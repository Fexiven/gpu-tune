//! Memory (VRAM) temperature through NvAPI (libnvidia-api.so.1, shipped with the NVIDIA
//! driver). NVML doesn't report memory temperature on GeForce cards; NvAPI's thermal channels
//! do, without kernel options or Secure Boot changes.
//!
//! Ported from LACT (github.com/ilya-zlobintsev/LACT, lact-daemon/src/server/gpu_controller/
//! nvidia/nvapi.rs), MIT License, Copyright (c) 2023 Ilya Zlobintsev. The thermal-channel calls
//! are undocumented; the IDs and struct layouts are the ones LACT uses.

use anyhow::{Context, Result, bail};
use libloading::Library;
use std::ffi::{CStr, c_char, c_void};
use std::sync::Mutex;

type Status = i32;
type Handle = *mut c_void;

const QUERY_INITIALIZE: u32 = 0x0150e828;
const QUERY_UNLOAD: u32 = 0xd22bdd7e;
const QUERY_ENUM_PHYSICAL_GPUS: u32 = 0xe5ac921f;
const QUERY_GPU_GET_BUS_ID: u32 = 0x1be0b8e5;
const QUERY_GET_ERROR_MESSAGE: u32 = 0x6c2d048c;
const QUERY_THERM_CHANNEL_GET_INFO: u32 = 0x0bc8163d;
const QUERY_THERM_CHANNEL_GET_STATUS: u32 = 0x65fe3aad;

const MAX_PHYSICAL_GPUS: usize = 64;
const SHORT_STRING_MAX: usize = 64;
const CHANNEL_TYPE_MEMORY: usize = 3;

const LIB_PATHS: &[&str] = &[
    "libnvidia-api.so.1",
    "/run/opengl-driver/lib/libnvidia-api.so.1", // NixOS
    "/usr/lib/x86_64-linux-gnu/libnvidia-api.so.1",
    "/usr/lib64/libnvidia-api.so.1",
    "/usr/lib/libnvidia-api.so.1",
];

const fn make_version<T>(version: usize) -> u32 {
    (std::mem::size_of::<T>() | (version << 16)) as u32
}

#[repr(C)]
struct Thermals {
    version: u32,
    mask: i32,
    rsvd: [u8; 32],
    temps: [i32; 32],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ThermChannel {
    class: u32,
    channel_type: u32,
    rel_loc: u32,
    target_gpu: u32,
    scaling: i32,
    offset_sw: i32,
    min_temp: i32,
    max_temp: i32,
    is_temp_sim_supported: u8,
    flags: u8,
    offset_hw: i32,
    rsvd: [u8; 28],
    data: [u8; 16],
}

#[repr(C)]
struct ThermChannelInfo {
    version: u32,
    mask: i32,
    rsvd: [u8; 32],
    channels: [ThermChannel; 32],
    primary_channel_idx: [u8; 5],
}

pub struct NvApiVram {
    lib: Library,
    handle: Handle,
    mask: i32,
    channel: usize,
    lock: Mutex<()>,
}

// NvAPI calls are serialized through `lock`; the handle is owned by the driver.
unsafe impl Send for NvApiVram {}
unsafe impl Sync for NvApiVram {}

impl NvApiVram {
    /// `pci_bus`: the PCI bus number of the GPU (e.g. 1 for 0000:01:00.0)
    pub fn new(pci_bus: u32) -> Result<Self> {
        let lib = LIB_PATHS
            .iter()
            .find_map(|p| unsafe { Library::new(*p) }.ok())
            .context("libnvidia-api.so.1 not found (comes with NVIDIA driver 525 and newer)")?;
        let mut s = NvApiVram {
            lib,
            handle: std::ptr::null_mut(),
            mask: 0,
            channel: 0,
            lock: Mutex::new(()),
        };
        unsafe {
            let init: unsafe extern "C" fn() -> Status =
                std::mem::transmute(s.query(QUERY_INITIALIZE)?);
            s.check(init())?;
            s.handle = s.find_gpu(pci_bus)?;
            let mut info = ThermChannelInfo {
                version: make_version::<ThermChannelInfo>(2),
                mask: 0,
                rsvd: [0; 32],
                channels: [ThermChannel::default(); 32],
                primary_channel_idx: [u8::MAX; 5],
            };
            s.gpu_query(QUERY_THERM_CHANNEL_GET_INFO, &mut info)
                .context("thermal channel info")?;
            let ch = info.primary_channel_idx[CHANNEL_TYPE_MEMORY] as usize;
            if ch >= 32 || info.mask & (1 << ch) == 0 {
                bail!("this GPU reports no memory temperature channel");
            }
            s.mask = info.mask;
            s.channel = ch;
        }
        if s.read().is_none() {
            bail!("memory temperature channel returned implausible values");
        }
        Ok(s)
    }

    /// Hottest memory temperature in °C (the memory junction value monitoring tools show).
    pub fn read(&self) -> Option<f64> {
        let _g = self.lock.lock().ok()?;
        let mut t = Thermals {
            version: make_version::<Thermals>(2),
            mask: self.mask,
            rsvd: [0; 32],
            temps: [0; 32],
        };
        unsafe { self.gpu_query(QUERY_THERM_CHANNEL_GET_STATUS, &mut t) }.ok()?;
        let v = t.temps[self.channel] / 256;
        (v > 0 && v < 255).then_some(v as f64)
    }

    unsafe fn query(&self, id: u32) -> Result<*const ()> {
        let qi = unsafe {
            self.lib
                .get::<unsafe extern "C" fn(u32) -> *const ()>(b"nvapi_QueryInterface\0")
        }
        .context("libnvidia-api has no nvapi_QueryInterface")?;
        let f = unsafe { qi(id) };
        if f.is_null() {
            bail!("NvAPI function {id:#x} not available in this driver");
        }
        Ok(f)
    }

    unsafe fn check(&self, status: Status) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        let mut text = [0 as c_char; SHORT_STRING_MAX];
        let msg = unsafe {
            match self.query(QUERY_GET_ERROR_MESSAGE) {
                Ok(f) => {
                    let f: unsafe extern "C" fn(Status, *mut c_char) -> Status =
                        std::mem::transmute(f);
                    if f(status, text.as_mut_ptr()) == 0 {
                        CStr::from_ptr(text.as_ptr()).to_string_lossy().into_owned()
                    } else {
                        String::new()
                    }
                }
                Err(_) => String::new(),
            }
        };
        bail!("NvAPI error {status}: {msg}")
    }

    unsafe fn find_gpu(&self, pci_bus: u32) -> Result<Handle> {
        unsafe {
            let enumerate: unsafe extern "C" fn(*mut Handle, *mut u32) -> Status =
                std::mem::transmute(self.query(QUERY_ENUM_PHYSICAL_GPUS)?);
            let mut handles = [std::ptr::null_mut(); MAX_PHYSICAL_GPUS];
            let mut count = 0u32;
            self.check(enumerate(handles.as_mut_ptr(), &mut count))?;
            let bus_id: unsafe extern "C" fn(Handle, *mut u32) -> Status =
                std::mem::transmute(self.query(QUERY_GPU_GET_BUS_ID)?);
            for h in handles.into_iter().take(count as usize) {
                let mut id = 0;
                self.check(bus_id(h, &mut id))?;
                if id == pci_bus {
                    return Ok(h);
                }
            }
        }
        bail!("NvAPI doesn't list a GPU on PCI bus {pci_bus:#x}")
    }

    unsafe fn gpu_query<T>(&self, id: u32, data: &mut T) -> Result<()> {
        unsafe {
            let f: unsafe extern "C" fn(Handle, *mut T) -> Status =
                std::mem::transmute(self.query(id)?);
            self.check(f(self.handle, data))
        }
    }
}

impl Drop for NvApiVram {
    fn drop(&mut self) {
        unsafe {
            if let Ok(f) = self.query(QUERY_UNLOAD) {
                let unload: unsafe extern "C" fn() -> Status = std::mem::transmute(f);
                unload();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn struct_sizes_match_the_driver_abi() {
        // version words encode the struct size; a wrong layout is rejected by the driver
        assert_eq!(std::mem::size_of::<Thermals>(), 4 + 4 + 32 + 32 * 4);
        assert_eq!(std::mem::size_of::<ThermChannel>(), 84); // 8 x 4-byte fields, 2 bytes + padding, 4, 28, 16
        assert_eq!(make_version::<Thermals>(2), 168 | (2 << 16));
    }
}

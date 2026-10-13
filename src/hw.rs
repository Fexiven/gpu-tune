//! GPU control and sensors: clock offsets (P0, read back after every change), clock cap,
//! power limit, readings, memory temperature, and NVIDIA driver errors from the kernel log.

use crate::nvapi::NvApiVram;
use crate::nvml::{self, Handle, Nvml};
use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::fs::OpenOptionsExt;
use std::sync::{Arc, Mutex};

pub const HARD_CORE: (i32, i32) = (0, 600);
pub const HARD_MEM: (i32, i32) = (0, 4000);

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Settings {
    pub core: i32,
    pub mem: i32,
    /// maximum GPU clock (locked clocks 210..max), None = stock
    pub clock: Option<u32>,
    pub power_mw: u32,
}

#[derive(Clone, Debug, Default)]
pub struct Snap {
    pub power: f64,
    pub util: u32,
    pub clk: u32,
    pub mclk: u32,
    pub temp: u32,
    pub pstate: Option<u32>,
    pub vram: Option<f64>,
    pub reasons: u64,
}

#[derive(Clone, Debug)]
pub struct GpuInfo {
    pub name: String,
    pub driver: String,
    pub bus: String,
    #[allow(dead_code)] // kept for diagnostics
    pub dev_id: u32,
    pub default_pl: u32,
    pub max_pl: u32,
    pub core_range: (i32, i32),
    pub mem_range: (i32, i32),
    pub api: String,
    /// stock maximum memory clock (what games get), MHz as nvidia-smi shows it
    pub max_mem_clk: u32,
}

pub trait Gpu: Send + Sync {
    fn info(&self) -> &GpuInfo;
    fn apply(&self, s: &Settings) -> Result<()>;
    fn reset(&self);
    fn snapshot(&self) -> Result<Snap>;
    fn power_limit(&self) -> u32;
    fn temp(&self) -> Result<u32>;
    fn vram(&self) -> Option<f64>;
    /// NVIDIA driver errors (Xid) logged since the last call
    fn new_xids(&self) -> Vec<String>;
    fn free_vram_mb(&self) -> u64;
}

// =================================================================== real GPU
pub struct RealGpu {
    nvml: Arc<Nvml>,
    h: Handle,
    info: GpuInfo,
    pstate_api: bool,
    klog: Mutex<KernelLog>,
    pub vram: VramSensor,
}

impl RealGpu {
    pub fn open(index: u32) -> Result<Self> {
        let nvml = Arc::new(Nvml::load()?);
        let h = nvml.handle(index)?;
        let (bus, dev_id) = nvml.pci(h)?;
        let (_, max_pl) = nvml.power_constraints(h)?;
        let pstate_api = nvml.has_pstate_offsets()
            && nvml
                .get_offset(h, nvml::CLOCK_GRAPHICS, nvml::PSTATE_0)
                .is_ok();
        let (mut core_range, mut mem_range) = (HARD_CORE, HARD_MEM);
        if pstate_api {
            let g = nvml.get_offset(h, nvml::CLOCK_GRAPHICS, nvml::PSTATE_0)?;
            let m = nvml.get_offset(h, nvml::CLOCK_MEM, nvml::PSTATE_0)?;
            if g.max_offset_mhz > 0 {
                core_range = (g.min_offset_mhz, g.max_offset_mhz);
            }
            if m.max_offset_mhz > 0 {
                mem_range = (m.min_offset_mhz, m.max_offset_mhz);
            }
        } else {
            let (c, m) = nvml.legacy_ranges(h);
            if let Some(c) = c {
                core_range = c;
            }
            if let Some(m) = m {
                mem_range = (0, m);
            }
        }
        let mut info = GpuInfo {
            name: nvml.name(h)?,
            driver: nvml.driver_version().unwrap_or_else(|_| "?".into()),
            bus: bus.clone(),
            dev_id,
            default_pl: nvml.power_default(h)?,
            max_pl,
            core_range,
            mem_range,
            api: if pstate_api {
                "per P-state (P0)".into()
            } else {
                "legacy (all P-states)".into()
            },
            max_mem_clk: 0,
        };
        let gpu = RealGpu {
            vram: VramSensor::new(&bus, dev_id),
            nvml,
            h,
            pstate_api,
            klog: Mutex::new(KernelLog::new()),
            info: info.clone(),
        };
        gpu.reset(); // start from real stock, even if LACT applied something before
        info.max_mem_clk = gpu.nvml.max_clock(h, nvml::CLOCK_MEM)?;
        Ok(RealGpu { info, ..gpu })
    }

    fn check_range(what: &str, v: i32, r: (i32, i32)) -> Result<()> {
        if v < r.0 || v > r.1 {
            bail!(
                "{what} offset {v:+} is outside the driver's range {}..{}",
                r.0,
                r.1
            );
        }
        Ok(())
    }

    fn set_offsets(&self, core: i32, mem: i32) -> Result<()> {
        Self::check_range("core", core, self.info.core_range)?;
        Self::check_range("memory", mem, self.info.mem_range)?;
        let n = &self.nvml;
        let (gc, gm) = if self.pstate_api {
            n.set_offset(self.h, nvml::CLOCK_GRAPHICS, nvml::PSTATE_0, core)?;
            n.set_offset(self.h, nvml::CLOCK_MEM, nvml::PSTATE_0, mem)?;
            (
                n.get_offset(self.h, nvml::CLOCK_GRAPHICS, nvml::PSTATE_0)?
                    .offset_mhz,
                n.get_offset(self.h, nvml::CLOCK_MEM, nvml::PSTATE_0)?
                    .offset_mhz,
            )
        } else {
            n.legacy_set(self.h, core, mem)?;
            n.legacy_get(self.h)?
        };
        if (gc - core).abs() > 20 || (gm - mem).abs() > 20 {
            bail!(
                "the driver reports core {gc:+}, memory {gm:+} after setting core {core:+}, memory {mem:+}"
            );
        }
        Ok(())
    }
}

impl Gpu for RealGpu {
    fn info(&self) -> &GpuInfo {
        &self.info
    }

    fn apply(&self, s: &Settings) -> Result<()> {
        self.nvml.set_power_limit(self.h, s.power_mw)?;
        self.set_offsets(s.core, s.mem)?;
        match s.clock {
            Some(c) => self.nvml.set_locked_clocks(self.h, 210, c),
            None => self.nvml.reset_locked_clocks(self.h),
        }
    }

    fn reset(&self) {
        let n = &self.nvml;
        let _ = n.reset_locked_clocks(self.h);
        let _ = n.set_power_limit(self.h, self.info.default_pl);
        if self.pstate_api {
            let _ = n.set_offset(self.h, nvml::CLOCK_GRAPHICS, nvml::PSTATE_0, 0);
            let _ = n.set_offset(self.h, nvml::CLOCK_MEM, nvml::PSTATE_0, 0);
        } else {
            let _ = n.legacy_set(self.h, 0, 0);
        }
    }

    fn snapshot(&self) -> Result<Snap> {
        let n = &self.nvml;
        Ok(Snap {
            power: n.power_mw(self.h)? as f64 / 1000.0,
            util: n.util(self.h)?,
            clk: n.clock(self.h, nvml::CLOCK_GRAPHICS)?,
            mclk: n.clock(self.h, nvml::CLOCK_MEM)?,
            temp: n.temp(self.h)?,
            pstate: n.pstate(self.h).ok(),
            vram: self.vram.read(),
            reasons: n.reasons(self.h),
        })
    }

    fn power_limit(&self) -> u32 {
        self.nvml
            .power_limit(self.h)
            .unwrap_or(self.info.default_pl)
    }

    fn temp(&self) -> Result<u32> {
        self.nvml.temp(self.h)
    }

    fn vram(&self) -> Option<f64> {
        self.vram.read()
    }

    fn new_xids(&self) -> Vec<String> {
        self.klog.lock().unwrap().new_xids()
    }

    fn free_vram_mb(&self) -> u64 {
        self.nvml.mem_free_mb(self.h).unwrap_or(4096)
    }
}

/// No NVIDIA management library: only for running the test loads (self-test on any device).
pub struct NullGpu(pub GpuInfo);
impl Gpu for NullGpu {
    fn info(&self) -> &GpuInfo {
        &self.0
    }
    fn apply(&self, _: &Settings) -> Result<()> {
        bail!("no GPU control without NVML")
    }
    fn reset(&self) {}
    fn snapshot(&self) -> Result<Snap> {
        bail!("no sensors without NVML")
    }
    fn power_limit(&self) -> u32 {
        0
    }
    fn temp(&self) -> Result<u32> {
        bail!("no sensors without NVML")
    }
    fn vram(&self) -> Option<f64> {
        None
    }
    fn new_xids(&self) -> Vec<String> {
        vec![]
    }
    fn free_vram_mb(&self) -> u64 {
        2048
    }
}

// =================================================================== kernel log
/// New NVIDIA Xid errors since the last call. Reads /dev/kmsg from its current end, so old
/// messages and the rotating ring buffer can't hide or fake an error. Falls back to dmesg.
pub struct KernelLog {
    kmsg: Option<std::fs::File>,
    seen: std::collections::HashSet<String>,
}

impl KernelLog {
    pub fn new() -> Self {
        let kmsg = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK)
            .open("/dev/kmsg")
            .and_then(|mut f| f.seek(SeekFrom::End(0)).map(|_| f))
            .ok();
        let seen = if kmsg.is_none() {
            Self::dmesg_xids().into_iter().collect()
        } else {
            Default::default()
        };
        KernelLog { kmsg, seen }
    }

    fn dmesg_xids() -> Vec<String> {
        std::process::Command::new("dmesg")
            .output()
            .map(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .lines()
                    .filter(|l| l.contains("NVRM: Xid"))
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// "6,1234,5678,-;NVRM: Xid (PCI:0000:01:00): 79, ..." -> the message text
    pub fn parse_record(raw: &[u8]) -> String {
        let t = String::from_utf8_lossy(raw);
        match t.split_once(';') {
            Some((_, msg)) => msg.trim().to_string(),
            None => t.trim().to_string(),
        }
    }

    pub fn new_xids(&mut self) -> Vec<String> {
        let Some(f) = self.kmsg.as_mut() else {
            let new: Vec<String> = Self::dmesg_xids()
                .into_iter()
                .filter(|l| !self.seen.contains(l))
                .collect();
            self.seen.extend(new.iter().cloned());
            return new;
        };
        let mut found = vec![];
        let mut buf = [0u8; 8192];
        loop {
            match f.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    let msg = Self::parse_record(&buf[..n]);
                    if msg.contains("NVRM: Xid") {
                        found.push(msg);
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(e) if e.raw_os_error() == Some(libc::EPIPE) => continue, // we fell behind
                Err(_) => break,
            }
        }
        found
    }
}

// =================================================================== VRAM temperature
/// Memory temperature: through NvAPI (comes with the driver, works out of the box), otherwise
/// straight from the card's register like the gddr6 tool (needs Secure Boot off and sometimes
/// the kernel option iomem=relaxed).
pub struct VramSensor {
    nvapi: Option<NvApiVram>,
    reg: Option<RegisterVram>,
    pub status: String,
}

impl VramSensor {
    pub fn new(bus: &str, dev_id: u32) -> Self {
        // "0000:01:00.0" -> PCI bus number 0x01
        let pci_bus = bus
            .split(':')
            .nth(1)
            .and_then(|b| u32::from_str_radix(b, 16).ok())
            .unwrap_or(0);
        let nv_err = match NvApiVram::new(pci_bus) {
            Ok(n) => {
                return VramSensor {
                    nvapi: Some(n),
                    reg: None,
                    status: "ok (NvAPI)".into(),
                };
            }
            Err(e) => format!("{e:#}"),
        };
        match RegisterVram::new(bus, dev_id) {
            Ok(r) => VramSensor {
                nvapi: None,
                reg: Some(r),
                status: "ok (register)".into(),
            },
            Err(e) => VramSensor {
                nvapi: None,
                reg: None,
                status: format!("not readable: NvAPI: {nv_err}; register: {e}"),
            },
        }
    }
    pub fn ok(&self) -> bool {
        self.nvapi.is_some() || self.reg.is_some()
    }
    pub fn read(&self) -> Option<f64> {
        match (&self.nvapi, &self.reg) {
            (Some(n), _) => n.read(),
            (None, Some(r)) => r.read(),
            _ => None,
        }
    }
}

/// PCI device id -> register offset, from the gddr6 tool's table
/// (github.com/olealgoritme/gddr6, lib/src/gddr6.c). Only cards listed there are read.
const GDDR6_REGS: &[(u32, u64)] = &[
    (0x2684, 0xE2A8),
    (0x2685, 0xE2A8),
    (0x2702, 0xE2A8),
    (0x2704, 0xE2A8),
    (0x2705, 0xE2A8),
    (0x2782, 0xE2A8),
    (0x2783, 0xE2A8),
    (0x2786, 0xE2A8),
    (0x2860, 0xE2A8),
    (0x28E0, 0xE2A8),
    (0x28A0, 0xE2A8),
    (0x2203, 0xE2A8),
    (0x2204, 0xE2A8),
    (0x2208, 0xE2A8),
    (0x2206, 0xE2A8),
    (0x2216, 0xE2A8),
    (0x2484, 0xEE50),
    (0x2488, 0xEE50),
    (0x2531, 0xE2A8),
    (0x2571, 0xE2A8),
    (0x24B0, 0xE2A8),
    (0x2232, 0xE2A8),
    (0x2231, 0xE2A8),
    (0x26B1, 0xE2A8),
    (0x27B8, 0xE2A8),
    (0x26B9, 0xE2A8),
    (0x2236, 0xE2A8),
];

/// GDDR6/6X temperature register in BAR0: temperature = (raw & 0xFFF) / 32. Needs root;
/// blocked when Secure Boot locks the kernel down; some kernels need iomem=relaxed.
pub struct RegisterVram {
    ptr: *const u32,
}
unsafe impl Send for RegisterVram {}
unsafe impl Sync for RegisterVram {}

impl RegisterVram {
    fn new(bus: &str, dev_id: u32) -> Result<Self> {
        let off = GDDR6_REGS
            .iter()
            .find(|(d, _)| *d == dev_id)
            .map(|(_, o)| *o)
            .ok_or_else(|| anyhow!("not supported on this GPU (device id {dev_id:04x})"))?;
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        let base = off / page * page;
        let dev = format!("/sys/bus/pci/devices/{bus}");
        let bar0 = std::fs::read_to_string(format!("{dev}/resource"))
            .ok()
            .and_then(|s| {
                s.split_whitespace()
                    .next()
                    .and_then(|x| u64::from_str_radix(x.trim_start_matches("0x"), 16).ok())
            });
        // the card's register window through sysfs, then /dev/mem; each without and with
        // O_SYNC (the kernel refuses a caching mode that conflicts with the driver's mapping)
        let mut tries = vec![(format!("{dev}/resource0"), base)];
        if let Some(b) = bar0 {
            tries.push(("/dev/mem".into(), b + base));
        }
        let mut errors = vec![];
        for (path, map_off) in tries {
            for sync in [0, libc::O_SYNC] {
                let c = std::ffi::CString::new(path.clone())?;
                let fd = unsafe { libc::open(c.as_ptr(), libc::O_RDONLY | sync) };
                if fd < 0 {
                    errors.push(format!("{path}: {}", std::io::Error::last_os_error()));
                    continue;
                }
                let p = unsafe {
                    libc::mmap(
                        std::ptr::null_mut(),
                        page as usize,
                        libc::PROT_READ,
                        libc::MAP_SHARED,
                        fd,
                        map_off as libc::off_t,
                    )
                };
                unsafe { libc::close(fd) };
                if p == libc::MAP_FAILED {
                    errors.push(format!("{path}: {}", std::io::Error::last_os_error()));
                    continue;
                }
                let r = RegisterVram {
                    ptr: unsafe { (p as *const u8).add((off - base) as usize) } as *const u32,
                };
                if r.read().is_some() {
                    return Ok(r);
                }
                errors.push(format!("{path}: implausible values"));
            }
        }
        errors.dedup();
        bail!("{}", errors.join("; "))
    }

    fn read(&self) -> Option<f64> {
        let raw = unsafe { std::ptr::read_volatile(self.ptr) };
        let t = (raw & 0xFFF) as f64 / 32.0;
        (15.0..=125.0).contains(&t).then_some(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn kmsg_record() {
        assert_eq!(
            KernelLog::parse_record(
                b"4,1234,567,-;NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.\n"
            ),
            "NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus."
        );
    }
}

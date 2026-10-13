//! Minimal hand-written binding to NVIDIA's management library (libnvidia-ml.so.1).
//! Only the ~30 functions this tool needs. Every call's return code is checked.

use anyhow::{Result, anyhow, bail};
use libloading::Library;
use std::ffi::{CStr, c_char, c_int, c_uint, c_ulonglong, c_void};

pub type Ret = c_int;
pub type Dev = *mut c_void;

pub const SUCCESS: Ret = 0;

pub const CLOCK_GRAPHICS: c_uint = 0;
pub const CLOCK_MEM: c_uint = 2;
pub const TEMPERATURE_GPU: c_uint = 0;
pub const PSTATE_0: c_uint = 0;

#[repr(C)]
#[derive(Default)]
pub struct PciInfo {
    pub bus_id_legacy: [c_char; 16],
    pub domain: c_uint,
    pub bus: c_uint,
    pub device: c_uint,
    pub pci_device_id: c_uint,
    pub pci_sub_system_id: c_uint,
    pub bus_id: [c_char; 32],
}

#[repr(C)]
#[derive(Default)]
pub struct Utilization {
    pub gpu: c_uint,
    pub memory: c_uint,
}

#[repr(C)]
#[derive(Default)]
pub struct MemInfo {
    pub total: c_ulonglong,
    pub free: c_ulonglong,
    pub used: c_ulonglong,
}

/// nvmlClockOffset_v1: offsets per clock domain and P-state (driver 555+).
#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct ClockOffset {
    pub version: c_uint,
    pub clock_type: c_uint,
    pub pstate: c_uint,
    pub offset_mhz: c_int,
    pub min_offset_mhz: c_int,
    pub max_offset_mhz: c_int,
}
pub const CLOCK_OFFSET_V1: c_uint = (std::mem::size_of::<ClockOffset>() as c_uint) | (1 << 24);

type F0 = unsafe extern "C" fn() -> Ret;
type FStr = unsafe extern "C" fn(*mut c_char, c_uint) -> Ret;
type FDevStr = unsafe extern "C" fn(Dev, *mut c_char, c_uint) -> Ret;
type FDevU = unsafe extern "C" fn(Dev, *mut c_uint) -> Ret;
type FDevUU = unsafe extern "C" fn(Dev, *mut c_uint, *mut c_uint) -> Ret;
type FDevSetU = unsafe extern "C" fn(Dev, c_uint) -> Ret;
type FDevSetUU = unsafe extern "C" fn(Dev, c_uint, c_uint) -> Ret;
type FDevArgU = unsafe extern "C" fn(Dev, c_uint, *mut c_uint) -> Ret;
type FDevI = unsafe extern "C" fn(Dev, *mut c_int) -> Ret;
type FDevII = unsafe extern "C" fn(Dev, *mut c_int, *mut c_int) -> Ret;
type FDevSetI = unsafe extern "C" fn(Dev, c_int) -> Ret;
type FDev = unsafe extern "C" fn(Dev) -> Ret;
type FDevU64 = unsafe extern "C" fn(Dev, *mut c_ulonglong) -> Ret;

pub struct Nvml {
    _lib: Library,
    error_string: unsafe extern "C" fn(Ret) -> *const c_char,
    driver_version: FStr,
    handle_by_index: unsafe extern "C" fn(c_uint, *mut Dev) -> Ret,
    name: FDevStr,
    pci_info: unsafe extern "C" fn(Dev, *mut PciInfo) -> Ret,
    pl_default: FDevU,
    pl_constraints: FDevUU,
    pl_get: FDevU,
    pl_set: FDevSetU,
    power_usage: FDevU,
    utilization: unsafe extern "C" fn(Dev, *mut Utilization) -> Ret,
    clock_info: FDevArgU,
    max_clock_info: FDevArgU,
    temperature: FDevArgU,
    pstate: FDevU,
    memory_info: unsafe extern "C" fn(Dev, *mut MemInfo) -> Ret,
    locked_set: FDevSetUU,
    locked_reset: FDev,
    reasons: Option<FDevU64>,
    offsets_set: Option<unsafe extern "C" fn(Dev, *mut ClockOffset) -> Ret>,
    offsets_get: Option<unsafe extern "C" fn(Dev, *mut ClockOffset) -> Ret>,
    gpc_set: Option<FDevSetI>,
    gpc_get: Option<FDevI>,
    mem_set: Option<FDevSetI>,
    mem_get: Option<FDevI>,
    gpc_minmax: Option<FDevII>,
    mem_minmax: Option<FDevII>,
}

// NVML is documented as thread-safe; the handles are plain pointers owned by the driver.
unsafe impl Send for Nvml {}
unsafe impl Sync for Nvml {}

#[derive(Clone, Copy)]
pub struct Handle(pub Dev);
unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}

const LIB_PATHS: &[&str] = &[
    "libnvidia-ml.so.1",
    "/run/opengl-driver/lib/libnvidia-ml.so.1", // NixOS
    "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
    "/usr/lib64/libnvidia-ml.so.1",
    "/usr/lib/libnvidia-ml.so.1",
];

fn cstr(buf: &[c_char]) -> String {
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

macro_rules! sym {
    ($lib:expr, $name:literal) => {
        *unsafe { $lib.get(concat!($name, "\0").as_bytes()) }
            .map_err(|e| anyhow!("libnvidia-ml has no {}: {e}", $name))?
    };
}
macro_rules! opt_sym {
    ($lib:expr, $name:literal) => {
        unsafe { $lib.get(concat!($name, "\0").as_bytes()) }
            .ok()
            .map(|s| *s)
    };
}

impl Nvml {
    pub fn load() -> Result<Self> {
        let mut last = None;
        let lib = LIB_PATHS
            .iter()
            .find_map(|p| match unsafe { Library::new(*p) } {
                Ok(l) => Some(l),
                Err(e) => {
                    last = Some(e);
                    None
                }
            })
            .ok_or_else(|| {
                anyhow!("can't load libnvidia-ml.so.1 (is the NVIDIA driver installed?): {last:?}")
            })?;
        let init: F0 = sym!(lib, "nvmlInit_v2");
        let n = Nvml {
            error_string: sym!(lib, "nvmlErrorString"),
            driver_version: sym!(lib, "nvmlSystemGetDriverVersion"),
            handle_by_index: sym!(lib, "nvmlDeviceGetHandleByIndex_v2"),
            name: sym!(lib, "nvmlDeviceGetName"),
            pci_info: sym!(lib, "nvmlDeviceGetPciInfo_v3"),
            pl_default: sym!(lib, "nvmlDeviceGetPowerManagementDefaultLimit"),
            pl_constraints: sym!(lib, "nvmlDeviceGetPowerManagementLimitConstraints"),
            pl_get: sym!(lib, "nvmlDeviceGetPowerManagementLimit"),
            pl_set: sym!(lib, "nvmlDeviceSetPowerManagementLimit"),
            power_usage: sym!(lib, "nvmlDeviceGetPowerUsage"),
            utilization: sym!(lib, "nvmlDeviceGetUtilizationRates"),
            clock_info: sym!(lib, "nvmlDeviceGetClockInfo"),
            max_clock_info: sym!(lib, "nvmlDeviceGetMaxClockInfo"),
            temperature: sym!(lib, "nvmlDeviceGetTemperature"),
            pstate: sym!(lib, "nvmlDeviceGetPerformanceState"),
            memory_info: sym!(lib, "nvmlDeviceGetMemoryInfo"),
            locked_set: sym!(lib, "nvmlDeviceSetGpuLockedClocks"),
            locked_reset: sym!(lib, "nvmlDeviceResetGpuLockedClocks"),
            reasons: opt_sym!(lib, "nvmlDeviceGetCurrentClocksEventReasons")
                .or(opt_sym!(lib, "nvmlDeviceGetCurrentClocksThrottleReasons")),
            offsets_set: opt_sym!(lib, "nvmlDeviceSetClockOffsets"),
            offsets_get: opt_sym!(lib, "nvmlDeviceGetClockOffsets"),
            gpc_set: opt_sym!(lib, "nvmlDeviceSetGpcClkVfOffset"),
            gpc_get: opt_sym!(lib, "nvmlDeviceGetGpcClkVfOffset"),
            mem_set: opt_sym!(lib, "nvmlDeviceSetMemClkVfOffset"),
            mem_get: opt_sym!(lib, "nvmlDeviceGetMemClkVfOffset"),
            gpc_minmax: opt_sym!(lib, "nvmlDeviceGetGpcClkMinMaxVfOffset"),
            mem_minmax: opt_sym!(lib, "nvmlDeviceGetMemClkMinMaxVfOffset"),
            _lib: lib,
        };
        n.check(unsafe { init() }, "init")?;
        Ok(n)
    }

    pub fn err(&self, r: Ret) -> String {
        let p = unsafe { (self.error_string)(r) };
        if p.is_null() {
            format!("NVML error {r}")
        } else {
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        }
    }

    fn check(&self, r: Ret, what: &str) -> Result<()> {
        if r == SUCCESS {
            Ok(())
        } else {
            bail!("{what}: {} ({r})", self.err(r))
        }
    }

    pub fn driver_version(&self) -> Result<String> {
        let mut b = [0 as c_char; 96];
        self.check(
            unsafe { (self.driver_version)(b.as_mut_ptr(), 96) },
            "driver version",
        )?;
        Ok(cstr(&b))
    }

    pub fn handle(&self, index: u32) -> Result<Handle> {
        let mut d: Dev = std::ptr::null_mut();
        self.check(
            unsafe { (self.handle_by_index)(index, &mut d) },
            "GPU handle",
        )?;
        Ok(Handle(d))
    }

    pub fn name(&self, h: Handle) -> Result<String> {
        let mut b = [0 as c_char; 96];
        self.check(unsafe { (self.name)(h.0, b.as_mut_ptr(), 96) }, "name")?;
        Ok(cstr(&b))
    }

    /// (sysfs-style bus id "0000:01:00.0", PCI device id like 0x2702)
    pub fn pci(&self, h: Handle) -> Result<(String, u32)> {
        let mut p = PciInfo::default();
        self.check(unsafe { (self.pci_info)(h.0, &mut p) }, "PCI info")?;
        let bus = cstr(&p.bus_id).to_lowercase();
        // NVML: "00000000:01:00.0" -> sysfs/Vulkan: "0000:01:00.0"
        let bus = match bus.split_once(':') {
            Some((dom, rest)) => {
                format!("{:04x}:{rest}", u32::from_str_radix(dom, 16).unwrap_or(0))
            }
            None => bus,
        };
        Ok((bus, p.pci_device_id >> 16))
    }

    fn get_u(&self, f: FDevU, h: Handle, what: &str) -> Result<u32> {
        let mut v = 0;
        self.check(unsafe { f(h.0, &mut v) }, what)?;
        Ok(v)
    }

    pub fn power_default(&self, h: Handle) -> Result<u32> {
        self.get_u(self.pl_default, h, "default power limit")
    }
    pub fn power_constraints(&self, h: Handle) -> Result<(u32, u32)> {
        let (mut a, mut b) = (0, 0);
        self.check(
            unsafe { (self.pl_constraints)(h.0, &mut a, &mut b) },
            "power limits",
        )?;
        Ok((a, b))
    }
    pub fn power_limit(&self, h: Handle) -> Result<u32> {
        self.get_u(self.pl_get, h, "power limit")
    }
    pub fn set_power_limit(&self, h: Handle, mw: u32) -> Result<()> {
        self.check(unsafe { (self.pl_set)(h.0, mw) }, "set power limit")
    }
    pub fn power_mw(&self, h: Handle) -> Result<u32> {
        self.get_u(self.power_usage, h, "power")
    }
    pub fn util(&self, h: Handle) -> Result<u32> {
        let mut u = Utilization::default();
        self.check(unsafe { (self.utilization)(h.0, &mut u) }, "utilization")?;
        Ok(u.gpu)
    }
    pub fn clock(&self, h: Handle, ty: u32) -> Result<u32> {
        let mut v = 0;
        self.check(unsafe { (self.clock_info)(h.0, ty, &mut v) }, "clock")?;
        Ok(v)
    }
    pub fn max_clock(&self, h: Handle, ty: u32) -> Result<u32> {
        let mut v = 0;
        self.check(
            unsafe { (self.max_clock_info)(h.0, ty, &mut v) },
            "max clock",
        )?;
        Ok(v)
    }
    pub fn temp(&self, h: Handle) -> Result<u32> {
        let mut v = 0;
        self.check(
            unsafe { (self.temperature)(h.0, TEMPERATURE_GPU, &mut v) },
            "temperature",
        )?;
        Ok(v)
    }
    pub fn pstate(&self, h: Handle) -> Result<u32> {
        self.get_u(self.pstate, h, "P-state")
    }
    pub fn mem_free_mb(&self, h: Handle) -> Result<u64> {
        let mut m = MemInfo::default();
        self.check(unsafe { (self.memory_info)(h.0, &mut m) }, "memory info")?;
        Ok(m.free / 1024 / 1024)
    }
    pub fn reasons(&self, h: Handle) -> u64 {
        let mut v = 0;
        match self.reasons {
            Some(f) if unsafe { f(h.0, &mut v) } == SUCCESS => v,
            _ => 0,
        }
    }
    pub fn set_locked_clocks(&self, h: Handle, min: u32, max: u32) -> Result<()> {
        self.check(unsafe { (self.locked_set)(h.0, min, max) }, "lock clocks")
    }
    pub fn reset_locked_clocks(&self, h: Handle) -> Result<()> {
        self.check(unsafe { (self.locked_reset)(h.0) }, "reset clocks")
    }

    // ---- clock offsets: per-P-state API (driver 555+) with the older VF-offset API as fallback
    pub fn has_pstate_offsets(&self) -> bool {
        self.offsets_set.is_some() && self.offsets_get.is_some()
    }
    pub fn get_offset(&self, h: Handle, clock_type: u32, pstate: u32) -> Result<ClockOffset> {
        let f = self
            .offsets_get
            .ok_or_else(|| anyhow!("nvmlDeviceGetClockOffsets missing"))?;
        let mut o = ClockOffset {
            version: CLOCK_OFFSET_V1,
            clock_type,
            pstate,
            ..Default::default()
        };
        self.check(unsafe { f(h.0, &mut o) }, "get clock offset")?;
        Ok(o)
    }
    pub fn set_offset(&self, h: Handle, clock_type: u32, pstate: u32, mhz: i32) -> Result<()> {
        let f = self
            .offsets_set
            .ok_or_else(|| anyhow!("nvmlDeviceSetClockOffsets missing"))?;
        let mut o = ClockOffset {
            version: CLOCK_OFFSET_V1,
            clock_type,
            pstate,
            offset_mhz: mhz,
            ..Default::default()
        };
        self.check(unsafe { f(h.0, &mut o) }, "set clock offset")
    }
    pub fn legacy_set(&self, h: Handle, core: i32, mem: i32) -> Result<()> {
        let (g, m) = (
            self.gpc_set
                .ok_or_else(|| anyhow!("no offset API in this driver"))?,
            self.mem_set
                .ok_or_else(|| anyhow!("no offset API in this driver"))?,
        );
        self.check(unsafe { g(h.0, core) }, "set core offset")?;
        self.check(unsafe { m(h.0, mem) }, "set memory offset")
    }
    pub fn legacy_get(&self, h: Handle) -> Result<(i32, i32)> {
        let (g, m) = (
            self.gpc_get.ok_or_else(|| anyhow!("no offset API"))?,
            self.mem_get.ok_or_else(|| anyhow!("no offset API"))?,
        );
        let (mut c, mut v) = (0, 0);
        self.check(unsafe { g(h.0, &mut c) }, "get core offset")?;
        self.check(unsafe { m(h.0, &mut v) }, "get memory offset")?;
        Ok((c, v))
    }
    /// (core max, mem max) from the legacy range functions, if available
    pub fn legacy_ranges(&self, h: Handle) -> (Option<(i32, i32)>, Option<i32>) {
        let core = self.gpc_minmax.and_then(|f| {
            let (mut a, mut b) = (0, 0);
            (unsafe { f(h.0, &mut a, &mut b) } == SUCCESS && b > 0).then_some((a, b))
        });
        // the min value of the memory range underflows on some drivers: only trust the max
        let mem = self.mem_minmax.and_then(|f| {
            let (mut a, mut b) = (0, 0);
            (unsafe { f(h.0, &mut a, &mut b) } == SUCCESS && b > 0 && b < 100_000).then_some(b)
        });
        (core, mem)
    }
}

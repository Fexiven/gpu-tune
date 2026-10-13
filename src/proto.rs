//! What the tuner sends to a worker process and what comes back.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// unpaced render load: full power, measures throughput
    Heavy,
    /// render bursts at a fixed frame rate: the card is not power-limited, like most games
    Paced,
    /// buffer copies + pattern check: memory bandwidth and errors
    Mem,
    /// memory test while the core renders (hot board)
    Combo,
    /// open the device and do a tiny job
    Ping,
}

impl Kind {
    pub fn text(self) -> &'static str {
        match self {
            Kind::Heavy => "Full load",
            Kind::Paced => "Game-like load (120 fps)",
            Kind::Mem => "Memory bandwidth",
            Kind::Combo => "Memory with a hot GPU",
            Kind::Ping => "Health check",
        }
    }
    pub fn short(self) -> &'static str {
        match self {
            Kind::Heavy => "full",
            Kind::Paced => "game",
            Kind::Mem => "mem",
            Kind::Combo => "mem-hot",
            Kind::Ping => "ping",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Params {
    pub kind: Kind,
    pub seconds: f64,
    pub warmup: f64,
    /// Vulkan device to use: PCI bus id like "0000:01:00.0", or "any" (tests only)
    pub bus: String,
    pub width: u32,
    pub height: u32,
    /// shader loop iterations per pixel and pass
    pub iters: u32,
    pub per_frame: u32,
    pub fps: u32,
    /// size of each of the two memory test buffer sets
    pub mem_mb: u64,
    /// self-test: deliberately corrupt some results to prove the checks catch them
    #[serde(default)]
    pub inject: bool,
}

impl Params {
    /// key for the cross-run reference hash: same image => same key
    pub fn golden_key(&self) -> String {
        format!("render:{}x{}:{}", self.width, self.height, self.iters)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct WorkResult {
    pub core_errors: u64,
    pub mem_errors: u64,
    /// full load: giga pixel-iterations per second (only meaningful relative to stock)
    pub perf: Option<f64>,
    /// game-like load: fraction of each frame the GPU was busy
    pub busy: Option<f64>,
    pub frames: Option<u64>,
    pub missed: Option<u64>,
    /// memory: GB/s (read + write)
    pub bw: Option<f64>,
    /// hash of the reference image this process rendered
    pub hash: Option<String>,
    pub adapter: String,
}

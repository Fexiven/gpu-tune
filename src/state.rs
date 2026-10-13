//! Saved progress (gpu_tune_state.json), written atomically after every step, so a crash,
//! a reboot or Ctrl+C never loses more than the test that was running.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub const STATE_VERSION: u32 = 9;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Noise {
    pub perf: f64,
    pub heavy_power: f64,
    pub paced_power: f64,
    pub clk: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Baseline {
    pub perf: f64,
    pub heavy_power: f64,
    pub heavy_clk: f64,
    pub paced_power: f64,
    pub paced_clk: f64,
    pub pstate: Option<u32>,
    pub noise: Noise,
    /// did the full load reach the power limit at stock (like a heavy game)?
    #[serde(default)]
    pub heavy_limited: bool,
    #[serde(default)]
    pub power_limit_w: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Calib {
    /// P-state the test loads run in (0 = like games)
    pub pstate: Option<u32>,
    /// memory clock under load at stock, MHz as nvidia-smi shows it
    pub mem_clk: f64,
    /// MHz of real memory clock per offset unit (0 = memory offset has no effect)
    pub k: f64,
}

/// One measured setting.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Prof {
    pub max_clock: u32,
    pub core: i32,
    pub mem: i32,
    pub max_stable: i32,
    pub perf: f64,
    pub heavy_power: f64,
    pub heavy_clk: f64,
    pub paced_power: f64,
    pub paced_clk: f64,
    pub missed: u64,
    pub temp: u32,
    pub limited_by: String,
    #[serde(default)]
    pub power_max: bool,
    /// a lower power limit found for this profile (None = default limit)
    #[serde(default)]
    pub power_limit_mw: Option<u32>,
    #[serde(default)]
    pub vram_max: Option<f64>,
    /// why the power-limit search stopped where it did (shown in the report)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub power_note: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PassState {
    pub golden: BTreeMap<String, String>,
    pub per_frame: Option<u32>,
    /// calibrated work per pass of the game-like load
    #[serde(default)]
    pub paced_iters: Option<u32>,
    /// calibrated math per step of the full load (until it reaches the power limit)
    #[serde(default)]
    pub heavy_alu: Option<u32>,
    pub baseline: Option<Baseline>,
    pub calib: Option<Calib>,
    pub core_check_done: bool,
    pub mem: Option<i32>,
    pub mem_bw: Option<f64>,
    pub mem_top_of_range: bool,
    /// lowest memory offset that showed errors (the pick stays MEM_ERROR_MARGIN below it)
    #[serde(default)]
    pub mem_err: Option<i32>,
    pub bw_noise: Option<f64>,
    pub clocks: BTreeMap<u32, Option<Prof>>,
    /// result of every test step, by key; survives crashes and reboots
    pub tried: BTreeMap<String, Value>,
    /// the step that was running when the state was saved (a crash => failure)
    pub pending: Option<String>,
    /// pass 2: the clock caps it fine-tunes (re-planned if pass 1's picture changes)
    #[serde(default)]
    pub plan: Vec<u32>,
    /// soak test per profile name ("efficient", "balanced", "max")
    #[serde(default)]
    pub soaks: BTreeMap<String, Soak>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Soak {
    /// settings being soak-tested when the state was saved (a crash => back off from here)
    pub at: Option<(i32, i32)>,
    pub done: bool,
    /// None after `done` = nothing passed
    pub result: Option<Prof>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Sig {
    pub v: u32,
    pub gpu: String,
    pub driver: String,
    pub clocks: Vec<u32>,
    pub core_step: i32,
    pub core_max: i32,
    pub core_margin: i32,
    pub mem_max: i32,
    pub fps: u32,
    pub width: u32,
    pub height: u32,
    pub iters_heavy: u32,
    pub iters_paced: u32,
    #[serde(default)]
    pub alu_heavy: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResultRec {
    /// the --goal profile (or the closest one that passed)
    pub recommended: Prof,
    #[serde(default)]
    pub recommended_name: String,
    pub baseline: Baseline,
    /// every profile that passed its soak test, by name
    pub profiles: BTreeMap<String, Prof>,
    /// profiles where nothing passed the soak test
    #[serde(default)]
    pub failed: Vec<String>,
}

/// Display name of a profile.
pub fn profile_label(name: &str) -> &'static str {
    match name {
        "efficient" => "Efficient",
        "max" => "Performance",
        _ => "Balanced",
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct State {
    pub sig: Sig,
    pub started: f64,
    pub goal: String,
    pub quick_only: bool,
    pub done: bool,
    pub p1: PassState,
    pub p2: PassState,
    pub result: Option<ResultRec>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pass {
    P1,
    P2,
}

impl State {
    pub fn pass_mut(&mut self, p: Pass) -> &mut PassState {
        match p {
            Pass::P1 => &mut self.p1,
            Pass::P2 => &mut self.p2,
        }
    }
}

/// Files the tuner writes go back to the user who ran sudo.
pub fn give_back(path: &Path) {
    let id = |k| std::env::var(k).ok().and_then(|v| v.parse::<u32>().ok());
    if let (Some(uid), Some(gid)) = (id("SUDO_UID"), id("SUDO_GID")) {
        let _ = std::os::unix::fs::chown(path, Some(uid), Some(gid));
    }
}

/// Write a file so that a crash or Ctrl+C never leaves it half-written.
pub fn write_atomic(path: &Path, text: &str) -> std::io::Result<()> {
    use std::io::Write;
    let tmp = path.with_extension("tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(text.as_bytes())?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    give_back(path);
    Ok(())
}

/// Holds the whole state; tuners hand in their pass's part whenever it changes.
pub struct Saver {
    pub path: Option<PathBuf>,
    pub state: Mutex<State>,
}

impl Saver {
    pub fn save_pass(&self, which: Pass, ps: &PassState) {
        let mut st = self.state.lock().unwrap();
        *st.pass_mut(which) = ps.clone();
        self.write(&st);
    }
    pub fn save(&self) {
        let st = self.state.lock().unwrap();
        self.write(&st);
    }
    fn write(&self, st: &State) {
        if let Some(p) = &self.path
            && let Err(e) = write_atomic(p, &serde_json::to_string_pretty(st).unwrap())
        {
            eprintln!("can't save progress to {}: {e}", p.display());
        }
    }
}

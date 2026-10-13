//! Runs one test in a worker process while sampling the GPU, and summarizes it.

use crate::hw::{Gpu, Snap};
use crate::proto::{Kind, Params, WorkResult};
use std::io::{BufRead, BufReader, Read};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Set by Ctrl+C (signal or dashboard key). Everything that waits checks it.
pub static INTERRUPTED: AtomicBool = AtomicBool::new(false);

pub fn interrupted() -> bool {
    INTERRUPTED.load(Ordering::SeqCst)
}

extern "C" fn on_signal(_: libc::c_int) {
    // only an atomic store: safe inside a signal handler
    INTERRUPTED.store(true, Ordering::SeqCst);
}

/// Ctrl+C, `kill` and a closed terminal all stop the tuner the safe way (GPU back to stock,
/// progress saved) instead of killing it mid-test.
pub fn catch_signals() {
    for sig in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP] {
        unsafe { libc::signal(sig, on_signal as *const () as libc::sighandler_t) };
    }
}

/// Why the tuning stops early.
#[derive(Debug, Clone, PartialEq)]
pub enum Stop {
    /// Ctrl+C: GPU back to stock, progress saved
    Interrupted,
    /// the card keeps overheating even after cooling down: a cooling problem, not a result
    Thermal,
    /// the GPU stopped responding: reboot and run again
    Fatal(String),
    /// any other reason to stop, with a message for the user
    Exit(String),
}

impl std::fmt::Display for Stop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Stop::Interrupted => write!(f, "stopped"),
            Stop::Thermal => write!(f, "overheating"),
            Stop::Fatal(m) | Stop::Exit(m) => write!(f, "{m}"),
        }
    }
}

pub type R<T> = std::result::Result<T, Stop>;

/// Time source, replaceable in tests so simulated runs take no real time.
pub trait Clock: Send + Sync {
    fn now(&self) -> f64;
    fn sleep(&self, secs: f64) -> R<()>;
}

pub struct RealClock(Instant);
impl RealClock {
    pub fn new() -> Self {
        RealClock(Instant::now())
    }
}
impl Clock for RealClock {
    fn now(&self) -> f64 {
        self.0.elapsed().as_secs_f64()
    }
    fn sleep(&self, secs: f64) -> R<()> {
        let end = Instant::now() + Duration::from_secs_f64(secs.max(0.0));
        while Instant::now() < end {
            if interrupted() {
                return Err(Stop::Interrupted);
            }
            std::thread::sleep(
                Duration::from_millis(100).min(end.saturating_duration_since(Instant::now())),
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Reason {
    Crash,
    Hung,
    Overheat,
}

#[derive(Clone, Debug, Default)]
pub struct Summary {
    pub power: f64,
    pub core_clk: f64,
    pub mem_clk: f64,
    pub temp: u32,
    pub limit: String,
    pub vram: Option<f64>,
    pub pstate: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct Outcome {
    pub res: Option<WorkResult>,
    pub summ: Summary,
    pub reason: Option<Reason>,
    pub why: String,
}

pub trait Runner: Send + Sync {
    fn run(&self, p: &Params, temp_limit: u32, vram_limit: f64) -> R<Outcome>;
}

const POWER_BITS: u64 = 0x4 | 0x80;
const THERMAL_BITS: u64 = 0x8 | 0x20 | 0x40;

/// The worker prints this line when the measured part of a test starts (after opening the
/// device, rendering the reference and the warm-up), so the averages cover only that part.
pub const MEASURE_MARK: &str = "MEASURE";

/// Average the samples of one test from `from` (seconds, same clock as the samples) on; for
/// loaded tests only count samples with the GPU busy (for the game-like load the idle gaps
/// between frames are part of the picture).
pub fn summarize(samples: &[(f64, Snap)], from: f64, loaded_only: bool) -> Summary {
    if samples.is_empty() {
        return Summary::default();
    }
    let mut win: Vec<&Snap> = samples
        .iter()
        .filter(|(t, _)| *t >= from)
        .map(|(_, s)| s)
        .collect();
    if win.is_empty() {
        win = samples.iter().map(|(_, s)| s).collect();
    }
    if loaded_only {
        let busy: Vec<&Snap> = win.iter().copied().filter(|s| s.util >= 80).collect();
        if !busy.is_empty() {
            win = busy;
        }
    }
    let n = win.len() as f64;
    let frac = |bits: u64| win.iter().filter(|s| s.reasons & bits != 0).count() as f64 / n;
    let mut limit = if frac(POWER_BITS) > 0.3 {
        "power".to_string()
    } else {
        String::new()
    };
    if frac(THERMAL_BITS) > 0.05 {
        limit = if limit.is_empty() {
            "thermal".into()
        } else {
            limit + "+thermal"
        };
    }
    let mut mclk: Vec<u32> = win.iter().map(|s| s.mclk).collect();
    mclk.sort_unstable();
    let mut ps: Vec<u32> = win.iter().filter_map(|s| s.pstate).collect();
    ps.sort_unstable();
    let pstate = ps
        .iter()
        .copied()
        .max_by_key(|p| ps.iter().filter(|x| *x == p).count());
    Summary {
        power: win.iter().map(|s| s.power).sum::<f64>() / n,
        core_clk: win.iter().map(|s| s.clk as f64).sum::<f64>() / n,
        mem_clk: mclk[mclk.len() / 2] as f64,
        temp: win.iter().map(|s| s.temp).max().unwrap_or(0),
        limit,
        vram: samples
            .iter()
            .filter_map(|(_, s)| s.vram)
            .fold(None, |m, v| Some(m.map_or(v, |m: f64| m.max(v)))),
        pstate,
    }
}

/// Real runner: starts this program again as a worker and samples the GPU meanwhile.
pub struct ProcessRunner {
    pub gpu: Arc<dyn Gpu>,
}

impl Runner for ProcessRunner {
    fn run(&self, p: &Params, temp_limit: u32, vram_limit: f64) -> R<Outcome> {
        let exe =
            std::env::current_exe().map_err(|e| Stop::Exit(format!("can't find myself: {e}")))?;
        self.gpu.new_xids(); // drain: only errors from this test count
        let samples: Arc<Mutex<Vec<(f64, Snap)>>> = Default::default();
        let too_hot = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicBool::new(false));
        let t0 = Instant::now();
        let sampler = {
            let (gpu, samples, too_hot, done) = (
                self.gpu.clone(),
                samples.clone(),
                too_hot.clone(),
                done.clone(),
            );
            std::thread::spawn(move || {
                while !done.load(Ordering::SeqCst) {
                    if let Ok(s) = gpu.snapshot() {
                        if s.temp >= temp_limit || s.vram.unwrap_or(0.0) >= vram_limit {
                            too_hot.store(true, Ordering::SeqCst);
                        }
                        samples
                            .lock()
                            .unwrap()
                            .push((t0.elapsed().as_secs_f64(), s));
                    }
                    std::thread::sleep(Duration::from_millis(200));
                }
            })
        };
        let mut child = Command::new(&exe)
            .arg("--worker")
            .arg(serde_json::to_string(p).unwrap())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null())
            .spawn()
            .map_err(|e| Stop::Exit(format!("can't start a test: {e}")))?;
        // read the pipes in threads so a chatty worker can't block on a full pipe; stdout is
        // read line by line to catch the moment the measurement starts
        let measure_at: Arc<Mutex<Option<f64>>> = Default::default();
        let out_t = {
            let (out, at) = (child.stdout.take(), measure_at.clone());
            std::thread::spawn(move || {
                let mut s = String::new();
                if let Some(out) = out {
                    for line in BufReader::new(out).lines().map_while(Result::ok) {
                        if line == MEASURE_MARK {
                            *at.lock().unwrap() = Some(t0.elapsed().as_secs_f64());
                        } else {
                            s.push_str(&line);
                            s.push('\n');
                        }
                    }
                }
                s
            })
        };
        let err_t = {
            let err = child.stderr.take();
            std::thread::spawn(move || {
                let mut s = String::new();
                if let Some(mut e) = err {
                    let _ = e.read_to_string(&mut s);
                }
                s
            })
        };
        let deadline = Instant::now() + Duration::from_secs_f64(p.seconds + p.warmup + 120.0);
        let mut reason = None;
        let mut stop = None;
        let status = loop {
            if let Ok(Some(st)) = child.try_wait() {
                break Some(st);
            }
            if interrupted() {
                stop = Some(Stop::Interrupted);
            } else if too_hot.load(Ordering::SeqCst) {
                reason = Some(Reason::Overheat);
            } else if Instant::now() > deadline {
                reason = Some(Reason::Hung);
            } else {
                std::thread::sleep(Duration::from_millis(200));
                continue;
            }
            let _ = child.kill();
            // a process stuck inside the driver can't be killed: don't hang with it
            let end = Instant::now() + Duration::from_secs(60);
            loop {
                if let Ok(Some(_)) = child.try_wait() {
                    break;
                }
                if Instant::now() > end {
                    done.store(true, Ordering::SeqCst);
                    return Err(Stop::Fatal(
                        "a test process is stuck inside the GPU driver and can't be stopped".into(),
                    ));
                }
                std::thread::sleep(Duration::from_millis(200));
            }
            break None;
        };
        done.store(true, Ordering::SeqCst);
        let _ = sampler.join();
        if let Some(s) = stop {
            return Err(s);
        }
        let out = out_t.join().unwrap_or_default();
        let err = err_t.join().unwrap_or_default();
        let res: Option<WorkResult> = out
            .lines()
            .find_map(|l| l.strip_prefix("RESULT "))
            .and_then(|j| serde_json::from_str(j).ok());
        let xids = self.gpu.new_xids();
        if reason.is_none()
            && (res.is_none() || !status.map(|s| s.success()).unwrap_or(false) || !xids.is_empty())
        {
            reason = Some(Reason::Crash);
        }
        let why = xids
            .last()
            .cloned()
            .or_else(|| {
                err.lines()
                    .rfind(|l| !l.trim().is_empty())
                    .map(String::from)
            })
            .unwrap_or_default();
        let samples = samples.lock().unwrap();
        // without the mark (old worker, crash before it): skip the warm-up from the first sample
        let from = measure_at
            .lock()
            .unwrap()
            .unwrap_or_else(|| samples.first().map_or(0.0, |s| s.0) + p.warmup);
        Ok(Outcome {
            res,
            summ: summarize(&samples, from, p.kind != Kind::Paced),
            reason,
            why: why.chars().take(120).collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn s(t: f64, util: u32, clk: u32, mclk: u32, reasons: u64) -> (f64, Snap) {
        (
            t,
            Snap {
                power: 100.0 + clk as f64 / 100.0,
                util,
                clk,
                mclk,
                temp: 60,
                pstate: Some(0),
                vram: None,
                reasons,
            },
        )
    }
    #[test]
    fn summary_skips_warmup_and_idle() {
        let v = vec![
            s(0.0, 99, 1000, 1, 0),
            s(1.0, 10, 500, 2, 0),
            s(2.0, 99, 2000, 3, 4),
            s(3.0, 99, 2000, 3, 4),
        ];
        let r = summarize(&v, 0.5, true);
        assert_eq!(r.core_clk, 2000.0);
        assert_eq!(r.limit, "power");
        assert_eq!(r.pstate, Some(0));
        let r = summarize(&v, 0.5, false);
        assert_eq!(r.core_clk, 1500.0);
        // the window starts where the worker said the measurement starts
        let r = summarize(&v, 2.0, false);
        assert_eq!(r.core_clk, 2000.0);
    }
}

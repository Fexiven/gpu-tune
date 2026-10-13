//! The tuning logic: stock baseline, calibration, memory search, core search, soak test.

use crate::hw::{Gpu, Settings};
use crate::proto::{Kind, Params};
use crate::runner::{Clock, R, Reason, Runner, Stop};
use crate::state::{Baseline, Calib, Noise, Pass, PassState, Prof, Saver};
use crate::ui::{Finding, Log, Row, RowState, Tone, Ui};
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// Settings for one pass (short tests in pass 1, full-length in pass 2) plus global options.
#[derive(Clone, Debug)]
pub struct PassArgs {
    pub warm: f64,
    pub measure: f64,
    pub warmup: f64,
    pub search_seconds: f64,
    pub mem_seconds: f64,
    pub combo_seconds: f64,
    pub combo_warmup: f64,
    pub mem_step: i32,
    pub soak: f64,
    pub clocks: Vec<u32>,
    pub hints: BTreeMap<u32, i32>,
    pub refine_mem: Option<i32>,
    pub refine_mem_top: bool,
    /// pass 2: lowest memory offset that showed errors in pass 1
    pub refine_mem_err: Option<i32>,
    // global
    pub core_max: i32,
    pub core_step: i32,
    pub core_margin: i32,
    pub mem_max: i32,
    pub temp_limit: u32,
    pub vram_limit: f64,
    pub vram_soft: f64,
    pub fps: u32,
    pub width: u32,
    pub height: u32,
    pub iters_heavy: u32,
    pub iters_paced: u32,
    pub alu_heavy: u32,
    pub alu_paced: u32,
    pub mem_mb: u64,
    pub bus: String,
    pub skip_offset_check: bool,
    pub ignore_pstate: bool,
}

impl PassArgs {
    pub fn for_pass(base: &PassArgs, quick: bool) -> PassArgs {
        let mut p = base.clone();
        if quick {
            (p.warm, p.measure, p.warmup, p.search_seconds) = (30.0, 25.0, 10.0, 15.0);
            (p.mem_seconds, p.combo_seconds, p.combo_warmup) = (10.0, 25.0, 10.0);
            (p.mem_step, p.soak) = (500, 120.0);
            let mut c: Vec<u32> = base.clocks.iter().map(|c| c / 15 * 15).collect();
            c.sort_unstable();
            c.dedup();
            p.clocks = c;
        } else {
            (p.warm, p.measure, p.warmup, p.search_seconds) = (30.0, 45.0, 15.0, 30.0);
            (p.mem_seconds, p.combo_seconds, p.combo_warmup) = (15.0, 40.0, 15.0);
            (p.mem_step, p.soak) = (250, 300.0);
        }
        p.hints.clear();
        p.refine_mem = None;
        p.refine_mem_top = false;
        p.refine_mem_err = None;
        p
    }
}

/// What one test measured.
#[derive(Clone, Debug, Default)]
pub struct TestRow {
    pub avg_power: f64,
    pub core_clk: f64,
    pub mem_clk: f64,
    pub temp: u32,
    pub limited_by: String,
    pub pstate: Option<u32>,
    pub vram_temp: Option<f64>,
    pub core_errors: u64,
    pub mem_errors: u64,
    pub perf: f64,
    pub busy: f64,
    pub missed: u64,
    pub bw: f64,
    pub hash: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Fail {
    Core,
    Mem,
}

/// The power limit a profile runs with: the card's maximum (performance profile, if it helped),
/// a lower limit found by the power search, or the default.
pub fn prof_power(p: &Prof, info: &crate::hw::GpuInfo) -> u32 {
    if p.power_max {
        info.max_pl
    } else {
        p.power_limit_mw.unwrap_or(info.default_pl)
    }
}

/// A power step is worth it while the performance it costs is at most this fraction of the power
/// it saves (relative). Near the top of the curve a GPU loses ~0.3% performance per 1% power
/// (voltage drops along with the clock); once the voltage is at its floor it's ~1:1.
pub const KNEE_ELASTICITY: f64 = 0.6;

/// Changes in power below this (relative) mean the limit didn't bite: ignore the step.
const MIN_STEP_SAVE: f64 = 0.005;

/// Why the power-limit search stopped at a step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PowerStop {
    /// this step cost clearly more performance than the power it saved
    Knee,
    /// the total performance loss went over the profile's budget
    LossCap,
}

/// `points[0]` = (performance, power) at the default limit, then lower limits in order.
/// The step that stopped the search and why (None = keep going).
pub fn power_stop(
    points: &[(f64, f64)],
    max_loss: f64,
    max_elast: f64,
) -> Option<(usize, PowerStop)> {
    pick_steps(points, max_loss, max_elast).1
}

/// The sweet spot among `points` (see `power_stop`): the lowest limit reached while every step
/// cost at most `max_elast` x the power it saved and the total loss stayed within `max_loss`.
/// None if it doesn't save at least `min_save` power.
pub fn pick_knee(
    points: &[(f64, f64)],
    max_loss: f64,
    min_save: f64,
    max_elast: f64,
) -> Option<usize> {
    let (best, _) = pick_steps(points, max_loss, max_elast);
    let (_, w0) = *points.first()?;
    best.filter(|&i| 1.0 - points[i].1 / w0 >= min_save)
}

type Steps = (Option<usize>, Option<(usize, PowerStop)>);

fn pick_steps(points: &[(f64, f64)], max_loss: f64, max_elast: f64) -> Steps {
    let Some(&(p0, w0)) = points.first() else {
        return (None, None);
    };
    let (mut best, mut last_loss, mut last_save) = (None, 0.0, 0.0);
    for (i, &(perf, power)) in points.iter().enumerate().skip(1) {
        let (loss, save) = (1.0 - perf / p0, 1.0 - power / w0);
        if loss > max_loss {
            return (best, Some((i, PowerStop::LossCap)));
        }
        let ds = save - last_save;
        if ds < MIN_STEP_SAVE {
            continue; // the limit didn't bite (yet): nothing to judge
        }
        if (loss - last_loss) / ds > max_elast {
            // the knee: from here on power is bought with performance
            return (best, Some((i, PowerStop::Knee)));
        }
        (last_loss, last_save) = (loss, save);
        best = Some(i);
    }
    (best, None)
}

pub fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// Run-to-run noise of repeated measurements: (max - min) / mean.
pub fn spread(v: &[f64]) -> f64 {
    let m = mean(v);
    if m.abs() <= 0.0 {
        return 0.0;
    }
    let (lo, hi) = v
        .iter()
        .fold((f64::MAX, f64::MIN), |(l, h), x| (l.min(*x), h.max(*x)));
    (hi - lo) / m.abs()
}

pub fn rel_noise(a: f64, b: f64) -> f64 {
    let m = (a.abs() + b.abs()) / 2.0;
    if m > 0.0 { (a - b).abs() / m } else { 0.0 }
}

/// Why the memory pick stopped where it did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MemStop {
    /// every offset tested kept its bandwidth: the pick is the highest one tested
    Top,
    /// from this offset on the bandwidth stopped following the clock (transfer retries)
    Retries(i32),
    /// the highest offset didn't add clearly more bandwidth than the noise: kept the lowest
    NoGain,
}

/// Memory offsets that passed, sorted ascending: (offset, bandwidth, measured clock).
/// GDDR6X detects transfer errors and replays them, so a too-high memory clock first shows up
/// as bandwidth that no longer follows the clock. The pick is the highest offset up to which
/// bandwidth per MHz stays within the noise of the best seen so far, judged over the whole
/// range (small steps add less than the noise each, so step-by-step tests can't see them), and
/// it must add at least half the bandwidth its clock increase should give, and clearly more
/// than the noise, over the lowest offset.
pub fn pick_mem(pts: &[(i32, f64, f64)], noise: f64) -> (usize, MemStop) {
    if pts.len() < 2 || pts[0].1 <= 0.0 || pts[0].2 <= 0.0 {
        return (0, MemStop::Top);
    }
    let tol = 0.005f64.max(2.0 * noise);
    let eff = |p: &(i32, f64, f64)| p.1 / p.2;
    let (mut best_eff, mut top, mut stop) = (eff(&pts[0]), 0, MemStop::Top);
    for (i, p) in pts.iter().enumerate().skip(1) {
        if p.2 <= 0.0 || eff(p) < best_eff * (1.0 - tol) {
            stop = MemStop::Retries(p.0);
            break;
        }
        best_eff = best_eff.max(eff(p));
        top = i;
    }
    let (p0, pt) = (&pts[0], &pts[top]);
    let (gain, ideal) = (pt.1 / p0.1 - 1.0, pt.2 / p0.2 - 1.0);
    if top > 0 && gain < (0.5 * ideal).max(2.0 * noise) {
        return (0, MemStop::NoGain);
    }
    (top, stop)
}

/// Memory offsets this far below the first one that showed errors are the most the tuner uses.
pub const MEM_ERROR_MARGIN: i32 = 500;

/// At low clock caps the voltage is already at its minimum: beyond some offset the power stops
/// falling, and more offset only raises the low (idle) points of the curve for nothing.
/// `pts` = (offset, game-like power) of the stable search steps at one clock cap. Returns the
/// smallest offset whose power is within `tol` of the lowest, if that's below the highest tested.
pub fn floor_offset(pts: &[(i32, f64)], tol: f64) -> Option<i32> {
    let mut v: Vec<(i32, f64)> = pts.iter().copied().filter(|p| p.1 > 0.0).collect();
    if v.len() < 3 {
        return None;
    }
    v.sort_by_key(|p| p.0);
    let min = v.iter().map(|p| p.1).fold(f64::MAX, f64::min);
    let flat = |p: &(i32, f64)| p.1 <= min * (1.0 + tol);
    let first = v.iter().find(|p| flat(p))?.0;
    // only meaningful if at least two higher offsets confirm the flat part, and all of them
    // stay flat (above the stock boost clock more offset means more clock and more power:
    // that's an overclock, not a floor)
    let above: Vec<_> = v.iter().filter(|p| p.0 > first).collect();
    (above.len() >= 2 && above.iter().all(|p| flat(p))).then_some(first)
}

/// Core clock in the game-like load vs stock. There the card is not power-limited (like in
/// most games), so this clock is what games get. A clock ratio, not fps, without the memory
/// overclock. (The full load is power-limited at stock, which would make low clock caps
/// look as fast as stock even though games would slow down.)
pub fn game_rel(r: &Prof, base: &Baseline) -> f64 {
    if base.paced_clk > 0.0 {
        r.paced_clk / base.paced_clk
    } else {
        1.0
    }
}

/// A core search step's saved result: `true` / `{"ok": true, ...}` = stable.
fn step_ok(v: &Value) -> bool {
    v.as_bool()
        .unwrap_or_else(|| v["ok"].as_bool().unwrap_or(false))
}

/// Binary search for the highest offset where `run` passes. Returns (lo, hi) with lo stable,
/// or None if even +0 fails. A hint (from pass 1) or the previous clock narrows the start.
pub fn search_bounds(
    mut run: impl FnMut(i32) -> R<bool>,
    hint: Option<i32>,
    prev: Option<i32>,
    core_max: i32,
    step: i32,
) -> R<Option<(i32, i32)>> {
    if !run(0)? {
        return Ok(None);
    }
    let (mut lo, mut hi) = (0, core_max + step);
    if let Some(h) = hint {
        let below = (h - 4 * step).max(0);
        let above = (h + 4 * step).min(core_max);
        if below > 0 {
            if run(below)? {
                lo = below;
            } else {
                hi = below;
            }
        }
        if lo < above && above < hi {
            if run(above)? {
                lo = above;
            } else {
                hi = above;
            }
        }
    } else if let Some(p) = prev.filter(|p| *p > 0 && *p <= core_max) {
        if run(p)? {
            lo = p;
        } else {
            hi = p;
        }
    }
    while hi - lo > step {
        let mid = lo + ((hi - lo) / (2 * step)).max(1) * step;
        if run(mid)? {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(Some((lo, hi)))
}

pub struct Profiles {
    pub efficient: Prof,
    pub balanced: Prof,
    pub max: Prof,
}

impl Profiles {
    pub fn get(&self, goal: &str) -> &Prof {
        match goal {
            "efficient" => &self.efficient,
            "max" => &self.max,
            _ => &self.balanced,
        }
    }
}

/// efficient = least power while gaming, clock at most ~2% below stock (not noticeable);
/// balanced  = highest game clock among settings using at most 90% of stock gaming power,
///             no lower than stock within the measurement noise (falls back to efficient);
/// max       = fastest overall.
/// All thresholds widen to twice the run-to-run noise measured at stock.
pub fn select_profiles(found: &[Prof], base: &Baseline) -> Profiles {
    let n = &base.noise;
    let perf_tol = 0.02f64.max(2.0 * n.perf);
    let near_tol = 0.02f64.max(2.0 * n.clk);
    let same_tol = 0.005f64.max(2.0 * n.clk);
    let power_cut = 0.90f64.min(1.0 - 3.0 * n.paced_power);
    let g = |r: &Prof| game_rel(r, base);
    let mut usable: Vec<&Prof> = found
        .iter()
        .filter(|r| r.missed == 0 && r.perf >= base.perf * (1.0 - perf_tol))
        .collect();
    if usable.is_empty() {
        usable = found.iter().collect();
    }
    let mut near: Vec<&Prof> = usable
        .iter()
        .copied()
        .filter(|r| g(r) >= 1.0 - near_tol)
        .collect();
    if near.is_empty() {
        let best = usable.iter().map(|r| g(r)).fold(f64::MIN, f64::max);
        near = usable
            .iter()
            .copied()
            .filter(|r| g(r) >= best - 0.01)
            .collect();
    }
    let eff = near
        .iter()
        .copied()
        .min_by(|a, b| a.paced_power.total_cmp(&b.paced_power))
        .unwrap()
        .clone();
    let frugal: Vec<&Prof> = usable
        .iter()
        .copied()
        .filter(|r| g(r) >= 1.0 - same_tol && r.paced_power <= base.paced_power * power_cut)
        .collect();
    let bal = frugal
        .iter()
        .copied()
        .max_by(|a, b| {
            g(a).total_cmp(&g(b))
                .then(b.paced_power.total_cmp(&a.paced_power))
        })
        .cloned()
        .unwrap_or_else(|| eff.clone());
    let fast = found
        .iter()
        .max_by(|a, b| (g(a) + a.perf / base.perf).total_cmp(&(g(b) + b.perf / base.perf)))
        .unwrap()
        .clone();
    Profiles {
        efficient: eff,
        balanced: bal,
        max: fast,
    }
}

/// Pass 1's clock caps relative to the card's stock boost clock (game-like load at stock):
/// from a deep undervolt to a little overclock.
pub const AUTO_CLOCKS: [i64; 4] = [-300, -150, 0, 150];

/// The clock caps to scan when `--clocks` isn't given: around the stock boost clock, in 15 MHz
/// steps (the V/F curve's resolution), never above the driver's maximum.
pub fn auto_clocks(stock_clk: f64, max_clk: u32) -> Vec<u32> {
    let base = (stock_clk as i64) / 15 * 15;
    let top = if max_clk > 0 {
        (max_clk / 15 * 15) as i64
    } else {
        i64::MAX
    };
    let mut v: Vec<u32> = AUTO_CLOCKS
        .iter()
        .map(|d| (base + d).min(top))
        .filter(|c| *c >= 1500)
        .map(|c| c as u32)
        .collect();
    v.dedup();
    v
}

pub fn interpolate_hints(found: &[Prof], clocks: &[u32]) -> BTreeMap<u32, i32> {
    let mut known: Vec<(u32, i32)> = found.iter().map(|r| (r.max_clock, r.max_stable)).collect();
    known.sort_unstable();
    let mut hints = BTreeMap::new();
    for &c in clocks {
        let lower = known.iter().rfind(|p| p.0 <= c);
        let upper = known.iter().find(|p| p.0 >= c);
        let est = match (lower, upper) {
            (Some(&(c1, o1)), Some(&(c2, o2))) => {
                if c1 == c2 {
                    o1 as f64
                } else {
                    o1 as f64 + (o2 - o1) as f64 * (c - c1) as f64 / (c2 - c1) as f64
                }
            }
            (Some(&(_, o)), None) | (None, Some(&(_, o))) => o as f64,
            (None, None) => continue,
        };
        hints.insert(c, ((est / 15.0).round() * 15.0) as i32);
    }
    hints
}

fn now_hms() -> String {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    let tt = t as libc::time_t;
    unsafe { libc::localtime_r(&tt, &mut tm) };
    format!("{:02}:{:02}:{:02}", tm.tm_hour, tm.tm_min, tm.tm_sec)
}

pub struct Tuner<'a> {
    pub gpu: &'a dyn Gpu,
    pub runner: &'a dyn Runner,
    pub clock: &'a dyn Clock,
    pub ui: &'a dyn Ui,
    pub log: &'a Log,
    pub saver: &'a Saver,
    pub a: PassArgs,
    pub st: PassState,
    pub which: Pass,
    pub tag: &'static str,
}

impl<'a> Tuner<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &'a dyn Gpu,
        runner: &'a dyn Runner,
        clock: &'a dyn Clock,
        ui: &'a dyn Ui,
        log: &'a Log,
        saver: &'a Saver,
        a: PassArgs,
        st: PassState,
        which: Pass,
    ) -> Self {
        let mut t = Tuner {
            gpu,
            runner,
            clock,
            ui,
            log,
            saver,
            a,
            st,
            which,
            tag: if which == Pass::P1 { "p1" } else { "p2" },
        };
        if let Some(p) = t.st.pending.take() {
            // the computer died during this step last time: that's a failure
            t.st.tried.insert(p.clone(), Value::Bool(false));
            t.save();
            t.ui.event(
                "!",
                &format!("the last run died during '{p}': counting it as unstable"),
                Tone::Warn,
            );
        }
        t
    }

    fn save(&self) {
        self.saver.save_pass(self.which, &self.st);
    }

    fn dpl(&self) -> u32 {
        self.gpu.info().default_pl
    }

    fn set(&self, core: i32, mem: i32, clock: Option<u32>) -> Settings {
        Settings {
            core,
            mem,
            clock,
            power_mw: self.dpl(),
        }
    }

    // ------------------------------------------------------------- crash-proof memo
    /// Run `f` once per key and remember the result, across reboots too: the key is saved as
    /// pending before the test, so a hard crash counts as a failure next time.
    fn probe(&mut self, key: &str, f: impl FnOnce(&mut Self) -> R<Value>) -> R<Value> {
        if let Some(v) = self.st.tried.get(key) {
            return Ok(v.clone());
        }
        self.st.pending = Some(key.to_string());
        self.save();
        let r = f(self);
        self.st.pending = None;
        match r {
            Ok(v) => {
                self.st.tried.insert(key.into(), v.clone());
                self.save();
                Ok(v)
            }
            Err(Stop::Fatal(m)) => {
                self.st.tried.insert(key.into(), Value::Bool(false));
                self.save();
                Err(Stop::Fatal(m))
            }
            Err(e) => {
                self.save();
                Err(e)
            }
        }
    }

    // ------------------------------------------------------------- running tests
    fn params(&self, kind: Kind, seconds: f64, warmup: f64, per_frame: Option<u32>) -> Params {
        let a = &self.a;
        Params {
            kind,
            seconds,
            warmup,
            bus: a.bus.clone(),
            width: a.width,
            height: a.height,
            iters: if kind == Kind::Paced {
                self.st.paced_iters.unwrap_or(a.iters_paced)
            } else {
                a.iters_heavy
            },
            alu: if kind == Kind::Paced {
                a.alu_paced
            } else {
                self.st.heavy_alu.unwrap_or(a.alu_heavy)
            },
            per_frame: per_frame.or(self.st.per_frame).unwrap_or(6),
            fps: a.fps,
            mem_mb: a.mem_mb,
            inject: false,
        }
    }

    fn healthy(&self) -> R<bool> {
        if self.gpu.temp().is_err() {
            return Ok(false);
        }
        let p = self.params(Kind::Ping, 0.0, 0.0, None);
        match self.runner.run(&p, 200, 200.0) {
            Ok(o) => Ok(o.reason.is_none()),
            Err(Stop::Interrupted) => Err(Stop::Interrupted),
            Err(_) => Ok(false),
        }
    }

    fn recover(&self) -> R<()> {
        self.ui.event(
            "↻",
            "driver crashed, waiting for the GPU to recover...",
            Tone::Warn,
        );
        self.gpu.reset();
        for attempt in 0..4 {
            self.clock.sleep(5.0 + attempt as f64 * 10.0)?;
            if self.healthy()? {
                self.ui.event("✓", "GPU recovered, continuing", Tone::Good);
                return Ok(());
            }
        }
        Err(Stop::Fatal(
            "the GPU stopped responding after a crash".into(),
        ))
    }

    fn cooldown(&self, target: u32, max_wait: f64, text: &str) -> R<()> {
        self.gpu.reset();
        self.ui.event("❄", text, Tone::Info);
        let end = self.clock.now() + max_wait;
        let lim = self.a.vram_limit - 15.0;
        while self.clock.now() < end {
            let hot = self.gpu.temp().map(|t| t > target).unwrap_or(false)
                || self.gpu.vram().is_some_and(|v| v > lim);
            if !hot {
                break;
            }
            self.clock.sleep(3.0)?;
        }
        Ok(())
    }

    /// Apply settings and run one test. Overheating is not a failure of the settings: it
    /// cools down and retries, and stops the run (Stop::Thermal) if cooling can't keep up.
    fn test(
        &mut self,
        kind: Kind,
        s: Settings,
        seconds: f64,
        warmup: f64,
        per_frame: Option<u32>,
        stock: bool,
    ) -> R<(bool, Option<TestRow>)> {
        let mut desc = if stock {
            format!("{} · stock", kind.text())
        } else {
            format!(
                "{} · core {:+} · mem {:+} · max {}",
                kind.text(),
                s.core,
                s.mem,
                s.clock.map_or("stock".into(), |c| c.to_string())
            )
        };
        if s.power_mw != self.dpl() {
            desc += &format!(" · {:.0} W", s.power_mw as f64 / 1000.0);
        }
        let short = if matches!(kind, Kind::Mem | Kind::Combo) {
            format!("mem {:+}", s.mem)
        } else {
            format!(
                "{:+} @{}",
                s.core,
                s.clock.map_or("stock".into(), |c| c.to_string())
            )
        };
        let p = self.params(kind, seconds, warmup, per_frame);
        let mut out = None;
        for attempt in 0..3 {
            let applied = if stock {
                self.gpu.reset();
                Ok(())
            } else {
                self.gpu.apply(&s)
            };
            if let Err(e) = applied {
                self.ui
                    .event("✗", &format!("could not apply settings: {e:#}"), Tone::Bad);
                return Ok((false, None));
            }
            self.clock.sleep(1.5)?;
            self.ui
                .test_start(kind.short(), &desc, seconds + warmup, &short);
            let o = self.runner.run(&p, self.a.temp_limit, self.a.vram_limit)?;
            if o.reason == Some(Reason::Overheat) {
                self.ui.test_end(false, "too hot", "too hot (not counted)");
                if attempt == 2 {
                    self.gpu.reset();
                    return Err(Stop::Thermal);
                }
                self.cooldown(
                    62 - 7 * attempt,
                    300.0,
                    &format!("too hot, cooling down to {}°C", 62 - 7 * attempt),
                )?;
                continue;
            }
            out = Some(o);
            break;
        }
        let o = out.unwrap();
        let sm = &o.summ;
        let mut row = TestRow {
            avg_power: sm.power,
            core_clk: sm.core_clk,
            mem_clk: sm.mem_clk,
            temp: sm.temp,
            limited_by: sm.limit.clone(),
            pstate: sm.pstate,
            vram_temp: sm.vram,
            ..Default::default()
        };
        let mut log = Row {
            time: now_hms(),
            pass: self.tag.into(),
            workload: format!("{kind:?}").to_lowercase(),
            core: s.core,
            mem: s.mem,
            max_clock: s.clock,
            power_limit: s.power_mw / 1000,
            avg_power: sm.power,
            core_clk: sm.core_clk,
            mem_clk: sm.mem_clk,
            pstate: sm.pstate,
            temp: sm.temp,
            vram_temp: sm.vram,
            limited_by: sm.limit.clone(),
            ..Default::default()
        };
        if let Some(reason) = o.reason {
            log.result = format!("{reason:?}").to_lowercase();
            self.log.write(&log);
            let mut msg = if reason == Reason::Hung {
                "HUNG".to_string()
            } else {
                "CRASH".to_string()
            };
            if !o.why.is_empty() && reason == Reason::Crash {
                msg += &format!(" ({})", o.why);
            }
            self.ui.test_end(false, &msg, &log.result);
            self.recover()?;
            return Ok((false, Some(row)));
        }
        let res = o.res.unwrap_or_default();
        row.core_errors = res.core_errors;
        row.mem_errors = res.mem_errors;
        if let Some(h) = &res.hash {
            let key = p.golden_key();
            if let Some(g) = self.st.golden.get(&key)
                && g != h
            {
                row.core_errors += 1; // this run's reference image differs from stock's
            }
            row.hash = Some(h.clone());
        }
        let errors = row.core_errors + row.mem_errors;
        let w = format!("{:.0} W", row.avg_power);
        let (info, brief) = match kind {
            Kind::Heavy => {
                row.perf = res.perf.unwrap_or(0.0);
                (
                    format!("{:.1} Gi/s", row.perf),
                    format!("{:.1} Gi/s {w}", row.perf),
                )
            }
            Kind::Paced => {
                row.busy = res.busy.unwrap_or(0.0);
                row.missed = res.missed.unwrap_or(0);
                (
                    format!("busy {:.0}%, {:.0} MHz", row.busy * 100.0, row.core_clk),
                    w.clone(),
                )
            }
            Kind::Mem | Kind::Combo => {
                row.bw = res.bw.unwrap_or(0.0);
                (format!("{:.1} GB/s", row.bw), format!("{:.0} GB/s", row.bw))
            }
            Kind::Ping => (String::new(), String::new()),
        };
        log.result = if errors == 0 {
            "ok".into()
        } else {
            "math errors".into()
        };
        log.core_errors = row.core_errors;
        log.mem_errors = row.mem_errors;
        log.perf = res.perf;
        log.bw = res.bw;
        log.busy = res.busy;
        log.missed = res.missed;
        log.frames = res.frames;
        self.log.write(&log);
        if errors > 0 {
            self.ui
                .test_end(false, &format!("MATH ERRORS ({errors})"), "math errors");
        } else {
            self.ui
                .test_end(true, &format!("{info}, {w}, {}°C", row.temp), &brief);
        }
        Ok((errors == 0, Some(row)))
    }

    // ------------------------------------------------------------- baseline
    pub fn baseline(&mut self) -> R<Baseline> {
        if let Some(b) = self.st.baseline.clone() {
            self.show_stock(&b);
            return Ok(b);
        }
        self.ui.set_phase(
            "Measuring stock performance (three times)",
            "Your card at default settings, warmed up. Measured three times to know how much results vary by \
             chance, so later decisions ignore that noise.",
        );
        let s0 = self.set(0, 0, None);
        let fail = |m: &str| Stop::Exit(format!("{m} See the log file."));
        let pl_w = self.dpl() as f64 / 1000.0;

        // Full load: make it heavier until it reaches the power limit at stock, like a heavy
        // game. An undervolt that only holds at low power can still crash at high power
        // (more current, more voltage drop).
        let mut alu = self.st.heavy_alu.unwrap_or(self.a.alu_heavy).max(1);
        let mut warm = self.a.warm;
        let (heavy, limited) = loop {
            self.st.heavy_alu = Some(alu);
            let (ok, r) = self.test(Kind::Heavy, s0, self.a.measure, warm, None, true)?;
            let r = r.filter(|_| ok).ok_or_else(|| {
                fail("The stock test failed, so something is wrong with the setup.")
            })?;
            let limited = r.limited_by.contains("power") || r.avg_power >= 0.9 * pl_w;
            if limited || alu >= 512 {
                break (r, limited);
            }
            self.ui.event(
                "•",
                &format!(
                    "full load drew only {:.0} W of {pl_w:.0} W: making it heavier",
                    r.avg_power
                ),
                Tone::Normal,
            );
            alu = (alu * 2).min(512);
            warm = 10.0;
        };
        if limited {
            self.ui.event(
                "✓",
                &format!(
                    "full load reaches the power limit ({:.0} W), like a heavy game",
                    heavy.avg_power
                ),
                Tone::Good,
            );
        } else {
            self.ui.event(
                "!",
                &format!("full load reaches only {:.0} W of {pl_w:.0} W: results are tested below gaming power", heavy.avg_power),
                Tone::Warn,
            );
        }
        let key = self.params(Kind::Heavy, 0.0, 0.0, None).golden_key();
        self.st
            .golden
            .insert(key, heavy.hash.clone().unwrap_or_default());
        let (ok, heavy2) = self.test(Kind::Heavy, s0, self.a.measure, 5.0, None, true)?;
        let heavy2 = heavy2
            .filter(|_| ok)
            .ok_or_else(|| fail("The second stock test failed: the card is unstable at stock."))?;
        let (ok, heavy3) = self.test(Kind::Heavy, s0, self.a.measure, 5.0, None, true)?;
        let heavy3 = heavy3
            .filter(|_| ok)
            .ok_or_else(|| fail("The third stock test failed: the card is unstable at stock."))?;

        // Game-like load: about 60% busy at the target frame rate, so there are real idle gaps
        // between frames (load changes every frame, like a game). Adjusts the passes per frame
        // and, when one pass is already too much, the work per pass.
        let mut pf = self.st.per_frame.unwrap_or(1).max(1);
        let mut it = self.st.paced_iters.unwrap_or(self.a.iters_paced).max(1);
        for attempt in 0..8 {
            self.st.per_frame = Some(pf);
            self.st.paced_iters = Some(it);
            let (ok, r) = self.test(Kind::Paced, s0, 8.0, 0.0, None, true)?;
            let r = r
                .filter(|_| ok)
                .ok_or_else(|| fail("The stock game-like test failed."))?;
            if (0.45..=0.75).contains(&r.busy) {
                break;
            }
            if attempt == 7 {
                self.ui.event(
                    "!",
                    &format!(
                        "couldn't tune the game-like load to 60% busy (now {:.0}%)",
                        r.busy * 100.0
                    ),
                    Tone::Warn,
                );
                break;
            }
            let work = (pf * it) as f64 * 0.6 / r.busy.max(0.01);
            if work >= 2.0 * it as f64 {
                pf = (work / it as f64).round().max(1.0) as u32;
            } else {
                pf = 1;
                it = (work.round() as u32).max(1);
            }
        }
        let (ok, paced) = self.test(Kind::Paced, s0, self.a.measure, 0.0, None, true)?;
        let paced = paced
            .filter(|_| ok)
            .ok_or_else(|| fail("The stock game-like test failed."))?;
        let key = self.params(Kind::Paced, 0.0, 0.0, None).golden_key();
        self.st
            .golden
            .insert(key, paced.hash.clone().unwrap_or_default());
        self.ui.event(
            "•",
            &format!(
                "game-like load: {pf} pass(es) of {it} steps per frame, {:.0}% busy at {} fps",
                paced.busy * 100.0,
                self.a.fps
            ),
            Tone::Normal,
        );
        let (ok, paced2) = self.test(Kind::Paced, s0, self.a.measure, 0.0, None, true)?;
        let paced2 = paced2
            .filter(|_| ok)
            .ok_or_else(|| fail("The second stock game-like test failed."))?;
        let (ok, paced3) = self.test(Kind::Paced, s0, self.a.measure, 0.0, None, true)?;
        let paced3 = paced3
            .filter(|_| ok)
            .ok_or_else(|| fail("The third stock game-like test failed."))?;
        let hs = [&heavy, &heavy2, &heavy3];
        let ps = [&paced, &paced2, &paced3];
        let (m, sp) = (mean, spread);
        let b = Baseline {
            perf: m(&hs.map(|r| r.perf)),
            heavy_power: m(&hs.map(|r| r.avg_power)),
            heavy_clk: m(&hs.map(|r| r.core_clk)),
            paced_power: m(&ps.map(|r| r.avg_power)),
            paced_clk: m(&ps.map(|r| r.core_clk)),
            pstate: paced3.pstate,
            noise: Noise {
                perf: sp(&hs.map(|r| r.perf)),
                heavy_power: sp(&hs.map(|r| r.avg_power)),
                paced_power: sp(&ps.map(|r| r.avg_power)),
                clk: sp(&ps.map(|r| r.core_clk)),
            },
            heavy_limited: limited,
            power_limit_w: pl_w,
        };
        let n = &b.noise;
        self.ui.event(
            "•",
            &format!(
                "stock noise: perf ±{:.1}%, clock ±{:.1}%, power ±{:.1}%",
                n.perf * 100.0,
                n.clk * 100.0,
                n.paced_power * 100.0
            ),
            Tone::Normal,
        );
        self.st.baseline = Some(b.clone());
        self.save();
        self.show_stock(&b);
        Ok(b)
    }

    fn show_stock(&self, b: &Baseline) {
        self.ui.finding(
            "stock",
            Finding {
                setting: Some("Stock".into()),
                offset: Some("+0".into()),
                perf: Some(format!("{:.0} MHz", b.paced_clk)),
                heavy: Some(format!("{:.0} W", b.heavy_power)),
                paced: Some(format!("{:.0} W", b.paced_power)),
                state: Some(RowState::Done),
                ..Default::default()
            },
        );
    }

    // ------------------------------------------------------------- calibration
    /// Before trusting any result: check the tests run in P0 like games, measure how much real
    /// memory clock one offset unit gives, and check the core offset really raises the clock.
    pub fn calibrate(&mut self) -> R<()> {
        if self.st.calib.is_none() {
            self.ui.set_phase(
                "Checking how the card runs the test loads",
                "Games run in the P0 power state. The test loads must too, or the results wouldn't apply to games.",
            );
            let (ok0, r0) = self.test(Kind::Mem, self.set(0, 0, None), 8.0, 0.0, None, true)?;
            let r0 = r0.filter(|_| ok0).ok_or_else(|| {
                Stop::Exit("The stock memory test failed. See the log file.".into())
            })?;
            if r0.pstate != Some(0) && !self.a.ignore_pstate {
                self.gpu.reset();
                return Err(Stop::Exit(format!(
                    "The test loads run in P-state P{} instead of P0 (the state games use), so results wouldn't apply \
                     to games.\nIs a CUDA program running at the same time? Close it and try again, or run with \
                     --ignore-pstate if you know why.",
                    r0.pstate.map_or("?".into(), |p| p.to_string())
                )));
            }
            if r0.mem_clk + 1.0 < self.gpu.info().max_mem_clk as f64 * 0.99 {
                self.ui.event(
                    "!",
                    &format!(
                        "memory ran at {:.0} MHz, below the card's {} MHz",
                        r0.mem_clk,
                        self.gpu.info().max_mem_clk
                    ),
                    Tone::Warn,
                );
            }
            let s500 = self.set(0, 500, None);
            let v = self.probe("calib:mem500", |t| {
                let (ok, r) = t.test(Kind::Mem, s500, 8.0, 0.0, None, false)?;
                Ok(r.filter(|_| ok)
                    .map_or(Value::Bool(false), |r| json!(r.mem_clk)))
            })?;
            let k = v.as_f64().map_or(0.0, |c| (c - r0.mem_clk) / 500.0);
            let k = if k > 0.05 { k } else { 0.0 };
            if k == 0.0 {
                self.ui.event(
                    "!",
                    "the memory clock doesn't respond to the offset: memory stays at stock",
                    Tone::Bad,
                );
            } else {
                self.ui.event(
                    "✓",
                    &format!(
                        "tests run in P0 like games; memory {:.0} MHz, +500 offset = +{:.0} MHz",
                        r0.mem_clk,
                        500.0 * k
                    ),
                    Tone::Good,
                );
            }
            self.st.calib = Some(Calib {
                pstate: r0.pstate,
                mem_clk: r0.mem_clk,
                k,
            });
            self.save();
        }
        self.check_core_offset()
    }

    fn check_core_offset(&mut self) -> R<()> {
        if self.st.core_check_done || self.a.skip_offset_check {
            return Ok(());
        }
        self.ui.set_phase(
            "Checking that the core offset works",
            "A +60 offset must raise the clock in the game-like load. If it doesn't, every later result would be meaningless.",
        );
        let base_clk = self.st.baseline.as_ref().map_or(0.0, |b| b.paced_clk);
        let s = self.set(60, 0, None);
        let v = self.probe("corecheck:60", |t| {
            let (ok, r) = t.test(Kind::Paced, s, 15.0, 0.0, None, false)?;
            Ok(r.filter(|_| ok).map_or(
                Value::Bool(false),
                |r| json!({"clk": r.core_clk, "pstate": r.pstate}),
            ))
        })?;
        match v["clk"].as_f64() {
            None => self.ui.event(
                "!",
                "already +60 failed: this card has very little headroom",
                Tone::Warn,
            ),
            Some(clk) => {
                let delta = clk - base_clk;
                if delta < 30.0 {
                    self.gpu.reset();
                    return Err(Stop::Exit(format!(
                        "The core offset has no clear effect: +60 changed the clock in the game-like load by only \
                         {delta:+.0} MHz (P-state P{}).\nThe driver probably applies the offset to a different P-state, \
                         so the search would find nothing real. If you know better, run with --skip-offset-check.",
                        v["pstate"].as_u64().map_or("?".into(), |p| p.to_string())
                    )));
                }
                self.ui.event(
                    "✓",
                    &format!("core offset works: +60 gave {delta:+.0} MHz"),
                    Tone::Good,
                );
            }
        }
        self.st.core_check_done = true;
        self.save();
        Ok(())
    }

    // ------------------------------------------------------------- memory
    pub fn memory(&mut self) -> R<i32> {
        let label = if self.which == Pass::P1 {
            "Mem scan"
        } else {
            "Mem fine"
        };
        if let Some(m) = self.st.mem {
            self.show_mem(label, m, self.st.mem_bw);
            return Ok(m);
        }
        self.ui.finding(
            &format!("{}:mem", self.tag),
            Finding {
                setting: Some(label.into()),
                status: Some("⏳".into()),
                state: Some(RowState::Testing),
                ..Default::default()
            },
        );
        let k = self.st.calib.as_ref().map_or(0.0, |c| c.k);
        let (chosen, bw) = if k == 0.0 {
            (0, None)
        } else if let Some(m) = self.a.refine_mem {
            let step = 250;
            let mut c: Vec<i32> = [m - 2 * step, m - step, m, m + step]
                .into_iter()
                .filter(|o| *o >= 0 && *o <= self.a.mem_max)
                .collect();
            if self.st.mem_err.is_none() {
                self.st.mem_err = self.a.refine_mem_err;
            }
            if self.a.refine_mem_top && m + 2 * step <= self.a.mem_max {
                c.push(m + 2 * step);
            }
            self.ui.set_phase(
                &format!("Fine-tuning memory around +{m}"),
                "Checking 250-step neighbours with a hot GPU, where the real limit shows.",
            );
            self.hot_pick(&c)?
        } else {
            self.ui.set_phase(
                "Memory: finding the bandwidth peak",
                "Too much memory clock causes transfer retries, which show up as lost bandwidth.",
            );
            let (c, bw, top) = self.mem_sweep()?;
            self.st.mem_top_of_range = top;
            (c, bw)
        };
        self.st.mem = Some(chosen);
        self.st.mem_bw = bw;
        self.save();
        self.show_mem(label, chosen, bw);
        Ok(chosen)
    }

    fn mem_limit(&self) -> i32 {
        self.a.mem_max.min(self.gpu.info().mem_range.1)
    }

    fn mem_probe(&mut self, key: &str, off: i32, hot: bool) -> R<Value> {
        let s = self.set(0, off, None);
        let (secs, warm) = if hot {
            (self.a.combo_seconds, self.a.combo_warmup)
        } else {
            (self.a.mem_seconds, 0.0)
        };
        self.probe(key, |t| {
            let (ok, r) = t.test(
                if hot { Kind::Combo } else { Kind::Mem },
                s,
                secs,
                warm,
                None,
                false,
            )?;
            Ok(match r.filter(|_| ok) {
                Some(r) => json!({"bw": r.bw, "mem_clk": r.mem_clk, "vram_temp": r.vram_temp}),
                None => Value::Bool(false),
            })
        })
    }

    fn mem_sweep(&mut self) -> R<(i32, Option<f64>, bool)> {
        let lim = self.mem_limit();
        let r0 = self.mem_probe("mem:0", 0, false)?;
        let r0b = self.mem_probe("mem:0b", 0, false)?;
        let (Some(b0), Some(b1)) = (r0["bw"].as_f64(), r0b["bw"].as_f64()) else {
            self.ui.event(
                "!",
                "memory test failed at +0, leaving memory at stock",
                Tone::Bad,
            );
            return Ok((0, None, false));
        };
        let noise = rel_noise(b0, b1);
        self.st.bw_noise = Some(noise);
        let mut results: Vec<(i32, f64, f64)> =
            vec![(0, (b0 + b1) / 2.0, r0["mem_clk"].as_f64().unwrap_or(0.0))];
        let (mut peak, mut drops) = (results[0].1, 0);
        let tol = 0.005f64.max(2.0 * noise);
        let mut off = self.a.mem_step;
        while off <= self.a.mem_max {
            if off > lim {
                self.ui.event(
                    "•",
                    &format!("+{off} is beyond the driver's offset limit"),
                    Tone::Warn,
                );
                break;
            }
            let r = self.mem_probe(&format!("mem:{off}"), off, false)?;
            let (Some(bw), Some(clk)) = (r["bw"].as_f64(), r["mem_clk"].as_f64()) else {
                self.ui.event(
                    "•",
                    &format!("memory limit reached at +{off} (errors or crash)"),
                    Tone::Warn,
                );
                self.note_mem_err(off);
                break;
            };
            results.push((off, bw, clk));
            if bw > peak {
                (peak, drops) = (bw, 0);
            } else if bw < peak * (1.0 - tol) {
                drops += 1;
                if drops >= 2 {
                    self.ui.event(
                        "•",
                        "bandwidth falling: transfer retries are taking over",
                        Tone::Warn,
                    );
                    break;
                }
            }
            off += self.a.mem_step;
        }
        if results.len() > 1 && results.last().unwrap().2 <= results[0].2 {
            self.ui.event(
                "!",
                "the memory clock doesn't change with the offset, leaving it at stock",
                Tone::Bad,
            );
            return Ok((0, None, false));
        }
        let cap = self.mem_cap();
        results.retain(|r| r.0 <= cap);
        let (i, _) = pick_mem(&results, noise);
        let best = results[i];
        let top = best.0 + self.a.mem_step > self.a.mem_max.min(lim);
        self.ui.event(
            "•",
            &format!(
                "cool peak at +{} ({:.1} vs {:.1} GB/s, noise ±{:.1}%)",
                best.0,
                best.1,
                results[0].1,
                noise * 100.0
            ),
            Tone::Normal,
        );
        self.ui.set_phase(
            "Memory: re-checking the best steps with a hot GPU",
            "Hot memory needs more retries, so the peak can move lower.",
        );
        let c: Vec<i32> = [
            best.0,
            best.0 - self.a.mem_step,
            best.0 - 2 * self.a.mem_step,
        ]
        .into_iter()
        .filter(|o| *o >= 0)
        .collect();
        let (chosen, bw) = self.hot_pick(&c)?;
        Ok((chosen, bw, top && chosen == best.0))
    }

    /// Highest memory offset the tuner may use: MEM_ERROR_MARGIN below the first errors.
    fn mem_cap(&self) -> i32 {
        self.st.mem_err.map_or(i32::MAX, |e| e - MEM_ERROR_MARGIN)
    }

    fn note_mem_err(&mut self, off: i32) {
        self.st.mem_err = Some(self.st.mem_err.map_or(off, |e| e.min(off)));
        self.save();
    }

    fn hot_pick(&mut self, cands: &[i32]) -> R<(i32, Option<f64>)> {
        let noise = self.st.bw_noise.unwrap_or(0.0);
        let mut c: Vec<i32> = cands.to_vec();
        c.sort_unstable();
        c.dedup();
        if let Some(e) = self.st.mem_err {
            let cap = self.mem_cap();
            if c.iter().any(|o| *o > cap) {
                self.ui.event(
                    "•",
                    &format!("errors started at +{e}: staying at +{cap} or below"),
                    Tone::Normal,
                );
            }
        }
        let mut hot: BTreeMap<i32, (f64, f64)> = BTreeMap::new();
        for off in c {
            if off > self.mem_limit() || off > self.mem_cap() {
                break;
            }
            let r = self.mem_probe(&format!("memhot:{off}"), off, true)?;
            let (Some(bw), Some(clk)) = (r["bw"].as_f64(), r["mem_clk"].as_f64()) else {
                self.ui
                    .event("•", &format!("+{off} failed with a hot GPU"), Tone::Warn);
                self.note_mem_err(off);
                break;
            };
            if let Some(vt) = r["vram_temp"].as_f64()
                && !hot.is_empty()
                && vt >= self.a.vram_soft
            {
                let safer = *hot.keys().last().unwrap();
                self.ui.event(
                    "•",
                    &format!("+{off} heats the memory to {vt:.0}°C: keeping the safer +{safer}"),
                    Tone::Normal,
                );
                break;
            }
            hot.insert(off, (bw, clk));
        }
        let cap = self.mem_cap();
        let pts: Vec<(i32, f64, f64)> = hot
            .iter()
            .filter(|(o, _)| **o <= cap)
            .map(|(&o, &(bw, clk))| (o, bw, clk))
            .collect();
        if pts.is_empty() {
            return Ok((0, None));
        }
        let (i, stop) = pick_mem(&pts, noise);
        let chosen = pts[i].0;
        match stop {
            MemStop::Retries(o) => self.ui.event(
                "•",
                &format!("+{o} loses bandwidth per MHz (transfer retries): keeping +{chosen}"),
                Tone::Normal,
            ),
            MemStop::NoGain => self.ui.event(
                "•",
                &format!(
                    "+{} adds no clear bandwidth over +{chosen}: keeping +{chosen}",
                    pts.last().unwrap().0
                ),
                Tone::Normal,
            ),
            MemStop::Top => {}
        }
        self.ui
            .event("★", &format!("memory offset +{chosen}"), Tone::Star);
        Ok((chosen, Some(pts[i].1)))
    }

    fn show_mem(&self, label: &str, mem: i32, bw: Option<f64>) {
        self.ui.finding(
            &format!("{}:mem", self.tag),
            Finding {
                setting: Some(label.into()),
                offset: Some(format!("+{mem}")),
                perf: Some(bw.map_or(String::new(), |b| format!("{b:.0} GB/s"))),
                status: Some("✓".into()),
                state: Some(RowState::Done),
                ..Default::default()
            },
        );
    }

    // ------------------------------------------------------------- core
    pub fn core(&mut self, mem: i32) -> R<()> {
        let base = self.st.baseline.clone().unwrap_or_default();
        let label = if self.which == Pass::P1 {
            "Scan"
        } else {
            "Fine"
        };
        let core_max = self.a.core_max.min(self.gpu.info().core_range.1);
        let step = self.a.core_step;
        for &clk in &self.a.clocks {
            self.ui.finding(
                &format!("{}:{clk}", self.tag),
                Finding {
                    setting: Some(format!("{label} {clk}")),
                    status: Some("·".into()),
                    state: Some(RowState::Pending),
                    ..Default::default()
                },
            );
        }
        let mut prev = None;
        for clk in self.a.clocks.clone() {
            let fkey = format!("{}:{clk}", self.tag);
            if let Some(r) = self.st.clocks.get(&clk).cloned() {
                self.show_clock(&fkey, r.as_ref(), &base);
                if let Some(r) = r {
                    prev = Some(r.max_stable);
                }
                continue;
            }
            self.ui.finding(
                &fkey,
                Finding {
                    status: Some("⏳".into()),
                    state: Some(RowState::Testing),
                    ..Default::default()
                },
            );
            self.ui.set_phase(
                &format!("Core: highest stable offset at max {clk} MHz"),
                "Higher offset = lower voltage at the same clock = less power. Each step: game-like bursts, then full \
                 load at the power limit (high current is where undervolts fail).",
            );
            let hint = self.a.hints.get(&clk).copied();
            let secs = self.a.search_seconds;
            let found = search_bounds(
                |off| {
                    let s = self.set(off, mem, Some(clk));
                    let v = self.probe(&format!("core:{clk}:{mem}:{off}"), |t| {
                        let (ok, p) = t.test(Kind::Paced, s, secs, 0.0, None, false)?;
                        let ok = ok && t.test(Kind::Heavy, s, secs * 0.5, 0.0, None, false)?.0;
                        // game-like power: shows where more offset stops lowering the voltage
                        Ok(match p.filter(|_| ok) {
                            Some(p) => json!({"ok": true, "pp": p.avg_power}),
                            None => Value::Bool(false),
                        })
                    })?;
                    Ok(step_ok(&v))
                },
                hint,
                prev,
                core_max,
                step,
            )?;
            let Some((lo, _)) = found else {
                self.st.clocks.insert(clk, None);
                self.save();
                self.show_clock(&fkey, None, &base);
                continue;
            };
            prev = Some(lo);
            if lo + step > core_max {
                self.ui.event(
                    "!",
                    &format!("max {clk}: still stable at the top of the range (+{lo}); raise --core-max to search further"),
                    Tone::Warn,
                );
            }
            let mut off = (lo - self.a.core_margin).max(0);
            self.ui.event(
                "•",
                &format!("max {clk}: stable up to +{lo}, measuring with +{off} (safety margin)"),
                Tone::Normal,
            );
            let pts: Vec<(i32, f64)> = (0..=lo)
                .filter_map(|o| {
                    let v = self.st.tried.get(&format!("core:{clk}:{mem}:{o}"))?;
                    Some((o, v["pp"].as_f64()?))
                })
                .collect();
            let tol = 0.015f64.max(2.0 * base.noise.paced_power);
            if let Some(f) = floor_offset(&pts, tol).filter(|f| *f < off) {
                self.ui.event(
                    "•",
                    &format!(
                        "max {clk}: power stops falling above +{f} (the voltage is at its minimum), so +{f} is enough"
                    ),
                    Tone::Normal,
                );
                off = f;
            }
            self.ui.set_phase(
                &format!("Core: measuring max {clk} MHz with a safety margin"),
                "Full load for performance, game-like load for power at a fixed frame rate.",
            );
            let mut row = None;
            for _ in 0..4 {
                let v = self.probe(&format!("meas:{clk}:{mem}:{off}"), |t| {
                    t.measure(off, mem, clk, lo)
                })?;
                if let Ok(p) = serde_json::from_value::<Prof>(v) {
                    row = Some(p);
                    break;
                }
                if off == 0 {
                    break;
                }
                off = (off - step).max(0);
            }
            self.st.clocks.insert(clk, row.clone());
            self.save();
            self.show_clock(&fkey, row.as_ref(), &base);
        }
        Ok(())
    }

    fn measure(&mut self, off: i32, mem: i32, clk: u32, lo: i32) -> R<Value> {
        let s = self.set(off, mem, Some(clk));
        let (ok1, h) = self.test(Kind::Heavy, s, self.a.measure, self.a.warmup, None, false)?;
        let Some(h) = h.filter(|_| ok1) else {
            return Ok(Value::Bool(false));
        };
        let (ok2, p) = self.test(Kind::Paced, s, self.a.measure, 0.0, None, false)?;
        let Some(p) = p.filter(|_| ok2) else {
            return Ok(Value::Bool(false));
        };
        Ok(serde_json::to_value(Prof {
            max_clock: clk,
            core: off,
            mem,
            max_stable: lo,
            perf: h.perf,
            heavy_power: h.avg_power,
            heavy_clk: h.core_clk,
            paced_power: p.avg_power,
            paced_clk: p.core_clk,
            missed: p.missed,
            temp: h.temp.max(p.temp),
            limited_by: h.limited_by,
            power_max: false,
            power_limit_mw: None,
            vram_max: None,
            power_note: None,
        })
        .unwrap())
    }

    fn show_clock(&self, fkey: &str, r: Option<&Prof>, base: &Baseline) {
        let f = match r {
            None => Finding {
                offset: Some("unstable".into()),
                limit: Some(String::new()),
                status: Some("✗".into()),
                state: Some(RowState::Failed),
                ..Default::default()
            },
            Some(r) => {
                let dg = (game_rel(r, base) - 1.0) * 100.0;
                Finding {
                    offset: Some(format!("+{}", r.core)),
                    limit: Some(format!("+{}", r.max_stable)),
                    perf: Some(format!("{dg:+.1}%{}", if dg < -1.0 { " ▼" } else { "" })),
                    heavy: Some(format!("{:+.0} W", r.heavy_power - base.heavy_power)),
                    paced: Some(format!("{:+.0} W", r.paced_power - base.paced_power)),
                    status: Some("✓".into()),
                    state: Some(RowState::Done),
                    ..Default::default()
                }
            }
        };
        self.ui.finding(fkey, f);
    }

    // ------------------------------------------------------------- final checks
    pub fn maxpower(&mut self, prof: Prof) -> R<Prof> {
        let info = self.gpu.info().clone();
        if info.max_pl <= info.default_pl {
            return Ok(prof);
        }
        let s = Settings {
            core: prof.core,
            mem: prof.mem,
            clock: Some(prof.max_clock),
            power_mw: info.max_pl,
        };
        let (measure, warmup) = (self.a.measure, self.a.warmup);
        let v = self.probe(
            &format!("maxpower:{}:{}:{}", prof.max_clock, prof.core, prof.mem),
            |t| {
                t.ui.set_phase(
                    &format!(
                        "Checking the max power limit ({:.0} W)",
                        info.max_pl as f64 / 1000.0
                    ),
                    "Only kept if it adds clearly more performance than the measurement noise.",
                );
                let (ok, r) = t.test(Kind::Heavy, s, measure, warmup, None, false)?;
                Ok(r.filter(|_| ok).map_or(
                    Value::Bool(false),
                    |r| json!({"perf": r.perf, "power": r.avg_power}),
                ))
            },
        )?;
        let noise = self.st.baseline.as_ref().map_or(0.0, |b| b.noise.perf);
        match (v["perf"].as_f64(), v["power"].as_f64()) {
            (Some(perf), Some(power)) if perf > prof.perf * (1.0 + 0.01f64.max(2.0 * noise)) => {
                Ok(Prof {
                    perf,
                    heavy_power: power,
                    power_max: true,
                    ..prof
                })
            }
            _ => Ok(prof),
        }
    }

    /// Looks for the power-limit sweet spot of a profile. Full load with its settings, first at
    /// the default limit, then in 10 W steps down from what it actually draws. It keeps going
    /// while each step saves clearly more power than it costs performance and stops at the knee:
    /// where the card has run out of voltage to drop and only loses clock speed. `max_loss` caps
    /// the total full-load performance loss. Stops above the first unstable limit: under a power
    /// limit the card runs lower points of the shifted voltage/frequency curve, which the soak
    /// test at the chosen limit then confirms.
    pub fn power_search(&mut self, name: &str, prof: Prof, max_loss: f64) -> R<Prof> {
        let info = self.gpu.info().clone();
        let label = crate::state::profile_label(name);
        let secs = (self.a.measure * 0.5).max(10.0);
        let noise = self
            .st
            .baseline
            .as_ref()
            .map(|b| b.noise.clone())
            .unwrap_or_default();
        let min_save = 0.03f64.max(3.0 * noise.heavy_power);
        self.ui.set_phase(
            &format!("{label}: looking for the power-limit sweet spot"),
            "Full load in 10 W steps down from what these settings draw. Stops where saving more power \
             starts to cost clearly more performance.",
        );
        let key = |w: u32| format!("pl:{}:{}:{}:{w}", prof.max_clock, prof.core, prof.mem);
        let run = |t: &mut Self, mw: u32, warm: f64| -> R<Option<(f64, f64)>> {
            let s = Settings {
                core: prof.core,
                mem: prof.mem,
                clock: Some(prof.max_clock),
                power_mw: mw,
            };
            let v = t.probe(&key(mw / 1000), |t| {
                let (ok, r) = t.test(Kind::Heavy, s, secs, warm, None, false)?;
                Ok(r.filter(|_| ok).map_or(
                    Value::Bool(false),
                    |r| json!({"perf": r.perf, "power": r.avg_power}),
                ))
            })?;
            Ok(v["perf"].as_f64().zip(v["power"].as_f64()))
        };
        // reference: the default limit
        let Some((p0, w0)) = run(self, info.default_pl, self.a.warmup)? else {
            self.ui.event(
                "•",
                &format!("{label}: full load failed at the default limit, keeping it"),
                Tone::Warn,
            );
            return Ok(prof);
        };
        let floor_w = (info.default_pl as f64 * 0.7).max(info.min_pl as f64) / 1000.0;
        // first limit that actually binds: the next 10 W step below the measured draw
        let mut w = ((w0 - 2.0) / 10.0).floor() * 10.0;
        let mut points: Vec<(u32, f64, f64)> = vec![(info.default_pl, p0, w0)];
        let mut ended = format!("{floor_w:.0} W is the lowest limit it tries");
        while w >= floor_w {
            let mw = (w * 1000.0) as u32;
            let Some((perf, power)) = run(self, mw, 5.0)? else {
                ended = format!("{w:.0} W was unstable with these settings");
                self.ui.event(
                    "•",
                    &format!(
                        "{label}: {w:.0} W is unstable with these settings, stopping above it"
                    ),
                    Tone::Warn,
                );
                break;
            };
            self.ui.event(
                "•",
                &format!(
                    "{label}: {w:.0} W limit: {:+.1}% performance, {:+.1}% power",
                    (perf / p0 - 1.0) * 100.0,
                    (power / w0 - 1.0) * 100.0
                ),
                Tone::Normal,
            );
            points.push((mw, perf, power));
            let pts: Vec<(f64, f64)> = points.iter().map(|p| (p.1, p.2)).collect();
            if let Some((_, why)) = power_stop(&pts, max_loss, KNEE_ELASTICITY) {
                let loss = (1.0 - perf / p0) * 100.0;
                ended = match why {
                    PowerStop::LossCap => format!(
                        "{w:.0} W would cost {loss:.1}% performance, more than the {:.0}% this profile allows",
                        max_loss * 100.0
                    ),
                    PowerStop::Knee => format!(
                        "below it every watt costs clearly more performance ({w:.0} W: {loss:.1}% slower)"
                    ),
                };
                break;
            }
            w -= 10.0;
        }
        let pts: Vec<(f64, f64)> = points.iter().map(|p| (p.1, p.2)).collect();
        match pick_knee(&pts, max_loss, min_save, KNEE_ELASTICITY) {
            Some(i) => {
                let (mw, perf, power) = points[i];
                self.ui.event(
                    "★",
                    &format!(
                        "{label}: power limit {:.0} W saves {:.0}% power at full load for {:.1}% performance",
                        mw as f64 / 1000.0,
                        (1.0 - power / w0) * 100.0,
                        (1.0 - perf / p0) * 100.0
                    ),
                    Tone::Star,
                );
                Ok(Prof {
                    power_limit_mw: Some(mw),
                    perf,
                    heavy_power: power,
                    power_note: Some(format!("stopped here because {ended}")),
                    ..prof
                })
            }
            None => {
                self.ui.event(
                    "•",
                    &format!("{label}: a lower power limit doesn't pay off, keeping the default"),
                    Tone::Normal,
                );
                Ok(Prof {
                    power_note: Some(format!(
                        "default kept: a lower limit doesn't save at least {:.0}% power within the allowed loss ({ended})",
                        min_save * 100.0
                    )),
                    ..prof
                })
            }
        }
    }

    /// Desktop idle with the settings applied (no test running, so the card drops to its idle
    /// P-state and low clocks), each followed by a sudden burst of load. The offset shifts these
    /// low points of the curve too, and that's where Linux undervolts often crash.
    fn idle_check(&mut self, s: Settings, cycles: u32, idle_s: f64) -> R<bool> {
        for i in 0..cycles {
            if let Err(e) = self.gpu.apply(&s) {
                self.ui
                    .event("✗", &format!("could not apply settings: {e:#}"), Tone::Bad);
                return Ok(false);
            }
            self.gpu.new_xids();
            self.ui.test_start(
                "idle",
                &format!(
                    "Desktop idle · core {:+} · mem {:+} ({}/{cycles})",
                    s.core,
                    s.mem,
                    i + 1
                ),
                idle_s,
                &format!(
                    "{:+} @{}",
                    s.core,
                    s.clock.map_or("stock".into(), |c| c.to_string())
                ),
            );
            let end = self.clock.now() + idle_s;
            let (mut low, mut crashed) = (u32::MAX, None);
            while self.clock.now() < end {
                self.clock.sleep(1.0)?;
                match self.gpu.snapshot() {
                    Ok(sn) => low = low.min(sn.clk),
                    Err(e) => crashed = Some(format!("GPU stopped answering ({e})")),
                }
                if let Some(x) = self.gpu.new_xids().pop() {
                    crashed = Some(x.chars().take(100).collect());
                }
                if crashed.is_some() {
                    break;
                }
            }
            if let Some(c) = crashed {
                self.ui
                    .test_end(false, &format!("CRASH at idle ({c})"), "crash at idle");
                self.log.write(&Row {
                    time: now_hms(),
                    pass: self.tag.into(),
                    workload: "idle".into(),
                    core: s.core,
                    mem: s.mem,
                    max_clock: s.clock,
                    result: "crash".into(),
                    ..Default::default()
                });
                self.recover()?;
                return Ok(false);
            }
            self.ui.test_end(
                true,
                &format!("lowest clock {low} MHz"),
                &format!("ok, down to {low} MHz"),
            );
            let (ok, _) = self.test(Kind::Paced, s, 4.0, 0.0, None, false)?;
            if !ok {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Longer mixed test of the result: cold start, full load, game-like load, memory with a
    /// hot GPU, desktop idle. Returns the confirmed profile with numbers from these longer
    /// tests, or None if nothing passes. Each profile (`name`) has its own soak record.
    pub fn soak(&mut self, name: &str, prof: &Prof) -> R<Option<Prof>> {
        let rec = self.st.soaks.entry(name.to_string()).or_default().clone();
        if rec.done {
            return Ok(rec.result);
        }
        let pl = prof_power(prof, self.gpu.info());
        let (mut core, mut mem, clk) = (prof.core, prof.mem, prof.max_clock);
        let step = self.a.core_step;
        if let Some((c, m)) = rec.at {
            // the computer died during the soak last time
            (core, mem) = (c, m);
            if core > 0 {
                core = (core - step).max(0);
            } else if mem > 0 {
                mem = (mem - 250).max(0);
            } else {
                return self.finish_soak(name, None);
            }
            self.ui.event("!", &format!("the last run died during the soak test: backing off to core {core:+}, mem {mem:+}"), Tone::Warn);
        }
        loop {
            self.st.soaks.entry(name.to_string()).or_default().at = Some((core, mem));
            self.save();
            let s = Settings {
                core,
                mem,
                clock: Some(clk),
                power_mw: pl,
            };
            self.ui.set_phase(
                &format!(
                    "Soak test: cold start, {:.0} min of mixed load, desktop idle",
                    self.a.soak / 60.0
                ),
                "Making sure the settings hold up cold, hot, for longer and at idle.",
            );
            self.cooldown(
                50,
                240.0,
                "cooling down for a cold-start test (up to 4 min)",
            )?;
            let parts = [
                (Kind::Paced, 45.0, 0.0),
                (Kind::Heavy, self.a.soak * 0.4, self.a.warmup),
                (Kind::Paced, self.a.soak * 0.3, 0.0),
                (Kind::Combo, self.a.soak * 0.3, 0.0),
            ];
            let (mut fail, mut rows, mut vmax) = (None, vec![], None::<f64>);
            for (kind, secs, warm) in parts {
                let (ok, r) = self.test(kind, s, secs, warm, None, false)?;
                if let Some(v) = r.as_ref().and_then(|r| r.vram_temp) {
                    vmax = Some(vmax.map_or(v, |m| m.max(v)));
                }
                if !ok {
                    // blame the part that failed; a crash in the memory test is likely memory
                    fail = Some(
                        if kind == Kind::Combo && r.as_ref().is_none_or(|r| r.core_errors == 0) {
                            Fail::Mem
                        } else {
                            Fail::Core
                        },
                    );
                    break;
                }
                rows.push(r.unwrap());
            }
            if fail.is_none() {
                self.ui.set_phase(
                    "Soak test: desktop idle and sudden load",
                    "Leave the computer alone for a minute (light browsing is fine).",
                );
                if !self.idle_check(s, 3, 20.0)? {
                    fail = Some(Fail::Core);
                }
            }
            match fail {
                None => {
                    let (h, p) = (&rows[1], &rows[2]);
                    let r = Prof {
                        core,
                        mem,
                        vram_max: vmax,
                        perf: h.perf,
                        heavy_power: h.avg_power,
                        paced_power: p.avg_power,
                        paced_clk: p.core_clk,
                        ..prof.clone()
                    };
                    return self.finish_soak(name, Some(r));
                }
                Some(f) => {
                    if f == Fail::Mem && mem > 0 {
                        mem = (mem - 250).max(0);
                    } else if core > 0 {
                        core = (core - step).max(0);
                    } else if mem > 0 {
                        mem = (mem - 250).max(0);
                    } else {
                        return self.finish_soak(name, None);
                    }
                    self.ui.event(
                        "!",
                        &format!("soak failed ({f:?}), backing off to core {core:+}, mem {mem:+}"),
                        Tone::Warn,
                    );
                }
            }
        }
    }

    fn finish_soak(&mut self, name: &str, r: Option<Prof>) -> R<Option<Prof>> {
        let s = self.st.soaks.entry(name.to_string()).or_default();
        s.result = r.clone();
        s.done = true;
        s.at = None;
        self.save();
        let label = crate::state::profile_label(name);
        match &r {
            Some(p) => self.ui.event(
                "★",
                &format!(
                    "{label}: soak test passed (max {} MHz, core {:+}, mem {:+})",
                    p.max_clock, p.core, p.mem
                ),
                Tone::Star,
            ),
            None => self.ui.event(
                "✗",
                &format!("{label}: soak test failed even at +0"),
                Tone::Bad,
            ),
        }
        Ok(r)
    }
}

// ================================================================= time estimate
const TEST_OVERHEAD: f64 = 5.0;

pub fn est_pass1(p: &PassArgs) -> f64 {
    let steps = (p.mem_max / p.mem_step + 2) as f64;
    let n = if p.clocks.is_empty() {
        AUTO_CLOCKS.len()
    } else {
        p.clocks.len()
    } as f64;
    // baseline (2 full, ~3 calibration + 2 game-like), calibration, memory, 6 search steps per
    // clock (game-like + half as long full load each), measurement
    let tests = 11.0 + 2.0 + steps + 3.0 + n * 14.0;
    p.warm
        + 7.0 * p.measure
        + 3.0 * 8.0
        + 31.0
        + steps * p.mem_seconds
        + 3.0 * (p.combo_seconds + p.combo_warmup)
        + n * (6.0 * 1.5 * p.search_seconds + 2.0 * p.measure + p.warmup)
        + tests * TEST_OVERHEAD
}

pub fn est_pass2(p: &PassArgs, n_clocks: usize) -> f64 {
    let n = n_clocks as f64;
    let tests = 8.0 + 5.0 + n * 10.0;
    p.warm
        + 7.0 * p.measure
        + 8.0
        + 5.0 * (p.combo_seconds + p.combo_warmup)
        + n * (4.0 * 1.5 * p.search_seconds + 2.0 * p.measure + p.warmup)
        + tests * TEST_OVERHEAD
}

/// power-limit search for one profile: reference + ~7 short full-load steps
pub fn est_power(p: &PassArgs) -> f64 {
    (p.measure * 0.5).max(10.0)
        + p.warmup
        + 7.0 * ((p.measure * 0.5).max(10.0) + 5.0)
        + 8.0 * TEST_OVERHEAD
}

pub fn est_soak(p: &PassArgs) -> f64 {
    p.soak + 45.0 + 120.0 + 3.0 * 24.0 + 10.0 * TEST_OVERHEAD
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mem_rule() {
        // the user's RTX 4080 Super, pass 2 with a hot GPU: bandwidth follows the clock all the
        // way (no retries); each 250 step adds ~1.1%, less than the 1.56% noise threshold, but
        // the range as a whole clearly gains
        let hot = [
            (2500, 674.5, 12751.0),
            (2750, 681.9, 12876.0),
            (3000, 688.6, 13001.0),
            (3250, 695.6, 13126.0),
        ];
        assert_eq!(pick_mem(&hot, 0.0078), (3, MemStop::Top));
        // bandwidth flattens from +3000 on: retries
        let retries = [
            (2500, 674.5, 12751.0),
            (2750, 681.9, 12876.0),
            (3000, 682.5, 13001.0),
            (3250, 683.0, 13126.0),
        ];
        assert_eq!(pick_mem(&retries, 0.003), (1, MemStop::Retries(3000)));
        // a small gain lost in the noise: keep the lowest
        assert_eq!(pick_mem(&hot[..2], 0.02), (0, MemStop::NoGain));
    }

    #[test]
    fn bisection_finds_limit_with_and_without_hints() {
        for limit in [0, 15, 90, 180, 255, 450] {
            for hint in [None, Some(60), Some(180), Some(400)] {
                let mut calls = vec![];
                let r = search_bounds(
                    |o| {
                        calls.push(o);
                        Ok(o <= limit)
                    },
                    hint,
                    None,
                    450,
                    15,
                )
                .unwrap();
                assert_eq!(
                    r.map(|x| x.0),
                    Some(limit),
                    "limit {limit} hint {hint:?} calls {calls:?}"
                );
            }
        }
        assert_eq!(
            search_bounds(|_| Ok(false), None, None, 450, 15).unwrap(),
            None
        );
    }

    #[test]
    fn failed_hint_lowers_upper_bound() {
        // limit 30, hint 180: below=120 fails -> 'above' (240) must not be tried
        let mut calls = vec![];
        search_bounds(
            |o| {
                calls.push(o);
                Ok(o <= 30)
            },
            Some(180),
            None,
            450,
            15,
        )
        .unwrap();
        assert!(!calls.contains(&240), "{calls:?}");
    }

    /// (performance, power) for a card drawing `natural` W: perf ~ (limit/natural)^0.3 while the
    /// voltage can still drop, ~ linear in power below `vmin` W.
    fn curve(natural: f64, vmin: f64, limits: &[f64]) -> Vec<(f64, f64)> {
        let mut v = vec![(100.0, natural)];
        for &l in limits {
            let w = l.min(natural);
            let mut perf = 100.0 * (w / natural).powf(0.3);
            if w < vmin {
                perf *= w / vmin;
            }
            v.push((perf, w));
        }
        v
    }

    #[test]
    fn knee_without_a_knee_stops_at_the_loss_cap() {
        let pts = curve(
            300.0,
            0.0,
            &[290.0, 280.0, 270.0, 260.0, 250.0, 240.0, 230.0],
        );
        // 3% cap: 270 W costs 3.1%, so 280 W (2.1%) is the lowest allowed
        assert_eq!(pick_knee(&pts, 0.03, 0.03, KNEE_ELASTICITY), Some(2));
        assert_eq!(
            power_stop(&pts[..4], 0.03, KNEE_ELASTICITY),
            Some((3, PowerStop::LossCap))
        );
        assert_eq!(power_stop(&pts[..3], 0.03, KNEE_ELASTICITY), None);
    }

    #[test]
    fn knee_stops_where_power_starts_costing_performance() {
        // voltage floor at 270 W: below that every watt costs ~1:1
        let pts = curve(300.0, 270.0, &[290.0, 280.0, 270.0, 260.0, 250.0]);
        assert_eq!(pick_knee(&pts, 0.10, 0.03, KNEE_ELASTICITY), Some(3)); // 270 W
    }

    #[test]
    fn knee_ignores_limits_that_dont_bite_and_needs_real_savings() {
        // the card only draws 275 W: 290 and 280 W limits change nothing
        let pts = curve(275.0, 0.0, &[290.0, 280.0, 270.0, 260.0, 250.0]);
        assert_eq!(pick_knee(&pts, 0.03, 0.03, KNEE_ELASTICITY), Some(5)); // 250 W: -2.8%
        // saving only ~2% isn't worth a lower limit
        assert_eq!(pick_knee(&pts[..4], 0.03, 0.03, KNEE_ELASTICITY), None);
    }

    #[test]
    fn floor_offset_finds_where_power_stops_falling() {
        // the user's card, pass 1, game-like power per stable offset
        let at_2400 = [
            (0, 107.1),
            (225, 101.0),
            (345, 100.7),
            (405, 100.2),
            (435, 100.1),
            (450, 99.8),
        ];
        assert_eq!(floor_offset(&at_2400, 0.0174), Some(225));
        let at_2550 = [(0, 118.2), (225, 101.3), (330, 100.3), (345, 100.6)];
        assert_eq!(floor_offset(&at_2550, 0.0174), Some(225));
        // still falling at 2700 and 2850: no flat part confirmed by two higher offsets
        let at_2700 = [
            (0, 140.6),
            (165, 117.6),
            (240, 115.0),
            (255, 112.0),
            (270, 108.3),
            (285, 110.2),
        ];
        assert_eq!(floor_offset(&at_2700, 0.0174), None);
        let at_2850 = [
            (0, 147.5),
            (135, 147.9),
            (210, 132.9),
            (240, 127.6),
            (255, 127.3),
            (270, 124.4),
        ];
        assert_eq!(floor_offset(&at_2850, 0.0174), None);
        assert_eq!(floor_offset(&at_2850[..2], 0.0174), None);
        // overclock region (cap above the stock boost): power rises with the offset
        let oc = [
            (0, 200.0),
            (30, 202.0),
            (60, 204.4),
            (90, 206.5),
            (105, 207.6),
        ];
        assert_eq!(floor_offset(&oc, 0.015), None);
    }

    #[test]
    fn spread_of_three_runs() {
        assert!((spread(&[490.775, 494.398, 498.8]) - 0.0162).abs() < 0.0002);
        assert_eq!(spread(&[5.0, 5.0, 5.0]), 0.0);
    }

    #[test]
    fn auto_clocks_follow_the_stock_boost() {
        // RTX 4080 Super: ~2800 MHz stock boost -> the old default list
        assert_eq!(auto_clocks(2800.0, 3105), vec![2490, 2640, 2790, 2940]);
        // a lower-clocked card scans lower
        assert_eq!(auto_clocks(2520.0, 0), vec![2220, 2370, 2520, 2670]);
        // never above the driver's maximum (and no duplicates)
        assert_eq!(auto_clocks(2900.0, 2950), vec![2595, 2745, 2895, 2940]);
        assert_eq!(auto_clocks(2900.0, 2900), vec![2595, 2745, 2895]);
    }

    #[test]
    fn hints_interpolate() {
        let mk = |c, s| Prof {
            max_clock: c,
            max_stable: s,
            ..Default::default()
        };
        let h = interpolate_hints(&[mk(2640, 255), mk(2790, 180)], &[2700, 2850]);
        assert_eq!(h[&2700], 225);
        assert_eq!(h[&2850], 180);
    }
}

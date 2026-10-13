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

pub fn rel_noise(a: f64, b: f64) -> f64 {
    let m = (a.abs() + b.abs()) / 2.0;
    if m > 0.0 { (a - b).abs() / m } else { 0.0 }
}

/// A higher memory offset only counts if it delivers at least half of the bandwidth gain its
/// measured clock increase should give, and clearly more than the measurement noise. Less
/// than that means transfer retries (error detection and replay) are eating the gain.
pub fn mem_step_worth_it(clk_hi: f64, bw_hi: f64, clk_lo: f64, bw_lo: f64, noise: f64) -> bool {
    if bw_lo <= 0.0 || clk_lo <= 0.0 || clk_hi <= clk_lo {
        return false;
    }
    let ideal = clk_hi / clk_lo - 1.0;
    bw_hi / bw_lo - 1.0 >= (0.5 * ideal).max(2.0 * noise)
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
            "Measuring stock performance (twice)",
            "Your card at default settings, warmed up. Measured twice to know how much results vary by chance, \
             so later decisions ignore that noise.",
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
        let m = |a: f64, b: f64| (a + b) / 2.0;
        let b = Baseline {
            perf: m(heavy.perf, heavy2.perf),
            heavy_power: m(heavy.avg_power, heavy2.avg_power),
            heavy_clk: m(heavy.core_clk, heavy2.core_clk),
            paced_power: m(paced.avg_power, paced2.avg_power),
            paced_clk: m(paced.core_clk, paced2.core_clk),
            pstate: paced2.pstate,
            noise: Noise {
                perf: rel_noise(heavy.perf, heavy2.perf),
                heavy_power: rel_noise(heavy.avg_power, heavy2.avg_power),
                paced_power: rel_noise(paced.avg_power, paced2.avg_power),
                clk: rel_noise(paced.core_clk, paced2.core_clk),
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
                self.ui
                    .event("•", "memory limit reached (errors or crash)", Tone::Warn);
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
        let mut best = results[0];
        for r in &results[1..] {
            if mem_step_worth_it(r.2, r.1, best.2, best.1, noise) {
                best = *r;
            }
        }
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

    fn hot_pick(&mut self, cands: &[i32]) -> R<(i32, Option<f64>)> {
        let noise = self.st.bw_noise.unwrap_or(0.0);
        let mut c = cands.to_vec();
        c.sort_unstable();
        c.dedup();
        let mut hot: BTreeMap<i32, (f64, f64)> = BTreeMap::new();
        for off in c {
            if off > self.mem_limit() {
                break;
            }
            let r = self.mem_probe(&format!("memhot:{off}"), off, true)?;
            let (Some(bw), Some(clk)) = (r["bw"].as_f64(), r["mem_clk"].as_f64()) else {
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
        let Some(&first) = hot.keys().next() else {
            return Ok((0, None));
        };
        let mut chosen = first;
        for (&off, &(bw, clk)) in hot.iter().skip(1) {
            let (cbw, cclk) = hot[&chosen];
            if mem_step_worth_it(clk, bw, cclk, cbw, noise) {
                chosen = off;
            } else {
                let gain = (bw / cbw - 1.0) * 100.0;
                let what = if gain < 0.0 {
                    format!("is slower ({gain:+.1}%)")
                } else {
                    format!("adds only {gain:+.1}% bandwidth")
                };
                self.ui.event(
                    "•",
                    &format!("+{off} {what}: keeping the safer +{chosen}"),
                    Tone::Normal,
                );
                break;
            }
        }
        self.ui
            .event("★", &format!("memory offset +{chosen}"), Tone::Star);
        Ok((chosen, Some(hot[&chosen].0)))
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
                        let ok = t.test(Kind::Paced, s, secs, 0.0, None, false)?.0
                            && t.test(Kind::Heavy, s, secs * 0.5, 0.0, None, false)?.0;
                        Ok(Value::Bool(ok))
                    })?;
                    Ok(v.as_bool().unwrap_or(false))
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
            vram_max: None,
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
    /// tests, or None if nothing passes (keep stock).
    pub fn soak(&mut self, prof: &Prof) -> R<Option<Prof>> {
        if self.st.soak_done {
            return Ok(self.st.soak_result.clone());
        }
        let pl = if prof.power_max {
            self.gpu.info().max_pl
        } else {
            self.dpl()
        };
        let (mut core, mut mem, clk) = (prof.core, prof.mem, prof.max_clock);
        let step = self.a.core_step;
        if let Some((c, m)) = self.st.soak_at {
            // the computer died during the soak last time
            (core, mem) = (c, m);
            if core > 0 {
                core = (core - step).max(0);
            } else if mem > 0 {
                mem = (mem - 250).max(0);
            } else {
                return self.finish_soak(None);
            }
            self.ui.event("!", &format!("the last run died during the soak test: backing off to core {core:+}, mem {mem:+}"), Tone::Warn);
        }
        loop {
            self.st.soak_at = Some((core, mem));
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
                    return self.finish_soak(Some(r));
                }
                Some(f) => {
                    if f == Fail::Mem && mem > 0 {
                        mem = (mem - 250).max(0);
                    } else if core > 0 {
                        core = (core - step).max(0);
                    } else if mem > 0 {
                        mem = (mem - 250).max(0);
                    } else {
                        return self.finish_soak(None);
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

    fn finish_soak(&mut self, r: Option<Prof>) -> R<Option<Prof>> {
        self.st.soak_result = r.clone();
        self.st.soak_done = true;
        self.st.soak_at = None;
        self.save();
        match &r {
            Some(_) => self.ui.event("★", "soak test passed", Tone::Star),
            None => self.ui.event("✗", "soak test failed even at +0", Tone::Bad),
        }
        Ok(r)
    }
}

// ================================================================= time estimate
const TEST_OVERHEAD: f64 = 5.0;

pub fn est_pass1(p: &PassArgs) -> f64 {
    let steps = (p.mem_max / p.mem_step + 2) as f64;
    let n = p.clocks.len() as f64;
    // baseline (2 full, ~3 calibration + 2 game-like), calibration, memory, 6 search steps per
    // clock (game-like + half as long full load each), measurement
    let tests = 9.0 + 2.0 + steps + 3.0 + n * 14.0;
    p.warm
        + 5.0 * p.measure
        + 3.0 * 8.0
        + 31.0
        + steps * p.mem_seconds
        + 3.0 * (p.combo_seconds + p.combo_warmup)
        + n * (6.0 * 1.5 * p.search_seconds + 2.0 * p.measure + p.warmup)
        + tests * TEST_OVERHEAD
}

pub fn est_pass2(p: &PassArgs) -> f64 {
    let n = 5.0;
    let tests = 6.0 + 5.0 + n * 10.0;
    p.warm
        + 5.0 * p.measure
        + 8.0
        + 5.0 * (p.combo_seconds + p.combo_warmup)
        + n * (4.0 * 1.5 * p.search_seconds + 2.0 * p.measure + p.warmup)
        + tests * TEST_OVERHEAD
}

pub fn est_soak(p: &PassArgs) -> f64 {
    p.soak + 45.0 + 120.0 + 3.0 * 24.0 + 10.0 * TEST_OVERHEAD
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mem_rule() {
        assert!(mem_step_worth_it(12126.0, 676.0, 12001.0, 669.0, 0.001));
        assert!(!mem_step_worth_it(12126.0, 670.0, 12001.0, 669.0, 0.001));
        assert!(!mem_step_worth_it(12126.0, 676.0, 12001.0, 669.0, 0.01));
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

//! Runs the tuning phases on the real (or simulated) GPU: every test goes through `test`, every
//! result through the crash-proof `probe`. The phases live in their own files.

mod baseline;
mod core;
mod finals;
mod memory;

use crate::hw::{Gpu, Settings};
use crate::proto::{Kind, Params};
use crate::runner::{Clock, R, Reason, Runner, Stop};
use crate::state::{Pass, PassState, Saver};
use crate::ui::{Log, Row, Tone, Ui, now_hms};
use serde_json::Value;
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

/// Everything a run talks to: the GPU, the test runner, time, the user, the log, the saved state.
#[derive(Clone, Copy)]
pub struct Env<'a> {
    pub gpu: &'a dyn Gpu,
    pub runner: &'a dyn Runner,
    pub clock: &'a dyn Clock,
    pub ui: &'a dyn Ui,
    pub log: &'a Log,
    pub saver: &'a Saver,
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
    pub fn new(e: &Env<'a>, a: PassArgs, st: PassState, which: Pass) -> Self {
        let mut t = Tuner {
            gpu: e.gpu,
            runner: e.runner,
            clock: e.clock,
            ui: e.ui,
            log: e.log,
            saver: e.saver,
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
    fn params(&self, kind: Kind, seconds: f64, warmup: f64) -> Params {
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
            per_frame: self.st.per_frame.unwrap_or(6),
            fps: a.fps,
            mem_mb: a.mem_mb,
            inject: false,
        }
    }

    fn healthy(&self) -> R<bool> {
        if self.gpu.temp().is_err() {
            return Ok(false);
        }
        let p = self.params(Kind::Ping, 0.0, 0.0);
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
                s.cap()
            )
        };
        if s.power_mw != self.dpl() {
            desc += &format!(" · {:.0} W", s.power_mw as f64 / 1000.0);
        }
        let short = if matches!(kind, Kind::Mem | Kind::Combo) {
            format!("mem {:+}", s.mem)
        } else {
            s.short()
        };
        let p = self.params(kind, seconds, warmup);
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
}

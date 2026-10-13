//! Stock baseline (and load calibration), P-state and offset checks.

use super::{TestRow, Tuner};
use crate::decide::{mean, spread};
use crate::proto::Kind;
use crate::runner::{R, Stop};
use crate::state::{Baseline, Calib, Noise};
use crate::ui::{Finding, RowState, Tone};
use serde_json::{Value, json};

impl Tuner<'_> {
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
        let pl_w = self.dpl() as f64 / 1000.0;

        // Full load: make it heavier until it reaches the power limit at stock, like a heavy
        // game. An undervolt that only holds at low power can still crash at high power
        // (more current, more voltage drop).
        let mut alu = self.st.heavy_alu.unwrap_or(self.a.alu_heavy).max(1);
        let mut warm = self.a.warm;
        let (heavy, limited) = loop {
            self.st.heavy_alu = Some(alu);
            let r = self.stock(
                Kind::Heavy,
                self.a.measure,
                warm,
                "The stock test failed, so something is wrong with the setup.",
            )?;
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
        let key = self.params(Kind::Heavy, 0.0, 0.0).golden_key();
        self.st
            .golden
            .insert(key, heavy.hash.clone().unwrap_or_default());
        let unstable = "A repeated stock test failed: the card is unstable at stock.";
        let heavy2 = self.stock(Kind::Heavy, self.a.measure, 5.0, unstable)?;
        let heavy3 = self.stock(Kind::Heavy, self.a.measure, 5.0, unstable)?;

        // Game-like load: about 60% busy at the target frame rate, so there are real idle gaps
        // between frames (load changes every frame, like a game). Adjusts the passes per frame
        // and, when one pass is already too much, the work per pass.
        let paced_failed = "The stock game-like test failed.";
        let mut pf = self.st.per_frame.unwrap_or(1).max(1);
        let mut it = self.st.paced_iters.unwrap_or(self.a.iters_paced).max(1);
        for attempt in 0..8 {
            self.st.per_frame = Some(pf);
            self.st.paced_iters = Some(it);
            let r = self.stock(Kind::Paced, 8.0, 0.0, paced_failed)?;
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
        let paced = self.stock(Kind::Paced, self.a.measure, 0.0, paced_failed)?;
        let key = self.params(Kind::Paced, 0.0, 0.0).golden_key();
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
        let paced2 = self.stock(Kind::Paced, self.a.measure, 0.0, paced_failed)?;
        let paced3 = self.stock(Kind::Paced, self.a.measure, 0.0, paced_failed)?;
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

    /// A test at stock that must pass: if it doesn't, the setup is broken, not the tuning.
    fn stock(&mut self, kind: Kind, secs: f64, warm: f64, failed: &str) -> R<TestRow> {
        let (ok, r) = self.test(kind, self.set(0, 0, None), secs, warm, true)?;
        r.filter(|_| ok)
            .ok_or_else(|| Stop::Exit(format!("{failed} See the log file.")))
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
            let r0 = self.stock(Kind::Mem, 8.0, 0.0, "The stock memory test failed.")?;
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
                let (ok, r) = t.test(Kind::Mem, s500, 8.0, 0.0, false)?;
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
            let (ok, r) = t.test(Kind::Paced, s, 15.0, 0.0, false)?;
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
}

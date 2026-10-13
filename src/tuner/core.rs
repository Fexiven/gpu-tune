//! Core offset: per clock cap, the highest stable offset, then a measurement with the margin.

use super::Tuner;
use crate::decide::{floor_offset, game_rel, search_bounds, step_ok};
use crate::proto::Kind;
use crate::runner::R;
use crate::state::{Baseline, Pass, Prof};
use crate::ui::{Finding, RowState, Tone};
use serde_json::{Value, json};

impl Tuner<'_> {
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
                        let (ok, p) = t.test(Kind::Paced, s, secs, 0.0, false)?;
                        let ok = ok && t.test(Kind::Heavy, s, secs * 0.5, 0.0, false)?.0;
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
        let (ok1, h) = self.test(Kind::Heavy, s, self.a.measure, self.a.warmup, false)?;
        let Some(h) = h.filter(|_| ok1) else {
            return Ok(Value::Bool(false));
        };
        let (ok2, p) = self.test(Kind::Paced, s, self.a.measure, 0.0, false)?;
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
}

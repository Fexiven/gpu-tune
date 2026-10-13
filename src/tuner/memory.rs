//! Memory offset: cool sweep for the bandwidth peak, then re-checks with a hot GPU.

use super::Tuner;
use crate::decide::{MEM_ERROR_MARGIN, MemStop, pick_mem, rel_noise};
use crate::proto::Kind;
use crate::runner::R;
use crate::state::Pass;
use crate::ui::{Finding, RowState, Tone};
use serde_json::{Value, json};
use std::collections::BTreeMap;

impl Tuner<'_> {
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
}

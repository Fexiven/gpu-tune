//! Per profile: maximum or lower power limit, then the soak test (incl. desktop idle).

use super::{Fail, Tuner};
use crate::decide::{KNEE_ELASTICITY, PowerStop, pick_knee, power_stop, prof_power};
use crate::hw::Settings;
use crate::proto::Kind;
use crate::runner::R;
use crate::state::Prof;
use crate::ui::{Row, Tone, now_hms};
use serde_json::{Value, json};

impl Tuner<'_> {
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
                let (ok, r) = t.test(Kind::Heavy, s, measure, warmup, false)?;
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
                let (ok, r) = t.test(Kind::Heavy, s, secs, warm, false)?;
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
                &s.short(),
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
            let (ok, _) = self.test(Kind::Paced, s, 4.0, 0.0, false)?;
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
                let (ok, r) = self.test(kind, s, secs, warm, false)?;
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

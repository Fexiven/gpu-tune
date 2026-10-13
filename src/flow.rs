//! The two passes, the choice of result and the soak test.

use crate::hw::Gpu;
use crate::runner::{Clock, R, Runner};
use crate::state::{Pass, Prof, ResultRec, Saver};
use crate::tuner::{PassArgs, Tuner, game_rel, interpolate_hints, select_profiles};
use crate::ui::{Finding, Log, RowState, Tone, Ui};
use std::collections::BTreeMap;

pub struct Env<'a> {
    pub gpu: &'a dyn Gpu,
    pub runner: &'a dyn Runner,
    pub clock: &'a dyn Clock,
    pub ui: &'a dyn Ui,
    pub log: &'a Log,
    pub saver: &'a Saver,
}

/// Runs everything and fills `state.result` (None = keep stock).
pub fn tune(e: &Env, base_args: &PassArgs, goal: &str, quick_only: bool) -> R<()> {
    let p1 = PassArgs::for_pass(base_args, true);
    let mut p2 = PassArgs::for_pass(base_args, false);
    let take = |p: Pass| {
        let mut st = e.saver.state.lock().unwrap();
        std::mem::take(st.pass_mut(p))
    };

    // ---------------- pass 1: wide scan with short tests
    e.ui.set_stage(if quick_only {
        "Quick scan with short tests"
    } else {
        "Pass 1 of 2 · scanning the whole range with short tests"
    });
    let mut t1 = Tuner::new(
        e.gpu,
        e.runner,
        e.clock,
        e.ui,
        e.log,
        e.saver,
        p1,
        take(Pass::P1),
        Pass::P1,
    );
    let r1 = (|| -> R<_> {
        let b1 = t1.baseline()?;
        t1.calibrate()?;
        let m1 = t1.memory()?;
        t1.core(m1)?;
        Ok((b1, m1))
    })();
    let p1_state = t1.st.clone();
    e.saver.save_pass(Pass::P1, &p1_state);
    let (b1, m1) = r1?;

    let found1: Vec<Prof> = p1_state.clocks.values().flatten().cloned().collect();
    if found1.is_empty() {
        let mut st = e.saver.state.lock().unwrap();
        st.result = None;
        st.done = true;
        drop(st);
        e.saver.save();
        return Ok(());
    }
    let prof1 = select_profiles(&found1, &b1);
    let center = prof1.get(goal).clone();
    e.ui.event(
        "★",
        &format!(
            "pass 1 points to max {} MHz, core +{}, mem +{}",
            center.max_clock, center.core, center.mem
        ),
        Tone::Star,
    );

    let (base, profs, rec) = if quick_only {
        let rec = prof1.get(goal).clone();
        let rec = if goal == "max" {
            t1.maxpower(rec)?
        } else {
            rec
        };
        e.ui.set_stage("Final check · soak test of the result");
        let res = t1.soak(&rec);
        e.saver.save_pass(Pass::P1, &t1.st);
        (b1, prof1, res?)
    } else {
        // ---------------- pass 2: zoom in with full-length tests
        let c = center.max_clock;
        p2.clocks = [
            c as i64 - 60,
            c as i64 - 30,
            c as i64,
            c as i64 + 30,
            c as i64 + 60,
        ]
        .into_iter()
        .filter(|x| (1500..=3300).contains(x))
        .map(|x| x as u32)
        .collect();
        p2.hints = interpolate_hints(&found1, &p2.clocks);
        p2.refine_mem = Some(m1);
        p2.refine_mem_top = p1_state.mem_top_of_range;
        let mut st2 = take(Pass::P2);
        if st2.center.is_some_and(|x| x != c) {
            st2.clocks.clear();
            st2.tried.clear();
            st2.soak_done = false;
            st2.soak_result = None;
            st2.soak_at = None;
            e.ui.event(
                "↻",
                &format!("area changed to max {c} MHz, re-running the fine-tune clocks"),
                Tone::Warn,
            );
        }
        st2.center = Some(c);
        if st2.golden.is_empty() {
            st2.golden = p1_state.golden.clone();
        }
        st2.per_frame = st2.per_frame.or(p1_state.per_frame);
        st2.calib = st2.calib.or_else(|| p1_state.calib.clone());
        st2.core_check_done |= p1_state.core_check_done;
        st2.bw_noise = st2.bw_noise.or(p1_state.bw_noise);
        e.ui.dim("p1:");
        e.ui.set_stage(&format!(
            "Pass 2 of 2 · fine-tuning around max {c} MHz with full-length tests"
        ));
        let mut t2 = Tuner::new(
            e.gpu,
            e.runner,
            e.clock,
            e.ui,
            e.log,
            e.saver,
            p2.clone(),
            st2,
            Pass::P2,
        );
        let r = (|| -> R<_> {
            let b2 = t2.baseline()?;
            t2.calibrate()?;
            let m2 = t2.memory()?;
            t2.core(m2)?;
            let found2: Vec<Prof> = t2
                .st
                .clocks
                .iter()
                .filter(|(k, _)| p2.clocks.contains(k))
                .filter_map(|(_, v)| v.clone())
                .collect();
            let (base, profs) = if found2.is_empty() {
                (b1.clone(), prof1)
            } else {
                (b2.clone(), select_profiles(&found2, &b2))
            };
            let rec = profs.get(goal).clone();
            let rec = if goal == "max" {
                t2.maxpower(rec)?
            } else {
                rec
            };
            e.ui.set_stage("Final check · soak test of the result");
            let res = t2.soak(&rec)?;
            Ok((base, profs, res))
        })();
        e.saver.save_pass(Pass::P2, &t2.st);
        r?
    };

    let mut st = e.saver.state.lock().unwrap();
    st.result = rec.as_ref().map(|r| ResultRec {
        recommended: r.clone(),
        baseline: base.clone(),
        profiles: BTreeMap::from([
            ("efficient".to_string(), profs.efficient.clone()),
            ("balanced".to_string(), profs.balanced.clone()),
            ("max".to_string(), profs.max.clone()),
        ]),
    });
    st.done = true;
    drop(st);
    e.saver.save();
    if let Some(r) = &rec {
        e.ui.finding(
            "result",
            Finding {
                setting: Some(format!("★ {}", r.max_clock)),
                offset: Some(format!("+{}", r.core)),
                limit: Some(format!("mem +{}", r.mem)),
                perf: Some(format!("{:+.1}%", (game_rel(r, &base) - 1.0) * 100.0)),
                heavy: Some(format!("{:+.0} W", r.heavy_power - base.heavy_power)),
                paced: Some(format!("{:+.0} W", r.paced_power - base.paced_power)),
                status: Some("★".into()),
                state: Some(RowState::Best),
            },
        );
    }
    e.ui.event("★", "all done", Tone::Star);
    Ok(())
}

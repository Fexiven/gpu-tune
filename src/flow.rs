//! The two passes, the three profiles and their soak tests.

use crate::hw::Gpu;
use crate::runner::{Clock, R, Runner};
use crate::state::{Baseline, Pass, PassState, Prof, ResultRec, Saver, profile_label};
use crate::tuner::{PassArgs, Profiles, Tuner, game_rel, interpolate_hints, select_profiles};
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

pub const PROFILE_NAMES: [&str; 3] = ["efficient", "balanced", "max"];

/// Soak result per profile name (None = nothing passed).
type Soaked = BTreeMap<String, Option<Prof>>;
/// (max clock, core offset, memory offset, power limit): profiles with equal keys are tested once
type SettingsKey = (u32, i32, i32, u32);

/// Which profiles to find: all three, or only the goal (`--single`).
pub fn wanted(goal: &str, single: bool) -> Vec<&str> {
    if single {
        vec![goal]
    } else {
        PROFILE_NAMES.to_vec()
    }
}

/// Clock caps pass 2 fine-tunes: around the goal's clock (single profile), or around each
/// profile's clock from pass 1.
pub fn plan_pass2(prof1: &Profiles, names: &[&str]) -> Vec<u32> {
    let around: &[i64] = if names.len() == 1 {
        &[-60, -30, 0, 30, 60]
    } else {
        &[-30, 0, 30]
    };
    let mut v: Vec<u32> = names
        .iter()
        .flat_map(|n| {
            let c = prof1.get(n).max_clock as i64;
            around.iter().map(move |d| c + d)
        })
        .filter(|x| (1500..=3300).contains(x))
        .map(|x| x as u32)
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Runs everything and fills `state.result` (None = keep stock).
pub fn tune(e: &Env, base_args: &PassArgs, goal: &str, quick_only: bool, single: bool) -> R<()> {
    let p1 = PassArgs::for_pass(base_args, true);
    let mut p2 = PassArgs::for_pass(base_args, false);
    let take = |p: Pass| {
        let mut st = e.saver.state.lock().unwrap();
        std::mem::take(st.pass_mut(p))
    };
    let names = wanted(goal, single);

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
    e.saver.save_pass(Pass::P1, &t1.st);
    let (b1, m1) = r1?;

    let found1: Vec<Prof> = t1.st.clocks.values().flatten().cloned().collect();
    if found1.is_empty() {
        finish(e, None);
        return Ok(());
    }
    let prof1 = select_profiles(&found1, &b1);
    for n in &names {
        let p = prof1.get(n);
        e.ui.event(
            "★",
            &format!(
                "pass 1: {} near max {} MHz, core +{}, mem +{}",
                profile_label(n),
                p.max_clock,
                p.core,
                p.mem
            ),
            Tone::Star,
        );
    }

    if quick_only {
        let r = soak_all(e, &mut t1, &prof1, &names);
        e.saver.save_pass(Pass::P1, &t1.st);
        finish(e, Some((b1, r?, goal)));
        return Ok(());
    }

    // ---------------- pass 2: zoom in with full-length tests
    p2.clocks = plan_pass2(&prof1, &names);
    p2.hints = interpolate_hints(&found1, &p2.clocks);
    p2.refine_mem = Some(m1);
    p2.refine_mem_top = t1.st.mem_top_of_range;
    let mut st2 = take(Pass::P2);
    if !st2.plan.is_empty() && st2.plan != p2.clocks {
        st2.clocks.clear();
        st2.tried.clear();
        st2.soaks.clear();
        e.ui.event(
            "↻",
            "the area to fine-tune changed, re-running pass 2's clocks",
            Tone::Warn,
        );
    }
    st2.plan = p2.clocks.clone();
    inherit(&mut st2, &t1.st);
    e.ui.dim("p1:");
    e.ui.set_stage(&format!(
        "Pass 2 of 2 · fine-tuning {} clock caps ({}-{} MHz) with full-length tests",
        p2.clocks.len(),
        p2.clocks.first().unwrap_or(&0),
        p2.clocks.last().unwrap_or(&0)
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
        let soaked = soak_all(e, &mut t2, &profs, &names)?;
        Ok((base, soaked))
    })();
    e.saver.save_pass(Pass::P2, &t2.st);
    let (base, soaked) = r?;
    finish(e, Some((base, soaked, goal)));
    Ok(())
}

/// Pass 2 reuses pass 1's calibration (same test loads, same reference images).
fn inherit(st2: &mut PassState, st1: &PassState) {
    if st2.golden.is_empty() {
        st2.golden = st1.golden.clone();
    }
    st2.per_frame = st2.per_frame.or(st1.per_frame);
    st2.paced_iters = st2.paced_iters.or(st1.paced_iters);
    st2.heavy_alu = st2.heavy_alu.or(st1.heavy_alu);
    st2.calib = st2.calib.clone().or_else(|| st1.calib.clone());
    st2.core_check_done |= st1.core_check_done;
    st2.bw_noise = st2.bw_noise.or(st1.bw_noise);
}

/// Soak-tests each wanted profile. Profiles with identical settings are tested once.
fn soak_all(e: &Env, t: &mut Tuner, profs: &Profiles, names: &[&str]) -> R<Soaked> {
    let mut out: Soaked = BTreeMap::new();
    let mut by_settings: Vec<(SettingsKey, Option<Prof>)> = vec![];
    for (i, n) in names.iter().enumerate() {
        let mut p = profs.get(n).clone();
        p = match *n {
            "max" => t.maxpower(p)?,
            // lower power limit, if it costs at most this much full-load performance
            "efficient" => t.power_search(n, p, 0.03)?,
            _ => t.power_search(n, p, 0.01)?,
        };
        let key = (
            p.max_clock,
            p.core,
            p.mem,
            crate::tuner::prof_power(&p, e.gpu.info()),
        );
        if let Some((_, r)) = by_settings.iter().find(|(k, _)| *k == key) {
            e.ui.event(
                "•",
                &format!(
                    "{}: same settings as a profile already tested",
                    profile_label(n)
                ),
                Tone::Normal,
            );
            out.insert(n.to_string(), r.clone());
            continue;
        }
        e.ui.set_stage(&format!(
            "Final check{} · soak test of the {} profile (max {} MHz, core +{}, mem +{})",
            if names.len() > 1 {
                format!(" {} of {}", i + 1, names.len())
            } else {
                String::new()
            },
            profile_label(n),
            p.max_clock,
            p.core,
            p.mem
        ));
        let r = t.soak(n, &p)?;
        by_settings.push((key, r.clone()));
        out.insert(n.to_string(), r);
    }
    Ok(out)
}

fn finish(e: &Env, res: Option<(Baseline, Soaked, &str)>) {
    let rec = res.and_then(|(base, soaked, goal)| {
        let passed: BTreeMap<String, Prof> = soaked
            .iter()
            .filter_map(|(k, v)| v.clone().map(|p| (k.clone(), p)))
            .collect();
        let failed: Vec<String> = soaked
            .iter()
            .filter(|(_, v)| v.is_none())
            .map(|(k, _)| k.clone())
            .collect();
        // recommended: the goal, or the closest profile that passed
        let order: &[&str] = match goal {
            "efficient" => &["efficient", "balanced", "max"],
            "max" => &["max", "balanced", "efficient"],
            _ => &["balanced", "efficient", "max"],
        };
        let name = order.iter().find(|n| passed.contains_key(**n))?;
        Some(ResultRec {
            recommended: passed[*name].clone(),
            recommended_name: name.to_string(),
            baseline: base,
            profiles: passed,
            failed,
        })
    });
    if let Some(r) = &rec {
        for name in PROFILE_NAMES {
            let Some(p) = r.profiles.get(name) else {
                continue;
            };
            let star = name == r.recommended_name;
            e.ui.finding(
                &format!("result:{name}"),
                Finding {
                    setting: Some(format!(
                        "{}{}",
                        if star { "★ " } else { "  " },
                        profile_label(name)
                    )),
                    offset: Some(format!("+{}", p.core)),
                    limit: Some(format!("{} MHz", p.max_clock)),
                    perf: Some(format!("{:+.1}%", (game_rel(p, &r.baseline) - 1.0) * 100.0)),
                    heavy: Some(format!("{:+.0} W", p.heavy_power - r.baseline.heavy_power)),
                    paced: Some(format!("{:+.0} W", p.paced_power - r.baseline.paced_power)),
                    status: Some("★".into()),
                    state: Some(RowState::Best),
                },
            );
        }
    }
    let mut st = e.saver.state.lock().unwrap();
    st.result = rec;
    st.done = true;
    drop(st);
    e.saver.save();
    e.ui.event("★", "all done", Tone::Star);
}

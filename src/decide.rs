//! The tuner's decision rules, free of hardware so they can be tested directly: offset search,
//! memory and power-limit picks, profile selection, noise, time estimates.

use crate::runner::R;
use crate::state::{Baseline, Prof};
use crate::tuner::PassArgs;
use serde_json::Value;
use std::collections::BTreeMap;

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
pub fn step_ok(v: &Value) -> bool {
    v.as_bool()
        .unwrap_or_else(|| v["ok"].as_bool().unwrap_or(false))
}

/// Binary search for the highest offset where `run` passes. Returns (lo, hi) with lo stable,
/// or None if even +0 fails. A hint (from pass 1) or the previous clock narrows the start.
/// A lower offset at the same clock cap means more voltage, so when an offset passes all lower
/// ones would too: +0 is only tested when nothing above it passed.
pub fn search_bounds(
    mut run: impl FnMut(i32) -> R<bool>,
    hint: Option<i32>,
    prev: Option<i32>,
    core_max: i32,
    step: i32,
) -> R<Option<(i32, i32)>> {
    let (mut lo, mut hi) = (0, core_max + step);
    let mut check = |o: i32, lo: &mut i32, hi: &mut i32| -> R<()> {
        if run(o)? {
            *lo = o;
        } else {
            *hi = o;
        }
        Ok(())
    };
    if let Some(h) = hint {
        // pass 1 found the limit with shorter tests: start right around it
        let below = (h - HINT_WINDOW * step).max(0);
        let above = (h + HINT_WINDOW * step).min(core_max);
        if below > 0 {
            check(below, &mut lo, &mut hi)?;
        }
        if lo < above && above < hi {
            check(above, &mut lo, &mut hi)?;
        }
    } else if let Some(p) = prev.filter(|p| *p > 0 && *p <= core_max) {
        check(p, &mut lo, &mut hi)?;
    }
    while hi - lo > step {
        // (never beyond the range: lo isn't always on hi's grid)
        let mid = (lo + ((hi - lo) / (2 * step)).max(1) * step).min(core_max);
        check(mid, &mut lo, &mut hi)?;
    }
    if lo == 0 && !run(0)? {
        return Ok(None);
    }
    Ok(Some((lo, hi)))
}

/// Pass 2 starts this many steps either side of pass 1's limit (pass 1 and 2 agreed within
/// 2 steps on a real 4080 Super; a wrong hint only costs one more test).
const HINT_WINDOW: i32 = 2;

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

// ================================================================= time estimate
const TEST_OVERHEAD: f64 = 5.0;

pub fn est_pass1(p: &PassArgs) -> f64 {
    let steps = (p.mem_max / p.mem_step + 2) as f64;
    let n = if p.clocks.is_empty() {
        AUTO_CLOCKS.len()
    } else {
        p.clocks.len()
    } as f64;
    // baseline (3 full, ~3 calibration + 3 game-like), calibration, memory, ~5 search steps
    // per clock (game-like + half as long full load each), measurement
    let tests = 11.0 + 2.0 + steps + 3.0 + n * 12.0;
    p.warm
        + 7.0 * p.measure
        + 3.0 * 8.0
        + 31.0
        + steps * p.mem_seconds
        + 3.0 * (p.combo_seconds + p.combo_warmup)
        + n * (5.0 * 1.5 * p.search_seconds + 2.0 * p.measure + p.warmup)
        + tests * TEST_OVERHEAD
}

pub fn est_pass2(p: &PassArgs, n_clocks: usize) -> f64 {
    let n = n_clocks as f64;
    // ~3 search steps per clock: pass 1's limit +-2 steps, then bisection
    let tests = 8.0 + 5.0 + n * 8.0;
    p.warm
        + 7.0 * p.measure
        + 8.0
        + 5.0 * (p.combo_seconds + p.combo_warmup)
        + n * (3.0 * 1.5 * p.search_seconds + 2.0 * p.measure + p.warmup)
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
        // limit 30, hint 180: below=150 fails -> 'above' (210) must not be tried
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
        assert!(!calls.contains(&210), "{calls:?}");
    }

    #[test]
    fn zero_is_only_tested_when_nothing_above_passes() {
        let mut calls = vec![];
        search_bounds(
            |o| {
                calls.push(o);
                Ok(o <= 270)
            },
            Some(270),
            None,
            450,
            15,
        )
        .unwrap();
        assert!(!calls.contains(&0), "{calls:?}");
        // a good hint: below, above, then two bisection steps
        assert_eq!(calls, vec![240, 300, 270, 285]);
        let mut calls = vec![];
        let r = search_bounds(
            |o| {
                calls.push(o);
                Ok(o == 0)
            },
            None,
            None,
            450,
            15,
        )
        .unwrap();
        assert_eq!(r, Some((0, 15)));
        assert_eq!(calls.last(), Some(&0));
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

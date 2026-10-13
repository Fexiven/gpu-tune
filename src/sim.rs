//! End-to-end tests of the tuning flow against a simulated RTX 4080 Super with known limits.
//! Simulated time: a full run takes milliseconds.

use crate::flow::{Env, tune};
use crate::hw::{Gpu, GpuInfo, Settings, Snap};
use crate::proto::{Kind, Params, WorkResult};
use crate::runner::{Clock, Outcome, R, Reason, Runner, Stop, Summary};
use crate::state::{Saver, State};
use crate::tuner::PassArgs;
use crate::ui::{Finding, Log, Tone, Ui};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct Card {
    p0_mem: f64,
    k: f64,
    mem_ok_cool: f64,
    mem_ok_hot: f64,
    retry_knee: f64,
    idle_core_limit: i32,
    offset_broken: bool,
    pstate: u32,
    stock_boost: f64,
    /// game-like load: frame time multiplier (8 = the load is 8x too big at first)
    paced_scale: f64,
    /// full load: most power the shader can draw however heavy it gets
    heavy_max_w: f64,
    /// below this power the voltage is at its floor: every watt saved costs ~1:1 performance
    vmin_w: f64,
}

impl Default for Card {
    fn default() -> Self {
        Card {
            p0_mem: 11501.0,
            k: 0.5,
            mem_ok_cool: 12400.0,
            mem_ok_hot: 12300.0,
            retry_knee: 12200.0,
            idle_core_limit: 200,
            offset_broken: false,
            pstate: 0,
            stock_boost: 2800.0,
            paced_scale: 1.0,
            heavy_max_w: 400.0,
            vmin_w: 0.0,
        }
    }
}

fn core_limit(cap: Option<u32>) -> i32 {
    match cap {
        None => 150,
        Some(c) => 330 - (c as i32 - 2490) / 2,
    }
}

struct FakeGpu {
    info: GpuInfo,
    s: Mutex<Settings>,
    running: Mutex<bool>,
    card: Card,
}

impl FakeGpu {
    fn new(card: Card) -> Self {
        FakeGpu {
            info: GpuInfo {
                name: "NVIDIA GeForce RTX 4080 SUPER (simulated)".into(),
                driver: "580.00".into(),
                bus: "0000:01:00.0".into(),
                dev_id: 0x2702,
                default_pl: 320_000,
                min_pl: 150_000,
                max_pl: 352_000,
                core_range: (-1000, 1000),
                mem_range: (-2000, 6000),
                api: "per P-state (P0)".into(),
                max_mem_clk: 11501,
                max_core_clk: 3105,
            },
            s: Mutex::new(Settings {
                power_mw: 320_000,
                ..Default::default()
            }),
            running: Mutex::new(false),
            card,
        }
    }
}

impl Gpu for FakeGpu {
    fn info(&self) -> &GpuInfo {
        &self.info
    }
    fn apply(&self, s: &Settings) -> anyhow::Result<()> {
        let r = self.info.core_range;
        if s.core < r.0 || s.core > r.1 {
            anyhow::bail!("core offset {} outside range", s.core);
        }
        *self.s.lock().unwrap() = *s;
        Ok(())
    }
    fn reset(&self) {
        *self.s.lock().unwrap() = Settings {
            power_mw: self.info.default_pl,
            ..Default::default()
        };
    }
    fn snapshot(&self) -> anyhow::Result<Snap> {
        let s = self.s.lock().unwrap();
        Ok(Snap {
            power: 25.0,
            util: 0,
            clk: (210 + s.core.max(0)) as u32,
            mclk: 810,
            temp: 40,
            pstate: Some(8),
            vram: None,
            reasons: 0,
        })
    }
    fn power_limit(&self) -> u32 {
        self.s.lock().unwrap().power_mw
    }
    fn temp(&self) -> anyhow::Result<u32> {
        Ok(40)
    }
    fn vram(&self) -> Option<f64> {
        None
    }
    fn new_xids(&self) -> Vec<String> {
        let core = self.s.lock().unwrap().core;
        if !*self.running.lock().unwrap() && core > self.card.idle_core_limit {
            vec![format!(
                "NVRM: Xid (PCI:0000:01:00): 13, idle crash at core {core}"
            )]
        } else {
            vec![]
        }
    }
    fn free_vram_mb(&self) -> u64 {
        15000
    }
}

type Hook = Box<dyn Fn(&Params, &Settings, u64) -> Option<R<Outcome>> + Send + Sync>;

struct FakeRunner {
    gpu: Arc<FakeGpu>,
    clock: Arc<FakeClock>,
    calls: AtomicU64,
    hook: Option<Hook>,
    seed: Mutex<u64>,
}

impl FakeRunner {
    fn noise(&self, x: f64, rel: f64) -> f64 {
        let mut s = self.seed.lock().unwrap();
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        let u = (*s >> 11) as f64 / (1u64 << 53) as f64;
        x * (1.0 + rel * (2.0 * u - 1.0))
    }
}

impl Runner for FakeRunner {
    fn run(&self, p: &Params, _: u32, _: f64) -> R<Outcome> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        *self.clock.0.lock().unwrap() += p.seconds + p.warmup + 4.0;
        let s = *self.gpu.s.lock().unwrap();
        if let Some(h) = &self.hook
            && let Some(r) = h(p, &s, n)
        {
            return r;
        }
        let c = &self.gpu.card;
        let mut summ = Summary {
            pstate: Some(c.pstate),
            temp: 70,
            ..Default::default()
        };
        let ok = |res: WorkResult, summ: Summary| {
            Ok(Outcome {
                res: Some(res),
                summ,
                reason: None,
                why: String::new(),
            })
        };
        let crash = |summ: Summary| {
            Ok(Outcome {
                res: None,
                summ,
                reason: Some(Reason::Crash),
                why: "NVRM: Xid 13".into(),
            })
        };
        if p.kind == Kind::Ping {
            return ok(WorkResult::default(), summ);
        }
        *self.gpu.running.lock().unwrap() = true;
        let core = if c.offset_broken { 0 } else { s.core };
        let mclk = c.p0_mem + s.mem as f64 * c.k;
        summ.mem_clk = mclk;
        let hot = p.kind == Kind::Combo;
        let mem_limit = if hot { c.mem_ok_hot } else { c.mem_ok_cool };
        let out = (|| {
            if matches!(p.kind, Kind::Mem | Kind::Combo) && mclk > mem_limit + 150.0 {
                return crash(summ.clone());
            }
            let mem_errors = (matches!(p.kind, Kind::Mem | Kind::Combo) && mclk > mem_limit) as u64;
            if matches!(p.kind, Kind::Heavy | Kind::Paced | Kind::Combo)
                && core > core_limit(s.clock)
            {
                return crash(summ.clone());
            }
            let clk = (s.clock.map_or(f64::MAX, |x| x as f64)).min(c.stock_boost + core as f64);
            let volt = (clk - core as f64) / c.stock_boost;
            let mut r = WorkResult {
                mem_errors,
                ..Default::default()
            };
            match p.kind {
                Kind::Paced => {
                    summ.core_clk = self.noise(clk, 0.001);
                    summ.power = self.noise(220.0 * volt * volt * clk / 2800.0, 0.002);
                    // busy grows with the work per frame (passes x steps), 60% at 1 x 16 steps
                    r.busy = Some(
                        0.6 * (p.per_frame * p.iters) as f64 / 16.0 * c.paced_scale * 2800.0 / clk,
                    );
                    r.missed = Some(if clk > 2300.0 { 0 } else { 5 });
                    r.frames = Some(1000);
                    r.hash = Some("H".into());
                }
                Kind::Heavy => {
                    let h = clk.min(2650.0 + core as f64 * 0.6);
                    summ.core_clk = h;
                    // more math per step = more power (needs alu 256 to reach 320 W at stock);
                    // an undervolt (lower voltage at the same clock) draws less: power ~ V^2
                    let want = (100.0 + p.alu as f64 * 0.85).min(c.heavy_max_w);
                    let natural = want * volt.min(1.0).powi(2);
                    // the power limit holds the card below its natural draw, costing some speed
                    let cap = s.power_mw as f64 / 1000.0 - 2.0;
                    let (power, mut slow) = if natural >= cap {
                        (cap, (cap / natural).powf(0.25))
                    } else {
                        (natural, 1.0)
                    };
                    // below the voltage floor only the clock can drop: ~1:1 performance per watt
                    let floor = c.vmin_w - 2.0;
                    if power < floor {
                        slow *= power / floor;
                    }
                    summ.power = self.noise(power, 0.002);
                    if natural >= cap {
                        summ.limit = "power".into();
                    }
                    r.perf = Some(self.noise(52.0 * h / 2650.0 * slow, 0.002));
                    r.hash = Some("H".into());
                }
                _ => {
                    let eff = if mclk <= c.retry_knee {
                        mclk
                    } else {
                        c.retry_knee - (mclk - c.retry_knee) * 2.0
                    };
                    r.bw = Some(self.noise(640.0 * eff / c.p0_mem, 0.002));
                    summ.power = 300.0;
                    if hot {
                        r.hash = Some("H".into());
                    }
                }
            }
            ok(r, summ.clone())
        })();
        *self.gpu.running.lock().unwrap() = false;
        out
    }
}

struct FakeClock(Mutex<f64>);
impl Clock for FakeClock {
    fn now(&self) -> f64 {
        *self.0.lock().unwrap()
    }
    fn sleep(&self, s: f64) -> R<()> {
        *self.0.lock().unwrap() += s.max(0.0);
        Ok(())
    }
}

#[derive(Default)]
struct QuietUi(Mutex<Vec<String>>);
impl Ui for QuietUi {
    fn set_stage(&self, t: &str) {
        self.0.lock().unwrap().push(format!("» {t}"));
    }
    fn set_phase(&self, _: &str, _: &str) {}
    fn test_start(&self, _: &str, _: &str, _: f64, _: &str) {}
    fn test_end(&self, _: bool, _: &str, _: &str) {}
    fn event(&self, i: &str, t: &str, _: Tone) {
        self.0.lock().unwrap().push(format!("{i} {t}"));
    }
    fn finding(&self, _: &str, _: Finding) {}
    fn dim(&self, _: &str) {}
}

fn args(gpu: &FakeGpu) -> PassArgs {
    PassArgs {
        warm: 30.0,
        measure: 25.0,
        warmup: 10.0,
        search_seconds: 15.0,
        mem_seconds: 10.0,
        combo_seconds: 25.0,
        combo_warmup: 10.0,
        mem_step: 500,
        soak: 120.0,
        clocks: vec![], // auto, from the stock boost clock
        hints: Default::default(),
        refine_mem: None,
        refine_mem_top: false,
        core_max: 450,
        core_step: 15,
        core_margin: 45,
        mem_max: 4000,
        temp_limit: 87,
        vram_limit: 102.0,
        vram_soft: 95.0,
        fps: 120,
        width: 2560,
        height: 1440,
        iters_heavy: 8,
        iters_paced: 16,
        alu_heavy: 64,
        alu_paced: 2,
        mem_mb: 5000,
        bus: gpu.info.bus.clone(),
        skip_offset_check: false,
        ignore_pstate: false,
    }
}

struct Run {
    res: R<()>,
    state: State,
    events: Vec<String>,
    sim_minutes: f64,
}

fn run_with(card: Card, state: State, goal: &str, quick: bool, hook: Option<Hook>) -> Run {
    let gpu = Arc::new(FakeGpu::new(card));
    let clock = Arc::new(FakeClock(Mutex::new(0.0)));
    let runner = FakeRunner {
        gpu: gpu.clone(),
        clock: clock.clone(),
        calls: AtomicU64::new(0),
        hook,
        seed: Mutex::new(0x9E3779B97F4A7C15),
    };
    let ui = QuietUi::default();
    let log = Log::new(None);
    let saver = Saver {
        path: None,
        state: Mutex::new(state),
    };
    let a = args(&gpu);
    let env = Env {
        gpu: gpu.as_ref(),
        runner: &runner,
        clock: clock.as_ref(),
        ui: &ui,
        log: &log,
        saver: &saver,
    };
    let (goal, single) = match goal.strip_prefix("single:") {
        Some(g) => (g, true),
        None => (goal, false),
    };
    let res = tune(&env, &a, goal, quick, single);
    let state = saver.state.lock().unwrap().clone();
    let events = ui.0.lock().unwrap().clone();
    Run {
        res,
        state,
        events,
        sim_minutes: clock.now() / 60.0,
    }
}

fn run(card: Card) -> Run {
    run_with(card, State::default(), "balanced", false, None)
}

/// Like run_with, but the hook gets the saver too, so it can capture the saved state exactly
/// as it would be on disk at that moment (pending step already recorded).
fn run_shared(card: Card, state: State, make_hook: impl FnOnce(Arc<Saver>) -> Hook) -> Run {
    let gpu = Arc::new(FakeGpu::new(card));
    let saver = Arc::new(Saver {
        path: None,
        state: Mutex::new(state),
    });
    let clock = Arc::new(FakeClock(Mutex::new(0.0)));
    let runner = FakeRunner {
        gpu: gpu.clone(),
        clock: clock.clone(),
        calls: AtomicU64::new(0),
        hook: Some(make_hook(saver.clone())),
        seed: Mutex::new(7),
    };
    let ui = QuietUi::default();
    let log = Log::new(None);
    let a = args(&gpu);
    let env = Env {
        gpu: gpu.as_ref(),
        runner: &runner,
        clock: clock.as_ref(),
        ui: &ui,
        log: &log,
        saver: &saver,
    };
    let res = tune(&env, &a, "balanced", false, false);
    let state = saver.state.lock().unwrap().clone();
    let events = ui.0.lock().unwrap().clone();
    Run {
        res,
        state,
        events,
        sim_minutes: clock.now() / 60.0,
    }
}

/// The computer dies the first time `pred` matches a test: returns the state on disk at that moment.
fn power_loss(pred: impl Fn(&Params, &Settings, &State) -> bool + Send + Sync + 'static) -> State {
    let disk: Arc<Mutex<Option<State>>> = Default::default();
    let d2 = disk.clone();
    let r = run_shared(Card::default(), State::default(), move |saver| {
        Box::new(move |p, s, _| {
            let disk = saver.state.lock().unwrap().clone();
            if d2.lock().unwrap().is_none() && pred(p, s, &disk) {
                *d2.lock().unwrap() = Some(disk);
                return Some(Err(Stop::Exit("power loss".into())));
            }
            None
        })
    });
    assert!(matches!(r.res, Err(Stop::Exit(_))));
    disk.lock()
        .unwrap()
        .take()
        .expect("the power loss happened")
}

#[test]
fn finds_the_built_in_limits() {
    let r = run(Card::default());
    assert_eq!(r.res, Ok(()), "{:?}", r.events);
    let rec = &r.state.result.as_ref().expect("a result").recommended;
    assert_eq!(
        rec.core,
        core_limit(Some(rec.max_clock)) / 15 * 15 - 45,
        "{rec:?}"
    );
    // the full load was made heavier until it reached the power limit
    assert_eq!(r.state.p1.heavy_alu, Some(256));
    assert!(r.state.p1.baseline.as_ref().unwrap().heavy_limited);
    assert_eq!(
        (r.state.p1.per_frame, r.state.p1.paced_iters),
        (Some(1), Some(16))
    );
    assert_eq!(
        rec.mem, 1250,
        "last step before the simulated retries: {rec:?}"
    );
    // all three profiles soak-tested; each within its clock's limit minus the margin (or lower,
    // if the soak test made it back off), and ordered by power
    let res = r.state.result.as_ref().unwrap();
    assert_eq!(res.recommended_name, "balanced");
    assert!(res.failed.is_empty(), "{:?}", res.failed);
    let pr = |n: &str| {
        res.profiles
            .get(n)
            .unwrap_or_else(|| panic!("{n} missing: {:?}", r.events))
    };
    for n in ["efficient", "balanced", "max"] {
        let p = pr(n);
        assert!(
            p.core <= core_limit(Some(p.max_clock)) / 15 * 15 - 45,
            "{n}: {p:?}"
        );
        assert!(p.core <= 200, "{n} must pass the idle check: {p:?}");
    }
    assert!(pr("efficient").paced_power <= pr("balanced").paced_power + 1.0);
    // power search: no voltage floor on this card, so efficient goes down in 10 W steps until
    // the 3% loss cap; balanced (1%) keeps the default
    assert_eq!(
        pr("efficient").power_limit_mw,
        Some(250_000),
        "{:?}",
        pr("efficient")
    );
    assert_eq!(pr("balanced").power_limit_mw, None, "{:?}", pr("balanced"));
    assert!(pr("max").paced_clk >= pr("balanced").paced_clk - 1.0);
    let cal = r.state.p1.calib.as_ref().unwrap();
    assert_eq!(cal.pstate, Some(0));
    assert!((cal.k - 0.5).abs() < 1e-9);
    assert!(r.state.p1.baseline.as_ref().unwrap().noise.perf > 0.0);
    assert!(
        r.state.p2.pending.is_none() && r.state.p2.soaks.values().all(|s| s.at.is_none() && s.done)
    );
    assert!(
        r.sim_minutes > 30.0 && r.sim_minutes < 120.0,
        "{} min",
        r.sim_minutes
    );
}

#[test]
fn power_search_stops_at_the_knee() {
    // voltage floor at 255 W: above it a lower limit mostly lowers the voltage (cheap), below it
    // only the clock can drop (every watt costs performance). Without it: 250 W (see above).
    let r = run(Card {
        vmin_w: 255.0,
        ..Default::default()
    });
    assert_eq!(r.res, Ok(()), "{:?}", r.events);
    let eff = &r.state.result.as_ref().unwrap().profiles["efficient"];
    assert_eq!(eff.power_limit_mw, Some(260_000), "{eff:?}");
    // it looked one step past the knee, then stopped
    assert!(
        r.events
            .iter()
            .any(|e| e.contains("Efficient: 250 W limit"))
    );
    assert!(
        !r.events
            .iter()
            .any(|e| e.contains("Efficient: 240 W limit"))
    );
}

#[test]
fn clock_scan_follows_the_cards_stock_boost() {
    let keys = |r: &Run| r.state.p1.clocks.keys().copied().collect::<Vec<_>>();
    let r = run(Card::default());
    assert_eq!(keys(&r), vec![2490, 2640, 2790, 2940]);
    let r = run(Card {
        stock_boost: 2560.0,
        ..Default::default()
    });
    assert_eq!(r.res, Ok(()), "{:?}", r.events);
    assert_eq!(keys(&r), vec![2250, 2400, 2550, 2700]);
}

#[test]
fn idle_crash_backs_off_below_the_idle_limit() {
    let r = run(Card {
        idle_core_limit: 100,
        ..Default::default()
    });
    for (n, p) in &r.state.result.as_ref().unwrap().profiles {
        assert!(p.core <= 100, "{n}: {p:?}");
    }
    assert!(r.events.iter().any(|e| e.contains("soak failed")));
}

#[test]
fn offset_without_effect_stops() {
    let r = run(Card {
        offset_broken: true,
        ..Default::default()
    });
    match r.res {
        Err(Stop::Exit(m)) => assert!(m.contains("no clear effect"), "{m}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn tests_not_in_p0_stop() {
    let r = run(Card {
        pstate: 2,
        ..Default::default()
    });
    match r.res {
        Err(Stop::Exit(m)) => assert!(m.contains("instead of P0"), "{m}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn power_loss_in_the_search_is_counted_as_unstable_after_reboot() {
    let disk = power_loss(|p, s, _| {
        p.kind == Kind::Paced && p.seconds == 15.0 && s.clock == Some(2790) && s.core >= 120
    });
    let died = disk
        .p1
        .pending
        .clone()
        .expect("the running step is on disk");
    assert!(died.starts_with("core:2790:"), "{died}");
    let r = run_with(Card::default(), disk, "balanced", false, None);
    assert_eq!(r.res, Ok(()), "{:?}", r.events);
    assert_eq!(
        r.state.p1.tried.get(&died),
        Some(&serde_json::Value::Bool(false)),
        "{died}"
    );
    assert!(r.events.iter().any(|e| e.contains("died during")));
    assert!(r.state.result.is_some());
}

#[test]
fn power_loss_in_the_soak_backs_off_instead_of_looping() {
    // dies in the first profile's cold-start test
    let disk = power_loss(|p, _, st| {
        p.kind == Kind::Paced && st.p2.soaks.values().any(|s| s.at.is_some())
    });
    let (name, at) = disk
        .p2
        .soaks
        .iter()
        .find_map(|(n, s)| s.at.map(|a| (n.clone(), a)))
        .expect("soak position on disk");
    let r = run_with(Card::default(), disk, "balanced", false, None);
    let p = &r.state.result.as_ref().unwrap().profiles[&name];
    assert_eq!(p.core, at.0 - 15, "{name}: {at:?} -> {p:?}");
    assert!(r.events.iter().any(|e| e.contains("died during the soak")));
}

#[test]
fn interrupt_clears_pending_and_resumes() {
    let hook: Hook = Box::new(|_, _, n| (n == 40).then_some(Err(Stop::Interrupted)));
    let r1 = run_with(
        Card::default(),
        State::default(),
        "balanced",
        false,
        Some(hook),
    );
    assert_eq!(r1.res, Err(Stop::Interrupted));
    assert!(r1.state.p1.pending.is_none() && r1.state.p2.pending.is_none());
    let r2 = run_with(Card::default(), r1.state, "balanced", false, None);
    assert_eq!(r2.res, Ok(()));
    assert!(!r2.events.iter().any(|e| e.contains("died during")));
}

#[test]
fn one_overheat_is_not_counted_as_instability() {
    let n = Arc::new(AtomicU64::new(0));
    let n2 = n.clone();
    let hook: Hook = Box::new(move |p, s, _| {
        if p.kind == Kind::Paced
            && s.clock == Some(2640)
            && s.core == 165
            && n2.fetch_add(1, Ordering::SeqCst) == 0
        {
            return Some(Ok(Outcome {
                res: None,
                summ: Summary::default(),
                reason: Some(Reason::Overheat),
                why: String::new(),
            }));
        }
        None
    });
    let r = run_with(
        Card::default(),
        State::default(),
        "balanced",
        false,
        Some(hook),
    );
    assert!(n.load(Ordering::SeqCst) >= 1);
    assert_eq!(
        r.state.p1.clocks[&2640].as_ref().unwrap().max_stable,
        core_limit(Some(2640)) / 15 * 15
    );
}

#[test]
fn persistent_overheat_stops_the_run() {
    let hook: Hook = Box::new(|p, s, _| {
        (p.kind == Kind::Heavy && s.core > 0).then(|| {
            Ok(Outcome {
                res: None,
                summ: Summary::default(),
                reason: Some(Reason::Overheat),
                why: String::new(),
            })
        })
    });
    let r = run_with(
        Card::default(),
        State::default(),
        "balanced",
        false,
        Some(hook),
    );
    assert_eq!(r.res, Err(Stop::Thermal));
}

#[test]
fn quick_only_and_other_goals_finish() {
    let r = run_with(Card::default(), State::default(), "balanced", true, None);
    assert!(r.state.result.is_some());
    for goal in ["efficient", "max"] {
        let r = run_with(Card::default(), State::default(), goal, false, None);
        assert_eq!(r.res, Ok(()));
        assert!(r.state.result.is_some(), "{goal}");
    }
}

#[test]
fn nothing_stable_keeps_stock() {
    let r = run(Card {
        idle_core_limit: -1,
        mem_ok_cool: 0.0,
        mem_ok_hot: 0.0,
        ..Default::default()
    });
    match r.res {
        Ok(()) => assert!(r.state.result.is_none()),
        Err(Stop::Exit(_)) => {}
        other => panic!("{other:?}"),
    }
}

#[test]
fn state_round_trips_through_json() {
    let r = run(Card::default());
    let txt = serde_json::to_string_pretty(&r.state).unwrap();
    let back: State = serde_json::from_str(&txt).unwrap();
    assert_eq!(serde_json::to_string_pretty(&back).unwrap(), txt);
    let rep = crate::report::report_text(&FakeGpu::new(Card::default()), &back, "balanced");
    assert!(
        rep.contains("GPU clock offset")
            && rep.contains("nvidia-smi should show the memory at about 12126 MHz"),
        "{rep}"
    );
}

#[test]
fn saved_progress_is_reused_or_discarded_correctly() {
    let dir = std::env::temp_dir().join(format!("gpu-tune-test-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("state.json");
    let r = run(Card::default());
    let mut st = r.state.clone();
    st.started = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    st.done = false;
    std::fs::write(&path, serde_json::to_string(&st).unwrap()).unwrap();
    let (s, note) = crate::load_state(&path, &st.sig, "balanced", false, false);
    assert!(note.contains("continuing") && s.p1.baseline.is_some());
    let mut other = st.sig.clone();
    other.driver = "999".into();
    let (s, note) = crate::load_state(&path, &other, "balanced", false, false);
    assert!(note.contains("changed") && s.p1.baseline.is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn oversized_game_like_load_is_scaled_down() {
    // like the real 4080 Super run: one pass already took 2x the frame time
    let r = run(Card {
        paced_scale: 8.0,
        ..Default::default()
    });
    assert_eq!(r.res, Ok(()), "{:?}", r.events);
    let st = &r.state.p1;
    let busy = 0.6 * (st.per_frame.unwrap() * st.paced_iters.unwrap()) as f64 / 16.0 * 8.0;
    assert!(
        (0.45..=0.75).contains(&busy),
        "{:?} {:?} -> busy {busy}",
        st.per_frame,
        st.paced_iters
    );
}

#[test]
fn full_load_that_cant_reach_the_power_limit_is_reported() {
    let r = run(Card {
        heavy_max_w: 200.0,
        ..Default::default()
    });
    assert_eq!(r.res, Ok(()));
    let b = r.state.p1.baseline.as_ref().unwrap();
    assert!(!b.heavy_limited);
    assert_eq!(r.state.p1.heavy_alu, Some(512));
    let rep = crate::report::report_text(&FakeGpu::new(Card::default()), &r.state, "balanced");
    assert!(rep.contains("untested at full gaming power"), "{rep}");
}

#[test]
fn time_estimate_matches_a_simulated_run() {
    let gpu = FakeGpu::new(Card::default());
    let a = args(&gpu);
    let (p1, p2) = (PassArgs::for_pass(&a, true), PassArgs::for_pass(&a, false));
    let r = run(Card::default());
    let n = r.state.p2.plan.len();
    let est = (crate::tuner::est_pass1(&p1)
        + crate::tuner::est_pass2(&p2, n)
        + 3.0 * crate::tuner::est_soak(&p2)
        + 2.0 * crate::tuner::est_power(&p2))
        / 60.0;
    eprintln!("pass 2 fine-tuned {n} clock caps");
    eprintln!("estimate {est:.0} min, simulated {:.0} min", r.sim_minutes);
    assert!(
        est > r.sim_minutes * 0.7 && est < r.sim_minutes * 1.4,
        "estimate {est:.0} vs simulated {:.0}",
        r.sim_minutes
    );
}

#[test]
fn single_profile_mode_soaks_only_the_goal() {
    let r = run_with(
        Card::default(),
        State::default(),
        "single:efficient",
        false,
        None,
    );
    assert_eq!(r.res, Ok(()));
    let res = r.state.result.as_ref().unwrap();
    assert_eq!(res.profiles.keys().collect::<Vec<_>>(), vec!["efficient"]);
    assert_eq!(r.state.p2.soaks.len(), 1);
    assert_eq!(r.state.p2.plan.len(), 5);
}

#[test]
#[ignore]
fn print_sample_report() {
    let r = run(Card::default());
    println!(
        "{}",
        crate::report::report_text(&FakeGpu::new(Card::default()), &r.state, "balanced")
    );
    println!("\n[simulated {:.0} min]", r.sim_minutes);
}

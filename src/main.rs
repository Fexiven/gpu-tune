//! gpu-tune: automatic overclock / undervolt finder for NVIDIA GPUs on Linux.
//!
//! Test loads are Vulkan, like games, so the card runs in P0 with game clocks (CUDA would
//! force P2). Every result is checked bit-for-bit; crashes, reboots and Ctrl+C are survived.

mod flow;
mod hw;
mod nvapi;
mod nvml;
mod proto;
mod report;
mod runner;
mod state;
mod tui;
mod tuner;
mod ui;
mod work;

#[cfg(test)]
mod sim;

use clap::Parser;
use hw::{Gpu, RealGpu, Settings};
use proto::{Kind, Params};
use runner::{ProcessRunner, RealClock, Runner, Stop};
use state::{Saver, Sig, State};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use tuner::PassArgs;
use ui::{Log, PlainUi, Tone, Ui};

#[derive(Parser, Debug)]
#[command(
    name = "gpu-tune",
    version,
    about = "Automatic overclock / undervolt finder for NVIDIA GPUs on Linux.",
    long_about = "Finds the best stable clock offset, clock cap and memory offset for your NVIDIA card, \
                        tested with Vulkan loads like games, and tells you what to enter in LACT.\n\n\
                        Takes about 75 minutes (--quick-only: about 35 min). Ctrl+C stops safely at any time \
                        (GPU goes back to stock); run it again to continue where it stopped."
)]
struct Cli {
    /// what to optimise for: balanced (stock speed, clearly less power), efficient (least power, ~2% slower at most), max
    #[arg(long, default_value = "balanced", value_parser = ["balanced", "efficient", "max"])]
    goal: String,
    /// only the rough pass (~35 min)
    #[arg(long)]
    quick_only: bool,
    /// ignore saved progress and start over
    #[arg(long)]
    fresh: bool,
    /// at the end, compare stock vs result in your own benchmark
    #[arg(long)]
    game_check: bool,
    /// plain text instead of the dashboard
    #[arg(long)]
    plain: bool,
    /// only test the memory temperature sensor
    #[arg(long)]
    check_vram: bool,
    /// check that the error detection works on this machine (injects errors on purpose), then exit
    #[arg(long)]
    self_test: bool,
    /// don't stop if the core offset seems to have no effect
    #[arg(long)]
    skip_offset_check: bool,
    /// don't stop if the test loads don't run in P0
    #[arg(long)]
    ignore_pstate: bool,
    /// which NVIDIA GPU (NVML index)
    #[arg(long, default_value_t = 0)]
    gpu: u32,
    /// max clocks scanned in pass 1 (MHz)
    #[arg(long, num_args = 1.., default_values_t = [2490u32, 2640, 2790, 2940])]
    clocks: Vec<u32>,
    #[arg(long, default_value_t = 450)]
    core_max: i32,
    #[arg(long, default_value_t = 15)]
    core_step: i32,
    /// safety margin below the highest stable core offset (MHz)
    #[arg(long, default_value_t = 45)]
    core_margin: i32,
    #[arg(long, default_value_t = 4000)]
    mem_max: i32,
    /// memory test size in MB per buffer set (0 = auto, ~35% of free VRAM each for source and copy)
    #[arg(long, default_value_t = 0)]
    mem_mb: u64,
    /// pause tests at this GPU temperature (°C)
    #[arg(long, default_value_t = 87)]
    temp_limit: u32,
    /// pause tests at this memory temperature (°C)
    #[arg(long, default_value_t = 102.0)]
    vram_limit: f64,
    /// don't pick a higher memory offset if it heats the memory to this (°C)
    #[arg(long, default_value_t = 95.0)]
    vram_soft: f64,
    #[arg(long, default_value_t = 120)]
    fps: u32,
    /// render size of the test loads
    #[arg(long, default_value_t = 2560)]
    width: u32,
    #[arg(long, default_value_t = 1440)]
    height: u32,
    /// self-test only: run on this Vulkan device (PCI bus id, or "any") without NVIDIA's NVML
    #[arg(long, hide = true)]
    vulkan_device: Option<String>,
    /// state, log and result files go here
    #[arg(long, default_value = ".")]
    dir: PathBuf,
}

/// Environment the workers need (Vulkan driver lookup, NixOS library paths, terminal).
const PASS_ENV: &[&str] = &[
    "TERM",
    "COLORTERM",
    "LD_LIBRARY_PATH",
    "VK_ICD_FILENAMES",
    "VK_DRIVER_FILES",
    "VK_ADD_DRIVER_FILES",
    "XDG_DATA_DIRS",
    "__EGL_VENDOR_LIBRARY_DIRS",
    "NIX_LD_LIBRARY_PATH",
];

fn ensure_root() {
    if unsafe { libc::geteuid() } == 0 {
        return;
    }
    let exe = std::env::current_exe().expect("can't find this program's path");
    eprintln!("Changing GPU clocks needs root, so sudo will ask for your password.");
    let mut cmd = std::process::Command::new("sudo");
    cmd.arg("env");
    for k in PASS_ENV {
        if let Ok(v) = std::env::var(k) {
            cmd.arg(format!("{k}={v}"));
        }
    }
    cmd.arg(exe).args(std::env::args().skip(1));
    use std::os::unix::process::CommandExt;
    let e = cmd.exec();
    eprintln!("could not run sudo: {e}");
    std::process::exit(1);
}

/// Resets the GPU to stock however the program ends (error, panic, Ctrl+C).
/// Blocks suspend and idle actions while the tuner runs (systemd-logind inhibitor). The lock is
/// held by `systemd-inhibit` running a helper that exits as soon as this process does, so even a
/// hard kill can't leave the system unable to sleep.
struct SleepGuard(Option<std::process::Child>);

impl SleepGuard {
    fn start() -> Result<Self, String> {
        let pid = std::process::id().to_string();
        let tool = [
            "systemd-inhibit",
            "/run/current-system/sw/bin/systemd-inhibit",
            "/usr/bin/systemd-inhibit",
            "/bin/systemd-inhibit",
        ]
        .into_iter()
        .find(|t| !t.starts_with('/') || std::path::Path::new(t).exists())
        .unwrap_or("systemd-inhibit");
        let mut child = std::process::Command::new(tool)
            .args([
                "--what=sleep:idle",
                "--who=gpu-tune",
                "--why=GPU tuning in progress",
                "--mode=block",
            ])
            .args(["tail", "--pid", &pid, "-f", "/dev/null"])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("systemd-inhibit not available ({e})"))?;
        std::thread::sleep(std::time::Duration::from_millis(300));
        if let Ok(Some(st)) = child.try_wait() {
            let mut err = String::new();
            if let Some(mut e) = child.stderr.take() {
                let _ = std::io::Read::read_to_string(&mut e, &mut err);
            }
            return Err(format!("systemd-inhibit exited ({st}): {}", err.trim()));
        }
        Ok(SleepGuard(Some(child)))
    }
}

impl Drop for SleepGuard {
    fn drop(&mut self) {
        if let Some(mut c) = self.0.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

struct ResetGuard(Arc<dyn Gpu>);
impl Drop for ResetGuard {
    fn drop(&mut self) {
        self.0.reset();
    }
}

fn base_args(cli: &Cli, gpu: &dyn Gpu) -> PassArgs {
    let free = gpu.free_vram_mb();
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
        clocks: cli.clocks.clone(),
        hints: Default::default(),
        refine_mem: None,
        refine_mem_top: false,
        core_max: cli.core_max.min(gpu.info().core_range.1).max(0),
        core_step: cli.core_step.max(1),
        core_margin: cli.core_margin.max(0),
        mem_max: cli.mem_max.min(gpu.info().mem_range.1).max(0),
        temp_limit: cli.temp_limit,
        vram_limit: cli.vram_limit,
        vram_soft: cli.vram_soft,
        fps: cli.fps.max(10),
        width: cli.width.max(64),
        height: cli.height.max(64),
        // full load: few memory steps, lots of math (calibrated up until it hits the power limit);
        // game-like load: memory/texture-heavy, calibrated to ~60% busy per frame
        iters_heavy: 8,
        iters_paced: 16,
        alu_heavy: 64,
        alu_paced: 2,
        mem_mb: if cli.mem_mb > 0 {
            cli.mem_mb
        } else {
            (free.saturating_sub(768) as f64 * 0.35) as u64
        },
        bus: gpu.info().bus.clone(),
        skip_offset_check: cli.skip_offset_check,
        ignore_pstate: cli.ignore_pstate,
    }
}

fn self_test(runner: &dyn Runner, a: &PassArgs) -> bool {
    println!(
        "Self-test: every test load runs once clean and once with deliberately corrupted results."
    );
    println!(
        "Clean runs must report 0 errors and the same reference image; corrupted runs must be caught.\n"
    );
    let mut ok = true;
    let mut hash: Option<String> = None;
    for kind in [Kind::Heavy, Kind::Paced, Kind::Mem, Kind::Combo] {
        for inject in [false, true] {
            let p = Params {
                kind,
                seconds: 4.0,
                warmup: 0.0,
                bus: a.bus.clone(),
                width: a.width,
                height: a.height,
                iters: a.iters_heavy,
                alu: a.alu_heavy,
                per_frame: 2,
                fps: a.fps,
                mem_mb: a.mem_mb.min(1024),
                inject,
            };
            print!(
                "  {:<26} {:<10} ",
                kind.text(),
                if inject { "corrupted" } else { "clean" }
            );
            let _ = std::io::Write::flush(&mut std::io::stdout());
            let o = match runner.run(&p, 200, 200.0) {
                Ok(o) => o,
                Err(e) => {
                    println!("stopped: {e}");
                    return false;
                }
            };
            let Some(r) = o.res.filter(|_| o.reason.is_none()) else {
                println!("FAILED to run ({})", o.why);
                ok = false;
                continue;
            };
            let errs = r.core_errors + r.mem_errors;
            let same = match (&hash, &r.hash) {
                (Some(a), Some(b)) => a == b,
                (None, Some(b)) => {
                    hash = Some(b.clone());
                    true
                }
                _ => true,
            };
            let good = if inject { errs > 0 } else { errs == 0 && same };
            ok &= good;
            println!(
                "{}  ({} core, {} memory errors{}; P{})",
                if good { "ok" } else { "WRONG" },
                r.core_errors,
                r.mem_errors,
                if same {
                    ""
                } else {
                    ", reference image differs!"
                },
                o.summ.pstate.map_or("?".into(), |p| p.to_string())
            );
        }
    }
    println!(
        "\n{}",
        if ok {
            "Self-test passed: errors are detected."
        } else {
            "Self-test FAILED: don't trust the tuner on this system."
        }
    );
    ok
}

fn unix_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

pub fn load_state(
    path: &std::path::Path,
    sig: &Sig,
    goal: &str,
    quick_only: bool,
    force_fresh: bool,
) -> (State, String) {
    let fresh = |note: &str| {
        if path.exists() {
            let old = path.with_extension("json.old");
            let _ = std::fs::rename(path, &old);
            state::give_back(&old);
        }
        (
            State {
                sig: sig.clone(),
                started: unix_now(),
                goal: goal.into(),
                quick_only,
                ..Default::default()
            },
            note.to_string(),
        )
    };
    if force_fresh || !path.exists() {
        return fresh("");
    }
    let parsed = std::fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_json::from_str::<State>(&t).ok());
    let Some(mut old) = parsed else {
        return fresh("saved progress unreadable: starting over");
    };
    let recent = unix_now() - old.started < 7.0 * 24.0 * 3600.0;
    if old.sig != *sig || !recent || old.p1.baseline.is_none() {
        let note = if old.sig != *sig && !old.done {
            "driver or test settings changed since the saved run: starting over"
        } else {
            ""
        };
        return fresh(note);
    }
    let note = if !old.done {
        "continuing the previous run from where it stopped"
    } else if old.quick_only && !quick_only {
        old.p1.soak_done = false;
        old.p1.soak_result = None;
        "reusing your quick scan, going straight to fine-tuning"
    } else {
        return fresh("");
    };
    old.result = None;
    old.done = false;
    old.goal = goal.into();
    old.quick_only = quick_only;
    (old, note.into())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 3 && args[1] == "--worker" {
        work::worker_main(&args[2]);
    }
    let cli = Cli::parse();
    if cli.self_test {
        if let Some(dev) = &cli.vulkan_device {
            let info = hw::GpuInfo {
                name: dev.clone(),
                driver: "?".into(),
                bus: dev.clone(),
                dev_id: 0,
                default_pl: 0,
                max_pl: 0,
                core_range: hw::HARD_CORE,
                mem_range: hw::HARD_MEM,
                api: "none".into(),
                max_mem_clk: 0,
            };
            let gpu: Arc<dyn Gpu> = Arc::new(hw::NullGpu(info));
            let runner = ProcessRunner { gpu: gpu.clone() };
            let base = base_args(&cli, gpu.as_ref());
            let _ = ctrlc::set_handler(|| runner::INTERRUPTED.store(true, Ordering::SeqCst));
            std::process::exit(if self_test(&runner, &base) { 0 } else { 1 });
        }
    } else {
        ensure_root();
    }
    let real = match RealGpu::open(cli.gpu) {
        Ok(g) => Arc::new(g),
        Err(e) => {
            eprintln!("Can't use the NVIDIA GPU: {e:#}");
            std::process::exit(1);
        }
    };
    if cli.check_vram {
        println!(
            "{}: memory temperature sensor {}",
            real.info().name,
            real.vram.status
        );
        for _ in 0..if real.vram.ok() { 5 } else { 0 } {
            match real.vram.read() {
                Some(v) => println!("  VRAM {v:.1}°C   GPU {}°C", real.temp().unwrap_or(0)),
                None => println!("  VRAM reading failed"),
            }
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
        if !real.vram.ok() {
            println!(
                "Easiest fix: make sure the LACT daemon (0.10 or newer) is running:  systemctl status lactd"
            );
        }
        return;
    }
    let gpu: Arc<dyn Gpu> = real;
    let _guard = ResetGuard(gpu.clone());
    let runner = ProcessRunner { gpu: gpu.clone() };
    let base = base_args(&cli, gpu.as_ref());
    let _ = ctrlc::set_handler(|| runner::INTERRUPTED.store(true, Ordering::SeqCst));
    if cli.self_test {
        std::process::exit(if self_test(&runner, &base) { 0 } else { 1 });
    }

    let _ = std::fs::create_dir_all(&cli.dir);
    let info = gpu.info().clone();
    let sig = Sig {
        v: state::STATE_VERSION,
        gpu: info.name.clone(),
        driver: info.driver.clone(),
        clocks: base.clocks.clone(),
        core_step: base.core_step,
        core_max: base.core_max,
        core_margin: base.core_margin,
        mem_max: base.mem_max,
        fps: base.fps,
        width: base.width,
        height: base.height,
        iters_heavy: base.iters_heavy,
        iters_paced: base.iters_paced,
        alu_heavy: base.alu_heavy,
    };
    let state_path = cli.dir.join("gpu_tune_state.json");
    let (st, note) = load_state(&state_path, &sig, &cli.goal, cli.quick_only, cli.fresh);
    let saver = Saver {
        path: Some(state_path),
        state: std::sync::Mutex::new(st),
    };
    saver.save();

    let p1 = PassArgs::for_pass(&base, true);
    let p2 = PassArgs::for_pass(&base, false);
    let est = tuner::est_pass1(&p1)
        + if cli.quick_only {
            tuner::est_soak(&p1)
        } else {
            tuner::est_pass2(&p2) + tuner::est_soak(&p2)
        };
    let t = unix_now() as libc::time_t;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    unsafe { libc::localtime_r(&t, &mut tm) };
    let log_path = cli.dir.join(format!(
        "gpu_tune_log_{:04}{:02}{:02}_{:02}{:02}{:02}.csv",
        tm.tm_year + 1900,
        tm.tm_mon + 1,
        tm.tm_mday,
        tm.tm_hour,
        tm.tm_min,
        tm.tm_sec
    ));
    let log_file = std::fs::File::create(&log_path).ok();
    state::give_back(&log_path);
    let log = Log::new(log_file);

    let tui = if cli.plain {
        None
    } else {
        tui::TuiUi::start(gpu.clone(), est).ok()
    };
    let plain = PlainUi::new();
    let ui: &dyn Ui = match &tui {
        Some(t) => t,
        None => &plain,
    };
    ui.event(
        "•",
        &format!(
            "offset control: {}, core up to +{}, memory up to +{}",
            info.api, info.core_range.1, info.mem_range.1
        ),
        Tone::Normal,
    );
    ui.event(
        "•",
        &format!("memory test size {} MB x 2", base.mem_mb),
        Tone::Normal,
    );
    match gpu.vram() {
        Some(v) => {
            ui.event(
                "🌡",
                &format!("memory temperature sensor found: {v:.0}°C"),
                Tone::Good,
            );
            ui.event(
                "!",
                "if LACT is running, turn off its automatic profile switching while tuning",
                Tone::Normal,
            );
        }
        None => ui.event(
            "!",
            "memory temperature not available (see --check-vram); continuing without it",
            Tone::Warn,
        ),
    }
    if !note.is_empty() {
        ui.event("↻", &note, Tone::Warn);
    }

    let _sleep_guard = match SleepGuard::start() {
        Ok(g) => {
            ui.event(
                "☾",
                "sleep and idle suspend are blocked while tuning",
                Tone::Normal,
            );
            Some(g)
        }
        Err(e) => {
            ui.event(
                "!",
                &format!("couldn't block sleep ({e}): make sure the computer doesn't suspend"),
                Tone::Warn,
            );
            None
        }
    };
    let clock = RealClock::new();
    let env = flow::Env {
        gpu: gpu.as_ref(),
        runner: &runner,
        clock: &clock,
        ui,
        log: &log,
        saver: &saver,
    };
    let t_start = std::time::Instant::now();
    let res = flow::tune(&env, &base, &cli.goal, cli.quick_only);
    if res.is_ok() {
        std::thread::sleep(std::time::Duration::from_millis(2500)); // show the final state a moment
    }
    if let Some(t) = &tui {
        t.close();
    }
    gpu.reset();
    match res {
        Ok(()) => {}
        Err(Stop::Interrupted) => {
            println!(
                "\nStopped. The GPU is back at stock. Progress is saved: run gpu-tune again to continue."
            );
            return;
        }
        Err(Stop::Thermal) => {
            println!(
                "\nStopped: the card kept overheating even after cooling down. That's a cooling problem, not a tuning result.\n\
                 Improve airflow, or raise --temp-limit / --vram-limit carefully, then run again to continue."
            );
            return;
        }
        Err(Stop::Fatal(m)) => {
            println!(
                "\n{m}. Reboot, then run gpu-tune again: it continues where it stopped and counts the crashed setting as unstable."
            );
            return;
        }
        Err(Stop::Exit(m)) => {
            println!("\n{m}");
            std::process::exit(1);
        }
    }
    let st = saver.state.lock().unwrap().clone();
    let text = report::report_text(gpu.as_ref(), &st, &cli.goal);
    report::save_report(&cli.dir, gpu.as_ref(), &text);
    println!(
        "\n===== ⚡ GPU TUNE: done in {:.0} min =====\n\n{text}\n\n(saved to gpu_tune_result.txt)",
        t_start.elapsed().as_secs_f64() / 60.0
    );
    if cli.game_check
        && let Some(r) = &st.result
    {
        let rec = &r.recommended;
        let pl = if rec.power_max {
            info.max_pl
        } else {
            info.default_pl
        };
        report::game_check(
            gpu.clone(),
            Settings {
                core: rec.core,
                mem: rec.mem,
                clock: Some(rec.max_clock),
                power_mw: pl,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn sleep_guard_fails_gracefully_or_holds_the_lock() {
        match super::SleepGuard::start() {
            Ok(g) => {
                assert!(g.0.is_some());
                drop(g); // releases the lock
            }
            Err(e) => assert!(e.contains("systemd-inhibit"), "{e}"),
        }
    }
}

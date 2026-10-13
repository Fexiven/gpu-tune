# gpu-tune: handoff notes

For whoever continues this project (human or AI). Read this first, then `README.md` (user-facing)
and the module docs at the top of each `src/*.rs` file.

## 1. What it is

A Linux CLI (Rust, one binary) that finds stable undervolt/overclock settings for NVIDIA GPUs
and prints what to enter in [LACT](https://github.com/ilya-zlobintsev/LACT): per profile a
**core clock offset**, a **maximum GPU clock (clock cap)**, a **VRAM clock offset** and a
**power limit**. Three profiles: `efficient`, `balanced`, `performance` (internal name `max`).

Linux has no voltage control for NVIDIA cards. The undervolt is offset + cap: the offset shifts
the whole V/F curve up, the cap stops the card at a clock it now reaches at lower voltage.
The two must always be used together (README explains this to users).

## 2. The user and their system

- RTX 4080 SUPER (PCI device id 0x2702, bus 0000:01:00.0), stock power limit 320 W, max 352 W.
  Stock memory clock 11501 MHz (as nvidia-smi shows it).
- NixOS, runs the tool with `nix run .` from the extracted repo (flake). Also has a Radeon iGPU
  (RADV) and llvmpipe as Vulkan devices; the tool picks the NVIDIA device by PCI bus id.
- Uses LACT 0.10+ to apply settings (it has a "balanced" LACT profile already).
- Prefers plain explanations; asked for honest, critical review of AI-written code.

## 3. How we got here (important decisions and why)

1. **Started as an AI-written Python script** (CuPy/CUDA loads, NVML). A review found many issues;
   the biggest: **CUDA runs GeForce cards in P2, games run in P0**, so CUDA tests ran at different
   (lower memory) clocks than games. Confirmed on the user's card: the CUDA game-like load was P2.
2. **Verified Vulkan runs in P0**: `memtest_vulkan` (headless Vulkan compute) ran in P0 at the full
   11501 MHz memory clock on the user's card. So the rewrite uses Vulkan (wgpu 30) for all loads.
   `--self-test` on the user's card later confirmed P0 for every load.
3. **Rewrite in Rust** (user's choice after asking "which is best"): single binary, same stack as
   LACT, no Python/pip/3 GB CUDA toolkit for users.
4. **Own NVML binding** (`nvml.rs`, libloading) instead of a crate: full control, every return
   code checked. The Python `nvidia-ml-py` wrappers for `nvmlDeviceSet/GetClockOffsets` pass the
   struct by value and ignore return codes; that bug is why we don't trust wrappers here.
5. **Offsets only on P0** (`nvmlDeviceSetClockOffsets`), read back after every set; legacy
   `nvmlDeviceSetGpcClkVfOffset/MemClkVfOffset` as fallback for old drivers.
6. **VRAM temperature through NvAPI** (`nvapi.rs`, `libnvidia-api.so.1`, ported from LACT's
   MIT-licensed code; thermal channel type 3 = memory). Register read (gddr6-tool method, table in
   `hw.rs`) is the fallback. Verified on the user's card: "ok (NvAPI)", ~60 °C idle, 2 °C steps.
   An earlier version read it from the LACT daemon socket; removed at the user's request.
7. **v0.2 real run revealed weak test loads** (see §6): full load drew only 186 W of 320 W and the
   game-like load was never paced (busy 2.18 = every frame missed). Fixed in v0.3 by
   self-calibrating loads (ALU block until power-limited; work-per-pass shrinks to ~60% busy) and
   adding full load to every search step.
8. v0.3 also: default safety margin 45 MHz (log showed longer tests find lower limits).
9. v0.4-0.5: three profiles, each soak-tested; per-profile power-limit search (efficient ≤3% loss,
   balanced ≤1%); performance checks the max power limit; sleep inhibitor (systemd-inhibit).
   Decided NOT to run every test at several power limits (see README "How it works" step 6).
10. v0.6: power search walks down in 10 W steps from the profile's measured draw and stops at
    the knee (`pick_knee` in `decide.rs`: step elasticity perf-loss/power-save > 0.6, steps
    saving < 0.5% ignored, total loss capped, total save ≥ max(3%, 3× noise)). The offset stays
    fixed while the limit changes (asked by the user; answer in README step 6). Pass 1's clock
    caps now default to the card's stock boost −300/−150/0/+150 (`auto_clocks`), capped at
    NVML's max graphics clock (`GpuInfo::max_core_clk`); `--clocks` still overrides. GitHub
    Actions CI added (`.github/workflows/ci.yml`: fmt, clippy, tests, nix build).
    Repo: https://github.com/Fexiven/gpu-tune. Commit as the user (Fexiven), not as the AI.

11. v0.7, from the user's first v0.6 run (§6b): memory pick judged over the whole range
    (`pick_mem`: bandwidth per MHz vs the best so far, gain vs the lowest), capped
    `MEM_ERROR_MARGIN` (500) below the first offset with errors (`PassState::mem_err`, carried
    into pass 2); smallest offset at the voltage floor (`floor_offset`, uses the game-like
    power saved with each core search step as `{"ok":true,"pp":W}`); stock measured three
    times; report says why each power limit was chosen (`Prof::power_note`) and compares
    Performance with Balanced.
12. v0.8, clean-up and speed: `tuner.rs` split into `decide.rs` (pure rules + their tests) and
    `tuner/` (one file per phase); offset search no longer tests +0 up front (implied by any
    higher offset passing) and pass 2 starts ±2 steps around pass 1's limit (was ±4): the
    simulated run went from 109 to 96 min with the same settings. Worker prints `MEASURE` when
    the measured part starts, so averages no longer include device start-up and reference
    rendering (this skewed the game-like clock/power of short tests); the full load keeps one
    batch queued so the GPU never waits for the CPU. `ctrlc` replaced by a libc signal handler,
    ratatui without default features (124 → 114 crates, clean release build 157 → 133 s).

## 4. Architecture (src/)

| file | role |
|---|---|
| `main.rs` | CLI (clap), sudo re-exec (passes Vulkan/NixOS env vars through), modes (`--self-test`, `--check-vram`), saved-progress loading (`load_state`), sleep guard, reset guard, report + `--game-check` |
| `flow.rs` | the overall flow: pass 1 (wide, short tests), pass 2 (planned around each profile's clock), per-profile maxpower / power search, soak per profile, result |
| `decide.rs` | the decision rules, no hardware: `search_bounds` (offset bisection), `pick_mem`, `floor_offset`, `power_stop`/`pick_knee`, `select_profiles`, `auto_clocks`, `interpolate_hints`, noise (`mean`, `spread`), time estimates; most unit tests live here |
| `tuner/mod.rs` | `Env` (GPU, runner, clock, UI, log, saver), `PassArgs`, `Tuner`: `probe` (crash-proof memo), `test` (apply, run, overheat retry, log), recovery, cooldown |
| `tuner/baseline.rs` | stock baseline + load calibration, `calibrate` (P0 check, memory offset units), core-offset effect check |
| `tuner/memory.rs` | memory sweep (cool), hot pick, error cap |
| `tuner/core.rs` | per clock cap: offset search, voltage-floor check, measurement with margin |
| `tuner/finals.rs` | `maxpower`, `power_search`, idle check, `soak` |
| `work.rs` | the worker process (`gpu-tune --worker <json>`): wgpu device by PCI bus id; render load (fragment shader: FP32 math, texture sampling, random storage reads, optional ALU block) with a GPU-side bit-exact compare against a reference image after every pass; memory load (buffer copies + address-dependent pattern check); combo; `inject` for the self-test |
| `proto.rs` | worker parameters/result (JSON), `golden_key` (reference image identity) |
| `runner.rs` | starts a worker, samples NVML every 200 ms meanwhile, overheat/hang handling, `summarize` (from the worker's `MEASURE` line on); Ctrl+C/SIGTERM/SIGHUP handler; `Stop` enum, `Clock` trait, `INTERRUPTED` flag |
| `hw.rs` | `Gpu` trait + `RealGpu` (NVML control, P0 offsets with read-back, ranges), `NullGpu` (self-test without NVML), kernel log (`/dev/kmsg` Xid watcher), VRAM sensor (NvAPI then register) |
| `nvml.rs` / `nvapi.rs` | hand-written FFI |
| `state.rs` | saved progress (`gpu_tune_state.json`, atomic writes, chown back to `SUDO_UID`), `Sig` (invalidates progress when driver/settings/STATE_VERSION change), result types |
| `report.rs` | result text (also `gpu_tune_result.txt`), score parsing, `--game-check` |
| `ui.rs` / `tui.rs` | `Ui` trait, plain output, CSV log; ratatui dashboard (Ctrl+C / q handled as keys in raw mode) |
| `sim.rs` | tests only: simulated RTX 4080 SUPER with known limits + scenario tests |

Key mechanisms worth knowing before changing anything:

- **Crash/reboot safety**: `Tuner::probe(key, f)` saves `pending = key` before a test and the result
  after. A hard crash leaves `pending` on disk; next run records it as failed. Soak keeps
  `soaks[name].at` the same way and backs off one step after a crash. Keys encode all settings.
- **Errors**: `Stop::Interrupted` (Ctrl+C), `Stop::Thermal` (cooling can't keep up; overheating
  is never counted as instability), `Stop::Fatal` (GPU gone, reboot), `Stop::Exit` (message).
- **Noise**: stock is measured three times (spread = (max-min)/mean); thresholds use max(fixed floor, 2× measured noise).
- **Memory decisions** use measured clocks (`mem_step_worth_it`): a step counts only if bandwidth
  rises by ≥ half the clock increase and > 2× noise (GDDR6X error detection & replay eats
  bandwidth when overclocked too far). Errors from the pattern check stop the sweep.
- **Profiles** (`select_profiles`): efficient = least game-like power with clock ≥ stock−2%;
  balanced = highest clock among ≤90% stock power and clock ≥ stock−0.5%; max = fastest.
  "Clock" = core clock in the paced load (not fps; memory OC not included).
- **Load calibration** (stock baseline): heavy `alu` doubles from 64 up to 512 until the run is
  power-limited (NVML reasons) or ≥90% of the limit; paced `per_frame`/`paced_iters` adjust until
  busy is 45-75% at `--fps` (default 120). Stored in pass state, inherited by pass 2.

## 5. Verification status

| what | how verified |
|---|---|
| whole tuning flow, crashes, reboots, Ctrl+C, overheat, three profiles, power search, calibrations, estimates | `cargo test`: 39 tests (+2 ignored printers: `print_sample_report`, `print_durations`) incl. simulated card (`sim.rs`) |
| Vulkan loads run, are deterministic, detect injected errors | lavapipe in the dev container: `gpu-tune --self-test --vulkan-device any --width 320 --height 192 --mem-mb 64`; user's 4080 SUPER: `--self-test` all ok, P0 |
| VRAM temperature via NvAPI | user's card: `--check-vram` ok |
| NixOS flake builds and runs | user's machine (v0.2); CI builds it on every push (`nix build`) |
| NVML control, full tuning run | user's card, **v0.2 only** (results §6) |
| **not yet verified on hardware** | v0.3+ load calibration (does heavy reach 320 W? how big does `alu` get?), paced sizing, full load in search steps, three profiles, power knee search (real knee position on a 4080 Super?), auto clock range, sleep inhibitor, dashboard after v0.2 |

First thing to do with new hardware results: read the CSV log (`gpu_tune_log_*.csv`). Check
`limited_by=power` on stock heavy rows, `busy` ≈ 0.45-0.75 on paced rows with `missed` ≈ 0,
`pstate` 0 everywhere, and where `math errors` vs `crash` appear relative to the chosen offsets.

## 6. Real data from the user's card (v0.2 run, 58 min)

- Stock: paced/heavy core clock ~2730 MHz, **heavy only 186 W, not power-limited** (fixed in v0.3).
  Paced `busy` 2.18 (not paced; fixed in v0.3).
- Memory: pattern errors first at **+3500** (13251 MHz); hot test fine to +3000; chosen **+2250**
  (12626 MHz). LACT shows P0 VRAM 12626 MHz with +2250, so **LACT offsets = NVML units**
  (2× DDR: +250 offset = +125 MHz).
- Core (pass 2, 30 s steps, mem +2250): math errors appear ~15-30 MHz below crashes, e.g. at
  2850 cap: +300 ok, +315/+330 errors, +390 crash. Pass 1 (15 s) found higher limits than pass 2
  (e.g. 2790: +345 vs +315) → longer tests find lower limits → margin raised to 45.
- Result v0.2 (balanced): cap 2850, core +270, mem +2250, 320 W. Soak passed incl. idle.
  Caveat told to the user: core only proven at ~130-170 W; suggested lowering to +240/+210 if a
  heavy game crashes. User applied it in LACT (screenshot confirmed values).
- During the memory sweep the user saw screen artifacts (expected: display reads VRAM).

## 6b. Real data from the user's card (v0.6 run, 98 min, driver 595.104.02)

- Stock: full load 319.5 W, limited by power, alu 64 was enough; game-like load 37 steps/pass,
  ~49% busy, 0 missed. Stock boost (game-like) ~2706 MHz → auto caps 2400/2550/2700/2850.
- Core: errors always appeared first in the **full load**, 15-30 MHz below crashes, often only
  5-30 wrong values; the game-like load passed offsets the full load failed. Pass 2 limits:
  2670 → +300, 2700 → +300, 2730 → +285, 2820/2850/2880 → +270. Pass 1 and 2 agreed (2850: +270).
- Voltage floor: at 2400 every offset up to +450 passed and game-like power stayed ~100 W from
  +225 on; same at 2550 from +225. Motivated `floor_offset`.
- Memory: cool errors at +3500 (13251 MHz); hot +2500..+3250 clean, bandwidth per MHz constant
  (no retries). v0.6 picked +2500 because each 250 step (+1.1%) was below 2× noise (1.56%):
  the bug fixed in v0.7. memtest_vulkan (user) showed the memory at 96 °C; the tuner's combo
  load peaked at 88 °C.
- Power search: no knee within the budgets; step cost 0.28-0.37 at 2670 MHz/+255.
  Efficient 270 W (−2.2%, −7.0%), stopped by the 3% cap at 260 W; Balanced 310 W (−0.7%,
  −3.3%), stopped by the 1% cap at 300 W. Max limit 352 W: +2.0% perf for +10% power.
- Result v0.6: Efficient 2670/+255/+2500/270 W, Balanced 2850/+225/+2500/310 W, Performance
  2880/+225/+2500/352 W; all soaks passed first try.

## 7. Open work / ideas (rough priority)

1. **Get a v0.7 run log from the user** (expect memory +3000, see §6b) and check §5's list. Likely tuning needed: `alu` range,
   paced sizing, time estimates on real hardware.
2. **V/F curve editing via NvAPI** (LACT has it: `QUERY_NVAPI_GPU_CLOCK_CLIENT_CLK_VF_POINTS_*` in
   LACT's `nvapi.rs`). A real undervolt curve would leave the low idle points untouched instead
   of shifting them (offset+cap), removing the main source of idle crashes. Big change; would
   need LACT's VF-curve export format or a different "enter in LACT" output.
3. Write results directly as LACT profiles (LACT API `get_gpu_config`/`set_gpu_config` over
   `/run/lactd.sock`, see LACT `docs/API.md`; config changes must be confirmed or they revert).
   Must ask the user first and never apply untested combinations.
4. Release binaries for non-Nix distros (dynamic glibc; needs libvulkan.so.1 + NVIDIA driver).
   Not built or tested on other distros yet.
5. `--game-check` is manual (user types a benchmark score); could automate with a known Vulkan
   benchmark, but keep it optional.
6. RTX 50 (GDDR7): NvAPI path should work (LACT supports Blackwell); register fallback table has
   no 5090 decode (gddr6 tool uses a different method there).

## 8. Development

- Rust ≥ 1.90 (wgpu 30 needs ordered-float 5, MSRV 1.90; CI builds with 1.90), edition 2024. wgpu **30** (API differs from older versions:
  `PollType::Wait{submission_index,timeout}`, `enumerate_adapters` is async, adapter info has
  `device_pci_bus_id`, pipeline layouts take `&[Option<&BindGroupLayout>]`).
- `serde_json` needs `float_roundtrip` (state files must round-trip floats exactly; a test caught it).
- `cargo test` (no GPU needed), `cargo clippy --all-targets` (keep at 0 warnings), `cargo fmt`.
- Real Vulkan without NVIDIA: install `mesa-vulkan-drivers` (lavapipe), use `--vulkan-device any`.
- NixOS: `nix build` / `nix run` / `nix develop`. In a git checkout, flakes only see committed
  files: commit before `nix build`.
- Bump `STATE_VERSION` (state.rs) whenever test loads or saved fields change meaning, so old
  progress isn't reused. Bump the version in both `Cargo.toml` and `flake.nix`.
- User-facing text: plain language, short sentences, say what it means for the user.
- Commits end with the co-author/session lines already used in `git log`.

## 9. Safety rules to keep

- Never raise voltage; only offsets/caps/power limits within the driver's reported ranges.
- Always reset the GPU to stock on exit (ResetGuard, Ctrl+C, panic).
- Never count overheating as instability; stop if cooling can't keep up.
- Never recommend an offset without its clock cap, or a setting that wasn't soak-tested.
- Be honest in reports about what was and wasn't tested (e.g. power reached by the full load).

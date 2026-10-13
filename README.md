# gpu-tune

Automatic overclock / undervolt finder for NVIDIA GPUs on Linux. It finds the best stable
**core clock offset**, **clock cap** and **memory offset** for your card and tells you what to
enter in [LACT](https://github.com/ilya-zlobintsev/LACT). It takes about an hour, and you can stop
and continue at any time.

Made for the RTX 4080 Super; works on RTX 30/40/50 cards with the proprietary driver (555+ recommended).

```
nix run github:<you>/gpu-tune          # NixOS / Nix
# or
cargo build --release && ./target/release/gpu-tune
```

It asks for your sudo password (changing clocks needs root), shows a live dashboard, and ends
with something like:

```
Recommended (balanced):
   game-like load:  core clock -0.4% vs stock, -25 W at a fixed 120 fps
   full load:       +3.2% performance, -1 W

Enter in LACT (for P0, the highest P-state):
   GPU clock offset    +150 MHz
   VRAM clock offset   +1250 MHz
   Maximum GPU clock   2790 MHz
   Power limit         320 W (default)
```

**Ctrl+C (or q) stops safely at any time**: the GPU goes back to stock and progress is saved.
Run it again to continue. Driver crashes during testing are expected (that's how limits are
found), and even a crash that needs a reboot is survived: the setting that was running is
counted as unstable next time.

## Options

| | |
|---|---|
| `--goal balanced` | stock speed in games with clearly less power (default) |
| `--goal efficient` | least power while at most ~2% slower |
| `--goal max` | most performance |
| `--quick-only` | only the rough pass (~30 min) |
| `--self-test` | proves the error detection works on your machine (injects errors on purpose) |
| `--game-check` | at the end, compare stock vs result in your own benchmark |
| `--check-vram` | only test the memory temperature sensor |
| `--plain` | plain text instead of the dashboard |
| `--fresh` | ignore saved progress |

Run `gpu-tune --help` for everything else (clock list, search range, temperature limits, ...).

## How it works

**Vulkan loads, like games.** NVIDIA runs CUDA programs in the P2 power state (often with lower
memory clocks), while games run in P0. Results found with CUDA don't transfer reliably to games.
All test loads here are Vulkan, so the card runs exactly like in a game. The tuner checks that
and stops if the tests don't run in P0.

**Every result is checked bit-for-bit.** Unstable clocks produce wrong results before they crash.
The game-like load renders an image (FP32 math, texture sampling, random memory reads); after
every pass the GPU compares it with a reference image, and the reference itself is compared with
the one rendered at stock. The memory test copies large buffers holding an address-dependent
pattern and checks every word. Each test runs in its own process, so a crash only kills the test.

**Steps:**

1. Stock performance, measured twice to know the run-to-run noise. Every later decision must
   beat twice that noise, so random variation can't pick the result.
2. Calibration: tests must run in P0; the memory offset's real effect is measured; a +60 core
   offset must actually raise the clock (otherwise the search would find nothing real).
3. Memory: the bandwidth peak. Too much memory clock causes transfer retries (GDDR6X error
   detection and replay), which show up as lost bandwidth. The best steps are re-checked with a
   hot GPU, and a step only counts if it clearly adds bandwidth.
4. Core: for several clock caps, a binary search for the highest stable offset under game-like
   load (bursts at 120 fps: the card isn't power-limited, like most games). A higher offset at the
   same clock cap means lower voltage, so less power. Then measured with a 30 MHz safety margin.
5. Pass 2 repeats steps 3-4 around the best area with full-length tests and finer steps.
6. Soak test of the result: cold start, full load, game-like load, memory with a hot GPU, and
   **desktop idle with sudden load**. An offset shifts the whole voltage/frequency curve,
   including the low idle points, which is where Linux undervolts often crash.

Overheating is never counted as instability: the tuner cools down and retries, and stops if the
cooling can't keep up.

## Memory temperature

GDDR6X gets hot, so the tuner watches it and won't pick a memory offset that heats it too much.
NVIDIA's NVML doesn't report memory temperature on GeForce cards, so it's read through NvAPI
(`libnvidia-api.so.1`, part of the driver since 525), the same way LACT does it; no setup needed.
On older drivers it falls back to the card's register like the
[gddr6](https://github.com/olealgoritme/gddr6) tool (needs Secure Boot off and sometimes the
kernel option `iomem=relaxed`). Without either, everything else still works. Check with
`gpu-tune --check-vram`.

If LACT is running, turn off its automatic profile switching while tuning.

## After tuning

1. Enter the values in LACT (P0 row if it shows one row per P-state). The report also tells you
   which memory clock `nvidia-smi` should show in a game, so you can check the offset units.
2. Run [memtest_vulkan](https://github.com/GpuZelenograd/memtest_vulkan) once: bandwidth tests
   can't prove the memory is error-free.
3. Play your games for a few days, and use the desktop normally (idle, videos).
   - Game crashes or driver reset: lower the GPU clock offset by 15-30
   - Crashes at idle / on the desktop: lower the GPU clock offset by 15-30
   - Artifacts / flickering: lower the VRAM offset by 250

## Files

Written to the current directory (or `--dir`), owned by your user:
`gpu_tune_state.json` (progress), `gpu_tune_log_*.csv` (every test), `gpu_tune_result.txt`.

## Building

Needs Rust 1.88+, the NVIDIA proprietary driver and the Vulkan loader (`libvulkan.so.1`).

```
cargo build --release
cargo test            # runs the whole tuning flow against a simulated GPU, no GPU needed
```

On NixOS: `nix build`, `nix run`, or `nix develop` for a shell with everything set up.

## Safety

Settings stay within the driver's offset and power limits. The GPU is reset to stock when the
tuner ends, however it ends. Use at your own risk: compute-stable isn't always game-stable,
so confirm with your own games.

## License

MIT

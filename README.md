# gpu-tune

Automatic overclock / undervolt finder for NVIDIA GPUs on Linux. It finds three soak-tested
profiles for your card (**efficient**, **balanced**, **performance**), each a core clock offset,
clock cap and memory offset, and tells you what to enter in
[LACT](https://github.com/ilya-zlobintsev/LACT) (save them as LACT profiles). It takes about
1 hour 30 minutes (`--single`: one profile, ~65 min), and you can stop and continue at any time.

Made for the RTX 4080 Super; works on RTX 30/40/50 cards with the proprietary driver (555+ recommended).

```
nix run github:Fexiven/gpu-tune         # NixOS / Nix
# or
cargo build --release && ./target/release/gpu-tune
```

It asks for your sudo password (changing clocks needs root), shows a live dashboard, and ends
with something like:

```
3 profiles, each soak-tested. Save them as LACT profiles and switch as you like.
Tested up to the power limit: full load drew 317 W at stock (320 W limit).

Efficient  (least power, at most ~2% slower than stock)
   game-like load:  core clock -1.4% vs stock, -32 W at a fixed 120 fps
   LACT (P0):       GPU clock offset +150, VRAM clock offset +1250, maximum GPU clock 2760 MHz, ...

★ Balanced  (stock speed or better, clearly less power)
   ...
Performance  (fastest)
   ...
```

While it runs, the computer won't go to sleep (it holds a systemd-logind inhibitor, released
automatically when the tuner ends, even if it's killed).

**Ctrl+C (or q) stops safely at any time**: the GPU goes back to stock and progress is saved.
Run it again to continue. Driver crashes during testing are expected (that's how limits are
found), and even a crash that needs a reboot is survived: the setting that was running is
counted as unstable next time.

## Options

| | |
|---|---|
| `--goal balanced` | which profile is marked as recommended (default; also `efficient`, `performance`) |
| `--single` | only find and soak-test the `--goal` profile (~65 min) |
| `--quick-only` | only the rough pass (~40 min, ~30 with `--single`) |
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

1. Stock performance, measured three times to know the run-to-run noise. Every later decision must
   beat twice that noise, so random variation can't pick the result. Both test loads calibrate
   themselves here: the full load gets heavier until it reaches the power limit (like a heavy
   game: an undervolt that holds at low power can still crash at high power, where more current
   means more voltage drop), and the game-like load is sized to ~60% busy per frame, so there are
   real idle gaps between frames.
2. Calibration: tests must run in P0; the memory offset's real effect is measured; a +60 core
   offset must actually raise the clock (otherwise the search would find nothing real).
3. Memory: too much memory clock causes transfer retries (GDDR6X error detection and replay),
   which show up as bandwidth that no longer follows the clock, and then as wrong data. The
   tuner keeps the highest offset where bandwidth per MHz still holds, judged over the whole
   range (one small step adds less than the noise, the range as a whole doesn't), re-checked with
   a hot GPU, and always at least 500 below the first offset that showed errors.
4. Core: for four clock caps around your card's stock boost clock (300 MHz below to 150 MHz
   above, never above the driver's maximum; or your own list with `--clocks`), a binary search for the highest stable offset. Each step runs
   the game-like load (bursts at 120 fps, load changing every frame) and then the full load at
   the power limit. A higher offset at the same clock cap means lower voltage, so less power. The
   result is used with a 45 MHz safety margin (longer tests find lower limits). At low clock
   caps the voltage is already at its minimum, so beyond some offset power stops falling; there
   the tuner uses the smallest offset that reaches it (more would only raise the idle points of
   the curve, where Linux undervolts crash, for no gain).
5. Pass 2 repeats steps 3-4 around each profile's area with full-length tests and finer steps.
6. Power limit (efficient and balanced profiles): full load with the profile's settings, first
   at the default limit, then in 10 W steps down from what it actually draws. At first a lower
   limit mostly lowers the voltage, which costs little performance; at some point the card has
   no voltage left to drop and only loses clock speed, so every watt saved costs performance.
   The search stops at that knee (a step that costs more than 0.6% performance per 1% power),
   or when the total loss passes 3% (efficient) or 1% (balanced). The last limit before it is
   kept if it saves clearly more power than the noise (at least 3%). The report says why the
   search stopped where it did.
   The performance profile instead checks whether the card's maximum limit adds performance.
   (The offset stays the same while the limit changes: stability is decided at the highest
   current, at the clock cap, and a lower limit only moves the card down the same curve. The
   soak test at the chosen limit confirms it.)
7. Soak test of each profile, at its power limit: cold start, full load, game-like load, memory with a hot GPU, and
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

## Why a clock offset *and* a lower maximum clock?

Linux has no direct voltage control for NVIDIA cards, so an undervolt takes two settings:

1. The **clock offset** shifts the whole voltage/frequency curve up: every voltage step now runs
   that many MHz faster.
2. The **maximum GPU clock** stops the card from using that headroom to boost even higher. It
   reaches the cap at a lower voltage than stock would need for that clock: less power and heat.

The cap only looks low next to the slider's range: that range is the driver's maximum, not what
the card actually runs (a 4080 Super boosts to ~2700-2800 MHz at stock). **Never use the offset
without the cap**: the card would then boost to stock voltage plus the offset, a pure overclock
the tuner never tested.

## After tuning

1. Enter the values in LACT (P0 row if it shows one row per P-state). The report also tells you
   which memory clock `nvidia-smi` should show in a game, so you can check the offset units.
2. Run [memtest_vulkan](https://github.com/GpuZelenograd/memtest_vulkan) once: bandwidth tests
   can't prove the memory is error-free. It heats the memory more than any game (mid-90s °C
   on a 4080 Super is normal there); what matters is that it finds no errors.
3. Play your games for a few days, and use the desktop normally (idle, videos).
   - Game crashes or driver reset: lower the GPU clock offset by 15-30
   - Crashes at idle / on the desktop: lower the GPU clock offset by 15-30
   - Artifacts / flickering: lower the VRAM offset by 250

## Files

Written to the current directory (or `--dir`), owned by your user:
`gpu_tune_state.json` (progress), `gpu_tune_log_*.csv` (every test), `gpu_tune_result.txt`.

## Building

Needs Rust 1.90+, the NVIDIA proprietary driver and the Vulkan loader (`libvulkan.so.1`).

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

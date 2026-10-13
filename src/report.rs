//! The final summary (also saved to gpu_tune_result.txt) and the optional benchmark check.

use crate::hw::{Gpu, Settings};
use crate::state::{State, give_back};
use crate::tuner::game_rel;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub fn report_text(gpu: &dyn Gpu, st: &State, goal: &str) -> String {
    let info = gpu.info();
    let cal = st.p1.calib.clone().unwrap_or_default();
    let mut l: Vec<String> = vec![];
    let Some(res) = &st.result else {
        return "No tuned setting passed all tests. Keep your card at stock.".into();
    };
    let (rec, b) = (&res.recommended, &res.baseline);
    let pl = if rec.power_max {
        format!("{:.0} W (max)", info.max_pl as f64 / 1000.0)
    } else {
        format!("{:.0} W (default)", info.default_pl as f64 / 1000.0)
    };
    l.push(format!("Recommended ({goal}):"));
    l.push(format!(
        "   game-like load:  core clock {:+.1}% vs stock, {:+.0} W at a fixed 120 fps",
        (game_rel(rec, b) - 1.0) * 100.0,
        rec.paced_power - b.paced_power
    ));
    l.push(format!(
        "   full load:       {:+.1}% performance, {:+.0} W",
        (rec.perf / b.perf - 1.0) * 100.0,
        rec.heavy_power - b.heavy_power
    ));
    l.push(
        "   (the clock figure doesn't include the memory overclock; check fps with --game-check)"
            .into(),
    );
    if b.heavy_limited {
        l.push(format!(
            "   Tested up to the power limit: full load drew {:.0} W at stock ({:.0} W limit), {:.0} W tuned.",
            b.heavy_power, b.power_limit_w, rec.heavy_power
        ));
    } else if b.power_limit_w > 0.0 {
        l.push(format!(
            "   Note: the full load only reached {:.0} W of {:.0} W, so the core offset is untested at full gaming power.",
            b.heavy_power, b.power_limit_w
        ));
        l.push("   Check it in your most demanding game; lower the GPU clock offset by 15-30 if it crashes.".into());
    }
    l.push(String::new());
    l.push(
        "Enter in LACT (for P0, the highest P-state, if LACT shows one row per P-state):".into(),
    );
    l.push(format!("   GPU clock offset    {:+} MHz", rec.core));
    l.push(format!("   VRAM clock offset   {:+} MHz", rec.mem));
    l.push(format!("   Maximum GPU clock   {} MHz", rec.max_clock));
    l.push(format!("   Power limit         {pl}"));
    if cal.k > 0.0 && rec.mem != 0 {
        l.push(format!(
            "Check: in a game, nvidia-smi should show the memory at about {:.0} MHz (stock {:.0} MHz).",
            cal.mem_clk + rec.mem as f64 * cal.k,
            cal.mem_clk
        ));
        l.push("   If it's off by half or double the change, LACT uses different offset units: adjust the VRAM value.".into());
    }
    if let Some(v) = rec.vram_max {
        l.push(format!(
            "Memory temperature: up to {v:.0}°C during the soak test{}",
            if v >= 90.0 {
                "  (warm: consider a lower VRAM offset for daily use)"
            } else {
                "  (fine)"
            }
        ));
    }
    l.push(String::new());
    let others: Vec<_> = res
        .profiles
        .iter()
        .filter(|(n, p)| n.as_str() != goal && p.max_clock != rec.max_clock)
        .collect();
    if !others.is_empty() {
        l.push("Other options from this run (not soak-tested):".into());
        for (n, p) in others {
            l.push(format!(
                "   {n:<10} max {} MHz, core {:+}, mem {:+}: clock {:+.1}%, {:+.0} W at 120 fps",
                p.max_clock,
                p.core,
                p.mem,
                (game_rel(p, b) - 1.0) * 100.0,
                p.paced_power - b.paced_power
            ));
        }
        l.push(String::new());
    }
    l.extend(
        [
            "Next:",
            "  1. Run memtest_vulkan once with these settings: bandwidth tests can't prove the memory is error-free.",
            "  2. Play your games for a few days, and leave the desktop idle / watch videos too.",
            "  Game crashes or driver reset  -> lower the GPU clock offset by 15-30",
            "  Crashes at idle / on desktop  -> lower the GPU clock offset by 15-30",
            "  Artifacts / flickering        -> lower the VRAM offset by 250",
            "The GPU was reset to stock; re-apply LACT with: sudo systemctl restart lactd",
        ]
        .map(String::from),
    );
    l.join("\n")
}

pub fn save_report(dir: &std::path::Path, gpu: &dyn Gpu, text: &str) {
    let p = dir.join("gpu_tune_result.txt");
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let body = format!(
        "gpu-tune result (unix time {t}), {}, driver {}\n\n{text}\n",
        gpu.info().name,
        gpu.info().driver
    );
    if crate::state::write_atomic(&p, &body).is_ok() {
        give_back(&p);
    }
}

/// "143.7 fps" -> 143.7, "12,345" -> 12345, "98,5" -> 98.5
pub fn parse_score(s: &str) -> Option<f64> {
    let start = s.find(|c: char| c.is_ascii_digit())?;
    let num: String = s[start..]
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == ',')
        .collect();
    let num = num.trim_end_matches(['.', ',']);
    let thousands = num
        .split(',')
        .skip(1)
        .all(|g| g.len() == 3 && g.chars().all(|c| c.is_ascii_digit()))
        && num.contains(',')
        && num
            .split(',')
            .next()
            .is_some_and(|g| (1..=3).contains(&g.len()));
    let n = if thousands {
        num.replace(',', "")
    } else if num.matches(',').count() == 1 && !num.contains('.') {
        num.replace(',', ".")
    } else {
        num.to_string()
    };
    n.parse().ok().filter(|v: &f64| *v > 0.0)
}

pub fn game_check(gpu: Arc<dyn Gpu>, rec: Settings) {
    println!("\nReal benchmark check: run the SAME benchmark with the same settings both times.");
    let mut out = vec![];
    for (name, s) in [("stock", None), ("tuned", Some(rec))] {
        match s {
            None => gpu.reset(),
            Some(s) => {
                if let Err(e) = gpu.apply(&s) {
                    println!("  could not apply the settings: {e:#}");
                    gpu.reset();
                    return;
                }
            }
        }
        println!("\n  {name} settings applied.");
        prompt("  Press Enter, THEN start the benchmark... ");
        let stop = Arc::new(AtomicBool::new(false));
        let sampler = {
            let (g, stop) = (gpu.clone(), stop.clone());
            std::thread::spawn(move || {
                let mut v = vec![];
                while !stop.load(Ordering::SeqCst) {
                    if let Ok(s) = g.snapshot()
                        && s.util >= 50
                    {
                        v.push(s.power);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(500));
                }
                if v.is_empty() {
                    0.0
                } else {
                    v.iter().sum::<f64>() / v.len() as f64
                }
            })
        };
        let score = loop {
            if let Some(v) = parse_score(&prompt(
                "  When it finishes, type the score (or average fps): ",
            )) {
                break v;
            }
        };
        stop.store(true, Ordering::SeqCst);
        out.push((name, score, sampler.join().unwrap_or(0.0)));
    }
    println!("\n  {:<8}{:>10}{:>8}", "", "score", "watts");
    for (n, s, w) in &out {
        println!("  {n:<8}{s:>10.1}{w:>8.0}");
    }
    let ((_, s0, w0), (_, s1, w1)) = (out[0], out[1]);
    println!(
        "\n  tuned vs stock: {:+.1}% score, {:+.0} W",
        (s1 / s0 - 1.0) * 100.0,
        w1 - w0
    );
    gpu.reset();
}

fn prompt(q: &str) -> String {
    print!("{q}");
    let _ = std::io::stdout().flush();
    let mut s = String::new();
    let _ = std::io::stdin().read_line(&mut s);
    s
}

#[cfg(test)]
mod tests {
    use super::parse_score;
    #[test]
    fn scores() {
        assert_eq!(parse_score("143.7 fps"), Some(143.7));
        assert_eq!(parse_score("12,345"), Some(12345.0));
        assert_eq!(parse_score("98,5"), Some(98.5));
        assert_eq!(parse_score("score: 15000"), Some(15000.0));
        assert_eq!(parse_score("abc"), None);
    }
}

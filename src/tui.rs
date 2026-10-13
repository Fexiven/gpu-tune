//! Full-screen live dashboard. Draws in its own thread; Ctrl+C or q stops the tuner safely.

use crate::hw::{Gpu, Snap};
use crate::runner::INTERRUPTED;
use crate::ui::{Finding, RowState, Tone, Ui};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Cell, Paragraph, Row, Table, Wrap};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

struct Cur {
    short_kind: String,
    desc: String,
    short: String,
    start: Instant,
    dur: f64,
}

struct St {
    stage: String,
    phase: String,
    why: String,
    cur: Option<Cur>,
    events: VecDeque<(String, String, String, Tone)>,
    rows: Vec<(String, Finding)>,
    tele: Option<(Snap, f64)>,
    hist: [VecDeque<f64>; 4], // power, temp, clock, vram
    t0: Instant,
    est: f64,
}

pub struct TuiUi {
    st: Arc<Mutex<St>>,
    stop: Arc<AtomicBool>,
    th: Mutex<Option<JoinHandle<()>>>,
}

fn hhmm() -> String {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as libc::time_t;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    unsafe { libc::localtime_r(&t, &mut tm) };
    format!("{:02}:{:02}", tm.tm_hour, tm.tm_min)
}

pub fn fmt_time(s: f64) -> String {
    let s = s.max(0.0) as u64;
    if s >= 3600 {
        format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
    } else {
        format!("{}:{:02}", s / 60, s % 60)
    }
}

impl TuiUi {
    pub fn start(gpu: Arc<dyn Gpu>, est: f64) -> std::io::Result<Self> {
        let st = Arc::new(Mutex::new(St {
            stage: "Starting".into(),
            phase: String::new(),
            why: String::new(),
            cur: None,
            events: VecDeque::new(),
            rows: vec![],
            tele: None,
            hist: Default::default(),
            t0: Instant::now(),
            est,
        }));
        let stop = Arc::new(AtomicBool::new(false));
        let mut term = ratatui::try_init()?;
        let th = {
            let (st, stop) = (st.clone(), stop.clone());
            std::thread::spawn(move || {
                let mut last_tele = Instant::now() - Duration::from_secs(1);
                while !stop.load(Ordering::SeqCst) {
                    if last_tele.elapsed() >= Duration::from_millis(500) {
                        last_tele = Instant::now();
                        if let Ok(s) = gpu.snapshot() {
                            let lim = gpu.power_limit() as f64 / 1000.0;
                            let mut g = st.lock().unwrap();
                            for (i, v) in [
                                Some(s.power),
                                Some(s.temp as f64),
                                Some(s.clk as f64),
                                s.vram,
                            ]
                            .into_iter()
                            .enumerate()
                            {
                                if let Some(v) = v {
                                    g.hist[i].push_back(v);
                                    if g.hist[i].len() > 120 {
                                        g.hist[i].pop_front();
                                    }
                                }
                            }
                            g.tele = Some((s, lim));
                        }
                    }
                    {
                        let g = st.lock().unwrap();
                        let info = gpu.info();
                        let _ = term.draw(|f| {
                            draw(f, &g, &info.name, &info.driver, info.max_pl as f64 / 1000.0)
                        });
                    }
                    if event::poll(Duration::from_millis(125)).unwrap_or(false)
                        && let Ok(Event::Key(k)) = event::read()
                    {
                        let ctrl_c = k.code == KeyCode::Char('c')
                            && k.modifiers.contains(KeyModifiers::CONTROL);
                        if k.kind == KeyEventKind::Press && (ctrl_c || k.code == KeyCode::Char('q'))
                        {
                            INTERRUPTED.store(true, Ordering::SeqCst);
                            let mut g = st.lock().unwrap();
                            g.stage =
                                "Stopping safely: GPU goes back to stock, progress is saved..."
                                    .into();
                        }
                    }
                }
                ratatui::restore();
            })
        };
        Ok(TuiUi {
            st,
            stop,
            th: Mutex::new(Some(th)),
        })
    }

    /// Leave the dashboard and give the terminal back.
    pub fn close(&self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.th.lock().unwrap().take() {
            let _ = t.join();
        }
    }
}

impl Drop for TuiUi {
    fn drop(&mut self) {
        self.close();
    }
}

impl Ui for TuiUi {
    fn set_stage(&self, text: &str) {
        self.st.lock().unwrap().stage = text.into();
        self.event("»", text, Tone::Stage);
    }
    fn set_phase(&self, name: &str, why: &str) {
        let mut g = self.st.lock().unwrap();
        g.phase = name.into();
        g.why = why.into();
    }
    fn test_start(&self, short_kind: &str, desc: &str, dur: f64, short: &str) {
        self.st.lock().unwrap().cur = Some(Cur {
            short_kind: short_kind.into(),
            desc: desc.into(),
            short: short.into(),
            start: Instant::now(),
            dur,
        });
    }
    fn test_end(&self, ok: bool, result: &str, brief: &str) {
        let cur = self.st.lock().unwrap().cur.take();
        if let Some(c) = cur {
            let b = if brief.is_empty() { result } else { brief };
            self.event(
                if ok { "✓" } else { "✗" },
                &format!("{} {} → {b}", c.short_kind, c.short),
                if ok { Tone::Good } else { Tone::Bad },
            );
        }
    }
    fn event(&self, icon: &str, text: &str, tone: Tone) {
        let mut g = self.st.lock().unwrap();
        g.events.push_back((hhmm(), icon.into(), text.into(), tone));
        if g.events.len() > 60 {
            g.events.pop_front();
        }
    }
    fn finding(&self, key: &str, f: Finding) {
        let mut g = self.st.lock().unwrap();
        match g.rows.iter_mut().find(|(k, _)| k == key) {
            Some((_, r)) => r.merge(f),
            None => g.rows.push((key.into(), f)),
        }
    }
    fn dim(&self, prefix: &str) {
        let mut g = self.st.lock().unwrap();
        for (k, r) in g.rows.iter_mut() {
            if k.starts_with(prefix) && r.state != Some(RowState::Failed) {
                r.state = Some(RowState::Old);
            }
        }
    }
}

const SPARK: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

fn spark(v: &VecDeque<f64>, lo: f64, hi: f64, width: usize) -> String {
    v.iter()
        .rev()
        .take(width)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(|x| SPARK[(((x - lo) / (hi - lo).max(1.0)) * 7.999).clamp(0.0, 7.0) as usize])
        .collect()
}

fn bar(frac: f64, width: usize) -> String {
    let n = (frac.clamp(0.0, 1.0) * width as f64).round() as usize;
    format!("{}{}", "█".repeat(n), "░".repeat(width - n))
}

fn tone_style(t: Tone) -> Style {
    match t {
        Tone::Normal => Style::default().fg(Color::Gray),
        Tone::Good => Style::default().fg(Color::Green),
        Tone::Bad => Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        Tone::Warn => Style::default().fg(Color::Yellow),
        Tone::Info => Style::default().fg(Color::Cyan),
        Tone::Stage => Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
        Tone::Star => Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD),
    }
}

fn draw(f: &mut ratatui::Frame, g: &St, name: &str, driver: &str, max_w: f64) {
    let [head, mid, low, foot] = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(14),
        Constraint::Min(6),
        Constraint::Length(3),
    ])
    .areas(f.area());
    let [now, gpu] =
        Layout::horizontal([Constraint::Ratio(3, 5), Constraint::Ratio(2, 5)]).areas(mid);
    let [find, log] =
        Layout::horizontal([Constraint::Ratio(3, 5), Constraint::Ratio(2, 5)]).areas(low);
    let dim = Style::default().fg(Color::DarkGray);
    let white = Style::default()
        .fg(Color::White)
        .add_modifier(Modifier::BOLD);

    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                "⚡ GPU TUNE  ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(name, white),
            Span::styled("   driver ", dim),
            Span::raw(driver),
            Span::styled("   running ", dim),
            Span::styled(fmt_time(g.t0.elapsed().as_secs_f64()), white),
        ]))
        .block(Block::bordered().border_style(Style::default().fg(Color::Yellow))),
        head,
    );

    // ---- now
    let mut lines = vec![
        Line::styled(g.stage.clone(), tone_style(Tone::Stage)),
        Line::styled(g.phase.clone(), white),
        Line::styled(
            g.why.clone(),
            Style::default()
                .fg(Color::Gray)
                .add_modifier(Modifier::ITALIC),
        ),
        Line::raw(""),
    ];
    match &g.cur {
        Some(c) => {
            let el = c.start.elapsed().as_secs_f64();
            let spin =
                ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'][(el * 10.0) as usize % 10];
            lines.push(Line::styled(
                format!("{spin} {}", c.desc),
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            ));
            lines.push(Line::from(vec![
                Span::styled(
                    bar(el / c.dur.max(1.0), 40),
                    Style::default().fg(Color::Magenta),
                ),
                Span::styled(format!("  {} / {}", fmt_time(el), fmt_time(c.dur)), dim),
            ]));
        }
        None => lines.push(Line::styled(
            "preparing next test...",
            Style::default().fg(Color::Cyan),
        )),
    }
    f.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: true }).block(
            Block::bordered()
                .title(" Now ")
                .border_style(Style::default().fg(Color::Cyan)),
        ),
        now,
    );

    // ---- live GPU
    let mut gl = vec![];
    match &g.tele {
        None => gl.push(Line::styled("reading sensors...", dim)),
        Some((s, lim)) => {
            let col = |v: f64, a: f64, b: f64| {
                if v < a {
                    Color::Green
                } else if v < b {
                    Color::Yellow
                } else {
                    Color::Red
                }
            };
            let pc = col(s.power, lim * 0.8, lim * 0.97);
            let tc = col(s.temp as f64, 70.0, 82.0);
            gl.push(Line::from(vec![
                Span::styled("Power ", dim),
                Span::styled(bar(s.power / max_w.max(1.0), 16), Style::default().fg(pc)),
                Span::styled(
                    format!(" {:.0} / {:.0} W", s.power, lim),
                    Style::default().fg(pc),
                ),
            ]));
            gl.push(Line::from(vec![
                Span::styled("Temp  ", dim),
                Span::styled(bar(s.temp as f64 / 95.0, 16), Style::default().fg(tc)),
                Span::styled(format!(" {}°C", s.temp), Style::default().fg(tc)),
            ]));
            if let Some(v) = s.vram {
                let vc = col(v, 85.0, 95.0);
                gl.push(Line::from(vec![
                    Span::styled("VRAM  ", dim),
                    Span::styled(bar(v / 110.0, 16), Style::default().fg(vc)),
                    Span::styled(format!(" {v:.0}°C"), Style::default().fg(vc)),
                ]));
            }
            gl.push(Line::from(vec![
                Span::styled("Load  ", dim),
                Span::styled(
                    bar(s.util as f64 / 100.0, 16),
                    Style::default().fg(Color::Cyan),
                ),
                Span::styled(format!(" {}%", s.util), Style::default().fg(Color::Cyan)),
            ]));
            gl.push(Line::from(vec![
                Span::styled("Core ", dim),
                Span::styled(format!("{} MHz", s.clk), white),
                Span::styled("   Mem ", dim),
                Span::styled(format!("{} MHz", s.mclk), white),
                Span::styled(s.pstate.map_or(String::new(), |p| format!("   P{p}")), dim),
            ]));
            gl.push(Line::raw(""));
            let w = (gpu.width as usize).saturating_sub(10).min(40);
            gl.push(Line::from(vec![
                Span::styled("Power ", dim),
                Span::styled(
                    spark(&g.hist[0], 0.0, max_w, w),
                    Style::default().fg(Color::Yellow),
                ),
            ]));
            gl.push(Line::from(vec![
                Span::styled("Temp  ", dim),
                Span::styled(
                    spark(&g.hist[1], 30.0, 95.0, w),
                    Style::default().fg(Color::Red),
                ),
            ]));
            if !g.hist[3].is_empty() {
                gl.push(Line::from(vec![
                    Span::styled("VRAM  ", dim),
                    Span::styled(
                        spark(&g.hist[3], 30.0, 110.0, w),
                        Style::default().fg(Color::Magenta),
                    ),
                ]));
            }
            gl.push(Line::from(vec![
                Span::styled("Clock ", dim),
                Span::styled(
                    spark(&g.hist[2], 200.0, 3200.0, w),
                    Style::default().fg(Color::Cyan),
                ),
            ]));
        }
    }
    f.render_widget(
        Paragraph::new(gl).block(
            Block::bordered()
                .title(" Live GPU ")
                .border_style(Style::default().fg(Color::Green)),
        ),
        gpu,
    );

    // ---- findings
    let rows: Vec<Row> = g
        .rows
        .iter()
        .map(|(k, r)| {
            let st = match r.state {
                Some(RowState::Testing) => Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
                Some(RowState::Failed) => Style::default().fg(Color::Red),
                Some(RowState::Pending) | Some(RowState::Old) => dim,
                Some(RowState::Best) => Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
                _ => Style::default().fg(Color::White),
            };
            let c = |v: &Option<String>| Cell::from(v.clone().unwrap_or_default());
            Row::new(vec![
                Cell::from(r.setting.clone().unwrap_or_else(|| k.clone())),
                c(&r.offset),
                c(&r.limit),
                c(&r.perf),
                c(&r.heavy),
                c(&r.paced),
                c(&r.status),
            ])
            .style(st)
        })
        .collect();
    let widths = [
        Constraint::Length(12),
        Constraint::Length(9),
        Constraint::Length(10),
        Constraint::Length(11),
        Constraint::Length(10),
        Constraint::Length(9),
        Constraint::Length(2),
    ];
    let header = Row::new([
        "Setting",
        "Core",
        "Limit",
        "Game clock",
        "Full load",
        "120 fps",
        "",
    ])
    .style(dim.add_modifier(Modifier::BOLD));
    f.render_widget(
        Table::new(rows, widths).header(header).block(
            Block::bordered()
                .title(" Findings ")
                .border_style(Style::default().fg(Color::Blue)),
        ),
        find,
    );

    // ---- log
    let n = (log.height as usize).saturating_sub(2);
    let ev: Vec<Line> = g
        .events
        .iter()
        .rev()
        .take(n)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(|(t, i, s, tone)| {
            Line::from(vec![
                Span::styled(format!("{t} "), dim),
                Span::styled(format!("{i} "), tone_style(*tone)),
                Span::styled(s.clone(), tone_style(*tone)),
            ])
        })
        .collect();
    f.render_widget(
        Paragraph::new(ev).block(
            Block::bordered()
                .title(" Log ")
                .border_style(Style::default().fg(Color::Magenta)),
        ),
        log,
    );

    // ---- footer
    let el = g.t0.elapsed().as_secs_f64();
    let frac = (el / g.est.max(1.0)).min(0.99);
    let left = g.est - el;
    let txt = format!(
        " {:3.0}%  {}   Ctrl+C or q stops safely",
        frac * 100.0,
        if left > 60.0 {
            format!("~{} left", fmt_time(left))
        } else {
            "almost done".into()
        }
    );
    let bw = (foot.width as usize).saturating_sub(txt.chars().count() + 4);
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(bar(frac, bw), Style::default().fg(Color::Yellow)),
            Span::styled(txt, Style::default().fg(Color::White)),
        ]))
        .block(Block::bordered().border_style(Style::default().fg(Color::Yellow))),
        foot,
    );
    let _ = Rect::default();
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;

    #[test]
    fn dashboard_renders() {
        let mut st = St {
            stage: "Pass 2 of 2 · fine-tuning around max 2790 MHz with full-length tests".into(),
            phase: "Core: highest stable offset at max 2790 MHz".into(),
            why: "Higher offset = lower voltage at the same clock = less power. Searching with game-like bursts.".into(),
            cur: Some(Cur { short_kind: "game".into(), desc: "Game-like load (120 fps) · core +165 · mem +1250 · max 2790".into(),
                            short: "+165 @2790".into(), start: Instant::now(), dur: 30.0 }),
            events: VecDeque::new(),
            rows: vec![],
            tele: Some((Snap { power: 214.0, util: 61, clk: 2790, mclk: 12126, temp: 66, pstate: Some(0), vram: Some(78.0), reasons: 0 }, 320.0)),
            hist: Default::default(),
            t0: Instant::now(),
            est: 3600.0,
        };
        for i in 0..60 {
            st.hist[0].push_back(150.0 + (i % 7) as f64 * 20.0);
            st.hist[1].push_back(55.0 + (i % 5) as f64);
            st.hist[2].push_back(2700.0 + (i % 3) as f64 * 40.0);
            st.hist[3].push_back(70.0 + (i % 4) as f64);
        }
        st.events.push_back((
            "10:21".into(),
            "★".into(),
            "memory offset +1250".into(),
            Tone::Star,
        ));
        st.events.push_back((
            "10:24".into(),
            "✓".into(),
            "game +150 @2790 → 196 W".into(),
            Tone::Good,
        ));
        st.events.push_back((
            "10:25".into(),
            "↻".into(),
            "driver crashed, waiting for the GPU to recover...".into(),
            Tone::Warn,
        ));
        st.rows.push((
            "stock".into(),
            Finding {
                setting: Some("Stock".into()),
                offset: Some("+0".into()),
                perf: Some("2800 MHz".into()),
                heavy: Some("318 W".into()),
                paced: Some("221 W".into()),
                state: Some(RowState::Done),
                ..Default::default()
            },
        ));
        st.rows.push((
            "p2:2790".into(),
            Finding {
                setting: Some("Fine 2790".into()),
                offset: Some("+150".into()),
                limit: Some("+180".into()),
                perf: Some("-0.4%".into()),
                heavy: Some("+0 W".into()),
                paced: Some("-25 W".into()),
                status: Some("✓".into()),
                state: Some(RowState::Done),
            },
        ));
        let mut term = ratatui::Terminal::new(TestBackend::new(140, 40)).unwrap();
        term.draw(|f| draw(f, &st, "NVIDIA GeForce RTX 4080 SUPER", "580.95", 352.0))
            .unwrap();
        let buf = term.backend().buffer().clone();
        let mut out = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                out.push_str(buf[(x, y)].symbol());
            }
            out.push('\n');
        }
        std::fs::write(std::env::temp_dir().join("gpu-tune-dashboard.txt"), &out).unwrap();
        assert!(out.contains("GPU TUNE") && out.contains("Findings") && out.contains("P0"));
    }
}

//! What the tuner reports while it works: the interface, plain-text output and the CSV log.
//! The full-screen dashboard lives in tui.rs.

use std::io::Write;
use std::sync::Mutex;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tone {
    Normal,
    Good,
    Bad,
    Warn,
    Info,
    Stage,
    Star,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RowState {
    Pending,
    Testing,
    Done,
    Failed,
    Old,
    Best,
}

/// One row of the findings table; `None` fields are left as they are.
#[derive(Clone, Debug, Default)]
pub struct Finding {
    pub setting: Option<String>,
    pub offset: Option<String>,
    pub limit: Option<String>,
    pub perf: Option<String>,
    pub heavy: Option<String>,
    pub paced: Option<String>,
    pub status: Option<String>,
    pub state: Option<RowState>,
}

impl Finding {
    pub fn merge(&mut self, o: Finding) {
        macro_rules! m {
            ($($f:ident),*) => { $( if o.$f.is_some() { self.$f = o.$f; } )* };
        }
        m!(setting, offset, limit, perf, heavy, paced, status, state);
    }
}

pub trait Ui: Send + Sync {
    fn set_stage(&self, text: &str);
    fn set_phase(&self, name: &str, why: &str);
    fn test_start(&self, short_kind: &str, desc: &str, dur: f64, short: &str);
    fn test_end(&self, ok: bool, result: &str, brief: &str);
    fn event(&self, icon: &str, text: &str, tone: Tone);
    fn finding(&self, key: &str, f: Finding);
    fn dim(&self, prefix: &str);
}

pub struct PlainUi {
    cur: Mutex<Option<String>>,
}

impl PlainUi {
    pub fn new() -> Self {
        PlainUi {
            cur: Mutex::new(None),
        }
    }
}

impl Ui for PlainUi {
    fn set_stage(&self, text: &str) {
        println!("\n===== {text} =====");
    }
    fn set_phase(&self, name: &str, why: &str) {
        println!("\n{name}");
        if !why.is_empty() {
            println!("  ({why})");
        }
    }
    fn test_start(&self, _: &str, desc: &str, _: f64, _: &str) {
        print!("  {desc} ...");
        let _ = std::io::stdout().flush();
        *self.cur.lock().unwrap() = Some(desc.into());
    }
    fn test_end(&self, _ok: bool, result: &str, _: &str) {
        println!(" {result}");
        *self.cur.lock().unwrap() = None;
    }
    fn event(&self, icon: &str, text: &str, _: Tone) {
        if self.cur.lock().unwrap().take().is_some() {
            println!();
        }
        println!("  {icon} {text}");
    }
    fn finding(&self, _: &str, _: Finding) {}
    fn dim(&self, _: &str) {}
}

/// One CSV row per test, flushed immediately so a hard crash doesn't lose it.
#[derive(Clone, Debug, Default)]
pub struct Row {
    pub time: String,
    pub pass: String,
    pub workload: String,
    pub core: i32,
    pub mem: i32,
    pub max_clock: Option<u32>,
    pub power_limit: u32,
    pub result: String,
    pub core_errors: u64,
    pub mem_errors: u64,
    pub perf: Option<f64>,
    pub bw: Option<f64>,
    pub busy: Option<f64>,
    pub missed: Option<u64>,
    pub frames: Option<u64>,
    pub avg_power: f64,
    pub core_clk: f64,
    pub mem_clk: f64,
    pub pstate: Option<u32>,
    pub temp: u32,
    pub vram_temp: Option<f64>,
    pub limited_by: String,
}

const HEADER: &str = "time,pass,workload,core,mem,max_clock,power_limit,result,core_errors,mem_errors,perf,bw,busy,\
missed,frames,avg_power,core_clk,mem_clk,pstate,temp,vram_temp,limited_by";

pub struct Log {
    f: Mutex<Option<std::fs::File>>,
}

impl Log {
    pub fn new(f: Option<std::fs::File>) -> Self {
        let log = Log { f: Mutex::new(f) };
        log.line(HEADER);
        log
    }
    fn line(&self, s: &str) {
        if let Some(f) = self.f.lock().unwrap().as_mut() {
            let _ = writeln!(f, "{s}");
            let _ = f.flush();
        }
    }
    pub fn write(&self, r: &Row) {
        let o = |v: Option<String>| v.unwrap_or_default();
        let esc = |s: &str| {
            if s.contains(',') || s.contains('"') {
                format!("\"{}\"", s.replace('"', "\"\""))
            } else {
                s.into()
            }
        };
        let cols = [
            r.time.clone(),
            r.pass.clone(),
            r.workload.clone(),
            r.core.to_string(),
            r.mem.to_string(),
            o(r.max_clock.map(|c| c.to_string())),
            r.power_limit.to_string(),
            esc(&r.result),
            r.core_errors.to_string(),
            r.mem_errors.to_string(),
            o(r.perf.map(|v| format!("{v:.3}"))),
            o(r.bw.map(|v| format!("{v:.1}"))),
            o(r.busy.map(|v| format!("{v:.3}"))),
            o(r.missed.map(|v| v.to_string())),
            o(r.frames.map(|v| v.to_string())),
            format!("{:.1}", r.avg_power),
            format!("{:.0}", r.core_clk),
            format!("{:.0}", r.mem_clk),
            o(r.pstate.map(|v| v.to_string())),
            r.temp.to_string(),
            o(r.vram_temp.map(|v| format!("{v:.1}"))),
            r.limited_by.clone(),
        ];
        self.line(&cols.join(","));
    }
}

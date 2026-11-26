//! Running the `cherrycolallm` command and streaming what it prints.
use eframe::egui;
use serde_json::Value;
use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

pub enum Event {
    Chunk(u64, String),
    Done(u64, Option<String>),
    Models(Result<Value, String>),
    Status(Result<Value, String>),
    Recommend(Result<Value, String>),
    PullLine(String),
    PullDone(Option<String>, egui::Context),
    Note(String),
    /// Model settings panel: (what, JSON or plain text / error)
    Data(&'static str, Result<Value, String>),
}

/// Next to this exe if it's there (the installer puts them together), else PATH.
fn cli_path() -> PathBuf {
    let name = if cfg!(windows) { "cherrycolallm.exe" } else { "cherrycolallm" };
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let next = dir.join(name);
            if next.is_file() {
                return next;
            }
        }
    }
    PathBuf::from(name)
}

fn command(args: &[String]) -> Command {
    let mut cmd = Command::new(cli_path());
    cmd.args(args).stdin(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    cmd
}

fn strs(args: &[&str]) -> Vec<String> {
    args.iter().map(|s| s.to_string()).collect()
}

fn run_text(args: &[String]) -> Result<String, String> {
    let out = command(args).output().map_err(|e| format!("could not run cherrycolallm: {e}. Is it installed?"))?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    } else {
        let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
        Err(if err.is_empty() { "cherrycolallm failed".into() } else { err })
    }
}

/// Run a quick command and send its result as `make(Ok(json or text))`.
pub fn run_json_event(tx: &Sender<Event>, ctx: &egui::Context, args: &[&str], make: fn(Result<Value, String>) -> Event) {
    let (tx, ctx, args) = (tx.clone(), ctx.clone(), strs(args));
    std::thread::spawn(move || {
        let r = run_text(&args).map(|t| serde_json::from_str::<Value>(&t).unwrap_or(Value::String(t)));
        let _ = tx.send(make(r));
        ctx.request_repaint();
    });
}

/// Run a command and show its one-line outcome.
pub fn run_note(tx: &Sender<Event>, ctx: &egui::Context, args: &[&str]) {
    let (tx, ctx, args) = (tx.clone(), ctx.clone(), strs(args));
    std::thread::spawn(move || {
        let note = match run_text(&args) {
            Ok(t) => t.lines().next().unwrap_or("").to_string(),
            Err(e) => format!("⚠ {e}"),
        };
        let _ = tx.send(Event::Note(note));
        ctx.request_repaint();
    });
}

/// Kill the answer that's streaming.
pub fn stop(slot: &Arc<Mutex<Option<Child>>>) {
    if let Ok(mut g) = slot.lock() {
        if let Some(c) = g.as_mut() {
            let _ = c.kill();
        }
    }
}

fn wait(slot: &Arc<Mutex<Option<Child>>>) -> bool {
    loop {
        {
            let mut g = slot.lock().unwrap();
            match g.as_mut().map(|c| c.try_wait()) {
                Some(Ok(Some(st))) => {
                    *g = None;
                    return st.success();
                }
                Some(Ok(None)) => {}
                _ => {
                    *g = None;
                    return false;
                }
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(40));
    }
}

#[allow(clippy::too_many_arguments)]
pub fn ask(
    tx: &Sender<Event>,
    ctx: &egui::Context,
    slot: Arc<Mutex<Option<Child>>>,
    id: u64,
    question: String,
    files: Vec<PathBuf>,
    history: Vec<(String, String)>,
    web: bool,
    model: Option<String>,
    preset: String,
) {
    let (tx, ctx) = (tx.clone(), ctx.clone());
    std::thread::spawn(move || {
        let fail = |e: String| {
            let _ = tx.send(Event::Done(id, Some(e)));
            ctx.request_repaint();
        };
        // `history` is everything before this question.
        let past: Vec<Value> = history.iter().map(|(r, t)| serde_json::json!({"role": r, "content": t})).collect();
        let hist_path = std::env::temp_dir().join(format!("cherrycola-history-{id}.json"));
        if std::fs::write(&hist_path, serde_json::to_string(&past).unwrap_or_default()).is_err() {
            return fail("could not write the chat history".into());
        }
        let mut args = strs(&["ask", "--preset", &preset, "--history"]);
        args.push(hist_path.to_string_lossy().into_owned());
        if web {
            args.push("--web".into());
        }
        if let Some(m) = model {
            args.push("-m".into());
            args.push(m);
        }
        for f in &files {
            args.push("--attach".into());
            args.push(f.to_string_lossy().into_owned());
        }
        args.push("--".into());
        args.push(question);
        let mut child = match command(&args).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
            Ok(c) => c,
            Err(e) => return fail(format!("could not run cherrycolallm: {e}. Is it installed?")),
        };
        let mut out = child.stdout.take();
        let mut err_pipe = child.stderr.take();
        *slot.lock().unwrap() = Some(child);
        let err_thread = std::thread::spawn(move || {
            let mut s = String::new();
            if let Some(p) = err_pipe.as_mut() {
                let _ = p.read_to_string(&mut s);
            }
            s
        });
        if let Some(out) = out.as_mut() {
            let mut buf = [0u8; 1024];
            let mut pending: Vec<u8> = vec![];
            loop {
                match out.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        pending.extend_from_slice(&buf[..n]);
                        let keep = match std::str::from_utf8(&pending) {
                            Ok(_) => pending.len(),
                            Err(e) => e.valid_up_to(),
                        };
                        if keep > 0 {
                            let text = String::from_utf8_lossy(&pending[..keep]).into_owned();
                            pending.drain(..keep);
                            let _ = tx.send(Event::Chunk(id, text));
                            ctx.request_repaint();
                        }
                    }
                }
            }
        }
        let stopped_or_ok = wait(&slot);
        let err = err_thread.join().unwrap_or_default().trim().to_string();
        let _ = std::fs::remove_file(&hist_path);
        let _ = tx.send(Event::Done(
            id,
            if stopped_or_ok { None } else if err.is_empty() { Some("stopped".into()) } else { Some(err) },
        ));
        ctx.request_repaint();
    });
}

/// `cherrycolallm pull SPEC`, reporting its \r-separated progress lines.
pub fn pull(tx: &Sender<Event>, ctx: &egui::Context, spec: String) {
    let (tx, ctx) = (tx.clone(), ctx.clone());
    std::thread::spawn(move || {
        let mut child = match command(&strs(&["pull", "--use", "--", &spec])).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
            Ok(c) => c,
            Err(e) => {
                let _ = tx.send(Event::PullDone(Some(format!("could not run cherrycolallm: {e}")), ctx.clone()));
                ctx.request_repaint();
                return;
            }
        };
        let mut readers = vec![];
        for pipe in [child.stdout.take().map(|p| Box::new(p) as Box<dyn Read + Send>), child.stderr.take().map(|p| Box::new(p) as Box<dyn Read + Send>)].into_iter().flatten() {
            let (tx, ctx) = (tx.clone(), ctx.clone());
            readers.push(std::thread::spawn(move || {
                let mut pipe = pipe;
                let (mut buf, mut line, mut last) = ([0u8; 512], Vec::<u8>::new(), String::new());
                while let Ok(n) = pipe.read(&mut buf) {
                    if n == 0 {
                        break;
                    }
                    for &b in &buf[..n] {
                        if b == b'\r' || b == b'\n' {
                            let t = String::from_utf8_lossy(&line).trim().to_string();
                            line.clear();
                            if !t.is_empty() {
                                last = t.clone();
                                let _ = tx.send(Event::PullLine(t));
                                ctx.request_repaint();
                            }
                        } else {
                            line.push(b);
                        }
                    }
                }
                last
            }));
        }
        let ok = child.wait().map(|s| s.success()).unwrap_or(false);
        let lasts: Vec<String> = readers.into_iter().filter_map(|r| r.join().ok()).collect();
        let err = if ok { None } else { Some(lasts.into_iter().rev().find(|l| !l.is_empty()).unwrap_or_else(|| "download failed".into())) };
        let _ = tx.send(Event::PullDone(err, ctx.clone()));
        ctx.request_repaint();
    });
}

/// Run quick commands in order and hand each result to the settings panel.
pub fn load(tx: &Sender<Event>, ctx: &egui::Context, jobs: Vec<(&'static str, Vec<String>)>) {
    let (tx, ctx) = (tx.clone(), ctx.clone());
    std::thread::spawn(move || {
        for (tag, args) in jobs {
            let r = run_text(&args).map(|t| serde_json::from_str::<Value>(&t).unwrap_or(Value::String(t)));
            let _ = tx.send(Event::Data(tag, r));
            ctx.request_repaint();
        }
    });
}

/// Run a command that changes something, report its last line (or its
/// error) as a panel note, then reload what it changed.
pub fn change(tx: &Sender<Event>, ctx: &egui::Context, args: Vec<String>, reload: Vec<(&'static str, Vec<String>)>) {
    change_all(tx, ctx, vec![args], reload)
}

/// Like `change`, for several commands that must run one after another.
/// Stops at the first one that fails.
pub fn change_all(tx: &Sender<Event>, ctx: &egui::Context, cmds: Vec<Vec<String>>, reload: Vec<(&'static str, Vec<String>)>) {
    let (tx, ctx) = (tx.clone(), ctx.clone());
    std::thread::spawn(move || {
        let mut note = Ok(Value::String("Saved.".into()));
        for args in cmds {
            match run_text(&args) {
                Ok(t) => {
                    let lines: Vec<&str> = t.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
                    if !lines.is_empty() {
                        note = Ok(Value::String(lines[lines.len().saturating_sub(2)..].join(" ")));
                    }
                }
                Err(e) => {
                    note = Err(e);
                    break;
                }
            }
        }
        let _ = tx.send(Event::Data("note", note));
        ctx.request_repaint();
        for (tag, args) in reload {
            let r = run_text(&args).map(|t| serde_json::from_str::<Value>(&t).unwrap_or(Value::String(t)));
            let _ = tx.send(Event::Data(tag, r));
            ctx.request_repaint();
        }
    });
}

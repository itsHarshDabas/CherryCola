//! CherryColaLLM: one window for chatting, managing models and settings.
//!
//! The window never talks to an engine itself. Everything goes through the
//! `cherrycolallm` command, so Ollama, llama.cpp and LM Studio behave exactly
//! as they do in a terminal and the command line stays the one stable API.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod cli;
mod store;

use cli::Event;
use eframe::egui::{self, Color32, CornerRadius, RichText, Stroke};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use store::{Chat, Msg, Who};

const CHERRY: Color32 = Color32::from_rgb(0xE0, 0x3E, 0x6E);
const CANDY: Color32 = Color32::from_rgb(0xFF, 0x9F, 0xC0);
const BG: Color32 = Color32::from_rgb(0x1B, 0x14, 0x1A);
const PANEL: Color32 = Color32::from_rgb(0x25, 0x1A, 0x22);
const BUBBLE_ME: Color32 = Color32::from_rgb(0x8E, 0x2B, 0x4E);
const BUBBLE_AI: Color32 = Color32::from_rgb(0x2F, 0x22, 0x2C);
const CODE_BG: Color32 = Color32::from_rgb(0x14, 0x0E, 0x13);
const MUTED: Color32 = Color32::from_rgb(0x9A, 0x8A, 0x95);

#[derive(PartialEq, Clone, Copy)]
enum View {
    Chat,
    Models,
    Settings,
}

struct App {
    view: View,
    chats: Vec<Chat>,
    cur: usize,
    input: String,
    attachments: Vec<PathBuf>,
    models: Vec<String>,
    model: Option<String>,
    preset: String,
    web: bool,
    busy: bool,
    stop: Arc<Mutex<Option<std::process::Child>>>,
    toast: String,
    // models page
    pull_text: String,
    pull_log: String,
    pulling: bool,
    recommend: Option<Value>,
    // settings page
    status: Option<Value>,
    backend: String,
    tx: std::sync::mpsc::Sender<Event>,
    rx: std::sync::mpsc::Receiver<Event>,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        style(&cc.egui_ctx);
        let (tx, rx) = std::sync::mpsc::channel();
        let mut chats = store::load_all();
        if chats.is_empty() {
            chats.push(Chat::new());
        }
        let app = App {
            view: View::Chat,
            chats,
            cur: 0,
            input: String::new(),
            attachments: vec![],
            models: vec![],
            model: None,
            preset: "balanced".into(),
            web: false,
            busy: false,
            stop: Arc::new(Mutex::new(None)),
            toast: "Looking for models…".into(),
            pull_text: String::new(),
            pull_log: String::new(),
            pulling: false,
            recommend: None,
            status: None,
            backend: "auto".into(),
            tx,
            rx,
        };
        app.refresh(&cc.egui_ctx);
        app
    }

    fn refresh(&self, ctx: &egui::Context) {
        cli::run_json_event(&self.tx, ctx, &["models"], Event::Models);
        cli::run_json_event(&self.tx, ctx, &["status", "--json"], Event::Status);
        cli::run_json_event(&self.tx, ctx, &["recommend", "--json"], Event::Recommend);
    }

    fn chat(&mut self) -> &mut Chat {
        let i = self.cur.min(self.chats.len() - 1);
        &mut self.chats[i]
    }

    fn new_chat(&mut self) {
        if self.busy {
            return;
        }
        if self.chats[self.cur].msgs.is_empty() {
            return;
        }
        self.chats.insert(0, Chat::new());
        self.cur = 0;
        self.view = View::Chat;
    }

    fn send(&mut self, ctx: &egui::Context) {
        let q = self.input.trim().to_string();
        if (q.is_empty() && self.attachments.is_empty()) || self.busy {
            return;
        }
        let q = if q.is_empty() { "Describe the attached file(s).".to_string() } else { q };
        let files = std::mem::take(&mut self.attachments);
        let (web, model, preset) = (self.web, self.model.clone(), self.preset.clone());
        let chat = self.chat();
        let history = chat.history();
        chat.msgs.push(Msg {
            who: Who::Me,
            text: q.clone(),
            files: files.iter().map(|p| p.to_string_lossy().into_owned()).collect(),
        });
        chat.msgs.push(Msg { who: Who::Ai, text: String::new(), files: vec![] });
        if chat.title.is_empty() {
            chat.title = q.chars().take(40).collect();
        }
        let id = chat.id;
        self.input.clear();
        self.busy = true;
        self.toast = "Thinking…".into();
        cli::ask(&self.tx, ctx, self.stop.clone(), id, q, files, history, web, model, preset);
    }

    fn regenerate(&mut self, ctx: &egui::Context) {
        if self.busy {
            return;
        }
        let chat = self.chat();
        if chat.msgs.len() < 2 {
            return;
        }
        chat.msgs.pop();
        if let Some(last) = chat.msgs.pop() {
            self.input = last.text;
            self.attachments = last.files.into_iter().map(PathBuf::from).collect();
        }
        self.send(ctx);
    }

    fn add_files(&mut self, files: impl IntoIterator<Item = PathBuf>) {
        for f in files {
            if !self.attachments.contains(&f) {
                self.attachments.push(f);
            }
        }
    }

    fn handle(&mut self, ev: Event) {
        match ev {
            Event::Chunk(id, t) => {
                if let Some(c) = self.chats.iter_mut().find(|c| c.id == id) {
                    if let Some(m) = c.msgs.last_mut() {
                        m.text.push_str(&t);
                    }
                }
            }
            Event::Done(id, err) => {
                self.busy = false;
                self.toast.clear();
                if let Some(c) = self.chats.iter_mut().find(|c| c.id == id) {
                    if let Some(m) = c.msgs.last_mut() {
                        if let Some(e) = err {
                            let sep = if m.text.is_empty() { "" } else { "\n\n" };
                            m.text.push_str(&format!("{sep}⚠ {e}"));
                        }
                    }
                    store::save(c);
                }
            }
            Event::Models(Ok(v)) => {
                let list: Vec<String> = v
                    .as_str()
                    .unwrap_or("")
                    .lines()
                    .map(|l| l.trim().to_string())
                    .filter(|l| !l.is_empty())
                    .collect();
                self.toast = if list.is_empty() { "No models yet: open Models and get one.".into() } else { String::new() };
                if self.model.as_ref().map_or(true, |m| !list.contains(m)) {
                    self.model = list.first().cloned();
                }
                self.models = list;
            }
            Event::Models(Err(e)) => self.toast = e,
            Event::Status(Ok(v)) => {
                if let Some(b) = v.get("configured_backend").and_then(|b| b.as_str()) {
                    self.backend = b.to_string();
                }
                if let Some(m) = v.get("model").and_then(|m| m.as_str()) {
                    if self.model.is_none() {
                        self.model = Some(m.to_string());
                    }
                }
                self.status = Some(v);
            }
            Event::Status(Err(_)) => {}
            Event::Recommend(Ok(v)) => self.recommend = Some(v),
            Event::Recommend(Err(_)) => {}
            Event::PullLine(l) => {
                self.pull_log = l;
            }
            Event::PullDone(err, ctx) => {
                self.pulling = false;
                self.pull_log = match err {
                    None => "Done.".into(),
                    Some(e) => format!("⚠ {e}"),
                };
                self.refresh(&ctx);
            }
            Event::Note(t) => self.toast = t,
        }
    }
}

fn style(ctx: &egui::Context) {
    let mut v = egui::Visuals::dark();
    v.panel_fill = BG;
    v.window_fill = PANEL;
    v.extreme_bg_color = CODE_BG;
    v.selection.bg_fill = CHERRY;
    v.hyperlink_color = CANDY;
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, Color32::from_rgb(0x3A, 0x2A, 0x35));
    for w in [&mut v.widgets.inactive, &mut v.widgets.hovered, &mut v.widgets.active] {
        w.corner_radius = CornerRadius::same(10);
    }
    v.widgets.hovered.bg_fill = Color32::from_rgb(0x4A, 0x2A, 0x3C);
    v.widgets.active.bg_fill = CHERRY;
    ctx.set_visuals(v);
    ctx.style_mut(|s| {
        s.spacing.item_spacing = egui::vec2(10.0, 10.0);
        s.spacing.button_padding = egui::vec2(14.0, 8.0);
    });
}

fn file_icon(path: &str) -> &'static str {
    let ext = path.rsplit('.').next().unwrap_or("").to_lowercase();
    match ext.as_str() {
        "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp" => "🖼",
        "mp3" | "wav" | "m4a" | "flac" | "ogg" | "opus" | "aac" => "🎵",
        "mp4" | "mkv" | "mov" | "webm" | "avi" | "m4v" => "🎬",
        _ => "📄",
    }
}

fn file_name(path: &str) -> String {
    std::path::Path::new(path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| path.into())
}

/// Text with ``` fenced code shown in a monospace box.
fn render_text(ui: &mut egui::Ui, text: &str) {
    for (i, part) in text.split("```").enumerate() {
        if i % 2 == 1 {
            let code = part.split_once('\n').map(|(_, rest)| rest).unwrap_or(part).trim_end();
            egui::Frame::new().fill(CODE_BG).corner_radius(CornerRadius::same(8)).inner_margin(egui::Margin::same(10)).show(ui, |ui| {
                ui.add(egui::Label::new(RichText::new(code).monospace().size(14.0)).selectable(true));
            });
        } else if !part.trim().is_empty() {
            ui.add(egui::Label::new(RichText::new(part.trim()).size(15.5)).selectable(true));
        }
    }
}

impl App {
    fn sidebar(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("side")
            .exact_width(250.0)
            .frame(egui::Frame::new().fill(PANEL).inner_margin(egui::Margin::same(14)))
            .show(ctx, |ui| {
                ui.label(RichText::new("🍒 CherryColaLLM").size(20.0).strong().color(CANDY));
                ui.add_space(4.0);
                if ui.add_sized([ui.available_width(), 34.0], egui::Button::new("＋  New chat").fill(CHERRY)).clicked() {
                    self.new_chat();
                }
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.view, View::Chat, "💬 Chat");
                    ui.selectable_value(&mut self.view, View::Models, "📦 Models");
                    ui.selectable_value(&mut self.view, View::Settings, "⚙");
                });
                ui.separator();
                ui.label(RichText::new("Chats").color(MUTED));
                let mut delete = None;
                let mut pick = None;
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    for (i, c) in self.chats.iter().enumerate() {
                        ui.horizontal(|ui| {
                            let title = if c.title.is_empty() { "New chat" } else { &c.title };
                            if ui.add_sized([ui.available_width() - 48.0, 26.0], egui::SelectableLabel::new(i == self.cur && self.view == View::Chat, title)).clicked() {
                                pick = Some(i);
                            }
                            if ui.small_button("🗑").on_hover_text("Delete this chat").clicked() {
                                delete = Some(i);
                            }
                        });
                    }
                });
                if let Some(i) = pick {
                    if !self.busy {
                        self.cur = i;
                        self.view = View::Chat;
                    }
                }
                if let Some(i) = delete {
                    if !self.busy {
                        store::remove(&self.chats[i]);
                        self.chats.remove(i);
                        if self.chats.is_empty() {
                            self.chats.push(Chat::new());
                        }
                        self.cur = self.cur.min(self.chats.len() - 1);
                    }
                }
            });
    }

    fn chat_view(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("top").frame(egui::Frame::new().fill(BG).inner_margin(egui::Margin::symmetric(14, 10))).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Model").color(MUTED));
                let shown = self.model.clone().unwrap_or_else(|| "none".into());
                egui::ComboBox::from_id_salt("model").selected_text(shown).width(200.0).show_ui(ui, |ui| {
                    for m in self.models.clone() {
                        ui.selectable_value(&mut self.model, Some(m.clone()), m);
                    }
                });
                egui::ComboBox::from_id_salt("preset").selected_text(self.preset.clone()).width(110.0).show_ui(ui, |ui| {
                    for p in ["precise", "balanced", "creative", "code"] {
                        ui.selectable_value(&mut self.preset, p.to_string(), p);
                    }
                }).response.on_hover_text("How adventurous the answers are");
                ui.checkbox(&mut self.web, "🌐 Web").on_hover_text("Search the web first and cite sources");
            });
        });

        egui::TopBottomPanel::bottom("input").frame(egui::Frame::new().fill(BG).inner_margin(egui::Margin::same(14))).show(ctx, |ui| {
            if !self.attachments.is_empty() {
                ui.horizontal_wrapped(|ui| {
                    let mut drop = None;
                    for (i, f) in self.attachments.iter().enumerate() {
                        let s = f.to_string_lossy();
                        if ui.add(egui::Button::new(format!("{} {}  ✕", file_icon(&s), file_name(&s))).fill(BUBBLE_AI)).on_hover_text("Remove").clicked() {
                            drop = Some(i);
                        }
                    }
                    if let Some(i) = drop {
                        self.attachments.remove(i);
                    }
                });
            }
            ui.horizontal(|ui| {
                if ui.add_sized([44.0, 44.0], egui::Button::new(RichText::new("📎").size(18.0))).on_hover_text("Attach images, documents, audio or video (or drop files here)").clicked() {
                    if let Some(files) = rfd::FileDialog::new().set_title("Attach files").pick_files() {
                        self.add_files(files);
                    }
                }
                let w = ui.available_width() - 90.0;
                let edit = ui.add_sized([w, 44.0], egui::TextEdit::multiline(&mut self.input).hint_text("Ask anything…  (Enter sends, Shift+Enter adds a line)").desired_rows(1).margin(egui::Margin::symmetric(12, 12)));
                let enter = edit.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);
                if enter {
                    while self.input.ends_with('\n') {
                        self.input.pop();
                    }
                }
                if self.busy {
                    if ui.add(egui::Button::new(RichText::new("Stop").strong()).fill(BUBBLE_ME).min_size(egui::vec2(76.0, 44.0))).clicked() {
                        cli::stop(&self.stop);
                    }
                } else if ui.add(egui::Button::new(RichText::new("Send").strong()).fill(CHERRY).min_size(egui::vec2(76.0, 44.0))).clicked() || enter {
                    self.send(ctx);
                    edit.request_focus();
                }
            });
            if !self.toast.is_empty() {
                ui.label(RichText::new(&self.toast).small().color(MUTED));
            }
        });

        let mut regen = false;
        egui::CentralPanel::default().show(ctx, |ui| {
            let busy = self.busy;
            let chat = &self.chats[self.cur.min(self.chats.len() - 1)];
            egui::ScrollArea::vertical().stick_to_bottom(true).auto_shrink([false, false]).show(ui, |ui| {
                if chat.msgs.is_empty() {
                    ui.add_space(ui.available_height() * 0.3);
                    ui.vertical_centered(|ui| {
                        ui.label(RichText::new("🍒").size(56.0));
                        ui.label(RichText::new("Ask your own model anything").size(22.0).color(CANDY));
                        ui.label(RichText::new("Runs on this PC. Drop in images, documents, audio or video.").color(MUTED));
                    });
                }
                let last = chat.msgs.len().saturating_sub(1);
                for (i, m) in chat.msgs.iter().enumerate() {
                    let me = m.who == Who::Me;
                    let layout = if me { egui::Layout::right_to_left(egui::Align::TOP) } else { egui::Layout::left_to_right(egui::Align::TOP) };
                    ui.with_layout(layout, |ui| {
                        egui::Frame::new().fill(if me { BUBBLE_ME } else { BUBBLE_AI }).corner_radius(CornerRadius::same(14)).inner_margin(egui::Margin::symmetric(14, 10)).show(ui, |ui| {
                            ui.set_max_width((ui.available_width() * 0.85).max(220.0));
                            ui.vertical(|ui| {
                                for f in &m.files {
                                    ui.label(RichText::new(format!("{} {}", file_icon(f), file_name(f))).small().color(CANDY));
                                }
                                if m.text.is_empty() && busy && i == last {
                                    ui.label("…");
                                } else {
                                    render_text(ui, &m.text);
                                }
                                if !me && !m.text.is_empty() && !(busy && i == last) {
                                    ui.horizontal(|ui| {
                                        if ui.small_button("Copy").clicked() {
                                            ui.ctx().copy_text(m.text.clone());
                                        }
                                        if i == last && ui.small_button("↻ Again").clicked() {
                                            regen = true;
                                        }
                                    });
                                }
                            });
                        });
                    });
                }
            });
        });
        if regen {
            self.regenerate(ctx);
        }
    }

    fn models_view(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().frame(egui::Frame::new().fill(BG).inner_margin(egui::Margin::same(20))).show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading(RichText::new("Models").color(CANDY));
                ui.label(RichText::new("Get a model by name (qwen3:8b), Hugging Face repo (owner/repo:Q4_K_M), a link to a .gguf, or a file on this PC.").color(MUTED));
                ui.horizontal(|ui| {
                    let w = ui.available_width() - 250.0;
                    ui.add_sized([w, 34.0], egui::TextEdit::singleline(&mut self.pull_text).hint_text("Model name, repo or link"));
                    if ui.add_enabled(!self.pulling && !self.pull_text.trim().is_empty(), egui::Button::new("⬇ Get").fill(CHERRY)).clicked() {
                        self.start_pull(ctx, self.pull_text.trim().to_string());
                    }
                    if ui.add_enabled(!self.pulling, egui::Button::new("📁 Browse…")).clicked() {
                        if let Some(f) = rfd::FileDialog::new().add_filter("GGUF model", &["gguf"]).pick_file() {
                            self.start_pull(ctx, f.to_string_lossy().into_owned());
                        }
                    }
                });
                if !self.pull_log.is_empty() {
                    ui.horizontal(|ui| {
                        if self.pulling {
                            ui.spinner();
                        }
                        ui.label(RichText::new(&self.pull_log).color(CANDY));
                    });
                }
                ui.add_space(8.0);
                ui.label(RichText::new("Installed").strong());
                if self.models.is_empty() {
                    ui.label(RichText::new("Nothing yet.").color(MUTED));
                }
                let mut remove = None;
                for m in self.models.clone() {
                    ui.horizontal(|ui| {
                        let current = self.model.as_deref() == Some(m.as_str());
                        ui.label(RichText::new(&m).size(15.0));
                        if current {
                            ui.label(RichText::new("● in use").color(CANDY));
                        } else if ui.small_button("Use").clicked() {
                            self.model = Some(m.clone());
                            cli::run_note(&self.tx, ctx, &["use", &m]);
                        }
                        if ui.small_button("Remove").on_hover_text("Delete this model (Ollama)").clicked() {
                            remove = Some(m.clone());
                        }
                    });
                }
                if let Some(m) = remove {
                    cli::run_note(&self.tx, ctx, &["remove", &m]);
                    self.refresh(ctx);
                }
                if let Some(r) = self.recommend.clone() {
                    ui.add_space(12.0);
                    ui.label(RichText::new("Fits this PC").strong());
                    ui.label(RichText::new(format!(
                        "Runs on the {} ({} GB). Engine: {}.",
                        if r["runs_on"] == "gpu" { "graphics card" } else { "processor" },
                        if r["runs_on"] == "gpu" { &r["vram_gb"] } else { &r["ram_gb"] },
                        r["engine_reason"].as_str().unwrap_or("")
                    )).color(MUTED));
                    let rec = r["recommended"].as_str().unwrap_or("").to_string();
                    for m in r["models"].as_array().cloned().unwrap_or_default() {
                        if m["fits"] != true {
                            continue;
                        }
                        let name = m["model"].as_str().unwrap_or("").to_string();
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(&name).size(15.0));
                            ui.label(RichText::new(format!("{} GB  {}", m["size_gb"], m["note"].as_str().unwrap_or(""))).color(MUTED));
                            if name == rec {
                                ui.label(RichText::new("★ recommended").color(CANDY));
                            }
                            if !self.models.contains(&name) && ui.add_enabled(!self.pulling, egui::Button::new("Get").small()).clicked() {
                                self.start_pull(ctx, name.clone());
                            }
                        });
                    }
                }
            });
        });
    }

    fn start_pull(&mut self, ctx: &egui::Context, spec: String) {
        self.pulling = true;
        self.pull_log = format!("Getting {spec}…");
        cli::pull(&self.tx, ctx, spec);
    }

    fn settings_view(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().frame(egui::Frame::new().fill(BG).inner_margin(egui::Margin::same(20))).show(ctx, |ui| {
            ui.heading(RichText::new("Settings").color(CANDY));
            ui.label(RichText::new("Engine").strong());
            ui.label(RichText::new("The program that runs your models. Auto uses Ollama if it's installed, else LM Studio.").color(MUTED));
            ui.horizontal(|ui| {
                for (key, label) in [("auto", "Auto"), ("ollama", "Ollama"), ("llamacpp", "llama.cpp"), ("lmstudio", "LM Studio")] {
                    if ui.selectable_label(self.backend == key, label).clicked() && self.backend != key {
                        self.backend = key.to_string();
                        cli::run_note(&self.tx, ctx, &["backend", key]);
                        self.refresh(ctx);
                    }
                }
            });
            ui.add_space(8.0);
            if let Some(s) = &self.status {
                let reach = s["reachable"] == true;
                ui.label(format!(
                    "Using {}: {}",
                    s["backend"].as_str().unwrap_or("?"),
                    if reach { "running" } else { "not running (it starts when you ask something)" }
                ));
                ui.label(format!("Default model: {}", s["model"].as_str().unwrap_or("none")));
            }
            if let Some(r) = &self.recommend {
                if r["engine_installed"] != true {
                    ui.add_space(6.0);
                    ui.label(RichText::new("No engine found. Install one:  winget install Ollama.Ollama  (or llama.cpp, or LM Studio)").color(CANDY));
                }
            }
            ui.add_space(12.0);
            if ui.button("↻ Refresh").clicked() {
                self.refresh(ctx);
            }
            ui.add_space(12.0);
            ui.label(RichText::new("Per-model options (context length, GPU layers, every llama.cpp flag) are in the terminal:  cherrycolallm settings --list").color(MUTED));
        });
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        while let Ok(ev) = self.rx.try_recv() {
            self.handle(ev);
        }
        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().filter_map(|f| f.path.clone()).collect());
        if !dropped.is_empty() {
            self.add_files(dropped);
            self.view = View::Chat;
        }
        self.sidebar(ctx);
        match self.view {
            View::Chat => self.chat_view(ctx),
            View::Models => self.models_view(ctx),
            View::Settings => self.settings_view(ctx),
        }
    }
}

fn main() -> eframe::Result {
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_title("CherryColaLLM").with_inner_size([1040.0, 700.0]).with_min_inner_size([640.0, 420.0]).with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native("CherryColaLLM", opts, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}

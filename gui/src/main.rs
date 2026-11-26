//! CherryColaLLM: one window for chatting, managing models and settings.
//!
//! The window never talks to an engine itself. Everything goes through the
//! `cherrycolallm` command, so Ollama, llama.cpp and LM Studio behave exactly
//! as they do in a terminal and the command line stays the one stable API.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod cli;
mod store;
mod theme;

use cli::Event;
use eframe::egui::{self, Align, Color32, CornerRadius, Layout, Margin, RichText, Sense, Stroke, UiBuilder, Vec2};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use store::{Chat, Msg, Who};
use theme::*;

const COLUMN: f32 = 780.0;
const ICON_PNG: &[u8] = include_bytes!("../assets/icon.png");

/// (icon, title, one-line hint, prompt; an empty prompt opens the file picker)
const SUGGESTIONS: [(&str, &str, &str, &str); 4] = [
    ("💡", "Explain something", "How do language models actually work?", "Explain how a transformer language model works, simply."),
    ("✏", "Help me write", "A friendly email to move a meeting", "Write a short, friendly email asking to move a meeting to Friday."),
    ("⌨", "Write code", "Rename photos by the date they were taken", "Write a Python script that renames every photo in a folder by its date."),
    ("📄", "Summarise a file", "Attach a document and get the gist", ""),
];

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
    focus_input: bool,
    icon: Option<egui::TextureHandle>,
    // models page
    pull_text: String,
    pull_log: String,
    pull_frac: Option<f32>,
    pull_spec: String,
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
        theme::apply(&cc.egui_ctx);
        let icon = eframe::icon_data::from_png_bytes(ICON_PNG).ok().map(|d| {
            let img = egui::ColorImage::from_rgba_unmultiplied([d.width as usize, d.height as usize], &d.rgba);
            cc.egui_ctx.load_texture("icon", img, egui::TextureOptions::LINEAR)
        });
        let (tx, rx) = std::sync::mpsc::channel();
        let mut chats = store::load_all();
        chats.insert(0, Chat::new());
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
            toast: String::new(),
            focus_input: true,
            icon,
            pull_text: String::new(),
            pull_log: String::new(),
            pull_frac: None,
            pull_spec: String::new(),
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
        self.view = View::Chat;
        self.focus_input = true;
        if self.busy {
            return;
        }
        if let Some(i) = self.chats.iter().position(|c| c.msgs.is_empty()) {
            self.cur = i;
            return;
        }
        self.chats.insert(0, Chat::new());
        self.cur = 0;
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
        chat.msgs.push(Msg { who: Who::Me, text: q.clone(), files: files.iter().map(|p| p.to_string_lossy().into_owned()).collect() });
        chat.msgs.push(Msg { who: Who::Ai, text: String::new(), files: vec![] });
        if chat.title.is_empty() {
            let first = q.lines().next().unwrap_or("");
            chat.title = if first.chars().count() > 60 { first.chars().take(58).collect::<String>().trim_end().to_string() + "…" } else { first.to_string() };
        }
        let id = chat.id;
        self.input.clear();
        self.busy = true;
        self.toast.clear();
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

    fn pick_files(&mut self) {
        if let Some(files) = rfd::FileDialog::new()
            .set_title("Attach files")
            .add_filter("Everything", &["*"])
            .add_filter("Images", &["png", "jpg", "jpeg", "gif", "webp", "bmp"])
            .add_filter("Documents", &["txt", "md", "pdf", "docx", "pptx", "xlsx", "odt", "csv", "json"])
            .add_filter("Audio", &["mp3", "wav", "m4a", "flac", "ogg", "opus"])
            .add_filter("Video", &["mp4", "mkv", "mov", "webm", "avi"])
            .pick_files()
        {
            self.add_files(files);
        }
    }

    fn handle(&mut self, ev: Event) {
        match ev {
            Event::Chunk(id, t) => {
                if let Some(m) = self.chats.iter_mut().find(|c| c.id == id).and_then(|c| c.msgs.last_mut()) {
                    m.text.push_str(&t);
                }
            }
            Event::Done(id, err) => {
                self.busy = false;
                if let Some(c) = self.chats.iter_mut().find(|c| c.id == id) {
                    if let (Some(m), Some(e)) = (c.msgs.last_mut(), err) {
                        let sep = if m.text.is_empty() { "" } else { "\n\n" };
                        m.text.push_str(&format!("{sep}⚠ {e}"));
                    }
                    store::save(c);
                }
            }
            Event::Models(Ok(v)) => {
                let list: Vec<String> = v.as_str().unwrap_or("").lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect();
                if self.model.as_ref().map_or(true, |m| !list.contains(m)) {
                    let default = self.status.as_ref().and_then(|s| s["model"].as_str()).map(String::from);
                    self.model = default.filter(|d| list.contains(d)).or_else(|| list.first().cloned());
                }
                self.models = list;
            }
            Event::Models(Err(e)) => self.toast = e,
            Event::Status(Ok(v)) => {
                if let Some(b) = v["configured_backend"].as_str() {
                    self.backend = b.to_string();
                }
                if self.model.is_none() {
                    self.model = v["model"].as_str().map(String::from).filter(|m| self.models.contains(m));
                }
                self.status = Some(v);
            }
            Event::Status(Err(_)) | Event::Recommend(Err(_)) => {}
            Event::Recommend(Ok(v)) => self.recommend = Some(v),
            Event::PullLine(l) => {
                self.pull_frac = l
                    .rsplit(' ')
                    .find_map(|w| w.strip_suffix('%'))
                    .and_then(|n| n.trim().parse::<f32>().ok())
                    .map(|p| (p / 100.0).clamp(0.0, 1.0))
                    .or(self.pull_frac);
                self.pull_log = l;
            }
            Event::PullDone(err, ctx) => {
                self.pulling = false;
                self.pull_frac = None;
                self.pull_log = match err {
                    None => "Downloaded and ready to chat.".into(),
                    Some(e) => format!("⚠ {e}"),
                };
                self.refresh(&ctx);
            }
            Event::Note(t) => self.toast = t,
        }
    }
}

// ---- helpers ---------------------------------------------------------------------

fn file_icon(path: &str) -> &'static str {
    match path.rsplit('.').next().unwrap_or("").to_lowercase().as_str() {
        "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp" => "🖼",
        "mp3" | "wav" | "m4a" | "flac" | "ogg" | "opus" | "aac" => "🎵",
        "mp4" | "mkv" | "mov" | "webm" | "avi" | "m4v" => "🎬",
        "pdf" => "📕",
        "py" | "rs" | "js" | "ts" | "c" | "cpp" | "go" | "java" | "sh" => "⌨",
        _ => "📄",
    }
}

fn file_name(path: &str) -> String {
    std::path::Path::new(path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| path.into())
}

/// A centred column at most `max` wide inside the current ui.
fn centered<R>(ui: &mut egui::Ui, max: f32, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    let avail = ui.available_rect_before_wrap();
    let w = avail.width().min(max);
    let left = avail.left() + (avail.width() - w) / 2.0;
    let rect = egui::Rect::from_min_size(egui::pos2(left, avail.top()), Vec2::new(w, avail.height()));
    let r = ui.scope_builder(UiBuilder::new().max_rect(rect).layout(Layout::top_down(Align::Min)), add);
    r.inner
}

/// Plain text with **bold**, `code` and ``` fenced blocks.
fn render_markdown(ui: &mut egui::Ui, text: &str) {
    for (i, part) in text.split("```").enumerate() {
        if i % 2 == 1 {
            let (lang, code) = part.split_once('\n').unwrap_or(("", part));
            let code = code.trim_end().to_string();
            egui::Frame::new().fill(CODE_BG).stroke(Stroke::new(1.0_f32, BORDER)).corner_radius(CornerRadius::same(10)).show(ui, |ui| {
                ui.set_width(ui.available_width());
                egui::Frame::new().fill(SURFACE).corner_radius(CornerRadius { nw: 10, ne: 10, sw: 0, se: 0 }).inner_margin(Margin::symmetric(12, 4)).show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(if lang.trim().is_empty() { "code" } else { lang.trim() }).size(12.0).color(MUTED));
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if ui.add(ghost("Copy")).clicked() {
                                ui.ctx().copy_text(code.clone());
                            }
                        });
                    });
                });
                egui::Frame::new().inner_margin(Margin::symmetric(14, 10)).show(ui, |ui| {
                    egui::ScrollArea::horizontal().id_salt(("code", ui.id(), i)).show(ui, |ui| {
                        ui.add(egui::Label::new(RichText::new(&code).monospace().color(Color32::from_rgb(0xD8, 0xDE, 0xE9))).extend());
                    });
                });
            });
        } else {
            for para in part.trim().split("\n\n") {
                if para.trim().is_empty() {
                    continue;
                }
                let mut job = egui::text::LayoutJob::default();
                let body = egui::FontId::proportional(15.0);
                for (j, seg) in para.split("**").enumerate() {
                    for (k, s) in seg.split('`').enumerate() {
                        let fmt = if k % 2 == 1 {
                            egui::TextFormat { font_id: egui::FontId::monospace(13.5), color: ACCENT_HI, background: SURFACE2, line_height: Some(23.0), valign: Align::Center, ..Default::default() }
                        } else if j % 2 == 1 {
                            egui::TextFormat { font_id: semibold(15.0), color: TEXT, line_height: Some(23.0), valign: Align::Center, ..Default::default() }
                        } else {
                            egui::TextFormat { font_id: body.clone(), color: TEXT, line_height: Some(23.0), valign: Align::Center, ..Default::default() }
                        };
                        job.append(s, 0.0, fmt);
                    }
                }
                job.wrap.max_width = ui.available_width();
                ui.add(egui::Label::new(job).wrap());
                ui.add_space(4.0);
            }
        }
    }
}

// ---- views -----------------------------------------------------------------------

impl App {
    fn sidebar(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("side")
            .exact_width(264.0)
            .resizable(false)
            .frame(egui::Frame::new().fill(SIDEBAR).inner_margin(Margin { left: 12, right: 12, top: 16, bottom: 12 }))
            .show_separator_line(false)
            .show(ctx, |ui| {
                let full = ui.max_rect().expand2(Vec2::new(12.0, 16.0));
                ui.painter().vline(full.right(), full.y_range(), Stroke::new(1.0_f32, BORDER));
                // brand
                ui.horizontal(|ui| {
                    ui.add_space(4.0);
                    if let Some(t) = &self.icon {
                        ui.add(egui::Image::new(t).fit_to_exact_size(Vec2::splat(28.0)).corner_radius(CornerRadius::same(7)));
                    }
                    ui.label(RichText::new("CherryColaLLM").font(semibold(17.0)));
                });
                ui.add_space(14.0);
                if ui.add_sized([ui.available_width(), 38.0], primary("+   New chat")).clicked() {
                    self.new_chat();
                }
                ui.add_space(10.0);
                for (v, icon, label) in [(View::Chat, "💬", "Chat"), (View::Models, "📦", "Models"), (View::Settings, "⚙", "Settings")] {
                    if self.nav_row(ui, icon, label, self.view == v).clicked() {
                        self.view = v;
                    }
                }
                ui.add_space(14.0);
                ui.label(RichText::new("RECENT").size(11.0).color(FAINT).font(semibold(11.0)));
                ui.add_space(2.0);

                let status_h = 46.0;
                let mut pick = None;
                let mut delete = None;
                egui::ScrollArea::vertical().max_height(ui.available_height() - status_h).auto_shrink([false, false]).show(ui, |ui| {
                    let mut any = false;
                    for (i, c) in self.chats.iter().enumerate() {
                        if c.msgs.is_empty() {
                            continue;
                        }
                        any = true;
                        let sel = i == self.cur && self.view == View::Chat;
                        let (rect, resp) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 34.0), Sense::click());
                        let hovered = ui.rect_contains_pointer(rect);
                        if sel || hovered {
                            ui.painter().rect_filled(rect, CornerRadius::same(8), if sel { SURFACE2 } else { SURFACE });
                        }
                        let mut text_rect = rect.shrink2(Vec2::new(10.0, 0.0));
                        if hovered {
                            text_rect.max.x -= 26.0;
                        }
                        ui.scope_builder(UiBuilder::new().max_rect(text_rect).layout(Layout::left_to_right(Align::Center)), |ui| {
                            ui.add(egui::Label::new(RichText::new(&c.title).size(13.5).color(if sel { TEXT } else { MUTED })).truncate().selectable(false));
                        });
                        if hovered {
                            let b = egui::Rect::from_center_size(egui::pos2(rect.right() - 18.0, rect.center().y), Vec2::splat(24.0));
                            let del = ui.put(b, egui::Button::new(RichText::new("🗑").size(12.0).color(MUTED)).fill(Color32::TRANSPARENT).frame(false));
                            if del.on_hover_text("Delete chat").clicked() {
                                delete = Some(i);
                            }
                        }
                        if resp.clicked() {
                            pick = Some(i);
                        }
                    }
                    if !any {
                        ui.label(RichText::new("Your chats show up here.").size(13.0).color(FAINT));
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
                        if self.chats.iter().all(|c| !c.msgs.is_empty()) {
                            self.chats.insert(0, Chat::new());
                        }
                        self.cur = self.cur.min(self.chats.len() - 1);
                    }
                }
                // engine status
                ui.with_layout(Layout::bottom_up(Align::Min), |ui| {
                    let (label, color) = match &self.status {
                        Some(s) if s["reachable"] == true => (format!("{} · ready", engine_name(s["backend"].as_str().unwrap_or(""))), GREEN),
                        Some(s) => (format!("{} · starts on demand", engine_name(s["backend"].as_str().unwrap_or(""))), AMBER),
                        None => ("Checking engine…".into(), FAINT),
                    };
                    egui::Frame::new().fill(SURFACE).corner_radius(CornerRadius::same(10)).inner_margin(Margin::symmetric(10, 8)).show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            dot(ui, color);
                            ui.label(RichText::new(label).size(12.5).color(MUTED));
                        });
                    });
                });
            });
    }

    fn nav_row(&self, ui: &mut egui::Ui, icon: &str, label: &str, selected: bool) -> egui::Response {
        let (rect, resp) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 36.0), Sense::click());
        let hovered = resp.hovered();
        if selected || hovered {
            ui.painter().rect_filled(rect, CornerRadius::same(9), if selected { SURFACE2 } else { SURFACE });
        }
        if selected {
            let bar = egui::Rect::from_min_size(egui::pos2(rect.left(), rect.top() + 9.0), Vec2::new(3.0, 18.0));
            ui.painter().rect_filled(bar, CornerRadius::same(2), ACCENT);
        }
        let color = if selected { TEXT } else { MUTED };
        ui.painter().text(egui::pos2(rect.left() + 16.0, rect.center().y), egui::Align2::LEFT_CENTER, icon, egui::FontId::proportional(14.0), color);
        ui.painter().text(egui::pos2(rect.left() + 42.0, rect.center().y), egui::Align2::LEFT_CENTER, label, if selected { semibold(14.0) } else { egui::FontId::proportional(14.0) }, color);
        resp.on_hover_cursor(egui::CursorIcon::PointingHand)
    }

    fn model_picker(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let shown = self.model.clone().unwrap_or_else(|| "No model".into());
        egui::ComboBox::from_id_salt("model")
            .selected_text(RichText::new(shown).font(semibold(14.0)))
            .width(230.0)
            .height(360.0)
            .show_ui(ui, |ui| {
                ui.set_min_width(230.0);
                for m in self.models.clone() {
                    ui.selectable_value(&mut self.model, Some(m.clone()), m);
                }
                ui.separator();
                if ui.selectable_label(false, "＋ Get more models…").clicked() {
                    self.view = View::Models;
                }
            });
        let _ = ctx;
    }

    fn chat_view(&mut self, ctx: &egui::Context) {
        // header
        egui::TopBottomPanel::top("top")
            .frame(egui::Frame::new().fill(BG).inner_margin(Margin::symmetric(20, 12)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    self.model_picker(ui, ctx);
                    let title = self.chats[self.cur.min(self.chats.len() - 1)].title.clone();
                    if !title.is_empty() {
                        ui.add_space(8.0);
                        ui.label(RichText::new("/").color(FAINT));
                        ui.add_space(4.0);
                        ui.add(egui::Label::new(RichText::new(title).size(13.5).color(MUTED)).truncate());
                    }
                });
            });

        // composer
        egui::TopBottomPanel::bottom("composer")
            .show_separator_line(false)
            .frame(egui::Frame::new().fill(BG).inner_margin(Margin { left: 20, right: 20, top: 6, bottom: 16 }))
            .show(ctx, |ui| {
                centered(ui, COLUMN, |ui| self.composer(ui, ctx));
            });

        // messages
        let mut regen = false;
        let mut suggestion: Option<String> = None;
        let mut attach_hint = false;
        egui::CentralPanel::default().frame(egui::Frame::new().fill(BG)).show(ctx, |ui| {
            let busy = self.busy;
            let icon = self.icon.clone();
            let chat = &self.chats[self.cur.min(self.chats.len() - 1)];
            egui::ScrollArea::vertical().stick_to_bottom(true).auto_shrink([false, false]).show(ui, |ui| {
                ui.add_space(20.0);
                centered(ui, COLUMN, |ui| {
                    ui.set_min_height(10.0);
                    if chat.msgs.is_empty() {
                        Self::empty_state(ui, &icon, &self.models, &mut suggestion, &mut attach_hint);
                        return;
                    }
                    let last = chat.msgs.len() - 1;
                    for (i, m) in chat.msgs.iter().enumerate() {
                        if m.who == Who::Me {
                            ui.with_layout(Layout::right_to_left(Align::Min), |ui| {
                                egui::Frame::new().fill(SURFACE2).corner_radius(CornerRadius { nw: 18, ne: 18, sw: 18, se: 6 }).inner_margin(Margin::symmetric(16, 11)).show(ui, |ui| {
                                    ui.set_max_width(COLUMN * 0.72);
                                    ui.vertical(|ui| {
                                        for f in &m.files {
                                            egui::Frame::new().fill(SURFACE).corner_radius(CornerRadius::same(8)).inner_margin(Margin::symmetric(8, 4)).show(ui, |ui| {
                                                ui.label(RichText::new(format!("{}  {}", file_icon(f), file_name(f))).size(12.5).color(MUTED));
                                            });
                                        }
                                        ui.add(egui::Label::new(RichText::new(&m.text).size(15.0)).wrap());
                                    });
                                });
                            });
                            ui.add_space(14.0);
                        } else {
                            ui.horizontal_top(|ui| {
                                if let Some(t) = &icon {
                                    ui.add(egui::Image::new(t).fit_to_exact_size(Vec2::splat(28.0)).corner_radius(CornerRadius::same(8)));
                                }
                                ui.add_space(4.0);
                                ui.vertical(|ui| {
                                    ui.set_max_width(ui.available_width());
                                    ui.add_space(4.0);
                                    if m.text.is_empty() && busy && i == last {
                                        typing(ui);
                                    } else {
                                        render_markdown(ui, &m.text);
                                    }
                                    if !m.text.is_empty() && !(busy && i == last) {
                                        ui.horizontal(|ui| {
                                            ui.spacing_mut().item_spacing.x = 2.0;
                                            if ui.add(ghost("⎘ Copy")).clicked() {
                                                ui.ctx().copy_text(m.text.clone());
                                            }
                                            if i == last && ui.add(ghost("↻ Regenerate")).clicked() {
                                                regen = true;
                                            }
                                        });
                                    }
                                });
                            });
                            ui.add_space(18.0);
                        }
                    }
                });
            });
        });
        if regen {
            self.regenerate(ctx);
        }
        if attach_hint {
            self.pick_files();
            if !self.attachments.is_empty() && self.input.is_empty() {
                self.input = "Summarise this for me.".into();
            }
            self.focus_input = true;
        }
        if let Some(s) = suggestion {
            self.input = s;
            self.send(ctx);
        }
    }

    fn empty_state(ui: &mut egui::Ui, icon: &Option<egui::TextureHandle>, models: &[String], suggestion: &mut Option<String>, attach: &mut bool) {
        ui.add_space((ui.available_height() * 0.16).max(20.0));
        ui.vertical_centered(|ui| {
            if let Some(t) = icon {
                ui.add(egui::Image::new(t).fit_to_exact_size(Vec2::splat(64.0)).corner_radius(CornerRadius::same(16)));
            }
            ui.add_space(12.0);
            ui.label(RichText::new("What can I help with?").font(semibold(28.0)));
            ui.add_space(2.0);
            let sub = if models.is_empty() { "Get a model from the Models page to start." } else { "Runs on this PC. Drop in images, documents, audio or video." };
            ui.label(RichText::new(sub).size(14.5).color(MUTED));
        });
        ui.add_space(28.0);
        let gap = 12.0;
        let w = (ui.available_width() - gap) / 2.0;
        for row in SUGGESTIONS.chunks(2) {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = gap;
                for (emoji, title, hint, prompt) in row {
                    let (rect, resp) = ui.allocate_exact_size(Vec2::new(w, 72.0), Sense::click());
                    let hovered = resp.hovered();
                    ui.painter().rect(rect, CornerRadius::same(14), if hovered { SURFACE2 } else { SURFACE }, Stroke::new(1.0_f32, if hovered { ACCENT.linear_multiply(0.6) } else { BORDER }), egui::StrokeKind::Inside);
                    ui.painter().text(rect.left_top() + Vec2::new(16.0, 16.0), egui::Align2::LEFT_TOP, format!("{emoji}  {title}"), semibold(14.5), TEXT);
                    ui.painter().text(rect.left_top() + Vec2::new(16.0, 44.0), egui::Align2::LEFT_TOP, *hint, egui::FontId::proportional(13.0), MUTED);
                    if resp.on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                        if prompt.is_empty() {
                            *attach = true;
                        } else {
                            *suggestion = Some(prompt.to_string());
                        }
                    }
                }
            });
            ui.add_space(gap - 8.0);
        }
    }

    fn composer(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let focused_border = ACCENT.linear_multiply(0.7);
        let id = egui::Id::new("composer-input");
        let has_focus = ctx.memory(|m| m.has_focus(id));
        egui::Frame::new()
            .fill(SURFACE)
            .stroke(Stroke::new(1.0_f32, if has_focus { focused_border } else { BORDER }))
            .corner_radius(CornerRadius::same(20))
            .inner_margin(Margin { left: 14, right: 10, top: 10, bottom: 10 })
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                if !self.attachments.is_empty() {
                    ui.horizontal_wrapped(|ui| {
                        let mut drop = None;
                        for (i, f) in self.attachments.iter().enumerate() {
                            let s = f.to_string_lossy();
                            egui::Frame::new().fill(SURFACE2).corner_radius(CornerRadius::same(10)).inner_margin(Margin { left: 10, right: 4, top: 4, bottom: 4 }).show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label(RichText::new(format!("{}  {}", file_icon(&s), file_name(&s))).size(13.0));
                                    if ui.add(egui::Button::new(RichText::new("✕").size(11.0).color(MUTED)).frame(false)).on_hover_text("Remove").clicked() {
                                        drop = Some(i);
                                    }
                                });
                            });
                        }
                        if let Some(i) = drop {
                            self.attachments.remove(i);
                        }
                    });
                    ui.add_space(4.0);
                }
                let hint = if self.models.is_empty() { "Get a model first (Models page)…" } else { "Message CherryColaLLM…" };
                let edit = ui.add(
                    egui::TextEdit::multiline(&mut self.input)
                        .id(id)
                        .hint_text(RichText::new(hint).color(FAINT))
                        .desired_rows(1)
                        .desired_width(f32::INFINITY)
                        .frame(false)
                        .font(egui::FontId::proportional(15.0))
                        .margin(Margin::symmetric(2, 6)),
                );
                if self.focus_input {
                    edit.request_focus();
                    self.focus_input = false;
                }
                let enter = edit.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);
                if enter {
                    while self.input.ends_with('\n') {
                        self.input.pop();
                    }
                }
                ui.add_space(2.0);
                ui.horizontal(|ui| {
                    if ui.add(egui::Button::new(RichText::new("📎").size(15.0)).fill(Color32::TRANSPARENT).stroke(Stroke::new(1.0_f32, BORDER)).corner_radius(CornerRadius::same(15)).min_size(Vec2::new(34.0, 30.0)))
                        .on_hover_text("Attach images, documents, audio or video. You can also drop files on the window.")
                        .clicked()
                    {
                        self.pick_files();
                    }
                    pill_toggle(ui, &mut self.web, "🌐  Web search").on_hover_text("Search the web first and cite sources");
                    egui::ComboBox::from_id_salt("preset")
                        .selected_text(RichText::new(format!("✨  {}", capitalize(&self.preset))).size(13.0).color(MUTED))
                        .width(120.0)
                        .show_ui(ui, |ui| {
                            for (p, d) in [("precise", "Focused, factual"), ("balanced", "Good default"), ("creative", "More varied"), ("code", "Tuned for code")] {
                                ui.selectable_value(&mut self.preset, p.to_string(), format!("{}  ·  {d}", capitalize(p)));
                            }
                        })
                        .response
                        .on_hover_text("How adventurous the answers are");
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        let size = Vec2::splat(34.0);
                        if self.busy {
                            let b = ui.add(egui::Button::new(RichText::new("■").size(13.0).color(TEXT)).fill(SURFACE2).corner_radius(CornerRadius::same(17)).min_size(size)).on_hover_text("Stop");
                            if b.clicked() {
                                cli::stop(&self.stop);
                            }
                        } else {
                            let ready = !self.input.trim().is_empty() || !self.attachments.is_empty();
                            let b = ui.add_enabled(ready, egui::Button::new(RichText::new("↑").size(17.0).color(Color32::WHITE).font(semibold(17.0))).fill(if ready { ACCENT } else { SURFACE2 }).corner_radius(CornerRadius::same(17)).min_size(size)).on_hover_text("Send (Enter)");
                            if b.clicked() || (enter && ready) {
                                self.send(ctx);
                                self.focus_input = true;
                            }
                        }
                    });
                });
            });
        let note = if !self.toast.is_empty() { self.toast.clone() } else { "Answers come from a model on this PC and can be wrong.".into() };
        ui.add_space(6.0);
        ui.vertical_centered(|ui| ui.label(RichText::new(note).size(11.5).color(FAINT)));
    }

    fn page_header(ui: &mut egui::Ui, title: &str, sub: &str) {
        ui.add_space(28.0);
        ui.label(RichText::new(title).font(semibold(26.0)));
        ui.label(RichText::new(sub).size(14.0).color(MUTED));
        ui.add_space(18.0);
    }

    fn models_view(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().frame(egui::Frame::new().fill(BG).inner_margin(Margin::symmetric(24, 0))).show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                centered(ui, COLUMN, |ui| {
                    Self::page_header(ui, "Models", "Download, switch between and remove the models on this PC.");

                    card().show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.label(RichText::new("Get a model").font(semibold(15.0)));
                        ui.label(RichText::new("A name like qwen3:8b, a Hugging Face repo (owner/repo:Q4_K_M), a link to a .gguf, or a file.").size(13.0).color(MUTED));
                        ui.add_space(6.0);
                        ui.horizontal(|ui| {
                            let w = ui.available_width() - 200.0;
                            let edit = ui.add_sized([w, 36.0], egui::TextEdit::singleline(&mut self.pull_text).hint_text(RichText::new("qwen3:8b").color(FAINT)).margin(Margin::symmetric(10, 9)));
                            let go = edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                            if (ui.add_enabled(!self.pulling && !self.pull_text.trim().is_empty(), primary("Download").min_size(Vec2::new(96.0, 36.0))).clicked() || go) && !self.pull_text.trim().is_empty() && !self.pulling {
                                self.start_pull(ctx, self.pull_text.trim().to_string());
                            }
                            if ui.add_enabled(!self.pulling, secondary("File…").min_size(Vec2::new(80.0, 36.0))).on_hover_text("Use a .gguf file on this PC").clicked() {
                                if let Some(f) = rfd::FileDialog::new().add_filter("GGUF model", &["gguf"]).pick_file() {
                                    self.start_pull(ctx, f.to_string_lossy().into_owned());
                                }
                            }
                        });
                        if self.pulling || !self.pull_log.is_empty() {
                            ui.add_space(8.0);
                            if self.pulling {
                                let bar = match self.pull_frac {
                                    Some(f) => egui::ProgressBar::new(f),
                                    None => egui::ProgressBar::new(0.0).animate(true),
                                };
                                ui.add(bar.fill(ACCENT).desired_height(6.0).corner_radius(CornerRadius::same(4)));
                            }
                            let color = if self.pull_log.starts_with('⚠') { AMBER } else if self.pulling { MUTED } else { GREEN };
                            let line = match (self.pulling, self.pull_frac) {
                                (true, Some(f)) => format!("Downloading {}  ·  {:.0}%", self.pull_spec, f * 100.0),
                                (true, None) => format!("Downloading {}  ·  {}", self.pull_spec, self.pull_log),
                                _ => self.pull_log.clone(),
                            };
                            ui.add(egui::Label::new(RichText::new(line).size(12.5).color(color)).truncate());
                        }
                    });

                    ui.add_space(22.0);
                    ui.label(RichText::new(format!("On this PC  ·  {}", self.models.len())).font(semibold(15.0)));
                    ui.add_space(6.0);
                    if self.models.is_empty() {
                        card().show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            ui.label(RichText::new("No models yet. Pick one below, or type a name above.").color(MUTED));
                        });
                    }
                    let mut remove = None;
                    for m in self.models.clone() {
                        card().inner_margin(Margin::symmetric(16, 12)).show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            ui.horizontal(|ui| {
                                if let Some(t) = &self.icon {
                                    ui.add(egui::Image::new(t).fit_to_exact_size(Vec2::splat(22.0)).corner_radius(CornerRadius::same(6)));
                                }
                                ui.label(RichText::new(&m).font(semibold(15.0)));
                                let current = self.model.as_deref() == Some(m.as_str());
                                if current {
                                    badge(ui, "IN USE", GREEN, Color32::from_rgb(0x1D, 0x33, 0x29));
                                }
                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    if ui.add(ghost("Remove")).on_hover_text("Delete this model from disk (Ollama)").clicked() {
                                        remove = Some(m.clone());
                                    }
                                    if !current && ui.add(secondary("Use")).clicked() {
                                        self.model = Some(m.clone());
                                        cli::run_note(&self.tx, ctx, &["use", &m]);
                                    }
                                    if ui.add(secondary("Chat")).clicked() {
                                        self.model = Some(m.clone());
                                        self.new_chat();
                                    }
                                });
                            });
                        });
                        ui.add_space(2.0);
                    }
                    if let Some(m) = remove {
                        cli::run_note(&self.tx, ctx, &["remove", &m]);
                        self.refresh(ctx);
                    }

                    if let Some(r) = self.recommend.clone() {
                        ui.add_space(22.0);
                        ui.label(RichText::new("Recommended for this PC").font(semibold(15.0)));
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            let gpu = r["runs_on"] == "gpu";
                            badge(ui, &format!("RAM {} GB", r["ram_gb"]), MUTED, SURFACE2);
                            if gpu {
                                badge(ui, &format!("GPU {} GB", r["vram_gb"]), MUTED, SURFACE2);
                            }
                            badge(ui, if gpu { "RUNS ON GPU" } else { "RUNS ON CPU" }, if gpu { GREEN } else { AMBER }, SURFACE2);
                        });
                        ui.add_space(8.0);
                        let rec = r["recommended"].as_str().unwrap_or("").to_string();
                        for m in r["models"].as_array().cloned().unwrap_or_default() {
                            if m["fits"] != true {
                                continue;
                            }
                            let name = m["model"].as_str().unwrap_or("").to_string();
                            let have = self.models.contains(&name);
                            card().inner_margin(Margin::symmetric(16, 12)).show(ui, |ui| {
                                ui.set_width(ui.available_width());
                                ui.horizontal(|ui| {
                                    ui.vertical(|ui| {
                                        ui.horizontal(|ui| {
                                            ui.label(RichText::new(&name).font(semibold(15.0)));
                                            if name == rec {
                                                badge(ui, "★ BEST FIT", ACCENT_HI, ACCENT_SOFT);
                                            }
                                            if m["slow"] == true {
                                                badge(ui, "SLOW HERE", AMBER, SURFACE2);
                                            }
                                        });
                                        ui.label(RichText::new(format!("{}  ·  {} GB download", m["note"].as_str().unwrap_or(""), m["size_gb"])).size(13.0).color(MUTED));
                                    });
                                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                        if have {
                                            ui.label(RichText::new("✓ Installed").size(13.0).color(GREEN));
                                        } else if ui.add_enabled(!self.pulling, primary("Download")).clicked() {
                                            self.start_pull(ctx, name.clone());
                                        }
                                    });
                                });
                            });
                            ui.add_space(2.0);
                        }
                    }
                    ui.add_space(30.0);
                });
            });
        });
    }

    fn start_pull(&mut self, ctx: &egui::Context, spec: String) {
        self.pulling = true;
        self.pull_frac = None;
        self.pull_log = "starting…".into();
        self.pull_spec = spec.clone();
        cli::pull(&self.tx, ctx, spec);
    }

    fn settings_view(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().frame(egui::Frame::new().fill(BG).inner_margin(Margin::symmetric(24, 0))).show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                centered(ui, COLUMN, |ui| {
                    Self::page_header(ui, "Settings", "Which engine runs your models, and what the app uses by default.");

                    card().show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.label(RichText::new("Engine").font(semibold(15.0)));
                        ui.label(RichText::new("Auto uses Ollama when it's installed, otherwise LM Studio.").size(13.0).color(MUTED));
                        ui.add_space(8.0);
                        // segmented control
                        egui::Frame::new().fill(BG).corner_radius(CornerRadius::same(10)).inner_margin(Margin::same(3)).show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.spacing_mut().item_spacing.x = 3.0;
                                for (key, label) in [("auto", "Auto"), ("ollama", "Ollama"), ("llamacpp", "llama.cpp"), ("lmstudio", "LM Studio")] {
                                    let on = self.backend == key;
                                    let b = egui::Button::new(RichText::new(label).size(13.5).color(if on { TEXT } else { MUTED }).font(if on { semibold(13.5) } else { egui::FontId::proportional(13.5) }))
                                        .fill(if on { SURFACE2 } else { Color32::TRANSPARENT })
                                        .stroke(Stroke::NONE)
                                        .corner_radius(CornerRadius::same(8))
                                        .min_size(Vec2::new(104.0, 32.0));
                                    if ui.add(b).clicked() && !on {
                                        self.backend = key.to_string();
                                        cli::run_note(&self.tx, ctx, &["backend", key]);
                                        self.refresh(ctx);
                                    }
                                }
                            });
                        });
                        ui.add_space(10.0);
                        if let Some(s) = &self.status {
                            let reach = s["reachable"] == true;
                            ui.horizontal(|ui| {
                                dot(ui, if reach { GREEN } else { AMBER });
                                ui.label(RichText::new(format!(
                                    "{} is {}",
                                    engine_name(s["backend"].as_str().unwrap_or("")),
                                    if reach { "running" } else { "not running yet. It starts when you send a message." }
                                )).size(13.5));
                            });
                        }
                        if let Some(r) = &self.recommend {
                            if r["engine_installed"] != true && self.backend != "lmstudio" {
                                ui.add_space(6.0);
                                egui::Frame::new().fill(ACCENT_SOFT).corner_radius(CornerRadius::same(10)).inner_margin(Margin::symmetric(12, 10)).show(ui, |ui| {
                                    ui.set_width(ui.available_width());
                                    ui.label(RichText::new("No engine installed yet").font(semibold(13.5)).color(ACCENT_HI));
                                    ui.label(RichText::new("Open a terminal and run:  winget install Ollama.Ollama   (or install llama.cpp or LM Studio)").size(13.0));
                                });
                            }
                        }
                    });
                    ui.add_space(12.0);

                    card().show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.label(RichText::new("Default model").font(semibold(15.0)));
                        ui.label(RichText::new("Used for new chats and by other apps that call CherryColaLLM.").size(13.0).color(MUTED));
                        ui.add_space(8.0);
                        let current = self.status.as_ref().and_then(|s| s["model"].as_str()).unwrap_or("none").to_string();
                        let mut pick = None;
                        egui::ComboBox::from_id_salt("default-model").selected_text(&current).width(260.0).show_ui(ui, |ui| {
                            for m in &self.models {
                                if ui.selectable_label(*m == current, m).clicked() {
                                    pick = Some(m.clone());
                                }
                            }
                        });
                        if let Some(m) = pick {
                            self.model = Some(m.clone());
                            cli::run_note(&self.tx, ctx, &["use", &m]);
                            self.refresh(ctx);
                        }
                    });
                    ui.add_space(12.0);

                    card().show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.label(RichText::new("Advanced").font(semibold(15.0)));
                        ui.label(RichText::new("Context length, GPU layers, sampling and every llama.cpp option are set per model from a terminal:").size(13.0).color(MUTED));
                        ui.add_space(6.0);
                        egui::Frame::new().fill(CODE_BG).corner_radius(CornerRadius::same(8)).inner_margin(Margin::symmetric(12, 8)).show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            ui.label(RichText::new("cherrycolallm settings --list").monospace());
                        });
                        ui.add_space(8.0);
                        if ui.add(secondary("↻  Check again")).clicked() {
                            self.refresh(ctx);
                        }
                    });
                    ui.add_space(14.0);
                    ui.label(RichText::new(format!("CherryColaLLM {}", env!("CARGO_PKG_VERSION"))).size(12.0).color(FAINT));
                    ui.add_space(30.0);
                });
            });
        });
    }
}

fn engine_name(key: &str) -> &'static str {
    match key {
        "ollama" => "Ollama",
        "llamacpp" => "llama.cpp",
        "lmstudio" => "LM Studio",
        _ => "Engine",
    }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        while let Ok(ev) = self.rx.try_recv() {
            self.handle(ev);
        }
        let (dropped, hovering) = ctx.input(|i| (i.raw.dropped_files.iter().filter_map(|f| f.path.clone()).collect::<Vec<_>>(), !i.raw.hovered_files.is_empty()));
        if !dropped.is_empty() {
            self.add_files(dropped);
            self.view = View::Chat;
            self.focus_input = true;
        }
        if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::N)) {
            self.new_chat();
        }
        self.sidebar(ctx);
        match self.view {
            View::Chat => self.chat_view(ctx),
            View::Models => self.models_view(ctx),
            View::Settings => self.settings_view(ctx),
        }
        if hovering {
            let screen = ctx.screen_rect();
            let p = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("drop")));
            p.rect_filled(screen, CornerRadius::ZERO, Color32::from_black_alpha(160));
            p.rect_stroke(screen.shrink(16.0), CornerRadius::same(18), Stroke::new(2.0_f32, ACCENT), egui::StrokeKind::Inside);
            p.text(screen.center(), egui::Align2::CENTER_CENTER, "Drop files to attach", semibold(22.0), TEXT);
        }
    }
}

fn main() -> eframe::Result {
    let mut viewport = egui::ViewportBuilder::default()
        .with_title("CherryColaLLM")
        .with_inner_size([1180.0, 760.0])
        .with_min_inner_size([760.0, 480.0])
        .with_drag_and_drop(true);
    if let Ok(icon) = eframe::icon_data::from_png_bytes(ICON_PNG) {
        viewport = viewport.with_icon(icon);
    }
    let opts = eframe::NativeOptions { viewport, ..Default::default() };
    eframe::run_native("CherryColaLLM", opts, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}

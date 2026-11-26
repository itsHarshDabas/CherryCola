//! The model settings panel: the same settings as the Linux app's sidebar.
//!
//! Every setting from `cherrycolallm settings --list --json` gets a row with
//! its explanation on hover. Numbers are sliders that stay in a safe range
//! (context length stops at 4x what the model was trained on). With the
//! llama.cpp engine there are model tools (info, GPU/RAM/disk plan,
//! requantize, download, import from Ollama) and every llama-server option,
//! searchable. Changes are saved through `cherrycolallm set/unset/flag`, for
//! this model or for all models.
use crate::cli::{self, Event};
use crate::theme::*;
use eframe::egui::{self, Align, Color32, CornerRadius, Layout, Margin, RichText, Stroke, Vec2};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::sync::mpsc::Sender;

const GROUPS: [(&str, &str); 5] = [("load", "Loading"), ("gpu", "GPU and memory"), ("rope", "Long context"), ("sampling", "Sampling"), ("other", "Behaviour")];
const ENGINES: [(&str, &str, &str); 4] = [
    ("auto", "Automatic", "Ollama when it's installed, else LM Studio's server."),
    ("ollama", "Ollama", "Easy downloads. Most load settings, but not every llama.cpp option."),
    ("llamacpp", "llama.cpp", "Every setting and llama.cpp option, requantizing and streaming big models from disk. Uses GGUF files."),
    ("lmstudio", "LM Studio", "Uses the model loaded in the LM Studio app. Only sampling settings apply here."),
];
const QUANT_TYPES: [&str; 33] = [
    "Q2_K", "Q2_K_S", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0", "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS",
    "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M", "IQ4_NL", "IQ4_XS", "TQ1_0", "TQ2_0", "MXFP4_MOE", "F16", "BF16", "F32",
];
const SECTIONS: [(&str, &str); 4] = [("common params", "Loading and hardware"), ("sampling params", "Sampling"), ("speculative params", "Speculative decoding"), ("example-specific params", "Server and chat")];

/// How a number setting's slider moves.
enum Slide {
    Steps(Vec<f64>),
    /// lo, hi, step, digits; hi < 0 means: -1 = this PC's CPUs, -2 = the model's layers
    Range(f64, f64, f64, usize),
}

fn slider_for(key: &str) -> Option<Slide> {
    let pow2 = |a: u32, b: u32| (a..b).map(|i| 2f64.powi(i as i32)).collect::<Vec<_>>();
    Some(match key {
        "context_length" => Slide::Steps(pow2(9, 21)),
        "threads" => Slide::Range(1.0, -1.0, 1.0, 0),
        "batch_size" => Slide::Steps(pow2(5, 14)),
        "ubatch_size" => Slide::Steps(pow2(5, 13)),
        "gpu_layers" => Slide::Range(-1.0, -2.0, 1.0, 0),
        "main_gpu" => Slide::Range(0.0, 8.0, 1.0, 0),
        "cpu_moe_layers" => Slide::Range(0.0, -2.0, 1.0, 0),
        "rope_freq_scale" => Slide::Range(0.0, 1.0, 0.05, 2),
        "temperature" => Slide::Range(0.0, 2.0, 0.05, 2),
        "top_k" => Slide::Range(0.0, 200.0, 1.0, 0),
        "top_p" => Slide::Range(0.0, 1.0, 0.01, 2),
        "min_p" => Slide::Range(0.0, 1.0, 0.01, 2),
        "repeat_penalty" => Slide::Range(0.8, 2.0, 0.01, 2),
        "repeat_last_n" => Slide::Range(-1.0, 4096.0, 1.0, 0),
        "presence_penalty" => Slide::Range(-2.0, 2.0, 0.05, 2),
        "frequency_penalty" => Slide::Range(-2.0, 2.0, 0.05, 2),
        "max_tokens" => Slide::Steps(std::iter::once(-1.0).chain(pow2(6, 16)).collect()),
        _ => return None,
    })
}

fn slider_start(key: &str) -> Option<f64> {
    Some(match key {
        "context_length" => 4096.0,
        "batch_size" => 2048.0,
        "ubatch_size" => 512.0,
        "gpu_layers" => -1.0,
        "temperature" => 0.8,
        "top_k" => 40.0,
        "top_p" => 0.95,
        "min_p" => 0.05,
        "repeat_penalty" => 1.0,
        "repeat_last_n" => 64.0,
        "max_tokens" => -1.0,
        "rope_freq_scale" => 0.0,
        _ => return None,
    })
}

fn short(v: f64) -> String {
    if v >= 1024.0 && (v as i64) % 1024 == 0 {
        format!("{}K", v as i64 / 1024)
    } else if v.fract() == 0.0 {
        format!("{}", v as i64)
    } else {
        format!("{v:.2}").trim_end_matches('0').to_string()
    }
}

/// A setting's value as the CLI takes it.
fn to_arg(v: &Value) -> String {
    match v {
        Value::Bool(b) => if *b { "on" } else { "off" }.into(),
        Value::Array(a) => a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join("|"),
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        _ => String::new(),
    }
}

fn shown(v: &Value) -> String {
    match v {
        Value::Bool(b) => if *b { "On" } else { "Off" }.into(),
        Value::Null => String::new(),
        _ => to_arg(v),
    }
}

#[derive(Default)]
pub struct Panel {
    pub open: bool,
    scope_all: bool,
    catalog: Vec<Value>,
    current: Option<Value>,
    options: Option<Result<Vec<Value>, String>>,
    folders: Vec<String>,
    own_dir: String,
    info: String,
    plan: String,
    pub loaded: (String, String, bool),
    text: HashMap<String, String>,
    slide: HashMap<String, f64>,
    search: String,
    quant: String,
    download: String,
    import: String,
    pub note: String,
    note_err: bool,
}

impl Panel {
    fn scope(&self, model: &str) -> Vec<String> {
        if self.scope_all {
            vec!["--all".into()]
        } else {
            vec!["-m".into(), model.into()]
        }
    }

    fn reloads(&self, model: &str) -> Vec<(&'static str, Vec<String>)> {
        vec![("current", vec!["settings".into(), "--json".into(), "--".into(), model.into()])]
    }

    /// Load what the panel shows (again when the model, engine or scope changes).
    pub fn ensure_loaded(&mut self, tx: &Sender<Event>, ctx: &egui::Context, model: &str, backend: &str, force: bool) {
        let key = (model.to_string(), backend.to_string(), self.scope_all);
        if !force && self.loaded == key {
            return;
        }
        let model_changed = self.loaded.0 != key.0 || self.loaded.1 != key.1;
        self.loaded = key;
        self.text.clear();
        self.slide.clear();
        let mut jobs: Vec<(&'static str, Vec<String>)> = vec![];
        if self.catalog.is_empty() {
            jobs.push(("catalog", vec!["settings".into(), "--list".into(), "--json".into()]));
        }
        jobs.extend(self.reloads(model));
        jobs.push(("folders", vec!["folder".into(), "list".into()]));
        if backend == "llamacpp" {
            if self.options.is_none() || force {
                jobs.push(("options", vec!["options".into(), "--json".into()]));
            }
            if model_changed || force {
                self.info.clear();
                self.plan.clear();
                jobs.push(("info", vec!["info".into(), "--".into(), model.into()]));
                jobs.push(("plan", vec!["plan".into(), "--".into(), model.into()]));
            }
        }
        cli::load(tx, ctx, jobs);
    }

    pub fn handle(&mut self, tag: &str, r: Result<Value, String>) {
        match (tag, r) {
            ("catalog", Ok(v)) => self.catalog = v["settings"].as_array().cloned().unwrap_or_default(),
            ("current", Ok(v)) => {
                self.current = Some(v);
                self.text.clear();
                self.slide.clear();
            }
            ("options", Ok(Value::Array(a))) => self.options = Some(Ok(a)),
            ("options", Err(e)) => self.options = Some(Err(e)),
            ("folders", Ok(v)) => {
                let t = v.as_str().unwrap_or("").to_string();
                let mut lines = t.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty());
                self.own_dir = lines.next().map(|l| l.trim_end_matches("(CherryColaLLM's own)").trim().to_string()).unwrap_or_default();
                self.folders = lines.collect();
            }
            ("info", r) => self.info = r.map(|v| v.as_str().map(String::from).unwrap_or_else(|| v.to_string())).unwrap_or_else(|e| e),
            ("plan", r) => self.plan = r.map(|v| v.as_str().map(String::from).unwrap_or_else(|| v.to_string())).unwrap_or_else(|e| e),
            ("note", Ok(v)) => {
                self.note = v.as_str().unwrap_or("Saved.").to_string();
                self.note_err = false;
            }
            ("note", Err(e)) => {
                self.note = e;
                self.note_err = true;
            }
            _ => {}
        }
    }

    fn save(&mut self, tx: &Sender<Event>, ctx: &egui::Context, model: &str, key: &str, value: Option<String>) {
        let mut args = vec![if value.is_some() { "set" } else { "unset" }.to_string()];
        args.extend(self.scope(model));
        args.push("--".into());
        args.push(key.into());
        if let Some(v) = value {
            args.push(v);
        }
        cli::change(tx, ctx, args, self.reloads(model));
    }

    fn save_flag(&mut self, tx: &Sender<Event>, ctx: &egui::Context, model: &str, flag: &str, negated: Option<&str>, value: Option<String>) {
        // Clear the flag (and its --no- twin) first, then set the new value.
        let mine = self.mine();
        let raw = mine.get("llama_args").and_then(|v| v.as_object()).cloned().unwrap_or_default();
        let mut args = vec!["flag".to_string()];
        args.extend(self.scope(model));
        let mut jobs: Vec<Vec<String>> = vec![];
        for f in std::iter::once(flag).chain(negated) {
            if raw.contains_key(f) {
                let mut a = args.clone();
                a.extend(["--remove".into(), f.into()]);
                jobs.push(a);
            }
        }
        if let Some(v) = value {
            let mut a = args.clone();
            a.push(flag.into());
            if v != "on" {
                a.push(v);
            }
            jobs.push(a);
        }
        if !jobs.is_empty() {
            cli::change_all(tx, ctx, jobs, self.reloads(model));
        }
    }

    fn mine(&self) -> Map<String, Value> {
        let c = match &self.current {
            Some(c) => c,
            None => return Map::new(),
        };
        let k = if self.scope_all { "defaults" } else { "model_settings" };
        c[k].as_object().cloned().unwrap_or_default()
    }

    fn effective(&self) -> Map<String, Value> {
        self.current.as_ref().and_then(|c| c["settings"].as_object().cloned()).unwrap_or_default()
    }

    // ---- drawing ---------------------------------------------------------------

    pub fn ui(&mut self, ui: &mut egui::Ui, tx: &Sender<Event>, ctx: &egui::Context, model: &str, backend_cfg: &str, backend: &str, on_engine: &mut Option<String>) {
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label(RichText::new("Model settings").font(semibold(17.0)));
                ui.label(RichText::new(if self.scope_all { "All models" } else { model }).size(12.5).color(MUTED));
            });
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui.add(ghost("×")).on_hover_text("Close (Ctrl+,)").clicked() {
                    self.open = false;
                }
                if ui.add(ghost("↺ Reset")).on_hover_text("Clear every setting here (back to engine defaults)").clicked() {
                    let mut a = vec!["unset".to_string()];
                    a.extend(self.scope(model));
                    cli::change(tx, ctx, a, self.reloads(model));
                }
            });
        });
        ui.add_space(6.0);
        // scope
        egui::Frame::new().fill(BG).corner_radius(CornerRadius::same(10)).inner_margin(Margin::same(3)).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 3.0;
                let w = (ui.available_width() - 3.0) / 2.0;
                for (all, label, tip) in [(false, "This model", "Changes apply to the model picked at the top"), (true, "All models", "Defaults for every model; a model's own settings win")] {
                    let on = self.scope_all == all;
                    let b = egui::Button::new(RichText::new(label).size(13.0).color(if on { TEXT } else { MUTED }))
                        .fill(if on { SURFACE2 } else { Color32::TRANSPARENT })
                        .stroke(Stroke::NONE)
                        .corner_radius(CornerRadius::same(8))
                        .min_size(Vec2::new(w, 28.0));
                    if ui.add(b).on_hover_text(tip).clicked() && !on {
                        self.scope_all = all;
                        self.ensure_loaded(tx, ctx, model, backend, true);
                    }
                }
            });
        });
        ui.add_space(6.0);
        // what the last change did; a fixed footer so rows never jump
        egui::TopBottomPanel::bottom("panel-note")
            .show_separator_line(false)
            .frame(egui::Frame::new().inner_margin(Margin { left: 0, right: 0, top: 8, bottom: 2 }))
            .show_inside(ui, |ui| {
                let (text, color) = if self.note.is_empty() {
                    ("Changes apply the next time the model loads.".to_string(), FAINT)
                } else {
                    (format!("{}  {}", if self.note_err { "⚠" } else { "✔" }, self.note.replace('\n', " ")), if self.note_err { AMBER } else { GREEN })
                };
                ui.add(egui::Label::new(RichText::new(&text).size(12.0).color(color)).truncate()).on_hover_text(&text);
            });

        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            ui.set_width(ui.available_width() - 6.0);
            self.engine_section(ui, tx, ctx, backend_cfg, on_engine);
            if self.current.is_none() || self.catalog.is_empty() {
                ui.add_space(20.0);
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(RichText::new("Reading settings…").color(MUTED));
                });
                return;
            }
            let engine = if backend.is_empty() { "ollama" } else { backend };
            let mine = self.mine();
            let eff = self.effective();
            let limits = self.current.as_ref().map(|c| c["limits"].clone()).unwrap_or(Value::Null);
            let trained = limits["context_length"].as_f64().unwrap_or(0.0);
            let layers = limits["layers"].as_f64().unwrap_or(0.0);
            let experts = limits["experts"].as_i64().unwrap_or(0);
            let ctx_now = eff.get("context_length").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let mut hidden = 0;
            let catalog = self.catalog.clone();
            for (group, title) in GROUPS {
                let rows: Vec<&Value> = catalog
                    .iter()
                    .filter(|s| s["group"] == group && s["key"] != "llama_args")
                    .filter(|s| {
                        let ok = s["engines"].as_array().map_or(false, |e| e.iter().any(|x| x == engine));
                        if !ok {
                            hidden += 1;
                        }
                        ok
                    })
                    .filter(|s| !(s["key"] == "cpu_moe_layers" && layers > 0.0 && experts == 0))
                    .collect();
                if rows.is_empty() {
                    continue;
                }
                ui.add_space(14.0);
                ui.label(RichText::new(title.to_uppercase()).size(11.0).color(FAINT).font(semibold(11.0)));
                ui.add_space(2.0);
                card().inner_margin(Margin::symmetric(12, 8)).show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    for (i, s) in rows.iter().enumerate() {
                        if i > 0 {
                            ui.add_space(2.0);
                            ui.separator();
                        }
                        self.row(ui, tx, ctx, model, s, mine.get(s["key"].as_str().unwrap_or("")), eff.get(s["key"].as_str().unwrap_or("")), trained, layers, ctx_now);
                    }
                });
            }
            if hidden > 0 && engine != "llamacpp" {
                ui.add_space(8.0);
                ui.label(RichText::new(format!("{hidden} more settings work with the llama.cpp engine.")).size(12.5).color(MUTED));
            }
            if engine == "llamacpp" {
                self.tools(ui, tx, ctx, model);
                self.raw_options(ui, tx, ctx, model, &mine);
            }
            ui.add_space(24.0);
        });
    }

    fn engine_section(&mut self, ui: &mut egui::Ui, tx: &Sender<Event>, ctx: &egui::Context, backend_cfg: &str, on_engine: &mut Option<String>) {
        ui.label(RichText::new("ENGINE").size(11.0).color(FAINT).font(semibold(11.0)));
        ui.add_space(2.0);
        card().inner_margin(Margin::symmetric(12, 10)).show(ui, |ui| {
            ui.set_width(ui.available_width());
            let tip = ENGINES.iter().map(|(_, l, h)| format!("{l}: {h}")).collect::<Vec<_>>().join("\n\n");
            ui.horizontal(|ui| {
                ui.label("Engine").on_hover_text(format!("What runs the model.\n\n{tip}"));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let cur = ENGINES.iter().find(|(k, _, _)| *k == backend_cfg).map(|e| e.1).unwrap_or("Automatic");
                    egui::ComboBox::from_id_salt("panel-engine").selected_text(cur).width(150.0).show_ui(ui, |ui| {
                        for (k, l, h) in ENGINES {
                            if ui.selectable_label(k == backend_cfg, l).on_hover_text(h).clicked() && k != backend_cfg {
                                *on_engine = Some(k.to_string());
                            }
                        }
                    });
                });
            });
            ui.separator();
            let n = self.folders.len() + 1;
            egui::CollapsingHeader::new(RichText::new(format!("Model folders  ·  {n} {}", if n == 1 { "folder" } else { "folders" })).size(14.0))
                .id_salt("folders")
                .show(ui, |ui| {
                    ui.label(RichText::new("Folders searched for GGUF models (for example LM Studio's). Models found here run with the llama.cpp engine.").size(12.0).color(MUTED));
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("CherryColaLLM").size(13.0));
                        ui.add(egui::Label::new(RichText::new(&self.own_dir).size(12.0).color(MUTED)).truncate()).on_hover_text("Downloads and requantized models go here");
                    });
                    for f in self.folders.clone() {
                        ui.horizontal(|ui| {
                            if ui.add(ghost("−")).on_hover_text("Stop looking in this folder (nothing is deleted)").clicked() {
                                cli::change(tx, ctx, vec!["folder".into(), "remove".into(), f.clone()], vec![("folders", vec!["folder".into(), "list".into()])]);
                            }
                            ui.add(egui::Label::new(RichText::new(&f).size(12.5)).truncate()).on_hover_text(&f);
                        });
                    }
                    if ui.add(secondary("Add folder…")).clicked() {
                        if let Some(d) = rfd::FileDialog::new().set_title("Add a folder of GGUF models").pick_folder() {
                            cli::change(tx, ctx, vec!["folder".into(), "add".into(), d.to_string_lossy().into_owned()], vec![("folders", vec!["folder".into(), "list".into()])]);
                        }
                    }
                });
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn row(&mut self, ui: &mut egui::Ui, tx: &Sender<Event>, ctx: &egui::Context, model: &str, s: &Value, own: Option<&Value>, eff: Option<&Value>, trained: f64, layers: f64, ctx_now: f64) {
        let key = s["key"].as_str().unwrap_or("").to_string();
        let label = s["label"].as_str().unwrap_or(&key).to_string();
        let kind = s["type"].as_str().unwrap_or("str");
        let mut tip = s["help"].as_str().unwrap_or("").to_string();
        if let (Some(lo), Some(hi)) = (s["min"].as_f64(), s["max"].as_f64()) {
            tip += &format!("\n\nRange: {} to {}.", short(lo), short(hi));
        }
        let only: Vec<&str> = s["engines"].as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect()).unwrap_or_default();
        if only.len() < 3 {
            tip += &format!("\nUsed by: {}.", only.join(", "));
        }

        if kind == "bool" || kind == "choice" {
            let choices: Vec<String> = if kind == "bool" { vec!["On".into(), "Off".into()] } else { s["choices"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default() };
            let current = own.map(shown);
            ui.horizontal(|ui| {
                ui.label(&label).on_hover_text(&tip);
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let text = current.clone().unwrap_or_else(|| match eff.map(shown) {
                        Some(e) if !e.is_empty() => format!("Default ({e})"),
                        _ => "Default".into(),
                    });
                    let mut pick: Option<Option<String>> = None;
                    egui::ComboBox::from_id_salt(("set", &key)).selected_text(RichText::new(text).size(13.0)).width(130.0).show_ui(ui, |ui| {
                        if ui.selectable_label(current.is_none(), "Default").clicked() {
                            pick = Some(None);
                        }
                        for c in &choices {
                            if ui.selectable_label(current.as_deref() == Some(c.as_str()), c).clicked() {
                                pick = Some(Some(c.clone()));
                            }
                        }
                    });
                    if let Some(p) = pick {
                        let v = p.map(|c| if kind == "bool" { if c == "On" { "on".into() } else { "off".into() } } else { c });
                        self.save(tx, ctx, model, &key, v);
                    }
                })
            });
            return;
        }

        if let Some(slide) = slider_for(&key) {
            self.slider_row(ui, tx, ctx, model, &key, &label, &tip, slide, own, eff, trained, layers, ctx_now, kind == "int");
            return;
        }

        if key == "system_prompt" {
            let buf = self.text.entry(key.clone()).or_insert_with(|| own.map(to_arg).unwrap_or_default());
            ui.label(&label).on_hover_text(&tip);
            ui.add(egui::TextEdit::multiline(buf).desired_rows(4).desired_width(f32::INFINITY).hint_text(RichText::new("You are a helpful assistant…").color(FAINT)));
            let text = buf.trim().to_string();
            ui.horizontal(|ui| {
                if ui.add(primary("Save prompt")).clicked() {
                    self.save(tx, ctx, model, &key, if text.is_empty() { None } else { Some(text.clone()) });
                }
                if own.is_some() && ui.add(ghost("Clear")).clicked() {
                    self.save(tx, ctx, model, &key, None);
                }
            });
            return;
        }

        // free text / numbers without a slider
        let hint = match (own, eff) {
            (None, Some(e)) => format!("Default ({})", shown(e)),
            _ => match key.as_str() {
                "stop" => "word1|word2".into(),
                "tensor_split" => "3,1".into(),
                "keep_alive" => "5m".into(),
                _ => "Default".into(),
            },
        };
        let buf = self.text.entry(key.clone()).or_insert_with(|| own.map(to_arg).unwrap_or_default());
        let (mut apply, mut reset) = (false, false);
        ui.horizontal(|ui| {
            ui.label(&label).on_hover_text(&tip);
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if own.is_some() && ui.add(ghost("↺")).on_hover_text("Back to the default").clicked() {
                    reset = true;
                    return;
                }
                // saved on Enter or when leaving the field
                let r = ui.add(egui::TextEdit::singleline(buf).desired_width(150.0).hint_text(RichText::new(hint).color(FAINT)));
                apply = r.lost_focus();
            });
        });
        if reset {
            self.save(tx, ctx, model, &key, None);
        } else if apply {
            let v = self.text.get(&key).cloned().unwrap_or_default().trim().to_string();
            let before = own.map(to_arg).unwrap_or_default();
            if v != before {
                self.save(tx, ctx, model, &key, if v.is_empty() { None } else { Some(v) });
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn slider_row(&mut self, ui: &mut egui::Ui, tx: &Sender<Event>, ctx: &egui::Context, model: &str, key: &str, label: &str, tip: &str, slide: Slide, own: Option<&Value>, eff: Option<&Value>, trained: f64, layers: f64, ctx_now: f64, int: bool) {
        let start = own.and_then(|v| v.as_f64()).or_else(|| eff.and_then(|v| v.as_f64())).or_else(|| slider_start(key));
        let mut caption = String::new();
        let is_set = own.is_some();
        let cpus = std::thread::available_parallelism().map(|n| n.get() as f64).unwrap_or(16.0);
        let layer_max = if layers > 0.0 { layers + 1.0 } else { 200.0 };

        // value shown and the slider position
        let (mut value, steps, range) = match slide {
            Slide::Steps(mut steps) => {
                if key == "context_length" {
                    if trained > 0.0 {
                        steps.retain(|s| *s <= 4.0 * trained);
                        if !steps.contains(&trained) {
                            steps.push(trained);
                        }
                        steps.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        caption = format!("Trained on {}. Past that needs RoPE scaling (yarn).", short(trained));
                    } else {
                        caption = "This model's trained window is unknown.".into();
                    }
                }
                let window = if ctx_now > 0.0 { ctx_now } else { trained };
                if key == "max_tokens" && window > 0.0 {
                    steps.retain(|s| *s <= window);
                    if !steps.contains(&window) {
                        steps.push(window);
                    }
                }
                if (key == "batch_size" || key == "ubatch_size") && window > 0.0 {
                    let first = steps[0];
                    steps.retain(|s| *s <= window);
                    if steps.is_empty() {
                        steps.push(first);
                    }
                }
                let v = start.unwrap_or(steps[0]);
                (v, Some(steps), None)
            }
            Slide::Range(lo, hi, step, digits) => {
                let hi = if hi == -1.0 { cpus } else if hi == -2.0 { if key == "cpu_moe_layers" { layer_max - 1.0 } else { layer_max } } else { hi };
                if hi == layer_max || key == "cpu_moe_layers" {
                    if layers > 0.0 {
                        caption = format!("This model has {} layers.", layers as i64);
                    }
                }
                let v = start.unwrap_or(lo).clamp(lo, hi);
                (v, None, Some((lo, hi, step, digits)))
            }
        };
        if let Some(v) = self.slide.get(key) {
            value = *v;
        }

        let mut text = short(value);
        if key == "gpu_layers" && (value == -1.0 || (layers > 0.0 && value > layers)) {
            text = "All".into();
        }
        if key == "max_tokens" && value == -1.0 {
            text = "No limit".into();
        }
        if key == "context_length" && trained > 0.0 && value > trained {
            text += &format!("  ·  {}x stretched", short(value / trained));
        }
        if !is_set && !self.slide.contains_key(key) && eff.is_none() {
            text = "Default".into();
        }

        ui.horizontal(|ui| {
            ui.label(label).on_hover_text(tip);
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if is_set && ui.add(ghost("↺")).on_hover_text("Back to the default").clicked() {
                    self.slide.remove(key);
                    self.save(tx, ctx, model, key, None);
                }
                ui.label(RichText::new(text).size(13.0).color(if is_set { ACCENT_HI } else { MUTED }));
            });
        });
        let mut commit = None;
        let saved = own.and_then(|v| v.as_f64());
        ui.scope(|ui| {
            ui.spacing_mut().slider_width = ui.available_width();
            ui.visuals_mut().selection.bg_fill = ACCENT;
            ui.visuals_mut().widgets.inactive.bg_fill = SURFACE2;
            if let Some(steps) = &steps {
                let mut idx = steps
                    .iter()
                    .enumerate()
                    .min_by(|a, b| {
                        let d = |x: f64| ((x.abs() + 1.0).log2() - (value.abs() + 1.0).log2()).abs() + if (x < 0.0) != (value < 0.0) { 100.0 } else { 0.0 };
                        d(*a.1).partial_cmp(&d(*b.1)).unwrap()
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let before = idx;
                let r = ui.add(egui::Slider::new(&mut idx, 0..=steps.len() - 1).show_value(false)).on_hover_text(tip);
                if idx != before {
                    self.slide.insert(key.to_string(), steps[idx]);
                }
                if (r.drag_stopped() || (idx != before && !r.dragged())) && Some(steps[idx]) != saved {
                    commit = Some(steps[idx]);
                }
            } else if let Some((lo, hi, step, _)) = range {
                let mut v = value;
                let r = ui.add(egui::Slider::new(&mut v, lo..=hi).step_by(step).show_value(false)).on_hover_text(tip);
                // the slider snaps values to its step by itself; only a real move counts
                let moved = (v - value).abs() > step / 2.0;
                if moved {
                    self.slide.insert(key.to_string(), v);
                }
                let differs = saved.map_or(true, |s| (v - s).abs() > step / 2.0);
                if (r.drag_stopped() || (moved && !r.dragged())) && differs {
                    commit = Some(v);
                }
            }
        });
        if !caption.is_empty() {
            ui.label(RichText::new(caption).size(11.5).color(FAINT));
        }
        if let Some(v) = commit {
            let s = if int { format!("{}", v.round() as i64) } else { format!("{v:.2}") };
            self.save(tx, ctx, model, key, Some(s));
        }
    }

    fn tools(&mut self, ui: &mut egui::Ui, tx: &Sender<Event>, ctx: &egui::Context, model: &str) {
        ui.add_space(14.0);
        ui.label(RichText::new("MODEL TOOLS").size(11.0).color(FAINT).font(semibold(11.0)));
        ui.add_space(2.0);
        card().inner_margin(Margin::symmetric(12, 10)).show(ui, |ui| {
            ui.set_width(ui.available_width());
            if !self.info.is_empty() || !self.plan.is_empty() {
                egui::Frame::new().fill(CODE_BG).corner_radius(CornerRadius::same(8)).inner_margin(Margin::symmetric(10, 8)).show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    for t in [&self.info, &self.plan] {
                        if !t.is_empty() {
                            ui.add(egui::Label::new(RichText::new(t.trim()).monospace().size(12.0).color(MUTED)).wrap());
                        }
                    }
                })
                .response
                .on_hover_text("What this model is, and how it splits between graphics memory, RAM and disk with the current settings. Estimates.");
                ui.add_space(6.0);
            }
            if self.quant.is_empty() {
                self.quant = "Q4_K_M".into();
            }
            ui.horizontal(|ui| {
                ui.label("Requantize to").on_hover_text("Write a copy of this model at another precision. Lower numbers are smaller and faster but less accurate; Q4_K_M is the usual sweet spot, Q8_0 is near lossless. Starting from an already quantized model loses extra quality.");
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui.add(secondary("Quantize")).clicked() {
                        self.note = format!("Quantizing {model} to {}… this can take a while.", self.quant);
                        self.note_err = false;
                        cli::change(tx, ctx, vec!["quantize".into(), "--allow-requantize".into(), "--".into(), model.into(), self.quant.clone()], vec![]);
                    }
                    egui::ComboBox::from_id_salt("quant").selected_text(&self.quant).width(110.0).height(300.0).show_ui(ui, |ui| {
                        for q in QUANT_TYPES {
                            ui.selectable_value(&mut self.quant, q.to_string(), q);
                        }
                    });
                });
            });
            ui.separator();
            let entry = |ui: &mut egui::Ui, buf: &mut String, label: &str, hint: &str, tip: &str| -> Option<String> {
                ui.label(label).on_hover_text(tip);
                let mut out = None;
                ui.horizontal(|ui| {
                    let r = ui.add(egui::TextEdit::singleline(buf).desired_width(ui.available_width() - 70.0).hint_text(RichText::new(hint).color(FAINT)));
                    let enter = r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    if (ui.add(secondary("Go")).clicked() || enter) && !buf.trim().is_empty() {
                        out = Some(buf.trim().to_string());
                    }
                });
                out
            };
            if let Some(spec) = entry(ui, &mut self.download, "Download from Hugging Face", "Qwen/Qwen3-4B-GGUF:Q4_K_M", "Download a GGUF model from Hugging Face. Downloads resume if interrupted.") {
                self.note = format!("Downloading {spec}…");
                self.note_err = false;
                cli::change(tx, ctx, vec!["get".into(), "--".into(), spec], vec![]);
            }
            if let Some(name) = entry(ui, &mut self.import, "Import from Ollama", "qwen3:4b", "Use a model Ollama already downloaded, without copying it.") {
                cli::change(tx, ctx, vec!["import-ollama".into(), "--".into(), name], vec![]);
            }
        });
    }

    fn raw_options(&mut self, ui: &mut egui::Ui, tx: &Sender<Event>, ctx: &egui::Context, model: &str, mine: &Map<String, Value>) {
        ui.add_space(14.0);
        ui.label(RichText::new("ALL LLAMA.CPP OPTIONS").size(11.0).color(FAINT).font(semibold(11.0)));
        ui.label(RichText::new("Every llama-server option, read from the installed version. Hover one to see what it does.").size(12.0).color(MUTED));
        ui.add_space(4.0);
        let opts = match &self.options {
            None => {
                ui.spinner();
                return;
            }
            Some(Err(e)) => {
                ui.label(RichText::new(format!("llama-server not found: {e}")).color(AMBER).size(12.5));
                return;
            }
            Some(Ok(o)) => o.clone(),
        };
        ui.add(egui::TextEdit::singleline(&mut self.search).hint_text(RichText::new("🔎  Search options").color(FAINT)).desired_width(f32::INFINITY).margin(Margin::symmetric(10, 7)));
        let raw = mine.get("llama_args").and_then(|v| v.as_object()).cloned().unwrap_or_default();
        let q = self.search.trim().to_lowercase();
        let mut order: Vec<String> = vec![];
        for o in &opts {
            let s = o["section"].as_str().unwrap_or("other").to_string();
            if !order.contains(&s) {
                order.push(s);
            }
        }
        for section in order {
            let in_sec: Vec<&Value> = opts.iter().filter(|o| o["section"].as_str().unwrap_or("other") == section).collect();
            let visible: Vec<&&Value> = in_sec
                .iter()
                .filter(|o| q.is_empty() || format!("{} {}", o["flag"].as_str().unwrap_or(""), o["help"].as_str().unwrap_or("")).to_lowercase().contains(&q))
                .collect();
            if visible.is_empty() {
                continue;
            }
            let set = in_sec.iter().filter(|o| raw.contains_key(o["flag"].as_str().unwrap_or("")) || o["negated"].as_str().map_or(false, |n| raw.contains_key(n))).count();
            let title = SECTIONS.iter().find(|(k, _)| *k == section).map(|(_, t)| t.to_string()).unwrap_or_else(|| {
                let mut c = section.chars();
                c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
            });
            let n = in_sec.len();
            let head = if set > 0 { format!("{title}  ·  {set} of {n} set") } else { format!("{title}  ·  {n} options") };
            egui::CollapsingHeader::new(RichText::new(head).size(14.0))
                .id_salt(("sec", &section))
                .open(if q.is_empty() { None } else { Some(true) })
                .show(ui, |ui| {
                    for o in visible {
                        self.raw_row(ui, tx, ctx, model, o, &raw);
                        ui.add_space(2.0);
                    }
                });
        }
    }

    fn raw_row(&mut self, ui: &mut egui::Ui, tx: &Sender<Event>, ctx: &egui::Context, model: &str, o: &Value, raw: &Map<String, Value>) {
        let flag = o["flag"].as_str().unwrap_or("").to_string();
        let arg = o["arg"].as_str().unwrap_or("").to_string();
        let negated = o["negated"].as_str().map(String::from).filter(|s| !s.is_empty());
        let help = o["help"].as_str().unwrap_or(&flag).to_string();
        let choices: Vec<String> = o["choices"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default();
        let mut current = raw.get(&flag).cloned();
        if current.is_none() && negated.as_deref().map_or(false, |n| raw.contains_key(n)) {
            current = Some(Value::Bool(false));
        }
        let mono = |t: &str| RichText::new(t.to_string()).monospace().size(12.5);
        if arg.is_empty() || !choices.is_empty() {
            let opts: Vec<String> = if choices.is_empty() { vec!["On".into(), "Off".into()] } else { choices.clone() };
            let cur_text = current.as_ref().map(|v| match v {
                Value::Bool(b) => if *b { "On".to_string() } else { "Off".to_string() },
                Value::String(s) if choices.is_empty() && s == "on" => "On".into(),
                other => to_arg(other),
            });
            let is_set = cur_text.is_some();
            ui.horizontal(|ui| {
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let mut pick: Option<Option<String>> = None;
                    let shown = RichText::new(cur_text.clone().unwrap_or_else(|| "Default".into())).size(12.5).color(if is_set { ACCENT_HI } else { TEXT });
                    egui::ComboBox::from_id_salt(("raw", &flag)).selected_text(shown).width(104.0).show_ui(ui, |ui| {
                        if ui.selectable_label(cur_text.is_none(), "Default").clicked() {
                            pick = Some(None);
                        }
                        for c in &opts {
                            if ui.selectable_label(cur_text.as_deref() == Some(c.as_str()), c).clicked() {
                                pick = Some(Some(c.clone()));
                            }
                        }
                    });
                    ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                        ui.add(egui::Label::new(mono(&flag).color(if is_set { TEXT } else { MUTED })).truncate()).on_hover_text(&help);
                    });
                    if let Some(p) = pick {
                        let v = match p {
                            None => None,
                            Some(c) if choices.is_empty() => {
                                if c == "On" {
                                    Some("on".to_string())
                                } else if let Some(n) = &negated {
                                    // "Off" for a flag with a --no- twin: set the twin
                                    self.save_flag(tx, ctx, model, n, Some(&flag), Some("on".into()));
                                    return;
                                } else {
                                    None
                                }
                            }
                            Some(c) => Some(c),
                        };
                        self.save_flag(tx, ctx, model, &flag, negated.as_deref(), v);
                    }
                });
            });
        } else {
            let k = format!("raw:{flag}");
            let buf = self.text.entry(k.clone()).or_insert_with(|| current.as_ref().map(to_arg).unwrap_or_default());
            let is_set = current.is_some();
            let (mut apply, mut reset) = (false, false);
            ui.horizontal(|ui| {
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if is_set && ui.add(ghost("↺")).on_hover_text("Back to the default").clicked() {
                        reset = true;
                    }
                    let r = ui.add(
                        egui::TextEdit::singleline(buf)
                            .desired_width(if is_set { 96.0 } else { 120.0 })
                            .text_color(if is_set { ACCENT_HI } else { TEXT })
                            .hint_text(RichText::new(if arg.is_empty() { "Default".to_string() } else { arg.clone() }).color(FAINT)),
                    );
                    apply = r.lost_focus();
                    ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                        ui.add(egui::Label::new(mono(&flag).color(if is_set { TEXT } else { MUTED })).truncate()).on_hover_text(format!("{flag} {arg}\n\n{help}"));
                    });
                });
            });
            if reset {
                self.text.remove(&k);
                self.save_flag(tx, ctx, model, &flag, negated.as_deref(), None);
                return;
            }
            if apply {
                let v = self.text.get(&k).cloned().unwrap_or_default().trim().to_string();
                let before = current.as_ref().map(to_arg).unwrap_or_default();
                if v != before {
                    self.save_flag(tx, ctx, model, &flag, negated.as_deref(), if v.is_empty() { None } else { Some(v) });
                }
            }
        }
    }
}

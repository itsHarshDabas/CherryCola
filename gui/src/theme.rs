//! Colours, fonts and the few custom widgets the window is built from.
use eframe::egui::{self, Color32, CornerRadius, FontData, FontDefinitions, FontFamily, FontId, Margin, RichText, Sense, Stroke, TextStyle, Vec2};
use std::sync::Arc;

pub const BG: Color32 = Color32::from_rgb(0x12, 0x14, 0x18);
pub const SIDEBAR: Color32 = Color32::from_rgb(0x17, 0x1A, 0x20);
pub const SURFACE: Color32 = Color32::from_rgb(0x1E, 0x22, 0x2A);
pub const SURFACE2: Color32 = Color32::from_rgb(0x27, 0x2C, 0x36);
pub const HOVER: Color32 = Color32::from_rgb(0x2C, 0x32, 0x3D);
pub const BORDER: Color32 = Color32::from_rgb(0x2F, 0x35, 0x41);
pub const TEXT: Color32 = Color32::from_rgb(0xEC, 0xEF, 0xF4);
pub const MUTED: Color32 = Color32::from_rgb(0x8E, 0x96, 0xA6);
pub const FAINT: Color32 = Color32::from_rgb(0x5E, 0x66, 0x75);
pub const ACCENT: Color32 = Color32::from_rgb(0xE8, 0x56, 0x6C);
pub const ACCENT_HI: Color32 = Color32::from_rgb(0xF0, 0x70, 0x84);
pub const ACCENT_SOFT: Color32 = Color32::from_rgb(0x3A, 0x22, 0x2A);
pub const GREEN: Color32 = Color32::from_rgb(0x6F, 0xD5, 0x9A);
pub const AMBER: Color32 = Color32::from_rgb(0xF2, 0xB8, 0x5A);
pub const CODE_BG: Color32 = Color32::from_rgb(0x0D, 0x0F, 0x12);

pub const SEMIBOLD: &str = "semibold";

pub fn semibold(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name(SEMIBOLD.into()))
}

/// The system UI font where we can find it (Segoe UI on Windows, Inter or
/// Noto/DejaVu on Linux), egui's built-in fonts for emoji and anything missing.
fn first_font(paths: &[&str]) -> Option<Vec<u8>> {
    paths.iter().find_map(|p| std::fs::read(p).ok())
}

pub fn install_fonts(ctx: &egui::Context) {
    let windir = std::env::var("WINDIR").unwrap_or_else(|_| "C:\\Windows".into());
    let w = |f: &str| format!("{windir}\\Fonts\\{f}");
    let (r1, r2, b1, b2) = (w("segoeui.ttf"), w("SegUIVar.ttf"), w("seguisb.ttf"), w("segoeuib.ttf"));
    let regular = first_font(&[
        &r1, &r2,
        "/usr/share/fonts/opentype/inter/Inter-Regular.otf",
        "/usr/share/fonts/inter/Inter-Regular.otf",
        "/usr/share/fonts/TTF/Inter-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "/System/Library/Fonts/SFNS.ttf",
    ]);
    let bold = first_font(&[
        &b1, &b2,
        "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf",
        "/usr/share/fonts/inter/Inter-SemiBold.otf",
        "/usr/share/fonts/TTF/Inter-SemiBold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-SemiBold.ttf",
        "/usr/share/fonts/noto/NotoSans-SemiBold.ttf",
    ]);
    let mono = first_font(&[&w("CascadiaMono.ttf"), &w("consola.ttf"), "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf"]);

    let mut fonts = FontDefinitions::default();
    let fallback: Vec<String> = fonts.families[&FontFamily::Proportional].clone();
    if let Some(r) = regular {
        fonts.font_data.insert("ui".into(), Arc::new(FontData::from_owned(r)));
        fonts.families.get_mut(&FontFamily::Proportional).unwrap().insert(0, "ui".into());
    }
    let mut semi = vec![];
    if let Some(b) = bold {
        fonts.font_data.insert("ui-bold".into(), Arc::new(FontData::from_owned(b)));
        semi.push("ui-bold".to_string());
    }
    semi.extend(fonts.families[&FontFamily::Proportional].clone());
    semi.extend(fallback);
    fonts.families.insert(FontFamily::Name(SEMIBOLD.into()), semi);
    if let Some(m) = mono {
        fonts.font_data.insert("mono".into(), Arc::new(FontData::from_owned(m)));
        fonts.families.get_mut(&FontFamily::Monospace).unwrap().insert(0, "mono".into());
    }
    ctx.set_fonts(fonts);
}

pub fn apply(ctx: &egui::Context) {
    install_fonts(ctx);
    let mut v = egui::Visuals::dark();
    v.override_text_color = Some(TEXT);
    v.panel_fill = BG;
    v.window_fill = SURFACE;
    v.window_stroke = Stroke::new(1.0_f32, BORDER);
    v.window_corner_radius = CornerRadius::same(12);
    v.menu_corner_radius = CornerRadius::same(10);
    v.popup_shadow = egui::epaint::Shadow { offset: [0, 6], blur: 24, spread: 0, color: Color32::from_black_alpha(120) };
    v.extreme_bg_color = BG; // text fields
    v.faint_bg_color = SURFACE;
    v.selection.bg_fill = ACCENT.linear_multiply(0.55);
    v.selection.stroke = Stroke::new(1.0_f32, ACCENT_HI);
    v.hyperlink_color = ACCENT_HI;
    v.text_cursor.stroke = Stroke::new(2.0_f32, ACCENT_HI);
    let r = CornerRadius::same(10);
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, BORDER);
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, TEXT);
    v.widgets.inactive.bg_fill = SURFACE2;
    v.widgets.inactive.weak_bg_fill = SURFACE2;
    v.widgets.inactive.bg_stroke = Stroke::NONE;
    v.widgets.inactive.fg_stroke = Stroke::new(1.0_f32, TEXT);
    v.widgets.hovered.bg_fill = HOVER;
    v.widgets.hovered.weak_bg_fill = HOVER;
    v.widgets.hovered.bg_stroke = Stroke::new(1.0_f32, BORDER);
    v.widgets.hovered.fg_stroke = Stroke::new(1.0_f32, TEXT);
    v.widgets.active.bg_fill = SURFACE2;
    v.widgets.active.weak_bg_fill = SURFACE2;
    v.widgets.active.bg_stroke = Stroke::new(1.0_f32, ACCENT);
    v.widgets.active.fg_stroke = Stroke::new(1.0_f32, TEXT);
    v.widgets.open.weak_bg_fill = SURFACE2;
    for w in [&mut v.widgets.noninteractive, &mut v.widgets.inactive, &mut v.widgets.hovered, &mut v.widgets.active, &mut v.widgets.open] {
        w.corner_radius = r;
        w.expansion = 0.0;
    }
    ctx.set_visuals(v);
    ctx.style_mut(|s| {
        s.spacing.item_spacing = Vec2::new(8.0, 8.0);
        s.spacing.button_padding = Vec2::new(12.0, 7.0);
        s.spacing.interact_size.y = 30.0;
        s.spacing.combo_width = 160.0;
        s.spacing.scroll = egui::style::ScrollStyle::thin();
        s.text_styles.insert(TextStyle::Body, FontId::proportional(14.5));
        s.text_styles.insert(TextStyle::Button, FontId::proportional(14.0));
        s.text_styles.insert(TextStyle::Small, FontId::proportional(12.0));
        s.text_styles.insert(TextStyle::Heading, semibold(22.0));
        s.text_styles.insert(TextStyle::Monospace, FontId::monospace(13.5));
        s.interaction.selectable_labels = true;
    });
}

// ---- widgets -----------------------------------------------------------------

pub fn primary(text: &str) -> egui::Button<'static> {
    egui::Button::new(RichText::new(text.to_string()).font(semibold(14.0)).color(Color32::WHITE))
        .fill(ACCENT)
        .corner_radius(CornerRadius::same(10))
}

pub fn secondary(text: &str) -> egui::Button<'static> {
    egui::Button::new(RichText::new(text.to_string()).color(TEXT))
        .fill(SURFACE2)
        .stroke(Stroke::new(1.0_f32, BORDER))
        .corner_radius(CornerRadius::same(10))
}

pub fn ghost(text: &str) -> egui::Button<'static> {
    egui::Button::new(RichText::new(text.to_string()).size(13.0).color(MUTED))
        .fill(Color32::TRANSPARENT)
        .stroke(Stroke::NONE)
        .corner_radius(CornerRadius::same(8))
}

/// A rounded, outlined card.
pub fn card() -> egui::Frame {
    egui::Frame::new()
        .fill(SURFACE)
        .stroke(Stroke::new(1.0_f32, BORDER))
        .corner_radius(CornerRadius::same(14))
        .inner_margin(Margin::same(16))
}

/// A small coloured label like "IN USE" or "5.2 GB".
pub fn badge(ui: &mut egui::Ui, text: &str, fg: Color32, bg: Color32) {
    egui::Frame::new()
        .fill(bg)
        .corner_radius(CornerRadius::same(6))
        .inner_margin(Margin::symmetric(7, 2))
        .show(ui, |ui| {
            ui.label(RichText::new(text).size(11.5).color(fg).font(semibold(11.5)));
        });
}

/// A pill toggle (web search on/off).
pub fn pill_toggle(ui: &mut egui::Ui, on: &mut bool, text: &str) -> egui::Response {
    let (fg, bg, stroke) = if *on { (ACCENT_HI, ACCENT_SOFT, ACCENT.linear_multiply(0.6)) } else { (MUTED, Color32::TRANSPARENT, BORDER) };
    let r = ui.add(
        egui::Button::new(RichText::new(text).size(13.0).color(fg))
            .fill(bg)
            .stroke(Stroke::new(1.0_f32, stroke))
            .corner_radius(CornerRadius::same(15))
            .min_size(Vec2::new(0.0, 30.0)),
    );
    if r.clicked() {
        *on = !*on;
    }
    r
}

/// A status dot.
pub fn dot(ui: &mut egui::Ui, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(10.0), Sense::hover());
    ui.painter().circle_filled(rect.center(), 4.0, color);
}

/// Three dots that pulse while a model is thinking.
pub fn typing(ui: &mut egui::Ui) {
    let t = ui.input(|i| i.time) as f32;
    let (rect, _) = ui.allocate_exact_size(Vec2::new(36.0, 20.0), Sense::hover());
    for k in 0..3 {
        let phase = (t * 4.0 - k as f32 * 0.7).sin() * 0.5 + 0.5;
        let c = MUTED.linear_multiply(0.35 + 0.65 * phase);
        ui.painter().circle_filled(egui::pos2(rect.left() + 6.0 + k as f32 * 11.0, rect.center().y), 3.5, c);
    }
    ui.ctx().request_repaint();
}

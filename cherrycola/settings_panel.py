"""The model settings side panel of the CherryCola chat app.

Every setting in profiles.SETTINGS gets a row; hovering a row explains it.
Settings are saved per model (or for all models) as soon as they change,
and load settings apply the next time the model loads. With the llama.cpp
engine, every llama-server option is listed too (read from its --help),
along with model tools: download, import from Ollama, requantize and the
GPU / RAM / disk plan.
"""

from __future__ import annotations

import subprocess
import threading

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GLib, Gtk  # noqa: E402

from . import config as config_mod  # noqa: E402
from . import gguf, llamacpp, profiles  # noqa: E402

ENGINES = [("auto", "Automatic"), ("ollama", "Ollama"),
           ("llamacpp", "llama.cpp"), ("lmstudio", "LM Studio")]
ENGINE_HELP = {
    "auto": "Ollama when it's installed, else LM Studio's server.",
    "ollama": "Easy downloads with cherrycola pull. Most load settings, "
              "but not every llama.cpp option.",
    "llamacpp": "Every setting and llama.cpp option, requantizing and "
                "streaming big models from disk. Uses GGUF files.",
    "lmstudio": "Uses the model loaded in the LM Studio app. Only sampling "
                "settings apply here; loading is set in LM Studio.",
}
DEFAULT = "Default"
SECTIONS = {"common params": "Loading and hardware",
            "sampling params": "Sampling",
            "speculative params": "Speculative decoding",
            "example-specific params": "Server and chat"}


def _fmt(value) -> str:
    if isinstance(value, list):
        return "|".join(value)
    if isinstance(value, float):
        return f"{value:g}"
    return "" if value is None else str(value)


class SettingsPanel(Gtk.Box):
    """Lives in the window's right sidebar."""

    def __init__(self, window):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.win = window
        self.model = ""
        self.backend = ""
        self.scope_all = False
        self._building = False

        header = Adw.HeaderBar(show_start_title_buttons=False,
                               show_end_title_buttons=False)
        self.title = Adw.WindowTitle(title="Model settings")
        header.set_title_widget(self.title)
        reset = Gtk.Button(icon_name="edit-undo-symbolic",
                           tooltip_text="Clear every setting for this model "
                                        "(back to engine defaults)")
        reset.connect("clicked", self._on_reset)
        header.pack_end(reset)
        self.append(header)

        self.scope_box = Gtk.Box(css_classes=["linked"], halign=Gtk.Align.CENTER,
                                 margin_bottom=6)
        self.btn_model = Gtk.ToggleButton(label="This model", active=True,
                                          tooltip_text="Changes apply to the "
                                          "model picked in the header")
        self.btn_all = Gtk.ToggleButton(label="All models", group=self.btn_model,
                                        tooltip_text="Defaults for every "
                                        "model; a model's own settings win")
        self.btn_model.connect("toggled", self._on_scope)
        self.scope_box.append(self.btn_model)
        self.scope_box.append(self.btn_all)
        self.append(self.scope_box)

        self.page = Adw.PreferencesPage()
        self.append(self.page)
        self.set_vexpand(True)
        self.set_size_request(340, -1)

    # ---- building -------------------------------------------------------------

    def show_for(self, backend: str, model: str):
        self.backend, self.model = backend, model
        self.title.set_subtitle(model if not self.scope_all else "All models")
        self.rebuild()

    def _target(self):
        return None if self.scope_all else self.model

    def rebuild(self):
        self._building = True
        self.remove(self.page)
        self.page = Adw.PreferencesPage()
        self.append(self.page)
        stored = profiles.load_profiles()
        mine = stored.get(self._target() or profiles.GLOBAL, {})
        eff = profiles.effective(self.model, profiles=stored)

        self.page.add(self._engine_group())
        groups: dict[str, Adw.PreferencesGroup] = {}
        engine = self.backend or "ollama"
        hidden = 0
        for key, spec in profiles.SETTINGS.items():
            if key == "llama_args":
                continue
            if engine not in spec.engines:
                hidden += 1
                continue
            group = groups.get(spec.group)
            if group is None:
                group = Adw.PreferencesGroup(title=profiles.GROUPS[spec.group])
                groups[spec.group] = group
                self.page.add(group)
            group.add(self._row(spec, mine.get(key), eff.get(key)))
        if hidden:
            note = Adw.PreferencesGroup(description=(
                f"{hidden} more settings work with the llama.cpp engine."
                if engine != "llamacpp" else ""))
            self.page.add(note)
        if engine == "llamacpp":
            self.page.add(self._tools_group())
            self._raw_group(mine.get("llama_args", {}))
        self._building = False

    def _engine_group(self):
        group = Adw.PreferencesGroup()
        names = Gtk.StringList.new([label for _, label in ENGINES])
        row = Adw.ComboRow(use_markup=False, title="Engine", model=names)
        cfg = config_mod.load_config()
        keys = [k for k, _ in ENGINES]
        row.set_selected(keys.index(cfg.backend) if cfg.backend in keys else 0)
        row.set_tooltip_text(
            "What runs the model.\n\n" + "\n".join(
                f"{label}: {ENGINE_HELP[k]}" for k, label in ENGINES))
        row.connect("notify::selected", self._on_engine, keys)
        group.add(row)
        return group

    def _row(self, spec: profiles.Setting, own, effective):
        tip = spec.help
        if spec.kind in ("int", "float") and spec.lo is not None:
            lo = int(spec.lo) if spec.kind == "int" else spec.lo
            hi = int(spec.hi) if spec.kind == "int" else spec.hi
            tip += f"\n\nRange: {lo:g} to {hi:g}." if spec.kind == "float" \
                else f"\n\nRange: {lo} to {hi}."
        if spec.kind in ("bool", "choice"):
            choices = ["On", "Off"] if spec.kind == "bool" else \
                list(spec.choices)
            row = Adw.ComboRow(use_markup=False, title=spec.label,
                               model=Gtk.StringList.new([DEFAULT] + choices))
            if own is not None:
                value = ("On" if own else "Off") if spec.kind == "bool" \
                    else own
                row.set_selected(choices.index(value) + 1)
            row.connect("notify::selected", self._on_combo, spec, choices)
        elif spec.key == "system_prompt":
            row = Adw.ExpanderRow(use_markup=False, title=spec.label,
                                  subtitle="Custom" if own else "Default")
            view = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR,
                                top_margin=8, bottom_margin=8, left_margin=8,
                                right_margin=8, height_request=110)
            view.get_buffer().set_text(own or "")
            save = Gtk.Button(label="Save", halign=Gtk.Align.END,
                              margin_top=6, margin_bottom=6, margin_end=6,
                              css_classes=["suggested-action", "pill"])
            save.connect("clicked", self._on_prompt, view, row)
            box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
            box.append(Gtk.Frame(child=view, margin_start=6, margin_end=6,
                                 margin_top=6))
            box.append(save)
            row.add_row(box)
        else:
            row = Adw.EntryRow(use_markup=False, title=spec.label, show_apply_button=True)
            row.set_text(_fmt(own))
            if own is None and effective is not None:
                # A preset's value: show it greyed, as a hint.
                row.set_title(f"{spec.label}  ·  {_fmt(effective)}")
            if spec.kind in ("int", "float"):
                row.set_input_purpose(Gtk.InputPurpose.NUMBER)
            row.connect("apply", self._on_entry, spec)
        row.set_tooltip_text(tip)
        return row

    def _tools_group(self):
        group = Adw.PreferencesGroup(title="Model tools")
        path = llamacpp.find_model(config_mod.load_config(), self.model)
        if path is not None:
            try:
                info = gguf.read_info(path)
                eff = profiles.effective(self.model)
                plan = llamacpp.make_plan(info, eff)
                size = (f"{info.parameters / 1e9:.1f}B"
                        if info.parameters >= 1e9
                        else f"{info.parameters / 1e6:.0f}M")
                about = Adw.ActionRow(
                    use_markup=False,
                    title=f"{info.architecture} · {info.file_type} · {size}",
                    subtitle=plan.describe(bool(eff.get("disk_stream"))),
                    subtitle_selectable=True)
                about.set_tooltip_text(
                    "How this model splits between graphics memory, RAM and "
                    "disk with the current settings. Estimates.")
                group.add(about)
            except gguf.GGUFError as exc:
                group.add(Adw.ActionRow(use_markup=False, title="Can't read this model",
                                        subtitle=str(exc)))
            types = list(llamacpp.QUANT_TYPES)
            q = Adw.ComboRow(use_markup=False, title="Requantize to",
                             model=Gtk.StringList.new(types))
            q.set_selected(types.index("Q4_K_M"))
            q.set_tooltip_text(
                "Write a copy of this model at another precision. Lower "
                "numbers are smaller and faster but less accurate; Q4_K_M "
                "is the usual sweet spot, Q8_0 is near lossless. Starting "
                "from an already quantized model loses extra quality.")
            go = Gtk.Button(label="Quantize", valign=Gtk.Align.CENTER,
                            css_classes=["flat"])
            go.connect("clicked", self._on_quantize, q, types)
            q.add_suffix(go)
            group.add(q)
        get = Adw.EntryRow(use_markup=False, title="Download (owner/repo:Q4_K_M)",
                           show_apply_button=True)
        get.set_tooltip_text(
            "Download a GGUF model from Hugging Face, for example "
            "Qwen/Qwen3-4B-GGUF:Q4_K_M. Downloads resume if interrupted.")
        get.connect("apply", self._on_download)
        group.add(get)
        imp = Adw.EntryRow(use_markup=False, title="Import from Ollama (qwen3:4b)",
                           show_apply_button=True)
        imp.set_tooltip_text(
            "Use a model Ollama already downloaded, without copying it.")
        imp.connect("apply", self._on_import)
        group.add(imp)
        return group

    def _raw_group(self, raw: dict):
        group = Adw.PreferencesGroup(
            title="All llama.cpp options",
            description="Every llama-server option, read from the installed "
                        "version. Hover one to see what it does.")
        search = Gtk.SearchEntry(placeholder_text="Search options")
        group.add(search)
        self.page.add(group)
        group = Adw.PreferencesGroup()  # the option sections, under search
        self.page.add(group)
        try:
            opts = llamacpp.server_options(
                config_mod.load_config().llama_server)
        except llamacpp.LlamaError as exc:
            group.add(Adw.ActionRow(use_markup=False, title="llama-server not found",
                                    subtitle=str(exc)))
            return group
        sections: dict[str, Adw.ExpanderRow] = {}
        rows: list[tuple[Gtk.Widget, Adw.ExpanderRow, str]] = []
        for opt in opts:
            exp = sections.get(opt.section)
            if exp is None:
                exp = Adw.ExpanderRow(use_markup=False, title=SECTIONS.get(
                    opt.section, opt.section.capitalize()))
                sections[opt.section] = exp
                group.add(exp)
            row = self._raw_row(opt, raw)
            exp.add_row(row)
            rows.append((row, exp, f"{opt.flag} {opt.help}".lower()))
        for name, exp in sections.items():
            count = sum(1 for o in opts if o.section == name and (
                o.flag in raw or (o.negated and o.negated in raw)))
            if count:
                exp.set_subtitle(f"{count} set")

        def on_search(entry):
            q = entry.get_text().strip().lower()
            for row, exp, text in rows:
                row.set_visible(not q or q in text)
            for exp in sections.values():
                exp.set_expanded(bool(q))
        search.connect("search-changed", on_search)
        return group

    def _raw_row(self, opt: llamacpp.ServerOption, raw: dict):
        current = raw.get(opt.flag)
        if current is None and opt.negated in raw:
            current = False
        title = opt.flag
        if not opt.arg or opt.choices:
            choices = list(opt.choices) or ["On", "Off"]
            row = Adw.ComboRow(use_markup=False, title=title,
                               model=Gtk.StringList.new([DEFAULT] + choices))
            if current is not None:
                value = current if opt.choices else (
                    "On" if current in (True, "on") else "Off")
                if value in choices:
                    row.set_selected(choices.index(value) + 1)
            row.connect("notify::selected", self._on_raw_combo, opt, choices)
        else:
            # EntryRow parses its title as markup whatever use_markup says.
            row = Adw.EntryRow(title=GLib.markup_escape_text(
                f"{title} {opt.arg}"),
                               show_apply_button=True)
            row.set_text("" if current is None else str(current))
            row.connect("apply", self._on_raw_entry, opt)
        row.set_tooltip_text(opt.help or opt.flag)
        return row

    # ---- saving ------------------------------------------------------------------

    def _save(self, key, value):
        try:
            if value is None:
                profiles.unset_value(self._target(), key)
            else:
                profiles.set_value(self._target(), key, value)
        except profiles.SettingError as exc:
            self.win.toast(str(exc))
            return False
        spec = profiles.SETTINGS.get(key)
        if spec and spec.group in ("load", "gpu", "rope") or \
                key == "llama_args":
            self.win.toast("Saved. Applies the next time the model loads.")
        return True

    def _on_scope(self, *_):
        self.scope_all = self.btn_all.get_active()
        self.show_for(self.backend, self.model)

    def _on_engine(self, row, _pspec, keys):
        if self._building:
            return
        engine = keys[row.get_selected()]
        config_mod.save_setting("backend", engine)
        self.win.cfg = config_mod.load_config()
        self.win.probe()

    def _on_combo(self, row, _pspec, spec, choices):
        if self._building:
            return
        i = row.get_selected()
        value = None if i == 0 else choices[i - 1]
        if spec.kind == "bool" and value is not None:
            value = value == "On"
        self._save(spec.key, value)
        if spec.key in ("preset", "disk_stream"):
            GLib.idle_add(self.rebuild)

    def _on_entry(self, row, spec):
        text = row.get_text().strip()
        if self._save(spec.key, text or None) and spec.key == "context_length":
            GLib.idle_add(self.rebuild)

    def _on_prompt(self, _btn, view, row):
        buf = view.get_buffer()
        text = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)
        if self._save("system_prompt", text.strip() or None):
            row.set_subtitle("Custom" if text.strip() else "Default")
            self.win.toast("System prompt saved.")

    def _raw_set(self, opt, value):
        stored = profiles.load_profiles()
        raw = dict(stored.get(self._target() or profiles.GLOBAL, {})
                   .get("llama_args", {}))
        raw.pop(opt.flag, None)
        if opt.negated:
            raw.pop(opt.negated, None)
        if value is not None:
            try:
                llamacpp.raw_args({opt.flag: value}, [opt])
            except llamacpp.LlamaError as exc:
                self.win.toast(str(exc))
                return
            raw[opt.flag] = value
        self._save("llama_args", raw or None)

    def _on_raw_combo(self, row, _pspec, opt, choices):
        if self._building:
            return
        i = row.get_selected()
        if i == 0:
            self._raw_set(opt, None)
        elif opt.choices:
            self._raw_set(opt, choices[i - 1])
        else:
            self._raw_set(opt, choices[i - 1] == "On")

    def _on_raw_entry(self, row, opt):
        text = row.get_text().strip()
        self._raw_set(opt, text or None)

    def _on_reset(self, *_):
        profiles.unset_value(self._target(), None)
        self.win.toast("Settings cleared.")
        self.rebuild()

    # ---- model tools -------------------------------------------------------------

    def _background(self, work, done_text):
        def run():
            try:
                result = work()
            except (llamacpp.LlamaError, gguf.GGUFError, OSError) as exc:
                GLib.idle_add(self.win.toast, str(exc))
                return
            GLib.idle_add(self.win.toast, done_text.format(result))
            GLib.idle_add(self.win.probe)
        threading.Thread(target=run, daemon=True).start()

    def _on_quantize(self, _btn, row, types):
        qtype = types[row.get_selected()]
        cfg = config_mod.load_config()
        path = llamacpp.find_model(cfg, self.model)
        allow = False
        if path is not None:
            try:
                allow = gguf.read_info(path).quantized
            except gguf.GGUFError:
                pass
        self.win.toast(f"Quantizing {self.model} to {qtype}…")
        self._background(lambda: llamacpp.quantize(
            cfg, self.model, qtype, allow_requantize=allow,
            run=lambda a: subprocess.run(a, capture_output=True)),
            "Wrote {}. Pick it in the model menu.")

    def _on_download(self, row):
        spec = row.get_text().strip()
        if not spec:
            return
        cfg = config_mod.load_config()
        self.win.toast(f"Downloading {spec}…")

        def progress(name, done, total):
            if total:
                GLib.idle_add(self.title.set_subtitle,
                              f"{name}: {100 * done // total}%")
        self._background(lambda: llamacpp.download(cfg, spec, progress),
                         "Downloaded {}.")

    def _on_import(self, row):
        name = row.get_text().strip()
        if name:
            cfg = config_mod.load_config()
            self._background(lambda: llamacpp.import_ollama(cfg, name),
                             "Imported as {}.")

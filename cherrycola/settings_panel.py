"""The model settings side panel of the CherryColaLLM chat app.

Every setting in profiles.SETTINGS gets a row; hovering a row explains it.
Settings are saved per model (or for all models) as soon as they change,
and load settings apply the next time the model loads. With the llama.cpp
engine, every llama-server option is listed too (read from its --help),
along with model tools: download, import from Ollama, requantize and the
GPU / RAM / disk plan.
"""

from __future__ import annotations

import math
import os
import subprocess
import threading

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GLib, Gtk  # noqa: E402

from . import config as config_mod  # noqa: E402
from . import backends, gguf, llamacpp, profiles  # noqa: E402

ENGINES = [("auto", "Automatic"), ("ollama", "Ollama"),
           ("llamacpp", "llama.cpp"), ("lmstudio", "LM Studio")]
ENGINE_HELP = {
    "auto": "Ollama when it's installed, else LM Studio's server.",
    "ollama": "Easy downloads with cherrycolallm pull. Most load settings, "
              "but not every llama.cpp option.",
    "llamacpp": "Every setting and llama.cpp option, requantizing and "
                "streaming big models from disk. Uses GGUF files.",
    "lmstudio": "Uses the model loaded in the LM Studio app. Only sampling "
                "settings apply here; loading is set in LM Studio.",
}
DEFAULT = "Default"

# Sliders for numbers, so a value can't be out of range.
#   ("steps", [values]): snaps to these (powers of two and the like)
#   ("range", lo, hi, step, digits)
# "layers" and "cpus" are filled in from the model and this PC.
_POW2 = [2 ** i for i in range(5, 14)]
SLIDERS = {
    "context_length": ("steps", [2 ** i for i in range(9, 21)]),
    "threads": ("range", 1, "cpus", 1, 0),
    "batch_size": ("steps", _POW2),
    "ubatch_size": ("steps", _POW2[:8]),
    "gpu_layers": ("range", -1, "layers", 1, 0),
    "main_gpu": ("range", 0, 8, 1, 0),
    "cpu_moe_layers": ("range", 0, "layers", 1, 0),
    "rope_freq_scale": ("range", 0, 1, 0.05, 2),
    "temperature": ("range", 0, 2, 0.05, 2),
    "top_k": ("range", 0, 200, 1, 0),
    "top_p": ("range", 0, 1, 0.01, 2),
    "min_p": ("range", 0, 1, 0.01, 2),
    "repeat_penalty": ("range", 0.8, 2, 0.01, 2),
    "repeat_last_n": ("range", -1, 4096, 1, 0),
    "presence_penalty": ("range", -2, 2, 0.05, 2),
    "frequency_penalty": ("range", -2, 2, 0.05, 2),
    "max_tokens": ("steps", [-1] + [2 ** i for i in range(6, 16)]),
}
# Where a slider starts when nothing is set (the usual engine default).
SLIDER_START = {"context_length": 4096, "batch_size": 2048,
                "ubatch_size": 512, "gpu_layers": -1, "temperature": 0.8,
                "top_k": 40, "top_p": 0.95, "min_p": 0.05,
                "repeat_penalty": 1.0, "repeat_last_n": 64,
                "max_tokens": -1, "rope_freq_scale": 0}


def _short(v: float) -> str:
    if v >= 1024 and v % 1024 == 0:
        return f"{int(v) // 1024}K"
    return f"{v:g}"
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
        self._layers, self._trained_ctx = 0, 0
        self._experts, self._ctx_now = -1, 0
        self._limit_cache: dict[tuple[str, str], object] = {}

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

    def _limits(self):
        """The model's own numbers (trained context, layers, experts),
        asked once per model and engine."""
        key = (self.backend or "ollama", self.model)
        if key not in self._limit_cache:
            self._limit_cache[key] = backends.model_limits(
                key[0], config_mod.load_config(), self.model, timeout=2)
        return self._limit_cache[key]

    def rebuild(self):
        self._building = True
        self.remove(self.page)
        self.page = Adw.PreferencesPage()
        self.append(self.page)
        stored = profiles.load_profiles()
        mine = stored.get(self._target() or profiles.GLOBAL, {})
        eff = profiles.effective(self.model, profiles=stored)

        limits = self._limits()
        self._layers, self._trained_ctx = limits.block_count, \
            limits.context_length
        self._experts = limits.expert_count if limits.source else -1
        self._ctx_now = eff.get("context_length") or 0
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
            if key == "cpu_moe_layers" and self._experts == 0:
                continue  # a dense model has no experts to keep on the CPU
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
        group.add(self._folders_row())
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
        elif spec.key in SLIDERS:
            row = self._slider_row(spec, own, effective)
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

    def _slider_row(self, spec, own, effective):
        kind, *conf = SLIDERS[spec.key]
        layers = self._layers or 200
        # The name and value on one line, the slider full width under it.
        row = Adw.PreferencesRow(title=spec.label)
        value_label = Gtk.Label(css_classes=["dim-label", "numeric"],
                                xalign=1)
        scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL,
                          hexpand=True, draw_value=False)
        start = own if own is not None else effective \
            if effective is not None else SLIDER_START.get(spec.key)
        trained = self._trained_ctx
        caption = ""
        if kind == "steps":
            steps = list(conf[0])
            if spec.key == "context_length" and trained:
                # Up to the model's own trained window (its exact value is
                # a stop, so 40960 or 131072 can be picked), then up to 4x
                # for RoPE/YaRN stretching.
                steps = sorted({s for s in steps if s <= 4 * trained}
                               | {trained})
                caption = (f"Trained on {_short(trained)}. Past the mark "
                           "needs RoPE scaling (yarn).")
            elif spec.key == "context_length":
                caption = "This model's trained window is unknown."
            if spec.key == "max_tokens":
                # A reply can't be longer than the window it lives in.
                window = self._ctx_now or trained
                if window:
                    steps = [s for s in steps if s <= window]
                    if window not in steps:
                        steps.append(window)
            if spec.key in ("batch_size", "ubatch_size") and (
                    self._ctx_now or trained):
                window = self._ctx_now or trained
                steps = [s for s in steps if s <= window] or steps[:1]
            scale.set_range(0, len(steps) - 1)
            scale.set_increments(1, 1)
            scale.set_round_digits(0)
            if start is None:
                start = steps[0]
            idx = min(range(len(steps)), key=lambda i: abs(
                math.log2(abs(steps[i]) + 1) - math.log2(abs(start) + 1))
                if start >= 0 else (0 if steps[i] < 0 else 1))

            def value_of(pos):
                return steps[int(round(pos))]
            scale.set_value(idx)
            if spec.key == "context_length" and self._trained_ctx in steps:
                scale.add_mark(steps.index(self._trained_ctx),
                               Gtk.PositionType.BOTTOM, None)
        else:
            lo, hi, step, digits = conf
            # llama.cpp counts the output layer too, so all = layers + 1.
            hi = {"cpus": os.cpu_count() or 16,
                  "layers": layers + 1 if self._layers else layers
                  }.get(hi, hi)
            if spec.key == "cpu_moe_layers":
                hi = layers
            if conf[1] == "layers" and self._layers:
                caption = f"This model has {self._layers} layers."
            scale.set_range(lo, hi)
            scale.set_increments(step, step * 10)
            scale.set_round_digits(digits)
            if start is None:
                start = lo

            def value_of(pos):
                v = round(round(pos / step) * step, digits)
                return int(v) if spec.kind == "int" else v
            scale.set_value(min(max(start, lo), hi))

        state = {"set": own is not None}

        def show(value):
            text = _short(value)
            if spec.key in ("gpu_layers", "max_tokens") and value == -1:
                text = "All" if spec.key == "gpu_layers" else "No limit"
            if spec.key == "gpu_layers" and self._layers and \
                    value > self._layers:
                text = "All"
            if spec.key == "context_length" and trained and value > trained:
                text += f"  ·  {value / trained:g}x stretched"
            if not state["set"]:
                # Not set here: a preset's or all-models value, or default.
                text = text if effective is not None else DEFAULT
            value_label.set_label(text)
        reset = Gtk.Button(icon_name="edit-undo-symbolic",
                           valign=Gtk.Align.CENTER,
                           css_classes=["flat", "circular"],
                           tooltip_text="Back to the default",
                           visible=own is not None)
        show(value_of(scale.get_value()))
        pending = {"id": 0}

        def changed(_scale):
            if self._building:
                return
            value = value_of(scale.get_value())
            state["set"] = True
            show(value)
            reset.set_visible(True)
            # Save once the slider stops moving.
            if pending["id"]:
                GLib.source_remove(pending["id"])

            def save():
                pending["id"] = 0
                self._save(spec.key, value, quiet=True)
                return False
            pending["id"] = GLib.timeout_add(400, save)
        scale.connect("value-changed", changed)

        def on_reset(_btn):
            self._save(spec.key, None, quiet=True)
            GLib.idle_add(self.rebuild)
        reset.connect("clicked", on_reset)
        top = Gtk.Box(spacing=6)
        top.append(Gtk.Label(label=spec.label, xalign=0, hexpand=True))
        top.append(value_label)
        top.append(reset)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2,
                      margin_top=10, margin_bottom=6, margin_start=12,
                      margin_end=8)
        box.append(top)
        box.append(scale)
        if caption:
            box.append(Gtk.Label(label=caption, xalign=0, wrap=True,
                                 css_classes=["dim-label", "caption"]))
        row.set_child(box)
        return row

    def _folders_row(self):
        cfg = config_mod.load_config()
        count = len(cfg.model_dirs) + 1
        row = Adw.ExpanderRow(use_markup=False, title="Model folders",
                              subtitle=f"{count} folder"
                              + ("s" if count != 1 else ""))
        row.set_tooltip_text(
            "Folders CherryColaLLM looks in for GGUF models, for example LM "
            "Studio's models folder. Models found here run with the "
            "llama.cpp engine and appear in the model menu.")
        home = os.path.expanduser("~")

        def short(path) -> str:
            text = str(path)
            return "~" + text[len(home):] if text.startswith(home) else text
        own = Adw.ActionRow(use_markup=False, title="CherryColaLLM models",
                            subtitle=short(llamacpp.models_dir(cfg)),
                            subtitle_lines=1)
        own.set_tooltip_text(f"{llamacpp.models_dir(cfg)}\nDownloads and "
                             f"requantized models go here.")
        row.add_row(own)
        for folder in cfg.model_dirs:
            item = Adw.ActionRow(use_markup=False,
                                 title=os.path.basename(folder) or folder,
                                 subtitle=short(folder), subtitle_lines=1)
            item.set_tooltip_text(folder)
            rm = Gtk.Button(icon_name="list-remove-symbolic",
                            valign=Gtk.Align.CENTER, css_classes=["flat"],
                            tooltip_text="Stop looking in this folder "
                                         "(nothing is deleted)")
            rm.connect("clicked", self._on_remove_folder, folder)
            item.add_suffix(rm)
            row.add_row(item)
        add = Gtk.Button(label="Add folder…", halign=Gtk.Align.START,
                         css_classes=["flat"], margin_top=6, margin_bottom=6,
                         margin_start=6)
        add.connect("clicked", self._on_add_folder)
        row.add_row(add)
        return row

    def _on_add_folder(self, *_):
        dialog = Gtk.FileDialog(title="Add a folder of GGUF models")

        def done(dlg, result):
            try:
                folder = dlg.select_folder_finish(result)
            except GLib.Error:
                return  # cancelled
            cfg = config_mod.load_config()
            try:
                path, count = llamacpp.add_folder(cfg, folder.get_path())
            except llamacpp.LlamaError as exc:
                self.win.toast(str(exc))
                return
            hint = "" if self.backend == "llamacpp" else \
                " Switch the engine to llama.cpp to use them."
            self.win.toast(f"Found {count} models in {path.name}.{hint}")
            self.win.cfg = config_mod.load_config()
            self.win.probe()
            self.rebuild()
        dialog.select_folder(self.win, None, done)

    def _on_remove_folder(self, _btn, folder):
        cfg = config_mod.load_config()
        llamacpp.remove_folder(cfg, folder)
        self.win.cfg = config_mod.load_config()
        self.win.probe()
        self.rebuild()

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

    def _save(self, key, value, quiet=False):
        try:
            if value is None:
                profiles.unset_value(self._target(), key)
            else:
                profiles.set_value(self._target(), key, value)
        except profiles.SettingError as exc:
            self.win.toast(str(exc))
            return False
        spec = profiles.SETTINGS.get(key)
        if not quiet and (spec and spec.group in ("load", "gpu", "rope")
                          or key == "llama_args"):
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

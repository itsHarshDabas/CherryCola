"""Model load and sampling settings, per model, with presets.

Stored as JSON in ~/.config/grace/cherrycola-models.json (respects
$XDG_CONFIG_HOME)::

    {"*": {"context_length": 8192},
     "qwen3:8b": {"preset": "code", "gpu_layers": 20, "temperature": 0.3}}

"*" holds defaults for every model. The effective settings for a model are,
lowest to highest: built-in defaults, "*", the model's preset, the model's
own keys, then one-off overrides (``ask -o key=value``).

Settings use engine-neutral names (close to LM Studio's). Each backend maps
the ones it understands and reports the rest as unsupported, so nothing is
silently dropped.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "GROUPS",
    "PRESETS",
    "SETTINGS",
    "Setting",
    "SettingError",
    "effective",
    "llamacpp_request",
    "lmstudio_request",
    "load_profiles",
    "ollama_request",
    "ollama_server_env",
    "parse_value",
    "profiles_path",
    "save_profiles",
    "set_value",
    "unset_value",
]

GLOBAL = "*"


class SettingError(ValueError):
    """A bad key or value; the message is shown to people as is."""


@dataclass(frozen=True)
class Setting:
    key: str
    kind: str  # int | float | bool | str | choice | list | dict
    group: str  # load | gpu | sampling | other
    label: str  # short name shown in the app
    help: str  # the hover explanation, plain words
    lo: float | None = None
    hi: float | None = None
    choices: tuple[str, ...] = ()
    engines: tuple[str, ...] = ("llamacpp", "ollama", "lmstudio")


_ALL = ("llamacpp", "ollama", "lmstudio")
_LOAD = ("llamacpp", "ollama")
_LLAMA = ("llamacpp",)
_KV_TYPES = ("f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl")

SETTINGS: dict[str, Setting] = {s.key: s for s in (
    # --- loading --------------------------------------------------------
    Setting("context_length", "int", "load", "Context length",
            "How many tokens the model can see at once: your chat so far, "
            "pasted text and its answer. Bigger remembers more but uses "
            "more memory.", 256, 1_048_576, engines=_LOAD),
    Setting("threads", "int", "load", "CPU threads",
            "Processor threads used to generate text. Usually best at the "
            "number of physical cores.", 1, 512, engines=_LOAD),
    Setting("batch_size", "int", "load", "Batch size",
            "How many prompt tokens are processed in one step. Higher reads "
            "long prompts faster but uses more memory.", 1, 65_536,
            engines=_LOAD),
    Setting("ubatch_size", "int", "load", "Micro-batch size",
            "The physical chunk size inside a batch. Lower it if loading "
            "runs out of GPU memory.", 1, 65_536, engines=_LLAMA),
    Setting("use_mmap", "bool", "load", "Memory-map the file",
            "Read the model straight from disk as it's needed instead of "
            "copying it into RAM first. Loads faster and lets the system "
            "share or drop pages.", engines=_LOAD),
    Setting("use_mlock", "bool", "load", "Keep in RAM",
            "Lock the model in memory so it is never swapped out. Steadier "
            "speed, but the RAM stays taken.", engines=_LOAD),
    Setting("disk_stream", "bool", "load", "Stream from disk",
            "For models bigger than your RAM and graphics memory: put what "
            "fits on the GPU and read the remaining layers from disk for "
            "each token. Works, but slow (often seconds per token).",
            engines=_LLAMA),
    Setting("keep_alive", "str", "load", "Keep loaded for",
            "How long the model stays in memory after an answer, such as "
            "5m or 1h. -1 keeps it loaded.", engines=("ollama",)),
    # --- gpu ------------------------------------------------------------
    Setting("gpu_layers", "int", "gpu", "GPU layers",
            "How many of the model's layers run on the graphics card. More "
            "is faster until graphics memory runs out. -1 means all, 0 "
            "means CPU only.", -1, 999, engines=_LOAD),
    Setting("main_gpu", "int", "gpu", "Main GPU",
            "Which graphics card to use first when there are several "
            "(0 is the first).", 0, 16, engines=_LLAMA),
    Setting("tensor_split", "str", "gpu", "Split across GPUs",
            "How to share the model between several graphics cards, as "
            "proportions such as 3,1.", engines=_LLAMA),
    Setting("cpu_moe_layers", "int", "gpu", "Experts on CPU",
            "Mixture-of-experts models only: keep the experts of the first "
            "N layers in RAM and everything else on the GPU. Lets big MoE "
            "models run on small graphics cards.", 0, 999, engines=_LLAMA),
    Setting("flash_attention", "bool", "gpu", "Flash attention",
            "A faster, leaner way to compute attention. Needed for a "
            "quantized KV cache.", engines=_LOAD),
    Setting("kv_cache_type", "choice", "gpu", "KV cache type",
            "How the model's working memory is stored. q8_0 halves it with "
            "almost no quality loss; q4_0 quarters it.",
            choices=_KV_TYPES, engines=_LOAD),
    # --- rope -----------------------------------------------------------
    Setting("rope_scaling", "choice", "rope", "RoPE scaling",
            "How positions are stretched when you run past the context the "
            "model was trained on. yarn works best for long context.",
            choices=("none", "linear", "yarn"), engines=_LLAMA),
    Setting("rope_freq_base", "float", "rope", "RoPE base",
            "The base frequency for positions. 0 uses the model's own "
            "value; change it only if a model card says so.", 0, 1e9,
            engines=_LOAD),
    Setting("rope_freq_scale", "float", "rope", "RoPE scale",
            "Shrinks positions to fit a longer context (0.5 doubles it). "
            "0 uses the model's own value.", 0, 100, engines=_LOAD),
    Setting("yarn_orig_ctx", "int", "rope", "Trained context",
            "The context length the model was trained with, for yarn "
            "scaling. 0 reads it from the model.", 0, 1_048_576,
            engines=_LLAMA),
    # --- sampling -------------------------------------------------------
    Setting("temperature", "float", "sampling", "Temperature",
            "Randomness. Low gives focused, repeatable answers; high gives "
            "varied, creative ones.", 0, 5),
    Setting("top_k", "int", "sampling", "Top K",
            "Only consider the K most likely next words. 0 turns it off.",
            0, 1000),
    Setting("top_p", "float", "sampling", "Top P",
            "Only consider the most likely words that together add up to "
            "this probability.", 0, 1),
    Setting("min_p", "float", "sampling", "Min P",
            "Drop words less likely than this share of the top word's "
            "chance. Cuts nonsense without dulling answers.", 0, 1),
    Setting("repeat_penalty", "float", "sampling", "Repeat penalty",
            "Makes repeating recent words less likely. 1 turns it off.",
            0, 5),
    Setting("repeat_last_n", "int", "sampling", "Repeat window",
            "How many recent tokens count for the repeat penalty. -1 means "
            "the whole context.", -1, 1_048_576, engines=_LOAD),
    Setting("presence_penalty", "float", "sampling", "Presence penalty",
            "Nudges the model toward new topics once a word has appeared.",
            -2, 2),
    Setting("frequency_penalty", "float", "sampling", "Frequency penalty",
            "Penalizes words the more often they have appeared.", -2, 2),
    Setting("max_tokens", "int", "sampling", "Max answer length",
            "Stop the answer after this many tokens. -1 means no limit.",
            -1, 1_048_576),
    Setting("seed", "int", "sampling", "Seed",
            "Fix this to get the same answer for the same question. -1 is "
            "random each time.", -1, 2**31 - 1),
    Setting("stop", "list", "sampling", "Stop strings",
            "The answer ends when the model writes any of these. Separate "
            "several with |."),
    # --- other ----------------------------------------------------------
    Setting("system_prompt", "str", "other", "System prompt",
            "Instructions the model reads before every chat. Replaces "
            "CherryCola's default."),
    Setting("preset", "choice", "other", "Preset",
            "A ready-made set of sampling values. Anything you set by hand "
            "wins over the preset.",
            choices=("precise", "balanced", "creative", "code")),
    Setting("llama_args", "dict", "advanced", "llama.cpp options",
            "Any other llama-server option, by its flag name. CherryCola "
            "checks each against the installed llama-server.",
            engines=_LLAMA),
    Setting("draft_model", "str", "other", "Draft model",
            "A small model of the same family that guesses ahead so the "
            "big one answers faster (speculative decoding).",
            engines=_LLAMA),
)}

GROUPS = {"load": "Loading", "gpu": "GPU and memory", "rope": "Long context",
          "sampling": "Sampling", "other": "Behaviour",
          "advanced": "All llama.cpp options"}

PRESETS: dict[str, dict] = {
    "precise": {"temperature": 0.2, "top_k": 20, "top_p": 0.8,
                "min_p": 0.05, "repeat_penalty": 1.05},
    "balanced": {"temperature": 0.7, "top_k": 40, "top_p": 0.9,
                 "min_p": 0.05, "repeat_penalty": 1.1},
    "creative": {"temperature": 1.1, "top_k": 80, "top_p": 0.95,
                 "min_p": 0.02, "repeat_penalty": 1.05},
    "code": {"temperature": 0.15, "top_k": 20, "top_p": 0.9,
             "min_p": 0.05, "repeat_penalty": 1.0},
}

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def profiles_path() -> Path:
    from . import osdeps
    return osdeps.config_home() / "grace" / "cherrycola-models.json"


def parse_value(key: str, raw):
    """Check and convert one value (a string from the CLI, or JSON)."""
    spec = SETTINGS.get(key)
    if spec is None:
        raise SettingError(f"unknown setting {key!r} "
                           f"(see: cherrycola settings --list)")
    try:
        if spec.kind == "int":
            if isinstance(raw, bool):
                raise ValueError
            value = int(raw)
        elif spec.kind == "float":
            if isinstance(raw, bool):
                raise ValueError
            value = float(raw)
        elif spec.kind == "bool":
            if isinstance(raw, bool):
                value = raw
            elif str(raw).strip().lower() in _TRUE:
                value = True
            elif str(raw).strip().lower() in _FALSE:
                value = False
            else:
                raise ValueError
        elif spec.kind == "dict":
            if not isinstance(raw, dict) or not all(
                    isinstance(k, str) and k.startswith("-")
                    and isinstance(v, (str, bool, int, float))
                    for k, v in raw.items()):
                raise ValueError
            value = {k: v if isinstance(v, bool) else str(v)
                     for k, v in raw.items()}
        elif spec.kind == "list":
            if isinstance(raw, list):
                value = [str(v) for v in raw if str(v)]
            else:
                value = [p for p in str(raw).split("|") if p]
        elif spec.kind == "choice":
            value = str(raw).strip().lower()
            if value not in spec.choices:
                raise SettingError(
                    f"{key} must be one of: {', '.join(spec.choices)}")
        else:
            value = str(raw)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, SettingError):
            raise
        raise SettingError(f"{key} needs a {spec.kind} value, "
                           f"not {raw!r}") from None
    if spec.kind in ("int", "float") and (
            (spec.lo is not None and value < spec.lo)
            or (spec.hi is not None and value > spec.hi)):
        lo = int(spec.lo) if spec.kind == "int" else spec.lo
        hi = int(spec.hi) if spec.kind == "int" else spec.hi
        raise SettingError(f"{key} must be between {lo} and {hi}")
    return value


def load_profiles(path: Path | None = None) -> dict[str, dict]:
    """Read the file, dropping anything invalid (never raises)."""
    try:
        data = json.loads((path or profiles_path()).read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict] = {}
    for model, values in data.items():
        if not isinstance(model, str) or not isinstance(values, dict):
            continue
        clean = {}
        for key, raw in values.items():
            try:
                clean[key] = parse_value(key, raw)
            except SettingError:
                continue
        out[model] = clean
    return out


def save_profiles(profiles: dict[str, dict],
                  path: Path | None = None) -> Path:
    target = path or profiles_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    keep = {m: v for m, v in profiles.items() if v}
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(keep, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, target)
    return target


def set_value(model: str | None, key: str, raw,
              path: Path | None = None):
    value = parse_value(key, raw)
    profiles = load_profiles(path)
    profiles.setdefault(model or GLOBAL, {})[key] = value
    save_profiles(profiles, path)
    return value


def unset_value(model: str | None, key: str | None,
                path: Path | None = None) -> bool:
    """Remove one key, or the whole model entry when key is None."""
    profiles = load_profiles(path)
    entry = profiles.get(model or GLOBAL)
    if entry is None:
        return False
    if key is None:
        del profiles[model or GLOBAL]
    elif key in entry:
        del entry[key]
    else:
        return False
    save_profiles(profiles, path)
    return True


def effective(model: str, overrides: dict | None = None,
              profiles: dict[str, dict] | None = None) -> dict:
    """The settings that apply to ``model``, presets expanded."""
    profiles = load_profiles() if profiles is None else profiles
    layers = [profiles.get(GLOBAL, {}), profiles.get(model, {}),
              overrides or {}]
    preset = None
    for layer in layers:
        preset = layer.get("preset", preset)
    out: dict = {}
    out.update({k: v for k, v in layers[0].items()})
    if preset:
        out.update(PRESETS[preset])
    for layer in layers[1:]:
        out.update(layer)
    if preset:
        out["preset"] = preset
    raw: dict = {}
    for layer in layers:  # raw llama.cpp flags add up instead of replacing
        raw.update(layer.get("llama_args", {}))
    if raw:
        out["llama_args"] = raw
    return out


# ---- mapping to engines ---------------------------------------------------

_OLLAMA_OPTIONS = {
    "context_length": "num_ctx",
    "gpu_layers": "num_gpu",
    "threads": "num_thread",
    "batch_size": "num_batch",
    "use_mmap": "use_mmap",
    "use_mlock": "use_mlock",
    "rope_freq_base": "rope_frequency_base",
    "rope_freq_scale": "rope_frequency_scale",
    "temperature": "temperature",
    "top_k": "top_k",
    "top_p": "top_p",
    "min_p": "min_p",
    "repeat_penalty": "repeat_penalty",
    "repeat_last_n": "repeat_last_n",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "max_tokens": "num_predict",
    "seed": "seed",
    "stop": "stop",
}
# Ollama only takes these for the whole server, as environment variables.
_OLLAMA_SERVER = {"kv_cache_type", "flash_attention"}
_HANDLED = {"system_prompt", "preset"}

_LLAMACPP_SAMPLING = {"temperature", "top_k", "top_p", "min_p",
                      "repeat_penalty", "repeat_last_n", "presence_penalty",
                      "frequency_penalty", "seed", "stop", "max_tokens"}


def ollama_request(settings: dict) -> tuple[dict, list[str]]:
    """Extra /api/chat body fields, and keys Ollama can't apply per request."""
    options, extra, skipped = {}, {}, []
    for key, value in settings.items():
        if key in _HANDLED:
            continue
        if key in _OLLAMA_OPTIONS:
            if key == "gpu_layers" and value < 0:
                value = 999  # Ollama: offload as many layers as exist
            if key in ("rope_freq_base", "rope_freq_scale") and value == 0:
                continue
            options[_OLLAMA_OPTIONS[key]] = value
        elif key == "keep_alive":
            extra["keep_alive"] = (int(value) if value.lstrip("-").isdigit()
                                   else value)
        elif key not in _OLLAMA_SERVER:
            skipped.append(key)
    if options:
        extra["options"] = options
    return extra, skipped


def ollama_server_env(settings: dict) -> dict[str, str]:
    """Environment for ``ollama serve`` (KV cache type, flash attention)."""
    env = {}
    if "flash_attention" in settings:
        env["OLLAMA_FLASH_ATTENTION"] = "1" if settings["flash_attention"] \
            else "0"
    if "kv_cache_type" in settings:
        kv = settings["kv_cache_type"]
        # Ollama knows f16, q8_0 and q4_0; quantized caches need flash attn.
        env["OLLAMA_KV_CACHE_TYPE"] = kv if kv in ("f16", "q8_0", "q4_0") \
            else "q8_0" if kv.startswith(("q8", "q5")) else "q4_0" \
            if kv.startswith(("q4", "iq4")) else "f16"
        if env["OLLAMA_KV_CACHE_TYPE"] != "f16":
            env["OLLAMA_FLASH_ATTENTION"] = "1"
    return env


_LMSTUDIO = {"temperature", "top_k", "top_p", "min_p", "repeat_penalty",
             "presence_penalty", "frequency_penalty", "seed", "stop"}


def lmstudio_request(settings: dict) -> tuple[dict, list[str]]:
    """Extra /v1/chat/completions fields. Loading is set in LM Studio."""
    extra, skipped = {}, []
    for key, value in settings.items():
        if key in _HANDLED:
            continue
        if key in _LMSTUDIO:
            extra[key] = value
        elif key == "max_tokens":
            if value >= 0:
                extra["max_tokens"] = value
        else:
            skipped.append(key)
    return extra, skipped


def llamacpp_request(settings: dict) -> dict:
    """Per-request sampling for llama-server; loading goes on its flags."""
    extra = {}
    for key in _LLAMACPP_SAMPLING & settings.keys():
        value = settings[key]
        if key == "max_tokens":
            extra["n_predict"] = value
        else:
            extra[key] = value
    return extra

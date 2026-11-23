"""Config: ~/.config/grace/cherrycola.conf (key = value, # comments)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "BACKENDS",
    "DEFAULT_DUCKDUCKGO_URL",
    "DEFAULT_HF_URL",
    "DEFAULT_LLAMACPP_URL",
    "DEFAULT_LMSTUDIO_URL",
    "DEFAULT_OLLAMA_URL",
    "DEFAULT_SEARXNG_URL",
    "MODEL_SIZES_GB",
    "SEARCH_BACKENDS",
    "Config",
    "config_path",
    "default_model_for_ram",
    "load_config",
    "save_setting",
    "total_ram_bytes",
]

BACKENDS = ("auto", "ollama", "lmstudio", "llamacpp")
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_LMSTUDIO_URL = "http://127.0.0.1:1234"
DEFAULT_LLAMACPP_URL = "http://127.0.0.1:11436"
DEFAULT_HF_URL = "https://huggingface.co"
SEARCH_BACKENDS = ("duckduckgo", "searxng", "off")
DEFAULT_SEARXNG_URL = "http://127.0.0.1:8888"
DEFAULT_DUCKDUCKGO_URL = "https://html.duckduckgo.com/html/"

#: Approximate download sizes for the default models (GB).
MODEL_SIZES_GB = {
    "qwen3:1.7b": 1.4,
    "qwen3:4b": 2.6,
    "qwen3:8b": 5.2,
}

_VALID_KEYS = ("backend", "model", "ollama_url", "lmstudio_url",
               "search", "searxng_url", "duckduckgo_url", "web_pages",
               "llamacpp_url", "models_dir", "hf_url", "llama_server",
               "llama_quantize", "llama_imatrix", "model_dirs")
_PLAIN_KEYS = ("models_dir", "llama_server", "llama_quantize",
               "llama_imatrix")


@dataclass
class Config:
    backend: str = "auto"
    model: str | None = None
    ollama_url: str = DEFAULT_OLLAMA_URL
    lmstudio_url: str = DEFAULT_LMSTUDIO_URL
    search: str = "duckduckgo"
    searxng_url: str = DEFAULT_SEARXNG_URL
    duckduckgo_url: str = DEFAULT_DUCKDUCKGO_URL
    web_pages: int = 3
    llamacpp_url: str = DEFAULT_LLAMACPP_URL
    models_dir: str = ""  # empty: $XDG_DATA_HOME/cherrycola/models
    hf_url: str = DEFAULT_HF_URL
    llama_server: str = "llama-server"
    llama_quantize: str = "llama-quantize"
    llama_imatrix: str = "llama-imatrix"
    # More folders of GGUF files, separated by | in the file.
    model_dirs: list[str] = field(default_factory=list)
    path: Path | None = None


def config_path() -> Path:
    from . import osdeps
    return osdeps.config_home() / "grace" / "cherrycola.conf"


def load_config(path: Path | None = None) -> Config:
    cfg = Config(path=path or config_path())
    try:
        text = cfg.path.read_text(encoding="utf-8")  # type: ignore[union-attr]
    except FileNotFoundError:
        return cfg
    except OSError:
        return cfg
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key not in _VALID_KEYS or not value:
            continue
        if key == "backend":
            if value in BACKENDS:
                cfg.backend = value
        elif key == "model":
            cfg.model = value
        elif key == "ollama_url":
            cfg.ollama_url = value.rstrip("/")
        elif key == "lmstudio_url":
            cfg.lmstudio_url = value.rstrip("/")
        elif key == "search":
            lowered = value.lower()
            if lowered in SEARCH_BACKENDS:
                cfg.search = lowered
        elif key == "searxng_url":
            cfg.searxng_url = value.rstrip("/")
        elif key == "duckduckgo_url":
            cfg.duckduckgo_url = value.rstrip("/")
        elif key in ("llamacpp_url", "hf_url"):
            setattr(cfg, key, value.rstrip("/"))
        elif key == "model_dirs":
            cfg.model_dirs = [d.strip() for d in value.split("|") if d.strip()]
        elif key in _PLAIN_KEYS:
            setattr(cfg, key, value)
        elif key == "web_pages":
            try:
                cfg.web_pages = max(0, min(5, int(value)))
            except ValueError:
                pass
    return cfg


def save_setting(key: str, value: str, path: Path | None = None) -> Path:
    """Rewrite one ``key = value`` line, keeping comments; append if absent."""
    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
        had_file = True
    except FileNotFoundError:
        lines = []
        had_file = False
    prefix = key.strip() + "="
    squished = key.strip()
    updated = False
    out: list[str] = []
    for raw in lines:
        stripped = raw.strip()
        if (
            not updated
            and stripped
            and not stripped.startswith("#")
            and "=" in raw
        ):
            k, _, _ = raw.partition("=")
            if k.strip() == squished:
                out.append(f"{squished} = {value}")
                updated = True
                continue
        out.append(raw)
    _ = prefix  # documented: match is on the key part before '='
    if not updated:
        out.append(f"{squished} = {value}")
    text = "\n".join(out) + "\n"
    if had_file or out:
        target.write_text(text, encoding="utf-8")
    return target


def total_ram_bytes(meminfo: str = "/proc/meminfo") -> int | None:
    from . import osdeps
    return osdeps.total_ram(meminfo)


def default_model_for_ram(total_bytes: int | None) -> str:
    if total_bytes is None:
        return "qwen3:4b"
    gb = total_bytes / 1024**3
    if gb < 6:
        return "qwen3:1.7b"
    if gb < 12:
        return "qwen3:4b"
    return "qwen3:8b"

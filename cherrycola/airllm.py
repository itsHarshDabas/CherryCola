"""The AirLLM engine: big Hugging Face models on a small graphics card.

AirLLM (https://github.com/lyogavin/airllm) runs a model one transformer
layer at a time, loading each layer from disk just before it is needed, so a
70B model runs in about 4 GB of graphics memory. It is slow (every token
reads the whole model from disk), but it runs models nothing else here can.

It needs PyTorch and transformers (a few GB), so it is optional:
``cherrycolallm airllm install`` makes a Python environment just for it under
CherryColaLLM's data folder (``airllm_python`` in the config points at
another one instead). CherryColaLLM itself still needs nothing but Python.

Models are Hugging Face repos with safetensors weights (not GGUF files):
``cherrycolallm pull owner/repo`` downloads one into the AirLLM models
folder, and ``cherrycolallm pull /path/to/folder`` uses a model folder
already on this PC, where it is. ``airllm_worker.py`` serves the loaded model
on ``airllm_url`` with an OpenAI-style chat endpoint, started on demand like
llama-server.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import config as config_mod
from . import osdeps

__all__ = [
    "AirError",
    "add_folder",
    "base_python",
    "download",
    "ensure_loaded",
    "find_model",
    "install",
    "installed",
    "list_local",
    "loaded_model",
    "python",
    "remove",
    "stop",
]

_LOAD_TIMEOUT = 3 * 3600  # the first load splits the model into layers
_REPO = re.compile(r"^[\w.-]+/[\w.-]+$")
# What a model needs; skips GGUF, ONNX, other formats and big .bin
# duplicates when safetensors are there.
_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken",
             "tokenizer*", "*.py"]


class AirError(Exception):
    """Expected failure; the message is shown to people as is."""


# ---- the Python environment ---------------------------------------------------

def home() -> Path:
    return osdeps.data_home() / "cherrycola" / "airllm"


def _venv() -> Path:
    return home() / "venv"


def python(cfg: config_mod.Config) -> str:
    """The Python that has AirLLM (configured, else CherryColaLLM's own env)."""
    if cfg.airllm_python:
        return os.path.expanduser(cfg.airllm_python)
    bin_dir = "Scripts" if osdeps.WINDOWS else "bin"
    exe = "python.exe" if osdeps.WINDOWS else "python"
    return str(_venv() / bin_dir / exe)


def _marker() -> Path:
    return home() / "installed.json"


def installed(cfg: config_mod.Config) -> bool:
    """AirLLM is set up (checked cheaply: no torch import on every status)."""
    if cfg.airllm_python:
        return Path(python(cfg)).is_file() or osdeps.which(
            cfg.airllm_python) is not None
    return Path(python(cfg)).is_file() and _marker().is_file()


def base_python() -> list[str] | None:
    """A Python 3.9+ to make the environment with.

    Running from source, that's the Python running CherryColaLLM. The
    Windows exe has no usable Python inside, so it looks for one (the py
    launcher first).
    """
    if not getattr(sys, "frozen", False) and sys.version_info >= (3, 9):
        return [sys.executable]
    for cmd in (["py", "-3"], ["python3"], ["python"]):
        exe = osdeps.which(cmd[0])
        if exe is None:
            continue
        try:
            out = subprocess.run(
                [exe, *cmd[1:], "-c",
                 "import sys,venv;print(sys.version_info>=(3,9))"],
                capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            continue
        if out.returncode == 0 and out.stdout.strip() == "True":
            return [exe, *cmd[1:]]
    return None


def _torch_index(gpus) -> str:
    """PyTorch's wheel index for this PC: CUDA with an NVIDIA card, else
    the small CPU-only build (AirLLM is much slower without CUDA)."""
    if any(g.vendor == "nvidia" for g in gpus):
        return "https://download.pytorch.org/whl/cu124"
    return "https://download.pytorch.org/whl/cpu"


def install_steps(cfg: config_mod.Config, base: list[str], gpus) -> list[list[str]]:
    """The commands ``install`` runs, in order."""
    py = python(cfg)
    nvidia = any(g.vendor == "nvidia" for g in gpus)
    extra = ["bitsandbytes"] if nvidia else []
    return [
        [*base, "-m", "venv", str(_venv())],
        [py, "-m", "pip", "install", "--upgrade", "pip"],
        [py, "-m", "pip", "install", "torch", "--index-url",
         _torch_index(gpus)],
        [py, "-m", "pip", "install", "airllm", "huggingface_hub", *extra],
    ]


def install(cfg: config_mod.Config, gpus, run=subprocess.run,
            on_status=None) -> str:
    """Make the AirLLM environment. Returns its Python. Output passes through
    (pip shows its progress)."""
    if cfg.airllm_python:
        raise AirError(f"airllm_python is set ({cfg.airllm_python}): install "
                       f"airllm in that Python yourself (pip install airllm)")
    base = base_python()
    if base is None:
        raise AirError("AirLLM needs Python 3.9 or newer to install into. "
                       "Install it from https://www.python.org/downloads/ "
                       "(or: winget install Python.Python.3.12), then try "
                       "again")
    home().mkdir(parents=True, exist_ok=True)
    for step in install_steps(cfg, base, gpus):
        if on_status:
            on_status(" ".join(Path(step[0]).name if i == 0 else a
                               for i, a in enumerate(step)))
        try:
            proc = run(step)
        except OSError as exc:
            raise AirError(f"{step[0]}: {exc}") from exc
        if proc.returncode != 0:
            raise AirError(f"installing AirLLM failed at: {' '.join(step[2:5])}"
                           f" (exit {proc.returncode})")
    _marker().write_text(json.dumps({"at": int(time.time()),
                                     "torch": _torch_index(gpus)}), "utf-8")
    return python(cfg)


def uninstall(cfg: config_mod.Config) -> bool:
    """Delete the environment (models stay)."""
    stop()
    if not _venv().exists():
        return False
    shutil.rmtree(_venv(), ignore_errors=True)
    try:
        _marker().unlink()
    except OSError:
        pass
    return True


def _need(cfg: config_mod.Config) -> None:
    if not installed(cfg):
        raise AirError("AirLLM isn't installed. It needs PyTorch (a few GB), "
                       "so it is optional: run cherrycolallm airllm install "
                       "(or press Install AirLLM in settings)")


# ---- models ---------------------------------------------------------------------

def models_dir() -> Path:
    return home() / "models"


def _folders_file() -> Path:
    return home() / "folders.json"


def _folders() -> dict[str, str]:
    try:
        data = json.loads(_folders_file().read_text("utf-8"))
        return {k: v for k, v in data.items() if isinstance(v, str)}
    except (OSError, ValueError, AttributeError):
        return {}


def is_model_folder(path: Path) -> bool:
    return (path / "config.json").is_file() and (
        any(path.glob("*.safetensors")) or any(path.glob("*.bin")))


def list_local(cfg: config_mod.Config) -> dict[str, Path]:
    """Model name -> folder. Downloads are named owner/repo; folders you
    added by their folder name."""
    _ = cfg
    out: dict[str, Path] = {}
    root = models_dir()
    if root.is_dir():
        for d in sorted(root.iterdir()):
            if d.is_dir() and "--" in d.name and is_model_folder(d):
                out[d.name.replace("--", "/", 1)] = d
    for name, folder in sorted(_folders().items()):
        path = Path(folder)
        if is_model_folder(path):
            out.setdefault(name, path)
    return out


def find_model(cfg: config_mod.Config, name: str) -> Path | None:
    local = list_local(cfg)
    if name in local:
        return local[name]
    path = Path(os.path.expanduser(name))
    if path.is_dir() and is_model_folder(path):
        return path
    return None


def add_folder(cfg: config_mod.Config, folder: str) -> str:
    """Use a model folder where it is (no copy). Returns its name."""
    _ = cfg
    path = Path(os.path.expanduser(folder)).resolve()
    if path.is_file() and path.suffix.lower() == ".gguf":
        raise AirError("AirLLM runs Hugging Face models (a folder with "
                       "config.json and .safetensors files), not .gguf "
                       "files: run .gguf files with the llama.cpp engine")
    if not path.is_dir() or not is_model_folder(path):
        raise AirError(f"{folder} isn't a model folder (it needs "
                       f"config.json and .safetensors files)")
    folders = _folders()
    name = path.name
    taken = {n for n, p in folders.items() if p != str(path)}
    n = 2
    while name in taken:
        name, n = f"{path.name}-{n}", n + 1
    folders[name] = str(path)
    home().mkdir(parents=True, exist_ok=True)
    _folders_file().write_text(json.dumps(folders, indent=1), "utf-8")
    return name


_DOWNLOAD = """
import os, sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2],
                  allow_patterns=sys.argv[3].split("|"))
"""


def download(cfg: config_mod.Config, repo: str, run=subprocess.run) -> str:
    """Download a Hugging Face repo with AirLLM's huggingface_hub (it shows
    progress and resumes). Returns the model name."""
    _need(cfg)
    if not _REPO.fullmatch(repo) or ".." in repo:
        raise AirError("use owner/repo, for example "
                       "Qwen/Qwen2.5-7B-Instruct")
    dest = models_dir() / repo.replace("/", "--", 1)
    dest.mkdir(parents=True, exist_ok=True)
    env = {**os.environ}
    if cfg.hf_url != config_mod.DEFAULT_HF_URL:
        env["HF_ENDPOINT"] = cfg.hf_url
    try:
        proc = run([python(cfg), "-c", _DOWNLOAD, repo, str(dest),
                    "|".join(_PATTERNS)], env=env)
    except OSError as exc:
        raise AirError(f"could not run AirLLM's Python: {exc}") from exc
    if proc.returncode != 0:
        raise AirError(f"downloading {repo} failed (exit {proc.returncode})")
    if not is_model_folder(dest):
        raise AirError(f"{repo} has no safetensors weights AirLLM can run")
    return repo


def remove(cfg: config_mod.Config, name: str) -> None:
    """Delete a downloaded model (an added folder is only forgotten)."""
    local = list_local(cfg)
    if name not in local:
        raise AirError(f"no AirLLM model named {name}")
    if loaded_model() == name:
        stop()
    folders = _folders()
    if name in folders:
        del folders[name]
        _folders_file().write_text(json.dumps(folders, indent=1), "utf-8")
    else:
        shutil.rmtree(local[name], ignore_errors=True)
    shutil.rmtree(_shards(name), ignore_errors=True)


def limits(cfg: config_mod.Config, name: str) -> tuple[int, int]:
    """(trained context, layers) from the model's config.json, 0 = unknown."""
    path = find_model(cfg, name)
    if path is None:
        return 0, 0
    try:
        conf = json.loads((path / "config.json").read_text("utf-8"))
    except (OSError, ValueError):
        return 0, 0
    conf = conf.get("text_config", conf) if isinstance(conf, dict) else {}
    ctx = conf.get("max_position_embeddings", 0)
    layers = conf.get("num_hidden_layers", 0)
    return (ctx if isinstance(ctx, int) else 0,
            layers if isinstance(layers, int) else 0)


# ---- the server -------------------------------------------------------------------

def _state_file() -> Path:
    return osdeps.runtime_dir() / "airllm.json"


def log_file() -> Path:
    return osdeps.state_home() / "cherrycola" / "airllm.log"


def _shards(name: str) -> Path:
    return home() / "layers" / re.sub(r"[^\w.-]+", "--", name)


def _read_state() -> dict:
    try:
        data = json.loads(_state_file().read_text("utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _alive(pid: int) -> bool:
    if not osdeps.pid_alive(pid):
        return False
    name = osdeps.process_name(pid)
    return name is None or name.startswith("python")


def loaded_model() -> str | None:
    state = _read_state()
    pid = state.get("pid")
    if isinstance(pid, int) and _alive(pid):
        return state.get("model")
    return None


def _health(url: str) -> tuple[str, str]:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health",
                                    timeout=3) as resp:
            return ("ok", "") if resp.status == 200 else ("loading", "")
    except urllib.error.HTTPError as exc:
        if exc.code == 503:
            return "loading", ""
        try:
            return "error", json.loads(exc.read()).get("error", "")
        except (ValueError, OSError, AttributeError):
            return "error", str(exc)
    except (urllib.error.URLError, OSError):
        return "down", ""


def stop() -> bool:
    state = _read_state()
    pid = state.get("pid")
    stopped = False
    if isinstance(pid, int) and _alive(pid):
        try:
            osdeps.terminate(pid)
            stopped = True
        except OSError:
            pass
        for _ in range(40):
            if not _alive(pid):
                break
            time.sleep(0.25)
        else:
            try:
                osdeps.terminate(pid, force=True)
            except OSError:
                pass
    try:
        _state_file().unlink()
    except OSError:
        pass
    return stopped


def _spawn(args: list[str], log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as log:
        return subprocess.Popen(
            args, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, **osdeps.popen_detached())


def _worker() -> Path:
    """The worker script, copied out of the package (the Windows exe keeps
    its files in a temporary folder that goes away when it exits)."""
    src = Path(__file__).with_name("airllm_worker.py")
    dest = home() / "airllm_worker.py"
    home().mkdir(parents=True, exist_ok=True)
    try:
        if not dest.is_file() or dest.read_bytes() != src.read_bytes():
            shutil.copyfile(src, dest)
    except OSError as exc:
        raise AirError(f"could not set up the AirLLM worker: {exc}") from exc
    return dest


def build_args(cfg: config_mod.Config, path: Path, name: str,
               settings: dict) -> list[str]:
    parsed = urllib.parse.urlsplit(cfg.airllm_url)
    args = [python(cfg), str(_worker()), "--model", str(path),
            "--name", name, "--host", parsed.hostname or "127.0.0.1",
            "--port", str(parsed.port or 11437),
            "--shards", str(_shards(name)),
            "--compression", str(settings.get("airllm_compression",
                                              "none")),
            "--prefetch", "1" if settings.get("airllm_prefetch", True)
            else "0"]
    if settings.get("context_length"):
        args += ["--max-seq-len", str(settings["context_length"])]
    return args


def ensure_loaded(cfg: config_mod.Config, model: str, settings: dict,
                  spawn=_spawn, on_status=None,
                  timeout: float = _LOAD_TIMEOUT) -> None:
    """Make the AirLLM worker serve ``model``; restart it when the model or
    a load setting changes."""
    _need(cfg)
    path = find_model(cfg, model)
    if path is None:
        raise AirError(f"no AirLLM model named {model}. Get one with: "
                       f"cherrycolallm pull owner/repo (Hugging Face), or a "
                       f"model folder on this PC")
    args = build_args(cfg, path, model, settings)
    state = _read_state()
    pid = state.get("pid")
    if isinstance(pid, int) and _alive(pid):
        if state.get("args") == args and _health(cfg.airllm_url)[0] == "ok":
            return
        stop()
    elif _health(cfg.airllm_url)[0] != "down":
        raise AirError(f"something else is using {cfg.airllm_url}; set "
                       f"airllm_url to a free port")
    first = not _shards(model).is_dir()
    if on_status:
        on_status(f"Loading {model} with AirLLM" + (
            " (the first time it splits the model into layers on disk, "
            "which takes a while)…" if first else "…"))
    try:
        proc = spawn(args, log_file())
    except OSError as exc:
        raise AirError(f"could not start AirLLM: {exc}") from exc
    _state_file().parent.mkdir(parents=True, exist_ok=True)
    _state_file().write_text(json.dumps(
        {"pid": proc.pid, "model": model, "args": args}), "utf-8")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, error = _health(cfg.airllm_url)
        if status == "ok":
            return
        if status == "error" or proc.poll() is not None:
            stop()
            raise AirError(f"AirLLM could not load {model}: "
                           f"{error or _log_tail()}")
        time.sleep(0.5)
    stop()
    raise AirError(f"AirLLM did not finish loading {model} (see {log_file()})")


def _log_tail(n: int = 6) -> str:
    try:
        lines = log_file().read_text("utf-8", errors="replace").splitlines()
    except OSError:
        return f"see {log_file()}"
    return "\n".join(lines[-n:]) or f"see {log_file()}"

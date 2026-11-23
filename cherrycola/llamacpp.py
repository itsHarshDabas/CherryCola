"""The llama.cpp engine: GGUF files on disk, run by ``llama-server``.

CherryCola starts one ``llama-server`` per user, on demand, with every load
setting turned into a command-line flag. When the model or a load setting
changes, the server is restarted with the new flags. Also here: listing and
downloading GGUF files, importing models Ollama already downloaded,
requantizing with ``llama-quantize``, and planning how a model splits
between the GPU, RAM and disk.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import config as config_mod
from . import gguf, osdeps

__all__ = [
    "LlamaError",
    "Plan",
    "build_args",
    "download",
    "ensure_loaded",
    "find_model",
    "import_ollama",
    "list_local",
    "loaded_model",
    "make_plan",
    "quantize",
    "remove",
    "server_caps",
    "server_options",
    "stop",
]

QUANT_TYPES = (
    "Q2_K", "Q2_K_S", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_1", "Q4_K_S",
    "Q4_K_M", "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0", "IQ1_S",
    "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_XS",
    "IQ3_S", "IQ3_M", "IQ4_NL", "IQ4_XS", "TQ1_0", "TQ2_0", "MXFP4_MOE",
    "F16", "BF16", "F32",
)
# Types that come out badly without an importance matrix.
NEEDS_IMATRIX = {"IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
                 "IQ3_XXS", "Q2_K_S"}

_GB = 1024**3
_SPLIT = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)
_LOAD_TIMEOUT = 900  # a big model read from a slow disk


class LlamaError(Exception):
    """Expected failure; the message is shown to people as is."""


# ---- files ----------------------------------------------------------------

def models_dir(cfg: config_mod.Config) -> Path:
    if cfg.models_dir:
        return Path(os.path.expanduser(cfg.models_dir))
    return osdeps.data_home() / "cherrycola" / "models"


def extra_dirs(cfg: config_mod.Config) -> list[Path]:
    """Folders people added (cherrycola folder add, or the app)."""
    return [Path(os.path.expanduser(d)) for d in cfg.model_dirs]


def _scan(root: Path, prefix: str, out: dict[str, Path]) -> None:
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*.gguf")):
        low = path.name.lower()
        if low.startswith("mmproj") or low.endswith(".imatrix.gguf"):
            continue
        split = _SPLIT.search(path.name)
        if split and split.group(1) != "00001":
            continue
        rel = prefix + path.relative_to(root).as_posix()[: -len(".gguf")]
        out.setdefault(rel, path)


def list_local(cfg: config_mod.Config) -> dict[str, Path]:
    """Model name -> first file.

    Models in CherryCola's own folder are named by their path in it (no
    .gguf); models in an added folder get that folder's name in front.
    """
    out: dict[str, Path] = {}
    _scan(models_dir(cfg), "", out)
    for folder in extra_dirs(cfg):
        _scan(folder, folder.name + "/", out)
    return out


def add_folder(cfg: config_mod.Config, folder: str) -> tuple[Path, int]:
    """Remember a folder of GGUF files. Returns it and its model count."""
    path = Path(os.path.expanduser(folder)).resolve()
    if not path.is_dir():
        raise LlamaError(f"{folder} is not a folder")
    if "|" in str(path):
        raise LlamaError("folder names with | are not supported")
    dirs = [d for d in cfg.model_dirs if Path(d).expanduser().resolve()
            != path] + [str(path)]
    config_mod.save_setting("model_dirs", "|".join(dirs), cfg.path)
    cfg.model_dirs = dirs
    found: dict[str, Path] = {}
    _scan(path, "", found)
    return path, len(found)


def remove_folder(cfg: config_mod.Config, folder: str) -> bool:
    path = Path(os.path.expanduser(folder)).resolve()
    dirs = [d for d in cfg.model_dirs
            if Path(d).expanduser().resolve() != path]
    if len(dirs) == len(cfg.model_dirs):
        return False
    config_mod.save_setting("model_dirs", "|".join(dirs), cfg.path)
    cfg.model_dirs = dirs
    return True


def find_model(cfg: config_mod.Config, name: str) -> Path | None:
    local = list_local(cfg)
    if name in local:
        return local[name]
    path = Path(os.path.expanduser(name))
    if path.suffix.lower() == ".gguf" and path.is_file():
        return path
    return None


def total_size(path: Path) -> int:
    """Bytes of a model, all parts of a split model counted."""
    split = _SPLIT.search(path.name)
    if not split:
        return path.stat().st_size
    total = 0
    for i in range(1, int(split.group(2)) + 1):
        part = path.with_name(_SPLIT.sub(
            f"-{i:05d}-of-{split.group(2)}.gguf", path.name))
        if part.exists():
            total += part.stat().st_size
    return total


def remove(cfg: config_mod.Config, name: str) -> None:
    """Delete a model from the models folder (a link: only the link)."""
    local = list_local(cfg)
    if name not in local:
        raise LlamaError(f"no model named {name} in {models_dir(cfg)}")
    path = local[name]
    if not path.is_relative_to(models_dir(cfg)):
        raise LlamaError(f"{name} is in a folder you added; CherryCola only "
                         f"deletes models in {models_dir(cfg)}")
    if loaded_model() == name:
        stop()
    split = _SPLIT.search(path.name)
    parts = [path]
    if split:
        parts = [path.with_name(_SPLIT.sub(
            f"-{i:05d}-of-{split.group(2)}.gguf", path.name))
            for i in range(1, int(split.group(2)) + 1)]
    for part in parts:
        try:
            part.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise LlamaError(f"could not delete {part}: {exc}") from exc


# ---- the server ------------------------------------------------------------

def _state_file() -> Path:
    return osdeps.runtime_dir() / "llamacpp.json"


def log_file() -> Path:
    return osdeps.state_home() / "cherrycola" / "llama-server.log"


_CAPS: dict[tuple[str, float], set[str]] = {}


def server_caps(binary: str) -> set[str]:
    """Flags this llama-server build knows, read once from --help.

    llama.cpp renames flags now and then (``--no-mmap``/``--mlock`` became
    ``--load-mode``), and distro packages lag behind, so ask the binary.
    """
    path = shutil.which(binary)
    if path is None:
        raise LlamaError(
            f"{binary} is not installed. Install llama.cpp (it provides "
            f"llama-server), or set llama_server in cherrycola.conf")
    key = (path, os.stat(path).st_mtime)
    if key not in _CAPS:
        try:
            out = subprocess.run([path, "--help"], capture_output=True,
                                 text=True, timeout=20)
            text = out.stdout + out.stderr
        except (OSError, subprocess.SubprocessError) as exc:
            raise LlamaError(f"could not run {binary}: {exc}") from exc
        caps = set(re.findall(r"(?<![\w-])(--?[a-z][\w-]*)", text))
        if "[on|off|auto]" in text:
            caps.add("fa-value")
        _CAPS[key] = caps
    return _CAPS[key]


# Flags CherryCola sets itself, or that would let the server reach the
# network, run tools, download models or write files outside its log:
# not offered as raw options.
_RESERVED = re.compile(
    r"^--(help|usage|version|cache-list|completion-bash|list-devices|"
    r"model|model-url|alias|host|port|path|api-prefix|api-key.*|ssl-.*|"
    r"hf-.*|docker-repo|log-file|models-.*|jinja|no-jinja|webui.*|"
    r"no-webui|reuse-port|offline|metrics|slots|props|"
    r"chat-template-file|.*-file|.*-url|.*-urls|.*-hf|tools|tools-.*|"
    r"agent|allow-unauthenticated-tools|mcp-.*|ui|ui-.*|connect|"
    r"connect-.*|cors-.*|rpc|log-prompts-dir|slot-save-path|"
    r".*-default|.*-spec|lookup-cache-.*|media-path)$")


@dataclass
class ServerOption:
    names: list[str]  # all spellings, e.g. ["-c", "--ctx-size"]
    flag: str  # the long name we write, e.g. "--ctx-size"
    arg: str  # "" for an on/off switch
    help: str
    section: str
    negated: str = ""  # "--no-perf" when the switch has an off form
    choices: tuple[str, ...] = ()


_OPTIONS: dict[tuple[str, float], list[ServerOption]] = {}


def parse_help(text: str) -> list[ServerOption]:
    """Turn ``llama-server --help`` into options with their help text."""
    opts: list[ServerOption] = []
    section = ""
    for line in text.splitlines():
        head = re.match(r"^----- (.+?) -----$", line.strip())
        if head:
            section = head.group(1)
            continue
        if not line.strip():
            continue
        if line.startswith("-"):
            if len(line) <= 40 or line[39] == " ":
                spec, desc = line[:40].rstrip(), line[40:].strip()
            else:  # a long spec spills past the description column
                m = re.search(r"\s{2,}(\S.*)$", line[40:])
                spec = line[:40 + m.start()] if m else line
                desc = m.group(1) if m else ""
            tokens = [t.strip() for t in re.split(r",\s*(?=-)", spec.strip())]
            names, arg = [], ""
            for tok in tokens:
                first, _, rest = tok.partition(" ")
                if first.startswith("-"):
                    names.append(first)
                    if rest:
                        arg = rest.strip()
            longs = [n for n in names if n.startswith("--")]
            pos = [n for n in longs if not n.startswith("--no-")]
            neg = [n for n in longs if n.startswith("--no-")]
            flag = (pos or longs or names or [""])[0]
            if not flag:
                continue
            choices = ()
            m = re.match(r"^[\[{<](.+?)[\]}>]$", arg)
            if m and ("|" in m.group(1) or "," in m.group(1)) and \
                    ".." not in m.group(1):
                choices = tuple(c.strip() for c in re.split(
                    r"[|,]", m.group(1)) if c.strip())
            opts.append(ServerOption(names, flag, arg, desc, section,
                                     neg[0] if neg and pos else "", choices))
        elif opts and line.startswith(" " * 20):
            more = line.strip()
            if more.startswith("(env:"):
                continue
            opts[-1].help = (opts[-1].help + " " + more).strip()
    return [o for o in opts if not _RESERVED.match(o.flag)]


def server_options(binary: str) -> list[ServerOption]:
    """Every option this llama-server offers, minus the reserved ones."""
    path = shutil.which(binary)
    if path is None:
        raise LlamaError(f"{binary} is not installed")
    key = (path, os.stat(path).st_mtime)
    if key not in _OPTIONS:
        try:
            out = subprocess.run([path, "--help"], capture_output=True,
                                 text=True, timeout=20)
        except (OSError, subprocess.SubprocessError) as exc:
            raise LlamaError(f"could not run {binary}: {exc}") from exc
        _OPTIONS[key] = parse_help(out.stdout + out.stderr)
    return _OPTIONS[key]


def raw_args(raw: dict, options: list[ServerOption]) -> list[str]:
    """``llama_args`` from a profile -> argv, checked against --help."""
    by_name = {}
    for o in options:
        for n in o.names + ([o.negated] if o.negated else []):
            by_name[n] = o
    out = []
    for name, value in raw.items():
        opt = by_name.get(name)
        if opt is None:
            raise LlamaError(f"this llama-server has no option {name}")
        if not opt.arg:
            if value is True or value == "on":
                out.append(opt.flag)
            elif opt.negated:
                out.append(opt.negated)
            continue
        value = str(value)
        if opt.choices and value not in opt.choices:
            raise LlamaError(f"{name} must be one of: "
                             f"{', '.join(opt.choices)}")
        out += [opt.flag, value]
    return out


def _num(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def build_args(path: Path, settings: dict, url: str, caps: set[str],
               plan: Plan | None = None, alias: str = "") -> list[str]:
    """llama-server flags for a model and its load settings."""
    parsed = urllib.parse.urlparse(url)
    args = ["-m", str(path), "--host", parsed.hostname or "127.0.0.1",
            "--port", str(parsed.port or 8080), "--jinja"]
    if alias:
        args += ["--alias", alias]
    s = dict(settings)
    stream = bool(s.get("disk_stream"))
    if stream:
        # Pages the weights that don't fit straight from the file.
        s["use_mmap"], s["use_mlock"] = True, False
        if "gpu_layers" not in s and plan is not None:
            s["gpu_layers"] = plan.gpu_layers
        if "--fit" in caps:
            args += ["--fit", "off"]  # our plan decides, not the fitter
        if "--no-repack" in caps:
            # Repacking copies the weights into RAM, which defeats mmap:
            # streamed weights must stay file pages the kernel can drop.
            args += ["--no-repack"]
        if "--parallel" in caps:
            args += ["--parallel", "1"]  # one context cache, not four
    simple = {"context_length": "-c", "threads": "-t", "batch_size": "-b",
              "ubatch_size": "-ub", "main_gpu": "-mg",
              "rope_freq_base": "--rope-freq-base",
              "rope_freq_scale": "--rope-freq-scale",
              "yarn_orig_ctx": "--yarn-orig-ctx",
              "cpu_moe_layers": "--n-cpu-moe"}
    for key, flag in simple.items():
        if key in s:
            if key.startswith("rope_freq") and not s[key]:
                continue
            if key == "cpu_moe_layers" and flag not in caps:
                raise LlamaError("this llama.cpp is too old for "
                                 "cpu_moe_layers (--n-cpu-moe)")
            args += [flag, _num(s[key])]
    if "gpu_layers" in s:
        ngl = s["gpu_layers"]
        args += ["-ngl", "999" if ngl < 0 else str(ngl)]
    if s.get("tensor_split"):
        args += ["-ts", str(s["tensor_split"])]
    if "kv_cache_type" in s:
        args += ["-ctk", s["kv_cache_type"], "-ctv", s["kv_cache_type"]]
    if "flash_attention" in s or s.get("kv_cache_type", "f16") not in (
            "f16", "bf16"):
        on = s.get("flash_attention", True)
        if "fa-value" in caps:
            args += ["-fa", "on" if on else "off"]
        elif on:
            args += ["-fa"]
    if s.get("rope_scaling"):
        args += ["--rope-scaling", s["rope_scaling"]]
    if "use_mmap" in s or "use_mlock" in s:
        mmap, mlock = s.get("use_mmap", True), s.get("use_mlock", False)
        if "--load-mode" in caps:
            mode = ("mmap+mlock" if mmap and mlock else "mmap" if mmap
                    else "mlock" if mlock else "none")
            args += ["--load-mode", mode]
        else:
            if not mmap:
                args += ["--no-mmap"]
            if mlock:
                args += ["--mlock"]
    if s.get("draft_model"):
        args += ["-md", str(s["draft_model"])]
    return args


def _read_state() -> dict:
    try:
        data = json.loads(_state_file().read_text("utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _alive(pid: int) -> bool:
    if not osdeps.pid_alive(pid):
        return False
    # A reused pid must not be mistaken for our server.
    name = osdeps.process_name(pid)
    return name is None or name.startswith(("llama-server", "python"))


def loaded_model() -> str | None:
    state = _read_state()
    pid = state.get("pid")
    if isinstance(pid, int) and _alive(pid):
        return state.get("model")
    return None


def _spawn(args: list[str], log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as log:
        return subprocess.Popen(
            args, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, **osdeps.popen_detached())


def _health(url: str) -> str:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health",
                                    timeout=3) as resp:
            return "ok" if resp.status == 200 else "loading"
    except urllib.error.HTTPError as exc:
        return "loading" if exc.code == 503 else "down"
    except (urllib.error.URLError, OSError):
        return "down"


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


def _log_tail(n: int = 6) -> str:
    try:
        lines = log_file().read_text("utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def ensure_loaded(cfg: config_mod.Config, model: str, settings: dict,
                  spawn=_spawn, on_status=None,
                  timeout: float = _LOAD_TIMEOUT) -> None:
    """Make llama-server serve ``model`` with ``settings``; restart if needed."""
    path = find_model(cfg, model)
    if path is None:
        raise LlamaError(f"no GGUF model named {model} in {models_dir(cfg)}")
    caps = server_caps(cfg.llama_server)
    plan = None
    if settings.get("disk_stream"):
        plan = make_plan(gguf.read_info(path), settings)
    args = [shutil.which(cfg.llama_server) or cfg.llama_server] + \
        build_args(path, settings, cfg.llamacpp_url, caps, plan, model)
    if settings.get("llama_args"):
        args += raw_args(settings["llama_args"],
                         server_options(cfg.llama_server))
    state = _read_state()
    pid = state.get("pid")
    if isinstance(pid, int) and _alive(pid):
        if state.get("args") == args and _health(cfg.llamacpp_url) == "ok":
            return
        if on_status:
            on_status(f"Reloading {model} with new settings…")
        stop()
    elif _health(cfg.llamacpp_url) != "down":
        raise LlamaError(f"something else is using {cfg.llamacpp_url}; set "
                         f"llamacpp_url to a free port")
    if on_status:
        on_status(f"Loading {model}…")
    try:
        proc = spawn(args, log_file())
    except OSError as exc:
        raise LlamaError(f"could not start llama-server: {exc}") from exc
    _state_file().parent.mkdir(parents=True, exist_ok=True)
    _state_file().write_text(json.dumps(
        {"pid": proc.pid, "model": model, "args": args}), "utf-8")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = _health(cfg.llamacpp_url)
        if status == "ok":
            return
        if proc.poll() is not None:
            break
        time.sleep(0.3)
    stop()
    tail = _log_tail()
    raise LlamaError(f"llama-server could not load {model}"
                     + (f":\n{tail}" if tail else f" (see {log_file()})"))


# ---- planning ----------------------------------------------------------------

@dataclass
class Plan:
    model_gb: float
    gpu_layers: int
    layers: int
    on_gpu_gb: float
    in_ram_gb: float
    from_disk_gb: float  # read from disk for every token
    kv_gb: float
    fits: str  # gpu | ram | disk

    def describe(self, streaming: bool = False) -> str:
        def gb(x: float) -> str:
            return f"{x:.1f} GB" if x >= 1 or x == 0 else \
                f"{x * 1024:.0f} MB"

        where = {"gpu": "fits on the graphics card",
                 "ram": "fits in graphics memory and RAM" if self.gpu_layers
                 else "fits in RAM",
                 "disk": "is bigger than graphics memory and RAM together"}
        lines = [f"Model {gb(self.model_gb)} {where[self.fits]}.",
                 f"GPU layers: {self.gpu_layers} of {self.layers} "
                 f"({gb(self.on_gpu_gb)} on the GPU)",
                 f"In RAM: {gb(self.in_ram_gb)}, context cache: "
                 f"{gb(self.kv_gb)}"]
        if self.fits == "disk":
            secs = self.from_disk_gb / 3.0
            lines.append(
                f"Streams {gb(self.from_disk_gb)} from disk for each "
                f"token: about {secs:.1f} s per token on a 3 GB/s NVMe SSD, "
                f"far slower on a hard disk."
                + ("" if streaming else " Turn on disk_stream to run it."))
        return "\n".join(lines)


def make_plan(info: gguf.ModelInfo, settings: dict,
              ram: int | None = None, vram: int | None = None) -> Plan:
    """Split a model across GPU, RAM and disk (estimates)."""
    from . import hardware
    if ram is None:
        ram = _available_ram()
    if vram is None:
        vram = max((g.vram for g in hardware.detect_gpus()), default=0)
    size = total_size(info.path) if info.path.exists() else info.size
    ctx = settings.get("context_length") or min(info.context_length or 4096,
                                                 8192)
    kv = info.kv_bytes_per_token(settings.get("kv_cache_type", "f16")) * ctx
    layers = sorted(info.layer_bytes.items())
    n_layers = len(layers) or info.block_count
    budget = int(vram * 0.9) - 512 * 1024**2 - kv if vram else 0
    on_gpu, count = 0, 0
    wanted = settings.get("gpu_layers")
    for _, nbytes in layers:
        if wanted is not None and wanted >= 0 and count >= wanted:
            break
        if wanted is None and on_gpu + nbytes > budget:
            break
        on_gpu += nbytes
        count += 1
    rest = max(0, size - on_gpu)
    ram_budget = max(0, ram - int(1.5 * _GB) - (kv if not count else 0))
    in_ram = min(rest, ram_budget)
    from_disk = rest - in_ram
    fits = "gpu" if rest <= info.other_bytes and count == n_layers else \
        "disk" if from_disk > 0 else "ram"
    return Plan(model_gb=size / _GB, gpu_layers=count, layers=n_layers,
                on_gpu_gb=on_gpu / _GB, in_ram_gb=in_ram / _GB,
                from_disk_gb=from_disk / _GB, kv_gb=kv / _GB, fits=fits)


def _available_ram() -> int:
    return osdeps.available_ram() or config_mod.total_ram_bytes() or 0


# ---- getting models ------------------------------------------------------------

def _hf_files(cfg: config_mod.Config, repo: str) -> list[dict]:
    url = f"{cfg.hf_url}/api/models/{repo}/tree/main?recursive=true"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 404):
            raise LlamaError(f"no public Hugging Face repo {repo}") from exc
        raise LlamaError(f"Hugging Face: {exc}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise LlamaError(f"could not reach Hugging Face: {exc}") from exc
    return [f for f in data if isinstance(f, dict)
            and f.get("type") == "file"
            and str(f.get("path", "")).lower().endswith(".gguf")]


def pick_files(files: list[dict], quant: str | None) -> list[str]:
    """The GGUF file(s) for one quantization, split parts included."""
    paths = [f["path"] for f in files
             if not Path(f["path"]).name.lower().startswith("mmproj")]
    if not paths:
        raise LlamaError("that repo has no GGUF files")
    if quant:
        q = quant.upper()
        hits = [p for p in paths if re.search(
            rf"(?<![A-Z0-9]){re.escape(q)}(?![A-Z0-9])", p.upper())]
        if not hits:
            have = sorted({m.group(0) for p in paths for m in [re.search(
                r"(I?Q\d_[A-Z0-9_]+|Q\d_\d|[BF]F?16|F32)", p.upper())] if m})
            raise LlamaError(f"no {q} file there; it has: "
                             f"{', '.join(have) or ', '.join(paths)}")
    else:
        pref = ("Q4_K_M", "Q4_K_S", "Q5_K_M", "Q8_0")
        hits = []
        for q in pref:
            hits = [p for p in paths if q in p.upper()]
            if hits:
                break
        hits = hits or paths[:1]
    first = hits[0]
    split = _SPLIT.search(first)
    if split:
        stem = first[: split.start()]
        return sorted(p for p in hits if p.startswith(stem))
    return [first]


def download(cfg: config_mod.Config, spec: str, on_progress=None) -> str:
    """``owner/repo[:QUANT]`` from Hugging Face; resumes. Returns the name."""
    repo, _, quant = spec.partition(":")
    if repo.count("/") != 1 or not all(
            re.fullmatch(r"[\w.-]+", p) and p.strip(".")
            for p in repo.split("/")):
        raise LlamaError("use owner/repo, for example "
                         "Qwen/Qwen3-4B-GGUF:Q4_K_M")
    files = pick_files(_hf_files(cfg, repo), quant or None)
    dest_dir = models_dir(cfg) / repo.split("/")[1]
    dest_dir.mkdir(parents=True, exist_ok=True)
    for rel in files:
        name = Path(rel).name
        dest = dest_dir / name
        if dest.exists():
            continue
        part = dest.with_name(name + ".part")
        have = part.stat().st_size if part.exists() else 0
        url = f"{cfg.hf_url}/{repo}/resolve/main/{urllib.parse.quote(rel)}"
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"}
                                     if have else {})
        try:
            resp = urllib.request.urlopen(req, timeout=60)
        except urllib.error.HTTPError as exc:
            if exc.code == 416:  # already complete
                part.rename(dest)
                continue
            raise LlamaError(f"download failed: {exc}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise LlamaError(f"download failed: {exc}") from exc
        with resp:
            if resp.status != 206:
                have = 0
            total = have + int(resp.headers.get("Content-Length") or 0)
            with open(part, "ab" if have else "wb") as out:
                done = have
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
                    done += len(chunk)
                    if on_progress:
                        on_progress(name, done, total)
        part.rename(dest)
    first = dest_dir / Path(files[0]).name
    return first.relative_to(models_dir(cfg)).as_posix()[: -len(".gguf")]


def ollama_dirs() -> list[Path]:
    dirs = []
    if os.environ.get("OLLAMA_MODELS"):
        dirs.append(Path(os.environ["OLLAMA_MODELS"]))
    dirs += [Path.home() / ".ollama" / "models",
             Path("/var/lib/ollama/.ollama/models"),
             Path("/var/lib/ollama/models"),
             Path("/usr/share/ollama/.ollama/models")]
    return [d for d in dirs if (d / "manifests").is_dir()]


def import_ollama(cfg: config_mod.Config, model: str) -> str:
    """Link an Ollama model's GGUF blob into the models folder (no copy)."""
    name, _, tag = model.partition(":")
    tag = tag or "latest"
    parts = name.split("/")
    if len(parts) == 1:
        parts = ["library"] + parts
    if not all(re.fullmatch(r"[\w.-]+", p) and p.strip(".")
               for p in parts + [tag]):
        raise LlamaError(f"bad model name {model}")
    for root in ollama_dirs():
        manifest = root / "manifests" / "registry.ollama.ai" / Path(
            *parts) / tag
        try:
            data = json.loads(manifest.read_text("utf-8"))
        except (OSError, ValueError):
            continue
        digest = next((layer.get("digest") for layer in data.get("layers", [])
                       if layer.get("mediaType")
                       == "application/vnd.ollama.image.model"), None)
        if not digest or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise LlamaError(f"{model} has no model file in its manifest")
        blob = root / "blobs" / digest.replace(":", "-")
        try:
            info = gguf.read_info(blob)
        except gguf.GGUFError as exc:
            raise LlamaError(f"{model}: {exc}") from exc
        out_name = f"ollama/{name.replace('/', '-')}-{tag}"
        dest = models_dir(cfg) / f"{out_name}.gguf"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        osdeps.link_file(blob, dest)
        _ = info
        return out_name
    raise LlamaError(f"Ollama has no model {model} downloaded "
                     f"(looked in: {', '.join(map(str, ollama_dirs())) or 'no Ollama model folders found'})")


# ---- requantizing ----------------------------------------------------------------

def quant_name(src_name: str, qtype: str) -> str:
    base = re.sub(r"[-_.]?(I?Q\d(_[A-Z0-9]+)*|[BF]F?16|F32|MXFP4_MOE)$", "",
                  src_name, flags=re.I)
    return f"{base}-{qtype}"


def quantize(cfg: config_mod.Config, src: str, qtype: str,
             out: str | None = None, imatrix: str | None = None,
             allow_requantize: bool = False, threads: int | None = None,
             run=subprocess.run) -> str:
    """Write a new GGUF in another quantization. Returns its name."""
    qtype = qtype.upper()
    if qtype not in QUANT_TYPES:
        raise LlamaError(f"unknown type {qtype}; pick one of: "
                         f"{', '.join(QUANT_TYPES)}")
    path = find_model(cfg, src)
    if path is None:
        raise LlamaError(f"no GGUF model named {src} in {models_dir(cfg)}")
    info = gguf.read_info(path)
    if info.file_type == qtype:
        raise LlamaError(f"{src} is already {qtype}")
    if info.quantized and not allow_requantize:
        raise LlamaError(
            f"{src} is already quantized ({info.file_type}). Quantizing it "
            f"again loses more quality than starting from an F16/BF16 file. "
            f"Add --allow-requantize to do it anyway")
    if qtype in NEEDS_IMATRIX and not imatrix:
        raise LlamaError(f"{qtype} needs an importance matrix: "
                         f"cherrycola imatrix {src} TEXT_FILE first, then "
                         f"pass --imatrix")
    binary = shutil.which(cfg.llama_quantize)
    if binary is None:
        raise LlamaError(f"{cfg.llama_quantize} is not installed "
                         f"(it comes with llama.cpp)")
    name = out or quant_name(path.relative_to(models_dir(cfg)).as_posix()
                             [: -len(".gguf")] if path.is_relative_to(
                                 models_dir(cfg)) else path.stem, qtype)
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise LlamaError("--out is a name inside the models folder")
    dest = models_dir(cfg) / f"{name}.gguf"
    if dest.exists():
        raise LlamaError(f"{name} already exists")
    dest.parent.mkdir(parents=True, exist_ok=True)
    args = [binary]
    if allow_requantize:
        args.append("--allow-requantize")
    if imatrix:
        args += ["--imatrix", str(Path(imatrix).expanduser())]
    args += [str(path), str(dest), qtype]
    if threads:
        args.append(str(threads))
    result = run(args)
    if result.returncode != 0:
        try:
            dest.unlink()
        except OSError:
            pass
        raise LlamaError(f"llama-quantize failed (exit {result.returncode})")
    return name


def imatrix(cfg: config_mod.Config, model: str, text_file: str,
            settings: dict | None = None, run=subprocess.run) -> Path:
    """Compute an importance matrix from a text file with llama-imatrix."""
    path = find_model(cfg, model)
    if path is None:
        raise LlamaError(f"no GGUF model named {model}")
    binary = shutil.which(cfg.llama_imatrix)
    if binary is None:
        raise LlamaError(f"{cfg.llama_imatrix} is not installed "
                         f"(it comes with llama.cpp)")
    out = path.with_name(path.stem + ".imatrix.gguf")
    args = [binary, "-m", str(path), "-f", str(Path(text_file).expanduser()),
            "-o", str(out)]
    settings = settings or {}
    if "gpu_layers" in settings:
        args += ["-ngl", str(max(settings["gpu_layers"], 0)
                             if settings["gpu_layers"] >= 0 else 999)]
    if "context_length" in settings:
        args += ["-c", str(settings["context_length"])]
    if run(args).returncode != 0:
        raise LlamaError("llama-imatrix failed")
    return out

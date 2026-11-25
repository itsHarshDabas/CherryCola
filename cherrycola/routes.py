"""Getting a model through the engine you picked, and loading it only for a moment.

``pull SPEC`` takes a model in any of the forms people paste:

- an Ollama tag (``qwen3:4b``, ``user/model:tag``)
- a Hugging Face GGUF repo (``Qwen/Qwen3-4B-GGUF:Q4_K_M``, ``hf.co/...``,
  ``https://huggingface.co/...``)
- a direct ``https://`` link to a ``.gguf`` file
- a ``.gguf`` file already on this computer (or, for Ollama, a model folder)

and gets it the way the selected engine (the "route") wants: Ollama pulls it
or imports the file, llama.cpp downloads the GGUF into its models folder (or
links Ollama's copy), and LM Studio uses its ``lms`` command.

``unload`` frees a model's memory on any engine, and ``was_loaded`` tells a
temporary run (``ask --temporary``) whether it should unload afterwards.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import backends as be
from . import config as config_mod

__all__ = ["Spec", "classify", "pull", "unload", "was_loaded"]

_HF_HOSTS = ("hf.co/", "huggingface.co/")
_NAME = re.compile(r"[\w.-]+")
_OLLAMA_TAG = re.compile(r"^[\w.-]+(/[\w.-]+)?(:[\w.-]+)?$")
_MAX_NAME = 200
# The default Ollama models, as the same model in GGUF on Hugging Face, so a
# bare `pull` works on llama.cpp and LM Studio too.
GGUF_EQUIVALENTS = {
    "qwen3:1.7b": "Qwen/Qwen3-1.7B-GGUF:Q8_0",
    "qwen3:4b": "Qwen/Qwen3-4B-GGUF:Q4_K_M",
    "qwen3:8b": "Qwen/Qwen3-8B-GGUF:Q4_K_M",
}


@dataclass(frozen=True)
class Spec:
    """What the user gave, sorted into one kind.

    kind: "ollama" (a registry tag), "hf" (owner/repo, quant optional),
    "url" (an https link to a .gguf file), "file" (a .gguf on disk) or
    "dir" (a model folder on disk, Ollama only).
    """
    kind: str
    value: str
    quant: str = ""

    @property
    def hf_spec(self) -> str:
        return f"{self.value}:{self.quant}" if self.quant else self.value


def _valid_repo(repo: str) -> bool:
    parts = repo.split("/")
    return len(parts) == 2 and all(_NAME.fullmatch(p) and p.strip(".")
                                   for p in parts)


def _hf_from_url(text: str) -> Spec | None:
    """https://huggingface.co/OWNER/REPO[/...] or hf.co/OWNER/REPO[:Q]."""
    rest = re.sub(r"^https://", "", text)
    for host in _HF_HOSTS:
        if rest.startswith(host):
            rest = rest[len(host):]
            break
    else:
        return None
    path, _, quant = rest.partition(":")
    parts = [p for p in path.split("/") if p]
    if len(parts) < 2:
        return None
    repo = "/".join(parts[:2])
    if not _valid_repo(repo):
        return None
    # A link straight to one file: .../resolve/main/name-Q4_K_M.gguf
    if len(parts) > 2 and parts[-1].lower().endswith(".gguf") and \
            text.startswith("https://"):
        return Spec("url", text)
    return Spec("hf", repo, quant.strip())


def classify(text: str, backend: str) -> Spec:
    """Sort a pasted model into a Spec. Raises BackendError when it's none."""
    text = (text or "").strip()
    if not text or len(text) > 2048 or any(c in text for c in "\n\r\0"):
        raise be.BackendError("give a model name, Hugging Face repo, "
                              "link or file")
    local = Path(os.path.expanduser(text))
    if text.startswith(("/", "~", ".")) or re.match(r"^[A-Za-z]:[\\/]", text) \
            or local.exists():
        if local.is_file() and local.suffix.lower() == ".gguf":
            return Spec("file", str(local.resolve()))
        if local.is_dir():
            return Spec("dir", str(local.resolve()))
        raise be.BackendError(f"{text} is not a .gguf file or model folder")
    if text.startswith("http://"):
        raise be.BackendError("only https:// links are downloaded")
    hf = _hf_from_url(text)
    if hf is not None:
        return hf
    if text.startswith("https://"):
        name = urllib.parse.urlsplit(text).path.rsplit("/", 1)[-1]
        if not name.lower().endswith(".gguf"):
            raise be.BackendError("a link must point at a .gguf file")
        return Spec("url", text)
    if not _OLLAMA_TAG.fullmatch(text) or len(text) > _MAX_NAME:
        raise be.BackendError(f"{text!r} doesn't look like a model name")
    path, _, quant = text.partition(":")
    # Ollama has namespaces (user/model) too, so on Ollama only repos that
    # are clearly GGUF go to Hugging Face.
    if "/" in path and _valid_repo(path) and (
            backend != "ollama" or "gguf" in path.lower()):
        return Spec("hf", path, quant)
    return Spec("ollama", text)


# ---- pulling ------------------------------------------------------------------

def _which(name: str) -> str | None:
    return shutil.which(name)


def _run(args: list[str], timeout: float | None = None) -> None:
    """Run a helper command, passing its output through (it shows progress)."""
    try:
        proc = subprocess.run(args, check=False, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise be.BackendError(f"{args[0]}: {exc}") from exc
    if proc.returncode != 0:
        raise be.BackendError(f"{' '.join(args[:2])} failed "
                              f"(exit {proc.returncode})")


def _safe_name(text: str) -> str:
    stem = re.sub(r"\.gguf$", "", Path(text).name, flags=re.IGNORECASE)
    stem = re.sub(r"[^\w.-]+", "-", stem).strip(".-").lower()
    return (stem or "model")[:80]


def download_url(cfg: config_mod.Config, url: str, on_progress=None) -> Path:
    """Download an https .gguf link into the llama.cpp models folder (resumes)."""
    from . import llamacpp
    name = urllib.parse.unquote(urllib.parse.urlsplit(url).path
                                .rsplit("/", 1)[-1])
    if not name.lower().endswith(".gguf") or "/" in name or "\\" in name \
            or name.startswith("."):
        raise be.BackendError("a link must point at a .gguf file")
    dest_dir = llamacpp.models_dir(cfg) / "downloads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    if dest.exists():
        return dest
    part = dest.with_name(name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(
        url, headers={"Range": f"bytes={have}-"} if have else {})
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and have:
            part.rename(dest)
            return dest
        raise be.BackendError(f"download failed: {exc}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise be.BackendError(f"download failed: {exc}") from exc
    with resp:
        if not resp.geturl().startswith("https://"):
            raise be.BackendError("the link redirected away from https")
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
    with open(part, "rb") as fh:
        if fh.read(4) != b"GGUF":
            part.unlink()
            raise be.BackendError(f"{name} is not a GGUF file")
    part.rename(dest)
    return dest


def _ollama_create(name: str, source: str) -> str:
    """``ollama create NAME`` from a GGUF file or model folder."""
    if _which("ollama") is None:
        raise be.BackendError("importing a file into Ollama needs the "
                              "ollama command")
    with tempfile.TemporaryDirectory() as tmp:
        modelfile = Path(tmp) / "Modelfile"
        # Forward slashes work on Windows too and need no escaping.
        modelfile.write_text(f"FROM {json.dumps(Path(source).as_posix())}\n",
                             encoding="utf-8")
        _run(["ollama", "create", name, "-f", str(modelfile)])
    return name


def _pull_ollama(cfg, spec: Spec, on_progress) -> str:
    if spec.kind == "ollama":
        be.pull_model("ollama", cfg, spec.value, on_progress)
        return spec.value
    if spec.kind == "hf":
        # Ollama pulls GGUF repos from Hugging Face itself.
        tag = f"hf.co/{spec.hf_spec}"
        be.pull_model("ollama", cfg, tag, on_progress)
        return tag
    if spec.kind == "url":
        path = download_url(cfg, spec.value, _bytes_progress(on_progress))
        return _ollama_create(_safe_name(path.name), str(path))
    return _ollama_create(_safe_name(spec.value), spec.value)


def _pull_llamacpp(cfg, spec: Spec, on_progress) -> str:
    from . import llamacpp
    try:
        if spec.kind == "hf":
            return llamacpp.download(cfg, spec.hf_spec,
                                     _bytes_progress(on_progress))
        if spec.kind == "ollama":
            try:
                return llamacpp.import_ollama(cfg, spec.value)
            except llamacpp.LlamaError:
                if spec.value not in GGUF_EQUIVALENTS:
                    raise
            return llamacpp.download(cfg, GGUF_EQUIVALENTS[spec.value],
                                     _bytes_progress(on_progress))
        if spec.kind == "url":
            path = download_url(cfg, spec.value, _bytes_progress(on_progress))
            return path.relative_to(llamacpp.models_dir(cfg)).as_posix()[:-5]
        if spec.kind == "file":
            path = Path(spec.value)
            root = llamacpp.models_dir(cfg)
            if path.is_relative_to(root):
                return path.relative_to(root).as_posix()[:-5]
            llamacpp.add_folder(cfg, str(path.parent))
            return f"{_folder_prefix(cfg, path.parent)}{path.stem}"
    except llamacpp.LlamaError as exc:
        raise be.BackendError(str(exc)) from exc
    raise be.BackendError("llama.cpp runs single .gguf files, not folders")


def _folder_prefix(cfg, folder: Path) -> str:
    """How list_local names models from an added folder."""
    from . import llamacpp
    local = llamacpp.list_local(cfg)
    for name, path in local.items():
        if path.parent == folder:
            return name[: -len(path.stem)]
    return ""


def _pull_lmstudio(cfg, spec: Spec) -> str:
    lms = _which("lms")
    if lms is None:
        raise be.BackendError("LM Studio downloads with its lms command, "
                              "which isn't installed (LM Studio > Developer)")
    if spec.kind == "file":
        _run([lms, "import", "-y", spec.value])
        return Path(spec.value).stem
    if spec.kind == "url":
        path = download_url(cfg, spec.value)
        _run([lms, "import", "-y", str(path)])
        return path.stem
    if spec.kind == "dir":
        raise be.BackendError("LM Studio imports single .gguf files")
    what = spec.hf_spec if spec.kind == "hf" else \
        GGUF_EQUIVALENTS.get(spec.value, spec.value)
    _run([lms, "get", "-y", what])
    return spec.value


def _bytes_progress(on_progress):
    if on_progress is None:
        return None

    def report(name, done, total):
        on_progress({"status": f"downloading {name}", "completed": done,
                     "total": total})
    return report


def pull(cfg: config_mod.Config, text: str, backend: str | None = None,
         on_progress=None) -> str:
    """Get a model through the selected engine. Returns the name to use."""
    backend = backend or be.resolve_backend(cfg)
    spec = classify(text, backend)
    if backend == "ollama":
        return _pull_ollama(cfg, spec, on_progress)
    if backend == "llamacpp":
        return _pull_llamacpp(cfg, spec, on_progress)
    return _pull_lmstudio(cfg, spec)


# ---- loading only for a moment --------------------------------------------------

def _post(url: str, payload: dict, timeout: float = 30) -> None:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=timeout).close()
    except (urllib.error.URLError, OSError) as exc:
        raise be.BackendError(str(exc) or type(exc).__name__) from exc


def loaded_models(backend: str, cfg: config_mod.Config) -> list[str]:
    """Models in memory right now (not just downloaded)."""
    if backend == "llamacpp":
        from . import llamacpp
        model = llamacpp.loaded_model()
        return [model] if model else []
    if backend == "ollama":
        try:
            payload = be.http_get_json(be._url(cfg.ollama_url, "/api/ps"))
            return [m["name"] for m in payload.get("models", [])]  # type: ignore[union-attr]
        except (be.BackendError, KeyError, TypeError, AttributeError):
            return []
    try:
        payload = be.http_get_json(be._url(cfg.lmstudio_url, "/api/v0/models"))
        return [m["id"] for m in payload.get("data", [])  # type: ignore[union-attr]
                if m.get("state") == "loaded"]
    except (be.BackendError, KeyError, TypeError, AttributeError):
        return []


def was_loaded(backend: str, cfg: config_mod.Config, model: str) -> bool:
    return be.model_listed(backend, model, loaded_models(backend, cfg))


def unload(backend: str, cfg: config_mod.Config, model: str | None) -> bool:
    """Free the model's memory. Returns False when nothing was loaded."""
    if backend == "llamacpp":
        from . import llamacpp
        return llamacpp.stop()
    if backend == "ollama":
        targets = [model] if model else loaded_models("ollama", cfg)
        for name in targets:
            # keep_alive 0 with no prompt: Ollama drops the model now.
            _post(be._url(cfg.ollama_url, "/api/generate"),
                  {"model": name, "keep_alive": 0})
        return bool(targets)
    lms = _which("lms")
    if lms is None:
        raise be.BackendError("LM Studio unloads with its lms command, "
                              "which isn't installed")
    _run([lms, "unload", model] if model else [lms, "unload", "--all"],
         timeout=60)
    return True

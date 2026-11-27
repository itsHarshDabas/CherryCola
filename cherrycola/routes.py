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
from . import osdeps

__all__ = ["Spec", "classify", "engine_for", "pull", "unload", "was_loaded"]

_HF_HOSTS = ("hf.co/", "huggingface.co/")
_NAME = re.compile(r"[\w.-]+")
_OLLAMA_TAG = re.compile(r"^[\w.-]+(/[\w.-]+)?(:[\w.-]+)?$")
_MAX_NAME = 200


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
    return osdeps.which(name)


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
        magic = fh.read(4)
    # Windows can't delete a file that is still open, so check after closing it.
    if magic != b"GGUF":
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
        try:
            return _ollama_create(_safe_name(path.name), str(path))
        finally:
            path.unlink(missing_ok=True)  # Ollama has its own copy now
    raise be.BackendError(f"{NO_COPY}, and Ollama can only run a model "
                          f"after copying it into its own store. Run "
                          f"{Path(spec.value).name} with the llama.cpp "
                          f"engine instead (it uses the file in place)")


# The recommended models (hardware.CATALOG, Ollama names) as GGUF repos on
# Hugging Face, so llama.cpp can download them without Ollama.
GGUF_REPOS = {
    "qwen3:0.6b": "unsloth/Qwen3-0.6B-GGUF",
    "qwen3:1.7b": "unsloth/Qwen3-1.7B-GGUF",
    "qwen3:4b": "unsloth/Qwen3-4B-GGUF",
    "qwen3:8b": "unsloth/Qwen3-8B-GGUF",
    "qwen3:14b": "unsloth/Qwen3-14B-GGUF",
    "qwen3:30b": "unsloth/Qwen3-30B-A3B-GGUF",
    "qwen3:32b": "unsloth/Qwen3-32B-GGUF",
    "gemma3:4b": "unsloth/gemma-3-4b-it-GGUF",
    "gemma3:12b": "unsloth/gemma-3-12b-it-GGUF",
    "qwen2.5-coder:7b": "unsloth/Qwen2.5-Coder-7B-Instruct-GGUF",
    "qwen2.5-coder:14b": "unsloth/Qwen2.5-Coder-14B-Instruct-GGUF",
    "gpt-oss:20b": "unsloth/gpt-oss-20b-GGUF",
}


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
                repo = GGUF_REPOS.get(spec.value)
                if repo is None:
                    raise
            # Not in Ollama: get the same model from Hugging Face.
            return llamacpp.download(cfg, repo, _bytes_progress(on_progress))
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
        if spec.kind == "dir":
            # A folder of .gguf files: searched where it is, like the app's
            # Model folders. The first model in it becomes the one to use.
            folder, count = llamacpp.add_folder(cfg, spec.value)
            if not count:
                raise llamacpp.LlamaError(f"{spec.value} has no .gguf files")
            names = [n for n, p in llamacpp.list_local(cfg).items()
                     if p.is_relative_to(folder)]
            return sorted(names)[0]
    except llamacpp.LlamaError as exc:
        raise be.BackendError(str(exc)) from exc
    raise be.BackendError("llama.cpp runs .gguf files")


def _pull_airllm(cfg, spec: Spec) -> str:
    """AirLLM runs Hugging Face models: a repo, or a model folder in place."""
    from . import airllm
    try:
        if spec.kind == "dir":
            return airllm.add_folder(cfg, spec.value)
        if spec.kind == "hf" and not spec.quant:
            return airllm.download(cfg, spec.value)
        if spec.kind == "file":
            return airllm.add_folder(cfg, spec.value)  # explains GGUF
    except airllm.AirError as exc:
        raise be.BackendError(str(exc)) from exc
    raise be.BackendError("AirLLM runs Hugging Face models: give owner/repo "
                          "(for example Qwen/Qwen2.5-7B-Instruct) or a model "
                          "folder; .gguf files and Ollama tags run with the "
                          "llama.cpp or Ollama engines")


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
    if spec.kind in ("file", "dir", "url"):
        # lms import copies into LM Studio's folder: engine_for sends these
        # to llama.cpp instead.
        raise be.BackendError(f"{NO_COPY}: run it with the llama.cpp engine "
                              f"(LM Studio would import a copy)")
    _run([lms, "get", "-y", spec.hf_spec if spec.kind == "hf" else spec.value])
    return spec.value


def _bytes_progress(on_progress):
    if on_progress is None:
        return None

    def report(name, done, total):
        on_progress({"status": f"downloading {name}", "completed": done,
                     "total": total})
    return report


NO_COPY = ("CherryColaLLM never copies model files: it runs them where "
           "they are")


def engine_for(cfg: config_mod.Config, spec: Spec, backend: str) -> str:
    """The engine that should get ``spec``. Raises BackendError when none
    can use it without copying it.

    Model files and folders on this computer are used where they are, never
    copied: a .gguf file (or a folder of them) runs with llama.cpp, which
    ships with CherryColaLLM, and a Hugging Face model folder (config.json
    + .safetensors) with AirLLM. Ollama and LM Studio only run models in
    their own stores, so they would have to copy the file in, which takes as
    long as a download and doubles the disk space; they are skipped. A link
    to a .gguf is downloaded once, straight into llama.cpp's folder.
    """
    from . import airllm
    llama = be.server_reachable("llamacpp", cfg)
    if spec.kind in ("file", "url") or (
            spec.kind == "dir" and _has_gguf(Path(spec.value))):
        if llama:
            return "llamacpp"
        if spec.kind == "url" and backend == "ollama":
            return backend  # no llama.cpp: Ollama downloads it itself
        raise be.BackendError(
            f"{NO_COPY}, and .gguf files run in place with llama.cpp, "
            f"which isn't installed here. It comes with CherryColaLLM's "
            f"packages; or install llama.cpp yourself")
    if spec.kind == "dir":
        if not airllm.is_model_folder(Path(spec.value)):
            raise be.BackendError(f"{spec.value} has no .gguf files and "
                                  f"isn't a Hugging Face model folder")
        if not airllm.installed(cfg):
            raise be.BackendError(
                f"{spec.value} is a Hugging Face model folder, which runs in "
                f"place with AirLLM. Install it first: cherrycolallm airllm "
                f"install")
        return "airllm"
    return backend


def _has_gguf(folder: Path) -> bool:
    return folder.is_dir() and any(folder.rglob("*.gguf"))


def pull(cfg: config_mod.Config, text: str, backend: str | None = None,
         on_progress=None) -> str:
    """Get a model through the given engine. Returns the name to use.

    Callers that want a local file to run in place pick the engine with
    ``engine_for`` first; this uses the engine it is given.
    """
    backend = backend or be.resolve_backend(cfg)
    spec = classify(text, backend)
    if backend == "ollama":
        return _pull_ollama(cfg, spec, on_progress)
    if backend == "llamacpp":
        return _pull_llamacpp(cfg, spec, on_progress)
    if backend == "airllm":
        return _pull_airllm(cfg, spec)
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
    if backend in ("llamacpp", "airllm"):
        from . import airllm, llamacpp
        engine = llamacpp if backend == "llamacpp" else airllm
        model = engine.loaded_model()
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
    if backend == "airllm":
        from . import airllm
        return airllm.stop()
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

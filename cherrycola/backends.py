"""HTTP backends: Ollama and LM Studio (OpenAI-compatible). Stdlib only."""

from __future__ import annotations

import http.client
import json
import shutil
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass

from . import config as config_mod
from . import profiles

__all__ = [
    "BackendError",
    "BackendResult",
    "ThinkStripper",
    "check_model_installed",
    "detect_backend",
    "http_get_json",
    "iter_chat_chunks",
    "list_models",
    "prepare",
    "pull_model",
    "resolve_backend",
    "resolve_model",
    "running_model",
    "server_reachable",
]

_TIMEOUT = 10

SYSTEM_PROMPT = (
    "You are CherryColaLLM, the local assistant built into GraceOS. "
    "Answer briefly and plainly."
)


class BackendError(Exception):
    """Expected backend failure (server down, HTTP error, bad payload)."""


@dataclass
class BackendResult:
    backend: str
    reachable: bool


class ThinkStripper:
    """Strip <think>...</think> from a stream, chunk by chunk.

    Tags may split across chunks, so hold a small tail buffer and only
    emit text once it cannot become part of a tag.
    """

    def __init__(self, keep: bool = False) -> None:
        self.keep = keep
        self._buf = ""
        self._in_think = False

    def feed(self, chunk: str) -> str:
        if self.keep:
            return chunk
        self._buf += chunk
        out: list[str] = []
        while True:
            if self._in_think:
                end = self._buf.find("</think>")
                if end == -1:
                    tail = len("</think>") - 1
                    if len(self._buf) > tail:
                        self._buf = self._buf[-tail:]
                    break
                self._buf = self._buf[end + len("</think>"):]
                self._in_think = False
                continue
            start = self._buf.find("<think>")
            if start == -1:
                # Hold back any tail that could start a tag.
                held = ""
                for tag in ("<think>", "</think>"):
                    for n in range(min(len(self._buf), len(tag) - 1), 0, -1):
                        if tag.startswith(self._buf[-n:]):
                            if len(self._buf[-n:]) > len(held):
                                held = self._buf[-n:]
                if held:
                    cut = len(self._buf) - len(held)
                    out.append(self._buf[:cut])
                    self._buf = held
                    break
                # A lone trailing "<" could also start a tag.
                if self._buf.endswith("<"):
                    out.append(self._buf[:-1])
                    self._buf = "<"
                    break
                out.append(self._buf)
                self._buf = ""
                break
            # Found "<think>": emit text before it, enter thinking.
            head = self._buf[:start]
            # Keep a trailing "<" of head? No: "<think>" matched fully,
            # so head is safe to emit.
            out.append(head)
            self._buf = self._buf[start + len("<think>"):]
            self._in_think = True
        return "".join(out)

    def flush(self) -> str:
        if self.keep:
            rest, self._buf = self._buf, ""
            return rest
        if self._in_think:
            self._buf = ""
            self._in_think = False
            return ""
        rest, self._buf = self._buf, ""
        return rest


def _url(base: str, path: str) -> str:
    return base.rstrip("/") + path


def http_get_json(url: str, timeout: float = _TIMEOUT) -> object:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, socket.timeout, OSError,
            http.client.HTTPException, ValueError) as exc:
        raise BackendError(str(exc) or type(exc).__name__) from exc


def server_reachable(backend: str, cfg: config_mod.Config) -> bool:
    if backend == "llamacpp":
        # llama-server starts per model, on demand: "reachable" means the
        # engine is installed.
        return shutil.which(cfg.llama_server) is not None
    base = cfg.ollama_url if backend == "ollama" else cfg.lmstudio_url
    probe = "/api/version" if backend == "ollama" else "/v1/models"
    try:
        http_get_json(_url(base, probe), timeout=3)
    except BackendError:
        return False
    return True


def detect_backend(cfg: config_mod.Config) -> str:
    """auto: Ollama if the CLI exists or its server answers, else LM Studio."""
    if shutil.which("ollama") is not None:
        return "ollama"
    if server_reachable("ollama", cfg):
        return "ollama"
    if server_reachable("lmstudio", cfg):
        return "lmstudio"
    return "ollama"


def resolve_backend(cfg: config_mod.Config) -> str:
    if cfg.backend == "auto":
        return detect_backend(cfg)
    return cfg.backend


def resolve_model(backend: str, cfg: config_mod.Config) -> str:
    if cfg.model:
        return cfg.model
    if backend in ("lmstudio", "llamacpp"):
        try:
            models = list_models(backend, cfg)
        except BackendError:
            models = []
        if models:
            return models[0]
    return config_mod.default_model_for_ram(config_mod.total_ram_bytes())


def list_models(backend: str, cfg: config_mod.Config) -> list[str]:
    if backend == "llamacpp":
        from . import llamacpp
        return list(llamacpp.list_local(cfg))
    if backend == "ollama":
        try:
            payload = http_get_json(_url(cfg.ollama_url, "/api/tags"))
        except BackendError as exc:
            raise BackendError(f"ollama server not reachable: {exc}") from exc
        try:
            return [m["name"] for m in payload["models"]]  # type: ignore[index]
        except (KeyError, TypeError) as exc:
            raise BackendError(
                f"bad response from {cfg.ollama_url}/api/tags") from exc
    try:
        payload = http_get_json(_url(cfg.lmstudio_url, "/v1/models"))
    except BackendError as exc:
        raise BackendError(f"lm studio server not reachable: {exc}") from exc
    try:
        return [m["id"] for m in payload["data"]]  # type: ignore[index]
    except (KeyError, TypeError) as exc:
        raise BackendError(
            f"bad response from {cfg.lmstudio_url}/v1/models") from exc


def running_model(backend: str, cfg: config_mod.Config) -> str | None:
    if backend == "llamacpp":
        from . import llamacpp
        return llamacpp.loaded_model()
    if backend == "ollama":
        try:
            payload = http_get_json(_url(cfg.ollama_url, "/api/ps"))
        except BackendError:
            return None
        try:
            models = payload["models"]  # type: ignore[index]
            if models:
                return models[0]["name"]  # type: ignore[index]
        except (KeyError, TypeError, IndexError):
            return None
        return None
    try:
        models = list_models("lmstudio", cfg)
    except BackendError:
        return None
    return models[0] if models else None


def check_model_installed(backend: str, model: str,
                          cfg: config_mod.Config) -> bool:
    try:
        return model in list_models(backend, cfg)
    except BackendError:
        return False


@dataclass
class ModelLimits:
    """What a model itself supports, read from its own metadata."""
    context_length: int = 0  # trained context window, 0 = unknown
    block_count: int = 0     # transformer layers
    expert_count: int = 0    # MoE experts, 0 = dense
    source: str = ""         # where the numbers came from

    @property
    def max_context(self) -> int:
        """The most context worth offering: trained length, or 4x with
        RoPE/YaRN scaling (beyond that quality falls apart)."""
        return self.context_length * 4


def _post_json(url: str, payload: dict, timeout: float) -> object:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, socket.timeout, OSError,
            http.client.HTTPException, ValueError) as exc:
        raise BackendError(str(exc) or type(exc).__name__) from exc


def _from_gguf_keys(meta: dict) -> ModelLimits:
    arch = str(meta.get("general.architecture", ""))

    def num(name: str) -> int:
        value = meta.get(f"{arch}.{name}")
        return int(value) if isinstance(value, (int, float)) else 0
    return ModelLimits(num("context_length"), num("block_count"),
                       num("expert_count"))


def model_limits(backend: str, cfg: config_mod.Config, model: str,
                 timeout: float = 3) -> ModelLimits:
    """Read the model's trained context, layers and experts from whichever
    engine has it. Unknown numbers stay 0; this never raises."""
    if not model:
        return ModelLimits()
    from . import gguf, llamacpp
    path = llamacpp.find_model(cfg, model)
    if path is not None:
        try:
            info = gguf.read_info(path)
            return ModelLimits(info.context_length, info.block_count,
                               info.expert_count, "gguf")
        except gguf.GGUFError:
            pass
    if backend == "ollama":
        try:
            data = _post_json(_url(cfg.ollama_url, "/api/show"),
                              {"model": model}, timeout)
            meta = data.get("model_info") or {}  # type: ignore[union-attr]
        except (BackendError, AttributeError):
            return ModelLimits()
        limits = _from_gguf_keys(meta)
        limits.source = "ollama"
        return limits
    if backend == "lmstudio":
        # LM Studio's own REST API (0.3+) lists max_context_length.
        try:
            data = http_get_json(_url(cfg.lmstudio_url, "/api/v0/models"),
                                 timeout=timeout)
            rows = data.get("data") or []  # type: ignore[union-attr]
        except (BackendError, AttributeError):
            return ModelLimits()
        for row in rows:
            if isinstance(row, dict) and row.get("id") == model:
                ctx = row.get("max_context_length")
                return ModelLimits(int(ctx) if isinstance(ctx, int) else 0,
                                   source="lmstudio")
    return ModelLimits()


def with_system_prompt(messages: list[dict], settings: dict) -> list[dict]:
    """Swap in the model's own system prompt, when it has one."""
    prompt = settings.get("system_prompt")
    if not prompt:
        return messages
    if messages and messages[0].get("role") == "system" and \
            messages[0].get("content") == SYSTEM_PROMPT:
        return [{"role": "system", "content": prompt}] + messages[1:]
    if messages and messages[0].get("role") == "system":
        # Web answers carry sources in the system prompt; keep them.
        return [{"role": "system", "content": prompt + "\n\n"
                 + messages[0]["content"]}] + messages[1:]
    return [{"role": "system", "content": prompt}] + messages


def prepare(backend: str, cfg: config_mod.Config, model: str,
            settings: dict, on_status=None) -> None:
    """Load the model with its settings where the engine needs that first."""
    if backend != "llamacpp":
        return
    from . import llamacpp
    try:
        llamacpp.ensure_loaded(cfg, model, settings, on_status=on_status)
    except llamacpp.LlamaError as exc:
        raise BackendError(str(exc)) from exc


def _open_chat_stream(backend: str, cfg: config_mod.Config,
                      model: str, messages: list[dict[str, str]],
                      settings: dict):
    body: dict = {"model": model, "stream": True,
                  "messages": with_system_prompt(messages, settings)}
    if backend == "ollama":
        url = _url(cfg.ollama_url, "/api/chat")
        body.update(profiles.ollama_request(settings)[0])
    elif backend == "llamacpp":
        url = _url(cfg.llamacpp_url, "/v1/chat/completions")
        body.update(profiles.llamacpp_request(settings))
    else:
        url = _url(cfg.lmstudio_url, "/v1/chat/completions")
        body.update(profiles.lmstudio_request(settings)[0])
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        # Loading a model on a slow CPU can take minutes before the first byte.
        return urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("error", "")
            if isinstance(detail, dict):
                detail = detail.get("message", "")
        except (ValueError, OSError, AttributeError):
            pass
        raise BackendError(str(detail or exc)) from exc
    except (urllib.error.URLError, socket.timeout, OSError,
            http.client.HTTPException) as exc:
        raise BackendError(str(exc) or type(exc).__name__) from exc


def _iter_ollama_text(resp):
    buf = b""
    for raw in resp:
        buf += bytes(raw)
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line.decode("utf-8"))
            except ValueError:
                continue
            if obj.get("error"):
                raise BackendError(str(obj["error"]))
            content = obj.get("message", {}).get("content", "")
            if content:
                yield content
            if obj.get("done"):
                return
    if buf.strip():
        try:
            obj = json.loads(buf.strip().decode("utf-8"))
        except ValueError:
            return
        content = obj.get("message", {}).get("content", "")
        if content:
            yield content


def _iter_lmstudio_text(resp):
    buf = ""
    thinking = False
    for raw in resp:
        buf += bytes(raw).decode("utf-8", errors="replace")
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                return
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            if obj.get("error"):
                err = obj["error"]
                msg = err.get("message", err) if isinstance(err, dict) else err
                raise BackendError(str(msg))
            try:
                d = obj["choices"][0]["delta"]
                delta = d.get("content") or ""
                reasoning = d.get("reasoning_content") or ""
            except (KeyError, TypeError, IndexError, AttributeError):
                continue
            # llama-server sends thinking separately; wrap it in <think>
            # so ThinkStripper and --show-thinking treat it as usual.
            if reasoning:
                if not thinking:
                    thinking = True
                    yield "<think>"
                yield reasoning
            if delta:
                if thinking:
                    thinking = False
                    yield "</think>"
                yield delta


def iter_chat_chunks(backend, cfg, model, messages, settings=None,
                     on_status=None):
    """Yield text chunks from a streaming chat request.

    ``settings`` are the model's effective settings (profiles.effective);
    None reads them from the profiles file.
    """
    if settings is None:
        settings = profiles.effective(model)
    prepare(backend, cfg, model, settings, on_status)
    with _open_chat_stream(backend, cfg, model, messages, settings) as resp:
        if backend == "ollama":
            yield from _iter_ollama_text(resp)
        else:
            yield from _iter_lmstudio_text(resp)


def remove_model(backend, cfg, model) -> None:
    """Ollama: DELETE /api/delete. llama.cpp: delete the GGUF file(s)."""
    if backend == "llamacpp":
        from . import llamacpp
        try:
            llamacpp.remove(cfg, model)
        except llamacpp.LlamaError as exc:
            raise BackendError(str(exc)) from exc
        return
    if backend != "ollama":
        raise BackendError("models are removed in the LM Studio app")
    req = urllib.request.Request(
        _url(cfg.ollama_url, "/api/delete"),
        data=json.dumps({"model": model}).encode(),
        headers={"Content-Type": "application/json"}, method="DELETE",
    )
    try:
        urllib.request.urlopen(req, timeout=_TIMEOUT).close()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise BackendError(f"no model named {model}") from exc
        raise BackendError(str(exc)) from exc
    except (urllib.error.URLError, socket.timeout, OSError,
            http.client.HTTPException) as exc:
        raise BackendError(str(exc) or type(exc).__name__) from exc


def pull_model(backend, cfg, model, on_progress=None) -> None:
    """Ollama only: POST /api/pull, streaming progress objects."""
    if backend != "ollama":
        raise BackendError("models are downloaded in the LM Studio app")
    url = _url(cfg.ollama_url, "/api/pull")
    data = json.dumps({"model": model, "stream": True}).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=3600)
    except (urllib.error.URLError, socket.timeout, OSError,
            http.client.HTTPException) as exc:
        raise BackendError(str(exc) or type(exc).__name__) from exc
    with resp:
        buf = b""
        for raw in resp:
            buf += bytes(raw)
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line.decode("utf-8"))
                except ValueError:
                    continue
                if obj.get("error"):
                    raise BackendError(str(obj["error"]))
                if on_progress is not None:
                    on_progress(obj)




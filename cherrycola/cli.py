"""CLI: ``cherrycola ask|chat|status|models|pull|use|backend|start|stop``,
model settings (``settings|set|unset|preset|flag|options``) and llama.cpp
model tools (``get|import-ollama|quantize|imatrix|info|plan``)."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from . import backends as be
from . import config as config_mod
from . import gguf, hardware, llamacpp, profiles
from . import web as web_mod

__all__ = ["build_parser", "main", "run"]

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_UNREACHABLE = 2
EXIT_MODEL_MISSING = 3


def _err(message: str) -> None:
    print(f"CherryCola: {message}", file=sys.stderr)


def _state_dir() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "state")
    return Path(base) / "cherrycola"


def _runtime_dir() -> Path:
    base = os.environ.get("XDG_RUNTIME_DIR")
    if base:
        return Path(base) / "cherrycola"
    return _state_dir()


def _pid_file() -> Path:
    return _runtime_dir() / "ollama.pid"


def _log_file() -> Path:
    return _state_dir() / "ollama.log"


def _read_pid() -> int | None:
    try:
        return int(_pid_file().read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _is_ollama(pid: int) -> bool:
    """A stale pid file may point at a reused pid; only ever signal our ollama."""
    try:
        return Path(f"/proc/{pid}/comm").read_text(encoding="utf-8").strip() == "ollama"
    except OSError:
        return False


def _spawn(args: list[str], log_path: Path, env: dict | None = None):
    """Spawn detached. Injectable seam: tests monkeypatch this."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab")
    try:
        proc = subprocess.Popen(
            args, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
            env={**os.environ, **env} if env else None,
        )
    finally:
        log.close()
    return proc


def start_server(cfg: config_mod.Config, spawn=_spawn,
                 timeout: float = 20) -> int:
    if be.server_reachable("ollama", cfg):
        return EXIT_OK
    if be.shutil.which("ollama") is None:
        _err("ollama is not installed, and its server is not running")
        return EXIT_UNREACHABLE
    log = _log_file()
    pid_path = _pid_file()
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    # KV cache type and flash attention are server-wide in Ollama.
    env = profiles.ollama_server_env(profiles.load_profiles().get("*", {}))
    try:
        proc = spawn(["ollama", "serve"], log, **({"env": env} if env else {}))
    except (OSError, be.BackendError) as exc:
        _err(f"could not start ollama: {exc}")
        return EXIT_UNREACHABLE
    try:
        pid_path.write_text(str(proc.pid) + "\n", encoding="utf-8")
    except OSError as exc:
        _err(f"could not write pid file: {exc}")
        return EXIT_UNREACHABLE
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if be.server_reachable("ollama", cfg):
            return EXIT_OK
        time.sleep(0.25)
    _err(f"ollama server did not start (see {log})")
    return EXIT_UNREACHABLE


def stop_server(cfg: config_mod.Config) -> int:
    _ = cfg
    pid = _read_pid()
    if pid is None:
        _err("cherrycola did not start an ollama server (no pid file)")
        return EXIT_UNREACHABLE
    if not _pid_alive(pid) or not _is_ollama(pid):
        try:
            _pid_file().unlink()
        except OSError:
            pass
        _err("cherrycola did not start an ollama server (not running)")
        return EXIT_UNREACHABLE
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        _err(f"could not stop ollama server (pid {pid}): {exc}")
        return EXIT_USAGE
    for _ in range(40):
        if not _pid_alive(pid):
            break
        time.sleep(0.25)
    try:
        _pid_file().unlink()
    except OSError:
        pass
    return EXIT_OK


def ensure_running(backend: str, cfg: config_mod.Config,
                   spawn=_spawn) -> int:
    """Start Ollama on demand. LM Studio has no local start path.

    llama.cpp starts per model when a question needs it; here we only
    check it is installed.
    """
    if backend == "llamacpp":
        if not be.server_reachable(backend, cfg):
            _err(f"{cfg.llama_server} is not installed (install llama.cpp, "
                 f"or set llama_server in {config_mod.config_path()})")
            return EXIT_UNREACHABLE
        return EXIT_OK
    if backend != "ollama":
        if not be.server_reachable(backend, cfg):
            _err(f"lm studio server is not reachable at {cfg.lmstudio_url}")
            return EXIT_UNREACHABLE
        return EXIT_OK
    return start_server(cfg, spawn=spawn)


def _missing_model_message(model: str, backend: str = "ollama") -> str:
    if backend == "llamacpp":
        return (f"no GGUF model named {model}. Get one with: cherrycola get "
                f"OWNER/REPO:Q4_K_M, or reuse an Ollama download with: "
                f"cherrycola import-ollama MODEL")
    size = config_mod.MODEL_SIZES_GB.get(model)
    if size is not None:
        return (
            f"the model {model} isn't downloaded yet "
            f"(about {size} GB). Run: cherrycola pull"
        )
    return f"the model {model} isn't downloaded yet. Run: cherrycola pull"


def _settings_for(args, model: str) -> dict:
    """The model's settings plus this run's --preset and -o overrides."""
    overrides = {}
    if getattr(args, "preset", None):
        overrides["preset"] = args.preset
    for item in getattr(args, "option", None) or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise profiles.SettingError(f"-o needs key=value, not {item!r}")
        overrides[key.strip()] = profiles.parse_value(key.strip(), value)
    return profiles.effective(model, overrides)


def _status_line(text: str) -> None:
    print(f"({text})", file=sys.stderr, flush=True)


def _stream_answer(backend: str, cfg: config_mod.Config, model: str,
                   messages: list[dict[str, str]], show_thinking: bool,
                   settings: dict | None = None) -> str:
    stripper = be.ThinkStripper(keep=show_thinking)
    full: list[str] = []
    for chunk in be.iter_chat_chunks(backend, cfg, model, messages,
                                     settings, on_status=_status_line):
        piece = stripper.feed(chunk)
        if piece:
            sys.stdout.write(piece)
            sys.stdout.flush()
            full.append(piece)
    tail = stripper.flush()
    if tail:
        sys.stdout.write(tail)
        sys.stdout.flush()
        full.append(tail)
    sys.stdout.write("\n")
    sys.stdout.flush()
    return "".join(full)


def cmd_ask(args, cfg: config_mod.Config) -> int:
    backend = be.resolve_backend(cfg)
    rc = ensure_running(backend, cfg)
    if rc != EXIT_OK:
        return rc
    model = args.model or be.resolve_model(backend, cfg)
    args.settings = _settings_for(args, model)
    if not be.check_model_installed(backend, model, cfg):
        _err(_missing_model_message(model, backend))
        return EXIT_MODEL_MISSING
    if args.question == ["-"]:
        question = sys.stdin.read().strip()
    else:
        question = " ".join(args.question).strip()
    if not question:
        _err("ask needs a question")
        return EXIT_USAGE
    use_web = bool(getattr(args, "web", False))
    if use_web:
        return _answer_with_sources(question, args, cfg, backend, model)
    messages = [
        {"role": "system", "content": be.SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    try:
        _stream_answer(backend, cfg, model, messages, args.show_thinking,
                       args.settings)
    except be.BackendError as exc:
        _err(str(exc) or "backend request failed")
        return EXIT_UNREACHABLE
    except BrokenPipeError:
        return EXIT_OK
    return EXIT_OK


def _answer_with_sources(question: str, args, cfg: config_mod.Config,
                         backend: str, model: str,
                         history: list[dict[str, str]] | None = None) -> int:
    """Search, read the top pages, stream a cited answer, print Sources.

    history (chat) keeps only the plain question and answer; the page text
    goes in this turn's system prompt only.
    """
    if history is None:
        history = [{"role": "system", "content": be.SYSTEM_PROMPT}]
    try:
        results = web_mod.search(question, cfg)
        if not results:
            _err("web search returned nothing")
    except be.BackendError as exc:
        _err(str(exc) or "web search failed")
        results = []
    turn = {"role": "user", "content": question}
    if results:
        pages = web_mod.fetch_many([r.url for r in results[:cfg.web_pages]])
        grounded = web_mod.web_messages(question, results, pages)
        messages = grounded[:1] + history[1:] + [turn]
    else:
        _err("answering without the web")
        messages = history + [turn]
    try:
        answer = _stream_answer(backend, cfg, model, messages,
                                args.show_thinking,
                                getattr(args, "settings", None))
    except be.BackendError as exc:
        _err(str(exc) or "backend request failed")
        return EXIT_UNREACHABLE
    except BrokenPipeError:
        return EXIT_OK
    history += [turn, {"role": "assistant", "content": answer}]
    if results:
        sys.stdout.write("\nSources:\n")
        for i, item in enumerate(results, start=1):
            sys.stdout.write(f"[{i}] {item.url}\n")
        sys.stdout.flush()
    return EXIT_OK


def cmd_search(args, cfg: config_mod.Config) -> int:
    query = " ".join(args.query).strip()
    if not query:
        _err("search needs a query")
        return EXIT_USAGE
    try:
        results = web_mod.search(query, cfg)
    except be.BackendError as exc:
        _err(str(exc) or "web search failed")
        return EXIT_UNREACHABLE
    if args.json:
        print(json.dumps([
            {"title": r.title, "url": r.url, "snippet": r.snippet}
            for r in results]))
        return EXIT_OK
    for i, item in enumerate(results, start=1):
        print(f"{i}. {item.title}")
        print(f"   {item.url}")
        print(f"   {item.snippet}")
    return EXIT_OK


def cmd_chat(args, cfg: config_mod.Config) -> int:
    backend = be.resolve_backend(cfg)
    rc = ensure_running(backend, cfg)
    if rc != EXIT_OK:
        return rc
    model = args.model or be.resolve_model(backend, cfg)
    args.settings = _settings_for(args, model)
    if not be.check_model_installed(backend, model, cfg):
        _err(_missing_model_message(model, backend))
        return EXIT_MODEL_MISSING
    history: list[dict[str, str]] = [
        {"role": "system", "content": be.SYSTEM_PROMPT}]
    web_mode = False
    print(f"CherryCola ({backend}/{model}). /exit quits, /clear resets.")
    while True:
        try:
            line = input("> ")
        except EOFError:
            print()
            return EXIT_OK
        except KeyboardInterrupt:
            print()
            return EXIT_OK
        text = line.strip()
        if not text:
            continue
        if text == "/exit":
            return EXIT_OK
        if text == "/clear":
            history = history[:1]
            print("(history cleared)")
            continue
        if text == "/web":
            web_mode = not web_mode
            print("Web search on." if web_mode else "Web search off.")
            continue
        if web_mode:
            _answer_with_sources(text, args, cfg, backend, model,
                                 history=history)
            continue
        history.append({"role": "user", "content": text})
        try:
            answer = _stream_answer(
                backend, cfg, model, history, args.show_thinking,
                args.settings)
        except be.BackendError as exc:
            _err(str(exc) or "backend request failed")
            history.pop()
            continue
        except BrokenPipeError:
            return EXIT_OK
        history.append({"role": "assistant", "content": answer})


def cmd_status(args, cfg: config_mod.Config) -> int:
    backend = be.resolve_backend(cfg)
    reachable = be.server_reachable(backend, cfg)
    models: list[str] = []
    running: str | None = None
    if reachable:
        model = be.resolve_model(backend, cfg)
        try:
            models = be.list_models(backend, cfg)
        except be.BackendError:
            models = []
        installed = model in models
        running = be.running_model(backend, cfg)
    else:
        # Unreachable: no probing (would just time out); RAM default.
        model = cfg.model or config_mod.default_model_for_ram(
            config_mod.total_ram_bytes())
        installed = False
    if args.json:
        print(json.dumps({
            "backend": backend,
            "configured_backend": cfg.backend,
            "reachable": reachable,
            "model": model,
            "installed": installed,
            "running": running,
            "models": models,
            "search": cfg.search,
        }))
        return EXIT_OK
    print(f"backend:   {backend}")
    print(f"reachable: {'yes' if reachable else 'no'}")
    print(f"model:     {model}")
    print(f"installed: {'yes' if installed else 'no'}")
    print(f"running:   {running if running else '-'}")
    return EXIT_OK


def cmd_models(args, cfg: config_mod.Config) -> int:
    _ = args
    backend = be.resolve_backend(cfg)
    rc = ensure_running(backend, cfg)
    if rc != EXIT_OK:
        return rc
    try:
        models = be.list_models(backend, cfg)
    except be.BackendError as exc:
        _err(str(exc) or "could not list models")
        return EXIT_UNREACHABLE
    for name in models:
        print(name)
    return EXIT_OK


def cmd_recommend(args, cfg: config_mod.Config) -> int:
    _ = cfg
    r = hardware.recommend(config_mod.total_ram_bytes(),
                           hardware.detect_gpus())
    if args.json:
        print(json.dumps(r))
        return EXIT_OK
    where = (f"graphics card ({r['vram_gb']} GB)" if r["runs_on"] == "gpu"
             else f"processor ({r['ram_gb']} GB RAM)")
    print(f"Models run on the {where}.")
    state = ("installed" if r["engine_installed"]
             else f"install it: sudo pacman -S {r['engine']}")
    print(f"Engine: {r['engine']} - {r['engine_reason']} ({state})")
    print(f"Recommended: {r['recommended']}\n")
    for m in r["models"]:
        if m["fits"]:
            flag = "  (slow here)" if m["slow"] else ""
            print(f"  {m['model']:<20} {m['size_gb']:>5} GB  {m['note']}{flag}")
    return EXIT_OK


def cmd_pull(args, cfg: config_mod.Config) -> int:
    backend = be.resolve_backend(cfg)
    model = args.model or be.resolve_model(backend, cfg)
    if backend == "llamacpp":
        _err("the llama.cpp engine downloads with: cherrycola get "
             "OWNER/REPO:QUANT (from Hugging Face)")
        return EXIT_USAGE
    if backend != "ollama":
        _err("models are downloaded in the LM Studio app")
        return EXIT_USAGE
    rc = ensure_running(backend, cfg)
    if rc != EXIT_OK:
        return rc
    print(f"Pulling {model} (downloads from the internet)...")

    def on_progress(obj: dict) -> None:
        total = obj.get("total")
        done = obj.get("completed")
        status = obj.get("status", "")
        if isinstance(total, (int, float)) and total and isinstance(
                done, (int, float)):
            pct = min(100.0, 100.0 * done / total)
            sys.stdout.write(f"\r{status} {pct:5.1f}%")
        else:
            sys.stdout.write(f"\r{status}".rstrip() + " " * 4)
        sys.stdout.flush()

    try:
        be.pull_model(backend, cfg, model, on_progress=on_progress)
    except be.BackendError as exc:
        sys.stdout.write("\n")
        _err(str(exc) or "pull failed")
        return EXIT_UNREACHABLE
    sys.stdout.write("\n")
    sys.stdout.flush()
    print(f"Pulled {model}.")
    return EXIT_OK


# ---- model settings ----------------------------------------------------------

def _fmt(value) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, list):
        return "|".join(value)
    if isinstance(value, dict):
        return " ".join(k if v is True else f"{k} {v}"
                        for k, v in value.items())
    return str(value)


def cmd_settings(args, cfg: config_mod.Config) -> int:
    if args.list:
        rows = []
        for key, spec in profiles.SETTINGS.items():
            rows.append({"key": key, "group": spec.group,
                         "label": spec.label, "type": spec.kind,
                         "min": spec.lo, "max": spec.hi,
                         "choices": list(spec.choices),
                         "engines": list(spec.engines), "help": spec.help})
        if args.json:
            print(json.dumps({"settings": rows, "presets": profiles.PRESETS}))
            return EXIT_OK
        for group, title in profiles.GROUPS.items():
            print(f"{title}:")
            for r in rows:
                if r["group"] == group:
                    only = "" if len(r["engines"]) == 3 else \
                        f" [{', '.join(r['engines'])}]"
                    print(f"  {r['key']:<18} {r['help']}{only}")
        print("Presets: " + ", ".join(profiles.PRESETS))
        return EXIT_OK
    backend = be.resolve_backend(cfg)
    model = args.model or be.resolve_model(backend, cfg)
    stored = profiles.load_profiles()
    eff = profiles.effective(model, profiles=stored)
    own = stored.get(model, {})
    if backend == "ollama":
        _, skipped = profiles.ollama_request(eff)
        skipped += [k for k in ("kv_cache_type", "flash_attention")
                    if k in own]
    elif backend == "lmstudio":
        _, skipped = profiles.lmstudio_request(eff)
    else:
        skipped = [k for k in eff if "llamacpp" not in
                   profiles.SETTINGS[k].engines]
    if args.json:
        print(json.dumps({"backend": backend, "model": model,
                          "settings": eff, "model_settings": own,
                          "defaults": stored.get("*", {}),
                          "not_applied": skipped}))
        return EXIT_OK
    print(f"{model} ({backend})")
    if not eff:
        print("  engine defaults (nothing set)")
    for key, value in eff.items():
        where = "model" if key in own else "preset" if (
            eff.get("preset") and key in profiles.PRESETS[eff["preset"]]
            and key not in stored.get("*", {})) else "all models"
        note = "  (not used by this engine)" if key in skipped else ""
        print(f"  {key:<18} {_fmt(value):<14} from {where}{note}")
    if backend == "ollama" and any(k in own for k in (
            "kv_cache_type", "flash_attention")):
        print("Ollama sets kv_cache_type and flash_attention for the whole "
              "server: use cherrycola set --all, then stop and start.")
    return EXIT_OK


def _target(args, cfg: config_mod.Config) -> str | None:
    if getattr(args, "all", False):
        return None
    if args.model:
        return args.model
    return be.resolve_model(be.resolve_backend(cfg), cfg)


def cmd_set(args, cfg: config_mod.Config) -> int:
    target = _target(args, cfg)
    value = profiles.set_value(target, args.key, " ".join(args.value))
    print(f"{args.key} = {_fmt(value)} for {target or 'all models'}.")
    if args.key in profiles.SETTINGS and profiles.SETTINGS[args.key].group \
            in ("load", "gpu", "rope"):
        print("Applies the next time the model loads.")
    return EXIT_OK


def cmd_unset(args, cfg: config_mod.Config) -> int:
    target = _target(args, cfg)
    if not profiles.unset_value(target, args.key):
        _err(f"{args.key or 'nothing'} is not set for "
             f"{target or 'all models'}")
        return EXIT_USAGE
    print(f"Cleared {args.key or 'every setting'} for "
          f"{target or 'all models'}.")
    return EXIT_OK


def cmd_flag(args, cfg: config_mod.Config) -> int:
    """Set any llama-server option by its flag."""
    if not args.flag.startswith("-"):
        _err("give the option as it appears in cherrycola options, "
             "for example --mirostat")
        return EXIT_USAGE
    target = _target(args, cfg)
    stored = profiles.load_profiles()
    raw = dict(stored.get(target or "*", {}).get("llama_args", {}))
    if args.remove:
        if raw.pop(args.flag, None) is None:
            _err(f"{args.flag} is not set")
            return EXIT_USAGE
    else:
        value = args.value if args.value is not None else True
        try:
            llamacpp.raw_args({args.flag: value},
                              llamacpp.server_options(cfg.llama_server))
        except llamacpp.LlamaError as exc:
            _err(str(exc))
            return EXIT_USAGE
        raw[args.flag] = value
    if raw:
        profiles.set_value(target, "llama_args", raw)
    else:
        profiles.unset_value(target, "llama_args")
    print(f"llama.cpp options for {target or 'all models'}: "
          f"{_fmt(raw) or 'none'}")
    return EXIT_OK


def cmd_options(args, cfg: config_mod.Config) -> int:
    try:
        opts = llamacpp.server_options(cfg.llama_server)
    except llamacpp.LlamaError as exc:
        _err(str(exc))
        return EXIT_UNREACHABLE
    if args.search:
        q = args.search.lower()
        opts = [o for o in opts if q in o.flag or q in o.help.lower()]
    if args.json:
        print(json.dumps([{"flag": o.flag, "names": o.names, "arg": o.arg,
                           "help": o.help, "section": o.section,
                           "negated": o.negated, "choices": list(o.choices)}
                          for o in opts]))
        return EXIT_OK
    section = None
    for o in opts:
        if o.section != section:
            section = o.section
            print(f"\n{section}:")
        spec = f"{o.flag} {o.arg}".strip()
        print(f"  {spec:<34} {o.help}")
    return EXIT_OK


# ---- llama.cpp models ----------------------------------------------------------

def _progress_bar(name: str, done: int, total: int) -> None:
    if total:
        sys.stdout.write(f"\r{name}: {100 * done / total:5.1f}% of "
                         f"{total / 1024**3:.1f} GB")
    else:
        sys.stdout.write(f"\r{name}: {done / 1024**3:.2f} GB")
    sys.stdout.flush()


def cmd_get(args, cfg: config_mod.Config) -> int:
    print(f"Downloading {args.spec} from Hugging Face...")
    try:
        name = llamacpp.download(cfg, args.spec, _progress_bar)
    except llamacpp.LlamaError as exc:
        sys.stdout.write("\n")
        _err(str(exc))
        return EXIT_UNREACHABLE
    print(f"\nSaved as {name}. Use it with: cherrycola use {name}")
    return EXIT_OK


def cmd_import_ollama(args, cfg: config_mod.Config) -> int:
    try:
        name = llamacpp.import_ollama(cfg, args.model)
    except llamacpp.LlamaError as exc:
        _err(str(exc))
        return EXIT_MODEL_MISSING
    print(f"Linked {args.model} as {name} (no copy). Use it with: "
          f"cherrycola use {name}")
    return EXIT_OK


def cmd_quantize(args, cfg: config_mod.Config) -> int:
    try:
        name = llamacpp.quantize(
            cfg, args.model, args.type, out=args.out, imatrix=args.imatrix,
            allow_requantize=args.allow_requantize, threads=args.threads)
    except (llamacpp.LlamaError, gguf.GGUFError) as exc:
        _err(str(exc))
        return EXIT_USAGE
    print(f"Wrote {name}.")
    return EXIT_OK


def cmd_imatrix(args, cfg: config_mod.Config) -> int:
    try:
        out = llamacpp.imatrix(cfg, args.model, args.text,
                               profiles.effective(args.model))
    except llamacpp.LlamaError as exc:
        _err(str(exc))
        return EXIT_USAGE
    print(f"Wrote {out}. Use it with: cherrycola quantize {args.model} "
          f"TYPE --imatrix {out}")
    return EXIT_OK


def _gguf_info(cfg, model: str) -> gguf.ModelInfo | None:
    path = llamacpp.find_model(cfg, model)
    if path is None:
        _err(f"no GGUF model named {model} (cherrycola models with the "
             f"llamacpp backend lists them)")
        return None
    try:
        return gguf.read_info(path)
    except gguf.GGUFError as exc:
        _err(str(exc))
        return None


def cmd_info(args, cfg: config_mod.Config) -> int:
    model = args.model or be.resolve_model("llamacpp", cfg)
    info = _gguf_info(cfg, model)
    if info is None:
        return EXIT_MODEL_MISSING
    data = {"model": model, "path": str(info.path),
            "size_gb": round(llamacpp.total_size(info.path) / 1024**3, 2),
            "architecture": info.architecture, "name": info.name,
            "quantization": info.file_type,
            "parameters_b": round(info.parameters / 1e9, 2),
            "layers": info.block_count, "context_length": info.context_length,
            "experts": info.expert_count, "split_files": info.split_count}
    if args.json:
        print(json.dumps(data))
        return EXIT_OK
    for key, value in data.items():
        print(f"{key:<15} {value}")
    return EXIT_OK


def cmd_plan(args, cfg: config_mod.Config) -> int:
    model = args.model or be.resolve_model("llamacpp", cfg)
    info = _gguf_info(cfg, model)
    if info is None:
        return EXIT_MODEL_MISSING
    settings = profiles.effective(model)
    plan = llamacpp.make_plan(info, settings)
    if args.json:
        print(json.dumps(plan.__dict__))
        return EXIT_OK
    print(plan.describe(bool(settings.get("disk_stream"))))
    return EXIT_OK


def _add_run_options(p) -> None:
    p.add_argument("-m", "--model", help="use this model instead of the default")
    p.add_argument("--preset", choices=list(profiles.PRESETS),
                   help="sampling preset for this run")
    p.add_argument("-o", "--option", action="append", metavar="KEY=VALUE",
                   help="override one setting for this run (repeatable)")


def _add_target(p) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("-m", "--model", help="the model (default: current)")
    g.add_argument("--all", action="store_true",
                   help="the default for every model")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cherrycola", description="GraceOS local model CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="ask one question (streams the answer)")
    ask.add_argument("question", nargs="+", help="question words, or - for stdin")
    ask.add_argument("--show-thinking", action="store_true",
                     help="keep <think> reasoning blocks")
    ask.add_argument("--web", action="store_true",
                     help="search the web first and cite sources")
    _add_run_options(ask)

    search_cmd = sub.add_parser("search", help="search the web (no model)")
    search_cmd.add_argument("query", nargs="+", help="search words")
    search_cmd.add_argument("--json", action="store_true",
                            help="machine-readable JSON")

    chat = sub.add_parser("chat", help="interactive chat loop")
    chat.add_argument("--show-thinking", action="store_true",
                      help="keep <think> reasoning blocks")
    _add_run_options(chat)

    st = sub.add_parser("settings", help="show a model's load and sampling "
                        "settings, or every setting with --list")
    st.add_argument("model", nargs="?", default=None)
    st.add_argument("--list", action="store_true",
                    help="every setting, what it does and which engines use it")
    st.add_argument("--json", action="store_true")

    sset = sub.add_parser("set", help="set a model setting "
                          "(see: settings --list)")
    _add_target(sset)
    sset.add_argument("key")
    sset.add_argument("value", nargs="+")

    sunset = sub.add_parser("unset", help="clear a setting (or all of them)")
    _add_target(sunset)
    sunset.add_argument("key", nargs="?", default=None)

    pre = sub.add_parser("preset", help="pick a sampling preset")
    _add_target(pre)
    pre.add_argument("name", choices=list(profiles.PRESETS))

    flag = sub.add_parser("flag", help="set any llama-server option "
                          "(see: cherrycola options)")
    _add_target(flag)
    flag.add_argument("--remove", action="store_true")
    flag.add_argument("flag")
    flag.add_argument("value", nargs="?", default=None)

    opts = sub.add_parser("options", help="every llama-server option, "
                          "with its help")
    opts.add_argument("search", nargs="?", default=None)
    opts.add_argument("--json", action="store_true")

    get = sub.add_parser("get", help="download a GGUF model from Hugging "
                         "Face (llama.cpp engine)")
    get.add_argument("spec", help="OWNER/REPO or OWNER/REPO:QUANT, "
                     "e.g. Qwen/Qwen3-4B-GGUF:Q4_K_M")

    imp = sub.add_parser("import-ollama", help="use a model Ollama already "
                         "downloaded with the llama.cpp engine (no copy)")
    imp.add_argument("model")

    qz = sub.add_parser("quantize", help="write a GGUF model in another "
                        "quantization (llama-quantize)")
    qz.add_argument("model")
    qz.add_argument("type", help="Q4_K_M, Q5_K_M, Q8_0, IQ4_XS, ...")
    qz.add_argument("--out", help="name for the new model")
    qz.add_argument("--imatrix", help="importance matrix file "
                    "(see: cherrycola imatrix)")
    qz.add_argument("--allow-requantize", action="store_true",
                    help="allow quantizing an already quantized model")
    qz.add_argument("--threads", type=int)

    im = sub.add_parser("imatrix", help="compute an importance matrix "
                        "from a text file (llama-imatrix)")
    im.add_argument("model")
    im.add_argument("text")

    inf = sub.add_parser("info", help="what a GGUF model is "
                         "(architecture, size, quantization)")
    inf.add_argument("model", nargs="?", default=None)
    inf.add_argument("--json", action="store_true")

    pl = sub.add_parser("plan", help="how a GGUF model splits across GPU, "
                        "RAM and disk")
    pl.add_argument("model", nargs="?", default=None)
    pl.add_argument("--json", action="store_true")

    status = sub.add_parser("status", help="show backend/model status")
    status.add_argument("--json", action="store_true",
                        help="machine-readable JSON")

    sub.add_parser("models", help="list installed models")

    rec = sub.add_parser("recommend",
                         help="models and engine that suit this PC")
    rec.add_argument("--json", action="store_true",
                     help="machine-readable JSON")

    rm = sub.add_parser("remove", help="delete a downloaded model (Ollama only)")
    rm.add_argument("model")

    pull = sub.add_parser("pull", help="download a model (Ollama only)")
    pull.add_argument("model", nargs="?", default=None)

    use = sub.add_parser("use", help="set the default model")
    use.add_argument("model")

    bck = sub.add_parser("backend", help="set the backend")
    bck.add_argument("name", choices=list(config_mod.BACKENDS))

    sub.add_parser("start", help="start the ollama server, or load the "
                   "model in llama.cpp")
    sub.add_parser("stop", help="stop the server cherrycola started")
    return parser


def _flag_argv(argv: list[str]) -> list[str]:
    """``flag --mirostat 2``: the llama.cpp flag isn't one of ours."""
    ours = {"--remove", "--all", "-m", "--model", "-h", "--help"}
    out, i = argv[:1], 1
    while i < len(argv):
        a = argv[i]
        if a in ("-m", "--model"):
            out += argv[i:i + 2]
            i += 2
            continue
        if a in ours or a == "--":
            out.append(a)
            i += 1
            continue
        return out + ["--"] + argv[i:]
    return out


def run(argv: Sequence[str] | None = None,
        cfg: config_mod.Config | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["flag"]:
        argv = _flag_argv(argv)
    args = build_parser().parse_args(argv)
    loaded = cfg if cfg is not None else config_mod.load_config()
    try:
        if args.command == "ask":
            return cmd_ask(args, loaded)
        if args.command == "search":
            return cmd_search(args, loaded)
        if args.command == "chat":
            return cmd_chat(args, loaded)
        if args.command == "status":
            return cmd_status(args, loaded)
        if args.command == "models":
            return cmd_models(args, loaded)
        if args.command == "recommend":
            return cmd_recommend(args, loaded)
        simple = {"settings": cmd_settings, "set": cmd_set,
                  "unset": cmd_unset, "flag": cmd_flag,
                  "options": cmd_options, "get": cmd_get,
                  "import-ollama": cmd_import_ollama,
                  "quantize": cmd_quantize, "imatrix": cmd_imatrix,
                  "info": cmd_info, "plan": cmd_plan}
        if args.command in simple:
            return simple[args.command](args, loaded)
        if args.command == "preset":
            args.key, args.value = "preset", [args.name]
            return cmd_set(args, loaded)
        if args.command == "pull":
            return cmd_pull(args, loaded)
        if args.command == "remove":
            backend = be.resolve_backend(loaded)
            rc = ensure_running(backend, loaded)
            if rc != EXIT_OK:
                return rc
            be.remove_model(backend, loaded, args.model)
            print(f"Removed {args.model}.")
            return EXIT_OK
        if args.command == "use":
            config_mod.save_setting("model", args.model, loaded.path)
            print(f"Using model {args.model}.")
            return EXIT_OK
        if args.command == "backend":
            config_mod.save_setting("backend", args.name, loaded.path)
            print(f"Using backend {args.name}.")
            return EXIT_OK
        if args.command == "start" and \
                be.resolve_backend(loaded) == "llamacpp":
            model = be.resolve_model("llamacpp", loaded)
            be.prepare("llamacpp", loaded, model, profiles.effective(model),
                       _status_line)
            print(f"{model} is loaded.")
            return EXIT_OK
        if args.command == "stop" and \
                be.resolve_backend(loaded) == "llamacpp":
            if not llamacpp.stop():
                _err("no llama-server started by cherrycola is running")
                return EXIT_UNREACHABLE
            return EXIT_OK
        if args.command == "start":
            if be.resolve_backend(loaded) != "ollama":
                _err("start only applies to the ollama backend")
                return EXIT_USAGE
            return start_server(loaded)
        if args.command == "stop":
            if be.resolve_backend(loaded) != "ollama":
                _err("stop only applies to the ollama backend")
                return EXIT_USAGE
            return stop_server(loaded)
    except be.BackendError as exc:
        _err(str(exc) or "backend request failed")
        return EXIT_UNREACHABLE
    except profiles.SettingError as exc:
        _err(str(exc))
        return EXIT_USAGE
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return EXIT_USAGE
    return EXIT_USAGE  # pragma: no cover - argparse enforces the subcommand


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point (``[project.scripts]``)."""
    return run(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())






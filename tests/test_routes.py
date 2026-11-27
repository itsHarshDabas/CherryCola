"""Pulling through the selected engine, and loading a model only for one answer."""

from __future__ import annotations

from typing import ClassVar

import pytest

from cherrycola import backends as be
from cherrycola import config as config_mod
from cherrycola import routes
from cherrycola.cli import run

from .conftest import OLLAMA_CALLS

# ---- sorting what people paste ----------------------------------------------------

@pytest.mark.parametrize("text,backend,kind,value,quant", [
    ("qwen3:4b", "ollama", "ollama", "qwen3:4b", ""),
    ("user/model:tag", "ollama", "ollama", "user/model:tag", ""),
    ("Qwen/Qwen3-4B-GGUF:Q4_K_M", "ollama", "hf", "Qwen/Qwen3-4B-GGUF", "Q4_K_M"),
    ("Qwen/Qwen3-4B-GGUF", "llamacpp", "hf", "Qwen/Qwen3-4B-GGUF", ""),
    ("owner/repo:Q8_0", "llamacpp", "hf", "owner/repo", "Q8_0"),
    ("hf.co/owner/repo:Q5_K_M", "ollama", "hf", "owner/repo", "Q5_K_M"),
    ("https://huggingface.co/owner/repo", "lmstudio", "hf", "owner/repo", ""),
    ("https://huggingface.co/owner/repo/resolve/main/m-Q4_K_M.gguf",
     "llamacpp", "url",
     "https://huggingface.co/owner/repo/resolve/main/m-Q4_K_M.gguf", ""),
    ("https://example.org/models/tiny.gguf", "ollama", "url",
     "https://example.org/models/tiny.gguf", ""),
    ("qwen3:4b", "llamacpp", "ollama", "qwen3:4b", ""),
])
def test_classify(text, backend, kind, value, quant):
    spec = routes.classify(text, backend)
    assert (spec.kind, spec.value, spec.quant) == (kind, value, quant)


@pytest.mark.parametrize("text", [
    "", "http://example.org/m.gguf", "https://example.org/m.bin",
    "bad name;rm -rf", "a\nb", "../../etc/passwd",
])
def test_classify_refuses(text):
    with pytest.raises(be.BackendError):
        routes.classify(text, "ollama")


def test_classify_local_file(tmp_path):
    model = tmp_path / "Tiny-Q4.gguf"
    model.write_bytes(b"GGUF....")
    spec = routes.classify(str(model), "llamacpp")
    assert spec.kind == "file" and spec.value == str(model.resolve())
    assert routes.classify(str(tmp_path), "ollama").kind == "dir"


# ---- pulling through each engine ---------------------------------------------------

def test_pull_hf_through_ollama_uses_hf_co(ollama_cfg, capsys):
    assert run(["pull", "Qwen/Qwen3-4B-GGUF:Q4_K_M"], cfg=ollama_cfg) == 0
    assert ("/api/pull", {"model": "hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M",
                          "stream": True}) in OLLAMA_CALLS
    assert "Pulled hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M." in capsys.readouterr().out


def test_pull_error_from_ollama(ollama_cfg, capsys):
    assert run(["pull", "missing:1b"], cfg=ollama_cfg) == 2
    assert "file does not exist" in capsys.readouterr().err


def test_pull_use_sets_default(ollama_cfg, capsys):
    assert run(["pull", "--use", "qwen3:8b"], cfg=ollama_cfg) == 0
    assert "model = qwen3:8b" in ollama_cfg.path.read_text()


def test_pull_hf_through_llamacpp(monkeypatch, tmp_path):
    cfg = config_mod.Config(backend="llamacpp", models_dir=str(tmp_path))
    seen = {}
    monkeypatch.setattr("cherrycola.llamacpp.download",
                        lambda c, spec, prog=None: seen.setdefault("s", spec)
                        and "Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M")
    assert routes.pull(cfg, "Qwen/Qwen3-4B-GGUF:Q4_K_M") == \
        "Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M"
    assert seen["s"] == "Qwen/Qwen3-4B-GGUF:Q4_K_M"


def test_pull_local_file_through_llamacpp_adds_its_folder(tmp_path):
    models = tmp_path / "models"
    elsewhere = tmp_path / "mine"
    elsewhere.mkdir()
    gguf = elsewhere / "Tiny.gguf"
    gguf.write_bytes(b"GGUF")
    cfg = config_mod.Config(backend="llamacpp", models_dir=str(models),
                            path=tmp_path / "c.conf")
    name = routes.pull(cfg, str(gguf))
    assert name.endswith("Tiny")
    from cherrycola import llamacpp
    assert name in llamacpp.list_local(cfg)


def test_pull_lmstudio_runs_lms_get(lmstudio_cfg, monkeypatch):
    ran = []
    monkeypatch.setattr("cherrycola.routes.shutil.which", lambda n: "/bin/lms")
    monkeypatch.setattr("cherrycola.routes._run", lambda a, timeout=None:
                        ran.append(a))
    routes.pull(lmstudio_cfg, "owner/repo-GGUF:Q4_K_M")
    assert ran == [["/bin/lms", "get", "-y", "owner/repo-GGUF:Q4_K_M"]]


def test_download_url_checks_gguf_magic(monkeypatch, tmp_path):
    import io

    class Resp(io.BytesIO):
        status = 200
        headers: ClassVar = {"Content-Length": "8"}

        def geturl(self):
            return "https://example.org/x.gguf"

    cfg = config_mod.Config(models_dir=str(tmp_path))
    monkeypatch.setattr("cherrycola.routes.urllib.request.urlopen",
                        lambda req, timeout=None: Resp(b"<html>.."))
    with pytest.raises(be.BackendError, match="not a GGUF"):
        routes.download_url(cfg, "https://example.org/x.gguf")
    assert not list((tmp_path / "downloads").iterdir())
    monkeypatch.setattr("cherrycola.routes.urllib.request.urlopen",
                        lambda req, timeout=None: Resp(b"GGUFdata"))
    path = routes.download_url(cfg, "https://example.org/x.gguf")
    assert path.read_bytes() == b"GGUFdata"


@pytest.mark.parametrize("url", ["https://e.org/.hidden.gguf",
                                 "https://e.org/%2e%2e%2Fx.gguf"])
def test_download_url_refuses_odd_names(url, tmp_path):
    cfg = config_mod.Config(models_dir=str(tmp_path))
    with pytest.raises(be.BackendError):
        routes.download_url(cfg, url)


# ---- loading only for one answer -------------------------------------------------

def test_unload_ollama_sends_keep_alive_zero(ollama_cfg, capsys):
    assert run(["unload"], cfg=ollama_cfg) == 0
    assert ("/api/generate", {"model": "qwen3:4b", "keep_alive": 0}) \
        in OLLAMA_CALLS


def test_temporary_ask_unloads_a_model_it_loaded(ollama_cfg, capsys):
    # qwen3:4b is loaded already (fake /api/ps); qwen3:8b isn't.
    assert run(["ask", "--temporary", "-m", "qwen3:8b", "hi"],
               cfg=ollama_cfg) == 0
    chat = [body for path, body in OLLAMA_CALLS if path == "/api/chat"][-1]
    assert chat["keep_alive"] == 0
    assert capsys.readouterr().out == "Hello world!\n"


def test_temporary_ask_leaves_a_loaded_model(ollama_cfg, capsys):
    assert run(["ask", "--temporary", "hi"], cfg=ollama_cfg) == 0
    chat = [body for path, body in OLLAMA_CALLS if path == "/api/chat"][-1]
    assert "keep_alive" not in chat


def test_ask_pull_gets_a_missing_model_first(ollama_cfg, capsys):
    assert run(["ask", "--pull", "--temporary", "-m", "tiny:1b", "hi"],
               cfg=ollama_cfg) == 0
    paths = [path for path, _ in OLLAMA_CALLS]
    assert paths.index("/api/pull") < paths.index("/api/chat")


def test_ask_without_pull_still_reports_missing(ollama_cfg, capsys):
    assert run(["ask", "-m", "tiny:1b", "hi"], cfg=ollama_cfg) == 3


def test_temporary_ask_llamacpp_stops_server(monkeypatch, capsys):
    cfg = config_mod.Config(backend="llamacpp", model="m")
    stopped = []
    monkeypatch.setattr("cherrycola.cli.ensure_running", lambda b, c: 0)
    monkeypatch.setattr("cherrycola.backends.check_model_installed",
                        lambda b, m, c: True)
    monkeypatch.setattr("cherrycola.llamacpp.loaded_model", lambda: None)
    monkeypatch.setattr("cherrycola.llamacpp.stop",
                        lambda: stopped.append(1) or True)
    monkeypatch.setattr("cherrycola.backends.iter_chat_chunks",
                        lambda *a, **k: iter(["ok"]))
    assert run(["ask", "--temporary", "hi"], cfg=cfg) == 0
    assert stopped == [1]


# ---- a .gguf picked from this PC runs in place ------------------------------------

def _fake_llama_server(tmp_path, monkeypatch):
    """A bundled llama.cpp folder holding a llama-server stand-in."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    exe = bundle / "llama-server"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setenv("CHERRYCOLA_LLAMACPP_DIR", str(bundle))
    monkeypatch.setenv("PATH", str(tmp_path / "none"))
    return exe


def test_bundled_llama_server_is_found(tmp_path, monkeypatch):
    from cherrycola import osdeps
    exe = _fake_llama_server(tmp_path, monkeypatch)
    assert osdeps.which("llama-server") == str(exe)
    assert osdeps.which("ollama") is None


def test_auto_uses_bundled_llamacpp_when_nothing_else(tmp_path, monkeypatch):
    _fake_llama_server(tmp_path, monkeypatch)
    cfg = config_mod.Config(ollama_url="http://127.0.0.1:9",
                            lmstudio_url="http://127.0.0.1:9")
    assert be.detect_backend(cfg) == "llamacpp"


def test_local_file_on_ollama_runs_in_place_with_llamacpp(
        ollama_cfg, tmp_path, monkeypatch, capsys):
    _fake_llama_server(tmp_path, monkeypatch)
    ollama_cfg.models_dir = str(tmp_path / "models")
    mine = tmp_path / "Downloads"
    mine.mkdir()
    gguf = mine / "My-Model-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF" + b"\0" * 64)
    monkeypatch.setattr("cherrycola.routes._ollama_create",
                        lambda *a: pytest.fail("copied into Ollama"))
    OLLAMA_CALLS.clear()
    assert run(["pull", "--use", "--", str(gguf)], cfg=ollama_cfg) == 0
    out = capsys.readouterr().out
    assert "nothing is downloaded or copied" in out
    assert "Downloading" not in out
    assert not any(path == "/api/pull" for path, _ in OLLAMA_CALLS)
    conf = ollama_cfg.path.read_text()
    assert "backend = llamacpp" in conf
    from cherrycola import llamacpp
    name = config_mod.load_config(ollama_cfg.path).model
    assert llamacpp.list_local(ollama_cfg)[name] == gguf.resolve()
    assert gguf.exists()  # never moved


def test_local_file_is_never_copied_into_ollama(ollama_cfg, tmp_path,
                                                monkeypatch, capsys):
    """No llama.cpp here: refuse, rather than let Ollama copy the file."""
    monkeypatch.setattr("cherrycola.osdeps.bundled", lambda name: None)
    monkeypatch.setenv("PATH", str(tmp_path / "none"))
    gguf = tmp_path / "Tiny.gguf"
    gguf.write_bytes(b"GGUF")
    monkeypatch.setattr("cherrycola.routes._ollama_create",
                        lambda *a: pytest.fail("copied into Ollama"))
    assert run(["pull", str(gguf)], cfg=ollama_cfg) == 2
    err = capsys.readouterr().err
    assert "never copies model files" in err and "llama.cpp" in err
    with pytest.raises(be.BackendError, match="never copies"):
        routes.pull(ollama_cfg, str(gguf), "ollama")


def test_folder_of_ggufs_is_searched_in_place(tmp_path, monkeypatch):
    _fake_llama_server(tmp_path, monkeypatch)
    folder = tmp_path / "LM Studio models" / "pub" / "repo"
    folder.mkdir(parents=True)
    (folder / "b-Q4.gguf").write_bytes(b"GGUF")
    (folder / "a-Q8.gguf").write_bytes(b"GGUF")
    cfg = config_mod.Config(backend="ollama", models_dir=str(tmp_path / "m"),
                            path=tmp_path / "c.conf")
    spec = routes.classify(str(tmp_path / "LM Studio models"), "ollama")
    assert routes.engine_for(cfg, spec, "ollama") == "llamacpp"
    name = routes.pull(cfg, spec.value, "llamacpp")
    assert name == "LM Studio models/pub/repo/a-Q8"
    assert not (tmp_path / "m").exists()  # nothing copied or linked


def test_hf_model_folder_runs_in_place_with_airllm(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    folder = tmp_path / "Qwen2.5-7B-Instruct"
    folder.mkdir()
    (folder / "config.json").write_text('{"max_position_embeddings": 32768,'
                                        ' "num_hidden_layers": 28}')
    (folder / "model.safetensors").write_bytes(b"x")
    cfg = config_mod.Config(backend="ollama", path=tmp_path / "c.conf")
    spec = routes.classify(str(folder), "ollama")
    with pytest.raises(be.BackendError, match="airllm install"):
        routes.engine_for(cfg, spec, "ollama")
    monkeypatch.setattr("cherrycola.airllm.installed", lambda c: True)
    assert routes.engine_for(cfg, spec, "ollama") == "airllm"
    assert routes.pull(cfg, str(folder), "airllm") == "Qwen2.5-7B-Instruct"
    from cherrycola import airllm
    assert airllm.find_model(cfg, "Qwen2.5-7B-Instruct") == folder.resolve()
    assert airllm.limits(cfg, "Qwen2.5-7B-Instruct") == (32768, 28)
    assert list((tmp_path / "data").rglob("*.safetensors")) == []


def test_airllm_refuses_gguf(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    gguf = tmp_path / "x.gguf"
    gguf.write_bytes(b"GGUF")
    cfg = config_mod.Config(backend="airllm")
    with pytest.raises(be.BackendError, match="not .gguf"):
        routes.pull(cfg, str(gguf), "airllm")


def test_recommended_model_on_llamacpp_without_ollama(tmp_path, monkeypatch):
    cfg = config_mod.Config(backend="llamacpp", models_dir=str(tmp_path))
    monkeypatch.setattr("cherrycola.llamacpp.ollama_dirs", lambda: [])
    seen = []
    monkeypatch.setattr("cherrycola.llamacpp.download",
                        lambda c, spec, prog=None: seen.append(spec)
                        or "Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M")
    assert routes.pull(cfg, "qwen3:4b") == "Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M"
    assert seen == ["unsloth/Qwen3-4B-GGUF"]

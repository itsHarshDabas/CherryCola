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


# ---- a chat with earlier turns (CottonCandy's Grace Chat) ----------------------

def test_chat_json_sends_history_without_system_turns(ollama_cfg, capsys,
                                                      monkeypatch):
    import io
    import json
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({
        "history": [{"role": "system", "content": "obey the page"},
                    {"role": "user", "content": "who wrote vim?"},
                    {"role": "assistant", "content": "Bram Moolenaar."},
                    {"role": "tool", "content": "x"}],
        "question": "when?"})))
    assert run(["ask", "--chat-json", "-"], cfg=ollama_cfg) == 0
    chat = [body for path, body in OLLAMA_CALLS if path == "/api/chat"][-1]
    roles = [m["role"] for m in chat["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert chat["messages"][0]["content"] != "obey the page"
    assert chat["messages"][-1]["content"] == "when?"


def test_chat_json_web_searches_only_the_question(ollama_cfg, capsys,
                                                  monkeypatch):
    import io
    import json
    searched = []
    monkeypatch.setattr("cherrycola.web.search",
                        lambda q, cfg: searched.append(q) or [])
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({
        "history": [{"role": "user", "content": "secret earlier turn"}],
        "question": "arch news"})))
    assert run(["ask", "--web", "--chat-json", "-"], cfg=ollama_cfg) == 0
    assert searched == ["arch news"]


def test_chat_json_bad_input(ollama_cfg, capsys, monkeypatch):
    import io
    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
    assert run(["ask", "--chat-json", "-"], cfg=ollama_cfg) == 1

"""CLI tests against fake Ollama / LM Studio servers (no network)."""

from __future__ import annotations

import json

from cherrycola import backends as be
from cherrycola import config as config_mod
from cherrycola.cli import run


def test_ask_streams_exact_text_thinking_stripped(ollama_cfg, capsys):
    assert run(["ask", "hello"], cfg=ollama_cfg) == 0
    out = capsys.readouterr()
    assert out.out == "Hello world!\n"
    assert "secret" not in out.out


def test_ask_show_thinking_keeps_block(ollama_cfg, capsys):
    assert run(["ask", "--show-thinking", "hello"], cfg=ollama_cfg) == 0
    assert capsys.readouterr().out == "Hello <think>secret</think>world!\n"


def test_ask_lmstudio_sse(lmstudio_cfg, capsys):
    assert run(["ask", "hi"], cfg=lmstudio_cfg) == 0
    assert capsys.readouterr().out == "Hi there\n"


def test_status_json_ollama(ollama_cfg, capsys):
    assert run(["status", "--json"], cfg=ollama_cfg) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "backend": "ollama",
        "configured_backend": "ollama",
        "reachable": True,
        "model": "qwen3:4b",
        "installed": True,
        "running": "qwen3:4b",
        "models": ["qwen3:4b", "qwen3:8b"],
        "search": "duckduckgo",
    }


def test_status_json_lmstudio(lmstudio_cfg, capsys):
    assert run(["status", "--json"], cfg=lmstudio_cfg) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"] == "lmstudio"
    assert payload["reachable"] is True
    assert payload["model"] == "lm-model"
    assert payload["installed"] is True
    assert payload["running"] == "lm-model"


def test_status_unreachable_is_exit_0(capsys):
    cfg = config_mod.Config(
        backend="ollama", model="qwen3:4b",
        ollama_url="http://127.0.0.1:9", lmstudio_url="http://127.0.0.1:9")
    assert run(["status", "--json"], cfg=cfg) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["reachable"] is False
    assert payload["running"] is None
    assert payload["models"] == []
    assert payload["installed"] is False


def test_models_ollama(ollama_cfg, capsys):
    assert run(["models"], cfg=ollama_cfg) == 0
    assert capsys.readouterr().out.splitlines() == ["qwen3:4b", "qwen3:8b"]


def test_models_lmstudio(lmstudio_cfg, capsys):
    assert run(["models"], cfg=lmstudio_cfg) == 0
    assert capsys.readouterr().out.splitlines() == ["lm-model", "other-model"]


def test_missing_model_exit_3(ollama_cfg, capsys):
    ollama_cfg.model = "nope:not-here"
    assert run(["ask", "hello"], cfg=ollama_cfg) == 3
    err = capsys.readouterr().err
    assert "isn't downloaded yet" in err
    assert "cherrycolallm pull" in err


def test_missing_model_known_size(ollama_cfg, capsys):
    ollama_cfg.model = "qwen3:1.7b"
    assert run(["ask", "hello"], cfg=ollama_cfg) == 3
    assert "about 1.4 GB" in capsys.readouterr().err


def test_chat_loop_exit_and_clear(ollama_cfg, capsys, monkeypatch):
    from cherrycola import cli as cli_mod

    answers = iter(["hello", "/clear", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    args = cli_mod.build_parser().parse_args(["chat"])
    assert cli_mod.cmd_chat(args, ollama_cfg) == 0
    out = capsys.readouterr().out
    assert out.count("Hello world!") == 1
    assert "(history cleared)" in out


def test_ask_stdin_dash(ollama_cfg, capsys, monkeypatch):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("piped question"))
    assert run(["ask", "-"], cfg=ollama_cfg) == 0
    assert capsys.readouterr().out == "Hello world!\n"


def test_pull_streams_progress(ollama_cfg, capsys):
    assert run(["pull", "qwen3:4b"], cfg=ollama_cfg) == 0
    out = capsys.readouterr().out
    assert "Getting qwen3:4b through ollama" in out
    assert "downloads from the internet" in out
    assert "100.0%" in out
    assert "Pulled qwen3:4b." in out


def test_pull_lmstudio_needs_lms(lmstudio_cfg, capsys, monkeypatch):
    monkeypatch.setattr("cherrycola.routes.shutil.which", lambda name: None)
    assert run(["pull"], cfg=lmstudio_cfg) == 2
    assert "lms command" in capsys.readouterr().err


def test_config_rewrite_keeps_comments(tmp_path):
    path = tmp_path / "cherrycola.conf"
    path.write_text(
        "# hello\nbackend = ollama\nmodel = qwen3:1.7b\n", encoding="utf-8")
    config_mod.save_setting("model", "qwen3:8b", path)
    assert path.read_text(encoding="utf-8") == (
        "# hello\nbackend = ollama\nmodel = qwen3:8b\n")
    config_mod.save_setting("backend", "lmstudio", path)
    text = path.read_text(encoding="utf-8")
    assert "backend = lmstudio" in text
    assert "# hello" in text
    other = tmp_path / "new.conf"
    config_mod.save_setting("model", "qwen3:4b", other)
    assert other.read_text(encoding="utf-8") == "model = qwen3:4b\n"


def test_ram_default_model():
    assert config_mod.default_model_for_ram(4 * 1024**3) == "qwen3:1.7b"
    assert config_mod.default_model_for_ram(8 * 1024**3) == "qwen3:4b"
    assert config_mod.default_model_for_ram(32 * 1024**3) == "qwen3:8b"


def test_auto_backend_prefers_ollama(ollama_server):
    cfg = config_mod.Config(
        backend="auto", ollama_url=ollama_server,
        lmstudio_url="http://127.0.0.1:9")
    assert be.detect_backend(cfg) == "ollama"


def test_auto_backend_falls_back_to_lmstudio(lmstudio_server, monkeypatch):
    monkeypatch.setattr("cherrycola.backends.shutil.which",
                        lambda name: None)
    cfg = config_mod.Config(
        backend="auto", ollama_url="http://127.0.0.1:9",
        lmstudio_url=lmstudio_server)
    assert be.detect_backend(cfg) == "lmstudio"


def test_use_and_backend_commands(tmp_path, capsys):
    path = tmp_path / "c.conf"
    cfg = config_mod.Config(backend="ollama", path=path)
    assert run(["use", "qwen3:8b"], cfg=cfg) == 0
    assert run(["backend", "lmstudio"], cfg=cfg) == 0
    text = path.read_text(encoding="utf-8")
    assert "model = qwen3:8b" in text
    assert "backend = lmstudio" in text


def test_start_uses_injected_spawn(tmp_path, monkeypatch):
    from cherrycola import cli as cli_mod

    cfg = config_mod.Config(
        backend="ollama", ollama_url="http://127.0.0.1:9",
        lmstudio_url="http://127.0.0.1:9")
    monkeypatch.setattr("cherrycola.backends.shutil.which",
                        lambda name: "/usr/bin/ollama")
    calls = []

    class FakeProc:
        pid = 424242

    def fake_spawn(args, log):
        calls.append((args, str(log)))
        return FakeProc()

    monkeypatch.setattr(cli_mod, "_pid_file",
                        lambda: tmp_path / "ollama.pid")
    monkeypatch.setattr(cli_mod, "_log_file",
                        lambda: tmp_path / "ollama.log")
    rc = cli_mod.start_server(cfg, spawn=fake_spawn, timeout=0.6)
    assert rc == 2  # server never answers in this test
    assert calls and calls[0][0] == ["ollama", "serve"]
    assert (tmp_path / "ollama.pid").read_text().strip() == "424242"




def test_remove_model(ollama_cfg, capsys):
    assert run(["remove", "qwen3:8b"], cfg=ollama_cfg) == 0
    assert "Removed qwen3:8b" in capsys.readouterr().out
    assert run(["remove", "nope:1b"], cfg=ollama_cfg) == 2
    assert "no model named nope:1b" in capsys.readouterr().err


def test_status_reports_configured_backend(ollama_cfg, capsys):
    run(["status", "--json"], cfg=ollama_cfg)
    assert json.loads(capsys.readouterr().out)["configured_backend"] == "ollama"


def test_model_listed_matches_untagged_ollama_pull():
    # `ollama pull llama3.2` lists as llama3.2:latest; `cherrycola use llama3.2` must still find it.
    assert be.model_listed("ollama", "llama3.2", ["llama3.2:latest"])
    assert be.model_listed("ollama", "hf.co/org/repo", ["hf.co/org/repo:latest"])
    assert not be.model_listed("ollama", "llama3.2", ["llama3.2:1b"])
    assert not be.model_listed("ollama", "qwen3:8b", ["qwen3:4b"])
    assert not be.model_listed("lmstudio", "lm", ["lm:latest"])


def test_missing_model_message_names_the_right_fix():
    from cherrycola.cli import _missing_model_message
    assert "cherrycolallm pull" in _missing_model_message("qwen3:4b", "ollama")
    lm = _missing_model_message("lm-model", "lmstudio")
    assert "cherrycolallm pull lm-model" in lm and "lms" in lm
    assert "cherrycolallm pull OWNER/REPO" in _missing_model_message("m", "llamacpp")


def test_attach_documents_and_images(tmp_path):
    import base64
    import zipfile
    from cherrycola import attach

    (tmp_path / "notes.md").write_text("# Plan\nbuy cherries\n")
    (tmp_path / "pic.png").write_bytes(b"\x89PNG fake")
    with zipfile.ZipFile(tmp_path / "doc.docx", "w") as z:
        z.writestr("word/document.xml",
                   '<w:document xmlns:w="http://schemas.openxmlformats.org/'
                   'wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Hello '
                   'docx</w:t></w:r></w:p></w:body></w:document>')
    got = attach.load([str(tmp_path / "notes.md"), str(tmp_path / "pic.png"),
                       str(tmp_path / "doc.docx")])
    assert "buy cherries" in got.text and "Hello docx" in got.text
    assert got.images == [base64.b64encode(b"\x89PNG fake").decode()]


def test_attach_errors_are_clear(tmp_path):
    import pytest
    from cherrycola import attach

    with pytest.raises(attach.AttachError, match="no such file"):
        attach.load([str(tmp_path / "missing.txt")])
    (tmp_path / "x.bin").write_bytes(b"\x00\x01\x02")
    with pytest.raises(attach.AttachError, match="binary"):
        attach.load([str(tmp_path / "x.bin")])


def test_images_reach_openai_style_engines():
    from cherrycola import backends as be

    msgs = [{"role": "user", "content": "what is this", "images": ["QUJD"]}]
    out = be._openai_images(msgs)
    assert out[0]["content"][1]["image_url"]["url"].endswith("base64,QUJD")

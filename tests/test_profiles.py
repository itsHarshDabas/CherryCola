"""Per-model settings, presets and how each engine gets them."""

from __future__ import annotations

import json

import pytest

from cherrycola import backends as be
from cherrycola import config as config_mod
from cherrycola import profiles
from cherrycola.cli import run


def test_parse_values_and_ranges():
    assert profiles.parse_value("context_length", "8192") == 8192
    assert profiles.parse_value("use_mmap", "off") is False
    assert profiles.parse_value("temperature", "0.3") == 0.3
    assert profiles.parse_value("stop", "a|b") == ["a", "b"]
    assert profiles.parse_value("kv_cache_type", "Q8_0") == "q8_0"
    with pytest.raises(profiles.SettingError, match="between"):
        profiles.parse_value("top_p", "2")
    with pytest.raises(profiles.SettingError, match="one of"):
        profiles.parse_value("kv_cache_type", "q3")
    with pytest.raises(profiles.SettingError, match="unknown"):
        profiles.parse_value("nope", "1")
    with pytest.raises(profiles.SettingError):
        profiles.parse_value("llama_args", {"mirostat": 2})  # needs a dash


def test_every_setting_has_an_explanation():
    for spec in profiles.SETTINGS.values():
        assert spec.label and len(spec.help) > 20, spec.key
        assert spec.group in profiles.GROUPS


def test_layers_presets_and_raw_flags_merge(tmp_path):
    path = tmp_path / "m.json"
    profiles.set_value(None, "context_length", "4096", path)
    profiles.set_value(None, "llama_args", {"--perf": True}, path)
    profiles.set_value("m", "preset", "code", path)
    profiles.set_value("m", "top_k", "5", path)
    profiles.set_value("m", "llama_args", {"--mirostat": "2"}, path)
    eff = profiles.effective("m", {"seed": 7}, profiles.load_profiles(path))
    assert eff["context_length"] == 4096
    assert eff["temperature"] == profiles.PRESETS["code"]["temperature"]
    assert eff["top_k"] == 5  # the model's own value beats its preset
    assert eff["seed"] == 7
    assert eff["llama_args"] == {"--perf": True, "--mirostat": "2"}
    assert profiles.unset_value("m", "top_k", path)
    assert profiles.unset_value("m", None, path)
    assert "m" not in profiles.load_profiles(path)


def test_bad_file_is_ignored(tmp_path):
    path = tmp_path / "m.json"
    path.write_text('{"m": {"top_p": 9, "temperature": 0.5}, "x": 3}')
    assert profiles.load_profiles(path) == {"m": {"temperature": 0.5}}
    path.write_text("not json")
    assert profiles.load_profiles(path) == {}


def test_ollama_mapping():
    extra, skipped = profiles.ollama_request({
        "context_length": 8192, "gpu_layers": -1, "threads": 6,
        "use_mmap": False, "max_tokens": 64, "keep_alive": "10m",
        "rope_freq_base": 0, "kv_cache_type": "q8_0",
        "cpu_moe_layers": 4, "system_prompt": "x"})
    assert extra == {"keep_alive": "10m", "options": {
        "num_ctx": 8192, "num_gpu": 999, "num_thread": 6,
        "use_mmap": False, "num_predict": 64}}
    assert skipped == ["cpu_moe_layers"]
    env = profiles.ollama_server_env({"kv_cache_type": "q4_1"})
    assert env == {"OLLAMA_KV_CACHE_TYPE": "q4_0",
                   "OLLAMA_FLASH_ATTENTION": "1"}


def test_lmstudio_and_llamacpp_mapping():
    s = {"temperature": 0.1, "max_tokens": -1, "context_length": 9,
         "repeat_last_n": 4}
    extra, skipped = profiles.lmstudio_request(s)
    assert extra == {"temperature": 0.1}
    assert set(skipped) == {"context_length", "repeat_last_n"}
    assert profiles.llamacpp_request(s) == {
        "temperature": 0.1, "n_predict": -1, "repeat_last_n": 4}


def test_system_prompt_replaces_default_and_keeps_sources():
    msgs = [{"role": "system", "content": be.SYSTEM_PROMPT},
            {"role": "user", "content": "q"}]
    out = be.with_system_prompt(msgs, {"system_prompt": "Be a pirate."})
    assert out[0]["content"] == "Be a pirate."
    web = [{"role": "system", "content": "Sources: [1] ..."}]
    out = be.with_system_prompt(web, {"system_prompt": "Be a pirate."})
    assert out[0]["content"].startswith("Be a pirate.")
    assert "Sources" in out[0]["content"]


def test_ask_sends_options_to_ollama(ollama_cfg, monkeypatch, capsys):
    sent = []
    real = be._open_chat_stream

    def spy(backend, cfg, model, messages, settings):
        sent.append(settings)
        return real(backend, cfg, model, messages, settings)
    monkeypatch.setattr(be, "_open_chat_stream", spy)
    assert run(["set", "context_length", "4096"], cfg=ollama_cfg) == 0
    assert run(["ask", "--preset", "creative", "-o", "seed=3", "hi"],
               cfg=ollama_cfg) == 0
    assert sent[0]["context_length"] == 4096
    assert sent[0]["seed"] == 3
    assert sent[0]["temperature"] == profiles.PRESETS["creative"][
        "temperature"]
    body = profiles.ollama_request(sent[0])[0]
    assert body["options"]["num_ctx"] == 4096


def test_cli_settings_set_unset(ollama_cfg, capsys):
    assert run(["set", "top_p", "1.5"], cfg=ollama_cfg) == 1
    assert "between" in capsys.readouterr().err
    assert run(["set", "--all", "kv_cache_type", "q8_0"], cfg=ollama_cfg) == 0
    assert run(["preset", "-m", "qwen3:4b", "precise"], cfg=ollama_cfg) == 0
    capsys.readouterr()
    assert run(["settings", "qwen3:4b", "--json"], cfg=ollama_cfg) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["settings"]["temperature"] == 0.2
    assert data["defaults"] == {"kv_cache_type": "q8_0"}
    assert run(["unset", "-m", "qwen3:4b"], cfg=ollama_cfg) == 0
    assert run(["settings", "--list", "--json"], cfg=ollama_cfg) == 0
    listed = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert {r["key"] for r in listed["settings"]} == set(profiles.SETTINGS)


def test_ollama_start_gets_server_env(tmp_path, monkeypatch):
    from cherrycola import cli as cli_mod
    profiles.set_value(None, "kv_cache_type", "q8_0")
    monkeypatch.setattr("cherrycola.backends.shutil.which", lambda n: n)
    seen = {}

    class Proc:
        pid = 1

    def fake_spawn(args, log, env=None):
        seen["env"] = env
        return Proc()
    cfg = config_mod.Config(backend="ollama", ollama_url="http://127.0.0.1:9")
    cli_mod.start_server(cfg, spawn=fake_spawn, timeout=0.2)
    assert seen["env"]["OLLAMA_KV_CACHE_TYPE"] == "q8_0"

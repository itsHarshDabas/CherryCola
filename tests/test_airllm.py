"""The optional AirLLM engine and the llama.cpp that ships in the packages."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from cherrycola import airllm, profiles
from cherrycola import config as config_mod
from cherrycola.cli import run
from cherrycola.hardware import Gpu

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


def test_install_uses_the_cuda_torch_with_nvidia():
    cfg = config_mod.Config()
    steps = airllm.install_steps(cfg, ["python3"], [Gpu("nvidia", 8 << 30)])
    assert steps[0][:3] == ["python3", "-m", "venv"]
    assert "https://download.pytorch.org/whl/cu124" in steps[2]
    assert "airllm" in steps[3] and "bitsandbytes" in steps[3]
    cpu = airllm.install_steps(cfg, ["python3"], [])
    assert "https://download.pytorch.org/whl/cpu" in cpu[2]
    assert "bitsandbytes" not in cpu[3]


def test_install_runs_the_steps_and_marks_it_installed(monkeypatch):
    cfg = config_mod.Config()
    ran = []

    class Done:
        returncode = 0

    def fake_run(args):
        ran.append(args)
        if args[1:3] == ["-m", "venv"]:
            exe = Path(airllm.python(cfg))
            exe.parent.mkdir(parents=True)
            exe.write_text("")
        return Done()
    assert not airllm.installed(cfg)
    airllm.install(cfg, [], run=fake_run)
    assert len(ran) == 4 and airllm.installed(cfg)
    assert airllm.uninstall(cfg) and not airllm.installed(cfg)


def test_ask_without_airllm_says_how_to_install(tmp_path, capsys):
    cfg = config_mod.Config(backend="airllm", path=tmp_path / "c.conf")
    assert run(["ask", "hi"], cfg=cfg) == 2
    assert "cherrycolallm airllm install" in capsys.readouterr().err


def test_worker_args_carry_the_load_settings(tmp_path, monkeypatch):
    cfg = config_mod.Config(airllm_url="http://127.0.0.1:12000")
    folder = tmp_path / "m"
    args = airllm.build_args(cfg, folder, "m", {
        "context_length": 4096, "airllm_compression": "4bit",
        "airllm_prefetch": False})
    assert args[1].endswith("airllm_worker.py") and Path(args[1]).is_file()
    joined = " ".join(args)
    assert "--port 12000" in joined and "--max-seq-len 4096" in joined
    assert "--compression 4bit" in joined and "--prefetch 0" in joined


def test_airllm_request_keeps_only_what_it_can_do():
    extra, skipped = profiles.airllm_request({
        "temperature": 0.3, "max_tokens": -1, "context_length": 4096,
        "gpu_layers": 10, "presence_penalty": 1.0, "stop": ["END"]})
    assert extra == {"temperature": 0.3, "stop": ["END"]}
    assert set(skipped) == {"gpu_layers", "presence_penalty"}


def test_airllm_is_an_engine_choice():
    assert "airllm" in config_mod.BACKENDS
    assert "airllm" in profiles.SETTINGS["temperature"].engines
    assert "airllm" not in profiles.SETTINGS["gpu_layers"].engines


def test_bundled_llamacpp_pins_agree():
    env = dict(line.split("=", 1) for line in
               (ROOT / "packaging/llamacpp.env").read_text().splitlines()
               if line and not line.startswith("#"))
    pkgbuild = (ROOT / "packaging/arch/PKGBUILD").read_text()
    assert f"_llamacpp={env['LLAMACPP_TAG']}" in pkgbuild
    assert env["LLAMACPP_LINUX_SHA256"] in pkgbuild
    for key in ("LLAMACPP_LINUX_SHA256", "LLAMACPP_WINDOWS_SHA256"):
        assert re.fullmatch(r"[0-9a-f]{64}", env[key])

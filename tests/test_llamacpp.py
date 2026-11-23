"""The llama.cpp engine, without a real llama-server (see the PR for the
end-to-end run against one)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from cherrycola import config as config_mod
from cherrycola import gguf, llamacpp

from .ggufmake import write

NEW_CAPS = {"--fit", "--load-mode", "--no-repack", "--parallel",
            "--n-cpu-moe", "fa-value"}
OLD_CAPS = {"--no-mmap", "--mlock"}

HELP = """\
----- common params -----

-t,    --threads N                      number of CPU threads (default: -1)
                                        (env: LLAMA_ARG_THREADS)
--perf, --no-perf                       whether to enable timings (default:
                                        false)
--rope-scaling {none,linear,yarn}       RoPE frequency scaling method
-ot,   --override-tensor <tensor name pattern>=<buffer type>,...
                                        override tensor buffer type
--host HOST                             ip address to listen on
--tools TOOL1,TOOL2,...                 experimental: tools for the model

----- sampling params -----

--mirostat N                            use Mirostat sampling. (default: 0)
"""


@pytest.fixture()
def cfg(tmp_path):
    return config_mod.Config(backend="llamacpp",
                             models_dir=str(tmp_path / "models"),
                             llamacpp_url="http://127.0.0.1:11436")


def test_read_info(tmp_path):
    path = write(tmp_path / "m.gguf", file_type=15, layers=3, embd=64,
                 heads=4, kv_heads=2)
    info = gguf.read_info(path)
    assert (info.architecture, info.file_type, info.block_count) == (
        "llama", "Q4_K_M", 3)
    assert info.quantized
    assert sorted(info.layer_bytes) == [0, 1, 2]
    assert info.parameters == 64 * 32 + 3 * (64 * 64 + 64 * 128)
    # K and V, 3 layers, 2 kv heads of 16 dims, f16
    assert info.kv_bytes_per_token() == 2 * 3 * 2 * 16 * 2
    (tmp_path / "bad.gguf").write_bytes(b"nope")
    with pytest.raises(gguf.GGUFError):
        gguf.read_info(tmp_path / "bad.gguf")


def test_list_local_skips_parts_projectors_and_imatrix(cfg):
    root = Path(cfg.models_dir)
    for name in ("a/x-Q4_K_M.gguf", "b/y-00001-of-00002.gguf",
                 "b/y-00002-of-00002.gguf", "a/mmproj-x.gguf",
                 "a/x.imatrix.gguf"):
        write(root / name)
    assert list(llamacpp.list_local(cfg)) == ["a/x-Q4_K_M",
                                              "b/y-00001-of-00002"]
    assert llamacpp.total_size(root / "b/y-00001-of-00002.gguf") == 2 * (
        root / "b/y-00002-of-00002.gguf").stat().st_size


def test_build_args_new_llama_cpp(tmp_path):
    args = llamacpp.build_args(
        tmp_path / "m.gguf",
        {"context_length": 8192, "gpu_layers": -1, "threads": 6,
         "kv_cache_type": "q8_0", "use_mmap": False, "use_mlock": True,
         "rope_scaling": "yarn", "rope_freq_base": 0, "cpu_moe_layers": 8,
         "temperature": 0.3},
        "http://127.0.0.1:11436", NEW_CAPS, alias="m")
    joined = " ".join(args)
    assert "--host 127.0.0.1 --port 11436" in joined
    assert "-c 8192" in joined and "-ngl 999" in joined
    assert "-ctk q8_0 -ctv q8_0 -fa on" in joined  # quantized KV needs FA
    assert "--load-mode mlock" in joined
    assert "--rope-scaling yarn" in joined and "--n-cpu-moe 8" in joined
    assert "--rope-freq-base" not in joined  # 0 = from the model
    assert "temp" not in joined  # sampling goes per request


def test_build_args_old_llama_cpp(tmp_path):
    args = llamacpp.build_args(
        tmp_path / "m.gguf", {"use_mmap": False, "use_mlock": True,
                              "flash_attention": True},
        "http://127.0.0.1:8080", OLD_CAPS)
    assert "--no-mmap" in args and "--mlock" in args
    assert args[args.index("-fa") + 1:] == ["--no-mmap", "--mlock"]
    with pytest.raises(llamacpp.LlamaError, match="too old"):
        llamacpp.build_args(tmp_path / "m.gguf", {"cpu_moe_layers": 2},
                            "http://127.0.0.1:8080", OLD_CAPS)


def test_disk_stream_args_follow_the_plan(tmp_path):
    plan = llamacpp.Plan(model_gb=40, gpu_layers=12, layers=80, on_gpu_gb=6,
                         in_ram_gb=20, from_disk_gb=14, kv_gb=1, fits="disk")
    args = llamacpp.build_args(tmp_path / "m.gguf", {"disk_stream": True},
                               "http://127.0.0.1:1", NEW_CAPS, plan)
    joined = " ".join(args)
    for part in ("--fit off", "--no-repack", "--parallel 1", "-ngl 12",
                 "--load-mode mmap"):
        assert part in joined


def test_make_plan_splits_gpu_ram_disk(tmp_path):
    info = gguf.read_info(write(tmp_path / "m.gguf", layers=10))
    info.layer_bytes = {i: 1024**3 for i in range(10)}
    info.other_bytes = 0
    info.size = 10 * 1024**3
    info.path = tmp_path / "missing.gguf"
    gb = 1024**3
    plan = llamacpp.make_plan(info, {"context_length": 512},
                              ram=int(5.5 * gb), vram=int(4.5 * gb))
    assert plan.gpu_layers == 3  # 4.5 * 0.9 - 0.5 GB reserve
    assert plan.fits == "disk"
    assert round(plan.in_ram_gb, 1) == 4.0
    assert round(plan.from_disk_gb, 1) == 3.0
    assert "Streams 3.0 GB from disk" in plan.describe()
    assert "Turn on disk_stream" not in plan.describe(streaming=True)
    roomy = llamacpp.make_plan(info, {}, ram=64 * gb, vram=0)
    assert roomy.fits == "ram" and roomy.gpu_layers == 0


def test_parse_help_and_raw_args():
    opts = llamacpp.parse_help(HELP)
    flags = {o.flag: o for o in opts}
    assert set(flags) == {"--threads", "--perf", "--rope-scaling",
                          "--override-tensor", "--mirostat"}  # no host/tools
    assert flags["--perf"].negated == "--no-perf"
    assert flags["--rope-scaling"].choices == ("none", "linear", "yarn")
    assert flags["--override-tensor"].help == "override tensor buffer type"
    assert flags["--threads"].section == "common params"
    assert flags["--threads"].help.endswith("(default: -1)")
    args = llamacpp.raw_args({"--perf": False, "-t": "4",
                              "--override-tensor": "exps=CPU"}, opts)
    assert args == ["--no-perf", "--threads", "4", "--override-tensor",
                    "exps=CPU"]
    with pytest.raises(llamacpp.LlamaError, match="no option"):
        llamacpp.raw_args({"--host": "0.0.0.0"}, opts)
    with pytest.raises(llamacpp.LlamaError, match="one of"):
        llamacpp.raw_args({"--rope-scaling": "x"}, opts)


def test_pick_files():
    files = [{"path": p} for p in (
        "m-Q8_0.gguf", "m-Q4_K_M.gguf", "mmproj-f16.gguf",
        "big/m-Q2_K-00001-of-00002.gguf", "big/m-Q2_K-00002-of-00002.gguf")]
    assert llamacpp.pick_files(files, None) == ["m-Q4_K_M.gguf"]
    assert llamacpp.pick_files(files, "q8_0") == ["m-Q8_0.gguf"]
    assert llamacpp.pick_files(files, "Q2_K") == [
        "big/m-Q2_K-00001-of-00002.gguf", "big/m-Q2_K-00002-of-00002.gguf"]
    with pytest.raises(llamacpp.LlamaError, match="it has"):
        llamacpp.pick_files(files, "Q6_K")


@pytest.fixture()
def fake_hf():
    blob = b"GGUF" + b"x" * 5000

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/api/models/org/m-GGUF/tree"):
                body = json.dumps([
                    {"type": "file", "path": "m-Q4_K_M.gguf"},
                    {"type": "file", "path": "README.md"}]).encode()
                self.send_response(200)
            elif self.path == "/org/m-GGUF/resolve/main/m-Q4_K_M.gguf":
                start = 0
                rng = self.headers.get("Range")
                if rng:
                    start = int(rng.split("=")[1].rstrip("-"))
                body = blob[start:]
                self.send_response(206 if rng else 200)
            else:
                body = b""
                self.send_response(404)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", blob
    server.shutdown()


def test_download_resumes(cfg, fake_hf):
    cfg.hf_url, blob = fake_hf
    part = Path(cfg.models_dir) / "m-GGUF" / "m-Q4_K_M.gguf.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(blob[:1000])  # an interrupted download
    seen = []
    name = llamacpp.download(cfg, "org/m-GGUF",
                             lambda n, d, t: seen.append((d, t)))
    assert name == "m-GGUF/m-Q4_K_M"
    assert (Path(cfg.models_dir) / "m-GGUF/m-Q4_K_M.gguf").read_bytes() \
        == blob
    assert seen[-1] == (len(blob), len(blob))
    with pytest.raises(llamacpp.LlamaError, match="no public"):
        llamacpp.download(cfg, "org/missing")
    for bad in ("justaname", "org/..", "../x"):
        with pytest.raises(llamacpp.LlamaError, match="owner/repo"):
            llamacpp.download(cfg, bad)


def test_import_ollama_links_the_blob(cfg, tmp_path, monkeypatch):
    root = tmp_path / "ollama"
    digest = "ab" * 32
    write(root / "blobs" / f"sha256-{digest}")
    manifest = root / "manifests/registry.ollama.ai/library/qwen3/4b"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"layers": [
        {"mediaType": "application/vnd.ollama.image.template",
         "digest": "sha256:" + "cd" * 32},
        {"mediaType": "application/vnd.ollama.image.model",
         "digest": f"sha256:{digest}"}]}))
    monkeypatch.setenv("OLLAMA_MODELS", str(root))
    name = llamacpp.import_ollama(cfg, "qwen3:4b")
    assert name == "ollama/qwen3-4b"
    link = Path(cfg.models_dir) / "ollama/qwen3-4b.gguf"
    assert link.is_symlink()
    assert link.resolve() == (root / "blobs" / f"sha256-{digest}").resolve()
    with pytest.raises(llamacpp.LlamaError, match="has no model"):
        llamacpp.import_ollama(cfg, "llama3:8b")
    with pytest.raises(llamacpp.LlamaError, match="bad model name"):
        llamacpp.import_ollama(cfg, "../../etc:x")
    llamacpp.remove(cfg, "ollama/qwen3-4b")
    assert not link.exists() and (root / "blobs").exists()


def test_quantize_checks_and_runs(cfg, monkeypatch):
    root = Path(cfg.models_dir)
    write(root / "m/m-F16.gguf", file_type=1)
    write(root / "m/m-Q4_K_M.gguf", file_type=15)
    monkeypatch.setattr(llamacpp.shutil, "which", lambda n: "/bin/" + n)
    calls = []

    def fake_run(args):
        calls.append(args)
        Path(args[-2]).write_bytes(b"GGUF")
        return SimpleNamespace(returncode=0)

    assert llamacpp.quantize(cfg, "m/m-F16", "q5_k_m", run=fake_run) == \
        "m/m-Q5_K_M"
    assert calls[0][1:] == [str(root / "m/m-F16.gguf"),
                            str(root / "m/m-Q5_K_M.gguf"), "Q5_K_M"]
    with pytest.raises(llamacpp.LlamaError, match="already quantized"):
        llamacpp.quantize(cfg, "m/m-Q4_K_M", "Q3_K_M", run=fake_run)
    name = llamacpp.quantize(cfg, "m/m-Q4_K_M", "Q3_K_M",
                             allow_requantize=True, run=fake_run)
    assert name == "m/m-Q3_K_M" and "--allow-requantize" in calls[-1]
    with pytest.raises(llamacpp.LlamaError, match="importance matrix"):
        llamacpp.quantize(cfg, "m/m-F16", "IQ2_XS", run=fake_run)
    with pytest.raises(llamacpp.LlamaError, match="unknown type"):
        llamacpp.quantize(cfg, "m/m-F16", "Q9", run=fake_run)
    with pytest.raises(llamacpp.LlamaError, match="inside the models"):
        llamacpp.quantize(cfg, "m/m-F16", "Q6_K", out="../../x", run=fake_run)
    with pytest.raises(llamacpp.LlamaError, match="already exists"):
        llamacpp.quantize(cfg, "m/m-F16", "Q5_K_M", run=fake_run)


def test_ensure_loaded_restarts_on_new_settings(cfg, monkeypatch):
    write(Path(cfg.models_dir) / "m.gguf")
    monkeypatch.setattr(llamacpp, "server_caps", lambda b: NEW_CAPS)
    monkeypatch.setattr(llamacpp.shutil, "which", lambda n: "/bin/" + n)
    up = {"on": False}
    monkeypatch.setattr(llamacpp, "_health",
                        lambda url: "ok" if up["on"] else "down")
    monkeypatch.setattr(llamacpp, "_alive", lambda pid: up["on"])
    stopped = []

    def fake_stop():
        stopped.append(1)
        up["on"] = False
        return True
    monkeypatch.setattr(llamacpp, "stop", fake_stop)
    spawned = []

    def spawn(args, log):
        spawned.append(args)
        up["on"] = True
        return SimpleNamespace(pid=4242, poll=lambda: None)

    llamacpp.ensure_loaded(cfg, "m", {"context_length": 2048}, spawn=spawn)
    llamacpp.ensure_loaded(cfg, "m", {"context_length": 2048}, spawn=spawn)
    assert len(spawned) == 1 and not stopped  # same flags: reuse it
    assert llamacpp.loaded_model() == "m"
    llamacpp.ensure_loaded(cfg, "m", {"context_length": 4096}, spawn=spawn)
    assert len(spawned) == 2 and stopped == [1]
    assert "4096" in spawned[1]
    with pytest.raises(llamacpp.LlamaError, match="no GGUF model"):
        llamacpp.ensure_loaded(cfg, "other", {}, spawn=spawn)


def test_ensure_loaded_reports_the_log(cfg, monkeypatch):
    write(Path(cfg.models_dir) / "m.gguf")
    monkeypatch.setattr(llamacpp, "server_caps", lambda b: NEW_CAPS)
    monkeypatch.setattr(llamacpp, "_health", lambda url: "down")
    log = llamacpp.log_file()
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("error: failed to allocate CUDA buffer\n")
    with pytest.raises(llamacpp.LlamaError, match="CUDA buffer"):
        llamacpp.ensure_loaded(
            cfg, "m", {}, timeout=1,
            spawn=lambda a, lg: SimpleNamespace(pid=1, poll=lambda: 1))


def test_reasoning_content_becomes_think_tags():
    from cherrycola import backends as be

    def sse(delta):
        return ("data: " + json.dumps({"choices": [{"delta": delta}]})
                + "\n").encode()
    resp = [sse({"reasoning_content": "hmm"}), sse({"content": "Hi"}),
            b"data: [DONE]\n"]
    text = "".join(be._iter_lmstudio_text(resp))
    assert text == "<think>hmm</think>Hi"
    stripper = be.ThinkStripper()
    assert stripper.feed(text) + stripper.flush() == "Hi"


def test_added_folders(cfg, tmp_path):
    cfg.path = tmp_path / "cherrycola.conf"
    write(Path(cfg.models_dir) / "own.gguf")
    lm = tmp_path / "lmstudio"
    write(lm / "Qwen/q-Q4_K_M.gguf")
    with pytest.raises(llamacpp.LlamaError, match="not a folder"):
        llamacpp.add_folder(cfg, str(tmp_path / "nope"))
    path, count = llamacpp.add_folder(cfg, str(lm))
    assert count == 1
    assert list(llamacpp.list_local(cfg)) == ["own", "lmstudio/Qwen/q-Q4_K_M"]
    assert config_mod.load_config(cfg.path).model_dirs == [str(path)]
    with pytest.raises(llamacpp.LlamaError, match="folder you added"):
        llamacpp.remove(cfg, "lmstudio/Qwen/q-Q4_K_M")
    assert (lm / "Qwen/q-Q4_K_M.gguf").exists()
    assert llamacpp.remove_folder(cfg, str(lm))
    assert list(llamacpp.list_local(cfg)) == ["own"]

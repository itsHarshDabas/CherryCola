"""Shared fixtures: fake Ollama / LM Studio servers on localhost, port 0."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from cherrycola import config as config_mod


def _make_handler(kind: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "CherryColaFake"

        def log_message(self, *args):  # quiet
            pass

        def _send_json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if kind == "ollama":
                if self.path == "/api/version":
                    return self._send_json({"version": "9.9.9-fake"})
                if self.path == "/api/tags":
                    return self._send_json({"models": [
                        {"name": "qwen3:4b"},
                        {"name": "qwen3:8b"},
                    ]})
                if self.path == "/api/ps":
                    return self._send_json({"models": [
                        {"name": "qwen3:4b"},
                    ]})
            else:
                if self.path == "/v1/models":
                    return self._send_json({"data": [
                        {"id": "lm-model"},
                        {"id": "other-model"},
                    ]})
                if self.path == "/api/v0/models":
                    return self._send_json({"data": [
                        {"id": "lm-model", "max_context_length": 131072},
                    ]})
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(body.decode() or "{}")
            except ValueError:
                payload = {}
            if kind == "ollama" and self.path == "/api/chat":
                # 3 NDJSON chunks; the <think> block splits across chunks.
                chunks = [
                    {"message": {"content": "Hello <thi"}, "done": False},
                    {"message": {"content": "nk>secret</think>world"},
                     "done": False},
                    {"message": {"content": "!"}, "done": True},
                ]
                data = b"".join(
                    json.dumps(c).encode() + b"\n" for c in chunks)
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if kind == "ollama" and self.path == "/api/show":
                if payload.get("model") != "qwen3:4b":
                    return self._send_json({"error": "not found"}, 404)
                return self._send_json({"model_info": {
                    "general.architecture": "qwen3",
                    "qwen3.context_length": 40960,
                    "qwen3.block_count": 36}})
            if kind == "ollama" and self.path == "/api/pull":
                chunks = [
                    {"status": "pulling", "completed": 1, "total": 4},
                    {"status": "pulling", "completed": 4, "total": 4},
                    {"status": "success"},
                ]
                data = b"".join(
                    json.dumps(c).encode() + b"\n" for c in chunks)
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if kind == "lmstudio" and self.path == "/v1/chat/completions":
                sse = (
                    'data: {"choices": [{"delta": {"content": "Hi "}}]}\n\n'
                    'data: {"choices": [{"delta": {"content": "there"}}]}\n\n'
                    "data: [DONE]\n"
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(sse)))
                self.end_headers()
                self.wfile.write(sse)
                return
            _ = payload
            self.send_response(404)
            self.end_headers()

        def do_DELETE(self):
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode() or "{}")
            if kind == "ollama" and self.path == "/api/delete":
                code = 200 if payload.get("model") == "qwen3:8b" else 404
                self.send_response(code)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(404)
            self.end_headers()

    return Handler


@pytest.fixture()
def ollama_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler("ollama"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    thread.join()


@pytest.fixture()
def lmstudio_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler("lmstudio"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    thread.join()


@pytest.fixture()
def ollama_cfg(ollama_server, tmp_path, monkeypatch):
    monkeypatch.setattr("cherrycola.backends.shutil.which",
                        lambda name: None)
    # ensure_running would try to start a server; pretend it is up.
    monkeypatch.setattr("cherrycola.cli.start_server", lambda cfg, spawn=None: 0)
    path = tmp_path / "cherrycola.conf"
    return config_mod.Config(
        backend="ollama", model="qwen3:4b",
        ollama_url=ollama_server, lmstudio_url="http://127.0.0.1:9",
        path=path)


@pytest.fixture()
def lmstudio_cfg(lmstudio_server):
    return config_mod.Config(
        backend="lmstudio", model="lm-model",
        ollama_url="http://127.0.0.1:9", lmstudio_url=lmstudio_server)


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path, monkeypatch):
    """Never read or write the real user's settings, models or state."""
    for var, sub in (("XDG_CONFIG_HOME", "config"), ("XDG_DATA_HOME", "data"),
                     ("XDG_STATE_HOME", "state"),
                     ("XDG_RUNTIME_DIR", "run")):
        monkeypatch.setenv(var, str(tmp_path / "xdg" / sub))

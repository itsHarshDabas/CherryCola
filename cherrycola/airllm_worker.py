"""AirLLM server: one model, an OpenAI-style chat endpoint, on localhost.

CherryColaLLM runs this file with the Python of its AirLLM environment (the
one ``cherrycolallm airllm install`` makes), never with its own: AirLLM needs
PyTorch and transformers, which CherryColaLLM itself does without. So this
file imports nothing from CherryColaLLM.

AirLLM runs a model one layer at a time, reading each layer from disk, so a
70B model runs on a 4 GB graphics card (slowly). The first load splits the
model into per-layer files, which takes a while and as much disk again.

Endpoints: GET /health (200 when loaded, 503 while loading),
GET /v1/models, POST /v1/chat/completions (stream or not).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"model": None, "name": "", "error": "", "device": "cpu"}
LOCK = threading.Lock()  # one answer at a time: AirLLM holds one model


def load(args) -> None:
    try:
        import torch
        from airllm import AutoModel
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        kwargs = {"max_seq_len": args.max_seq_len,
                  "prefetching": args.prefetch and device != "cpu"}
        if args.compression != "none":
            kwargs["compression"] = args.compression
        if args.shards:
            kwargs["layer_shards_saving_path"] = args.shards
        try:
            model = AutoModel.from_pretrained(args.model, device=device,
                                              **kwargs)
        except TypeError:  # AirLLM versions without a device argument
            model = AutoModel.from_pretrained(args.model, **kwargs)
        STATE.update(model=model, device=device)
        print(f"loaded {args.model} on {device}", flush=True)
    except Exception as exc:  # shown to people through /health
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        print(STATE["error"], file=sys.stderr, flush=True)


def _prompt_ids(tok, messages):
    msgs = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):  # OpenAI content parts: text only
            content = "".join(p.get("text", "") for p in content
                              if isinstance(p, dict))
        msgs.append({"role": m.get("role", "user"), "content": content})
    if getattr(tok, "chat_template", None):
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt")
        # transformers 5 returns a BatchEncoding here, 4 a tensor.
        return ids["input_ids"] if hasattr(ids, "keys") else ids
    text = "".join(f"{m['role']}: {m['content']}\n" for m in msgs)
    return tok(text + "assistant: ", return_tensors="pt",
               return_attention_mask=False)["input_ids"]


def generate(body: dict):
    """Yield the answer's text in pieces."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    from transformers import TextIteratorStreamer

    model = STATE["model"]
    tok = model.tokenizer
    ids = _prompt_ids(tok, body.get("messages") or [])
    if STATE["device"] != "cpu":
        ids = ids.to(STATE["device"])
    stop = body.get("stop") or []
    stop = [stop] if isinstance(stop, str) else list(stop)
    halt = threading.Event()

    class Halt(StoppingCriteria):
        def __call__(self, *a, **k):
            return halt.is_set()

    gen = {"max_new_tokens": body.get("max_tokens") or 512,
           "use_cache": True, "stopping_criteria": StoppingCriteriaList([Halt()])}
    temp = body.get("temperature", 0.7)
    if temp and temp > 0:
        gen.update(do_sample=True, temperature=float(temp))
        for k in ("top_k", "top_p", "min_p"):
            if body.get(k) is not None:
                gen[k] = body[k]
    if body.get("repeat_penalty"):
        gen["repetition_penalty"] = float(body["repeat_penalty"])
    if isinstance(body.get("seed"), int) and body["seed"] >= 0:
        torch.manual_seed(body["seed"])
    streamer = TextIteratorStreamer(tok, skip_prompt=True,
                                    skip_special_tokens=True)
    gen["streamer"] = streamer
    errors = []

    def run():
        try:
            model.generate(ids, **gen)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            errors.append(f"{type(exc).__name__}: {exc}")
            streamer.end()

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    sent = ""
    try:
        for piece in streamer:
            text = sent + piece
            cut = min((text.find(s) for s in stop if s and s in text),
                      default=-1)
            if cut >= 0:
                if cut > len(sent):
                    yield text[len(sent):cut]
                halt.set()
                break
            sent = text
            if piece:
                yield piece
    finally:
        halt.set()
        worker.join()
    if errors:
        raise RuntimeError(errors[0])


class Handler(BaseHTTPRequestHandler):
    server_version = "CherryColaLLM-AirLLM"

    def log_message(self, *args):
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            if STATE["error"]:
                return self._json({"status": "error",
                                   "error": STATE["error"]}, 500)
            if STATE["model"] is None:
                return self._json({"status": "loading"}, 503)
            return self._json({"status": "ok", "device": STATE["device"]})
        if self.path == "/v1/models":
            return self._json({"data": [{"id": STATE["name"]}]})
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self._json({"error": "not found"}, 404)
        try:
            body = json.loads(self.rfile.read(
                int(self.headers.get("Content-Length", 0))) or b"{}")
        except ValueError:
            return self._json({"error": "bad JSON"}, 400)
        if STATE["model"] is None:
            return self._json({"error": STATE["error"]
                               or "the model is still loading"}, 503)
        with LOCK:
            if not body.get("stream"):
                try:
                    text = "".join(generate(body))
                except Exception as exc:
                    return self._json({"error": str(exc)}, 500)
                return self._json({"choices": [{"index": 0, "message": {
                    "role": "assistant", "content": text},
                    "finish_reason": "stop"}]})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def send(obj):
                self.wfile.write(b"data: " + json.dumps(obj).encode()
                                 + b"\n\n")
                self.wfile.flush()
            try:
                for piece in generate(body):
                    send({"choices": [{"index": 0,
                                       "delta": {"content": piece}}]})
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as exc:
                send({"error": {"message": str(exc)}})
            try:
                self.wfile.write(b"data: [DONE]\n\n")
            except OSError:
                pass


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="folder or owner/repo")
    p.add_argument("--name", default="")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=11437)
    p.add_argument("--compression", choices=("none", "4bit", "8bit"),
                   default="none")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--shards", default="")
    p.add_argument("--prefetch", type=int, default=1)
    args = p.parse_args(argv)
    STATE["name"] = args.name or args.model
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    threading.Thread(target=load, args=(args,), daemon=True).start()
    print(f"AirLLM server on http://{args.host}:{args.port} "
          f"({time.strftime('%H:%M:%S')})", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

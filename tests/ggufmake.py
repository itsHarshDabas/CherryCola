"""Write small GGUF headers for tests (stdlib only, no real weights)."""

from __future__ import annotations

import struct


def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def write(path, *, arch="llama", file_type=1, layers=4, embd=64, heads=4,
          kv_heads=2, ctx=2048, tensor_type=1, ffn=128, vocab=32):
    kv = [
        ("general.architecture", 8, _s(arch)),
        ("general.name", 8, _s("test")),
        ("general.file_type", 4, struct.pack("<I", file_type)),
        (f"{arch}.block_count", 4, struct.pack("<I", layers)),
        (f"{arch}.context_length", 4, struct.pack("<I", ctx)),
        (f"{arch}.embedding_length", 4, struct.pack("<I", embd)),
        (f"{arch}.attention.head_count", 4, struct.pack("<I", heads)),
        (f"{arch}.attention.head_count_kv", 4, struct.pack("<I", kv_heads)),
        # an array of strings, like the token list, to be skipped
        ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 2) + _s("a")
         + _s("b")),
    ]
    tensors = [("token_embd.weight", [embd, vocab])]
    for i in range(layers):
        tensors += [(f"blk.{i}.attn_q.weight", [embd, embd]),
                    (f"blk.{i}.ffn_up.weight", [embd, ffn])]
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for key, vtype, value in kv:
        out += _s(key) + struct.pack("<I", vtype) + value
    for name, dims in tensors:
        out += _s(name) + struct.pack("<I", len(dims))
        out += b"".join(struct.pack("<Q", d) for d in dims)
        out += struct.pack("<IQ", tensor_type, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out + b"\0" * 64)
    return path

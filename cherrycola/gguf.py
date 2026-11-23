"""Read a GGUF file's header: metadata and tensor sizes. Stdlib only.

Only the header is read (never the weights), so this is quick even for
very large files. Used to show what a model is, check a quantization
request, and plan how much of it fits on the GPU and in RAM.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["FILE_TYPES", "GGUFError", "ModelInfo", "read_info"]

MAGIC = b"GGUF"

# llama.h LLAMA_FTYPE_* -> the name llama-quantize uses.
FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0",
    9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
    14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
    19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS",
    24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
    29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0",
    37: "TQ2_0", 38: "MXFP4_MOE",
}
UNQUANTIZED = {"F32", "F16", "BF16"}

# ggml_type -> (block size in elements, bytes per block)
_GGML_TYPES = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22),
    7: (32, 24), 8: (32, 34), 9: (32, 36), 10: (256, 84), 11: (256, 110),
    12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292),
    16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50),
    20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136),
    24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
    29: (256, 56), 30: (1, 2), 34: (256, 54), 35: (256, 66),
    39: (32, 17),
}

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
            6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9


class GGUFError(Exception):
    pass


@dataclass
class ModelInfo:
    path: Path
    size: int
    architecture: str = ""
    name: str = ""
    file_type: str = ""
    block_count: int = 0
    context_length: int = 0
    embedding_length: int = 0
    head_count: int = 0
    head_count_kv: int = 0
    expert_count: int = 0
    parameters: int = 0
    split_count: int = 1
    layer_bytes: dict[int, int] = field(default_factory=dict)
    other_bytes: int = 0  # embeddings, output head, norms

    @property
    def quantized(self) -> bool:
        return bool(self.file_type) and self.file_type not in UNQUANTIZED

    def kv_bytes_per_token(self, cache_type: str = "f16") -> int:
        """K and V for one token across all layers."""
        if not (self.block_count and self.embedding_length
                and self.head_count):
            return 0
        head_dim = self.embedding_length // self.head_count
        kv_heads = self.head_count_kv or self.head_count
        per_elem = {"f32": 4, "f16": 2, "bf16": 2, "q8_0": 34 / 32,
                    "q5_1": 24 / 32, "q5_0": 22 / 32, "q4_1": 20 / 32,
                    "q4_0": 18 / 32, "iq4_nl": 18 / 32}.get(cache_type, 2)
        return int(2 * self.block_count * kv_heads * head_dim * per_elem)


class _Reader:
    def __init__(self, handle):
        self.h = handle

    def read(self, n: int) -> bytes:
        data = self.h.read(n)
        if len(data) != n:
            raise GGUFError("file ends inside the header")
        return data

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        n = self.unpack("<Q")
        if n > 1 << 24:
            raise GGUFError("string too long; not a GGUF file?")
        return self.read(n).decode("utf-8", errors="replace")

    def value(self, vtype: int, keep: bool = True):
        if vtype in _SCALARS:
            return self.unpack(_SCALARS[vtype])
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            itype, n = self.unpack("<I"), self.unpack("<Q")
            if itype in _SCALARS and not keep:
                self.h.seek(n * struct.calcsize(_SCALARS[itype]), 1)
                return None
            items = [self.value(itype, keep) for _ in range(n)]
            return items if keep else None
        raise GGUFError(f"unknown value type {vtype}")


def _layer_of(name: str) -> int | None:
    if name.startswith("blk."):
        part = name.split(".", 2)[1]
        if part.isdigit():
            return int(part)
    return None


def read_info(path: str | Path) -> ModelInfo:
    path = Path(path)
    try:
        handle = open(path, "rb")
    except OSError as exc:
        raise GGUFError(str(exc)) from exc
    with handle:
        r = _Reader(handle)
        if r.read(4) != MAGIC:
            raise GGUFError(f"{path.name} is not a GGUF file")
        version = r.unpack("<I")
        if version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version {version}")
        n_tensors, n_kv = r.unpack("<Q"), r.unpack("<Q")
        meta: dict[str, object] = {}
        for _ in range(n_kv):
            key = r.string()
            vtype = r.unpack("<I")
            # Token lists are big; skip arrays we don't use.
            meta[key] = r.value(vtype, keep=vtype != _ARRAY)
        info = ModelInfo(path=path, size=path.stat().st_size)
        arch = str(meta.get("general.architecture", ""))
        info.architecture = arch
        info.name = str(meta.get("general.name", ""))
        ftype = meta.get("general.file_type")
        info.file_type = FILE_TYPES.get(ftype, "") if isinstance(
            ftype, int) else ""

        def num(key: str) -> int:
            v = meta.get(f"{arch}.{key}")
            return int(v) if isinstance(v, (int, float)) and not \
                isinstance(v, bool) else 0

        info.block_count = num("block_count")
        info.context_length = num("context_length")
        info.embedding_length = num("embedding_length")
        info.head_count = num("attention.head_count")
        info.head_count_kv = num("attention.head_count_kv")
        info.expert_count = num("expert_count")
        split = meta.get("split.count")
        info.split_count = int(split) if isinstance(split, int) and split else 1
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.unpack("<Q") for _ in range(r.unpack("<I"))]
            ttype = r.unpack("<I")
            r.unpack("<Q")  # offset
            elems = 1
            for d in dims:
                elems *= d
            info.parameters += elems
            block, nbytes = _GGML_TYPES.get(ttype, (1, 2))
            size = elems // block * nbytes
            layer = _layer_of(name)
            if layer is None:
                info.other_bytes += size
            else:
                info.layer_bytes[layer] = info.layer_bytes.get(layer, 0) + size
        if not info.file_type:
            info.file_type = "unknown"
        return info

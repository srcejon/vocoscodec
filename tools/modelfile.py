"""Reader and writer for vocoscodec model files (format version 1).

Layout, all integers little-endian:

  offset 0   char[8]  magic "VOCOSCM\\0"
         8   u32      format version (1)
        12   u32      tensor count
        16   u32      metadata length in bytes
        20   u32      reserved, 0
        24   metadata: UTF-8 text, one "key=value" per line
             tensor table, one entry per tensor:
               u16 name length, name (UTF-8), u8 dtype (0 = float32),
               u8 rank, u32 dims[rank], u64 data offset from the file start
             data: each tensor row-major, starting on a 64-byte boundary
"""
import struct
from pathlib import Path

import numpy as np

MAGIC = b"VOCOSCM\0"
VERSION = 1
ALIGN = 64


def write(path: Path, meta: dict, tensors: dict):
    meta_bytes = "".join(f"{k}={v}\n" for k, v in meta.items()).encode()
    arrays = {name: np.ascontiguousarray(value, dtype="<f4") for name, value in tensors.items()}
    table_size = sum(2 + len(n.encode()) + 2 + 4 * a.ndim + 8 for n, a in arrays.items())
    offset = 24 + len(meta_bytes) + table_size
    offsets = {}
    for name, array in arrays.items():
        offset = (offset + ALIGN - 1) // ALIGN * ALIGN
        offsets[name] = offset
        offset += array.nbytes

    out = bytearray(MAGIC + struct.pack("<4I", VERSION, len(arrays), len(meta_bytes), 0))
    out += meta_bytes
    for name, array in arrays.items():
        encoded = name.encode()
        out += struct.pack("<H", len(encoded)) + encoded
        out += struct.pack("<BB", 0, array.ndim) + struct.pack(f"<{array.ndim}I", *array.shape)
        out += struct.pack("<Q", offsets[name])
    for name, array in arrays.items():
        out += bytes(offsets[name] - len(out))
        out += array.tobytes()
    path.write_bytes(bytes(out))


def read(path: Path):
    data = Path(path).read_bytes()
    if data[:8] != MAGIC:
        raise ValueError(f"{path}: not a vocoscodec model file")
    version, count, meta_length, _ = struct.unpack_from("<4I", data, 8)
    if version != VERSION:
        raise ValueError(f"{path}: format version {version}, expected {VERSION}")
    position = 24
    meta = dict(line.split("=", 1) for line in
                data[position:position + meta_length].decode().splitlines() if line)
    position += meta_length
    tensors = {}
    for _ in range(count):
        (length,) = struct.unpack_from("<H", data, position)
        name = data[position + 2:position + 2 + length].decode()
        position += 2 + length
        dtype, rank = struct.unpack_from("<BB", data, position)
        dims = struct.unpack_from(f"<{rank}I", data, position + 2)
        (offset,) = struct.unpack_from("<Q", data, position + 2 + 4 * rank)
        position += 2 + 4 * rank + 8
        if dtype != 0:
            raise ValueError(f"{path}: tensor {name} has unsupported dtype {dtype}")
        tensors[name] = np.frombuffer(data, dtype="<f4", count=int(np.prod(dims)),
                                      offset=offset).reshape(dims)
    return meta, tensors

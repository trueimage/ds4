#!/usr/bin/env python3
"""Read both V4.1 Engram tables once through the page cache (warm state).
Usage: warm_engram.py MODEL.gguf [--dry-run]"""
import struct, sys

path = sys.argv[1]
f = open(path, "rb")
u32 = lambda: struct.unpack("<I", f.read(4))[0]
u64 = lambda: struct.unpack("<Q", f.read(8))[0]
def s(): return f.read(u64()).decode("utf-8", "replace")
SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
def val(t):
    if t in SIZES: return f.read(SIZES[t])
    if t == 8: return s()
    if t == 9:
        at, n = u32(), u64()
        return [val(at) for _ in range(n)]
    raise ValueError(f"unknown GGUF type {t}")
assert f.read(4) == b"GGUF"
u32(); n_tensors, n_kv = u64(), u64()
align = 32
for _ in range(n_kv):
    k, t = s(), u32()
    v = val(t)
    if k == "general.alignment": align = struct.unpack("<I", v)[0]
ranges = []
for _ in range(n_tensors):
    name, nd = s(), u32()
    dims = [u64() for _ in range(nd)]
    u32(); off = u64()
    if name in ("blk.1.engram_embd.weight", "blk.14.engram_embd.weight"):
        ranges.append((name, off, dims[0] * dims[1]))  # I8 rows of 264 bytes
data = (f.tell() + align - 1) // align * align
for name, off, n in ranges:
    print(f"{name}: offset {data + off}, {n / 2**30:.1f} GiB", flush=True)
    if "--dry-run" in sys.argv: continue
    with open(path, "rb", buffering=0) as g:
        g.seek(data + off)
        while n > 0:
            chunk = g.read(min(n, 64 << 20))
            if not chunk: break
            n -= len(chunk)

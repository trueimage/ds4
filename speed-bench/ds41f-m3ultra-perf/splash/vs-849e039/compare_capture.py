#!/usr/bin/env python3
"""Require identical position, vocabulary and FP32 bits in every captured row."""
import argparse
import hashlib
import json
import struct


def row(fp):
    header = fp.read(8)
    if not header:
        return None
    if len(header) != 8:
        raise ValueError("truncated header")
    position, vocab = struct.unpack("<II", header)
    if not 0 < position <= 32768 or not 0 < vocab <= 1000000:
        raise ValueError("invalid capture dimensions")
    values = fp.read(vocab * 4)
    if len(values) != vocab * 4:
        raise ValueError("truncated logits")
    return position, vocab, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control")
    parser.add_argument("candidate")
    args = parser.parse_args()
    rows = floats = 0
    positions = []
    digest = hashlib.sha256()
    with open(args.control, "rb") as control, open(args.candidate, "rb") as candidate:
        while True:
            a, b = row(control), row(candidate)
            if a is None and b is None:
                break
            if a != b:
                raise SystemExit(f"logit mismatch at row {rows}: "
                                 f"{a[:2] if a else None} / {b[:2] if b else None}")
            rows += 1
            floats += a[1]
            positions.append(a[0])
            digest.update(a[2])
    if not rows:
        raise SystemExit("empty captures")
    print(json.dumps({"exact": True, "rows": rows, "floats": floats,
                      "max_position": max(positions), "logits_sha256": digest.hexdigest()}))


if __name__ == "__main__":
    main()

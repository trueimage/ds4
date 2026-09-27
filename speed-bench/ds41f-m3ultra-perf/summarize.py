#!/usr/bin/env python3
"""Collect paired benchmark summaries without turning profile timings into wins."""
import argparse
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
args = parser.parse_args()
results = []
for path in sorted(args.directory.glob("*.log")):
    metadata = path.with_suffix(".json")
    if not metadata.exists():
        continue
    record = json.loads(metadata.read_text())
    text = path.read_text()
    variants = {}
    for match in re.finditer(r"^(?:aggregate )?variant=(control|candidate) .*tokens_per_second=([\d.]+)$", text, re.M):
        variants[match[1]] = float(match[2])
    if len(variants) != 2:
        continue
    command = record["command"]
    prefix = int(command[command.index("--prefix-tokens") + 1])
    metric = "decode" if "decode_schedule" in command[0] else "prefill"
    results.append({"name": path.stem, "metric": metric, "prefix": prefix,
        "returncode": record.get("returncode"), **variants,
        "change_percent": 100 * (variants["candidate"] / variants["control"] - 1),
        "command": command, "overrides": record["overrides"]})
print(json.dumps(results, indent=2))

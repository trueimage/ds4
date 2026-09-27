#!/usr/bin/env python3
"""Run one campaign command with clean tuning controls and retained evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command or Path(args.name).name != args.name:
        parser.error("a command and a plain evidence name are required")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("DS4_", "MTL_", "ASTRA_"))}
    overrides = {}
    for setting in args.env:
        key, sep, value = setting.partition("=")
        if not sep or not key:
            parser.error("--env requires NAME=VALUE")
        overrides[key] = value
    env.update(overrides)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / args.name
    record = {
        "command": command, "cwd": str(Path.cwd()), "overrides": overrides,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path("ds4.c"), Path("ds4_metal.m"), *sorted(Path("metal").glob("*.metal"))]
        },
        "started": time.time(),
    }
    # Refuse to overwrite earlier measurements accidentally.
    with stem.with_suffix(".json").open("x") as metadata:
        metadata.write(json.dumps(record, indent=2) + "\n")
    with stem.with_suffix(".log").open("x") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    record.update(returncode=result.returncode, finished=time.time())
    stem.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"name": args.name, "returncode": result.returncode,
                      "seconds": record["finished"] - record["started"]}))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

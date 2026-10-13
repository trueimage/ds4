#!/usr/bin/env python3
"""Compare ordinary greedy CLI output with prompt-copy disabled/enabled.

Requires the supported resident V4.1 Q4_K model on M3 Ultra Metal. Runs two
model processes serially; retain the printed result directory on failure.
Small token caps check output parity; caps >= 32 also require a copied token.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
PREFIXES = ('DS4_', 'MTL_', 'METAL_', 'ASTRA_')


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def run(output, model, tokens):
    output.mkdir(parents=True, exist_ok=False)
    print(f'Results and retained logs: {output}', flush=True)
    binary = REPO / 'ds4'
    fixture = REPO / 'tests/prompt-copy-fixtures/copy-code.txt'
    binary_hash = sha(binary)
    model_stat = model.stat()
    removed = sorted(k for k in os.environ if k.startswith(PREFIXES))
    clean = {k: v for k, v in os.environ.items() if not k.startswith(PREFIXES)}
    # --prompt-file selects ordinary run_generation, not the persistent REPL.
    command = [str(binary), '--model', str(model), '--metal', '--ctx', '4096',
               '--prompt-file', str(fixture), '--system', '', '--temp', '0',
               '--nothink', '--tokens', str(tokens), '--seed', '123']
    records = []
    baseline = None
    for arm in ('scalar', 'copy'):
        directory = output / arm
        directory.mkdir()
        assert sha(binary) == binary_hash, 'CLI binary changed between arms'
        overrides = {'DS4_PROMPT_COPY_LOG': '1'}
        if arm == 'scalar':
            overrides['DS4_METAL_DISABLE_V41_PROMPT_COPY'] = '1'
        env = dict(clean, **overrides)
        record = {'arm': arm, 'command': command, 'cwd': str(REPO),
                  'token_cap': tokens, 'activation_required': arm == 'copy' and tokens >= 32,
                  'environment_overrides': overrides, 'removed_environment_keys': removed,
                  'binary_sha256': binary_hash, 'fixture_sha256': sha(fixture),
                  'model_bytes': model_stat.st_size, 'model_mtime_ns': model_stat.st_mtime_ns,
                  'started_utc': now()}
        save(directory / 'invocation.json', record)
        started = time.monotonic()
        proc = None
        failure = None
        try:
            with (directory / 'stdout.bin').open('wb') as stdout, \
                    (directory / 'stderr.log').open('wb') as stderr:
                proc = subprocess.Popen(command, cwd=REPO, env=env,
                                        stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
                proc.wait(timeout=600)
        except BaseException as exc:
            failure = repr(exc)
            raise
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            record.update(finished_utc=now(), elapsed_seconds=time.monotonic() - started,
                          exit_code=None if proc is None else proc.returncode, failure=failure)
            save(directory / 'invocation.json', record)
        assert proc.returncode == 0, (arm, proc.returncode)
        assert sha(binary) == binary_hash, 'CLI binary changed during run'
        stdout = (directory / 'stdout.bin').read_bytes()
        calls = [tuple(map(int, match)) for match in re.findall(
            r'prompt-copy pos=(\d+) proposed=(\d+) accepted=(\d+)',
            (directory / 'stderr.log').read_text(errors='replace'))]
        activated = any(proposed > 1 and accepted > 1 for _, proposed, accepted in calls)
        if arm == 'scalar':
            baseline = stdout
        exact = baseline == stdout
        passed = bool(stdout) and exact and (not calls if arm == 'scalar' else
                                             tokens < 32 or activated)
        record.update(stdout_bytes=len(stdout), stdout_sha256=hashlib.sha256(stdout).hexdigest(),
                      verified_cycles=len(calls), accepted_draft_tokens=sum(k - 1 for _, _, k in calls),
                      activated=activated, stdout_exact=exact, passed=passed)
        if not exact:
            record['first_differing_byte'] = next(
                (i for i, (a, b) in enumerate(zip(baseline, stdout)) if a != b),
                min(len(baseline), len(stdout)))
        save(directory / 'result.json', record)
        records.append(record)
        save(output / 'results.json', {
            'all_equal': all(r['stdout_exact'] for r in records),
            'passed': len(records) == 2 and all(r['passed'] for r in records),
            'complete': len(records) == 2,
            'scope': 'CLI stdout parity and actual R1 activation; elapsed time includes loading',
            'records': records})
        assert stdout, 'CLI generated no stdout'
        assert not (arm == 'scalar' and calls), 'rollback CLI unexpectedly used prompt-copy'
        assert exact, 'CLI stdout differs; see raw stdout.bin files and result.json'
        assert not record['activation_required'] or activated, \
            'CLI never accepted a copied continuation'
        print(f'{arm}: cap={tokens}, stdout={len(stdout)} bytes, '
              f'verifier_cycles={len(calls)}, exact={exact}', flush=True)
    print(output / 'results.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=os.environ.get('DS4_TEST_MODEL'),
                        help='GGUF model path (or set DS4_TEST_MODEL)')
    parser.add_argument('--tokens', type=int, default=256, help='generation cap, 1..256 (default: 256)')
    parser.add_argument('--output', type=Path, help='new result directory; default: retained temporary directory')
    args = parser.parse_args()
    if args.model is None:
        parser.error('--model or DS4_TEST_MODEL is required')
    if not 1 <= args.tokens <= 256:
        parser.error('--tokens must be in 1..256')
    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        parser.error(f'model is not a file: {model}')
    output = args.output or Path(tempfile.mkdtemp(prefix='ds4-prompt-copy-cli-')) / 'results'
    run(output.expanduser().resolve(), model, args.tokens)

#!/usr/bin/env python3
"""Clean 849e039 versus frozen Splash, two official local GGUFs, at most 32K."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

OUT = Path(__file__).resolve().parent
REPO = Path('/Users/jw/ds4')
FIXTURES = REPO / 'speed-bench/ds41f-m3ultra-perf/splash'
TREES = {
    'base': Path('/Users/jw/.cache/ds4-bench/wt-849e039b4c80'),
    'splash': Path('/tmp/ds4-splash-20260926/final'),
}
COMMITS = {'base': '849e039b4c8048d73423c12cdf201ccbfee8528b',
           'splash': 'f768551d842942f5eed4b257f6c997f8eb5ec29e'}
MODELS = {'v41': REPO/'gguf/DeepSeek-V4.1-Flash-Q4.gguf',
          'glm': REPO/'gguf/GLM-5.3-Flash-Q4_K.gguf'}
PROMPT = REPO/'speed-bench/promessi_sposi.txt'
ORDER = ['base', 'splash', 'splash', 'base']
ENV = {k: v for k, v in os.environ.items()
       if not k.startswith(('DS4_', 'MTL_', 'METAL_'))}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()

def shell_output(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()

def identity():
    result = {}
    for arm, tree in TREES.items():
        assert shell_output(['git', 'rev-parse', 'HEAD'], tree) == COMMITS[arm]
        assert not shell_output(['git', 'diff', 'HEAD', '--'], tree)
        names = shell_output(['git', 'ls-files'], tree).splitlines()
        sources = {n: digest(tree/n) for n in names
                   if n.endswith(('.c', '.h', '.m', '.metal')) or n == 'Makefile'}
        result[arm] = {'commit': COMMITS[arm], 'worktree': str(tree),
                       'source_sha256': sources,
                       'binaries': {n: digest(tree/n) for n in ('ds4-bench', 'ds4-server')},
                       'capture_sha256': digest(OUT/('capture-'+arm)/'capture-bench')}
    return result

def save():
    tmp = OUT/'manifest.json.tmp'
    tmp.write_text(json.dumps(manifest, indent=2)+'\n')
    tmp.replace(OUT/'manifest.json')

def run(label, cmd, cwd=None, extra_env=None):
    previous = next((r for r in manifest['runs'] if r['label'] == label), None)
    if previous:
        assert previous['exit_code'] == 0, f'{label}: previous failure requires inspection'
        print('SKIP', label, flush=True)
        return
    print('START', label, flush=True)
    env = dict(ENV)
    if extra_env:
        env.update(extra_env)
    args = [str(x) for x in cmd]
    rec = {'label': label, 'command': args, 'cwd': str(cwd or OUT),
           'extra_env': extra_env or {},
           'started': datetime.datetime.now().astimezone().isoformat()}
    start = time.monotonic()
    with (OUT/(label+'.stdout')).open('w') as stdout, (OUT/(label+'.stderr')).open('w') as stderr:
        proc = subprocess.run(args, cwd=cwd or OUT, env=env, stdout=stdout, stderr=stderr)
    rec.update(exit_code=proc.returncode, seconds=time.monotonic()-start)
    manifest['runs'].append(rec)
    save()
    print(f'END {label}: exit={proc.returncode}, seconds={rec["seconds"]:.2f}', flush=True)
    if proc.returncode:
        raise RuntimeError(f'{label} failed; see {OUT/(label+".stderr")}')

def bench_args(tag):
    return ['-m', MODELS[tag], '--prompt-file', PROMPT,
            '--ctx-start', '512' if tag == 'v41' else '2048',
            '--ctx-max', '32639', '--ctx-alloc', '32768', '--step-mul', '2',
            '--gen-tokens', '128']

def exactness(tag):
    for arm, tree in TREES.items():
        dst = OUT/('exact-'+tag+'-'+arm)
        (dst/'snapshots').mkdir(parents=True, exist_ok=True)
        run('capture-'+tag+'-'+arm,
            [OUT/('capture-'+arm)/'capture-bench', *bench_args(tag),
             '--csv', dst/'instrumented.csv'], cwd=tree,
            extra_env={'DS4_BENCH_TRACE_LOGITS': str(dst/'logits.bin'),
                       'DS4_BENCH_SNAPSHOT_DIR': str(dst/'snapshots'),
                       'DS4_BENCH_FORCE_SNAPSHOT': '1'})
    base = OUT/('exact-'+tag+'-base')
    candidate = OUT/('exact-'+tag+'-splash')
    run('compare-logits-'+tag,
        ['python3', OUT/'compare_capture.py', base/'logits.bin', candidate/'logits.bin'])
    aa, bb = sorted((base/'snapshots').glob('*.bin')), sorted((candidate/'snapshots').glob('*.bin'))
    assert [p.name for p in aa] == [p.name for p in bb] and aa
    snapshots = []
    for a, b in zip(aa, bb):
        ha, hb = digest(a), digest(b)
        assert a.stat().st_size == b.stat().st_size and ha == hb, f'snapshot mismatch: {tag}/{a.name}'
        snapshots.append({'frontier': int(a.stem), 'bytes': a.stat().st_size,
                          'sha256': ha, 'bit_identical': True})
    (OUT/('exactness-'+tag+'.json')).write_text(json.dumps({
        'logits': json.loads((OUT/('compare-logits-'+tag+'.stdout')).read_text()),
        'snapshots': snapshots}, indent=2)+'\n')

def performance(tag):
    for i, arm in enumerate(ORDER, 1):
        label = f'perf-{tag}-{i}-{arm}'
        run(label, ['./ds4-bench', *bench_args(tag), '--csv', OUT/(label+'.csv')], TREES[arm])
        rows = list(csv.DictReader((OUT/(label+'.csv')).open()))
        assert len(rows) == (7 if tag == 'v41' else 5)
        assert all(int(r['gen_tokens']) == 128 for r in rows)
        assert all(int(r['ctx_tokens']) + int(r['gen_tokens']) < 32768 for r in rows)

def http(tag):
    prompts = OUT/('http-prompts-'+tag)
    prompts.mkdir(exist_ok=True)
    for source in [FIXTURES/('checkpoint-prompts-'+tag)/'http-10240.txt',
                   FIXTURES/('prompts-'+tag)/'http-32000.txt']:
        dst = prompts/source.name
        if dst.exists():
            assert dst.read_bytes() == source.read_bytes()
        else:
            shutil.copyfile(source, dst)
    for i, arm in enumerate(ORDER, 1):
        label = f'http-{tag}-{i}-{arm}'
        run(label, ['python3', OUT/'http_bench.py', '--checkout', TREES[arm],
                    '--model', MODELS[tag], '--prompts', prompts,
                    '--out', OUT/label, '--expect-continued-frontier', '10240'])
    baseline = OUT/f'http-{tag}-1-base'
    ref = json.loads((baseline/'results.json').read_text())
    ref_images = json.loads((baseline/'images.json').read_text())
    assert len(ref) == 4 and ref_images
    fields = ['name', 'usage', 'parts', 'finish']
    for i, arm in enumerate(ORDER, 1):
        target = OUT/f'http-{tag}-{i}-{arm}'
        values = json.loads((target/'results.json').read_text())
        assert [{k: x[k] for k in fields} for x in values] == [{k: x[k] for k in fields} for x in ref], f'HTTP output/usage drift: {target}'
        assert json.loads((target/'images.json').read_text()) == ref_images, f'HTTP checkpoint drift: {target}'
    (OUT/('http-exactness-'+tag+'.json')).write_text(json.dumps({
        'requests': 16, 'output_usage_finish_identical': True,
        'checkpoint_images_identical_except_header_timestamps': True,
        'images_per_round': ref_images}, indent=2)+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(MODELS), required=True)
    parser.add_argument('--phase', choices=['all', 'exactness', 'perf', 'http'], default='all')
    args = parser.parse_args()
    current = identity()
    path = OUT/'manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text())
        assert manifest['trees'] == current, 'Measurement source/binaries changed'
    else:
        manifest = {'created': datetime.datetime.now().astimezone().isoformat(),
                    'comparison': 'clean 849e039 versus final Splash f768551',
                    'os': shell_output(['sw_vers']), 'compiler': shell_output(['cc', '--version']),
                    'trees': current, 'models': {}, 'order': ORDER,
                    'allocated_context': 32768, 'max_frontier': 32639,
                    'generation_tokens': 128, 'prompt_sha256': digest(PROMPT), 'runs': []}
        for tag, path in MODELS.items():
            s = path.stat()
            with path.open('rb') as f:
                first = hashlib.sha256(f.read(8 << 20)).hexdigest()
            manifest['models'][tag] = {'path': str(path), 'bytes': s.st_size,
                                       'mtime_ns': s.st_mtime_ns,
                                       'first_8_MiB_sha256': first}
        save()
    assert digest(PROMPT) == manifest['prompt_sha256']
    for tag, path in MODELS.items():
        s = path.stat()
        assert s.st_size == manifest['models'][tag]['bytes'] and s.st_mtime_ns == manifest['models'][tag]['mtime_ns']
    if args.model == 'v41':
        run('warm-engram-v41', ['python3', OUT/'warm_engram.py', MODELS['v41']])
    if args.phase in ('all', 'exactness'):
        exactness(args.model)
    if args.phase in ('all', 'perf'):
        performance(args.model)
    if args.phase in ('all', 'http'):
        http(args.model)

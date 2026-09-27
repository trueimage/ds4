#!/usr/bin/env python3
"""Serial current-main/final ABBA sweep of the two official GGUFs, <=32K."""
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

OUT = Path(__file__).resolve().parent
REPO = Path('/Users/jw/ds4')
PREVIOUS = REPO.parent/'ds4-bench-results/splash-vs-849e039-20260926'
TREES = {'main': Path('/tmp/ds4-pr1090-main-20260926'),
         'final': Path('/tmp/ds4-splash-20260926/final')}
COMMITS = {'main': '0aaea5a238fb41a35106a551e73c8409dfb751ac',
           'final': 'f768551d842942f5eed4b257f6c997f8eb5ec29e'}
MODELS = {'v41': REPO/'gguf/DeepSeek-V4.1-Flash-Q4.gguf',
          'glm': REPO/'gguf/GLM-5.3-Flash-Q4_K.gguf'}
ENV = {k:v for k,v in os.environ.items()
       if not k.startswith(('DS4_', 'MTL_', 'METAL_', 'ASTRA_'))}

def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        while data := f.read(8 << 20): h.update(data)
    return h.hexdigest()

def command(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()

manifest = {'created': datetime.datetime.now().astimezone().isoformat(),
            'os': command(['sw_vers']), 'compiler': command(['cc','--version']),
            'order': ['main','final','final','main'], 'trees': {},
            'models': {}, 'context_allocation': 32768, 'runs': []}
for arm, tree in TREES.items():
    assert command(['git','rev-parse','HEAD'], tree) == COMMITS[arm]
    assert not command(['git','diff','HEAD','--'], tree)
    names = command(['git','ls-files'], tree).splitlines()
    manifest['trees'][arm] = {'commit': COMMITS[arm], 'checkout': str(tree),
        'source_sha256': {n:digest(tree/n) for n in names
                         if n.endswith(('.c','.h','.m','.metal')) or n=='Makefile'},
        'benchmark_sha256': digest(tree/'ds4-bench')}
for tag,path in MODELS.items():
    with path.open('rb') as f: first = hashlib.sha256(f.read(8 << 20)).hexdigest()
    manifest['models'][tag] = {'path':str(path), 'bytes':path.stat().st_size,
        'mtime_ns':path.stat().st_mtime_ns, 'first_8_MiB_sha256':first}
manifest['prompt_sha256'] = digest(REPO/'speed-bench/promessi_sposi.txt')

def save():
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

def run(label, args, cwd=None, extra=None, allow_difference=False):
    print('START',label,flush=True)
    args = [str(x) for x in args]
    env = dict(ENV, **(extra or {}))
    start = time.monotonic()
    with (OUT/(label+'.stdout')).open('w') as stdout, (OUT/(label+'.stderr')).open('w') as stderr:
        p = subprocess.run(args,cwd=cwd or OUT,env=env,stdout=stdout,stderr=stderr)
    manifest['runs'].append({'label':label,'command':args,'cwd':str(cwd or OUT),
        'extra_env':extra or {},'exit_code':p.returncode,'seconds':time.monotonic()-start})
    save()
    print('END',label,p.returncode,flush=True)
    if p.returncode and not allow_difference: raise RuntimeError(label)

save()
run('warm-engram',['python3',OUT/'warm_engram.py',MODELS['v41']])
for tag,path in MODELS.items():
    args = ['-m',path,'--prompt-file',REPO/'speed-bench/promessi_sposi.txt',
        '--ctx-start',512 if tag=='v41' else 2048,'--ctx-max',32639,
        '--ctx-alloc',32768,'--step-mul',2,'--gen-tokens',128]
    exact = OUT/('exact-'+tag+'-main')
    (exact/'snapshots').mkdir(parents=True)
    run('capture-'+tag+'-main',[OUT/'capture-main/capture-bench',*args,
        '--csv',exact/'instrumented.csv'],TREES['main'],
        {'DS4_BENCH_TRACE_LOGITS':str(exact/'logits.bin'),
         'DS4_BENCH_SNAPSHOT_DIR':str(exact/'snapshots'),
         'DS4_BENCH_FORCE_SNAPSHOT':'1'})
    reference = PREVIOUS/('exact-'+tag+'-splash')
    run('compare-'+tag,['python3',OUT/'compare_capture.py',exact/'logits.bin',
        reference/'logits.bin'],allow_difference=True)
    snapshots=[]
    for p in sorted((exact/'snapshots').glob('*.bin')):
        q=reference/'snapshots'/p.name
        snapshots.append({'frontier':int(p.stem),'main_sha256':digest(p),
                          'final_sha256':digest(q),'bytes':p.stat().st_size})
    (OUT/('snapshots-'+tag+'.json')).write_text(json.dumps(snapshots,indent=2)+'\n')
    for i,arm in enumerate(manifest['order'],1):
        label=f'perf-{tag}-{i}-{arm}'
        run(label,['./ds4-bench',*args,'--csv',OUT/(label+'.csv')],TREES[arm])
        rows=list(csv.DictReader((OUT/(label+'.csv')).open()))
        assert len(rows)==(7 if tag=='v41' else 5)
        assert all(int(r['ctx_tokens'])+int(r['gen_tokens'])<32768 for r in rows)
print('DONE',flush=True)

#!/usr/bin/env python3
"""Derive paired throughput, HTTP timings and exactness from the retained runs."""
import csv
import json
from pathlib import Path
import re
from statistics import mean, harmonic_mean

P = Path(__file__).resolve().parent
ORDER = ['base', 'splash', 'splash', 'base']

def stats(values):
    return {'runs': values, 'mean': mean(values), 'harmonic_mean': harmonic_mean(values),
            'spread_percent': 100 * (max(values)-min(values))/mean(values)}

result = {}
for tag in ['v41', 'glm']:
    model = {}
    paths = [P/f'perf-{tag}-{i}-{arm}.csv' for i, arm in enumerate(ORDER, 1)]
    if all(p.exists() for p in paths):
        runs = [list(csv.DictReader(p.open())) for p in paths]
        if all(len(r) == (7 if tag == 'v41' else 5) for r in runs):
            performance = []
            for index in range(len(runs[0])):
                record = {'frontier': int(runs[0][index]['ctx_tokens']),
                          'prefill_tokens': int(runs[0][index]['prefill_tokens'])}
                for metric in ['prefill_tps', 'gen_tps', 'gen_first_ms', 'gen_steady_tps']:
                    arms = {a: stats([float(r[index][metric]) for r, arm in zip(runs, ORDER) if arm == a])
                            for a in ['base', 'splash']}
                    aggregation = 'mean' if metric.endswith('_ms') else 'harmonic_mean'
                    arms['change_percent'] = 100 * (arms['splash'][aggregation]/arms['base'][aggregation]-1)
                    record[metric] = arms
                performance.append(record)
            model['performance'] = performance
            totals = {a: stats([sum(float(r['prefill_tokens'])/float(r['prefill_tps']) for r in rows)
                               for rows, arm in zip(runs, ORDER) if arm == a]) for a in ['base', 'splash']}
            totals['change_percent'] = 100 * (totals['splash']['mean']/totals['base']['mean']-1)
            model['cumulative_prefill_seconds'] = totals
    path = P/f'exactness-{tag}.json'
    if path.exists():
        model['exactness'] = json.loads(path.read_text())
    paths = [P/f'http-{tag}-{i}-{arm}' for i, arm in enumerate(ORDER, 1)]
    if all((p/'results.json').exists() for p in paths):
        runs = [json.loads((p/'results.json').read_text()) for p in paths]
        if all(len(r) == 4 for r in runs):
            cases = []
            for index in range(4):
                record = {'name': runs[0][index]['name'], 'usage': runs[0][index]['usage']}
                for metric in ['ttft_ms', 'e2e_ms']:
                    arms = {a: stats([r[index][metric] for r, arm in zip(runs, ORDER) if arm == a])
                            for a in ['base', 'splash']}
                    arms['saved_ms'] = arms['base']['mean']-arms['splash']['mean']
                    arms['change_percent'] = 100 * (arms['splash']['mean']/arms['base']['mean']-1)
                    record[metric] = arms
                cases.append(record)
            model['http'] = cases
            stores = []
            for path, arm in zip(paths, ORDER):
                for line in (path/'server.log').read_text().splitlines():
                    if 'kv cache stored ' in line:
                        stores.append({'run': path.name, 'arm': arm, 'line': line})
            model['checkpoint_store_records'] = stores
    path = P/f'http-exactness-{tag}.json'
    if path.exists():
        model['http_exactness'] = json.loads(path.read_text())
    result[tag] = model

(P/'results.json').write_text(json.dumps(result, indent=2)+'\n')
for tag, model in result.items():
    print(tag)
    for row in model.get('performance', []):
        print(row['frontier'], 'prefill', *(round(row['prefill_tps'][a]['harmonic_mean'], 3) for a in ['base', 'splash']),
              f'{row["prefill_tps"]["change_percent"]:+.3f}%',
              'decode', *(round(row['gen_tps'][a]['harmonic_mean'], 3) for a in ['base', 'splash']),
              f'{row["gen_tps"]["change_percent"]:+.3f}%')
    for row in model.get('http', []):
        v = row['ttft_ms']
        print(row['name'], 'TTFT', *(round(v[a]['mean'], 3) for a in ['base', 'splash']),
              'saved_ms', round(v['saved_ms'], 3), 'prompt_tokens', row['usage']['prompt_tokens'])
    if 'exactness' in model:
        print('Exactness', model['exactness']['logits'])

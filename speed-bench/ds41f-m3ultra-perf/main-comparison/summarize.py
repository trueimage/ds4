#!/usr/bin/env python3
"""Summarize completed ABBA sweeps without pooling historical baselines."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent
manifest = json.loads((OUT/'manifest.json').read_text())
history = json.loads((OUT/'history.json').read_text())
assert all(r['exit_code'] == 0 for r in manifest['runs'])
result = {'base': manifest['trees']['main']['commit'],
          'measured_final': manifest['trees']['final']['commit'],
          'curated_implementation': history['scoped_implementation_commits'][-1]['commit'], 'models': {}}
for tag in ('v41','glm'):
    raw = {arm: [list(csv.DictReader(p.open())) for p in sorted(OUT.glob('perf-'+tag+'-*-'+arm+'.csv'))]
           for arm in ('main','final')}
    assert all(len(v)==2 for v in raw.values())
    n = 7 if tag=='v41' else 5
    assert all(len(rows)==n for runs in raw.values() for rows in runs)
    rows=[]
    for i in range(n):
        row={k:int(raw['main'][0][i][k]) for k in ('ctx_tokens','prefill_tokens','gen_tokens')}
        assert row['ctx_tokens']+row['gen_tokens']<32768
        for metric in ('prefill_tps','gen_tps','gen_first_ms'):
            values={arm:[float(run[i][metric]) for run in runs] for arm,runs in raw.items()}
            means={arm:statistics.harmonic_mean(v) if metric.endswith('tps') else statistics.mean(v)
                   for arm,v in values.items()}
            row[metric]={'runs':values,'means':means,
                'change_percent':100*(means['final']/means['main']-1),
                'spread_percent':{arm:100*(max(v)-min(v))/statistics.mean(v) for arm,v in values.items()}}
        rows.append(row)
    exact=json.loads((OUT/('compare-'+tag+'.stdout')).read_text())
    snaps=json.loads((OUT/('snapshots-'+tag+'.json')).read_text())
    assert exact['exact'] and all(x['main_sha256']==x['final_sha256'] for x in snaps)
    result['models'][tag]={'rows':rows,'exactness':exact,'snapshots':snaps}
(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')

lines=['# Current main versus the final Flash optimizations', '',
    'Measured on 2026-09-26 against upstream `main` **`0aaea5a`**, the base of PR #1090.',
    'The final measured snapshot is `f768551`; the consolidated implementation at',
    f"`{result['curated_implementation']}` has byte-identical runtime sources and Makefile.",
    'History consolidation changes commit boundaries, not the measured implementation.', '',
    'Apple M3 Ultra, 80 GPU cores, 512 GiB, macOS 27.0. Only the upstream maintainer’s',
    '`DeepSeek-V4.1-Flash-Q4.gguf` and `GLM-5.3-Flash-Q4_K.gguf` were used.',
    'Resident Metal, default tuning, no SSD streaming or drafter; Engram tables warmed',
    'before the DeepSeek runs. One model process at a time.', '',
    'Each model ran **main / final / final / main**, two processes per arm, with',
    '128 greedy decode steps per frontier and a fixed 32,768-token allocation.',
    'The largest frontier is 32,639, leaving space for generation and the sentinel.',
    'Rates are harmonic means of the two rounded CSV rates. Prefill is incremental',
    'between frontiers; the table shows the work in each interval. These are local',
    'measurements of one corpus, not confidence intervals or cold-cache guarantees.', '']
for tag,name in [('v41','DeepSeek V4.1 Flash Q4'),('glm','GLM 5.3 Flash Q4_K')]:
    data=result['models'][tag]
    lines += ['## '+name,'',
        '| Frontier | New prefill tokens | Prefill t/s, main → final | Change | Decode t/s, main → final | Change |',
        '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in data['rows']:
        p,g=row['prefill_tps'],row['gen_tps']
        lines.append(f"| {row['ctx_tokens']:,} | {row['prefill_tokens']:,} | {p['means']['main']:.2f} → {p['means']['final']:.2f} | {p['change_percent']:+.2f}% | {g['means']['main']:.2f} → {g['means']['final']:.2f} | {g['change_percent']:+.2f}% |")
    ex=data['exactness']
    lines += ['',f"**Exactness against main:** {ex['floats']:,} FP32 logit values across {ex['rows']:,} complete",
        f"vocabulary rows match bit for bit, through position {ex['max_position']:,}.",
        f"All {len(data['snapshots'])} complete session snapshots match byte for byte.",
        f"Logit SHA-256: `{ex['logits_sha256']}`.",'']
    maxspread={m:max(max(r[m]['spread_percent'].values()) for r in data['rows']) for m in ('prefill_tps','gen_tps')}
    lines += [f"Largest within-arm repeat spread: {maxspread['prefill_tps']:.2f}% prefill and {maxspread['gen_tps']:.2f}% decode.",'']
lines += ['## Checkpoint latency and additional checks','',
    'The separate [clean-849e039 comparison](../splash/vs-849e039/README.md) isolates',
    'the current changes from the previously published branch tip. At the 10,240-token',
    'checkpoint continuation, HTTP time to first content fell by **13.9 ms for DeepSeek**',
    'and **60.7 ms for GLM**. Those are measured HTTP TTFT savings against `849e039`,',
    'not against main; bulk prefill stayed flat in that incremental comparison.',
    'All 32 HTTP responses matched, as did 40 cache images after masking only timestamps.', '',
    'The [Splash record](../splash/README.md) includes zero-tolerance GPU oracles,',
    'the independent 637-million-logit GLM HC soak, verification-prefix parity,',
    'SSD checkpoint save/load tests, bounded server tests and CPU builds.',
    'Older campaign pages retain their original baselines and historical context limits.',
    'This refresh did not run above 32K or test another model, CUDA or physical TP.', '',
    '## Evidence and reproduction','',
    '`manifest.json` records all commands, model identities, source and binary hashes.',
    '`results.json` retains both raw rates, aggregate rates and repeat spreads.',
    'Raw performance CSVs, logs and exactness comparison results are adjacent.',
    'Instrumented capture times are excluded from throughput results. The captures',
    'compare fresh main output with the saved, identically configured final captures',
    'from the immediately preceding clean-849e039 comparison. Large logit/snapshot',
    'images remain local; complete hashes identify them.', '',
    'Build `ds4-bench` separately at main and at the consolidated implementation, and',
    'run each binary from its matching checkout so it loads that checkout’s shaders.',
    'Clear diagnostic DS4/MTL/METAL/ASTRA environment overrides. Use `run.py` as the',
    'record of the original absolute paths, or repeat this command in ABBA order:', '',
    '```sh', './ds4-bench -m /path/to/DeepSeek-V4.1-Flash-Q4.gguf \\',
    '  --prompt-file speed-bench/promessi_sposi.txt \\',
    '  --ctx-start 512 --ctx-max 32639 --ctx-alloc 32768 \\',
    '  --step-mul 2 --gen-tokens 128 --csv /path/to/run.csv', '```','',
    'For GLM, use its named GGUF and `--ctx-start 2048`. The public benchmark’s',
    '`gen_first_ms` is the first decode evaluation after prefill; it is not HTTP TTFT.','']
(OUT/'README.md').write_text('\n'.join(lines))
print(json.dumps({tag:{m:[min(r[m]['change_percent'] for r in d['rows']),max(r[m]['change_percent'] for r in d['rows'])]
    for m in ('prefill_tps','gen_tps')} for tag,d in result['models'].items()},indent=2))

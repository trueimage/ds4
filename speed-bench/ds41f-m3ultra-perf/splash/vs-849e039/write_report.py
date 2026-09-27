#!/usr/bin/env python3
import json
from pathlib import Path
from statistics import mean

P = Path(__file__).resolve().parent
data = json.loads((P/'results.json').read_text())
manifest = json.loads((P/'manifest.json').read_text())
assert all(r['exit_code'] == 0 for r in manifest['runs'])
for name in ['v41', 'glm']:
    assert all(k in data[name] for k in ['performance', 'http', 'exactness', 'http_exactness'])

lines = ['## Results', '',
         'These are fresh measurements of **clean `849e039` versus Splash `f768551`**.',
         'Positive throughput changes are faster; positive TTFT savings are shorter waits.', '']
for tag, title in [('v41', 'DeepSeek V4.1 Flash Q4'), ('glm', 'GLM-5.3 Flash Q4_K')]:
    d = data[tag]
    lines += [f'### {title}', '',
              '| Frontier | New prefill tokens | Prefill tokens/s, base → Splash | Change | Decode tokens/s, base → Splash | Change |',
              '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in d['performance']:
        a, b = r['prefill_tps'], r['gen_tps']
        lines.append(f'| {r["frontier"]:,} | {r["prefill_tokens"]:,} | {a["base"]["harmonic_mean"]:.2f} → {a["splash"]["harmonic_mean"]:.2f} | {a["change_percent"]:+.2f}% | {b["base"]["harmonic_mean"]:.2f} → {b["splash"]["harmonic_mean"]:.2f} | {b["change_percent"]:+.2f}% |')
    t = d['cumulative_prefill_seconds']
    lines += ['', f'Total measured incremental prefill time: {t["base"]["mean"]:.4f} → {t["splash"]["mean"]:.4f} seconds ({t["change_percent"]:+.3f}%).', '',
              '| Frontier | Prefill repeat spread, base / Splash | Decode repeat spread, base / Splash |',
              '| ---: | ---: | ---: |']
    for r in d['performance']:
        a, b = r['prefill_tps'], r['gen_tps']
        lines.append(f'| {r["frontier"]:,} | {a["base"]["spread_percent"]:.3f}% / {a["splash"]["spread_percent"]:.3f}% | {b["base"]["spread_percent"]:.3f}% / {b["splash"]["spread_percent"]:.3f}% |')
    lines += ['', 'Repeat spreads are descriptive ranges from two runs per arm, not confidence intervals.', '',
              '| HTTP case | Actual input tokens | TTFT ms, base → Splash | Saved ms | Individual base runs, ms | Individual Splash runs, ms |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in d['http']:
        t = r['ttft_ms']
        name = r['name'].replace('http-', '').replace('-continued', ' continuation').replace('-cold', ' cold')
        a, b = t['base'], t['splash']
        av, bv = ' / '.join(f'{x:.2f}' for x in a['runs']), ' / '.join(f'{x:.2f}' for x in b['runs'])
        lines.append(f'| {name} | {r["usage"]["prompt_tokens"]:,} | {a["mean"]:.2f} → {b["mean"]:.2f} | {t["saved_ms"]:+.2f} | {av} | {bv} |')
    e = d['exactness']
    lines += ['', f'**Exactness:** {e["logits"]["floats"]:,} FP32 logit values across {e["logits"]["rows"]:,} rows are bit-identical, through position {e["logits"]["max_position"]:,}. All {len(e["snapshots"])} complete snapshots match byte-for-byte. All 16 HTTP responses have matching output, usage and finish reasons; all saved cache images match except their two header timestamps.', '']

lines += ['### Limits', '',
          'This comparison covers the two named GGUFs on resident Metal and the recorded',
          'corpus/HTTP prompts. It does not establish universal performance or exactness',
          'for other prompts, models, contexts above 32K, SSD streaming, CUDA or distributed',
          'hardware. No production source changed as part of this comparison.', '']
readme = (P/'README.md').read_text()
start, end = '<!-- generated-results:start -->', '<!-- generated-results:end -->'
block = start+'\n'+'\n'.join(lines)+'\n'+end+'\n\n'
if start in readme:
    before, after = readme.split(start, 1)
    _, after = after.split(end, 1)
    readme = before+block+after.lstrip('\n')
else:
    readme = readme.replace('## Measurement method\n', block+'## Measurement method\n')
(P/'README.md').write_text(readme)
print('Wrote', P/'README.md')

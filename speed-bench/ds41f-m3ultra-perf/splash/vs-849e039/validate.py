#!/usr/bin/env python3
import csv
import hashlib
import json
from pathlib import Path
import subprocess

P = Path(__file__).resolve().parent
ROOT = Path('/Users/jw/ds4')
M = json.loads((P/'manifest.json').read_text())

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()

assert len(M['runs']) == 23
assert all(r['exit_code'] == 0 for r in M['runs'])
for arm, rec in M['trees'].items():
    tree = Path(rec['worktree'])
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tree, text=True).strip() == rec['commit']
    assert not subprocess.check_output(['git', 'diff', 'HEAD', '--'], cwd=tree)
    for name, expected in rec['source_sha256'].items():
        assert digest(tree/name) == expected, (arm, name)
    for name, expected in rec['binaries'].items():
        assert digest(tree/name) == expected, (arm, name)
    assert digest(P/('capture-'+arm)/'capture-bench') == rec['capture_sha256']
for tag, rec in M['models'].items():
    path = Path(rec['path']); s = path.stat()
    assert s.st_size == rec['bytes'] and s.st_mtime_ns == rec['mtime_ns']
    with path.open('rb') as f:
        assert hashlib.sha256(f.read(8 << 20)).hexdigest() == rec['first_8_MiB_sha256']

floats = snapshots = requests = images = 0
for tag in ['v41', 'glm']:
    e = json.loads((P/f'exactness-{tag}.json').read_text())
    assert e['logits']['exact'] and e['logits']['max_position'] == 32767
    assert all(s['bit_identical'] for s in e['snapshots'])
    floats += e['logits']['floats']; snapshots += len(e['snapshots'])
    h = json.loads((P/f'http-exactness-{tag}.json').read_text())
    assert h['output_usage_finish_identical'] and h['checkpoint_images_identical_except_header_timestamps']
    for i, arm in enumerate(M['order'], 1):
        rows = list(csv.DictReader((P/f'perf-{tag}-{i}-{arm}.csv').open()))
        assert len(rows) == (7 if tag == 'v41' else 5)
        assert all(int(r['gen_tokens']) == 128 and int(r['ctx_tokens'])+128 < 32768 for r in rows)
        assert 'ctx=32768' in (P/f'perf-{tag}-{i}-{arm}.stderr').read_text()
        folder = P/f'http-{tag}-{i}-{arm}'
        values = json.loads((folder/'results.json').read_text())
        assert len(values) == 4
        for v in values:
            u = v['usage']; requests += 1
            assert u['completion_tokens'] == 64 and u['prompt_tokens']+u['completion_tokens'] <= 32768
            assert (u['prompt_tokens_details']['cached_tokens'] == 0) == v['name'].endswith('-cold')
        assert 'kv cache stored tokens=10240 trimmed=0 reason=continued' in (folder/'server.log').read_text()
        assert not list((folder/'kv').glob('*.tmp.*'))
        images += len(json.loads((folder/'images.json').read_text()))

head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
assert head == M['trees']['base']['commit']
status = subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=ROOT)
assert status == (P/'working-tree-before.txt').read_bytes()
original_validation = json.loads((ROOT/'speed-bench/ds41f-m3ultra-perf/splash/validation.json').read_text())
for name, expected in original_validation['changed_file_sha256'].items():
    assert digest(ROOT/name) == expected, name

report = {'status': 'passed', 'completed_run_records': len(M['runs']),
          'bit_identical_logit_values': floats, 'bit_identical_complete_snapshots': snapshots,
          'matching_http_requests': requests, 'matching_http_cache_images_with_only_timestamps_masked': images,
          'maximum_allocated_context': 32768, 'maximum_captured_position': 32767,
          'clean_source_and_binary_identities_verified': True,
          'working_branch_head': head, 'working_tree_status_unchanged': True,
          'current_27_splash_files_still_match_tested_snapshot': True}
(P/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))

checksums = {str(path.relative_to(P)): digest(path) for path in sorted(P.rglob('*'))
             if path.is_file() and path.name != 'artifact-sha256.json'}
(P/'artifact-sha256.json').write_text(json.dumps(checksums, indent=2)+'\n')
print('Hashed artifacts:', len(checksums))

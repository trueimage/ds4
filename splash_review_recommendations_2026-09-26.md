# Splash review: implementation guide for ds4 (2026-09-26)

This guide turns a read-only review of `~/splash` (commit `c64a578`, a Metal
inference engine for Qwen models) into work items for ds4 on branch
`ds41f-m3ultra-perf`. It is written for a coding agent that has not seen the
review. Nothing in it has been built or measured: every gain is an estimate,
and several items end in a measurement that decides whether to continue.

Read these first, in this order:

1. `AGENT.md` (repo rules; they override anything here).
2. The rest of this section (ground rules, environment, verification workflow).
3. `speed-bench/ds41f-m3ultra-perf/README.md`, `host-pipeline/README.md` and
   `mxfp4/README.md` beside it: what has already been done on this branch, how
   it was proven exact, and what was rejected.
4. The item you are implementing.

## Why Splash is fast, and why most of it does not port

Splash has no one-token decode. Every step is an 8-row verify (one committed
row plus seven draft rows) on Apple9 simdgroup-matrix tiles whose per-row
output does not depend on how many rows are live, so extra rows cost almost
nothing. Its kernels reassociate sums and are not bit-identical to anything
(99.3% top-1 agreement with llama.cpp). Its host side is simpler than ds4's:
one serial compute encoder, one command buffer in flight, Metal 4 only for
sparse KV mapping. So under ds4's exactness bar the useful material is
(a) patterns ds4 already uses but has not applied everywhere, (b) the
overheads that keep ds4's exact speculative decoder (DSpark) slower than plain
decode, (c) server-side time-to-first-token work, and (d) one policy decision
(item 22) that only the user can make.

Baseline arithmetic for V4.1 decode on the M3 Ultra: a token reads about
12.6 GB of weights, roughly 16-18 ms at practical bandwidth, against 33.1 ms
measured (MXFP4, 1K context). Stage costs: router + routed experts 9.09 ms,
attention output 5.19, attention projections 4.55, attention core/index/window
4.18, hyper-connections 2.83, shared expert 2.75, head 1.00. The dense Q8_0
stages run near 560 GB/s and the head near 700 GB/s; HC and attention core are
almost pure latency. About 1,900 dispatches run per token; removing ~200 tiny
dispatches measured +0.8%, so one tiny dispatch is worth about 1.3 us.

## Ground rules

- **Exact or gone.** A change to a default path must leave outputs
  bit-identical to the path it replaces: every full-vocabulary FP32 logit,
  every serialized session snapshot byte, every greedy generation byte. Float
  reordering is not an acceptable explanation for a difference. The only
  non-exact item here (22) is opt-in, off by default, and needs the user's
  approval before any integration work.
- **One admission rule.** V4.1 M3 Ultra exact paths are admitted through
  `ds4_gpu_dsv41_exact_admitted(test_flag, rollback_env)` (`ds4_metal.m`,
  near line 10460). It refuses quality mode, SSD streaming and tensor
  parallelism, requires the graph's published measured configuration
  (`ds41_measured_config` in `ds4.c`, published with
  `ds4_gpu_dsv41_set_measured_config`) on an M3 Ultra or the test flag, and
  honours the rollback variable. New V4.1 work uses it with a new rollback
  variable named `DS4_METAL_DISABLE_V41_<FEATURE>` (host-side:
  `DS4_DISABLE_V41_<FEATURE>`). GLM work uses `ds4_gpu_glm53_tuning_available()`
  / `glm53_flash_feature_enabled()` and keeps the aggregate rollback
  `DS4_METAL_DISABLE_GLM53_FLASH_TUNING` working. SSD streaming, TP, CUDA,
  vision, imatrix and `--quality` keep their existing paths unless an item
  says otherwise.
- **Rollback variables are diagnostic oracles, not features.** `AGENT.md`
  forbids permanent semantic variants behind flags; a rollback switch that
  restores the identical original path is fine.
- **Every new kernel gets a model-free GPU oracle** that compares the new and
  original paths bit for bit, under `MTL_DEBUG_LAYER=1`, including NaN, both
  infinities, subnormals, signed zeros, shape edges (tails, single row, max
  rows) and untouched guard bytes around every output, plus a dispatch counter
  proving the new path actually ran. Extend the existing fixtures where they
  fit: `tests/test_deepseek41_fusions.c` (`make test-deepseek41-fusions`),
  `tests/test_deepseek41_topk.c`, `tests/test_deepseek41_q4_tail.c`,
  `tests/test_deepseek41_engram_admission.c`; add new V4.1 oracles to
  `DEEPSEEK41_EXACT_DEFAULT_TESTS` in the `Makefile` so `make test` runs them.
  Test flags live in the enum in `ds4_gpu.h` (near line 255) and are set with
  `ds4_gpu_test_set_flags`.
- **Fast-math is on** in the runtime Metal library. A fused or rewritten
  kernel must evaluate the same expressions as the original: call the same
  inline helper, or pin rounding points the way existing exact kernels do
  (`volatile` copies, `#pragma clang fp contract(off)`), then prove it with the
  oracle. Do not trust "same source text" without the test.
- **Cross-threadgroup data.** Any partial written by one threadgroup and read
  by another in the same dispatch must go through atomics or `coherent(device)`
  memory. Splash commit `f04ad4e` found plain `device` pointers read stale data
  on the two-die M3 Ultra. See section G before adding a last-arriver.
- **History.** One item per commit where possible; no intermediate commit may
  be unsafe (squash a bug with its fix). Commit messages and PR text carry no
  AI-tool attribution or session links.
- **Scope of verification** (from `AGENT.md`): test the normal Metal path and
  its speed; test SSD streaming if a shared path changed; ask the user before
  distributed/TP tests or CUDA access.

## Environment

- Machine: Apple M3 Ultra, 80 GPU cores (Apple9 family: simdgroup matrices,
  no M5 neural accelerator), 512 GiB. One huge model process at a time: the
  instance lock is intentional. Do not run large CPU inference.
- Models (`gguf/`, gitignored): `DeepSeek-V4.1-Flash-Q4.gguf` (Q4_K routed
  experts; 294 GiB resident plus 188.8 GiB of Engram tables read from SSD),
  `DeepSeek-V4.1-Flash-MXFP4.gguf` (native MXFP4 experts, 278 GiB resident),
  `DeepSeek-V4.1-Flash-DSpark-support.gguf` (DSpark draft),
  `GLM-5.3-Flash-Q4_K.gguf`.
- **The working tree has uncommitted work** (MXFP4 admission, DSpark,
  `tests/test_deepseek41_dspark.c` untracked). Never revert or stash it
  without asking. Ask the user to commit it (or how to handle it) before
  starting, and use that commit as the baseline.
- **Shaders compile at startup** from `metal/*.metal` of the checkout the
  binary runs in, plus the prelude string in `ds4_metal.m` (near line 4825).
  Build the baseline in its own worktree
  (`git worktree add ../ds4-base <baseline-commit>`), run every binary from its
  own checkout, and never switch commits while a run is in flight.
- `make cpu` overwrites the Metal `ds4`/`ds4-bench` binaries with CPU builds;
  delete them and `make all` to restore.
- `ds4_test` reports 9 pre-existing failures when `ds4flash.gguf` links to the
  V4.1 model (its vectors target V4), which stops `make test` before the GPU
  oracles. Run the remaining lines of the `test:` recipe directly and report
  the 9 as pre-existing.
- **Engram page-cache state skews V4.1 decode numbers.** Cold tables cost
  ~0.5% after the host-pipeline work (much more on older code). Warm both
  tables before every decode comparison and record the state. This script
  reads them once through the cache (about 31 s):

```python
#!/usr/bin/env python3
"""Read both V4.1 Engram tables once through the page cache (warm state).
Usage: warm_engram.py MODEL.gguf [--dry-run]"""
import struct, sys

path = sys.argv[1]
f = open(path, "rb")
u32 = lambda: struct.unpack("<I", f.read(4))[0]
u64 = lambda: struct.unpack("<Q", f.read(8))[0]
def s(): return f.read(u64()).decode("utf-8", "replace")
SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
def val(t):
    if t in SIZES: return f.read(SIZES[t])
    if t == 8: return s()
    if t == 9:
        at, n = u32(), u64()
        return [val(at) for _ in range(n)]
    raise ValueError(f"unknown GGUF type {t}")
assert f.read(4) == b"GGUF"
u32(); n_tensors, n_kv = u64(), u64()
align = 32
for _ in range(n_kv):
    k, t = s(), u32()
    v = val(t)
    if k == "general.alignment": align = struct.unpack("<I", v)[0]
ranges = []
for _ in range(n_tensors):
    name, nd = s(), u32()
    dims = [u64() for _ in range(nd)]
    u32(); off = u64()
    if name in ("blk.1.engram_embd.weight", "blk.14.engram_embd.weight"):
        ranges.append((name, off, dims[0] * dims[1]))  # I8 rows of 264 bytes
data = (f.tell() + align - 1) // align * align
for name, off, n in ranges:
    print(f"{name}: offset {data + off}, {n / 2**30:.1f} GiB", flush=True)
    if "--dry-run" in sys.argv: continue
    with open(path, "rb", buffering=0) as g:
        g.seek(data + off)
        while n > 0:
            chunk = g.read(min(n, 64 << 20))
            if not chunk: break
            n -= len(chunk)
```

## Verification workflow (every item)

1. **Baseline.** Build the baseline commit in its own worktree
   (`make -j8 all` plus any bench/test targets you need). Capture its outputs
   once and keep them:
   - Full logits and snapshots: in each checkout run
     `python3 speed-bench/ds41f-m3ultra-perf/build_capture.py OUTDIR`, then
     ```sh
     mkdir -p OUTDIR/snap
     DS4_BENCH_TRACE_LOGITS=OUTDIR/logits.bin DS4_BENCH_SNAPSHOT_DIR=OUTDIR/snap \
       OUTDIR/capture-bench -m gguf/DeepSeek-V4.1-Flash-Q4.gguf \
       --prompt-file speed-bench/promessi_sposi.txt --ctx-start 512 \
       --ctx-max 32703 --step-mul 4 --ctx-alloc 32768 --gen-tokens 64
     ```
     (`DS4_BENCH_FORCE_SNAPSHOT=1` adds the continued sweep after snapshot
     restoration.) Compare with
     `python3 speed-bench/ds41f-m3ultra-perf/compare_capture.py BASE/logits.bin CAND/logits.bin`
     and `cmp` each snapshot. The script accepts positions up to 32,768; items
     that change long-context behaviour need a longer capture (see
     `host-pipeline/records/long-context-comparison.json` for the 130,816
     sweep that was run before).
   - Record the baseline hash yourself. Recorded values for reference only:
     Q4 `8e06b20e…` and MXFP4 `4d25b0a1…` at `849e039`; the older
     `e022c41d…` in `logit-comparison.json` is not what current code produces.
   - Greedy text: `./ds4 -m MODEL -c CTX -n 400 --temp 0 --nothink --prompt-file FILE`
     for a short, a ~8K and a ~64K prompt; compare bytes.
2. **Oracle.** The item's model-free GPU test passes under
   `MTL_DEBUG_LAYER=1`, and its dispatch counter shows the new path ran.
3. **Full-model exactness.** Logit capture, snapshots and greedy text are
   bit-identical to the baseline, with the default path, and the rollback
   variable restores the baseline path.
4. **Throughput.** Interleaved A/B within one process where a harness exists:
   ```sh
   make -j8 all metal-decode-schedule-bench metal-prefill-variant-bench
   python3 speed-bench/ds41f-m3ultra-perf/run.py --out /tmp/ITEM --name decode-8k -- \
     ./speed-bench/metal_decode_schedule_bench -m gguf/DeepSeek-V4.1-Flash-Q4.gguf \
     --prompt-file speed-bench/promessi_sposi.txt \
     --candidate-env DS4_METAL_DISABLE_V41_<FEATURE> \
     --prefix-tokens 8192 --ctx 32768 --warmup 16 --tokens 256 --include-selection
   python3 speed-bench/ds41f-m3ultra-perf/run.py --out /tmp/ITEM --name prefill-8k -- \
     ./speed-bench/metal_prefill_variant_bench -m gguf/DeepSeek-V4.1-Flash-Q4.gguf \
     --prompt-file speed-bench/promessi_sposi.txt \
     --candidate-env DS4_METAL_DISABLE_V41_<FEATURE> \
     --prefix-tokens 8192 --ctx 32768 --prefill-chunk 8192 --repeats 2
   ```
   With a rollback variable as the candidate, the harness's control is the new
   default, so a gain shows as a negative delta. `run.py` strips inherited
   `DS4_`/`MTL_` variables, records hashes and refuses to overwrite evidence;
   use a fresh `--name` each run. Confirm with uninstrumented `ds4-bench`
   processes in base/candidate/candidate/base order:
   `./ds4-bench -m MODEL --prompt-file speed-bench/promessi_sposi.txt --ctx-start 512 --ctx-max 32703 --step-mul 4 --ctx-alloc 32768 --gen-tokens 64 --csv out.csv`.
5. **Keep rule.** Keep a change only if the gain repeats across two
   interleaved screens and exceeds the run-to-run spread; otherwise keep the
   original path. Record the numbers, commands and hashes beside the existing
   evidence under `speed-bench/ds41f-m3ultra-perf/`.

GLM items use greedy byte-compare
(`ds4 -m MODEL --prompt-file P --raw-prompt --temp 0 -n 128 -c CTX`),
`ds4-bench --dump-frontier-logits-dir` (max |delta| = 0) for prefill, and
`ds4-bench -m MODEL --prompt-file speed-bench/promessi_sposi.txt --ctx-start 2048 --ctx-max 16384 --step-mul 2 --gen-tokens 128 --csv out.csv`
in base/branch/branch/base order.

## Item index and order

Each item section has the same parts: goal and expected gain; why it is
exact; code map (file:line references read on 2026-09-26; re-check them,
the tree moves); implementation steps; gating and rollback; tests;
measurement and acceptance; pitfalls; stop conditions.

**Test-flag bits.** Several items add a test flag (for example
`DS4_GPU_TEST_V41_MXFP4_DECODE`, `_MXFP4_PREFILL`, `_TAIL_CULL8`). As of
2026-09-26 the enum in `ds4_gpu.h` uses bits 0-11, and the GLM coverage
enum uses bits 0-10. Give each new flag the next free bit when its item
lands; never reuse a number from this guide without checking the enum.

| # | Item | Section | Area | Estimated gain | Size | Depends on |
| --- | --- | --- | --- | --- | --- | --- |
| G1 | GLM HC handoff: `device volatile` → atomic publish | G | correctness | none (fix) | S | coherence stress test |
| 1 | Skip redundant `gather_kv` in one-token decode | A | V4.1 decode | 0.3-0.5% | S | — |
| 2 | Per-shape NR0 for Q8_0 matvecs | A | V4.1 decode, verify | 1.5-6% if the sweep finds it | S-M | — |
| 3 | Fuse attention prepare/epilogue chains (A-F) | A | V4.1 decode | ~1.5% | L | — |
| 4 | MXFP4 decode kernels for V4.1's shape | A | V4.1 decode (MXFP4) | 0.5-2% | M | — |
| 5 | Router top-6 fast path | A | V4.1 decode | 0.3-0.6% | M | — |
| 6 | Norm load ordering | A | V4.1 decode | 0.3-0.8% | S | — |
| 7 | Drop the attention reduce dispatch | A | V4.1 decode | 0.5-1% | M | 3A |
| 8 | Cheaper MXFP4 code lookups (experiment) | A | V4.1 decode (MXFP4) | 0.3-1% | M | 4 |
| 15 | Rows-kernel rewrite; batch head, o_a/o_b, router in verify | A | DSpark verify | up to ~40 ms of a 170 ms 6-row verify | M | 2 (NR0 choice) |
| 9 | MXFP4 prefill switches below 2,048 rows | B | V4.1 prefill (MXFP4) | several % at 128-1,536 rows | S | — |
| 10 | Eight-row tail cull | B | V4.1 prefill | 1-2% at 512 | M | — |
| 11 | Half input to mm_id pair kernels | B | V4.1 prefill | 0.5%+ | M | 9, 10 |
| 12 | Split versus fused mm_id A/B (no code) | B | V4.1 prefill | measurement | S | — |
| 13 | Long-context indexer (tiled scorer, top-k) | B | V4.1 decode > 32K | 0.5-1 ms/token at 130K (unverified) | L | measurement; user approval above 32K |
| 14a | GLM KDA decode prefetch and split | C | GLM decode | 0.3-1% | M | G1 |
| 14b | GLM KDA prefill staging | C | GLM prefill | 0.5-1% | M | G1 |
| 16 | Verify with the decode host pipeline | D | DSpark | largest per-row saving, esp. cold | L | Step 0 |
| 17 | One wait per cycle, GPU acceptance | D | DSpark | removes 3-4 waits per cycle | L | 16 |
| 18 | Cost-model gate instead of fixed threshold | D | DSpark | DSpark never slower than decode | M | Step 0 |
| 19 | Cheaper drafter | D | DSpark | mostly prose | M | 18 |
| 20 | Seed draft windows during prefill | D | DSpark | first cycles of each reply | M | — |
| 21 | Expert-grouped verify MoE | D | DSpark | bounded by expert overlap | L | overlap ≥ 15-20% |
| 22 | Row-invariant MMA probe (non-exact; go/no-go) | E | policy | decision data only | M | user go-ahead; 2, 15 |
| 23 | Write-behind kvstore checkpoints | F | server TTFT | 70-400 ms per stored checkpoint | M | — |
| 24 | Single-pass prefix hashing, cached index | F | server TTFT | small, per lookup | S | — |
| 25 | Delta checkpoints (design sketch) | F | server disk and TTFT | large | L | 23, 24 |
| 26 | Memory-pressure admission; `F_NOCACHE` | F | server robustness | avoids Engram eviction | S-M | — |

Item numbers are kept from the first version of this review so older
references still work; section H lists what not to adopt, and the last
section gives the order.

## A. V4.1 decode (exact)

### 1. Skip redundant `gather_kv` in one-token decode

**Goal and expected gain.** Only the eight index-source layers need to
gather. Each gather is one `kernel_get_rows_f32` dispatch of 1,024
threadgroups moving 1 MiB in and 1 MiB out. Removing 30 per token (15 at
pos 0, where the ratio-2 spans have `n_comp = 0`) should save about
0.1-0.15 ms per token (0.3-0.5%) for a few lines of host code.

**Why it is exact.** In `ds41_graph_step` (`ds4.c:41555`; layer loop at
41635), `ds41_attention(g, …, il)` runs for il 0..39 of one token, in order,
on the graph's own scratch. For a non-index-source layer every input of the
gather equals the one at its span's index-source layer:
- `owner` changes only at 2/8/14/20, all index sources (`ds4.c:1384-1390`).
- `selected_comp` is written only by `ds41_attention_pick` at index-source
  layers (40756 returns before any top-k otherwise;
  `ds41_attention_candidates` touches only `index_scores` and `block_*`).
- `n_comp` and `attended` depend only on pos and ratio, constant within each
  span (2-7, 8-13, 14-19 at ratio 2; 20-23 … 36-39 at ratio 1).
- `compressed[owner]` is written only by `ds41_attention_publish` (40705) at
  the kv-source layer, the span's first layer, before its own gather.

`g->selected_kv` is a dedicated `DS41_SCRATCH` buffer (40238) with no views or
aliases; its only writer is the gather (40818), its only reader the attention
call (40821-40825); snapshots exclude it (`ds41_state_spans`, 63793). The
serial encoder and in-order command buffers already order the same kind of
read-after-write for `compressed[owner]`.

What would break it: `selected_kv` is not in `DS41_PREFILL_ROWS`
(40168-40193), so every layer-major `row = *g` copy shares one
`g->selected_kv` across rows. The skip is valid only inside
`ds41_graph_step`, never in the prefill per-row paths (42249-42250, 42262) or
`ds41_graph_step_batch_spec` (42505, DSpark verify and session batches).

**Code map.** Gather call `ds4.c:40816-40819`; wrapper
`ds4_gpu_dsv41_gather_kv` (`ds4_metal.m:49453`); graph struct `ds4.c:40257-40281`;
measured config published at step entry (41558).

**Implementation steps.**
1. In `ds4_metal.m` add `int ds4_gpu_dsv41_gather_reuse_admitted(void)`
   returning
   `ds4_gpu_dsv41_exact_admitted(DS4_GPU_TEST_V41_FUSIONS, "DS4_METAL_DISABLE_V41_GATHER_REUSE")`;
   declare it in the `#ifdef __APPLE__` block of `ds4_deepseek41_gpu.h`.
2. Add `bool gather_reuse;` to `ds41_gpu_graph` with the comment "true only
   during ds41_graph_step's one-token sweep; struct copies made anywhere else
   must see false".
3. In `ds41_graph_step` set `g->gather_reuse = ds4_gpu_dsv41_gather_reuse_admitted();`
   (Apple only) immediately before the loop at 41635 and clear it immediately
   after (before 41679; the loop has no `return`).
4. In `ds41_attention`: `const bool reuse = g->gather_reuse && !ds41_index_source(il);`
   and gather only when `n_comp && !reuse`. Comment the span invariant.
5. No cross-token invalidation is needed: restore, rewind, prefill, verify and
   failed steps either start a fresh sweep or never set the flag.

**Gating and rollback.** The admission call above;
`DS4_METAL_DISABLE_V41_GATHER_REUSE`. CUDA/TP island decode (41299) never sets
the flag.

**Tests.** Admission oracle in `tests/test_deepseek41_fusions.c` (1 under the
test flag, 0 under quality, SSD and the env var). Default-versus-rollback
arms in `tests/test_deepseek41_live_edges.c`: ≥ 16 steps from the near-32K
seed plus steps at pos 0-3; full-vocabulary logits bitwise, greedy tokens,
snapshots byte-for-byte; per-step `ds4_gpu_encoder_count()` delta exactly −30
(−15 at pos 0). `--verify-parity` (verify still gathers every row while
decode reuses). Standard capture including continued prefill after restore.

**Measurement and acceptance.** Standard decode screen with
`--candidate-env DS4_METAL_DISABLE_V41_GATHER_REUSE` at 8192 and 32495.
`DS4_METAL_V41_DECODE_HOST_PROFILE=1` should show ~30 fewer compute encodes
per step. Keep if every comparison is bitwise identical and the default is
faster in both screens.

**Pitfalls.** Never derive reuse from `il` alone outside the sweep. A future
`row = *g` copy inside the decode loop would inherit the flag. Do not also
skip the F32→F16 comp staging in `g_flash_attn_kv_buffer` (a global scratch
shared with the drafter).

**Stop if.** Any logit, token or snapshot mismatch; or the default measures
slower (that would mean the removed gather was hiding an ordering bug, which
needs investigation, not shipping).

### 2. Per-shape rows per threadgroup (NR0) for V4.1 Q8_0 decode matvecs

**Goal and expected gain.** Every V4.1 Q8_0 matvec computes a fixed NR0 = 2
output rows per threadgroup. Let each shape pick NR0 ∈ {1, 2, 4, 8}. Each
threadgroup reads the whole fp32 input, so on-chip activation traffic is
128/(34·NR0) of the weight bytes: 3.8×, 1.9×, 0.94×, 0.47× for NR0 = 1, 2, 4,
8. The trade-off is threadgroup count (at NR0 = 2): large shapes probably want
4 or 8 (q_b 16,384 threadgroups, head 64,640, o_a 4,096, o_b and shared down
2,560 each); small shapes may want 1 (kv 256, q_a 640). Dense Q8_0 is 172 MB
per layer, 6.9 GB per token plus the 703 MB head, running at ~480-620 GB/s
against the head's ~700 GB/s. Ceiling ~2.7 ms per token; realistic 0.5-2 ms
(1.5-6%). Only the sweep can say.

**Why it is exact.** In `kernel_mul_mv_q8_0_f32_impl`
(`metal/dense.metal:132-200`) neither the block walk (ib0 = sgitg·8 + ix,
stride NSG·8) nor the per-row chain (`sumq` over i = 0..7 from 0.f, then
`sumf[row] += sumq*d`) references NR0; rows share only `yl`.
`helper_mv_reduce_and_write` (85-129) gives each row its own zero-filled
32-float shmem slot, so for any NR0 the second `simd_sum` sees the same
partials in lanes < NSG and +0 elsewhere. The policy must never change NSG
(4, or 8 when out_dim > 65536; `DS4_METAL_Q8_MV_NSG` applies to both paths
alike). Under fast-math identical source is not proof; the oracle is.

Tails (ne01 % NR0 ≠ 0) stay arithmetically exact (invalid rows accumulate
only into their own slot; the write is guarded by `r0 + row < ne01`), but
their weight reads run up to NR0−1 rows past the range. So the policy
requires out_dim % NR0 == 0 and otherwise uses 2. All V4.1 shapes are
multiples of 8. If a tail is ever needed, clamp the row pointer in the new
template only, as `kernel_mul_mv_q8_0_f32_pair` does (`dense.metal:476-483`).

**Code map.** NR0 is baked in by the prelude `#define N_R0_Q8_0 2`
(`ds4_metal.m:4837`), used as a template argument; host side
`ds4_gpu_make_q8_0_mv_dispatch` (`ds4_metal.m:5536-5546`) sets nr0 = 2 and
smem = 32·2·4 bytes; grid = ceil(N/nr0).

| Shape | Kernel | Host site | Graph caller |
|---|---|---|---|
| q_a, kv, o_b, shared down | `kernel_dsv41_mul_mv_q8_0_f32_bf16` (`dense.metal:287-298`) | `ds4_gpu_dsv41_matmul_q8_0_bf16` (`ds4_metal.m:52010-52053`; strcmp on the dispatch name at 52026) | `ds41_matmul` (`ds4.c:40510`) |
| q_b | `kernel_mul_mv_q8_0_f32` (204-215), then a separate BF16 dispatch | one-token branch of `ds4_gpu_matmul_q8_0_legacy_tensor` (19706-19726) | `ds41_matmul_rows` → `ds41_matmul_batch`, count 1 (`ds4.c:40524`) |
| head | `kernel_mul_mv_q8_0_f32` | same branch, nsg = 8 at 19708 | `ds41_output_projection` (40575) |
| shared gate/up | `kernel_dsv41_shared_gate_up` (729-747; impl `<NR0,…>` 582-708) | `ds4_gpu_shared_gate_up_swiglu_q8_0_impl` (20694; dispatch 20757-20785; smem 2·dispatch.smem) | — |
| o_a | `kernel_dsv4_attn_out_low_q8_0_f32` (`metal/moe.metal:3392`; `N_R0_Q8_0` at 3436) | `ds4_gpu_attention_output_low_q8_tensor` (27246; nsg 4 fixed at 27324) | — |

Precedents: F16 NR0 = 4 for out 512/1024 with in ≥ 4096 (21381-21384);
`ds4_gpu_plain_mv_single_row` (5553). Not in V4.1 decode:
`kernel_mul_mv_q8_0_f32_pair` (V4 only) and the TP flag kernel.

**Implementation steps.**
1. In `metal/dense.metal`, leave existing entries untouched. Add a thin entry
   `template<short NR0, bool BF16> kernel void kernel_mul_mv_q8_0_f32_nr(...)`
   calling `kernel_mul_mv_q8_0_f32_impl<NR0, constant ds4_metal_args_mul_mv &, BF16>`;
   instantiate `kernel_mul_mv_q8_0_f32_nr{1,4,8}` and
   `kernel_dsv41_mul_mv_q8_0_f32_bf16_nr{1,4,8}`; add
   `kernel_dsv41_shared_gate_up_nr{1,4,8}` (impl `<NR0,true,0,true>`) and an
   NR0-template twin of the o_a kernel in `metal/moe.metal`. Use templates:
   a runtime `switch(args.nr0)` taxes NR0 = 2's register allocation, and
   function constants cannot size `sumf[NR0]`.
2. Beside `ds4_gpu_make_q8_0_mv_dispatch`, add
   `static int16_t ds4_gpu_dsv41_q8_nr0(uint64_t in, uint64_t out)`: return 2
   unless `ds4_gpu_dsv41_exact_admitted(DS4_GPU_TEST_V41_FUSIONS, "DS4_METAL_DISABLE_V41_Q8_NR0")`;
   then the diagnostic override `DS4_METAL_V41_Q8_NR0` (`all:4` or
   `1280x32768:8,5120x512:1`); then a const `{in, out, nr0}` table filled
   from the sweep. Reject NR0 ∉ {1,2,4,8} or out % NR0 ≠ 0. Add a name
   helper that maps NR0 = 2 to the existing names.
3. Apply the policy after `ds4_gpu_make_q8_0_mv_dispatch()` at the four
   one-token sites: nr0, smem = 32·nr0·4 (×2 for gate/up), args.nr0, name. Do
   not change `ds4_gpu_make_q8_0_mv_dispatch` itself: V4, GLM and Qwen use it,
   and 52026 compares its name.
4. Set a test counter bit (e.g. `g_test_v41_fusions_dispatches |= 32u`) when
   NR0 ≠ 2 is dispatched.

**Gating and rollback.** M3 Ultra measured config or the test flag; never
quality, SSD or TP. `DS4_METAL_DISABLE_V41_Q8_NR0=1` restores NR0 = 2
everywhere. `DS4_METAL_V41_Q8_NR0` is the diagnostic sweep override.

**Tests.** New section in `tests/test_deepseek41_fusions.c` modelled on the
BF16-store block (152-211): raise the fixture's `bytes` (line 21) to ≥ 46 MB so
the real q_b (1280→32768) and o_b (8192→5120) fit; also 5120→1280, 5120→512,
5120→2304, 2304→5120, the o_a groups (4096→1024 ×8) and a 512→65544 head
surrogate for the nsg = 8 branch. Force NR0 ∈ {1,4,8} through the override and
compare against the rollback for the plain, BF16-store, gate/up and o_a
kernels, reusing the exceptional scales and inputs (NaN, ±Inf, −0, subnormal,
1e18, zeros) and 16-byte guards. out_dim % NR0 ≠ 0 (1282, 516) must fall back
(counter bit clear); quality, streaming and rollback must keep NR0 = 2.

**Measurement and acceptance.**
- Sweep: add `--q8-nr0-sweep` to `tests/test_deepseek41_dspark.c` beside the
  q_b loop (313-327): per site × NR0, set the override, encode 40 layers, best
  of 4, print ms and GB/s. Adopt a non-2 NR0 only if ≥ 3% faster.
- A/B: the standard decode screen with
  `--candidate-env DS4_METAL_DISABLE_V41_Q8_NR0` at prefixes 512, 8192 and
  32495, and once on the MXFP4 file. Keep only if ≥ +1.0% at every context.
- Exactness: standard capture, default versus rollback and versus the
  baseline worktree.

**Pitfalls.** New names occupy fast pipeline cache slots (64;
`ds4_metal.m:640`); when full, lookups silently fall back to NSDictionary.
q_b is rounded by a separate dispatch, so the policy must reach the legacy
branch as well as the BF16 function.

**Stop if.** Any oracle bit differs (do not add pragmas to force a match: the
default kernel's compiled contraction is the reference), no shape improves
≥ 3%, the A/B is below +1% at any context, or any capture differs.

### 3. Fuse V4.1 attention prepare and epilogue chains

**Goal and expected gain.** In the admitted configuration
`ds4_gpu_tensor_copy` is a compute dispatch (`kernel_ds4_copy_words`,
`ds4_metal.m:9554`), so each copy counts. Current per-token dispatches:

| Chain | Dispatches per token |
| --- | --- |
| Per layer: rope q, rope kv, FP8 quantize, window copy, BF16(heads), inverse rope | 6 × 40 = 240 |
| Adjacent `ds41_norm(kv)` (RMS + BF16) | 80 |
| q_b BF16 (`ds41_matmul_rows` → matvec then quantize) | 40 |
| q_a norm BF16 | 40 |
| Index q rope + quantize | 16 |
| Compression boundary (~10 per event, ~2.5 events/token) | ~25 |

Proposed fusions in priority order:

| ID | Fusion | Saved per token |
| --- | --- | --- |
| A | Epilogue (BF16 + inverse RoPE) into the attention reduce | 80 |
| B | Prepare kernel (kv RMS + BF16 + RoPE + FP8 + window store; q RoPE) | 200 |
| C | Compression-boundary finalize | ~19 |
| D | Index q RoPE + quantize | 8 |
| E | RMS + BF16 kernel (q_a, compressor norms) | ~42 |
| F | q_b through the existing BF16-store matvec | 40 |

About 390 dispatches, roughly 0.5 ms or 1.5% at ~1.3 µs each.

**Why V4's fused kernels do not apply.**
`kernel_dsv4_qkv_rms_norm_kv_rope_fp8_store_f32` (`metal/norm.metal:255`;
wrapper `ds4_metal.m:22984`) uses V4's YaRN affine rope
(`ds4_rope_tail_pair_affine_row`, `dsv4_rope.metal:435`) on F32 without BF16
and a different FP8 quantizer. `kernel_flash_attn_ext_vec_reduce_rope`
(`dsv4_rope.metal:594`) and the packed32 kernel (gate at `ds4_metal.m:29766`)
rotate the unrounded F32 output; V4.1 applies BF16 first (`ds4.c:40826-40827`).

**Why it is exact (preservation rules).**
1. BF16: call `dsv41_bf16` (`dsv41.metal:9`); integer-only, and it also stops
   FP contraction between stages.
2. RoPE: copy `kernel_dsv41_rope` (`dsv41.metal:47`) verbatim into one
   `static inline` helper used only by the new kernels (leave the original as
   the reference). Lane p handles the pair at `width − 64 + 2p`;
   `theta = float(pos) * freq[p]`, `c = precise::cos`, `s = ±precise::sin`,
   then `dsv41_bf16(re*c − im*s)` and `dsv41_bf16(re*s + im*c)`. Share the
   host frequency table: move the `dispatch_once` table
   (`ds4_metal.m:49211-49228`) into a helper inside the same
   `#pragma float_control(precise)` region.
3. Quantize: match `kernel_dsv41_quantize` (`dsv41.metal:62-86`) exactly: one
   full 32-lane simdgroup per block (lanes ≥ 16 in mode 3 contribute 0.0f);
   `value = dsv41_bf16(x)`, `amax = simd_max(abs(value))`; same formulas,
   constants and helpers (`dsv41_pow2_ceil`, `dsv4_e4m3fn_dequant`/`e2m1` at
   `dsv4_kv.metal:66/93`); keep the `abs(value)/scale` form.
4. RMSNorm: reproduce `kernel_rms_norm_fuse_impl<float4,2>`
   (`norm.metal:17-71`) as dispatched by `ds4_metal.m:22674`: nth =
   `ds4_gpu_rms_norm_threads(n)` (128 for 512, 32 for 128, 320 for 1280);
   strided `sumf += dot(x, x)` from 0.0f; `simd_sum`; simdgroup 0 zeroes the 32
   shmem slots before the first barrier; lane 0 stores; barrier;
   `simd_sum(shmem[tiisg])`; `mean = sumf/ne00` (runtime int32);
   `1.0f/sqrt(mean + eps)` (runtime eps); `(x*scale)*w`; then BF16. If the
   oracle ever differs, pin with `volatile` as `kernel_dsv41_hc_norm` does
   (`dsv41.metal:362`). All threads execute every barrier.
5. Pass at runtime anything the original receives at runtime (width, eps,
   pos, table).
6. Reduce epilogue: call the same noinline `ds4_flash_attn_vec_reduce_row`
   (`flash_attn.metal:1404`), the precedent that keeps
   `kernel_flash_attn_ext_vec_reduce_rope` exact.
7. Write the same bits to the scratch tensor and the cache slot or row.

**Code map.** `ds4.c`: `ds41_attention` 40803-40834,
`ds41_attention_project` 40792, `ds41_attention_publish` 40705-40733,
`ds41_attention_select_published` 40765-40784, `ds41_norm` 40644,
`ds41_rope` 40663, `ds41_matmul_rows` 40569. `ds4_metal.m` wrappers: 49200
(rope), 49257 (quantize). Kernels go in `metal/dsv41.metal`, which follows
`flash_attn.metal` and `dsv4_kv.metal` in the source concatenation
(`ds4_metal.m:4904-4930`). Wrappers return 1 when encoded, 0 when unsupported
(including a missing pipeline, which must fall back) and −1 on failure, like
`ds4_gpu_dsv41_hc_norm`.

**Implementation steps.**
- A (epilogue): add `kernel_flash_attn_ext_vec_reduce_dsv41` with function
  constants 500/501 and a getter modelled on `ds4_metal.m:3923`; it calls
  `reduce_row`, then `threadgroup_barrier(mem_device)`; threads handle BF16 for
  indices < 448; threads 0..31 do BF16 then inverse RoPE on pairs 448..511. Arm
  it with a new global (not V4's `g_decode_attn_rope_fuse`):
  `ds4_gpu_dsv41_arm_attention_epilogue(pos, compressed)`, consumed by both
  `gathered_heads` (29690) and `raw_heads` (28112). In `ds4.c` run `ds41_bf16`
  + inverse rope only if `!ds4_gpu_dsv41_attention_epilogue_used()`; disarm on
  failure. If item 7 variant B lands, A's epilogue moves into that kernel.
- B (prepare): stop `ds41_attention_project` after the kv matmul when
  `!projected`; `kernel_dsv41_decode_prepare` does kv RMS + BF16 →
  barrier(mem_device) → RoPE → barrier → FP8 quantize (4 simdgroups × 4
  blocks) → store to `g->kv` and the `window[il]` slot; extra threadgroups
  rope the q heads, one simdgroup per head.
- C (compression boundary): `kernel_dsv41_compress_finalize` with 2
  threadgroups: index_k (RMS128 + BF16 + RoPE + FP4_E8M0 → `index_k` and the
  `index_cache` row) and latent (RoPE + FP4_E4M3 → `latent` and the
  `compressed` row). For ratio 1, matmul directly into `latent` instead of
  copying from `pool_kv`.
- D (index q): RoPE + FP4_E8M0 per head in one dispatch.
- E: `kernel_dsv41_rms_norm_mul_bf16` for the q_a and compressor norms.
- F (q_b): when `tp_world == 1`, call `ds41_matmul(g->q, …, true)`; first add
  the 1280→32768 shape (and 1280→4096 for the index q) to the matvec oracle
  shape list (`tests/test_deepseek41_fusions.c:156`).

**Gating and rollback.** Every sub-step via
`ds4_gpu_dsv41_exact_admitted(DS4_GPU_TEST_V41_FUSIONS, …)` with one env var
each: `DS4_METAL_DISABLE_V41_ATTN_EPILOGUE`, `_ATTN_PREPARE`,
`_COMPRESS_FINALIZE`, `_INDEX_Q_PREPARE`, `_NORM_BF16`, `_QB_BF16`. Only the
decode path (`projected == false`) changes; prefill, verify and the drafter
keep the old chain.

**Tests.** New `tests/test_deepseek41_attention.c`, target
`test-deepseek41-attention` (`MTL_DEBUG_LAYER=1`), added to
`DEEPSEEK41_EXACT_DEFAULT_TESTS`. For each wrapper run the rollback chain and
the fused kernel on identical inputs and compare every output tensor, cache
row and window slot bitwise plus 0xa5 guards. Inputs: random, ×1e30,
subnormals, ±0, ±Inf; NaN with payload only in the low 16 bits (BF16 turns it
into Inf); blocks at each amax clamp; positions 0/1/127/128/32767/1048575;
both frequency tables, forward and inverse. For A, the full attention call
with n_raw {1, 5, 127, 128} × attended {0, 1, 31, 32, 33, 511, 512} and sinks
at ±huge. Refusal under quality, SSD and rollback. Model level: captures,
snapshots, greedy tokens, DSpark parity.

**Measurement and acceptance.** Screen each sub-step at 8K and 32,495 with
its rollback env. Keep a sub-step only with a repeatable gain ≥ 0.3% across
two interleaved screens (the campaign kept T at +0.35%, rejected P at +0.13%).

**Pitfalls.** Quantize threads must form full 32-lane simdgroups. Keep
RoPE → quantize and norm → RoPE barriers as `mem_device` when going through
device memory. The DSpark drafter uses the batch wrappers and is unchanged.

**Stop if.** An oracle mismatch can only be fixed by editing a reference
kernel, or a sub-step shows < 0.2% twice.

### 4. Admit the MXFP4 decode kernels for V4.1's expert shape

**Goal and expected gain.** On the MXFP4 GGUF, routed experts are 9.09 ms
of the 33.1 ms token. V4's MXFP4 decode path (fixed-route pair, full-rows
down, static variants) cannot run for V4.1 because the host gates hard-code
V4's expert shape. Expect +0.5-2% decode on the MXFP4 file; nothing on Q4.

**Premise correction.** Only the two static kernels are literally shaped:
`DS4_MXFP4_PAIR_STATIC_NB` / `_ROW_BLOCKS 128` (`metal/moe.metal:4661-4672`)
and `DS4_MXFP4_DOWN_STATIC_NB 64` / `_ROW_BLOCKS 64` / `_SLOTS 6`
(6538-6546). The fixed-route and full-rows kernels read the shape at runtime;
only the host gates (`ds4_metal.m:41425-41493`) hard-code V4.

**V4.1 values.** Gate/up row 2720 B (160 blocks of 17 B); down row 1224 B (72
blocks); expert bytes 6,266,880 for each matrix. Gate K = 160 blocks = 10
steps per lane pair (uniform). Down K = 72 blocks: 5 or 4 steps per lane (not
uniform; the static down kernel must keep a tail guard).

**Why it is exact.** The same kernels already run for V4 with the same
per-block arithmetic; only the trip counts change, and each output keeps its
own lane-local chain in block order followed by the same `simd_sum`. A static
instantiation must make exactly the same per-lane block assignment and
reduction as the runtime-shaped kernel it replaces; the oracle below proves
that, not the argument.

**Implementation steps.**
1. Template the static impls on `short NB` (and down's `ROW_BLOCKS`); keep the
   V4 instantiations' names and bodies unchanged (AIR diff to confirm), and add
   `kernel_mul_mv_id_mxfp4_pair_swiglu_fixed_route_static_v41_f32` (NB 160) and
   `kernel_mul_mv_id_mxfp4_down_full_rows_static_v41_f32` (NB 72).
2. Add pipeline globals next to the V4 ones (`ds4_metal.m:8409-8414`,
   8428-8429) and their creation next to 11892-11893.
3. Add one predicate before 41437:
   ```c
   const bool v41_mxfp4_decode = use_mxfp4_moe_decode_tg_multiple &&
       expert_in_dim == 5120 && expert_mid_dim == 2304 && out_dim == 5120 &&
       gate_row_bytes == 2720 && gate_expert_bytes == 6266880 &&
       down_row_bytes == 1224 && down_expert_bytes == 6266880 &&
       (n_total_expert == 384 || (g_test_flags & DS4_GPU_TEST_V41_MXFP4_DECODE)) &&
       ds4_gpu_dsv41_exact_admitted(DS4_GPU_TEST_V41_MXFP4_DECODE,
                                    "DS4_METAL_DISABLE_V41_MXFP4_DECODE_SHAPE");
   ```
4. Widen each V4 gate to `(existing V4 literals || v41_mxfp4_decode)`, picking
   the `_v41` static pipeline when the shape is V4.1's. Give each stage its own
   rollback env (fixed-route pair, full-rows down, static pair, static down).
5. Record dispatch bits 1/2/4/8/16 for the five stages so tests can prove
   which kernels ran.
6. Do not touch the M5 TP static kernels.

**Gating and rollback.** The admission call above plus
`DS4_METAL_DISABLE_V41_MXFP4_DECODE_SHAPE` for the whole path and one env per
stage. New test flag `DS4_GPU_TEST_V41_MXFP4_DECODE` (next free bit; see the
bit note in the preamble).

**Tests.** New `tests/test_deepseek41_mxfp4_decode.c`, target
`test-deepseek41-mxfp4-decode` (`MTL_DEBUG_LAYER=1`, added to the default
exact tests): 8 synthetic experts under the test flag; E8M0 exponents
including 0, 1, 254 and 0xff; random, zero and signed-zero codes; activations
with ±0, subnormals, ±Inf, NaN; every stage bitwise against the rollback path
and against each single-stage rollback; dispatch bits prove the new kernels
ran. Also `make test-mxfp4-metal`. Model level: the MXFP4 capture hash and
greedy byte compare.

**Measurement and acceptance.** MXFP4 file, decode screen at 512 and 8K with
`--candidate-env DS4_METAL_DISABLE_V41_MXFP4_DECODE_SHAPE`. Keep the path at
≥ +0.5% in both. Keep each static v41 kernel only if its own A/B (its stage
env) is ≥ +0.2%.

**Pitfalls.** The down static kernel's 72-block K is not a multiple of the
lane-pair stride; a missing tail guard reads past the row silently. Engram
page-cache state moves decode numbers by more than this item's gain: warm
first, interleave arms.

**Stop if.** The total gain is under +0.3%, or any stage differs bitwise.

### 5. Router top-6 without the full bitonic network

**Goal and expected gain.** `kernel_dsv41_router` (`metal/dsv41.metal:295-356`)
sorts all 384 scores with a 512-thread bitonic network (322-335, 45
barriers) to take 6. One simdgroup can find the top 6 in 7 rounds when the
scores are well separated. Router work is ~40 dispatches per token; expected
+0.3-0.6% decode.

**Network semantics that must be reproduced.** Strict comparators; padding
enters as −∞; among equal keys the final order depends on the network's
wiring (it is not "lowest id first"); NaN breaks the order. So a simple
arg-max loop is not equivalent in general.

**Acceptance rule (exact by construction).** Take the fast path only if
(a) all 384 stored scores are finite and not subnormal (integer exponent
test on the stored bits, not a float compare under fast-math), and (b) the
top 7 are strictly decreasing. Under (a) and (b) the first six positions of
any correct sort are unique, so `ids[0..5]` equal the network's, in slot
order. Slot order matters: sum6 accumulates weights in slot order
(`moe.metal:6424-6435`). Anything else falls back to the unchanged network.

**Implementation steps.**
1. Turn the kernel into `template<bool TOP6> kernel_dsv41_router_impl`; host
   names `kernel_dsv41_router` (TOP6 = false, must be unchanged; AIR diff) and
   `kernel_dsv41_router_top6`.
2. After the existing score transform and its barrier, simdgroup 0 reads 12
   stored scores per lane, flags any bad exponent, then runs 7 uniform rounds
   of `simd_max` on the value plus `simd_min` on the id among lanes holding
   it; a round is a tie if the chosen value equals the previous one or occurs
   twice. Lanes `tid < 6` write `ids`.
3. Publish `network = bad || tie` through threadgroup memory and a barrier
   that all 512 threads execute; if set, run the existing sort loops verbatim.
   Keep 512 threads per threadgroup.
4. Host dispatch at `ds4_metal.m:34313-34340`; fallback chain 34505-34610;
   record dispatch bit 32 (1, 2, 4, 8, 16 are used).

**Gating and rollback.** `ds4_gpu_dsv41_exact_admitted(DS4_GPU_TEST_V41_FUSIONS, "DS4_METAL_DISABLE_V41_ROUTER_TOP6")`.

**Tests.** In `tests/test_deepseek41_fusions.c:30-60` run three passes
(unfused, network, top6) and compare ids and weights bitwise. Crafted inputs:
ties at ranks (r, r+1) for r = 0..5; ties at 6/7 and 7/8; a 1-ulp near-tie at
each rank; NaN, ±Inf and subnormals placed below rank 7 (must fall back);
all-equal scores; the fast path must report bit 32 only when taken. Live-edge
arms default versus rollback.

**Measurement and acceptance.** Decode screens at 8K with the rollback env;
keep at ≥ +0.3% in two screens with prefill no worse than −0.2%.

**Pitfalls.** Fast-math may fold `x != x`; use integer tests on `as_type<uint>`.
Do not early-return any thread before the shared barrier.

**Stop if.** Gain under +0.3%, or any crafted case diverges.

### 6. Norm load ordering

**Goal and expected gain.** Hide one dependent DRAM round trip per weighted
norm: today the weight row is read only after the reduction and two barriers.
Per decode token on the admitted path:

| Kernel | Uses per token | Width / threads |
|---|---|---|
| `kernel_dsv41_hc_norm` (attn_norm, ffn_norm) | 80 | 5120 / 1024 |
| `kernel_rms_norm_mul_f32_4`: q_a | 40 | 1280 / 320 |
| `kernel_rms_norm_mul_f32_4`: kv | 40 | 512 / 128 |
| `kernel_rms_norm_mul_f32_4`: output | 1 | 5120 / 1024 |
| `kernel_rms_norm_mul_f32_4`: compressor + indexer_k | ≈5 (kv-source layers) | 512 / 128, 128 / 32 |
| `kernel_rms_norm_f32_4` (HC flat norm, no weight) | 80 | 20480 / 1024 |

The weights are cold. Expect 0.5-1.5 µs per weighted norm, about 0.1-0.25 ms
per token (0.3-0.8%), plus the batched norms in verify. Splash's
14.8 → 8.2 µs was measured on 8-row, 256-thread norms; these single-row
1024-thread norms have less to hide.

**Why it is exact.** Only when a load is issued changes: same thread count
(`ds4_gpu_rms_norm_threads`, `ds4_metal.m:5776`; the partition is the
reduction tree), same elements per thread in the same order, verbatim
reduction loop, same `simd_sum`/shmem/`simd_sum` tree,
`mean = sumf/args.ne00` with a runtime ne00, `scale = 1.0f/sqrt(mean+eps)`,
and output `(x*scale)*w` per element.

**Code map.** `metal/norm.metal:19-79` `kernel_rms_norm_fuse_impl`
(reduction 48-62, output 67-78, instantiations 84-85), shared by every model
family. (`norm.metal:87-135` is `kernel_add_rms_norm_mul_f32_4`, GLM only.)
`metal/dsv41.metal:362-396` `kernel_dsv41_hc_norm` (HC collapse with four
volatile FMAs, BF16, weighted RMSNorm, BF16 store; weight read at 392). Host:
`ds4_gpu_rms_norm_weight_rows_tensor` (`ds4_metal.m:22675-22731`, pipeline at
22717), `ds4_gpu_rms_norm_plain_rows_tensor` (22471), `ds4_gpu_dsv41_hc_norm`
(51971-52008; returns 0 if `maxTotalThreadsPerThreadgroup < 1024`). Graph:
`ds41_norm` (`ds4.c:40644`), `ds41_hc_mix` (40669), `ds41_hc_collapse_norm`
(40950), `ds41_attention_project` (40792), `ds41_attention_publish` (40705),
`ds41_graph_logits_encode` (40933), `ds41_norm_batch` (40992) in verify.

**Implementation steps.**
1. `kernel_dsv41_hc_norm`: at entry preload
   `const float4 w0 = weight[tid], w1 = tid < 256u ? weight[tid + 1024u] : float4(0.0f);`
   and use `(values[i] * scale) * (i == tid ? w0 : w1)` at 392. Nothing else
   changes.
2. New `kernel_rms_norm_mul_f32_4_wpre` (template `MAXW = 2`): source
   identical to `kernel_rms_norm_fuse_impl<float4,2>` except that `float4
   wv[MAXW]` is loaded for i = tpitg.x + k·ntg.x < ne00_t before the reduction
   loop and used by the output loop. Separate host name; the original stays
   untouched.
3. In `ds4_gpu_rms_norm_weight_rows_tensor`, use the new pipeline (same
   grid, threads, smem) only when admitted via
   `ds4_gpu_dsv41_exact_admitted(DS4_GPU_TEST_V41_FUSIONS, "DS4_METAL_DISABLE_V41_NORM_PRELOAD")`,
   ceil(ne00_t/nth) ≤ MAXW, and `maxTotalThreadsPerThreadgroup ≥ nth`. Set a
   counter bit.
4. Optional, measured separately: a plain F = 1 variant that keeps x in
   registers (5 float4 per thread at 20480) to avoid the second read of x. It
   changes the reduction loop's shape: add `#pragma clang fp reassociate(off)`
   and accept only on an oracle pass.
5. Companion: the 86 weighted norms are each followed by
   `kernel_dsv41_bf16_linear` (`ds41_bf16`). A BF16-store norm variant using
   the `ds4_mv_bf16` expression removes 86 dispatches per token (~0.3%).

**Gating and rollback.** Steps 2-3: `DS4_METAL_DISABLE_V41_NORM_PRELOAD`.
Step 1: the existing `DS4_METAL_DISABLE_V41_HC_NORM` (reverts to the original
chain; compare step 1 against the baseline worktree).

**Tests.** New section in `tests/test_deepseek41_fusions.c`: n ∈ {128, 512,
1280, 5120, 8192}; rows {1, 2, 7}; eps {1e-20, 1e-6}; inputs random, all zero
(scale 1e10), 1e18 (overflow → Inf → scale 0), NaN, ±Inf, −0, subnormal;
weights with NaN, Inf, −0, subnormals. New versus rollback byte-equal
including guards; n = 16384 (4 per thread) must fall back. In the HC oracle
(~60-108) compute the reference chain with the rollback set and keep
`take_dispatches() == 2`, which proves the fused kernel still passes the
1024-thread check.

**Measurement and acceptance.** `--stage-timing` ("attention projections",
"hyper-connection mixes and norms") with and without the rollback; decode A/B
with `--candidate-env DS4_METAL_DISABLE_V41_NORM_PRELOAD` at 512 and 8192.
Keep only if decode improves ≥ +0.4% at both contexts (the earlier campaign
rejected +0.14-0.20% items). Standard capture identical.

**Pitfalls.** Never change the thread count. Keep `args.ne00` and `width`
runtime values (a constant invites a reciprocal multiply; see the comment at
`dsv41.metal:358-361`). Extra registers can push
`maxTotalThreadsPerThreadgroup` below 1024, which silently disables the HC
fusion (slower, not wrong). Never edit `kernel_rms_norm_fuse_impl` in place.

**Stop if.** Any bit differs, the fused HC path stops being selected, or
decode improves by less than 0.4%.

### 7. Remove V4.1 decode attention's separate reduce dispatch

**Premise correction.** V4.1 decode does not use `heads8_split` /
`split_reduce` (those serve V4 indexed decode and V4.1 batch prefill). The
V4.1 decode path (`ds4.c:40821`):
1. `ds4_gpu_attention_decode_heads_tensor` (`ds4_metal.m:31341`) calls
   `gathered_heads` (29690) when n_comp > 0, or `raw_heads` (28112) for layers
   0-1 and ratio-2 layers at pos 0.
2. F16 staging (comp is F32, so the fused stage kernel is skipped, 27914):
   ring copy + `copy_to_f16_1d`, optional pad.
3. `kernel_flash_attn_ext_vec_f16_dk512_dv512` (`flash_attn.metal:970/1391`)
   with nwg = 32, nsg = 1 (n_keys ≤ 640); grid (1, 64, 32) threadgroups of 32
   threads; partials to `g_flash_attn_tmp_buffer` (4 MiB + 16 KiB per layer).
4. `kernel_flash_attn_ext_vec_reduce` (1442), 64 × 1024 threads, calling the
   noinline `ds4_flash_attn_vec_reduce_row` (1404).

**Goal and expected gain.** Drop the reduce dispatch and, with item 3A, the
BF16 + RoPE dispatches: 120 dispatches per token plus 8 MiB of partial
traffic per layer. The V4 equivalent measured +0.99% decode on M3 Ultra
(commit e6a6ab5). Expect +0.5-1%.

**Recommended variant B: reuse the existing no-partials packed kernel.**
`ds4_flash_attn_vec_packed8_reduce_f16_512` (`flash_attn.metal:1457`)
time-slices all 32 splits through 8 simdgroups and reduces with the same
32-lane `simd_sum` topology; V4 uses it by default on M3 Ultra. V4.1 cannot
reach it today because `packed_shape` requires `comp_kv_f16` and V4's RoPE
fuse, although the kernel only reads the staged F16 K/V.
1. Add `kernel_dsv41_flash_attn_vec_packed32_epilogue_f16_dk512_dv512` in
   `dsv41.metal`: the V4 kernel's uniform guard minus the rope fields, the same
   body, `barrier(mem_device)`, then item 3A's BF16 + inverse RoPE on 256
   threads.
2. Select it in both encoders when the V4.1 epilogue is armed and admitted
   and `use_mask == 0`, n_head 64, head_dim 512, nsg 1, nwg 32, n_keys ≤ 1024,
   pipeline execution width 32, ≥ 256 threads, threadgroup memory ≥
   `packed_shared_bytes` (29818).
3. Keep staging and pad; set the `used` flag.
4. Exactness rests on the body's documented one-block-per-split property
   (≤ 1024 keys); the oracle is the proof.

**Variant A: last-arriver fold, only if B measures slower.**
1. Give `kernel_flash_attn_ext_vec` a `bool FOLD = false` template parameter
   with extra buffers (`counters[64]`, `heads`, epilogue args). The FOLD = false
   instantiation must produce identical code: diff the AIR
   (`xcrun metal -S -emit-llvm`) of old and new.
2. Store partials through `device coherent(device) float4 *` (MSL ≥ 3.2;
   guard `#if __METAL_VERSION__ >= 320` and refuse admission if the pipeline
   fails).
3. Arrival as Splash's `split_reduce.h`: `threadgroup_barrier(mem_device)`;
   thread 0 does a `thread_scope_device` seq_cst fence, `atomic_fetch_add` on
   `counter[iq2]`, another fence; barrier; return unless `old == 31`.
4. Make `ds4_flash_attn_vec_reduce_row` a noinline template over the partial
   pointer type (AIR diff shows the existing instantiation unchanged); call the
   coherent instantiation 32 times with virtual `sgitg` 0..31.
5. Barrier, the V4.1 epilogue, `atomic_store(counter, 0)`.
6. Counters: a 64 × u32 zeroed shared buffer allocated once, registered in
   `ds4_gpu_invalidate_completion_counters` (`ds4_metal.m:1456`; called on
   command-buffer error at 1513) so a failed buffer retires it.
7. Refuse the fold while `g_batch_encoder_concurrent` is set.

**Gating and rollback.** `DS4_METAL_DISABLE_V41_PACKED_ATTN` (B),
`DS4_METAL_DISABLE_V41_ATTN_FOLD` (A), both through the admission call.

**Tests.** In `tests/test_deepseek41_attention.c` compare vec + reduce + BF16
+ RoPE against the new kernel: n_keys 1..160 exhaustively and a sweep to 640,
with and without pad, both encoders; F16 overflow/NaN/±0/subnormal-half K/V;
sinks −FLT_MAX / +huge / NaN and an S == 0 row; heads output plus guards. For
A also ≥ 10⁴ repeated dispatches on the M3 Ultra and counter poisoning with
`ds4_gpu_test_invalidate_completion_counters` recovery (pattern at
`tests/test_deepseek41_topk.c:52-66`). Model level: captures, snapshots,
DSpark parity.

**Measurement and acceptance.** First `DS4_METAL_ENCODER_TIMELINE=/tmp/tl.txt`
for per-dispatch time of vec, reduce, BF16 and RoPE (diagnostic). Then decode
screens at 8K and 32,495. Keep with ≥ 0.3% repeatable.

**Pitfalls.** Variant A puts a whole row's reduce (64 KiB of partials) on one
simdgroup of the last arriver, which may cost more than the launch it saves.
The comment at `ds4_metal.m:29738` records unexplained downstream effects from
changing the attention grid.

**Stop if.** Neither variant gains ≥ 0.3%, or A's AIR diff shows the non-fold
kernel changed (that also changes V4 and needs V4 bitwise evidence plus user
approval).

### 8. Cheaper MXFP4 code lookups (experiment first)

**Goal.** The inner loop decodes each 4-bit code through a 16-entry LUT of
unscaled values; the E8M0 scale multiplies the finished 16-element partial
(`sum += e8m0(e) * ((acc.x+acc.y)+(acc.z+acc.w))`). Two cheaper decodes may
cut ALU work on the MXFP4 decode path. Expected small (+0.3-1%), and only
after item 4.

**Option A (pair LUT).** A 256-entry table
`pairs[b] = float2(values[b & 15], values[b >> 4])`, one lookup per byte.
Values are identical to two single lookups, so exactness is immediate.

**Option B (code bits).** Build the half directly:
`as_type<half>(((c & 8) << 12) | ((c & 7) << 9))`. Codes 1 and 9 give half
subnormals ±2⁻¹⁵, so the scale is off by 2¹⁴ and y must be prescaled by 2¹⁴,
with a guard that no |y| overflows (`(as_type<uint4>(y) & 0x7fffffff) < 0x78800000u`,
else fall back). The required float values of the eight magnitudes are
0, 0x38000000, 0x38800000, 0x38C00000, 0x39000000, 0x39400000, 0x39800000,
0x39C00000 and their signed forms. Half-subnormal to float conversion on the
device is unproven: the microbenchmark must check all 16 codes bitwise.

**Code map.** LUT definitions and pair walks at `metal/moe.metal:4516`,
4540-4556, 4695, 4721-4737; down walks 6341-6348, 6387-6394, 6573-6580.

**Process.**
1. Experiment first under `experiments/` (not in `metal/`):
   `experiments/mxfp4_decode_lookup.c` plus a kernel file comparing the
   current walk, A and B on V4.1's shapes with cold weights.
2. Accept an option only if it cuts kernel GPU time by ≥ 3% and is bitwise
   identical on every code, scale and exceptional activation.
3. Productize only inside item 4's static kernels through one shared helper
   `ds4_mxfp4_block_dot<P>`; do not touch V4's kernels.

**Gating and rollback.** `DS4_METAL_DISABLE_V41_MXFP4_PAIR_LUT` /
`DS4_METAL_DISABLE_V41_MXFP4_CODE_BITS`, under item 4's admission.

**Tests.** Item 4's test gains arms for each option; all 16 codes × E8M0
{0, 1, 127, 254, 0xff} × activations {±0, subnormal, ±Inf, NaN, max finite}.

**Measurement and acceptance.** Decode ≥ +0.3% in two screens on the MXFP4
file.

**Stop if.** The microbenchmark shows < 3%, or option B's device conversion
differs on any code.

### 15. Exact verify rows: rewrite `kernel_mul_mv_q8_0_f32_rows`, and batch the head, o_a/o_b and router

**Goal and expected gain.** The rows kernel takes q_b (40 layers) from 2.86 ms
for 1 row to 9.63 ms for 6 rows: 1.35 ms per extra row, about 11× the
0.12 ms FP32 FMA floor. Causes visible in `metal/dense.metal:221-281`: the
runtime `rows` bound makes `sumf[MAX_ROWS][NR0]` dynamically indexed (so it
lives in private memory); qs and d are reloaded and int8→float reconverted for
every input row; there are `rows` separate two-barrier reductions; activation
L1/L2 traffic is R·128/(34·NR0) ≈ 11× the weight bytes at R = 6. Verify
(`ds41_graph_step_batch_spec`, `ds4.c:42418`, via `ds41_graph_verify` at
42592) also runs o_a and o_b once per row (42505-42506), the head once per row
(42536-42547) and the router once per row (40541-40551). Rough 6-row budget
out of 169.69 ms: rows-kernel dense work 25 ms → ~12; per-row o_a+o_b 31 ms →
~8-10; head 6 ms → ~1.6; router dispatches ~0.5-2 ms. Up to ~40 ms, all
exact. Kernel target: q_b at 6 rows ≤ 4.6 ms (≤ 0.35 ms per extra row).
Production verify is at most 6 rows (the draft block size is 5).

**Why it is exact.** For each (input row r, output row) each lane walks the
same ib sequence with the same NSG, computing `sumq` = Σ_{i=0..7} q_i·y_{r,i}
from 0.f, then `sumf += sumq*d` in ib order. Converting int8→float once rather
than per use yields the same float; the chains are independent accumulators,
so loop order and load placement do not matter. Reduction: the same
first-stage `simd_sum` on the same lane values; lane 0 of each simdgroup
stores its partial; after one barrier the reducer reads
`v = tiisg < NSG ? part[tiisg] : 0.0f` (lanes < NSG hold exactly the values
the original shmem slots held, in the same lanes; lanes ≥ NSG hold +0.0 as the
original zero fill did), then `simd_sum(v)`. `simd_sum` is a function of its
32 lane inputs, so which simdgroup runs the second stage is irrelevant. The
oracle proves it for the compiled library.

**Code map.** Only dispatch site of the rows kernel:
`ds4_gpu_matmul_q8_0_decode_rows_exact_tensor` (`ds4_metal.m:19997-20087`);
gate at 20053-20060 (rows 2..8 and
`ds4_gpu_dsv41_exact_admitted(DS4_GPU_TEST_V41_SHARED_ROWS, "DS4_METAL_DISABLE_V41_SHARED_ROWS")`);
fallback is the one-token kernel with grid.y = rows (also exact); nsg = 8
above 65536 (20042). Callers: the `ds41_matmul_batch` Q8 branch
(`ds4.c:40531-40537`, which excludes `DS4_N_VOCAB` on Apple), reached from
`ds41_attention_project_batch` (41052), `ds41_moe_batch` (41351), the drafter
(42669-42733), `ds41_decoder_prepare` (41945) and prefill o_b (42282). Router:
F32 one-token `kernel_mul_mv_f32_f32_4`, nsg 8, NR0 2
(`ds4_gpu_matmul_f32_tensor`, 22218; one-token branch 22266). Precedent: the
F16 `exact_rows` branch (21376). (`ds4_gpu_dsv41_projection_rows` at 21529 is
the F16 per-row path, not a site of this kernel.)

**Implementation steps.**
1. Add `template<short R, short NR0> kernel_mul_mv_q8_0_f32_rows_t`,
   instantiated `kernel_mul_mv_q8_0_f32_rows{2..8}` with NR0 = `N_R0_Q8_0`
   (NR0 = 4 twins only if item 2's sweep favours them). Delete the old kernel
   once the oracle passes.
2. For each ib and output row, load `half d` and eight `int8_t` once (byte
   loads: `qs + 8·il` is only 2-byte aligned) and convert to float once. Load
   y for all R rows up front (float4 loads are value-identical) when
   R·8 + R·NR0 ≤ ~96 registers, otherwise stream y per r. Keep the expressions
   unchanged, fully unrolled, no dynamic indices.
3. Do all simd_sums; store `part[(r·NR0 + row)·NSG + sgitg]`; one barrier;
   simdgroup (o mod NSG) reduces output o and writes `dst[r·ne0 + r0 + row]` if
   `r0 + row < ne01`. smem = R·NR0·NSG·4 bytes rounded to 16 (≤ 512 B).
4. Host (20053-20083): `snprintf` the name from n_rows, set smem, keep the grid
   and the fallback.
5. Head: add `ds41_output_projection_rows_exact` to `ds4.c`, slicing the head
   exactly as `ds41_output_projection` (40575) does, then one call to
   `ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(spec->logits, …, 5120, vocab, active.norm, rows)`
   (`rows_view[i].norm` are contiguous rows of `batch.norm`, 40483-40486). In
   the spec loop compute each row's x and norm first, then make one call.
   Apple only, Q8_0 head, no `tp_logits_half`. Keep `ds41_matmul_batch`'s
   vocabulary exclusion: the batch and drafter heads use `mul_mv_ext` and must
   keep their bits.
6. o_b in verify (tp_world == 1): after the per-row attention and
   `ds41_attention_low`, call `ds41_matmul_batch(active.block, o_b, active.low, rows, false)`
   (the one-token call is `ds41_matmul` with round = false, the plain nsg-4
   kernel, so this is exact). Then, as a separately measured step, o_a: 8
   group dispatches of the rows kernel with offsets g·1024·4352 (weights),
   g·4096 floats (x), g·1024 floats (out); nb11 = 32768·4; ne0 = 8192;
   ne01 = 1024; nsg 4 fixed as at 27324; then
   `ds4_gpu_dsv41_quantize(low, 8192, rows, BF16)`.
7. Router: add `exact_rows` to the F32 path mirroring F16 (one
   `kernel_mul_mv_f32_f32_4` dispatch with grid.y = rows and
   `ds4_gpu_make_f32_mv_args(in, out, rows)`), replacing 40541-40551. A
   shared-weight F32 rows kernel only if it measures.

**Gating and rollback.** The rows kernel keeps
`DS4_GPU_TEST_V41_SHARED_ROWS` / `DS4_METAL_DISABLE_V41_SHARED_ROWS`. Add
`DS4_METAL_DISABLE_V41_VERIFY_HEAD_ROWS`, `DS4_METAL_DISABLE_V41_VERIFY_ATTN_OUT_ROWS`
and `DS4_METAL_DISABLE_V41_VERIFY_ROUTER_ROWS`, each restoring its per-row
loop, all through `ds4_gpu_dsv41_exact_admitted`. TP keeps the current paths.

**Tests.** Extend the shared-rows block
(`tests/test_deepseek41_fusions.c:213-279`): every R = 2..8; add 1280→32768,
4096→1024, 512→65544 (nsg 8) and one {5120, 129280} head case if memory
allows (enlarge the buffer as in item 2); `DS4_METAL_Q8_MV_NSG` = 1, 2, 4, 8;
exceptional values in every row position; keep per-row equality against the
one-token kernel. Add o_a rows versus per-row, and F32 router rows versus
per-row with NaN, ±Inf, −0. Model check:
`make tests/test_deepseek41_dspark` then
`./tests/test_deepseek41_dspark gguf/DeepSeek-V4.1-Flash-MXFP4.gguf --verify-parity speed-bench/promessi_sposi.txt`
(every logit and the snapshot after every kept prefix).

**Measurement and acceptance.** `--stage-timing`: the q_b rows table must
reach ≤ 5.5 ms at 6 rows. `--verify-timing` per step; keep a step if it saves
≥ 2% of verify time and parity holds. Final: DSpark with 400 greedy tokens on
both prompts, identical output, record t/s. Standard capture unchanged
(decode does not use these paths).

**Pitfalls.** Register spill at R = 8: time it. The head path must never reach
`mul_mv_ext`. o_a needs nsg 4 even when `DS4_METAL_Q8_MV_NSG` is set. The old
kernel races benignly on shmem (sgitg 0 zeroes slots for r+1 while other
simdgroups may still read r; their results are discarded); the rewrite removes
it, so do not copy it.

**Stop if.** Any bit differs, 6-row q_b is not under 7 ms, or
`--verify-parity` fails.

## B. V4.1 prefill and long-context decode

### 9. MXFP4 prefill switches below 2,048 rows

**Goal and expected gain.** Five MXFP4 prefill switches in
`ds4_metal.m:44318-44424` are gated on `n_tokens >= 2048u`: compact tile
(44343), map scatter (44359), pair tail cull (44380), down tail cull (44392)
and down half-LUT (44409). `ds4.c:42039-42044` caps chunks at 2,048 for
prompts under 8,192 tokens, so short and mid prompts, and the last chunk of
every prompt, never get them. Expect several percent of prefill at 128-1,536
rows on the MXFP4 file.

**Why it is exact.** Row count does not enter the per-output arithmetic of
any of these kernels. `tests/test_mxfp4_metal.c:444-720` already proves each
switch bitwise at 48 rows. `use_mm_id` itself requires ≥ 32 rows, so that is
the floor.

**Implementation steps.**
1. Add `v41_mxfp4_small_prefill` =
   `ds4_gpu_dsv41_exact_admitted(DS4_GPU_TEST_V41_MXFP4_PREFILL, "DS4_METAL_DISABLE_V41_MXFP4_SMALL_PREFILL")`
   and the V4.1 expert shape (as in item 4).
2. Give each switch its own minimum:
   `DS41_MXFP4_<SWITCH>_MIN_ROWS` constants (compile-time, chosen from the
   measurements below). When the V4.1 predicate is true, the gate becomes
   `n_tokens >= DS41_..._MIN_ROWS`; V4 keeps 2,048.
3. Keep the other conditions of each gate unchanged.

**Measurement (to choose the minimums).** For each switch, prefill bench on
the MXFP4 file at chunks 128, 512, 1,024 and 1,536 with `--repeats 4`, arms
isolating one switch by passing the others' disables in both arms:
`--control-env` / `--extra-env` over
`DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_{PAIR_SWIGLU_COMPACT_TILE,MAP_SCATTER,PAIR_TAIL_SIMDGROUP_CULL,DOWN_TAIL_SIMDGROUP_CULL,DOWN_HALF_LUT}`.
The minimum is the smallest size where the switch gains ≥ +0.5% in both
halves of the ABBA run.

**Tests.** New `tests/test_deepseek41_mxfp4_prefill.c` (default exact tests):
sizes 32, 33, 47-49, 63-65, 127-129, 511-513; each switch on versus off,
bitwise, V4.1 shape under the test flag; admission refusals.

**Acceptance.** Keep at ≥ +2% prefill at 512 and ≥ +1% at 1,536. Re-run
after item 10, whose cull changes the tail cost.

**Stop if.** No switch clears +0.5% below 2,048.

### 10. Eight-row tail cull in the mm_id pair, compact and down kernels

**Goal.** The simdgroup-matrix mm_id kernels compute 16-row halves; a tail
of ≤ 8 live rows still loads `mb[1]` and runs the `mc[4..7]` MMAs. Culling
the upper 8 when they are dead saves work on every expert's last tile.
Expect +1-2% prefill at 512 rows, less at 2,048.

**Why it is exact.** The skipped MMAs write only rows that are never stored;
live rows' accumulators are untouched and their MMA sequence is unchanged.

**Implementation steps.**
1. Add `bool CULL_TAIL_8 = false` to the pair (`metal/moe.metal:8426-8430`,
   body 8507-8531, store 8547-8552), compact (8640, 8732-8755, 8771-8776) and
   down (7970-7974, 8102-8124, 8132-8136) templates.
2. Compute `half_rows = nr1 - 16*(sgitg/2)` (compact: `sgitg`) and
   `mma_hi = !CULL_TAIL_8 || half_rows > 8`; guard the `mb[1]` load, the
   `mc[4..7]` MMAs and their store with `mma_hi`. `mma_hi` must be uniform
   across the simdgroup.
3. Instantiate `..._tail_cull8` for Q4_K and MXFP4; the default
   instantiations stay byte-identical (AIR diff).
4. Select them on V4.1 when admitted.

**Gating and rollback.** `DS4_GPU_TEST_V41_TAIL_CULL8`,
`DS4_METAL_DISABLE_V41_TAIL_CULL8`.

**Tests.** Sizes 40, 41, 56, 57 (and item 9's list) per expert, three passes
(original, cull16, cull8) bitwise, Q4 and MXFP4.

**Acceptance.** ≥ +1% at 512 and no size worse than −0.2%.

### 11. Half input to the mm_id pair kernels

**Goal.** The pair kernels convert F32 activations to half while staging
(`metal/moe.metal:8488`; compact 8711/8713), once per expert tile. A single
pre-pass converting the whole input once removes the repeated conversion and
halves activation reads.

**Why it is exact.** The pre-pass must use the same conversion as staging
(round-to-nearest-even F32→F16, same handling of overflow, NaN and
subnormals). Prove it with an exhaustive 2³² oracle: every F32 bit pattern
through the pre-pass kernel versus the staging conversion.

**Implementation steps.** Add a `TY` input template parameter and `_h`
instantiations; pre-pass with `ds4_gpu_encode_cpy_f32_f16_1d`
(`ds4_metal.m:27565`) into a new `g_moe_half_input_buffer`; set
`src1_elem_size = 2`.

**Gating and rollback.** `DS4_METAL_DISABLE_V41_PAIR_HALF_INPUT` under the
admission call.

**Order and acceptance.** After items 9 and 10. Keep at ≥ +0.5% at two
sizes.

### 12. Zero-code A/B: split mm_id versus fused pair on V4.1

**Goal.** Check whether the older split path (`kernel_mul_mm_id_{q4_K,mxfp4}_f32`
plus `kernel_dsv4_moe_swiglu_weight_f16`, `metal/moe.metal:518-550`) beats the
fused pair on V4.1 at some size. No code change.

**Confound.** `DS4_METAL_DISABLE_MOE_MM_ID_PAIR_SWIGLU` (44319) also disables
the Q4 down cull and the MXFP4 compact and tail culls, so a raw A/B mixes
several changes. Isolate: `--control-env DS4_METAL_DISABLE_V41_Q4_TAIL_CULL=1 --extra-env DS4_METAL_DISABLE_V41_Q4_TAIL_CULL=1`
(and the MXFP4 equivalents) so only the pair differs.

**Command loop.** Prefill bench at n = 512, 2,048 and 8,192 with
`--warmup-tokens 32 --repeats 4`, both GGUFs.

**Outcome.** If the split path is ≥ +1% at some size, open a follow-up item
with a row-count switch (exactness proven by the same kind of oracle);
otherwise record the numbers under `speed-bench/ds41f-m3ultra-perf/` and
close.

### 13. Long-context indexer: tiled scorer and top-k

**Premise corrections.** A survivor-threshold top-k in (value desc, id asc)
order does not reproduce the existing chain in general: the 1024-wide bitonic
block sort (`argsort.metal:52`) is unstable, the merge takes left on `>=`
(166), −0 equals +0, and NaN breaks the order. It is exact only on the
accepted class (top-512 finite, normal, pairwise distinct and strictly above
the 513th) with fallback otherwise, and the existing histogram path
(`argsort.metal:406-650`) already is that selector. "Full sort" above 32,768
is really per-block top-512 plus non-prefix merge-path merges over the whole
`work_width`. Masked layers (24/28/32/36, −Inf masks at `ds4.c:40757-40762`)
never use the histogram. At 65K only layer 20 newly exceeds 32,768 (layers
2/8/14 have `n_comp = (pos + 1)/2`). Full-model runs have never allocated or
advanced beyond 32,768 tokens: any 65K/130K model run needs the user's
approval.

**Goal and expected gain.** Scorer: `kernel_glm_indexer_score_one_direct`
(`dsv4_misc.metal:1978`; dispatch `ds4_metal.m:36400`; V4.1 call
`ds4.c:40776`) runs one 128-thread threadgroup per row, re-reading the 16 KB
q vector with 16 barriers per row, ~3C/2 + 5C rows per token at context C.
Top-k: the chain's merges dominate. Review estimate 0.5-1 ms per token at
130K and nothing at ≤ 8K; unverified.

**Measure first (model-free, no allocation issue).** A small harness around
`ds4_gpu_glm_indexer_score_one_tensor` and
`ds4_gpu_indexer_topk_tensor`/`ds4_gpu_dsv41_indexer_topk` at widths 16K, 32K,
65,536, 131,072, unmasked and masked like `kernel_dsv41_candidate_filter`
(≤ 16,384 finite, rest −Inf); record the histogram accept/overflow rate from
`ds4_gpu_test_glm53_topk_stats`. Compute ms per token at 65K and 130K from the
call mix. If under 0.3 ms at 130K, stop.

**Why the tiled scorer is exact.** One threadgroup (4 simdgroups) scores R
rows (16 or 32; R does not affect arithmetic). Simdgroup s keeps `q4[lane]`
and `weights[h]` for h = s, s+4, …, s+28 in registers. Per row:
`k = keys4[row*32 + lane]`; per head `d = simd_sum(dot(q_h[lane], k))`; lane 0
stores `max(d*scale, 0.0f)*w[h]` into `threadgroup volatile float psum[R][32]`.
After the barrier, thread r < R computes `acc = 0.0f; for h in 0..31: acc += psum[r][h]`:
the original's head order and the same materialized product (no FMA into
acc), same lanes, same `dot` operands, same `simd_sum`, runtime `scale`.
Guard rows beyond n_rows.

**Top-k exact options.** (a) Non-causal prefix merge: add
`kernel_argsort_merge_f32_i32<DESC, false, true>` for V4.1 decode calls with
top_k ≤ 512, computing `chunk` from the untruncated `len0 + len1` so each
thread's `k0` and merge-path partition equal the original's for every
k < 512 (exact even with NaN); merge grids unchanged so the histogram's
fallback `fb_grid` stays valid. (b) Raise the V4.1 histogram bound from 32,768
(`ds4_metal.m:19369`) toward 131,072 (the `fb_count ≤ 8` limit, 19380-19388)
only if the measured accept rate justifies it. (c) Admit masked rows: use
`args.pad0` as a flag so V4.1 alone treats exactly 0xff800000 as not
NONFINITE (TIE/SHORT already reject any −Inf reaching rank 511/512), then
call `ds4_gpu_dsv41_indexer_topk` for il > 20 too.

**Code map.** Top-k implementation `ds4_metal.m:19281-19523`;
`ds4_gpu_dsv41_indexer_topk` 19544; pick site `ds4.c:40750-40763`. The GLM
callers of the scorer (`ds4.c:55480`, 57440) stay on the old kernel.

**Implementation steps.** (1) `ds4_gpu_dsv41_indexer_score_one` (Apple): the
tiled kernel when admitted, else the old function; call it from 40776.
(2) The prefix merge for policy-41 calls and a masked-policy entry for
il > 20. (3) Options (b) and (c) only after the measurements.

**Gating and rollback.** Admission call; env vars
`DS4_METAL_DISABLE_V41_INDEX_SCORE_TILE`, `_TOPK_DECODE_PREFIX`,
`_TOPK_MASKED`, plus the existing `_TOPK_FAST`. Admit the tiled scorer and
prefix merge at all widths so the ≤ 32K model sweeps exercise the same kernels.

**Tests.** New `tests/test_deepseek41_indexer.c` (in the default exact
tests): n_rows 1/31/32/33/4095/65,536/131,072 (64 MB of keys);
NaN/Inf/±0/subnormal q and keys, negative and zero weights, Inf·0; bitwise
scores plus guards. Extend `tests/test_deepseek41_topk.c`: widths
65,536/98,304/131,072 and 131,073 (host refusal); masked kinds with exactly
511/512/513 finite values, a tie at the cut against −Inf, block-structured
masks; NaN/−0 kinds for the prefix merge against the non-prefix chain. Beyond
32K at model level: replay real score rows captured at ≤ 32K, tiled to
65K/131K.

**Measurement and acceptance.** Model-free timings first; then ≤ 32K
interleaved decode screens, which must be non-negative; long-context
throughput only with approval.

**Stop if.** The model-free estimate is under 0.3 ms per token at 130K; the
histogram accept rate at 65K-131K is under ~80%; or the user does not approve
contexts above 32,768.

## C. GLM 5.3 (KDA)

Do audit fix G1 (GLM's HC handoff, section G) before either item here: both
touch GLM decode and should be measured against the corrected baseline.

### 14a. KDA decode: prefetch, then split the recurrence across threadgroups

**Goal and expected gain.** `kernel_glm53_kda_decode_values<VALUES>`
(`metal/glm53_kda.metal:14-196`) runs one 128-thread threadgroup per head
(grid (n_rows, n_heads) × 128; callers `ds4.c:49319` and 79063 always pass
n_rows = 1). With 64 heads, most of the 80-core M3 Ultra is idle during the
recurrence. Step A overlaps state loads with stores; step B splits each head
over 2 or 4 threadgroups with a last-arriver finish, the pattern Splash uses
for split-K. Expect +0.3-1% decode, only if the kernel is a real share of the
token.

**Code map.** Kernel: conv shift 82-90; VALUES > 1 loop 155-177; output norm
181-195; reference kernel 210-356. Host `ds4_gpu_glm53_kda_decode`
(`ds4_metal.m:48726`): values4 predicate 48801-48806, threadgroup memory 656
floats (48856), grid (48857). GLM coverage enum `ds4_gpu.h:280-291`.

**Why it is exact.** Step A only moves loads earlier; each value is read
from the same address before any store that could alias it (check with the
kernel's own buffer layout, and refuse if `ds4_gpu_buffer_ranges_overlap`
says the state in and out overlap). Step B keeps each output element's
arithmetic in one thread in the same order; the split only distributes
which threadgroup computes which values, and the finishing phase (conv shift,
output norm) runs once, in the last arriver, on data published before its
arrival.

**Implementation steps.**
1. Step A: `kernel_glm53_kda_decode_v4p`, a copy of the VALUES = 4 kernel that
   prefetches the next rows' raw state into registers before the current
   stores. Measure alone first.
2. Step B: `kernel_glm53_kda_decode_split<SPLIT ∈ {2, 4}>`, grid
   (SPLIT, n_heads) × 128.
   - New buffers: `device coherent(device) float *partials [[buffer(16)]]`
     (64 × 128 floats) and `device atomic_uint *arrivals [[buffer(17)]]`
     (64 counters, zeroed once, registered in
     `ds4_gpu_invalidate_completion_counters`).
   - Phases 1-2 unchanged, but each threadgroup skips the conv shift at 82-90.
   - Arrival as Splash's `split_reduce.h`: publish partials;
     `threadgroup_barrier(mem_device)`; thread 0 does a device-scope fence,
     `atomic_fetch_add` on `arrivals[head]` and another fence; the ticket goes
     to all threads through threadgroup memory (660 floats total); all but the
     last arriver return.
   - The last arriver resets its counter, loads `so[tid] = partials[…]`, does
     the conv shift and phase 4 (output norm).
   - If `coherent(device)` is unavailable or fails the stress test (section
     G), fall back to publishing through atomics (pattern A) or to two
     dispatches (pattern C).
3. Env vars: `DS4_METAL_DISABLE_GLM53_DECODE_KDA_PREFETCH`,
   `DS4_METAL_GLM53_DECODE_KDA_SPLIT=1|2|4` (sweep),
   `DS4_METAL_DISABLE_GLM53_DECODE_KDA_SPLIT`. Admit only with the values4
   predicate, `!g_batch_encoder_concurrent` and no reference-kernel env.
4. Coverage bits `DS4_GPU_GLM53_DECODE_KDA_PREFETCH 1u<<11` and
   `DS4_GPU_GLM53_DECODE_KDA_SPLIT 1u<<12`.

**Tests.** Add arms to `check_glm53_kda_decode_values4`
(`tests/test_glm53_kda.c:1501-1582`): original, prefetch, split2, split4,
bitwise on outputs and state. A 4,096-step soak with NaN-poisoned partials
between steps (the split must never read stale data), and a check that every
arrival counter is zero after each dispatch. Model level: the GLM greedy
byte compare and capture from the preamble.

**Measurement and acceptance.** First `DS4_METAL_ENCODER_TIMELINE` to get the
kernel's per-token time. Then GLM decode screens with each env. Keep at
≥ +0.3%.

**Stop if.** The kernel is under 0.7 ms per token (not worth it), or split4
cuts the kernel's time by less than 20%.

### 14b. KDA prefill: stage the recurrence inputs in threadgroup memory

**Goal.** `kernel_glm53_kda_prefill_recurrence_values<VALUES>`
(`metal/glm53_kda.metal:663-708`; host 48902, selection 48987-48993, dispatch
49141-49161, grid (n_heads, 32/values) × 128) reads q, k, decay, v and beta
from device memory at every step. Staging blocks of steps in threadgroup
memory and prefetching the next block into registers should cut its time.
Expect +0.5-1% GLM prefill.

**Why it is exact.** Only where inputs are read from changes; every thread
performs the same operations in the same order on the same values. Inputs are
distinct allocations (`ds4.c:48478-48486`); assert no overlap with
`ds4_gpu_buffer_ranges_overlap` (`ds4_metal.m:24710`) and refuse otherwise.

**Implementation steps.** New
`kernel_glm53_kda_prefill_recurrence_staged<VALUES = 2, SGS, BLOCK>`: stage a
BLOCK of steps of q/k/decay/v/beta per threadgroup in threadgroup memory
(sizes: SGS 16 / BLOCK 8 ≈ 13.3 KB; SGS 32 / BLOCK 8 ≈ 14.4 KB; SGS 8 /
BLOCK 16 ≈ 25.7 KB), prefetch the next block into registers while computing
the current one; handle a final partial block. Profile with
`DS4_METAL_PROFILE_KDA_PREFILL`.

**Gating and rollback.** `DS4_METAL_DISABLE_GLM53_PREFILL_KDA_STAGED`; sweep
envs `DS4_METAL_GLM53_PREFILL_KDA_STAGED_SGS` / `_BLOCK` (diagnostic only;
ship one fixed choice); coverage bit `1u<<13`.

**Tests.** Arms in `check_glm53_kda_prepare_blocked`
(`tests/test_glm53_kda.c:1195-1349`) for each (SGS, BLOCK), plus a 47-token
case (partial block), bitwise against the current kernel.

**Acceptance.** ≥ +0.5% GLM prefill and ≥ 15% cut in the kernel's own time.
Do not stack VALUES = 4 on top: it measured −0.55% before.

## D. DSpark: an exact route to beating plain decode

Today verifying 2, 4, 6 or 8 rows costs 2.08, 3.57, 5.13 or 6.63 decode steps
(33.1 ms each), so DSpark loses to plain decode (29.3 vs 30.3 t/s on code,
24.6 vs 30.2 on prose) despite 80-86% acceptance. To win by 10%, code needs a
6-row verify at about 4.3 steps; prose needs each extra row to cost at most
~0.4 step. Every item here keeps outputs bit-identical: verification stays
exact; drafter-side changes can move acceptance and speed, never output.
Item 15 (section A) belongs to this route too.

**Premise corrections (checked in code and the support GGUF header).**
`dspark.block_size = 5`, `markov_rank = 256`, `target_layer_ids = [37, 38, 39]`,
3 stages, 128 experts with 3 used. `ds41_draft_block` drafts at most 5
tokens, so a production verify is at most 6 rows; the 8-row figure comes
only from `--verify-timing` (`DS4_TP_BATCH_MAX_ROWS = 8`, `ds4_tp.h:34`, is a
cap). The drafter's head is not per row: `ds41_draft_block` calls
`ds41_output_projection(g, logits, m, w, active.norm, rows)` once
(`ds4.c:42785`), which goes to `mul_mv_ext`; the head is per row only in
verify (42534-42546). `ds4_gpu_argmax_tensor` (`ds4_metal.m:19549`) does not
reproduce `sample_argmax` (bitonic top-1 with unordered ties, no −1e30 floor,
fast-math). `--verify-timing` includes the serial Engram reads (~5.3 ms per 48
cold, ~0.3 ms warm), so measure c(k) warm and cold. After the draft sync a
cycle waits four more times (layer-13 drain, end of verify, commit buffer,
seed buffer). Confidences are computed on the CPU in double after the draft
sync (42815-42825).

### Order and measurement plan

**Step 0: attribute c(k) and the draft cost D before changing behaviour**
(diagnostics only; the default path does not change).
1. `tests/test_deepseek41_dspark.c` `time_verify` (~186-235): run rows 1-6
   (keep 8 for continuity); per k print hash + serial Engram time, encode time
   before the first commit, each command buffer's GPU span
   (`DS4_METAL_CB_TIMES=1`; `ds4_gpu_finish_command_buffer`,
   `ds4_metal.m:1704`), commit and seed buffer times, CPU `sample_argmax`
   time. Once warm, once cold; record which.
2. `time_stages` (245-335): add a ROWS argument; for k = 1..6 time the batch
   forms verify runs (`ds41_attention_project_batch`; the per-row
   `ds41_attention`/`ds41_attention_output` loops; `ds41_moe_batch`;
   `ds41_before_attention_batch`/`ds41_after_attention_batch`;
   `ds41_spec_layer_begin`, `ds41_spec_layer_pairs`, `ds41_spec_capture`; k
   per-row heads versus one rows-exact head). Print ms and marginal ms per row.
3. New `--draft-timing`: split `ds41_draft_block` into embed + 3 stages, head,
   Markov chain and host confidence, with end/begin boundaries as the
   `DS41_STAGE` macro does (42204-42213).
4. Fill the V4.1 stats fields that print zero today in
   `ds4_session_print_dspark_stats` (74122): `verify_upload` (hash + Engram),
   `verify_layer`, `verify_head`, `verify_read` (argmax), `snapshot` (commit +
   seed), `prop_logits`, `prop_markov`, `prop_confidence`.
5. The expert-overlap diagnostic (item 21).

Deliverable: a per-stage c(k) table (warm and cold) plus D per stage. It
decides which items are worth doing and seeds item 18's initial costs.

**Order.** 18 (bounds the downside and measures online) → 16 (probably the
largest per-row saving, especially cold) → 15 (section A: rows kernel, head,
o_a/o_b, router) → 17 → 19 (D matters most for prose) → 20 → 21 (only if
expert overlap is ≥ 15-20%).

**After every item:** `--verify-parity` on the full case list; greedy
byte-compare of DSpark against plain decode; `make test` on Darwin; plain
decode speed unchanged; DSpark A/B in ABBA order, 3 runs, Engram state
recorded.

**Commands** (M = `gguf/DeepSeek-V4.1-Flash-MXFP4.gguf`,
S = `gguf/DeepSeek-V4.1-Flash-DSpark-support.gguf`):

```sh
make ds4 ds4-bench tests/test_deepseek41_dspark   # explicit target, not in make test
./tests/test_deepseek41_dspark $M --verify-parity speed-bench/promessi_sposi.txt   # needs >1100 prompt tokens
DS4_TEST_VERIFY_CASE=250,6 ./tests/test_deepseek41_dspark $M --verify-parity speed-bench/promessi_sposi.txt
DS4_METAL_CB_TIMES=1 ./tests/test_deepseek41_dspark $M --verify-timing speed-bench/promessi_sposi.txt
./tests/test_deepseek41_dspark $M --stage-timing speed-bench/promessi_sposi.txt
# greedy byte compare + throughput: save the two README prompts (C hash table,
# prose explanation) as files and record their sha256
for P in code.txt prose.txt; do
  ./ds4 -m $M -c 4096 -n 400 --temp 0 --nothink --prompt-file $P >plain.out 2>plain.err
  DS4_DSPARK_STATS=1 ./ds4 -m $M -c 4096 -n 400 --temp 0 --nothink --prompt-file $P \
      --dspark --mtp-model $S >dspark.out 2>dspark.err
  cmp plain.out dspark.out && grep -h 'generation:' plain.err dspark.err && grep -h 'DSpark stats' dspark.err
done
DS4_DSPARK_FIXTURE_REQUIRE_IDENTICAL=1 DS4_DSPARK_FIXTURE_TOKENS=128 \
  make dspark-acceptance DS4_DSPARK_MODEL=$M DS4_DSPARK_SUPPORT=$S
DS4_BENCH_SPEC_TRACE=1 ./ds4-bench -m $M --dspark --mtp-model $S --prompt-file $P \
  --ctx-start 2048 --ctx-max 2048 --gen-tokens 400   # average cycle ms by tokens per cycle
```

`DS4_DSPARK_STATS` is read by `ds4_dspark_stats_enabled` (`ds4.c:74090`); the
stats line prints at session free. The acceptance fixture defaults to V4
Flash, so always pass the V4.1 model and support file. Warm the Engram
tables first for warm runs.

### 16. Verify with the decode host pipeline

**Goal and expected gain.** `ds41_graph_verify` hashes and reads 48 Engram
rows per verify row one after another before anything is committed, encodes
layers 0-13 with the GPU idle, drains at layer 13, then encodes layers 14-39
and the heads with the GPU idle again. Decode removed exactly these gaps
(+12.95% cold from trial E, +6.7% from P, +0.6% from O). Expected saving per
verified row: the serial Engram time (cold ≈ 5.3 ms, ~0.9 step for 6 rows;
warm ≈ 0.3 ms) plus the part of host encode time the GPU waits for. Step 0
gives the real split.

**Why output stays exact.** The same bytes are read and each row is decoded
by the same `ds4_engram_read` inside the step readers; the dispatches are the
same and in the same order; only command-buffer boundaries move, and table 1's
input gets separate storage (same values at a different address). Decode
proved this bit-identical over 58.8M logits.

**Code map.** Decode `ds41_graph_step` (`ds4.c:41555`): `queue_layers` 41576;
`pipeline_layers` 41589 (`DS4_METAL_DISABLE_V41_DECODE_PIPELINE`); readers
41600-41622 (`DS4_DISABLE_V41_ENGRAM_STEP_READERS`,
`DS4_DISABLE_V41_ENGRAM_STEP_OVERLAP`); join 41641; late input 41646; flush
after layers 0 and 4 at 41667; drain rule 41669; join on failure 41680.
Readers: `ds4_engram_read_step_begin/end` (`ds4_engram.h`;
`ds4_engram.c:248-280`). Verify: `ds41_graph_verify` (42592: begin → batch →
end; on failure `ds4_gpu_synchronize()` and `g->valid = false`);
`ds41_graph_step_batch_spec` (42418): host `engram` array 42449; loop-local
`ids` and serial reads 42466-42481; Engram uploads 42485-42492;
`ds41_spec_layer_begin` 42494; `il == 13` split 42521; per-row heads
42534-42546. Engram layers 1 and 14 (`ds41_engram_layer`, 1392).
`g->batch.engram_rows` is managed memory with `prefill_cap` rows (40455).

**Implementation steps.**
1. In `batch_spec`, only when `spec != NULL`, compute on Apple:
   `measured_m3 = ds41_measured_config(graphs[0], w) && ds4_gpu_device_is_m3_ultra()`;
   `readers = measured_m3 && !DS4_DISABLE_V41_ENGRAM_STEP_READERS && !DS4_DISABLE_V41_VERIFY_ENGRAM_READERS`;
   `pipeline = measured_m3 && !DS4_METAL_DISABLE_V41_DECODE_PIPELINE && !DS4_METAL_DISABLE_V41_VERIFY_PIPELINE && g->prefill_cap >= 2*DS4_TP_BATCH_MAX_ROWS`;
   `overlap = readers && pipeline && !DS4_DISABLE_V41_ENGRAM_STEP_OVERLAP`.
   With `spec == NULL` (session batches, CUDA short prefill) nothing changes.
2. Move `ids` to function scope as
   `uint32_t ids[DS4_TP_BATCH_MAX_ROWS][2][DS4_ENGRAM_COLS]` and hash every row
   first with the same chained `history[]`.
3. If `readers`, call
   `ds4_engram_read_step_begin(&steps[i], s->table, ids[i], (float*[2]){engram[i][0], engram[i][1]})`
   per row and record it as pending. Without `overlap`, end every step
   immediately; without `readers`, keep the serial loop. Reuse the tested
   single-token API k times; add a batched API only if step 0 shows the join
   wait still matters.
4. Keep the embeds and the `pre`/`prefill_tokens` writes before layer 0.
5. At `il == 1`, end every pending step exactly once (any failure sets
   `ok = false`), then upload table 0 as today.
6. At `il == 14` with `pipeline`: upload table 1 into
   `rows_view[DS4_TP_BATCH_MAX_ROWS + i].engram_rows`; call
   `ds41_before_attention_batch` with a copy of `active` whose `engram_rows` is
   a view of rows [8, 8 + count); free the view after the layer.
7. Add `if (ok && pipeline && (il == 0 || il == 4)) ok = ds4_gpu_flush_commands();`
   and apply the layer-13 split only when `!pipeline`.
8. After the loop, end any step still pending, then free `engram`.
9. Leave the heads in the last buffer; `ds41_graph_verify`'s
   `ds4_gpu_end_commands` becomes the only wait.

**Gating and rollback.** Verify-only switches
`DS4_METAL_DISABLE_V41_VERIFY_PIPELINE` and
`DS4_DISABLE_V41_VERIFY_ENGRAM_READERS`; each decode switch also disables its
mechanism in verify. Admission: measured configuration on an M3 Ultra, spec
path only.

**Tests.** `--verify-parity` on every case in the default arm and each
rollback arm (all arms identical); assert the schedule with
`ds4_gpu_command_buffer_count()` as `tests/test_deepseek41_live_edges.c` does
(3 committed buffers and 1 wait with the pipeline, 2 and 2 without); extend
`tests/test_deepseek41_engram_admission.c` with the verify reader rollbacks
and an invalid row ID; inject a read failure under `MTL_DEBUG_LAYER=1`
(verify returns false, `g->valid == false`, no step left pending); greedy
byte-compare and the fixture with `REQUIRE_IDENTICAL=1`.

**Measurement and acceptance.** `--verify-timing` warm and cold with
`DS4_METAL_CB_TIMES=1`, then end-to-end t/s. Accept if c(k) per row drops by
≥ 0.05 step warm or ≥ 0.1 step cold, with decode unchanged.

**Pitfalls.** `ds4_gpu_tensor_write` is an immediate CPU memcpy into shared
memory (`ds4_metal.m:9515`): writing a buffer that a committed, unfinished
command buffer reads is a data race, which is why the late region is
mandatory once the layer-13 drain goes. `ids`, `engram` and the `out` pointers
must outlive the joins. The drafter shares `g->batch`, `rows_view` and
`prefill_tokens` with verify; safe only because the draft buffer has finished
before verify starts. The offset view needs `prefill_cap >= 16`, else fall
back to the drain. A read error now surfaces after layer 0 is committed: mark
the graph invalid and require a restore (as decode does); never return with a
step pending.

**Stop if.** Any parity arm differs, or the per-row gain is within noise both
warm and cold.

### 17. One wait per cycle after the draft, with GPU acceptance

**Goal and expected gain.** After the draft sync, encode verify → accept →
commit → seed between one `begin` and one `end` with a single wait; read back
`keep` and the per-row argmax; copy the kept row into `s->logits`. Removes two
command-buffer round trips and k CPU scans of 129,280 floats: roughly 1-2 ms
per cycle (0.03-0.06 step). Also the prerequisite for any later cross-cycle
overlap (Splash `runtime/model/Runtime.mm:2257-2349`,
`runtime/metal/kernels/decode/sampling.metal:866-917`).

**Why output stays exact.** `keep` equals the CPU keep only if the GPU argmax
has `sample_argmax`'s exact semantics; the GPU commit copies the same bytes as
`ds41_spec_commit`; `s->logits` receives the same row; the seed affects only
the drafter.

**`sample_argmax` semantics (verified).** It returns the smallest index whose
value is maximal among values that are not NaN and strictly greater than
−1e30, or 0 if there is none. −inf and NaN never win; +inf wins; +0 and −0 tie
(IEEE `==`), so the smaller index wins. The
`DS4_CPU_DISABLE_UNROLLED_ARGMAX` path follows the same rule.

**Code map.** Cycle `ds4_session_ds41_spec_cycle` (`ds4.c:85586-85697`):
verify 85654, CPU keep 85662-85666, commit 85667-85673, memcpy 85674, seed
85677-85680. `ds41_spec_commit` (41505-41531); `ds41_draft_seed`
(42661-42686); `sample_argmax` (44709), `sample_argmax_unrolled8` (44680),
`argmax_f32_unrolled8_range` (44622), `DS4_NEG_INF = -1.0e30f` (407).
`ds4_gpu_argmax_tensor` (`ds4_metal.m:19549`) → `ds4_gpu_indexer_topk_tensor(…,1,1)`
→ `metal/argsort.metal:60-150` (bitonic) and 230-318 (merge). The library is
fast-math (`ds4_metal.m:7064-7090`; `DS4_METAL_MATH_SAFE` is diagnostic).

**Implementation steps.**
1. `kernel_dsv41_spec_accept` in `metal/dsv41.metal`, one threadgroup per row
   over count+1 rows, using integer keys so fast-math cannot matter:
   `u = as_type<uint>(v)`; skip if `(u & 0x7fffffff) > 0x7f800000` (NaN); if
   `(u & 0x7fffffff) == 0` set `u = 0` (canonical zero);
   `key = (u >> 31) ? ~u : (u | 0x80000000)`; candidate only if
   `key > key(-1e30f)`. Reduce (max key, then min index) with each thread
   scanning its stride in increasing index order, then SIMD, then threadgroup
   (associative, so deterministic). Rows with no candidate return 0.
2. A one-thread keep kernel:
   ```
   keep = 1
   while keep <= count && am[keep-1] == tok[keep]:
       keep++
       if !ignore_eos && eos >= 0 && tok[keep-1] == eos: break
   ```
   writing `keep` and `am[]`.
3. `ds41_spec_commit_gpu(g, spec, keep_t)`: one conditional-copy dispatch per
   layer (the 40 `g->window[il]` are separate allocations) copying
   ring → window for rows `i >= keep`, and the 3 owners' pair restores with
   the slot derived from `keep` exactly as at 41517-41520, only when
   `keep < rows`. The host sets `g->history = spec->history[keep-1]` and
   `g->pos = spec->pos + keep` after the readback.
4. Seed variant: compute all `spec->rows` rows (rows-exact batch kernels make
   each row independent of the row count) but write window slots only for
   `i < keep`; the host adds `keep` to `d->seeded` and clamps `d->first` after
   the readback.
5. Split `ds41_graph_verify` into `ds41_graph_verify_encode` plus the existing
   wrapper (the parity test calls the wrapper). In the cycle: upload
   `tokens[]`; `begin` → `verify_encode` → accept → `commit_gpu` → `seed_gpu`
   → `end`; read `keep` and `am`, bounds-check `1 ≤ keep ≤ count+1`; memcpy row
   `keep−1` into `s->logits`; push tokens, update `ds41_draft_tail`, stats.
6. Stays on the host: drafted token IDs (Engram hashes and reads need them,
   which is why the draft sync stays); choosing `count` (gate, `room`,
   `max_tokens`, ctx, cutting at a drafted EOS); `checkpoint` and
   `accepted[]`; `g->pos`, `history`, `d->seeded`; the `s->logits` copy.
   Callers pick the next token themselves (`ds4_session_sample`,
   `ds4_cli.c:611`; `ds4_session_argmax_excluding` in ds4-bench): do not
   substitute the GPU argmax for that choice.

**Gating and rollback.** `DS4_METAL_DISABLE_V41_SPEC_ACCEPT` returns to
today's flow; admission via `ds4_gpu_dsv41_exact_admitted(DS4_GPU_TEST_V41_FUSIONS, …)`.

**Tests.** A model-free argmax oracle in `DEEPSEEK41_EXACT_DEFAULT_TESTS`
against both CPU variants: ties in the same lane and across lanes
(index mod 8), at index 0 and the last index; ±0 ties; NaN at the maximum; all
NaN; several +inf; −inf; exactly −1e30 and values ≤ −1e30 (expect 0);
subnormals; n = 129,280 and odd sizes. A keep oracle: EOS at every position,
`ignore_eos` both ways, `eos = -1` (the batch API passes −1 at
`ds4.c:81506`). `--verify-parity`: for every `keep`, `commit_gpu` gives a
snapshot byte-identical to `ds41_spec_commit`'s and to decode's. A new
`--cycle-parity` mode: N fused cycles against decode, comparing tokens,
`s->logits` after each cycle and the final snapshot.
`DS4_DSPARK_ACCEPT_CHECK=1` re-derives `keep` with `sample_argmax` every cycle
and aborts on mismatch; enable it for all benchmarks of this item. Greedy
byte-compare and the fixture.

**Measurement and acceptance.** Command buffers and waits per cycle
(`ds4_gpu_command_buffer_count()`, `DS4_METAL_CB_TIMES`); cycle ms by `keep`
(`DS4_BENCH_SPEC_TRACE`); `verify_ms`. Accept at ≥ 0.5 ms saved per cycle.

**Pitfalls.** Read `spec->logits` only after the wait. Keep the host
`ds41_spec_commit` for the rollback and tests. The `count == 0` path is
separate; optionally encode its seed into `ds41_graph_step`'s last buffer next
to the head (41661-41664).

**Stop if.** Any oracle case mismatches, or less than 0.5 ms per cycle is
saved.

### 18. Cost-model gate instead of the fixed confidence threshold

**Goal and expected gain.** Replace
`confidence[count] >= e->dspark_confidence_threshold` (`ds4.c:85627-85631`;
default 0.6 on Metal, 71927-71932) with a gate that maximizes tokens per
millisecond and falls back to plain decode when drafting loses. With today's
c(k) the gate will mostly choose plain steps (one draft needs
1 + q₁ > c(2) = 2.08 steps, impossible; a full verify needs q ≈ 0.97 per
draft), which brings DSpark to about decode speed now (prose loses 19% today,
code 3%). As later items lower c(k) and D, the gate starts drafting where it
pays. Current D is roughly 10 ms (~0.3 step) per drafted cycle.

**Why output stays exact.** The gate chooses only how many drafts to verify;
verification and acceptance are unchanged.

**Code map.** Cycle 85586-85697, `drafting` predicate 85598-85600, `room`
85619-85622; stats `ds4_dspark_spec_stats` and printer 74122. Qwen precedent
`qwen4_ema` / `qwen4_batch_spec_next` (81319-81327) under
`#ifdef DS4_HAS_QWEN4_METAL` (commits 3077786, 9ba464d); V4.1 needs its own
helper or an unguarded copy. Engine fields near 42926, session fields near
61653.

**Implementation steps.**
1. Pure function `n* = ds41_spec_choose(q[], drafts[], drafted, eos, ignore_eos, room, cost_ms[])`:
   ```
   T(n) = 1 + Σ_{j≤n} Π_{i≤j} q_i   (n stops at a drafted EOS unless ignore_eos, and at room)
   C(n) = c(n+1)                     (ms after the draft sync; c(1) = plain step with capture + seed)
   n*   = argmax_{0≤n} T(n)/C(n)     (D is already sunk; n* = 0 → plain step)
   ```
2. Engine state: EMA `c_ms[1..7]` and `D_ms`; extrapolate unseen k with a
   least-squares a + b·k over observed EMAs (before two k values exist, use
   the step 0 fit scaled by measured c(1)). Confidence calibration in 20 bins
   as EMA-weighted hits and trials with a Beta prior centred on the confidence
   (weight ~8); the confidence is a conditional (acceptance given the earlier
   drafts) and is calibrated as one.
3. Free labels: keep the last few draft blocks (start pos, drafts,
   confidences). When the checkpoint covers pos+1+j, label draft j accepted if
   `drafts[j] == checkpoint.v[pos+1+j]`, using it only if every earlier draft
   in the block also matched. This labels drafts that were never verified, so
   calibration needs no verify probes. Clear the ring when the checkpoint
   shrinks, on reset or restore.
4. Whether to draft this cycle: draft while the realized EMA rate of drafted
   cycles `keep/(D + c)` exceeds `1/c(1)` with 2% hysteresis; otherwise a plain
   step with capture + seed so the windows keep following; while plain, a
   draft probe 1 cycle in 32; while drafting, 2 forced plain cycles every 128
   to refresh c(1) (the shape of 3077786); the first 8 cycles verify full
   blocks so costs get priced.
5. Update each EMA only from steady cycles of the matching kind. Trace under
   `DS4_DSPARK_POLICY_TRACE=1`: confidences, q, T(n), C(n), n*, keep, costs.

**Gating and rollback.** `DS4_DISABLE_V41_DSPARK_POLICY=1` restores the fixed
threshold; an explicit `--dspark-confidence` also keeps the fixed rule;
diagnostic `DS4_DSPARK_POLICY_FORCE=plain|full|threshold`.

**Tests.** Unit tests of `ds41_spec_choose` (EOS, `room`, extrapolation,
monotone cases) and of the label bookkeeping; greedy byte-compare of DSpark
against plain decode in every FORCE mode and the default; `--verify-parity`;
the fixture.

**Measurement and acceptance.** Both 400-token prompts, 3× ABBA, warm.
Required: DSpark ≥ decode on code and ≥ decode − 1% on prose. Measure
`FORCE=plain` against plain decode (no `--dspark`) to size the seeding
overhead; above 0.5%, switch to lazy seeding (capture only into a
128-position hidden ring, seed in bulk when drafting resumes; shares
machinery with item 20). Tell the user acceptance statistics change because
verified lengths change.

**Pitfalls.** c(k) must exclude D. Transition cycles price neither kind. Keep
hardware costs and calibration in the engine, realized rates in the session.
`room` and EOS cap n exactly as the current loop does.

**Stop if.** DSpark with the gate stays more than 1% below decode even when
forced plain and with lazy seeding.

### 19. Cheaper drafter

**Goal and expected gain.** Cut D (~0.3 step per drafted cycle); matters most
for prose (1.96 tokens per cycle) and directly raises what the gate can gain.
Output is unaffected because every draft is verified exactly; speed and
acceptance move, and acceptance changes must be reported to the user.

**Code map (per 5-row draft block).** `ds41_draft_block` (`ds4.c:42752-42826`):
host writes to `prefill_tokens` and `pre`; one `ds4_gpu_embed_token_hc_tensor`
per row; `ds41_draft_stage` ×3 (42689-42746), each: batched HC mix and norm;
`ds41_attention_project_batch` (rows-exact Q8_0 kernels); one
`ds4_gpu_attention_decode_heads_tensor` per row (42708-42711) over the
≤128-slot window plus the block; one `ds41_attention_output` per row
(42713-42720); router, rows-exact shared expert, routed MoE with 128 experts /
3 used (tiny-pair kernels, 15 pairs); HC expand. Final norm, one batched head
(42785). The Markov chain runs sequentially per row (42790-42803): w1 embed,
`ds41_matmul(bias, markov_w2)` (F16 256 → 129,280, 66 MB), add,
`ds4_gpu_argmax_tensor` (full sort-based top-1). Host confidence: dot products
of 5,120 + 256 values per row (42815-42825).

**Implementation steps** (measure each with `--draft-timing` first).
1. Draft-only fast matmuls: plain batched kernels
   (`metal_graph_matmul_plain_tensor` n_tok = 5 → `mul_mv_ext`) instead of
   `ds41_matmul_batch`'s rows-exact branch, via a `fast` flag or a
   `ds41_draft_matmul` helper used only by the drafter.
2. One attention dispatch for all rows with rows in the grid (they share K and
   V; only q differs).
3. Batched attention output with the prefill kernel
   `ds4_gpu_dsv41_attention_output_batch` (see 42272) plus one quantize.
4. Replace the sort-based argmax with item 17's reduction argmax.
5. Top-K Markov: after the head, one dispatch computes the top-16 base-logit
   candidates for all 5 rows; then one single-threadgroup chain kernel that,
   for i = 0..4, embeds `ids[i]` from w1 (a 256-wide F16 row), computes the
   bias only for row i's 16 candidates (w2 rows are 256 contiguous F16
   values), adds, takes the argmax (smallest index on ties), writes
   `ids[i+1]` and `d->markov` row i, and optionally the confidence. This
   replaces ~50 serially dependent dispatches with 2 and differs from the full
   Markov only when the bias would promote a token outside the base top-16.

**Gating and rollback.** `DS4_METAL_DISABLE_V41_DRAFT_FAST_KERNELS` (steps
1-4) and `DS4_METAL_DISABLE_V41_DRAFT_MARKOV_TOPK` (step 5); diagnostic
switches, fast paths default.

**Tests.** `--verify-parity` (unchanged path, as a guard); greedy
byte-compare of DSpark against plain decode on both prompts and the fixture
with `REQUIRE_IDENTICAL=1` (must be byte-identical although the drafts
change); a unit check that the top-K chain equals the full chain whenever the
full argmax lies inside the candidate set.

**Measurement and acceptance.** A new `--draft-agreement PROMPT` mode: along a
teacher-forced prompt, seed the windows identically and run the old and new
drafters from the same state; report per-position draft agreement and
realized acceptance against the known target greedy tokens (accept rate, mean
accepted length, histogram). End to end: `DS4_DSPARK_STATS` `accept_rate`,
`avg_accept`, `accepted_len_hist`, `propose_ms` and t/s on both prompts plus
the 5 fixture prompts, 3 runs each. Keep a change only if t/s improves; an
acceptance drop above 1 percentage point must be justified by net speed and
reported.

**Pitfalls.** The drafter shares `g->batch`, `rows_view` and `prefill_tokens`
with verify; keep its command buffer finished before verify writes from the
host. The MoE path depends on `n_tokens <= 5` for tiny pairs
(`use_tiny_pair_mv`). Confidences must come from whichever `final_x` and
`markov` the new path writes.

**Stop if.** D falls by less than 20%, or acceptance falls enough to cancel
the gain.

### 20. Seed the draft windows during prefill

**Goal and expected gain.** After a prefill or restore,
`ds41_session_draft_follows` (`ds4.c:85574-85580`) fails and the windows
reset (85615). With empty windows `ds41_draft_block` returns 0 (it checks
`d->seeded == d->first` at 42760), so the first cycle is a plain step; the
windows then grow one position per token and reach 128 only after ~128
generated tokens. Seeding from the last ≤128 prompt positions gives
full-context drafts from the first cycle, which helps short generations and
every new chat turn. Splash does the same
(`runtime/model/DraftContextPlan.cpp:8-100`, `QwenTarget.cpp:236-251`).

**What the drafter needs (verified).** For each window position p,
`ds41_draft_seed` (42661-42686) consumes `hidden(p)`: the concatenation over
target layers [37, 38, 39] of BF16(mean over the 4 HC copies, weight 1/HC, of
the residual entering that layer). `ds41_spec_capture` (41457-41476) computes
it with `quarter` + `ds4_gpu_hc_weighted_sum_tensor` + BF16 quantize. From it
`ds41_draft_seed` computes per stage main_proj → main_norm → attn_kv → kv norm
→ RoPE at p → FP8 into `window[st][p % 128]` (each stage window 128 × 512).

**Why output stays exact.** The capture adds dispatches that only read the
prefill residual and write side buffers; no prefill buffer, dispatch order or
synchronization changes, so prefill logits, snapshots and the following decode
stay byte-identical. The windows affect only drafts.

**Code map.** Session prefill loop `ds4.c:76833-76905` (layer-major
`ds41_graph_prefill` 76876; token-major `ds41_graph_step(…, NULL)` 76878;
final logits 76908). `ds41_graph_prefill_sweep` (42050-42366): decoder suffix
`needed = 1 + (39 − il)·127` at 42143 (382, 255, 128 rows at layers 36, 37,
38); end of chunk work 42316-42321. `g->capture` (40277) and its use at 41653.
Lazy draft allocation 85603; freed at 74722.

**Implementation steps.**
1. Allocate the draft at the first prefill when the engine will draft (same
   predicate as 85598-85600), with side buffers `prefill_hidden`
   (128 × 3 × 5120 floats, ~7.5 MiB) plus `mean` and `quarter` for 128 rows.
2. Add `g->prefill_capture` holding the tensors and the span start
   `S = total − W`, W = min(128, prompt tokens being prefilled); set it in the
   session loop and clear it afterwards.
3. In the sweep, for every chunk at layer `il = T−1` (T a target layer),
   capture rows [max(off, S − start), off + count) from the residual after
   layer il's HC expand, before the carry store and `ds4_gpu_end_commands`
   (the exit of T−1 equals the entry of T; this also covers the decoder
   suffix, where layer 39 computes only the last row).
4. Token-major tail steps (a remainder under 8 rows goes token-major at
   76878, so the span can straddle both forms): in `ds41_graph_step`, next to
   41653, capture the entry of each target layer into `prefill_hidden` at row
   pos − S.
5. After the final logits: if the windows followed the session before this
   prefill (checked at its start), keep them contiguous and clamp
   `first = max(first, total − 128)`; otherwise `first = seeded = S`. Seed the
   W rows in chunks of 8 via a generalized
   `ds41_draft_seed(d, hidden_view, rows, pos)`; set `d->seeded = total`;
   update `ds41_draft_tail` from the checkpoint.
6. Snapshots and restore: keep the snapshot format unchanged (the windows are
   never serialized). After `load_snapshot` or a KV-cache restore the windows
   reset; the next prefill seeds from its own suffix (partially if shorter
   than 128 tokens); a restore with no prefill starts cold, as today.

**Gating and rollback.** `DS4_DISABLE_V41_DSPARK_PREFILL_SEED`.

**Tests.** Prefill logits and snapshot with and without capture
byte-identical (extend `--short-prefill` or add a mode covering chunked,
wide ≥ 8192 with the decoder suffix, and token-major-tail prompts);
`--verify-parity`; greedy byte-compare of DSpark against plain decode; a unit
check that window positions stay contiguous across prefill → cycle → prefill.

**Measurement and acceptance.** Split acceptance and tokens per cycle by
window fill (<32, <64, <128, full); compare against baseline over the first
128 generated tokens and on `-n 64/128/400`. Prefill t/s unchanged within
noise.

**Pitfalls.** HC helpers infer the row count from buffer sizes (comment at
42169): size views exactly. Positions are absolute (`start + t`).
`encoder_only` sweeps must not capture. With
`DS4_METAL_DISABLE_V41_BATCH_HC`, `active` is absent; use `rows_view` or
explicit views of `g->batch.residual`. Prefill arithmetic is not bitwise
decode arithmetic, so seeded means differ slightly from decode-captured ones
(drafts only).

**Stop if.** Early-window acceptance does not improve, or prefill slows by
more than 0.5%.

### 21. Expert-grouped verify MoE

(The rows-exact verify head is part of item 15.)

**Goal and expected gain.** Routed experts are the largest stage (9.09 ms per
token), and verify reads each (row, expert) pair's weights separately. The
gain is bounded by the fraction of pairs that share an expert across verify
rows, so measure that first. With random routing, 6 rows × 6 experts over 384
experts share almost nothing; real locality is unknown.

**Why output stays exact.** Each (row, expert) keeps its one-row walk,
accumulation order and destination; only weight reads are shared.

**Code map.** `ds41_moe_batch` (`ds4.c:41351`);
`ds4_gpu_routed_moe_batch_tensor` (`ds4_metal.m:43868`) with
`v41_decode_batch` at 44237 and `use_tiny_pair_mv` at 44245; pair-swiglu grid
`(row_groups, 1, pairs)` at 32251-32303; sum6 down grid
`(row_groups, tokens, 1)` at 32642; `kernel_mul_mv_id_mxfp4_sum6_f32`
(`metal/moe.metal:6403`) accumulates each lane's partial over the slots in
slot order, then one `simd_sum`. V4 precedent diagnostic
`DS4_DSPARK_EXPERT_OVERLAP` (`ds4.c:38589`).

**Implementation steps.**
1. Measure: when `DS4_DSPARK_EXPERT_OVERLAP=1`, copy `g->batch.selected`
   (rows × 6 int32) after each layer's router into a [40][8][6] tensor; after
   the verify print, per k, the mean unique experts per pair, per layer and
   over the run.
2. If shared pairs are ≥ 15-20%, try scheduling only first: make pairs the
   fastest grid dimension within a row group for pair-swiglu and tokens the
   fastest for sum6, or have a tiny GPU pass sort pairs by expert so the
   kernel reads `pair = perm[tgpig.z]`. Same-expert pairs then run back to
   back and hit L2/SLC; per-thread arithmetic is unchanged.
3. Only if that is insufficient: a grouped gate/up kernel modelled on
   `kernel_mul_mv_q8_0_f32_rows` (a threadgroup per (row group, expert)
   iterates over that expert's pairs inside the K loop, with per-pair
   accumulators and the per-pair `mid` destination).
4. Down/sum6 can be grouped exactly only by storing per-(token, slot) lane
   partials and adding them in slot order; complex, so do gate/up (2/3 of the
   expert bytes) first.

**Gating and rollback.** `DS4_METAL_DISABLE_V41_VERIFY_MOE_GROUP`, admitted
via `ds4_gpu_dsv41_exact_admitted`.

**Tests.** A pair oracle with forced shared experts comparing grouped against
per-pair outputs; `--verify-parity` on all cases and both arms; greedy
byte-compare; the fixture.

**Measurement and acceptance.** `--stage-timing ROWS` for `ds41_moe_batch`
with k = 2..6 and `--verify-timing`. Accept at ≥ 1.5 ms per 6-row verify.

**Pitfalls.** The fixed threadgroup memory size must match decode's dispatch.
Confirm with the parity test that the verify tiny-pair kernels still equal
decode's kernels for MXFP4.

**Stop if.** Shared pairs are below ~15%, or the scheduling-only variant shows
no measurable gain.

## E. Decision for the user: row-invariant MMA decode (not default)

### 22. Go/no-go probe: row-invariant 8-slot fp32 simdgroup-matrix Q8_0 kernel

Do not start this item without the user's go-ahead. It only builds a
standalone probe; integration is a separate decision.

**Goal.** Decide with data whether one kernel should serve both one-token
decode and 1-8-row verify, so verify rows cost almost nothing extra and
DSpark equals decode by construction. This is Splash's actual design. On
Apple9 the 8×8×8 fp32 MMA runs on the FP32 pipe, so 8 slots with 1 live row
do 8× the FMAs: head 5.3 GFMA ≈ 0.37 ms at peak against 1.0 ms of memory
time; q_b 13.4 GFMA ≈ 0.94 ms over 40 layers against 2.86 ms. One live row
avoids a regression only if the MMA overlaps the weight stream; that is what
the probe measures.

**Why it is not exact, and what must stay invariant.** It changes the
summation order (MMA over 8 k at a time, optional scale folding, split-K), so
its decode logits differ from today's; it can never be the default under the
bit-identical rule. Four properties must hold:
1. Row independence: a row's bits do not depend on its slot, on other slots'
   contents (including NaN/Inf) or on the live-row count.
2. Decode equals verify: 1 live and L live give identical bits for a row.
3. Determinism: identical bits across runs and across both dies (fixed split
   order, coherent partials).
4. Error bound: |y − y_fp64| ≤ γ_{K+S+2}·Σ|q·d·x| with
   γ_n = n·2⁻²⁴/(1 − n·2⁻²⁴), valid for any summation tree.

**Code map (probe only; nothing in `ds4_metal.m` or `metal/` changes).** New
`tests/bench_v41_sgmatrix_q8.m` and `tests/bench_v41_sgmatrix_q8.metal`,
compiled at runtime with default options (fast-math, like ds4) plus
`#pragma clang fp reassociate(off)`; new Makefile target
`bench-v41-sgmatrix-q8` linking `$(CORE_OBJS)`, NOT part of `test` or
`DEEPSEEK41_EXACT_DEFAULT_TESTS`. Baselines through the public API:
`ds4_gpu_matmul_q8_0_tensor` (1 row),
`ds4_gpu_matmul_q8_0_decode_rows_exact_tensor` with
`DS4_GPU_TEST_V41_SHARED_ROWS` (6 rows), `DS4_METAL_V41_Q8_NR0` for item 2's
best NR0. Splash references: `runtime/metal/kernels/common/sgmatrix.h`
(`lane_map`; `mma_acc` with MMA objects local to the call and a register
float2 accumulator), `common/split_reduce.h` (`coherent(device)` partials,
device fences, last arriver adds in split order), `decode/linear_gguf_sgmatrix.metal`
(lanes × 8 rows, loads issued before waits), `dev/tuning/LinearNumerics.hpp`
(gamma bounds).

**Implementation steps.**
1. Kernel: threadgroup = 4 simdgroups × 8 output rows. B is always 8 slots
   read from a host-padded [8][K] activation buffer; the kernel never sees the
   live count. A from int8 q (exact in fp32). Variant V1: 4 MMAs per 32-block
   into `dot`, then `acc = fma(dot, d, acc)`. Variant V2: fold
   A = float(q)·float(d) (exact, ≤ 19 significant bits). Fixed split-K per
   shape (q_b S = 1, head S = 1, o_b S = 4) with coherent partials and a
   per-tile counter; the last arriver adds s = 0..S−1 in order and resets it.
2. Data: cold weights (40 distinct matrices for q_b 1280→32768 and o_b
   8192→5120, 4 for the head 5120→129280); q uniform int8; d half in
   ±[2⁻¹², 2⁻⁶]; x BF16-rounded N(0,1); edges K = 1312 (41 blocks),
   N = 5123.
3. Timing: 40 dispatches per command buffer, best of 5, wall and
   `GPUStartTime`/`GPUEndTime`; 1 and 6 live rows for both variants against
   today's kernel (1 row), today's rows kernel (6 rows) and item 15's version.
4. Correctness: fp64 reference on 4096 sampled rows per shape (all rows for
   o_b) against the bound (today's kernel must pass it too, or the bound is
   wrong); slot independence for L ∈ {1, 2, 6, 8} with the other slots filled
   with zeros, random, NaN, ±Inf, 1e38, subnormals, −0; the same row in slots
   0-7 identical; L = 1 versus L = 6 identical; 50 repeats identical; once
   under `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`.
5. Output: one JSON line per (shape, variant, live rows): ms, GB/s, max
   err/bound, violations, checks. Archive under
   `speed-bench/ds41f-m3ultra-perf/sgmatrix-probe/`.

**Go/no-go rule.** GO only if all hold: (a) every invariance, determinism
and bound check passes; (b) for each of q_b, o_b and head, 1-live time ≤ 1.02×
today's one-token time at item 2's best NR0; (c) 6-live ≤ 1.25× its own
1-live time and ≤ 0.6× item 15's exact rows kernel at 6 rows; (d) using
measured `--stage-timing`/`--verify-timing`, projected DSpark t/s on the code
prompt ≥ 1.10× exact DSpark with decode losing ≤ 2%. Otherwise NO-GO; archive
the results.

**What an opt-in integration would need** (only with the user's approval; it
conflicts with `AGENT.md`'s "no permanent semantic variants behind flags"):
`DS4_METAL_V41_SGMATRIX_DECODE=1` read once per session; every V4.1 Q8_0 site
switched together for decode and verify (q_a, q_b, kv, o_a, o_b, gate/up,
down, head, DSpark main_proj); split counts fixed per shape, independent of
row count and load; coherent partials; quality, SSD and TP excluded; the mode
recorded in snapshots (or refused across modes); its own quality check
against today's logits (KL, top-1 agreement) and `--verify-parity` within the
mode. Do not extend it to MXFP4 routed experts: at 0.53 bytes per weight, 8
slots risk becoming compute-bound.

**Pitfalls.** Warm weights inflate everything. A live-count-dependent split
or grid silently breaks invariant 2. Without `coherent(device)` partials,
stale reads appear only on large grids spanning both dies. Judge only against
the post-item-2/15 exact baselines. Run only on an idle GPU.

**Stop if.** Any invariance check fails even once, any bound violation
appears, or (b) fails for q_b or the head.

## F. Server time to first token

These items change no model arithmetic. Their bar is that every checkpoint
file stays byte-identical to what the synchronous path writes (mask header
bytes 24-39, the timestamps, when comparing across runs) and that lookups
return the same entries.

### 23. Write-behind (asynchronous) kvstore checkpoint writes

**Premise corrections.** The cost is higher than the review assumed. Each
store serializes the whole payload twice: `ds4_session_stage_payload`
(`ds4.c:63994-64043`) writes it to `/tmp/ds4-session-payload.XXXXXX`, then
`ds4_session_write_staged_payload` (`ds4.c:63966`) copies it through
`payload_copy_file_bytes` (`ds4.c:62255`) into `<sha>.kv.tmp.<pid>` (`fopen`
at `ds4_kvstore.c:1064`, `rename` at 1113, no fsync). Commit 4844d55 added the
staging for distributed sessions, whose size is unknown
(`ds4_session_payload_bytes` returns 0 when `s->distributed`, `ds4.c:63915`);
local sessions know their size. The logged `save=… ms` starts after staging
(1063), so it under-reports. The server holds `inference_mu` and `kv_mu` for
the whole store (`ds4_server.c:11301-11320`), which in batched mode stalls
every slot and the decode thread. Call sites: evict before load 13633 (load at
13636); cold anchor mid-prefill 13779; progress callbacks 12735 and 12776
(inside `ds4_session_sync` under `inference_mu`, 12467-12470); continued store
after the prompt 13829; cold store at prompt end 13838; decode-time continued
13968; shutdown 16156. Snapshot source: `ds41_save_payload` (`ds4.c:63820`)
calls `ds4_gpu_synchronize`, then `payload_write_tensor_span` (62460), then
`ds4_gpu_tensor_read`, which on Metal is a `memcpy` from shared buffer
contents (`ds4_metal.m:9525-9533`) and on CUDA a device-to-host copy.

**Goal and expected gain.** Take checkpoint file I/O off TTFT and out of the
global lock. The caller then pays one memory copy of the payload (220,441,392 B
at 32,703 tokens; 848,748,596 B at 130,816) and a background thread writes the
file. Stage 0 alone (direct write for local sessions) removes one full payload
write and read-back from every store. Measure the saving; do not assume it.

**Why outputs are unchanged.** The writer never touches the session. The
payload comes from the same `ds4_session_save_payload`, written into memory
instead of `/tmp`. The file image is assembled in today's order: 48-byte
header with `now` captured at enqueue, `text_len`, text, payload, trailer
captured at enqueue. Lookup and eviction must behave as if the write had
finished: pending entries are visible to lookup and can be evicted, and a load
that picks a pending entry waits for it. This matters because evict, shutdown
and anchor checkpoints are not on the 2048-token grid, so a hit versus a miss
can change the bits of a continued prefill.

**Code map.**
- `ds4_kvstore.c`: `ds4_kvstore_store_live_prefix_text` 929-1161;
  `kv_cache_existing_compatible` 849-871; `kv_cache_rewrite_trailer` 889-927;
  header and `now` 1078-1086 (`ext_flags` from the trailer estimate at 1080);
  final size check 1103-1112; `ds4_kvstore_evict` 567-613;
  `ds4_kvstore_try_load_text` 1221-1346 (fopen 1243, touch 1327, corrupt-file
  unlink 1302); `ds4_kvstore_open`/`ds4_kvstore_close` 615-657.
- `ds4.c`: `ds4_session_save_snapshot` 65263-65304 (comment at 65281: fmemopen
  writes a trailing NUL).
- `ds4_server.c`: trailer hooks 11266-11290; `kv_tool_map_write` 11016 takes
  `tool_mu`; `server_close_resources` 15534 calls `kv_cache_close`; a second
  signal calls `_exit` (line 58).

**Implementation steps.**
0. Direct write for local sessions: `payload_bytes = ds4_session_payload_bytes(s)`;
   if nonzero, skip staging and call `ds4_session_save_payload(session, fp)`
   into the tmp file, check the `ftello` delta equals `payload_bytes` before
   the rename. Keep staging for distributed sessions only. Time the whole call.
   Land and measure this alone first.
1. Split the store into *prepare* (today's checks, text, SHA, path, trailer
   estimate, size check, eviction, `now`) and *emit*.
2. Add `int ds4_session_save_payload_mem(ds4_session *, uint8_t *dst, uint64_t bytes, char *, size_t)`:
   fmemopen on `dst` with capacity `bytes + 1`, `ds4_session_save_payload`,
   require `ftello == bytes`, reject distributed sessions. Build the file image
   in one 16 KiB-aligned buffer (`posix_memalign`): head, payload, trailer.
   Copy the trailer after `fclose`, because the fmemopen NUL lands on the first
   trailer byte.
3. Capture the trailer with `open_memstream` + `kv_trailer_write`, keeping the
   order: size estimate, payload, trailer. Run the final size check with the
   real trailer size before enqueueing.
4. Add to `ds4_kvstore`: `write_behind`, `io_mu`, `io_cv`, `pending_bytes`, a
   job list and a writer thread, initialized in `ds4_kvstore_open`, writer
   started lazily on first enqueue; add `ds4_kvstore_flush()`;
   `ds4_kvstore_close` flushes then joins.
5. Writer: open `<path>.tmp.<pid>.<seq>` with `O_CREAT|O_EXCL|O_CLOEXEC`,
   write 8 MiB aligned slices, close, mark the job done/cancelled/failed under
   `io_mu`, rename only on success and not cancelled, otherwise unlink; log
   `stored`/`failed` with wall time and queue delay. It never calls hooks or
   takes server locks; `server_log` is mutex-protected (9858).
6. Index (builds on item 24): entries are published or pending; only caller
   threads holding `kv_mu` change the array and reap finished jobs at the
   start of each API call. Eviction totals include pending bytes; evicting a
   pending entry cancels it (dequeue if not started, else set the cancel
   flag). `find_text_prefix` sees pending entries; `try_load` on one waits on
   `io_cv` and redoes the lookup if the job failed. A store whose SHA matches a
   pending entry waits for it, then runs the existing compatible and
   trailer-rewrite path.
7. Staging budget: at most 2 GiB pending (constant). A larger image uses the
   Stage 0 synchronous path; otherwise wait on `io_cv` until it fits (safe: the
   writer needs neither `kv_mu` nor `inference_mu`).
8. Return true on enqueue (the server records it at 11445, 13780, 13839); a
   later failure is only logged, as failed continued targets are not retried
   today either.
9. In `ds4_kvstore_open`, remove `*.kv.tmp.<pid>*` files whose pid is dead
   (today they leak and are never counted against the budget).
10. Rewind, invalidate and load need no blocking, because each job owns its
    copy. Optional Stage B (Metal and V4.1 only, only if the copy cost
    matters): copy synchronously the header, tokens, logits, 40 window spans
    and `previous_kv`/`previous_score` (~11 MiB) and stream the append-only
    `compressed`/`index_cache` spans later. That needs engine pins in
    `ds4_session_rewind`, `ds4_session_invalidate`, `ds4_session_sync` (when
    common < pos), `ds4_session_load_payload`, `ds4_session_load_snapshot`,
    `ds4_session_rewrite_from_common`, `ds4_session_free` and the batched
    invalidations at 13379 and 14253; evict-then-load must then wait.

**Gating and rollback.** `DS4_KVSTORE_SYNC_WRITES=1` runs the same job inline
on the caller thread. No `DS4_KVSTORE_*` variables exist yet; parse it like the
`DS4_SERVER_*` variables (13294). The file format does not change.

**Tests.** Extend `./ds4_test --server` (server unit group,
`ds4_server.c:23240-23282`):
- inline and asynchronous images with a fixed `now` match byte for byte;
- with the writer blocked by a test latch, lookup finds the pending entry,
  only `*.tmp.*` exists on disk, and a load waits;
- injected write failure (read-only directory or a hook): nothing published,
  tmp file gone, entry removed;
- eviction during a pending write cancels it; `close` drains three jobs;
- the stale-tmp sweep removes dead-pid files and keeps live-pid files.
Model-backed: the V4.1 zero-weight fixture
(`tests/test_deepseek41_manifest.py --fixture`), store at 512, 2048 and an odd
position in both modes and compare files (mask header bytes 24-39). Live: add
`--kv-cache-min-tokens 16` to `tests/test_deepseek41_http.py` (with `--ctx 512`
it never stores anything today), diff the kv directories between modes, then
`kill -9` mid-write and restart: no partial `.kv` file may remain.

**Measurement and acceptance.** `speed-bench/serve_concurrency_bench.py` with
`--concurrency 1` and 8,192/32,768-token prompts (cold anchors, continued
stores), then `--batched-session 4`; read TTFT and ITL p99; with and without
`DS4_KVSTORE_SYNC_WRITES=1`, in normal Metal and `--ssd-streaming`. Keep if
files are byte-identical, TTFT improves by roughly the removed store time, and
decode/prefill during background writes stays within 0.5%.

**Pitfalls.** `ds4_kvstore` becomes non-copyable, but
`kv_cache_slot_continued_target` copies `s->kv` by value without `kv_mu`
(11390): pass only `opt` and `last`. Tests zero-initialize the struct and a
zeroed `pthread_mutex_t` is invalid on macOS: initialize only in
`ds4_kvstore_open` and keep async off for tests that never call it.
Background writes compete with SSD expert streaming and cold Engram reads.
`ds4-agent` has its own writer (`ds4_agent.c:5048`) and is unaffected.

**Stop if.** Any file byte differs, the snapshot copy exceeds half of today's
full store time, or background writes cost more than 1% of decode or prefill
that pacing cannot fix.

### 24. Single-pass prefix hashing and a cached directory index

**Premise corrections.** `ds4_kvstore_find_text_prefix` (1196-1219) calls
`kv_cache_refresh` (474-489), which clears the index, runs `opendir`, and for
every `<40hex>.kv` does `stat` + `fopen` + a 52-byte read
(`ds4_kvstore_read_entry_file` 448-472); it then hashes `prompt[0:text_bytes]`
from scratch for each candidate in readdir order, skipping shorter candidates
only after a hit. Keys are SHA1 of the full rendered prefix (998-999). Two
more hotspots: `ds4_kvstore_evict` (567-613) also rescans, runs on every store
(1058), and re-hashes the incoming text per continued entry inside its victim
loop through `kv_cache_incoming_supersedes_continued` (510-530), which is
O(victims × entries × text); and `kv_cache_restore_tool_memory_for_messages`
(`ds4_server.c:11128-11167`) opens every file for each request that carries
tool ids (called at 4271, 4497, 5528, without `kv_mu`). Disk lookups run only
on live misses (13635), so store-triggered rescans are the more common cost.
`sha1_ctx` (215-220) is plain data and `sha1_final` (295-310) mutates only its
own context, so copying the struct and finalizing the copy is valid.

**Goal and expected gain.** Hash the prompt once, O(prompt + candidates), and
stop reopening every header per call. Likely milliseconds at the default
4096 MiB budget, more with hundreds of 14-80 MB anchors at large budgets.
Measure it.

**Why outputs are unchanged.** For a given `text_bytes` at most one entry can
match (equal SHA means the same file name). The new search picks the same
largest-`text_bytes` match. The index reproduces the file set a rescan would.

**Code map.** As above. Existing tests: lookup `ds4_server.c:21724-21830`,
eviction 22128-22438.

**Implementation steps.**
1. `kv_prefix_sha_scan(text, len, sorted_unique_lens[], n, out[][41])`: one
   `sha1_update` pass that copies the context at each length and finalizes the
   copy.
2. `find`: filter as today, sort candidates by `text_bytes`, scan once, keep
   the largest match (keep the tie rule on `tokens`).
3. `evict`: compute a `superseded[]` flag once per call and use it in an
   internal score function; keep the public
   `ds4_kvstore_entry_eviction_score` unchanged.
4. Add `index_valid` and `dir_mtime` to `ds4_kvstore` (`ds4_kvstore.h:67-80`)
   and replace the refresh with `kv_index_sync`: `stat(dir)` (`st_mtimespec`
   on Darwin, `st_mtim` on Linux); rescan only when the mtime changed or the
   recorded mtime is under 2 s older than its scan (coarse timestamps); record
   the mtime from a stat taken before `readdir`.
5. Apply the kvstore's own changes in memory: publish adds an entry from the
   header just written; evict, incompatible-replace and corrupt-unlink remove
   entries; touch updates `hits` and `last_used`. Then re-stat the directory
   and adopt the new mtime only if the pre-change mtime matched, otherwise
   invalidate. `ENOENT` in `try_load` invalidates the index.
6. Optionally let the tool-memory scan use the index and cached trailer
   offsets.

**Gating and rollback.** `DS4_KVSTORE_RESCAN=1` forces a full rescan on every
call. Single-pass hashing always runs; an oracle test proves it matches.

**Tests.** In `./ds4_test --server`: all existing lookup and eviction tests
unchanged (they zero-initialize `kc`, so `index_valid=false` forces the first
scan); a randomized oracle with the old algorithm copied into the test, 200
stub entries with random prefix lengths plus decoys with other model, quant,
ctx or text, identical results; index tests: files added after the first
lookup are found, an external unlink disappears, an own store does not
rescan (test counter), touch updates hits in memory, eviction picks the same
victim as a rescan.

**Measurement and acceptance.** Add `lookup=%.2f ms entries hashed_bytes` to
the KVCACHE hit/miss log; benchmark 20, 200 and 1000 stubs against a 500 KB
prompt, old versus new. Keep if results are identical and time is saved.

**Pitfalls.** Tests access `kc.entry[idx]` directly, so keep the array.
Another process touching headers does not change the directory mtime (affects
eviction scores only). Keep all index changes on the caller side (item 23
step 6).

**Stop if.** The index ever returns a stale result, or it saves under 1 ms at
realistic directory sizes; then keep only the single-pass hashing.

### 25. Delta (chained) checkpoints (large; design sketch)

**Premise corrections.** Sizes match `ds41_state_spans` (`ds4.c:63794-63810`):
payload = 52 + 4·pos + 4·129,280 + 40·min(pos,128)·2048 + Σ live_i·2560, plus
12,288 when pos is odd. Append-only rows cost 6,400 B/token, so a 2048-token
chunk is 13,107,200 B (1,024 rows for owners 0-2, 2,048 for owner 3); chunk
boundaries align with row boundaries. The mutable part is ~11.0 MiB. The
continued interval rounds up to 10,240 (`kv_cache_continued_step` 736-745).
One 130K lineage writes ~5.25 GB of full waypoints versus ~0.9 GB as deltas.
The default 4096 MiB budget cannot hold today's waypoints; the
superseded-prefix rule (546-563) evicts them. Restore reads the same bytes, so
there is no load speedup. Anchor, evict and shutdown positions are generally
unaligned. "Append-only" holds only until a rewind, invalidate or load.

**Goal and expected gain.** About 10× fewer bytes per late waypoint (~77 MB
instead of ~798 MB at 122,880 tokens), ~5× less write volume per session, and
many more waypoints in the same budget.

**Why outputs are unchanged.** A payload materialized from its chunks plus
tail must be byte-identical to `ds4_session_save_payload` at the same state;
the loader writes the same bytes into the same tensors.

**Worth doing only if** all hold: after item 23, background writes measurably
slow decode or prefill (>1%) or eviction logs show useful waypoints lost at
the user's budget or write volume matters; row stability is proven (S1); long
linear sessions dominate.

**Decision points.**
- D1 scope: V4.1 first; shape the engine API as generic "append-only spans +
  tail".
- D2 identity: never share chunks by token hash alone (prefill and decode
  rows can differ in bits). Chunks are immutable, uniquely named
  (`<chainhash>-<rand64>.kvc`); a session reuses a chunk only if it wrote or
  loaded it and has not rewritten rows below it since. Chain hash =
  SHA1(prev ‖ model, quant, ABI ‖ chunk tokens), stored in the chunk header.
- D3 granularity: 2048 tokens, the exactness grid.
- D4 container: one file per chunk (per-lineage pack files break on branch).
- D5 reference counting: mark-and-sweep from checkpoint headers in the item 24
  index, no persistent counts; sweep at open and after evictions; chunks of
  pending jobs are protected.
- D6 budget: Σ tails + Σ unique chunks; eviction score uses marginal bytes.
- D7 format: `KV_CACHE_PAYLOAD_ABI 3` (old binaries ignore it,
  `ds4_kvstore_read_header` 429); payload = "DSVD" magic, version, k,
  k × (name, chain hash, bytes), then the tail (same 13-u32 header, all tokens,
  logits, windows, rows in [k·2048, pos), `previous_*`).
- D8 crash consistency: chunks tmp→rename, then checkpoint tmp→rename; delete
  the checkpoint before sweeping its chunks; a missing or invalid chunk
  discards the checkpoint. No payload checksum (parity with today).
- D9 engine hook: `uint32_t ds4_session_take_rows_floor(ds4_session *)`
  returns the lowest position rewritten since the last call; lowered by
  invalidate, rewind, load, `rewrite_from_common` and a sync with common <
  pos, not by speculative writes above the committed position.

**Staged plan.** S0 measure after item 23 and decide. S1 engine test for row
stability (after continued prefill, decode, DSpark and rewinds above P, rows
below P are byte-equal) plus the rows-floor API and a materialize hook. S2
engine-owned chunk/tail serialization (`ds4_session_save_chunk`,
`ds4_session_save_tail`, `ds4_session_load_chunked`) verified materialized ==
`save_payload` at 2048, 4097, 10240, 32703, 32768 on the fixture and the real
model. S3 kvstore write path with `chunks/`, ordering and sweep. S4 index
references and marginal-byte eviction. S5 load path with discard fallback. S6
live validation (restore at each waypoint versus cold, logits and bytes) and
crash tests (`kill -9` mid-chunk and mid-checkpoint).

**Gating and rollback.** `DS4_KVSTORE_MONOLITHIC=1` writes ABI-2 files;
readers keep both formats; other model families always use monolithic files.

**Tests.** S1, S2, S6, plus randomized eviction and GC cases in the server
unit group.

**Measurement and acceptance.** Bytes written per session, waypoint hit rate
at 4 GiB, decode during writes. Keep if materialized bytes are identical and
write volume drops at least 4×.

**Pitfalls.** Lineage bugs corrupt state silently, so S2's byte test is
mandatory. Up to 64 chunk files open at 130K. Pair with item 26's F_NOCACHE.

**Stop if.** S1 finds any unstable row, or S0 shows no pressure.

### 26. Memory-pressure admission and hygiene; F_NOCACHE for kvstore I/O

**Premise corrections.** Nothing reads pressure dynamically (no
`memorystatus`, `host_statistics64` or dispatch memory-pressure source). There
are two static budgets: the GLM guard (`ds4.c:45555-45617`, `hw.memsize` at
45620-45628, `iogpu.wired_limit_mb` at 46058-46068) and, for V4.1,
`ds41_memory_admit_for_host`/`ds41_memory_admit` (71808-71854) with budget =
min(7/8 × memsize, `recommendedMaxWorkingSetSize`) (`ds4_metal.m:4792-4796`),
checked at engine open (72240, 72710) and each `ds4_session_create` (74268).
Linux has `ds4_linux_memory.h` (tested by `test-linux-memory`). Two review
premises were wrong: ds4-server creates every session at startup
(16000-16010), so there is no per-request session allocation to admit; and
"flushing" an idle slot frees nothing (`ds4_session_invalidate`,
86738-86757, keeps the buffers). What can be released: item 23's staging and
the image cache (≤128 MiB, 10045). Engram tables are opened with `F_NOCACHE`
and `F_RDAHEAD=0` (`ds4_engram.c:106-108`) and are outside the weight mapping
(`ds4.c:2761-2797`), but reads still hit resident pages. kvstore I/O today
pushes about 3× the payload through the page cache per store and 1× per load.
Cold versus warm decode is 25.70 versus 25.83 tok/s after the host pipeline,
so ~0.5% is the most this can gain for decode.

**Goal and expected gain.** Keep checkpoint I/O from evicting Engram pages
(up to ~0.5% decode), avoid growing memory under pressure, and return a clean
503 at critical pressure.

**Why outputs are unchanged.** F_NOCACHE changes only caching; admission
rejects only requests that have not started; the synchronous fallback writes
identical files.

**Code map.** As above, plus `enqueue` 14940 and the existing 503 at
15383-15384; `http_error` 6932 and `http_response` 6906 (no extra-header
support); `server_image_cache_clear` (under `inference_mu`); kvstore opens at
`ds4_kvstore.c:1064` and 1243.

**Implementation steps.**
1. `ds4_host_memory.h`, static inline like `ds4_linux_memory.h`:
   `ds4_host_memory_pressure()` via
   `sysctlbyname("kern.memorystatus_vm_pressure_level")` mapping 1/2/4 to
   NORMAL/WARN/CRITICAL, else UNKNOWN; `ds4_host_available_bytes()` via
   `host_statistics64(HOST_VM_INFO64)` with Splash's estimate as a pure,
   testable function (free − speculative + external + purgeable + compressor
   savings when not critical; `MemoryGovernor.cpp:20-80`). Linux returns
   UNKNOWN, so CUDA builds are unchanged.
2. Poll only at decision points (microseconds); no dispatch source for now.
3. Staging admission (item 23): asynchronous only at NORMAL/UNKNOWN pressure
   with available ≥ image + 4 GiB; otherwise the synchronous direct write. Log
   once per state transition.
4. Request admission: before `enqueue`, at CRITICAL return 503 "server under
   memory pressure" with `Retry-After: 5` (needs an `http_response` variant
   with extra headers). Never touch running jobs.
5. At job start (worker holds `inference_mu`), at ≥ WARN clear the image cache
   and call `ds4_kvstore_flush`. Do not invalidate slots; releasing idle KV for
   real (session free/re-create or Metal purgeable state) is a separate design.
6. F_NOCACHE on Apple: `fcntl(fd, F_NOCACHE, 1)` in the item 23 writer with
   16 KiB-aligned offsets and buffers; on `fileno(fp)` in the synchronous path
   with `setvbuf` at 1 MiB; on the load fp in `try_load`, keeping read-ahead
   on. Leave the distributed `/tmp` stage cached (read back immediately).
7. At startup, log available memory against the planned budget.

**Gating and rollback.** `DS4_KVSTORE_CACHED_IO=1` skips F_NOCACHE;
`DS4_SERVER_IGNORE_MEMORY_PRESSURE=1` makes the probe return UNKNOWN.

**Tests.** Unit test of the estimate function in the `test-linux-memory` style
(synthetic counts, speculative > free returns 0, overflow); a smoke test that
the sysctl returns a valid level; server unit tests with a test hook forcing
the level (CRITICAL → 503 on a socketpair; WARN → synchronous stores, no
writer thread); files identical with and without `DS4_KVSTORE_CACHED_IO`;
`mmap` + `mincore` on a fresh checkpoint shows few resident pages.

**Measurement and acceptance.** A small `mincore` diagnostic over the two
Engram ranges (offsets from `warm_engram.py --dry-run`; `mincore` does not
fault pages in), before and after N stores and loads; decode tok/s after the
stores, warm, with `DS4_METAL_V41_DECODE_HOST_PROFILE=1`;
`memory_pressure -S -l warn|critical` to exercise admission. Keep if
residency is preserved, store/load times regress at most 5%, and no 503
appears in normal runs.

**Pitfalls.** Evict-then-load of the same file now reads from SSD. Unaligned
F_NOCACHE writes fall back to cached copies. The pressure level is
system-wide. A 503 changes API behaviour: CRITICAL only.

**Stop if.** Any output or file byte changes, spurious 503s appear, or
residency does not improve while store/load times regress.

## G. Correctness audit: cross-threadgroup handoffs

**Why this is here.** Splash commit f04ad4e found that split-K partials
written through plain `device` pointers and read by another threadgroup were
sometimes stale on the two-die M3 Ultra; declaring them `coherent(device)`
fixed it. ds4 has several kernels where one threadgroup publishes data that
another threadgroup of the same dispatch reads. Their status:

| # | Site | Runs by default on M3 Ultra? | Risk |
| --- | --- | --- | --- |
| 1 | `kernel_dsv4_hc_rms_norm_mix_{bf16,f16}_cluster2_pre_norm` (`metal/dsv4_hc.metal:1396`; producer stores 1434/1490; arrival 1569-1602; consumer reads 400-419 through `device volatile`) | Yes: GLM BF16 (~90 dispatches per token) and V4 F16 before M5. Not V4.1. | Medium-high |
| 2 | `expand4` (1631-1693): plain `device` handoff, spin capped at 1e8 iterations and then silently proceeds | M5 TP only | High |
| 3 | `kernel_dsv4_router_project_select_fused` (`metal/dsv4_misc.metal:6068-6224`) | M5 only | High |
| 4 | GLM router atomics (`metal/dsv4_misc.metal:5595-5632`) | Yes | Safe; the pattern to copy |
| 5-10 | Remaining cross-threadgroup sites | — | None found |

**MSL rules that matter.** `coherent(device)` exists from MSL 3.2.
`volatile` stops the compiler caching the value but says nothing about
hardware coherence between threadgroups. Atomics are coherent. ds4 never
sets `languageVersion`, so it compiles at the newest version the OS offers
(check that `coherent(device)` compiles before relying on it, and guard with
`#if __METAL_VERSION__ >= 320`).

**Fix patterns.**
- A. Atomic publish: store each handed-off word with
  `atomic_store_explicit(…, as_type<uint>(v), memory_order_relaxed)` and read
  it with `atomic_load_explicit` then `as_type<float>`; keep the existing
  arrival protocol. Exact (bits move unchanged). Use for #1.
- B. `coherent(device)` buffers plus device-scope fences (Splash's
  `split_reduce.h`). Use for new split kernels such as 14a.
- C. Two dispatches: the encoder boundary orders them. Always correct; costs
  a dispatch.

**Stress test first.** New `tests/test_metal_coherence.c`
(`make test-metal-coherence`, not in the default exact tests because of run
time): one producer/consumer kernel in four modes (PLAIN, VOLATILE, ATOMIC,
COHERENT) with G ∈ {1, 16, 160, 640} threadgroups and S ∈ {2, 4, 6} producers
per consumer, 100,000 dispatches each, counting stale reads. The expected
result is stale reads in PLAIN (and probably VOLATILE) on large G and none in
ATOMIC and COHERENT. Record it under `speed-bench/ds41f-m3ultra-perf/`.
Whatever it shows, also run production soaks of the affected kernel (GLM
decode 4,096 tokens, bitwise against the non-cluster path).

**Order.** (1) the stress test; (2) fix #1 with pattern A, bitwise against
the current kernel and the non-cluster path, then a GLM decode screen (the
fix must cost < 0.3%, otherwise use pattern C where it is cheaper); (3) use
B or C in item 14a; (4) hand #2 and #3 to whoever owns the M5 TP path,
with the stress-test evidence, rather than changing them here.

**Treat #1 as a correctness fix, not an optimization.** It goes first in the
suggested order, is committed on its own, and needs no speed gain to be kept.

### Incidental finding: Apple stub row-count mismatch

The Apple stub at `ds4.c:266` (inside the `#if defined(__APPLE__)` block
starting at 163) accepts any `n_rows`, but the real implementation supports
only one. On Metal every caller is compiled out or CUDA-only, so it is
unreachable today. Fix for safety: `if (n_rows != 1) return 0;` with a
comment saying why. No test beyond the build is needed.

## H. Do not adopt

These were considered and rejected. Do not reopen them without new
evidence.

- **Non-exact numerics as defaults:** BF16 activation tables, the 128+q Q4
  offset trick, split-K or reassociated kernels, and any kernel choice that
  depends on the live row count (it breaks decode == verify). Item 22 is the
  only non-exact idea kept, and only as an opt-in probe.
- **Splash's host model:** one serial encoder and one command buffer in
  flight. ds4's pipelined decode step is already ahead of it.
- **Memory and cache designs:** the 10-minute residency unwire; repacking
  GGUF blocks into planes (it only serves the non-exact kernels); INT8 KV;
  32-token prefix blocks (off ds4's 2,048-token grid).
- **A BF16 router:** Splash measured that BF16 router scores change the
  top-8 expert set on 12-15% of rows
  (`runtime/metal/kernels/shared/moe.metal:290-298` in `~/splash`). That is
  evidence against the BF16-router idea from the MXFP4 work.
- **Survivor-threshold top-k as a drop-in:** not exact in general (see item
  13's premise corrections); ds4's histogram path already is the exact
  version.
- **`kernel_moe_pack_rhs` as a drop-in for item 11:** it is not one; use the
  plain F32→F16 copy pre-pass with the exhaustive conversion oracle.

## Suggested order

1. **Correctness first:** the coherence stress test, then audit fix G1 (GLM
   HC handoff → atomic publish), committed on its own.
2. **Cheap exact decode and prefill wins:** 1, 2, 9, 15 (15's NR0 twins wait
   for 2's sweep).
3. **Larger exact decode work:** 3 (sub-steps A → F, each kept or dropped on
   its own), then 7 on top of 3A; 4, then 8; 5; 6.
4. **Prefill follow-ups:** 10, then 11; 12 at any time (no code).
5. **GLM:** 14a, 14b, after G1.
6. **DSpark:** Step 0 diagnostics, then 18 → 16 → 15 (if not already done) →
   17 → 19 → 20 → 21.
7. **Decide item 22 with the user** before any large DSpark kernel work: if
   the probe says GO and the user approves an opt-in mode, parts of 15-21
   change shape.
8. **Server:** 23 and 24 are independent of the model work and can run in
   parallel; 25 only after both; 26 at any time.
9. **Long context (13):** measure model-free first; model runs above 32,768
   tokens need the user's approval.

When an item stops (its stop condition fires), record the measurement under
`speed-bench/ds41f-m3ultra-perf/` and move on; do not stack speculative
changes to rescue it.

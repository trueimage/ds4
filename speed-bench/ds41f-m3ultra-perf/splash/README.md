# Splash implementation campaign — 2026-09-26

The retained changes improve DeepSeek decode by **2.6–3.2%** and reduce
checkpoint-related TTFT on both targets. Bulk prefill has no meaningful gain;
GLM compute throughput stays within 0.1% of baseline. Proposals that failed
exactness or the guide's performance gates were removed from production.

Only `DeepSeek-V4.1-Flash-Q4.gguf` and `GLM-5.3-Flash-Q4_K.gguf` were used.
Every model allocation and advanced context stayed at or below **32,768**
tokens. GPU workloads ran serially on an Apple M3 Ultra (80 GPU cores, 512 GiB).
No drafter GGUF was added. The starting dirty tree was preserved as baseline
`280366e8553da111e0f7b1f63c90755b124ba9ff`; the final measurement tree is
`f768551d842942f5eed4b257f6c997f8eb5ec29e`. The user's branch and index were
not replaced by these detached measurement snapshots.

## Measured results

Uninstrumented `ds4-bench` processes ran baseline/final/final/baseline.
Rates below are harmonic means of the two runs per arm. This is an
**incremental prefill sweep**: the work column gives newly prefilled tokens,
not the full frontier. Raw runs are in `combined-results.json`.

| Model | Frontier | Prefill work | Prefill tokens/s, base → final | Decode tokens/s, base → final | Decode gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek | 512 | 512 | 294.51 → 291.26 | 29.02 → 29.93 | +3.15% |
| DeepSeek | 2,048 | 1,536 | 358.71 → 356.83 | 29.30 → 30.06 | +2.59% |
| DeepSeek | 8,192 | 6,144 | 374.70 → 375.08 | 28.81 → 29.72 | +3.16% |
| DeepSeek | 32,703 | 24,511 | 690.68 → 689.27 | 28.25 → 29.10 | +3.01% |
| GLM | 2,048 | 2,048 | 512.97 → 512.87 | 26.85 → 26.85 | +0.00% |
| GLM | 4,096 | 2,048 | 438.35 → 438.17 | 26.71 → 26.72 | +0.02% |
| GLM | 8,192 | 4,096 | 438.17 → 438.03 | 26.64 → 26.63 | -0.04% |
| GLM | 16,384 | 8,192 | 434.99 → 434.96 | 26.50 → 26.48 | -0.09% |

DeepSeek's short-prefill aggregate differences (−1.10% at 512 and −0.52% at
2048) are inside the spread between the two baseline runs; the closing baseline
matches the final runs closely. Isolated interleaved router screens give
+0.16% at 512 and +0.12% at 8192. These do not establish a meaningful bulk
prefill improvement. GLM's prefill differences stay within 0.05%.

The focused HTTP case uses the default checkpoint policy: the 10,000-token
interval rounds to **10,240**. The cold request contains 10,166 tokens,
generates 64, and its continuation adds 10. Logs confirm a continued checkpoint
is written before the first response token. Each model runs staged/direct/
direct/staged, with identical output, usage and checkpoint images.

| Model | TTFT ms, staged → direct | TTFT saved ms | Checkpoint store ms, staged → direct |
| --- | ---: | ---: | ---: |
| DeepSeek | 314.48 → 305.71 | 8.77 | 19.90 → 9.55 |
| GLM | 386.33 → 326.74 | 59.59 | 112.55 → 55.90 |

See `checkpoint-http-results.json` for both independent measurements and
stream-chunk interval p99. The saving repeats in both pairs and exceeds the
within-arm spread. This comparison isolates direct checkpoint writing; both
arms use the final compute and prefix-lookup code.

The broader HTTP sweep uses the same write-path ordering. Counts below are
actual API prompt counts: the CLI-generated 8192/32000 fixtures render as
8186/31994 through the API. The live continuations do not land on a continued
checkpoint boundary, so their times are not evidence of checkpoint savings.

| Model | API prompt / request | Staged TTFT ms | Direct TTFT ms | Saved ms |
| --- | --- | ---: | ---: | ---: |
| DeepSeek | 8,186 / cold | 22079.96 | 22108.24 | -28.28 |
| DeepSeek | 8,260 / live continuation | 321.23 | 289.27 | +31.96 |
| DeepSeek | 31,994 / cold | 43740.57 | 43882.02 | -141.45 |
| DeepSeek | 32,068 / live continuation | 306.09 | 306.12 | -0.03 |
| GLM | 8,186 / cold | 18124.79 | 18081.78 | +43.01 |
| GLM | 8,260 / live continuation | 295.99 | 266.32 | +29.67 |
| GLM | 31,994 / cold | 73743.79 | 73576.15 | +167.64 |
| GLM | 32,068 / live continuation | 281.19 | 281.24 | -0.04 |

GLM's cold-request saving repeats at both sizes. DeepSeek cold TTFT varies by
more than the removed store time, so no cold-TTFT improvement is claimed for
that model. Raw runs, exact output/usage checks, file hashes and stream-chunk
interval p99 are in `http-results.json`. Intervals measure client-visible
nonempty SSE chunks, which need not be one token each.

## Exactness and validation

- **Starting-tree comparison:** 260 DeepSeek vocabulary rows (33,612,800 FP32
  values) through position 32,767 and 516 GLM rows (79,918,080 values) through
  position 16,512. All **113,530,880 values** and **eight complete snapshots**
  match without tolerance, including continued prefill after restoration.
- DeepSeek snapshot frontiers: 512/2048/8192/32703. GLM: 2048/4096/8192/16384.
  Logit hashes are `8e06b20e…cdc0e` and `344488b3…4323d03`. See the final
  `*-coherent-logit-comparison.json` and `*-coherent-snapshot-comparison.json`
  records for full hashes and snapshot lengths.
- **Independent HC reference:** the retained coherent producer matches the
  separate HC chain for a 4096-token GLM soak: 637,021,440 additional logit
  values and 4,112 selected IDs. Counters report 370,080 executions on each
  path. This is independent reference evidence, not a starting-tree capture.
- Short and approximately 8K greedy outputs match byte for byte on both
  models (limits 400 for DeepSeek, 128 for GLM). No 64K prompt was used.
- All six new/expanded GPU oracle suites pass under `MTL_DEBUG_LAYER=1`,
  including exceptional values, guards, tails, fallback/admission and dispatch
  coverage. Live-state error/recovery, queued scheduling and gather reuse
  pass. Target-only verify parity passes every row and retained prefix at all
  nine test cases, including snapshots and subsequent continuation.
- SSD public-session/recovery checks pass. Both models pass staged/direct
  checkpoint save/load comparisons at 512 and 2048 in SSD mode. Resident
  checkpoint comparisons also cover 8193 and 32703. Full images match after
  masking **only creation/touch timestamps at bytes 24–39**; saving preserves
  state, and restored tokens and complete snapshots match exactly.
- All 48 HTTP requests complete. Both write paths produce identical logical
  output, usage, and timestamp-normalized checkpoint images in every four-run
  group; successful shutdown leaves no checkpoint temporary files.
- Ordinary builds, CPU core/server object builds, bounded unit/server tests,
  model-free TP transport tests and 18 cached-switch refusal checks pass.
  `validation.json` records the final source hashes and completed checks.

The full default `make test` contains a hard-coded 100,000-token model session.
Its model-free and applicable bounded checks were run separately to honor the
user's limit. CUDA, distributed inference and M5 hardware were not tested.
The DeepSeek compute admissions preserve the existing quality, SSD, TP,
vision and imatrix exclusions. The M5 TP HC handoff is unchanged.

## Decisions

| Item | Evidence | Status |
| --- | --- | --- |
| G1 HC handoff | Coherent transfers on MSL 3.2+, atomic fallback; 4.8M stress dispatches with zero stale reads; F16/BF16 exceptional/guard oracles pass | Keep: 0.075% cost in 1024-token ABBA; final coherent capture and 4096-token independent soak exact |
| 1 Gather reuse | Two 8K and two near-32K screens, full logits and selections exact; +0.60–0.71% decode | Final DeepSeek capture passes; combined decode +2.6–3.2% |
| 2 Q8 rows per group | 840 exact oracle dispatches; real Q4 weight sweep, no shape reaches 3% | Rejected; archived patch |
| 3A Attention epilogue | 384 GPU cases exact; repeated 8K/near-32K screens +0.48–0.58% | Final DeepSeek full-logit/snapshot capture passes |
| 3B/C Attention prepare/compression | Depend on in-place RMS/BF16 fusion; its executable reference disagreed | Stopped at exactness dependency |
| 3D Index-Q prepare | 240 exact GPU cases; two 8K screens +0.070/+0.029% | Rejected: below 0.2% twice |
| 3E BF16 norm store | Separate-output oracle passes admitted sizes; expanded in-place fixture fails | Rejected; archived patch |
| 3F q_b BF16 store | 180 matvec oracle cases; two 8K screens +0.59/+0.64%, two near-32K screens +0.45/+0.57%, every logit exact | Final DeepSeek capture passes; combined decode +2.6–3.2% |
| 5 Router top-six | 320 exact GPU cases, 182 fast accepts/138 fallbacks; repeated 8K +0.98/+0.87% | Prefill +0.16% at 512, +0.12% at 8192; final DeepSeek capture passes |
| 6 Norm load ordering | Synthetic oracle passes; full-model logits differ at 512 and 8K | Rejected; archived patch |
| 7 Packed/fold attention | Packed: −0.20 to −0.41%; fold: 10,000 GPU cases and model logits exact, but −6.6 to −7.0% decode in four screens | Both rejected |
| 10 Eight-row Q4 cull | Exact intermediate/output oracle; repeated 512 prefill about 3.4% slower | Rejected; archived patch |
| 11 Half input to pair | All 2³² F32 conversions exact; exceptional Q4 intermediates/output exact; 512 gains +0.55/+0.48%, 2048 +0.42/+0.41% | Rejected: no 0.5% at two sizes |
| 12 Split versus fused pair | Four ABBA/BAAB repeats at 512/2048/8192; full logits exact; split is 11.18/4.51/3.02% slower | Retain existing fused pair |
| 13 Long-context indexer | User limits this pass to 32K | Closed by scope/stop condition |
| 14a GLM decode prefetch/split | Prefetch: −0.002/+0.044%; split2: +0.046/+0.074%; split4: +0.104/+0.077%; every full-model logit exact; 4096-step poisoned-partial split soak exact | Rejected: under 0.3% model gain and under 20% kernel gain |
| 14b GLM prefill staging | Three tile choices pass GPU oracle; recurrence totals 158.94–214.92 ms versus 126.79 ms reference at 2K | Rejected; archived patch |
| 15 Exact rows and verify batches | 252 Q8 plus 56 output-A oracle cases; 6-row q_b 9.61→3.89 ms; all kept-prefix verification parity cases pass | Keep Q8 rows, output-B, head for rows 4–8, output-A for rows 4–6; reject F32 router batching below its 2% gate |
| 23 Direct checkpoint writes | All file bytes apart from timestamps, restored tokens and snapshots exact on both targets at 512, 2048, 8193, 32703 | Keep Stage 0; asynchronous stage stopped at the post-Stage-0 copy-cost gate |
| 24 Prefix hashing | Old/new lookup and repeated eviction agree on randomized fixtures; 20/200/1000-entry 500KB lookup: 8.32–18.80 ms → 1.05–1.48 ms | Keep single-pass hashing; directory caching deferred to preserve external in-place header updates |
| 25 Delta checkpoints | Four measured checkpoints total below the default 4 GiB budget on either model; >32K excluded | S0: no capacity pressure demonstrated in this scope |
| 26 Memory pressure / I/O | Estimate overflow, injected critical admission and warning image release pass; F_NOCACHE files/state exact but writes substantially slower | Keep admission/hygiene; reject F_NOCACHE |

MXFP4-only items 4/8/9 do not apply to the two named Q4 files. Item 22 changes
numerics and was excluded by the exactness requirement. Item 13 stopped at the
user's 32K limit. Drafter-dependent items 16–21 were excluded; item 15 was
validated through the target-only verification interface.

## Important implementation details

The attention epilogue shares the reference's noinline reduction and retains
the original runtime inverse-RoPE expression. Hard-coding its sign changed
compiled arithmetic and failed the oracle. Original RoPE and RMS reference
kernels were not changed to make a test pass.

The Q8 rows rewrite preserves each row's accumulation and SIMD reduction
order while sharing weights and a single publication barrier. Six-row q_b
falls from 9.61 to 3.89 ms for 40 layers. Head batching is retained for rows
4–8, output-A for rows 4–6, and output-B for rows 2–8. Repeated isolated
verify-time savings clear the 2% gate at the retained sizes; smaller gains and
F32 router batching were rejected. No end-to-end DSpark speedup is claimed.

HC publication uses `coherent(device)` vector access on MSL 3.2+ and atomic
bit transfers on older MSL. The atomic-only prototype cost 0.35% in the longer
2K-context comparison; the separate HC chain was about 5% slower. Coherent
access passes the same exceptional F16/BF16 oracles and costs **0.075%** in
1024-token ABBA runs (26.86 baseline, 26.84 final in both pairs). The general
4.8-million-dispatch stress test found zero stale reads in every mode,
including plain and volatile: it did not reproduce the hardware failure.

Stage 0 checkpoint writes serialize a known local payload directly into the
unpublished temporary file and verify its exact length before publishing.
Unknown-size/distributed payloads retain staging. At 32703 tokens, resident
DeepSeek stores about 220.5 MB in 26 ms instead of 57–73 ms; GLM stores about
936.3 MB in 138 ms instead of 262–265 ms. After Stage 0, GLM's warm snapshot
copy costs 74–75 ms versus 136–141 ms for the complete direct store, exceeding
the guide's half-store copy-cost stop condition. There is no asynchronous
writer, pending-entry index or append-only engine pinning in this change.

Prefix lookup finalizes copies of one running SHA state at eligible lengths,
then retains original directory order and tie rules. Eviction reuses computed
scores. Randomized lookup and repeated-eviction oracles agree with the old
logic. A cached directory index was deferred because external in-place header
updates need not change directory mtime; rescanning preserves their visibility.
Four measured checkpoints fit within 4 GiB on either model, so delta storage
stopped at S0 without demonstrated capacity pressure in this scope.

Critical memory pressure rejects new queued work with 503 and `Retry-After: 5`;
warning pressure releases disposable image cache entries under the inference
lock. Existing requests retain their owned embeddings. Pure integer estimate
and injected admission tests pass. `F_NOCACHE` removes fresh checkpoint page
residency but raises 32K writes from about 26→42 ms for DeepSeek and 139→214 ms
for GLM, failing the 5% latency budget; buffered I/O remains.

## Diagnostic rollbacks

| Diagnostic rollback | Scope |
| --- | --- |
| `DS4_METAL_DISABLE_V41_GATHER_REUSE=1` | Restore per-layer gathers in scalar decode |
| `DS4_METAL_DISABLE_V41_QB_BF16=1` | Restore separate q_b BF16 conversion |
| `DS4_METAL_DISABLE_V41_ATTN_EPILOGUE=1` | Restore separate attention BF16 and inverse RoPE |
| `DS4_METAL_DISABLE_V41_ROUTER_TOP6=1` | Use the full router sorting network |
| `DS4_METAL_DISABLE_V41_SHARED_ROWS=1` | Restore independent Q8 matvec rows |
| `DS4_METAL_DISABLE_V41_VERIFY_HEAD_ROWS=1` | Restore scalar verify heads |
| `DS4_METAL_DISABLE_V41_VERIFY_OA_ROWS=1` | Restore scalar verify output-A |
| `DS4_METAL_DISABLE_V41_VERIFY_ATTN_OUT_ROWS=1` | Restore scalar verify output-A/output-B |
| `DS4_KVSTORE_STAGE_PAYLOAD=1` | Restore temporary-file payload staging |
| `DS4_SERVER_IGNORE_MEMORY_PRESSURE=1` | Treat the pressure probe as unknown |

`DS4_METAL_DISABLE_GLM53_HC_PRODUCER_FUSE=1` selects the separate HC reference
when set **before engine startup**. It does not restore volatile publication.
Both ordinary variant benchmarks now reject the nine cached GLM switches
because changing them after initialization cannot select a second path.

## Evidence and corrections

`records/` retains commands, source hashes, oracle output, CSVs and server
logs. `experiments/` retains rejected patches, absent from production dispatch.
A rollback is the candidate in isolated feature screens, so a positive
candidate delta means the proposed default was slower. `manifest.json`
identifies model lengths and **8-MiB prefix hashes**, not whole-file checksums.
`run_coherent_final.py` and `http_checkpoint.py` record the final validation
commands. Driver scripts contain local checkout paths; large captures and
checkpoint files remain under `/tmp/ds4-splash-20260926`.

Three evidence issues were corrected and their original records retained:
(1) the empty-session live-edge fixture initially called eval before sync;
`live-edges-fixed` supplies the missing initial sync and passes; (2) an early
fixture-only detached commit omitted the rest of its tree, although tested
working files and binaries were unchanged—its correction is recorded in the
manifest; (3) the original HC toggle test used a cached switch and therefore
compared the fused path with itself. `invalid-reference-comparisons.json`
marks those records. The replacement private comparison object uncaches only
HC and reports actual fused/separate execution counts. Both its atomic and
retained coherent 4096-token comparisons pass.

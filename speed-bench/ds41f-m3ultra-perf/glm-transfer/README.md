# GLM strategy transfer to DeepSeek V4.1 Flash

Follow-up to the first M3 Ultra campaign. The reference for the combined results
on this page is `3c16a1d50b9e8d155f484fa0f1f4cd9540a8d11b`, which already includes
resident decode queueing and Q4 expert-tile culling. The original GGUF, weight
precision, compiler math settings and 32,768-token allocation limit are retained.

The requested order is router, HC, shared expert, independent projections, then
histogram top-k. Trials run serially on the same 80-core M3 Ultra with 512 GiB.
Full-logit comparisons are bitwise, without a tolerance or argmax-only shortcut.

## Component trials

| ID | Attempt | Comparison | Original path → candidate, tokens/s | Decision |
| --- | --- | --- | --- | --- |
| R | Fuse 384-expert probability transform, original bitonic selection and weight normalization | 8K, 256 measured decode steps per arm | 23.9012 → 24.8845 (+4.11%) | Keep |
| H | Fuse previous-mixer HC collapse, BF16 stores and weighted RMSNorm | 8K, 256 steps | 24.8524 → 25.4065 (+2.23%) | Keep |
| S | Fuse Q8 shared gate/up projections and SwiGLU with BF16 boundaries | 8K, 256 steps | 25.3816 → 25.7369 (+1.40%) | Keep |
| P | Pair independent Q8 query-low/KV and F16 compressor KV/gate projections | 8K, 256 steps | 25.7611 → 25.7938 (+0.13%) | Reject: gain too small |
| T | Reuse guarded histogram selection on unmasked indexer layers | Near 32K, 256 then 512 steps | 25.1235 → 25.2121 (+0.35%); 25.1416 → 25.2501 (+0.43%) | Keep: small repeatable gain |

Each screen includes the previously retained candidates. Its control is the
optimized default; the benchmark's named candidate sets that feature's rollback.
The table reverses those labels for readability. Gains are incremental and
must not be added or compared as independent whole-branch measurements.
Decode alternates variant order and session assignment per token, after 16 warmup
steps. Every 256-step screen compares 273 complete 129,280-float vocabulary rows
and 272 greedy selections. Router prefill was unchanged: eight balanced 8K runs
measured 652.8247 → 652.5147 tokens/s (-0.0475%), with all full logits exact.

The HC trial intentionally leaves mixer projection and Sinkhorn separate: V4.1
consumes the preceding sublayer's mixer, unlike the GLM producer fusion. The
prototype initially exposed a signed-zero mismatch on zero residuals; retaining
the accumulator's F32 boundary fixed it before model testing. Its four ordered
accumulations and 1,024-thread RMS reduction remain unchanged.

The shared trial preserves BF16 gate/up inputs to SwiGLU and its BF16 result.
It leaves the shared down-projection, routed sum and HC expansion separate.
The projection trial uses the existing exact pair kernels and leaves all
normalization and BF16 boundaries in place; it is archived rather than enabled.

Histogram selection is enabled only for the early, unmasked decode indexers,
12,288–32,768 scores, top-512, serial resident M3 Ultra, and normal precision.
Later layers contain intentional `-Inf` masks and retain the ordinary sort.
The histogram path rejects nonfinite scores, oversized candidate sets, and
ambiguous ties on the GPU, then runs the original sort/merge chain. The two
model screens accepted 1,040/1,088 and 2,030/2,112 calls; every rejected call
fell back exactly. Its modest benefit is supported by two interleaved screens
and requires no new shader. It should not be interpreted as a broad top-k speedup.

## Combined performance

Uninstrumented processes ran in baseline/final/final/baseline order. Each
process used four corpus frontiers with 64 greedy decode tokens, restoring
the corpus prefix before continuing. Rates below aggregate total tokens over
total time from the two-decimal CSV rates. Prefill measures incremental
intervals, not independent cold-prefix runs. See [combined-results.json](combined-results.json)
and [raw commands, logs and CSVs](records/).

| Context | Prefill interval | Baseline → final prefill t/s | Change | Baseline → final decode t/s | Change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 512 | 272.15 → 274.37 | +0.82% | 23.93 → 25.93 | +8.38% |
| 2,048 | 1,536 | 335.02 → 334.92 | -0.03% | 23.98 → 25.96 | +8.28% |
| 8,192 | 6,144 | 355.31 → 355.48 | +0.05% | 23.77 → 25.68 | +8.06% |
| 32,703 | 24,511 | 661.53 → 662.17 | +0.10% | 23.30 → 25.25 | +8.37% |

Decode improved **8.06–8.38%** on top of the first campaign. Incremental prefill
was essentially flat. These are local observations for one model, prompt and
machine, without confidence intervals; they do not establish results on other
hardware or concurrent serving workloads. The four accepted changes are enabled
only in their admitted M3 Ultra paths, with diagnostic rollback controls.
Mapped weight residency and graph allocation remain unchanged; the histogram
reuses the existing small top-k scratch allocation. Observed swap remained
5.81 MiB before and after the combined sweep.

## Combined exactness

A fresh reference build and the final build captured **260 full vocabulary
rows (33,612,800 FP32 values), all bit-identical**, through position 32,767.
The serialized snapshots at 512, 2,048, 8,192 and 32,703 also match byte for
byte, including the 220,441,392-byte final snapshot. These runs cover prefill,
decode, and continued prefill after restoration. Instrumented capture timings
are excluded from performance results.

`make test-deepseek41-fusions` compares the executable old and new GPU paths:
240 router cases and 120 cases each for HC and full-shaped Q8 shared experts.
It checks intermediate outputs as well as final BF16 outputs, exceptional
floating-point values, untouched router output guards, actual fused dispatches,
and ownership/precision rollback gates. `make test-deepseek41-topk` checks
600 score vectors, acceptance and fallback, and poisoned-state failure recovery.
Its synthetic width of 32,769 proves host refusal; full-model tests never
allocate or advance beyond 32,768 tokens. Both pass under Metal API validation.
Existing V4.1 Metal and GLM router/shared, top-k and Q8 input tests also pass.
The real-model SSD session fixture passes under Metal API validation, including
prefix reuse, snapshot restore, cancellation, bounds and imatrix paths. The GGUF
fixture, `make all`, and CPU-only object build also pass. CUDA and remote TP
hardware tests were not run; the new helper call sites are Apple-only and the
resident optimizations exclude TP and SSD ownership.

### Resident live-edge check

The explicit `tests/test_deepseek41_live_edges` fixture extends those GPU
oracles through resident model sessions. It restores the same near-32K seed
for rollback/default replay and compares live boundary inputs, full downstream
logits, greedy tokens and whole serialized snapshots. Test-only interposition
injects router ties and zero probabilities, HC zero/cancellation inputs, shared
zero inputs, and selector ties, nonfinite scores and candidate overflow. It
checks actual fused dispatches, histogram acceptance and exact GPU fallback,
plus stale-selector recovery and graph error propagation with snapshot rejection
and recovery after restoration.

Quality and SSD ownership refusal checks toggle admission only at the resident
dispatch boundary; they do not run full quality, SSD or TP sessions. The fixture
does not modify the production backend, shaders or GGUF. Its completed run,
branch counters, comparisons, source identity and limits are recorded in
[live-edge-validation.json](records/live-edge-validation.json). Instrumentation
timings are not performance evidence; the uninstrumented sweep above remains
the throughput measurement.

Run from this checkout with the existing local Q4 GGUF, one huge model process
at a time. This fixture has a fixed 32,768-token context and is deliberately
outside `make test`:

```sh
make tests/test_deepseek41_live_edges
MTL_DEBUG_LAYER=1 ./tests/test_deepseek41_live_edges \
  /path/to/DeepSeek-V4.1-Flash-Q4.gguf speed-bench/promessi_sposi.txt
```

## Evidence and reproduction

The component benchmark is `speed-bench/metal_decode_schedule_bench`, using
`speed-bench/promessi_sposi.txt`, context allocation 32,768 and selection included
in the timed step. `../run.py` records the complete command, tuning overrides,
source hashes and process status. Timing runs have no stage profiler or logit
capture instrumentation. Full-logit comparisons happen outside timed steps.

The retained stages are automatic in their admitted paths. To reproduce an
original stage, set its diagnostic rollback control to `1`; unset it for the
optimized default. These controls are exactness oracles, not precision options:

| Stage | Rollback control |
| --- | --- |
| Router | `DS4_METAL_DISABLE_V41_ROUTER_FUSION` |
| HC collapse and normalization | `DS4_METAL_DISABLE_V41_HC_NORM` |
| Shared expert gate/up and SwiGLU | `DS4_METAL_DISABLE_V41_SHARED_FUSION` |
| Histogram top-k | `DS4_METAL_DISABLE_V41_TOPK_FAST` |

Local working evidence: `/tmp/ds41f-glm-transfer-20260919/`.
Frozen reference checkout: `/tmp/ds41f-transfer-base-3c16a1d/`.
The patches in `experiments/` apply independently to the reference commit with
`git apply --unidiff-zero`; each includes its earlier retained prerequisites.
Run binaries from their own source checkout because Metal sources load at runtime.

To reproduce the sweep, use the commands in each record from its recorded
checkout. Build capture binaries with `../build_capture.py` and compare with
`../compare_capture.py`; use ordinary `ds4-bench` binaries for timing. Model and
capture binaries are local artifacts, not committed. `records/final-manifest.json`
records build identity and source hashes. The archived top-k patch captures the
screening stage; production additionally isolates V4.1 admission from GLM flags.
The projection fixture is archived only and is not a production test target.

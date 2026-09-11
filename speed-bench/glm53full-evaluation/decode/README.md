# Full GLM: exact IQ2 decode fusion

This follow-up measures the unchanged local `GLM-5.3-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf` on the M3 Ultra. The baseline is `cc1ab21`, which already includes the earlier IQ2 prefill tail cull. This comparison isolates the new decode change.

## Retained change

The full model previously projected each selected expert into device scratch, then launched a separate eight-expert sum. The new kernel assigns one SIMDgroup to each route, sharing the IQ2 lookup tables across all eight. It preserves each expert's original dequantization, accumulation, SIMD reduction and F32 scaling. Rounded results cross a volatile threadgroup-memory boundary before the original route-order addition sequence. It avoids the expert-output device writes and reads and removes 75 sum dispatches per generated token. The graph's scratch allocation remains unchanged.

The runtime selects this only for resident, serial M3 Ultra inference with 256 experts, eight routes, IQ2 gate/down weights and 6144/2048/6144 dimensions. Quality mode, graph dumps, SSD streaming, tensor parallelism and other hardware keep the existing dispatches. No CPU, CUDA, ROCm, graph or public API source changes are retained.

## Measurements

Final order: baseline → candidate → candidate → baseline. Each arm ran twice with 128 generated tokens at each frontier; all model processes were serial. Profiling, logits tracing and Metal validation were disabled for throughput measurements.

| Context frontier | Baseline generation tok/s | Candidate generation tok/s | Gain | Baseline prefill tok/s | Candidate prefill tok/s |
|---|---:|---:|---:|---:|---:|
| 512 | 18.125 | 18.465 | +1.88% | 174.745 | 174.505 |
| 2,048 | 16.535 | 16.785 | +1.51% | 165.505 | 165.505 |

Prefill is unchanged within observed variation. The 2K prefill measurement adds 1,536 tokens to the restored 512-token prefix; it is not a cold 2K prefill. Context allocation is 4,096. These are local results on one benchmark prompt and one M3 Ultra, not long-context or cross-machine claims. [Raw runs and summary](final-results.json).

## Screened alternatives

- Paired Q8 query/KV projections: 66 complete logit frames matched baseline exactly, with 4,992 paired dispatches confirmed in a separate timeline run. The initial 64-token screen moved 18.14 → 18.17 tok/s at 512 and 16.50 → 16.53 at 2K. That small change did not justify retaining the graph change; the experiment is saved locally for a future screen.
- Eight output rows per fused IQ2 tile: passed the kernel exactness cases, but the initial screen was no faster than four rows (18.44/16.72 versus 18.45/16.78 tok/s). Four rows are retained. These single-run screens rank experiments; they are not confidence intervals. [Screen results](screen-summary.json).

## Profiling

The existing encoder timeline uses GPU timestamps without adding command-buffer waits, but separates dispatch groups into compute passes. It adds serialization, so the numbers describe the workload and should not be read as release throughput.

Across eight sampled decode tokens at each frontier, ordinary Q8 matvecs account for about 24 ms of measured encoder work per token (41% at 512; 38% at 2K). IQ2 gate/up is about 10.5 ms, down is about 5.8 ms and the separate expert sum about 1.1 ms. Q8 remains the largest measured category and a useful next target. [Baseline profile](profile-summary.json).

## Correctness and regression checks

- Final full-logit capture: all 154,880 vocabulary values at prefill and 32 greedy decode steps at both frontiers, 66 frames / 40,888,848 bytes including headers. Every value is finite and the file is byte-identical to baseline. The coverage run records 4,800 fused dispatches, exactly 75 routed layers × 64 generated tokens. [Logit evidence](logits.json).
- `test_glm_full_metal`: the existing 21 exact prefill cases plus 36 new decode cases pass. New fixtures use 16 synthetic experts with eight selected and cover the full projection dimensions, shorter dimensions, output tails relative to the original eight-row dispatch, repeated and highest expert IDs, zero and signed route weights, poisoned outputs, quality mode and the resident fallback inside SSD mode. Comparisons cover consumed activations and outputs; fused gate/up scratch is not an observable output. The three fixtures each record eight fused calls, with no calls contributed by quality or SSD mode.
- Metal API validation passes the full-kernel fixtures. The Metal kernel suite, GLM attention tests and SSD expert-cache / eviction / mapping tests pass; SSD tests record zero fused calls. The final build has no warnings. [Checks](checks.json), [coverage and source verification](validation.json).
- A separate greedy chat run (512-token context, 64-token generation cap) produces the same coherent response on baseline and candidate. [Chat comparison](chat.json).
- The final source and binary hashes match those frozen before throughput runs, and the GGUF size, inode and modification time remain unchanged. [Manifest](manifest.json), [frozen hashes](frozen-final.json).

The final instrumented profile measures the fused down/sum at about 5.8 ms per token, versus roughly 5.8 ms for the old down projection plus 1.1 ms for its separate sum. It contains no decode dispatches of the old down or sum kernels. Only the matching first eight decode positions at each frontier are compared. [Final profile](final-profile.json).

## Reproduce

Build each arm in its own checkout and run from that checkout so the binary loads matching Metal sources. Clear inherited `DS4_`, `MTL_` and `ASTRA_` overrides before setting benchmark controls.

```sh
DS4_BENCH_FORCE_SNAPSHOT=1 ./ds4-bench \
  -m /Users/jw/ds4/gguf/GLM-5.3-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 512 --ctx-max 2048 --step-mul 4 --ctx-alloc 4096 \
  --gen-tokens 128 --csv /tmp/glm-full-decode.csv
```

`DS4_METAL_DISABLE_GLM_FULL_IQ2_DOWN_SUM=1` selects the previous down/sum sequence in the same build. `DS4_METAL_GLM_FULL_STATS=1` prints the actual fused dispatch count; use it for coverage, separately from timing. The existing [capture builder](../build_capture.py) reproduces full-logit comparisons with `DS4_BENCH_TRACE_LOGITS` and 32 generated tokens per frontier.

Full logs, raw timelines, logit captures, discarded patches and the frozen baseline checkout are retained at `/Users/jw/.cache/ds4-bench/glm53full-decode-20260910/`.

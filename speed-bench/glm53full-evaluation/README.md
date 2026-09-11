# GLM full: short-context Metal port evaluation

Branch `glm53full-metal-perf` starts from `origin/main` at `6289c516273979173abbc062209a81dd3706b804`. The donor is `glm53flash-metal-exact` at `6d328a09962289f8b8865eb9fe1624d5ca745792`.

The retained port skips unused half tiles in IQ2_XXS expert prefill. It preserves every active output's existing dequantization and accumulation, and keeps all threads participating in staging and barriers. The runtime selects it only for the resident, serial M3 Ultra path with 256 experts, eight routes, and 6144/2048/6144 dimensions. SSD streaming, tensor parallelism, other hardware and other weight formats retain their existing dispatches. No graph, CPU, CUDA or ROCm inference source changes are needed.

The unchanged local file is `gguf/GLM-5.3-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf`, 211,075,860,864 bytes. Its name metadata is `GLM-5.3`, architecture `glm-dsa`; the engine runs it through its GLM 5.2/full path. It has 78 inference layers plus an MTP layer, 256 experts with eight selected, 6144-wide activations, and IQ2_XXS routed weights. This is different from Flash's `glm5-next` architecture, 4096-wide activations and 288 experts. The file was not requantized or modified.

A subsequent [decode evaluation](decode/README.md) retains an exact IQ2 down/sum fusion and measures a further 1.5–1.9% generation improvement against this prefill-only baseline.

## Results

Final retest: main → final → final → main, two serial processes per arm, with matching source files and no timing instrumentation.

| Frontier | Main prefill tok/s | Port prefill tok/s | Gain | Main generation tok/s | Port generation tok/s |
|---|---:|---:|---:|---:|---:|
| 512 | 150.68 | 174.86 | +16.05% | 18.04 | 18.05 |
| 512 → 2,048 | 158.81 | 165.65 | +4.31% | 16.42 | 16.41 |

[Final raw measurements](evidence/final-model.json) and [unrounded summary](evidence/final-summary.json).

The first row measures a fresh 512-token prefill. The second measures an additional 1,536 prompt tokens after restoring the 512-token session, bringing it to 2,048. It is not a separate cold 2K prefill. Context allocation is 4,096 and the generation limit is 128 tokens per frontier. Generation throughput includes all 128 steps.

These are short experiments on one prompt and one Apple M3 Ultra (80 GPU cores, 512 GiB), macOS 26.5.2 (25F84). They establish a local prefill improvement, with no resolved generation improvement. They do not establish long-context throughput or cross-machine performance.

## Port decisions

| Flash change | Full-model finding | Decision |
|---|---|---|
| Expert prefill tail cull | Same existing matmul template supports IQ2_XXS. Skipping padding work improves short prefill and preserves every captured logit. | Retained for the full IQ2 shape. |
| Single-SIMDgroup top-eight router | Adapted from 288 to 256 experts. Passed 120 exact cases with ties, saturation, infinities and NaNs. Added only about 0.25% generation throughput over the tail-only arm in two runs per arm. | Removed from the final implementation; evidence retained for a longer screen. |
| Router/shared fusion | Hard-coded to Flash's 4096 inputs and 288 experts. Full uses 6144 inputs and 256 experts. Moving shared computation also needs independent intermediate storage. | Requires a separate adaptation and measurement; not imported. |
| KDA input/gate/recurrent fusions and BF16 tuning | This `glm-dsa` model has no Flash KDA blocks or BF16 weights. | Not applicable. |
| Hyperconnection producer/output fusions | Full has no Flash hyperconnection streams. | Not applicable. |
| Exact staged decode attention and prefill QK/head tiles | Flash variants require zero RoPE and different QK shapes. Full has a 64-value RoPE tail; main already has its coalesced GLM-5.2 QK-low kernel. | Cannot enable Flash variants unchanged. |
| Histogram DSA selector | Donor targets pooled top-512 selection with at least 12,288 scores. Full selects top-2048, and this campaign deliberately uses short contexts. | Defer to a separate full-model selector study. |

Running the whole unmodified donor branch initially produced no meaningful speed change on this file: about 18.1 tok/s at 512 and 16.4–16.5 at 2,048, matching main's complete captured logits. Most donor paths are excluded by this model's types and shapes. Importing the entire branch would therefore add unrelated changes without its Flash speedup.

The component confirmation used main → tail → tail+router → tail+router → tail → main, with two runs per arm. Tail-only prefill improved 16.06% and 4.29%; generation moved -0.17% and 0.00%. See [component measurements](evidence/confirmation.json) and [summary](evidence/confirmation-summary.json). The final table above is a separate retest after removing the router experiment, not a pool of both campaigns.

## Correctness and coverage

- The final model capture compares every vocabulary float immediately after prefill and after each of 32 greedy steps, at both 512 and 2,048: 66 frames, 40,888,848 bytes including headers, identical to main. No tracing is enabled in throughput runs.
- `test_glm_full_metal` checks all gate/up/intermediate/down/summed outputs against the existing kernels, poisons buffers before repeated runs, checks finiteness, and covers uneven expert groups, 32/33/47/64/65/129/257 rows, F16/F32 intermediate storage and the resident fallback inside streaming. All 21 cases pass. The dispatch counter records 28 fast calls; streaming cases retain the original path.
- The Metal kernel suite, GLM attention tests and SSD expert-cache tests pass. The SSD tests cover exact cached outputs, eight-expert eviction and mapping lifetime; they record zero new tail-cull dispatches.
- `make -j8 all ds4_test tests/test_glm_full_metal tests/test_metal_ssd_experts tests/test_glm_attention` succeeds without compiler warnings.
- CUDA/ROCm source is unchanged; the added header flag and build target are Apple-only. No distributed hardware campaign or end-to-end SSD throughput claim is made.

The preceding shader-validation failure is shared with main and is outside this change. All model comparisons clear `DS4_`, `MTL_` and `ASTRA_` environment overrides before applying recorded benchmark controls. `MTL_SHADER_VALIDATION` is not enabled.

## Reproduce

Build each arm in its own checkout and run from that directory: Metal source files load at runtime from the working directory. Use the same unchanged model and prompt for both. Clear existing diagnostic overrides first, then run:

```sh
DS4_BENCH_FORCE_SNAPSHOT=1 ./ds4-bench \
  -m /Users/jw/ds4/gguf/GLM-5.3-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 512 --ctx-max 2048 --step-mul 4 --ctx-alloc 4096 \
  --gen-tokens 128 --csv /tmp/glm-full-short.csv
```

For a same-build rollback comparison, set `DS4_METAL_DISABLE_GLM_FULL_MOE_TAIL_CULL=1`. For coverage only, `DS4_METAL_GLM_FULL_STATS=1` prints the new dispatch count at cleanup. Avoid profiling and tracing in timed runs.

```sh
make test-glm-full-metal
./tests/test_metal_ssd_experts
./tests/test_glm_attention
./ds4_test --metal-kernels
```

The [capture builder](build_capture.py), reused from the donor campaign, wraps the public session API and produces `campaign-bench` in the supplied source tree. Build each arm's engine objects first, then run `python3 build_capture.py /absolute/path/to/arm`. The same benchmark command with `./campaign-bench`, `--gen-tokens 32`, and `DS4_BENCH_TRACE_LOGITS=/absolute/path/to/trace.bin` reproduces the full-logit capture.

Compact evidence is in [evidence/](evidence/). Source and binary hashes are in [the final manifest](evidence/final-manifest.json); the model stat and prompt hash are in [the experiment manifest](evidence/ports-v1-manifest.json). Full local logs, traces, frozen baseline/donor trees and the discarded combined patch are preserved under `/Users/jw/.cache/ds4-bench/glm53full-20260910/`.

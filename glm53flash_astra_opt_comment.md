This update fixes a reproduced BF16 HC producer exactness bug and adds six further GLM 5.3 Flash Metal optimizations. **The earlier “over 30% faster, bit-exact” decode claim is superseded by the corrected measurements below.**

The fused BF16 HC producer reused DeepSeek V4's F16 reduction: vector dots across eight SIMDgroups. GLM's standalone BF16 projection uses ordered scalar FMAs within one SIMDgroup per row. Those different accumulation orders changed actual decode logits. Matching prefill-frontier logits or generated tokens was insufficient to establish bit-exact decode.

The fix preserves the BF16 projection's original FMA sequence, SIMD reduction, and explicit F32 rounding boundary after normalization. **Producer fusion remains enabled**, with one dispatch per site instead of four. DeepSeek's F16 path retains its own reduction. Both weight types are now compared byte-for-byte against their respective unfused chains, including mix values, split weights, collapsed activations, and weighted normalization.

A separate inherited GLM router issue was also corrected: normalization reads probabilities and selected IDs written by other threads, so publication now includes a device-memory fence. DeepSeek's routers already provide that visibility. No router failure was reproduced; this was distinct from the demonstrated BF16 mismatch.

The corrected HC fusion retains approximately **5.1%** more decode throughput than disabling it. Before the subsequent improvements, the corrected branch measured **23.34% / 22.55%** faster than `b0a147a` at 2K / 8K, with the same router fence applied to both revisions.

The subsequent improvements preserve the reference arithmetic while changing independent work scheduling:

- Single-SIMDgroup top-eight routing, preserving expert tie order and normalization; NaNs use the original sorting-network behavior.
- 32-column QK-low decode tiles using the unchanged quantized dot helper.
- More independent BF16 output rows per group for the short gate projections.
- Smaller DSA score tiles, preserving the softmax and value-reduction trees.
- Four interleaved independent KDA state rows, preserving each row's recurrence arithmetic.
- One combined BF16 dispatch for Q/K/V, `f_a`, `g_a`, and beta; the dependent second gate projections remain separate.

Final measurements on the **80-core M3 Ultra / 512 GB**, using the original resident `GLM-5.3-Flash-Q4_K.gguf` with BF16 projections:

| Context | Corrected branch before these six changes | Final branch | Additional decode gain |
| --- | ---: | ---: | ---: |
| 2K | 26.050 tok/s | 27.040 tok/s | **+3.80%** |
| 8K | 25.400 tok/s | 26.367 tok/s | **+3.81%** |

Three interleaved runs per arm, 256 greedy tokens at each frontier, `promessi_sposi.txt`, 16K allocated context, profiling disabled, and frozen binaries **and runtime Metal sources**. All generated texts match. These are additional gains over the corrected branch; percentages from separate experiments should not be added. Prefill is effectively unchanged in this pass (+0.12% / +0.15%). Slower or inconclusive prefill and weight-packing experiments were rejected.

Validation goes beyond generated-text equality:

- Full F32 logit traces match byte-for-byte at 2K and 8K in default and `--quality` modes, including snapshot continuation: 258 frames per mode.
- A different text segment at 16K also matches across 64 greedy updates.
- Kernel regressions compare intermediates, all six projections, convolution history, and recurrent state, with dispatch coverage and rollback checks.
- `make -j4 test-glm53-kda`, `./ds4_test --metal-kernels`, and CPU-only syntax checking pass.

The new defaults are limited to resident, single-device M3 Ultra execution at the validated GLM shapes, with individual and aggregate rollback controls. Model weights, precision settings, and compiler math flags are unchanged. Full measurements, trace hashes, and rejected experiments are recorded in `speed-bench/glm53_decode_findings.md`.

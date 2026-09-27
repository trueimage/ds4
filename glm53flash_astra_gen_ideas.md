# GLM 5.3 Flash Metal generation ideas

Recorded 2026-09-04 from a read-only review of ds4 `4b00b59`, local MLX
`b6368984b`, and local llama.cpp `4d9176092`. Keep this file untracked.
The original ideas below were proposals. The dated campaign entries record
which experiments have since been implemented and checked.

## Contract and scope

- Metal and GLM 5.3 Flash generation and prefill (scope expanded by the user
  for the optimization campaign). Other model/backend code is reference
  material; MLX and llama.cpp remain read-only.
- Correctness and quality take priority over speed. Preserve model bits,
  precision, per-output operands and accumulation order, reduction topology,
  conversion/rounding boundaries, expert ordering, and persistent state.
- Preserve the baseline compiler math settings. Metal fast math means that
  matching source expressions alone does not prove matching generated arithmetic.
- Most existing performance evidence concerns the fully resident Q4_K artifact
  with BF16 KDA projections on M3 Ultra. Reprofile the current tip; do not
  extrapolate those measurements to other artifacts, GPUs, or SSD streaming.
- Existing HC producer, KDA pair/trio, HC epilogue, shared-down, exact DSA, and
  prefill optimizations are already present and are not new proposals here.

## First: resolve the inherited router synchronization concern

`kernel_glm_router_select_one` in `metal/dsv4_misc.metal` writes probabilities
and selected IDs to device memory, then reads other threads' writes during
normalization with only threadgroup-memory barriers in between. The sort itself
uses synchronized threadgroup scratch. The concern is device-memory visibility
during normalization; no failure was reproduced during the analysis.

This shared GLM 5.2/5.3 kernel is byte-identical in `110afdd`, `b0a147a`, and
`4b00b59`. The relevant code originated in `005afed`; GLM 5.3 support widened
the sort in `147109a`. The optimization branch did not remove these fences.
DeepSeek Flash's fused routers explicitly use `mem_device` before re-reading
probabilities and selected IDs; its separate kernels use dispatch boundaries.

Accepted diagnostic approach: minimally add device scope to the barrier before
normalization, leaving arithmetic/storage unchanged. Compare probabilities,
selected IDs, weights, and actual decode logits byte-for-byte. Apply the same
correction to historical baseline and branch for interleaved performance runs.
An inherited synchronization issue need not cancel between arms, but there is
currently no evidence that it explains the branch's reported 30-32% decode gain.

## BF16 HC blocker: initial diagnosis (corrected in the follow-up below)

Full decode-logit capture found an existing, separate mismatch in the BF16 HC
producer fusion. At both 2,048 and 8,192 context, branch and `b0a147a` frontier
logits match byte-for-byte, but all 128 subsequent teacher-forced decode frames
differ. The first-step maximum absolute logit differences were approximately
`1.02e-5` and `1.37e-5`, respectively. Matching argmax does not satisfy the
bit-perfect contract. Adding the router fence leaves each revision's own full
trace byte-identical; the fence is not responsible for this difference.

The aggregate rollback switch restores the baseline trace exactly. Independent
four-step checks with each decode optimization enabled in isolation identify
`GLM53_FLASH_HC_PRODUCER_FUSE` as the differing path; the other seven checks
match the baseline. Gate trio and shared-down checks include their prerequisite
pair and FFN-tail switches.

The fused template in `metal/dsv4_hc.metal:1346` uses eight SIMDgroups per row,
grouped `float4` dots, and a second reduction across SIMDgroups
(`metal/dsv4_hc.metal:1400`). GLM's unfused BF16 projection uses the ordered
scalar FMA sequence and one SIMDgroup per row in `metal/glm53_bf16.metal:32`.
Borrowing DeepSeek's F16 arithmetic therefore does not preserve GLM's BF16
reference arithmetic. The existing test at `tests/test_glm53_kda.c:1436`
compares fused F16 and fused BF16 with equivalent weights; it does not compare
against the unfused GLM producer.

Resolve this before accepting the entire default branch as bit-perfect. A
corrected BF16 fusion must retain GLM's original projection reduction and its
rounding boundaries; leave DeepSeek's F16 path alone. The existing
`DS4_METAL_DISABLE_GLM53_HC_PRODUCER_FUSE=1` switch allows an immediate isolated
comparison while retaining the other optimizations. No HC implementation change
has been made as part of the router diagnostic.

With only that fusion disabled, the full branch matches `b0a147a` byte-for-byte
for both 128-step decode sequences, including the restored-session continuation.
The 258-frame trace (two prefill frames plus 256 decode frames) contains
159,840,288 bytes and has SHA-256
`c90004872c8e3bd0536bc3bc9f0af988b7abdbda9e01b21a2fb23007ca836c5f`.
The default branch's largest differences over the same teacher-forced sequences
are `0.00043201446533203125` at 2K and `0.00040531158447265625` at 8K; argmax
matches on every teacher-forced frame, showing that checking argmax alone can
miss the violation. These teacher-forced inputs differ from greedy feedback.
These observations concern the original Q4_K artifact with BF16 projections on
M3 Ultra, not a proof covering every input, artifact, or GPU.

### Router validation and retained performance, 2026-09-04

The minimal router correction adds `mem_device` to the existing barrier before
normalization. The complete GLM GPU test target passes with both frozen original
and corrected shaders. Added router cases cover 256/288 experts, one/17 rows,
ties, cross-SIMD selections, and varied/saturated sigmoid inputs. All 165,888
captured bytes of probabilities, IDs, and weights match between shaders across
144 rows. Both historical baseline and branch also retain their own exact
258-frame full-logit trace after the correction.

Performance: M3 Ultra, original `GLM-5.3-Flash-Q4_K.gguf`, fully resident,
`promessi_sposi.txt`, 128 greedy decode tokens, context allocation 16,384,
frontiers 2,048 and 8,192. Two runs per arm, ordered baseline-fixed /
branch-original / branch-fixed / branch-fixed-HC-off / branch-fixed-HC-off /
branch-fixed / branch-original / baseline-fixed. Timing uses uninstrumented
`ds4-bench`; the separate logit-capture binaries are not used for timings.

| Context | Baseline + fence, tok/s | Default branch + fence, tok/s | Branch + fence, HC fusion disabled, tok/s | Gain of the bitwise-checked configuration |
| --- | ---: | ---: | ---: | ---: |
| 2,048 | 21.140 | 27.770 | 24.765 | +17.15% |
| 8,192 | 20.770 | 27.020 | 24.175 | +16.39% |

On the default branch, original-router means were 27.915 and 26.985 tok/s;
the fence changes these by -0.52% and +0.13%. This short run set does not
establish an exact fence cost, but gives no evidence of a penalty large enough
to explain away the branch's gain. The default branch's +31.36%/+30.09%
measurements fail the strict bit-perfect acceptance criterion because of HC.
The default branch's greedy text also diverges from the baseline at 2K in
every run, starting at output byte 192, while its 8K text matches. The 2K
default-branch timing therefore includes different generated tokens/routing.
Original and corrected router versions produce matching text within the
default branch. Both HC-disabled runs match the baseline's text at both
contexts. That configuration retains a measured +16-17% gain while matching
the full-logit reference on the teacher-forced sequences.

Raw logs, CSVs, frozen shader sources, trace harnesses, and comparison metadata
are in `/var/folders/zw/xt8y4nbj6m3dxyws1yt44_rm0000gn/T/ds4-router-barrier-zm_h9om0/`.
The directory is temporary; the conclusions, configuration, and key trace hash
are recorded here so this follow-up does not depend on its retention.

## Follow-up: exact BF16 HC fusion retained

The BF16 projection inside the compound HC producer now uses one SIMDgroup
per output row, matching `glm53_mul_mv_bf16_f32_row_sum`: the same lane-strided
eight operands, ordered scalar FMAs, and one SIMD reduction. Normalized
activation products cross an explicit volatile F32 boundary before each FMA,
preserving the standalone RMS kernel's stored results under Metal fast math.
The normalization, split/Sinkhorn, collapse, weighted normalization, and
completion protocol remain fused. This still replaces four dispatches with
one at each of 90 sites per token. The F16 specialization keeps DeepSeek's
original eight-SIMDgroup vector-dot reduction.

The revised regression compares each type against its own unfused chain,
checking mix, all split weights, collapsed output, and weighted normalization
with `memcmp`. It covers varied/cancelling inputs, zero residuals, one/20
Sinkhorn iterations, and repeated completion-counter reuse with poisoned
split output. The original shader fails the BF16 mix comparison; the corrected
shader passes both BF16 and F16, plus the full GLM GPU and Metal kernel suites.

Actual decode checks pass with fusion enabled: 128 steps at each of 2K and 8K,
including continuation after snapshot restore, for teacher-forced decoding,
greedy decoding, and `--quality` greedy decoding. The teacher-forced full
trace matches the historical baseline hash above. Greedy fused/unfused trace
hashes match within each mode:

- Default: `3c600ac8994cf3327b2b19e7d2de0d8ad6b0dc261fe7c44eee2358974b0d1bf1`.
- Quality: `fc83d2ab8b6bdb8dca7a4d259f03f3a35e70e77a9ad855d6520ec58c1250bc2a`.

Interleaved uninstrumented measurements (baseline / corrected / HC-off /
HC-off / corrected / baseline, two runs each) retain the fusion's speedup:

| Context | Baseline + router fence | Branch, HC off | Branch, corrected HC | Fusion gain | Whole branch gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2K | 21.125 tok/s | 24.780 tok/s | 26.055 tok/s | +5.15% | +23.34% |
| 8K | 20.730 tok/s | 24.165 tok/s | 25.405 tok/s | +5.13% | +22.55% |

All six greedy benchmark outputs match. Hardware, model, prompt, allocation,
and generation length are the same as the router diagnostic above. The
corrected default branch retains about 23% over the historical baseline;
the previous roughly 30-31% figure included the non-exact HC reduction.

MLX's `mlx/backend/metal/kernels/gemv.h` separates independent output-row
scheduling from reduction width; with `BN=1` it needs no cross-SIMDgroup
reduction. Its own shuffle tree is not GLM's exact oracle. MLX's RMS kernel
also makes intermediate dtype conversion explicit. llama.cpp's Metal HC pre
kernel broadcasts coefficients across lanes and independently computes output
channels, while its comb kernel uses lane-shuffle reductions. Those are useful
scheduling examples, but transplanting their arithmetic would change this
reference. This fix therefore reuses ds4's existing BF16 arithmetic and HC
finalization; no external project was modified.

Potential later experiment: reduce BF16's physical threadgroup size while
retaining the existing 1,024-virtual-thread normalization trees and each
projection row's lane/FMA order. The current fix keeps the established dispatch
shape to isolate correctness. Do not split a BF16 dot to fill idle SIMDgroups.

Diagnostic artifacts for this follow-up are in
`/var/folders/zw/xt8y4nbj6m3dxyws1yt44_rm0000gn/T/ds4-hc-bf16-exact-8vepte72/`.

## 1. Exact top-eight router selection

GLM selects eight of 288 experts, while the current router pads to 512 and
performs a full bitonic sort: 45 compare/exchange stages, each with a threadgroup
barrier. Explore SIMD-local selection and a deterministic merge of candidates.

Preserve the stable sigmoid, descending biased score, lower-ID tie break, and
the ordered eight-term sum of unbiased probabilities. The same expert set in a
different order is insufficient: both normalization and routed accumulation can
change. Handle exceptional values consistently with the supported baseline.

After the minimal fence check, threadgroup-resident probabilities and IDs are
another experiment: avoid the device round trip while retaining required output
writes and rounding boundaries. Do not combine this with the initial diagnosis.

References: `metal/dsv4_misc.metal:5379`, `ds4.c:45467`, and
`~/llama.cpp/ggml/src/ggml-metal/kernels/argsort.metal:234`. The latter's atomic
top-k emission is not a direct replacement: output order and cutoff ties do not
meet GLM's current ordering contract.

The older router projection-plus-selection measurement was 0.95 ms/token.
Only a fraction is removable; that is a historical stage cost, not a forecast.

## 2. Schedule independent projection outputs more effectively

### Decode QK-low and exact DSA

Decode QK-low launches one threadgroup per head (64 groups). Split independent
output columns into more groups while retaining `glm_quant_dot_row_dev_f32`
unchanged. Do not split a dot across lanes. Consider staging the shared query
only as a separate, independently checked experiment.

The exact DSA dispatcher fixes score staging at 16 rows and value projection at
64 threads. Try smaller score tiles and different value-output group sizes while
retaining each output's arithmetic. Keep the 256-thread softmax partition/tree
and selected-row accumulation order unchanged.

The exact LoRA stage already prefetches effectively. Investigate staging depth
and register pressure only after profiling. Its loader couples eight heads,
256 threads, and 32 staged rows: changing `heads_per_group` alone can leave
staging incomplete. Validate tail stages and invalid/sentinel selections.

References: `metal/dsv4_misc.metal:2347`, `metal/dsv4_misc.metal:3340`,
`metal/dsv4_misc.metal:3455`, `ds4_metal.m:36691`.

### BF16 shape-specific row scheduling

The BF16 host selector uses four SIMDgroups on M3 Ultra and eight elsewhere
across different shapes: small low-rank projections, large KDA projections,
and the vocabulary head. Sweep grouping of independent output rows by shape.
One SIMDgroup must still own the same row, with the same lane/FMA sequence and
SIMD reduction. Multiple independent row accumulators are a later experiment.

Do not apply this rule blindly to generic Q8/F32 matvec NSG settings: some
kernels partition the reduction dimension across SIMDgroups, changing results.

References: `metal/glm53_bf16.metal:32`, `ds4_metal.m:46310`,
`~/mlx/mlx/backend/metal/kernels/gemv.h:29`,
`~/mlx/mlx/backend/metal/quantized.cpp:461`. Borrow MLX's shape-specific
scheduling and register-pressure tradeoffs, not its different dot arithmetic.

## 3. Remove remaining small decode dispatches

- Combine BF16 Q/K/V with `f_a/g_a/beta`: all six read `attn_norm` and use
  independent row dots. Generalize the existing heterogeneous trio scheduling
  while preserving the row helper. Potentially removes 34 dispatches/token;
  the dependent second gate-projection stage stays separate. BF16 artifacts only.
- Fuse the three leading dense FFN tails. They still perform down-projection,
  a copy into `next`, and HC expansion. Investigate the existing Q8 projection
  plus HC epilogue used for DSA output. Only three sites, so expect modest scope.

Preserve directional-steering, debug-output, tensor-type, resource-lifetime,
and streaming restrictions. Match the separate operations' rounding boundaries.

References: `ds4.c:44482`, `ds4.c:45981`, `ds4.c:52970`.

## 4. Interleave independent KDA decode value rows

The fused decode kernel has four SIMDgroups, each walking 32 independent value
rows. Carry/interleave two rows to expose independent instructions while keeping
each row's dot, SIMD sum, FMA, and state ownership unchanged. More SIMDgroups is
another possible experiment, but the original normalization reductions must be
retained exactly, including their four-SIMDgroup topology.

Decode already holds Q/K/decay outside the value loop. The input-reuse benefit
of the existing multi-value prefill kernel therefore does not transfer directly.
The older whole recurrence measurement was 1.23 ms/token; payoff is bounded.

References: `metal/glm53_kda.metal:133`, `metal/glm53_kda.metal:474`, and
`~/llama.cpp/ggml/src/ggml-metal/kernels/gated_delta_net.metal:151`.
llama.cpp illustrates independent state ownership but differs in layout and
arithmetic placement. Do not transplant its recurrence directly.

## 5. Lossless BF16 weight rearrangement for wider loads

The rejected widened-load experiment changed which operands each lane summed.
Instead, consider an in-memory permutation that makes a lane's original eight
weights contiguous. For an original 256-element block, preserve logical operands
`block + lane + 32*i` while storing them at `block + lane*8 + i` (i=0..7).
Two `ushort4` loads can then feed the original eight ordered scalar FMAs and the
same SIMD reduction. Keep the activation operands and BF16-to-F32 conversion.

This changes physical layout only, preserving every weight bit and the original
GGUF. It avoids cross-lane redistribution, but speed remains unproven: activation
loads, cache behavior, compiler output, register pressure, and packing costs
matter. The earlier rejected kernel's roughly 5.4% gain is not a forecast.

Start with resident weights. Retaining both layouts for all four large KDA
projections adds approximately 8.5 GiB on the documented artifact. Account for
startup amortization, allocation limits, view lifetimes, and cache invalidation.
Do not introduce per-token repacking of the large matrices.

References: `metal/glm53_bf16.metal:32`,
`speed-bench/glm53_decode_findings.md:693`.

## 6. Selective concurrency and secondary MoE work

Borrow explicit dependency tracking from MLX's resource tracking and llama.cpp's
overlapping memory-range checks. Only overlap independent GLM branches with
proven buffer ownership and barriers. Routed/shared FFN paths reuse `ffn_mid`;
large projections also compete for bandwidth. ds4 already has parallel FFN
infrastructure worth examining before adding another scheduling subsystem.

References: `~/mlx/mlx/backend/metal/device.cpp:350`,
`~/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:213`,
`ds4_metal.m:9681`, `ds4.c:45467`.

Routed MoE output-row tiling could similarly preserve independent per-row
arithmetic, but existing kernels are relatively close to their measured memory
bandwidth ceiling. Preserve quantized block math, route-weight placement, expert
slot order, and intermediate precision. Lower priority without a fresh profile.

## Exclusions and validation

Exclude requantization, reduced state/cache precision, split-K reductions,
regrouped expert sums, online-softmax substitutions, and parallel formulations
of the recurrent token chain. Existing measurements found little benefit from
decode flush-cadence changes or GPU-side greedy selection on this workload.
Do not silently alter prefill math when experimenting with decode kernels.

Acceptance requires immutable old-kernel comparisons, bytewise affected outputs
and persistent state, and full logits from actual decode steps under identical
token inputs. Cover continuation after prefill, snapshot restore, tails,
sentinels, ties, relevant artifact types, and supported execution modes.
Matching text/argmax does not establish matching logits. Frontier dumps exercise
prefill only; several current BF16 tests use tolerances rather than exact oracles.

Freeze shader sources for each arm: binaries load `metal/*.metal` at runtime.
Use isolated trees or per-file source overrides. Apply fixes consistently to
baseline and candidate, verify exactness first, then benchmark interleaved with
profiling disabled. Ablation can change routing; repeated stages can benefit
from warm weights. Do not sum historical stage estimates into a current budget.

## Exact optimization campaign, 2026-09-05

Reference: corrected BF16 HC producer plus router device fence, frozen before
this campaign. Artifacts and immutable reference sources are under
`/Users/jw/.cache/ds4-bench/glm53-exact-campaign-20260904-233959/`.
The hardware/model/prompt remain the original resident Q4_K artifact on the
80-core M3 Ultra. No weight conversion or approximation is accepted.

### First implemented candidates: router and independent output scheduling

- Router: one SIMDgroup holds nine scores per lane and selects the ordered
  top eight with SIMD max/min. Lower expert ID wins ties. The sigmoid,
  probability stores, selected-ID stores, ordered denominator, and final
  weight arithmetic remain unchanged. Bitwise NaN detection selects an
  emulation of the original 512-wire sorting network, preserving exceptional
  ordering. A 512-thread reduction prototype was also exact but slower on
  prefill, so the single-SIMDgroup version was selected. The NaN fallback is
  slower than the old kernel; finite inputs use the fast path.
- Decode QK-low: 32 independent columns per threadgroup, giving 1,024 groups
  instead of 64 for the actual 64-head / 512-column / 256-input Q8_0 shape.
  The original per-row quantized dot helper is unchanged.
- Short BF16 projections: sixteen independent SIMDgroups per group for the
  128-input gate projections. Each row still has one SIMDgroup and the same
  eight-FMA operand sequence. Large projections retain four SIMDgroups;
  sweeping one/two/four/eight/sixteen did not justify changing those defaults.

Defaults are limited to resident, single-device M3 Ultra execution; synthetic
test mode explicitly exercises dispatch coverage. Individual rollback switches:
`DS4_METAL_DISABLE_GLM53_ROUTER_TOP8`,
`DS4_METAL_DISABLE_GLM53_DECODE_QK_LOW`, and
`DS4_METAL_DISABLE_GLM53_BF16_SHORT_ROWS`. The aggregate rollback covers all
three. MLX's independent row scheduling is useful precedent; llama.cpp's
atomic top-k emission cannot directly satisfy this router's expert ordering.

Kernel screening: router probability/ID/weight dumps match for 1/17/2048 rows,
including ties, saturation, infinities, and NaNs. Finite 2048-row selection is
about 11x faster; single-row selection is about 2x faster. Warm QK output tiling
reduces the real-shape microbenchmark from about 54 to 32 microseconds. Short
BF16 matvec grouping reduces about 11.94 to 8.25 microseconds. These kernel
ratios are not whole-engine gains.

Six interleaved engine screening runs (reference/router/combined/combined/router/
reference, 128 greedy tokens at 2K and 8K) give:

| Context | Corrected reference | Router only | Combined | Combined gain |
| --- | ---: | ---: | ---: | ---: |
| 2K | 26.145 tok/s | 26.290 tok/s | 26.620 tok/s | +1.82% |
| 8K | 25.420 tok/s | 25.550 tok/s | 25.885 tok/s | +1.83% |

All six greedy outputs match. Prefill changes are within the run variation:
509.905 versus 510.710 tok/s at 2K, and 459.600 versus 459.815 for 2K-to-8K.
These are screening measurements using temporary host selectors, followed by
equivalent gated production dispatches; final combined benchmarking follows
the remaining prefill experiments.

The production dispatches pass the complete GLM GPU target and Metal kernel
suite. Added regression cases compare every output bit against rollback and
assert which policy dispatched, so a disabled path cannot silently pass.
Router coverage includes 256/288/512 experts, ties, signed zero, infinities,
NaN payloads, and both individual/aggregate rollback. QK uses the existing
signed and subnormal quantization-scale fixture. BF16 checks the real gate
shape. Full greedy traces (258 frames, 159,840,288 bytes per mode, including
snapshot continuation) retain the corrected reference hashes:

- Default: `3c600ac8994cf3327b2b19e7d2de0d8ad6b0dc261fe7c44eee2358974b0d1bf1`.
- Quality: `fc83d2ab8b6bdb8dca7a4d259f03f3a35e70e77a9ad855d6520ec58c1250bc2a`.

### Prefill experiments: paired Q4_K MoE GEMM and vector dequantization

GLM's grouped prefill path still dispatches gate GEMM, up GEMM, and weighted
SwiGLU separately. The repo already has a Q4_K paired GEMM with the same
per-output MMA order and explicit F32 staging before its F16 activation store,
used by DeepSeek's six-expert path. Its indexing accepts eight selected experts.
Test sharing the B tile and fusing the activation for GLM, with resident-only
ownership guards, exact intermediate/output comparisons, and full logits.
All three paired variants passed the native intermediate/output oracle and
four fresh-8K ABBA runs with full frontier-logit equality. None justified a
default: plain 64-row tiles lost 2.77%, tail-culling 64-row tiles lost 0.34%,
and compact tail-culling 32-row tiles gained only 0.14%, within noise.
Sharing the activation tile and removing dispatches did not overcome the
register/occupancy cost. Keep the separate GLM chain.

A GLM-only Q4_K dequantization prototype loaded four packed bytes at once
and converted four independent F32 results to F16 together, preserving the
scale/min operations and using the old helper for nonfinite scales. Native
tests and all four fresh-8K frontier comparisons passed, but prefill fell
from 472.48 to 460.78 tok/s (-2.48%). Keep scalar dequantization. These results
also caution against assuming vector loads or explicit pairing beat the
compiler's scheduling on this GPU.

### Additional scheduling and lossless-layout experiments

- Exact DSA decode scores: reduce rows staged per group from sixteen to two
  for the real 64-head / 512-latent / 256-value shape and at most 2,051 selected
  entries (including the indexer's three possible sentinels). Every score,
  denominator, latent vector and head output matches. The softmax and value
  reduction trees are unchanged. Paired 512-token engine screens improved
  0.14% at 2K and 0.25% at 8K; all 529 full-logit rows and 528 token choices
  match. Implemented with `DS4_METAL_DISABLE_GLM53_DSA_SCORE_TILE` rollback;
  larger selections retain sixteen rows.
- Lossless BF16 lane-interleaved weight packing: two vector loads replace
  eight scalar loads while preserving all FMAs. Rotating large matrices
  remain bit-exact, but roughly 98-microsecond projections barely improve.
  The extra model-sized storage and packing machinery are not justified.
- Applying the same packing only to BF16 HC is also exact and flat at about
  37 microseconds. Keep original weights and the corrected producer fusion.
- BF16 HC with eight/four physical SIMDgroups emulating the same virtual RMS
  and collapse trees is exact, but slower: about 41/50 versus 37 microseconds.
  Fewer idle SIMDgroups do not compensate for reduced parallelism elsewhere.
- KDA prefill recurrence with four values per SIMDgroup, versus the existing
  two, passed all fresh-8K frontier checks but lost 0.55%. Keep two.
- KDA prefill prepare blocks of 8/16/64 rows, versus the existing 32, passed
  every fresh-8K frontier check in separate ABBA screens. Changes were
  -0.16%/-0.002%/+0.03%; keep 32. Together with the recurrence sweep, this
  provides no evidence that a different existing KDA prefill knob helps.
- KDA decode can interleave four independent value rows while retaining each
  row's original SIMDgroup and arithmetic sequence. Output, convolution
  history, and every recurrent-state bit match after 256 synthetic updates.
  Warm kernel time falls from about 23 to 14.5 microseconds. Paired 512-token
  screens improve 0.97% at 2K and 0.95% at 8K; every full logit and selected
  ID matches. Implemented the four-row variant with an original-path rollback
  (`DS4_METAL_DISABLE_GLM53_DECODE_KDA_VALUES4`) and persistent-state regression.

Scratch sources, harnesses and raw results for accepted and rejected variants
are retained in the campaign artifact directory. Whole-model gains must be
remeasured together; the separate percentages above must not be added.

### Combined BF16 KDA input projections

Flatten Q/K/V (8,192 rows each), f_a/g_a (128 each), and beta (64) into one
24,896-row dispatch, all reading the same 4,096-element normalized input.
The existing BF16 row helper is unchanged; the dependent f_b/g_b pair remains
separate. All six graph outputs have distinct allocations. Ablation/repetition
of either substage, non-BF16 weights, nonresident/TP execution, other hardware,
and existing prerequisite rollbacks retain the original path.

Four rotating weight sets reproduce all six outputs exactly and save roughly
9–12 microseconds per layer in the warm component test. Paired 512-token model
screens improve 26.7752 to 26.9718 tok/s at 2K (+0.73%) and 26.1368 to 26.3384
at 8K (+0.77%). Both compare 529 full-logit rows and 528 selected IDs exactly.
The production variant has fixed validated dimensions, checks every buffer
before encoding, and adds `DS4_METAL_DISABLE_GLM53_KDA_INPUTS` rollback.
Native regression compares all six projections with separate matvecs using
varied/cancelling/zero/large inputs and signed/subnormal BF16 weights.

Remaining lower-priority ideas: the three dense FFN tails could save at most
six small dispatches per token; output-row scheduling in routed MoE and selective
engine concurrency need more targeted profiling. The latter must first address
shared `ffn_mid` ownership and bandwidth contention. No unproven concurrent
execution or weight-layout change is enabled by this campaign.

For a later prefill experiment, consider staging 64 K elements in grouped
Q4_K GEMM while issuing the same ordered 32-element MMA steps. This could
reduce staging barriers, but increases shared memory/register pressure; the
paired-GEMM and vector-load regressions show why that tradeoff needs measurement.
Do not change the MMA accumulation order or quantization conversion boundary.
This is an unimplemented proposal, not an established speedup.

### Final combined result

The six promoted policies pass the complete GLM GPU target and Metal kernel
suite. Default and quality 2K/8K traces retain the exact hashes recorded above.
A different novel segment at 16K, 64 greedy updates, and 32K allocation also
matches every bit: 65 frames / 40,269,840 bytes, SHA-256
`f49ed697561fdb2206b5414cddcadfcf9cd88b3ee0cd9af834347ec451607d22`.

Six quiet, uninstrumented runs with frozen binaries/shaders, three per arm,
ordered reference/candidate/candidate/reference/reference/candidate, use 256
greedy tokens at both frontiers. Every generated text matches.

| Context | Corrected reference | Final candidate | Additional decode gain |
| --- | ---: | ---: | ---: |
| 2K | 26.050 tok/s | 27.040 tok/s | +3.80% |
| 8K | 25.400 tok/s | 26.367 tok/s | +3.81% |

Prefill is effectively unchanged: 510.237 versus 510.870 tok/s at 2K and
459.803 versus 460.483 tok/s for the 2K-to-8K continuation (+0.12%/+0.15%).
Keep the rejected prefill variants out of the default path. Full tables,
rollback controls, validation details and limitations are recorded in
`speed-bench/glm53_decode_findings.md`; raw final CSVs are under
`glm53-exact-campaign-20260904-233959/final-benchmark/` in the cache directory.
The corrected BF16 HC producer fusion remains enabled. This file remains
untracked, as requested.

# Where GLM 5.3 Flash decode time goes on an M3 Ultra

Apple M3 Ultra, 80 GPU cores, 512 GB, macOS 26.5.2, Metal backend.
Model: `GLM-5.3-Flash-Q4_K.gguf`, 177.8 GiB, fully resident, no SSD streaming.

The short version: the routed-expert kernels are already close to the hardware
ceiling, and the largest single consumer of decode bandwidth is not the experts
at all.  It is the KDA (linear attention) projections, which this artifact
stores as **BF16** while its experts are Q4_K.  Requantizing them to Q8_0 --
which is what `gguf-tools/glm53_quantize.py` already specifies for
`role == "linear_attention"` -- is worth **+13.4% decode** with no measurable
quality cost.

## Decode budget

Measured with `DS4_GLM_DECODE_ABLATE`, which removes a stage and reports the
resulting speed.  Baseline 21.19 tok/s, two baseline runs 0.38% apart.

**The KDA rows below were re-measured.**  The original 18.37 ms figure was
produced by instrumentation that was never committed -- `DS4_GLM_DECODE_ABLATE`
had no `kda` bit, and the KDA path returned before the mask was read -- so it
could not be reproduced from this tree.  `kda`, `kda_qkv`, `kda_gate`,
`kda_recur` and `kda_out` now exist, and the table is what they report.  KDA is
**15.99 ms/token, 35.8% of decode**, not 18.37 ms and 38.9%.  Every other row
reproduced within noise on the same machine, at a 22.375 tok/s baseline
(four interleaved baselines, 0.85% spread) rather than 21.19.

| component | ms/token | share | bytes/token | GB/s | % of ceiling |
|---|---:|---:|---:|---:|---:|
| KDA attention (34 layers) | 18.37 | 38.9% | 8.50 GiB | 497 | 67% |
| routed MoE (42 layers) | 8.08 | 17.1% | 4.43 GiB | 589 | **80%** |
| DSA attention core (11 layers) | 7.91 | 16.8% | -- | -- | -- |
| shared expert (42 layers) | 2.13 | 4.5% | 0.55 GiB | 279 | 38% |
| attn_output projection (11 layers) | 1.26 | 2.7% | 0.73 GiB | 622 | **84%** |
| q_path (11 layers) | 0.59 | 1.3% | -- | -- | -- |
| indexer (11 layers) | 0.35 | 0.7% | -- | -- | -- |
| mHC producer chain (90 sites) | 3.99 | 8.9% | -- | -- | -- |
| output head (norm + logits matvec) | 1.80 | 4.0% | -- | -- | -- |
| remaining norms, residual, hc_expand | ~3.0 | ~6.7% | -- | -- | -- |

`% of ceiling` is against 736.9 GB/s, the sequential-read ceiling measured on
this machine by `speed-bench/metal_bandwidth_probe`.

Two things follow.

**The weight-streaming kernels are not the problem.**  Routed MoE runs at 80%
of the achievable sequential-read bandwidth and the `attn_output` projection at
84%.  There is very little left in them.

**A per-token bandwidth figure computed over the whole decode step is
misleading.**  Dividing total bytes by total decode time gives roughly a fifth
of peak, but the bandwidth-bound kernels only occupy about a fifth of the step.
The kernels themselves are near the ceiling; the rest of the step is other
work.

## The BF16 KDA projections

`blk.N.kda_q`, `kda_k`, `kda_v` and `kda_output` are BF16 in this artifact, on
all 34 KDA layers.  They are dense -- every one is read on every decoded token:

| | bytes/token | share of decode traffic |
|---|---:|---:|
| KDA q/k/v/output (BF16) | 8.50 GiB | **60%** |
| routed experts (Q4_K) | 4.43 GiB | 31% |
| shared expert (Q8_0) | 0.55 GiB | 4% |
| attn_output (Q8_0) | 0.73 GiB | 5% |

The KDA projections alone read nearly twice what all routed experts read.

This is not what the repo's own quantizer produces.  `regular_qtype()` in
`gguf-tools/glm53_quantize.py` maps `role == "linear_attention"` to `Q8_0` for
its default `--artifact q4`, and embedding/output likewise.  This artifact has
all of them at BF16, so it was not produced by that path.

### Why BF16 is not simply a mistake

Metal has a fused three-way QKV matmul, `ds4_gpu_glm53_matmul_bf16_qkv`, which
requires all three of q/k/v to be BF16 and issues one dispatch instead of
three.  It is gated behind `DS4_METAL_DISABLE_M3_ULTRA_GLM53_DECODE`, so it was
added as an M3 Ultra optimisation.  Quantizing KDA forfeits it and falls back
to the generic per-tensor matmul.

Measured, so the trade is not a guess:

| | decode tok/s | Δ |
|---|---:|---:|
| baseline (fused BF16 QKV) | 21.12 | -- |
| `DS4_METAL_DISABLE_GLM53_BF16_QKV=1` | 20.97 | -0.7% |
| `DS4_METAL_DISABLE_M3_ULTRA_GLM53_DECODE=1` | 20.96 | -0.8% |

The fusion is worth **0.7%**.  The BF16 storage it requires costs an order of
magnitude more than that.  No fused Q8_0 QKV kernel is needed to capture the
win; the generic fallback is nearly free.

## Result of requantizing KDA to Q8_0

`gguf-tools/glm53-requant-bf16` converts the 136 KDA tensors from BF16 to Q8_0
into a **new file**, copying every other byte verbatim, through the same
`quants.c` facade the other tools use.  It refuses an output that resolves to
the input (same path, hard link or symlink), because the input stays mmapped
for the whole run; there is no in-place mode.

    make -C gguf-tools glm53-requant-bf16
    ./gguf-tools/glm53-requant-bf16 in.gguf out.gguf --type q8_0 --tensors kda

`--tensors` also takes `head` (`output.weight`), `embd` (`token_embd.weight`)
and `all`; it defaults to `kda`.

8.50 GiB of KDA weights become 4.52 GiB; the file goes 177.8 -> 173.8 GiB.

Speed, arms interleaved O-Q-Q-O with the same binary and only the model file
changing, 8 context frontiers:

| ctx | original | kda Q8_0 | Δ |
|---:|---:|---:|---:|
| 2,048 | 21.05 | 23.91 | +13.54% |
| 8,192 | 20.65 | 23.42 | +13.44% |
| 16,384 | 20.56 | 23.31 | +13.40% |
| **mean (8 ctx)** | | | **+13.37%** |

Prefill -0.25%.  Within-arm drift 0.15-0.19%, so the effect is far outside the
noise, and it is within 0.3 points at every context.

Quality, teacher-forced over 18,672 tokens of `promessi_sposi.txt`:

| | avg NLL | perplexity |
|---|---:|---:|
| original (BF16 KDA) | 1.838851 | 6.289309 |
| requantized (Q8_0 KDA) | 1.834773 | **6.263711** |

No degradation -- marginally better, which at this size is noise.  Greedy
generations from both are coherent and track word for word until a late
paraphrase.

### The projection was too optimistic, and why -- corrected

The earlier reading of this was wrong, and it mattered, because it set the
priority for the whole KDA path.

It went: scaling KDA's 497 GB/s by the byte reduction predicts +22%, we
measured +13.4%, therefore only ~62% of KDA was weight streaming and the
remaining **~7 ms/token** is conv1d, gating and the recurrent state update --
"the next thing to attack, and not a bandwidth problem."

Direct substage ablation says otherwise.  Splitting KDA on the original BF16
artifact:

| substage | ms/token | share of decode |
|---|---:|---:|
| qkv projections | 9.68 | 21.7% |
| output projection | 3.16 | 7.1% |
| gate/beta low-rank chain | 1.37 | 3.1% |
| recurrence kernel (conv1d + gating + state) | **1.23** | **2.8%** |
| unattributed (dispatch, interaction) | 0.55 | 1.2% |

The conv1d, the gating and the recurrent state update together are **1.23
ms/token**, not ~7.  KDA is about 90% weight streaming, not 62%.  Requantizing
the same tensors to Q8_0 and re-ablating confirms it directly -- qkv goes 9.68
-> 5.61 ms and the output projection 3.16 -> 1.85 ms against a pure-bandwidth
prediction of 5.14 and 1.68, so both are ~90% bandwidth-scaled:

| | original | Q8_0 KDA |
|---|---:|---:|
| decode | 22.375 tok/s | 25.545 tok/s (**+14.2%**) |
| KDA total | 15.99 ms | 10.40 ms |

The +13.4% headline reproduces (+14.2% here).  Only the explanation was wrong.

Where the original inference went astray: it assumed everything in KDA that
did not scale with weight bytes was recurrence work.  Most of it is instead the
projections failing to scale *perfectly* -- they are ~90% bandwidth-bound, not
100% -- plus fixed dispatch cost.  Attributing that gap to the recurrence
inflated a 1.23 ms stage into a 7 ms one.

The practical consequence: **the recurrence kernel is not where the time is.**
Work on `metal/glm53_kda.metal` is capped at 2.8% of decode no matter how good
it gets.  The qkv projections, at 21.7%, are the KDA target that matters.

## Splitting the old "norms, hyper-connections, residual, LM head" row

That row was a residual -- whatever the other arms did not account for -- and
at ~18% of decode it was the second largest line in the budget with nothing
measured inside it.  The `hc` and `head` ablation arms split it:

| | ms/token | share |
|---|---:|---:|
| mHC producer chain | 3.99 | 8.9% |
| output head | 1.80 | 4.0% |
| everything else in the row | ~3.0 | ~6.7% |

`hc,head` together measure 5.77 ms against 5.79 for the two separately, so the
split is additive and the arms are not interacting.

**The mHC producer was the largest unoptimised item in the decode step.**
`glm53_graph_hc_pre` issued four dispatches -- plain RMSNorm, the 16384->24 mix
matvec, the split/mix, and the weighted RMSNorm -- twice per layer over 45
layers, so 360 small dispatches per token for 3.99 ms of work.  DeepSeek V4
already fused the F16 equivalent; GLM 5.3 now uses the same kernel with BF16
mix weights, one dispatch per site instead of four:

| ctx | four dispatches | fused | delta |
|---:|---:|---:|---:|
| 2,048 | 22.282 | 23.545 | **+5.67%** |
| 4,096 | 21.94 | 23.195 | +5.72% |
| 16,384 | 21.795 | 23.00 | +5.53% |

Bit-exact: all 154,880 logits match to max|delta| = 0.  Prefill is unchanged,
since only the decode path is fused.  2.41 ms of the 3.99 ms is gone; the
remaining 1.58 ms is the fused kernel's own arithmetic.

**This is the largest engine-only decode gain found on this path**, and it is
worth contrasting with the KDA recurrence work the earlier budget pointed at:
that stage is 2.8% of decode in total, while this one change is +5.67%.

**The output head is nearly all matvec.**  1.80 ms for a [4096 -> 154880]
BF16 matvec is close to what its 1.27 GB costs at this machine's measured
bandwidth, so there is no dispatch overhead worth chasing there.

### Why GPU-side argmax is not worth doing

A natural suggestion is to stop reading all 154,880 logits back and scanning
them on the CPU, and instead do a hierarchical argmax/top-k on the GPU and
return only the token.  Measured directly on this machine:

    logits readback (memcpy of 605 KiB)   0.0143 ms
    CPU argmax scan over 154,880 floats   0.1668 ms
    combined                              0.1811 ms

That is **0.40% of a 44.76 ms decode step, below the run-to-run spread**, so
the change could not be shown to work even if it were free.  It also would not
remove a synchronisation: the token is needed before the next step can start
either way.  Not worth the complexity.

## A trap in the stage profiler

`DS4_METAL_DECODE_STAGE_PROFILE` reports a stage named `attn_output` on all 45
layers, and on KDA layers it is the largest stage in the run.  It is **not**
measuring the output projection there.  The profile boundary sits after the
`glm53_attention_done:` label, and KDA layers reach that label by `goto`, so on
those layers the `attn_output` sample times the entire KDA attention.

The real output projection is 2.7% of decode, not 39%.  Ablation and the
profiler agree to within 0.3 points once the label is read correctly (2.7% vs
3.0% on the 11 DSA layers, where the label means what it says).

Two further cautions when using that profiler: it flushes the command buffer at
every boundary, which on this workload adds a uniform ~0.206 ms per boundary
and roughly triples the measured decode time; and because the floor is uniform,
stages that do little real work all read as roughly the floor.  Prefer
`DS4_GLM_DECODE_ABLATE` for attribution and use the stage profiler to localise.

## What is left, priced

With KDA split and the residual row split, every large line in the budget has a
measured cost and a measured bandwidth.  Byte counts are exact, read from the
GGUF tensor table; the ceiling is 736.9 GB/s.

| stage | bytes/token | ms | GB/s | % of ceiling |
|---|---:|---:|---:|---:|
| kda q/k/v (3 x [4096,8192] BF16 x34) | 6.845 GB | 9.68 | **707** | **96%** |
| kda_output ([8192,4096] BF16 x34) | 2.282 GB | 3.16 | **722** | **98%** |
| kda gate/beta (f_a, f_b, beta, g_a, g_b) | 0.232 GB | 1.37 | 169 | 23% |
| kda recurrence (136 MiB state, r+w) | 0.285 GB | 1.23 | 232 | 31% |

**The KDA projections are finished.**  At 96% and 98% of the ceiling there is
nothing left in them.  Specialising the BF16 matvec for the 4096 and 8192
shapes -- function constants to unroll the loops, two output rows per
simdgroup, staging the activation row in threadgroup memory -- cannot pay,
because the kernel already moves bytes about as fast as the machine will.

This also corrects the 497 -> 547 GB/s figure recorded when the widened loads
landed.  That came from the 18.37 ms KDA row, which was never measured; against
the measured 9.68 ms the q/k/v projections run at 707 GB/s.

**What is left is dispatch overhead, not bandwidth.**  The two stages far below
the ceiling are the ones made of many small launches.  The mHC fusion prices a
dispatch directly: 270 removed for 2.41 ms, about **8.9 us each**.

The gate/beta chain moves 232 MB, which is 0.33 ms at the rate the big
projections achieve, and costs 1.37 ms.  The other ~1.04 ms is 170 dispatches
(5 matvecs x 34 layers) at ~6 us, agreeing with the mHC figure.  So the
remaining KDA work is worth about **1.0 ms, 2.3% of decode**:

- `f_a` and `g_a` are both [4096 -> 128] from the same `attn_norm` input, so
  they pair the way `ds4_gpu_glm53_matmul_bf16_qkv` already pairs q/k/v.
  `beta` is [4096 -> 64] off the same input at a different width.
- `f_b` and `g_b` are both [128 -> 8192] but read different activations, so
  pairing them needs a two-input kernel.
- Both need a second low-rank buffer: `g->kda_lowrank` is written by `f_a`,
  read by `f_b`, then overwritten by `g_a`.

Fusing the projection consumers with the HC expansion
(`ds4_gpu_hc_expand_tensor`, 90 dispatches per token) is the same kind of play
in the ~3.0 ms "everything else" bucket -- dispatch count, not bandwidth.

**FP16 storage for the recurrent state is not worth pursuing.**  At 31% of
ceiling the state is latency-bound rather than bandwidth-bound, so halving it
would not halve the 1.23 ms; the whole stage is 2.8% of decode, and the upside
is well under 1% against an accumulating-error risk over long contexts.

## Two tuning knobs that turn out not to matter

Both were expected to be worth something on an 80-core GPU and neither is.

**Decode command-buffer flush cadence.**  Indexed decode flushes every 4
layers, and `DS4_GLM_DECODE_FLUSH_INTERVAL` overrides it.  Sweeping 0, 2, 3, 4,
6, 8, 12, 16, 32 at ctx 2048: everything from 3 to 12 lands in 22.25-22.31
tok/s, inside the run-to-run spread.  Only the extremes lose -- 0 (never flush)
at 21.91 and 32 at 22.06.  Confirmed at ctx 16384, where 2/4/8 give
21.80/21.81/21.78.  **The default of 4 is already right.**

**DSA split-attention rows per block.**  The choice steps straight from 32 to
128 at 1024 selected rows and had never been swept, so
`DS4_GLM_DECODE_SPLIT_BLOCK_ROWS` was added to force one value:

| rows | ctx 2048 | ctx 16384 |
|---:|---:|---:|
| default (32/128) | 23.50 | 23.00 |
| 32 | 23.51 | 22.97 |
| 64 | 23.51 | 23.02 |
| 96 | -- | 23.01 |
| 128 | 23.51 | 23.02 |
| 256 | 23.49 | 23.02 |

Flat to within 0.2% at both contexts.  The selection count is capped by
`glm53_graph_indexer_selected_limit()`, which does not grow with context, so
this does not become interesting at longer contexts either.  The knob is kept
as instrumentation for other GPUs, not because it found anything here.

## A trap when A/B-testing a shader change

`ds4_gpu_full_source()` reads `metal/*.metal` from disk at run time and there
is no embedded fallback, so building two binaries around a shader edit does
**not** compare two shaders -- both read whatever is on disk when they run.
Use the per-file overrides (`DS4_METAL_GLM53_KDA_SOURCE` and its siblings) with
a single binary instead.  A measurement in this file was wrong for exactly
this reason before it was caught.

## Scope and caveats

- This is a **model-file** change, not an engine change.  It does not speed up
  an artifact you already have; it produces a better one.
- Why the shipped artifact is BF16 is not established here.  It contradicts the
  repo's own quantizer, which suggests the artifact pipeline rather than a
  deliberate choice, but if it was deliberate the fix belongs upstream.
- Quality evidence is one perplexity run on one text plus a greedy generation.
  That is good evidence for a near-lossless type like Q8_0, not proof.
- `--artifact q4` also specifies Q8_0 for the embedding and output tensors,
  which are BF16 here too (~1.2 GiB more per token through the LM head).  Not
  measured; the same tool could be extended to cover them.
- Neither the constants recorded below nor a Q4_K KDA variant were measured.
  The tool accepts `q4_K` as a target, which would take KDA to 2.39 GiB, but
  Q4_K on attention projections is a materially bigger quality question than
  Q8_0 and was not attempted.

## Untested constants noticed while reading

Recorded so the next person does not re-derive them.  None were measured.

- `glm_graph_full_attention_cap` gives the **SSD-streaming** path a full
  attention cap of 8192 and the **fully-resident** path 4096.  The
  memory-constrained machine gets the larger window.  Above a 65536 context
  both clamp to 4096, so the asymmetry is only reachable below that -- and
  `ds4-bench` defaults `--ctx-alloc` to `ctx-max + gen-tokens + 1`, which
  exceeds 65536 on a 65536 sweep, hiding it.
- `DS4_GLM53_PREFILL_CHUNK_TOKENS` is a flat 2048 with no device, memory or
  residency input; the GLM 5.2 path uses 4096.
- `DS4_GLM_METAL_INDEXED_PREFILL_SCORE_SCRATCH_MB` is a fixed 256 MiB that
  clamps how many rows are scored per dispatch.  Observed as
  `score_scratch=64.00 MiB` at runtime, so something derives it down; worth
  checking which value actually binds.
- A trap for anyone testing the prefill chunk: `DS4_GLM53_PREFILL_CHUNK_TOKENS`
  (2048) and `DS4_GLM_METAL_FULL_ATTN_LAYER_FLUSH_CONTEXT` (2048) are the same
  number and the comparison is a strict `>`, so the default chunk sits exactly
  on the boundary where per-layer command-buffer flushing switches off.
  Raising the chunk to 4096 also switches flushing on across 46 layers.  Those
  are two changes, not one.

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

**The original KDA row was never measured.**  `DS4_GLM_DECODE_ABLATE` had no
`kda` bit and the KDA path returned before the mask was read, so the 18.37 ms
and 38.9% recorded here first could not have come from this harness.  `kda`,
`kda_qkv`, `kda_gate`, `kda_recur`, `kda_out`, `hc` and `head` now exist, and
the whole budget below was re-measured through them.

Everything in the table is from **one** run of arms against **one** baseline of
22.375 tok/s = 44.693 ms/token (four interleaved baselines, 0.85% spread).
Do not mix it with the earlier 21.19 tok/s figures; those are superseded.

| component | ms/token | share | superseded figure |
|---|---:|---:|---|
| KDA attention (34 layers) | 15.99 | 35.8% | was 18.37 / 38.9% |
| DSA attention core (11 layers) | 8.02 | 18.0% | was 7.91 / 16.8% |
| routed MoE (42 layers) | 7.89 | 17.6% | was 8.08 / 17.1% |
| mHC producer chain (90 sites) | 3.99 | 8.9% | was inside the residual row |
| shared expert (42 layers) | 1.98 | 4.4% | was 2.13 / 4.5% |
| output head (norm + logits matvec) | 1.80 | 4.0% | was inside the residual row |
| attn_output projection (11 layers) | 1.33 | 3.0% | was 1.26 / 2.7% |
| q_path (11 layers) | 0.44 | 1.0% | was 0.59 / 1.3% |
| indexer (11 layers) | 0.21 | 0.5% | was 0.35 / 0.7% |
| remaining norms, residual, hc_expand | 3.04 | 6.8% | residual, not ablated |
| **total** | **44.69** | **100%** | |

Only KDA moved outside noise; every other carried-over row reproduced.

Bandwidth is deliberately **not** a column here.  It is only meaningful where
the byte count is exactly derivable, which is the dense projections; see "What
is left, priced" below for those, computed against 736.9 GB/s.  The routed-MoE
and shared-expert byte figures recorded in the first version of this document
(4.43 and 0.55 GiB) depend on which experts a token selects and were never
re-derived here, so they are omitted rather than restated.

**A per-token bandwidth figure computed over the whole decode step is
misleading.**  Dividing total bytes by total decode time gives roughly a fifth
of peak, but the bandwidth-bound kernels only occupy about a fifth of the step.
The kernels themselves are near the ceiling; the rest of the step is other
work.

**Ablation arms are destructive.**  A skipped stage leaves stale contents in
its output buffer, which is fine for timing the dispatches that remain but can
in principle change data-dependent routing downstream (expert selection, index
selection).  The arms agree with each other -- `hc,head` measures 5.77 ms
against 5.79 for the two separately, and the KDA substages sum to 15.44 against
15.99 for the whole stage -- so the effect is small here, but these are
skip-ablation estimates, not per-kernel timings.

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

## A non-destructive second instrument

The ablation arms are destructive: a skipped stage leaves stale contents, so
the run is timing-only and can in principle perturb data-dependent routing.
`DS4_GLM_DECODE_REPEAT` is the other half of the pincer.  Every stage it
accepts is a pure function of its inputs, so dispatching it one extra time per
site writes the same bytes; the whole-token delta is then one extra execution
of that stage and **the model output is unchanged**.  Verified: all six arms
dump logits identical to the baseline at max|delta| = 0.

Only idempotent stages get a bit.  The KDA recurrence advances conv and
recurrent state, and directional steering updates in place, so neither can be
repeated and neither is offered.

Where the two instruments agree, the number is trustworthy:

| stage | ablate | repeat | agreement |
|---|---:|---:|---|
| kda_qkv | 9.69 | 9.76 | 0.7% |
| head | 1.80 | 1.76 | 2% |
| kda_gate | 1.37 | 1.45 | 6% |
| **kda_out** | **3.33** | **2.47** | **35%** |

The `kda_out` disagreement is reproducible across rounds, and the exact byte
count settles it.  Both instruments agree that kda_qkv costs 9.69 ms for
6.845 GB, i.e. 706 GB/s; kda_output is 2.282 GB, which at that rate is 3.23 ms
-- next to the ablation figure, not the repeat one.  Repeat underestimates here
because the second dispatch re-reads a 67 MB per-layer weight set that is
partly still resident, while kda_qkv's 201 MB per layer does not survive.

**So: repeat is the right instrument for dispatch-bound stages and undercounts
cache-friendly bandwidth-bound ones; ablation is the reverse.**  Use both, and
let exact bytes arbitrate when they disagree.

### The remaining bucket, partly split

`hc_expand` measured **0.55 ms/token, 1.3% of decode** by repeat -- a
dispatch-bound stage, so that figure was the reliable one.  All 90 sites are
now folded into whatever produces their input, and between them they returned
0.37 ms of it:

| site | count | mechanism | gain |
|---|---:|---|---:|
| kda_output (BF16 matvec) | 34 | new epilogue kernel | +0.46% |
| attn_output (Q8_0 matvec) | ~12 | DeepSeek's existing fused kernel | +0.11% |
| FFN tail (routed+shared add) | 43 | existing `has_add` path on the expand | +0.14% |

The last two together are +0.48% (t=9.65, n=8), 0.199 ms over 54 sites, or
3.7 us per site -- below the 4.6 us launch cost and the KDA epilogue's 5.6 us,
which fits: those two remove a cheap elementwise add and a Q8_0 matvec rather
than a BF16 matvec plus a 64 KiB round-trip.

Two of the three needed no new kernel at all.  `ds4_gpu_matmul_q8_0_hc_expand_tensor`
already existed for DeepSeek and reads post/comb from `hc_split` at the offsets
GLM uses; `ds4_gpu_hc_expand_add_tensor` already exposed the expand kernel's
`has_add` path.  Only the BF16 matvec needed an epilogue written.

That leaves roughly 2.5 ms in the residual row for the residual adds,
directional steering, the remaining norms and the final HC collapse, none of
which are separated yet.

With the mHC producer now fused, `hc_pre` measures 1.36 ms by repeat, down from
the 3.99 ms the four-dispatch chain cost.

## What is left, priced

With KDA split and the residual row split, the dense stages can be checked
against the memory system.  The ceiling is 736.9 GB/s.

The **weight** byte counts are exact -- summed from the GGUF tensor table, per
token, over all 34 KDA layers.  They are a lower bound on total traffic: they
exclude activations, intermediate writes, and (for the recurrence row) the
conv state, q/k/v inputs, gate inputs, conv weights and biases, and the output
write.  For the two dense projection rows the weights dominate so completely
that the omission does not matter; for the two small rows it does, and the
GB/s shown for them is correspondingly an **under**estimate.

| stage | weight bytes/token | ms | GB/s (weights only) | vs ceiling |
|---|---:|---:|---:|---:|
| kda q/k/v (3 x [4096,8192] BF16 x34) | 6.845 GB | 9.68 | **707** | **96%** |
| kda_output ([8192,4096] BF16 x34) | 2.282 GB | 3.16 | **722** | **98%** |
| kda gate/beta (f_a, f_b, beta, g_a, g_b) | 0.232 GB | 1.37 | >=169 | >=23% |
| kda recurrence (136 MiB state, r+w) | 0.285 GB | 1.23 | >=232 | >=31% |

**The KDA projections are done, on this machine.**  At 96% and 98% of a
736.9 GB/s ceiling there is no room for a faster inner loop; what remains is
within measurement error of the memory system.  This is an M3 Ultra result --
a part with a different bandwidth-to-compute ratio could sit lower and have
something to gain.  Specialising the BF16 matvec for the 4096 and 8192
shapes -- function constants to unroll the loops, two output rows per
simdgroup, staging the activation row in threadgroup memory -- cannot pay,
because the kernel already moves bytes about as fast as the machine will.

This also corrects the 497 -> 547 GB/s figure recorded when the widened loads
landed.  That came from the 18.37 ms KDA row, which was never measured; against
the measured 9.68 ms the q/k/v projections run at 707 GB/s.

**What is left is per-launch cost, not bandwidth.**  The two stages far below
the ceiling are the ones made of many small dispatches.

How much of that is launch overhead specifically is *not* established here, and
the mHC result should not be read as a per-dispatch price.  Collapsing four
dispatches into one removed 2.41 ms across 90 sites, but it removed three
intermediate round-trips per site (`hc_flat`, `hc_mix`, `hc_split` each written
then re-read) along with the launches, and a fused kernel also gets better
occupancy on small work than four sequential ones.  Dividing 2.41 ms by 270
gives 8.9 us per dispatch only if launches were the whole cost, and they were
not.

The same caution applied to the gate/beta chain, and the benchmark bore it out.
The chain moves at least 232 MB, which would be 0.33 ms at the rate the big
projections achieve, and it cost 1.37 ms, so ~1.04 ms looked available.
**Pairing it recovered 0.31 ms of that, not 1.0 ms** -- the upper bound was
three times the prize, which is why it was written as one.

That result also gives the first clean per-dispatch number.  Pairing removes 68
dispatches per token and removes *nothing else*: the same buffers are written
and the same weight bytes are read, so the saving is launch overhead and
nothing but:

    0.310 ms / 68 dispatches = 4.6 us per dispatch

Applying that back to the mHC fusion decomposes its 2.41 ms honestly:

| | ms |
|---|---:|
| launch overhead (270 x 4.6 us) | 1.23 |
| intermediate traffic + occupancy | 1.18 |

So roughly half of the mHC win was dispatch count and half was the three
intermediate round-trips per site that the fused kernel no longer materialises.
The earlier 8.9 us per dispatch inferred from that fusion alone was about twice
the real launch cost, exactly because it absorbed the traffic half.

The remaining shape of the KDA gate work, now measured rather than projected:

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

## Cumulative engine-only result

Individual commits report gains against whatever baseline was current when they
landed, which does not compose into a branch number.  This is the direct
measurement: the pre-series commit and the branch tip, each built in its own
tree so each reads its own `metal/*.metal`, run against the **same unchanged
GGUF** with the same harness, contexts and interleaving.

    ctx 2048, 128 generated tokens, arms interleaved, 3 pairs

    base (110afdd)   21.180 tok/s   47.21 ms/token
    tip              23.863 tok/s   41.91 ms/token
    engine-only      +12.67%

Contributions, each measured against the baseline current when it landed: the
widened BF16 loads ~+5.4%, the mHC producer fusion +5.67%, the KDA gate pairing
+0.74%, the kda_output HC-expand epilogue +0.46%.

Note the base reproduces the 21.19 tok/s of the original budget almost exactly,
which is a useful check that machine conditions have not drifted between the
first measurements in this document and the last.

Stacking the model-artifact changes on top of the same tip, all at ctx 2048:

| model file | tok/s | vs base engine + original artifact |
|---|---:|---:|
| GLM-5.3-Flash-Q4_K | 23.59 | +11.2% |
| GLM-5.3-Flash-Q4_K-kdaQ8 | 27.21 | +28.2% |
| GLM-5.3-Flash-Q4_K-kdaHeadQ8 | 27.84 | +31.2% |

Only the first row is an engine result.  The other two combine it with the
requantized artifacts and should never be quoted as engine tuning.

## A trap when verifying a decode-path change

`ds4-bench --dump-frontier-logits-dir` writes one file per **frontier**, which
is the logits at the end of prefill.  It does not exercise the single-token
decode graph at all.

This was found the hard way.  A change that skipped the FFN-side mHC producer
on every KDA layer -- catastrophic, garbage output after the first token --
produced frontier logits **bit-identical** to the baseline, because the bug was
entirely in the decode path the dump never touches.  A four-token greedy
generation caught it immediately.

For anything that touches decode, compare **greedy generations** instead: fixed
prompts, `--temp 0`, 128 tokens, byte-compared.  Decode is deterministic across
runs (verified), and any bit difference diverges within a few tokens.  The
frontier dump is still the right tool for a prefill-path change.

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
  which are BF16 here too (~1.2 GiB more per token through the LM head).  This
  was subsequently done: `--tensors head,embd` covers them, and converting the
  head is worth a further +1.80% decode over a KDA-only artifact.  `token_embd`
  is deliberately not in the default -- it is a single-row lookup per token, so
  it saves resident memory rather than decode bandwidth.
- Neither the constants recorded below nor a Q4_K KDA variant were measured.
  The tool accepts `q4_K` as a target, which would take KDA to 2.39 GiB, but
  Q4_K on attention projections is a materially bigger quality question than
  Q8_0 and was not attempted.

## Prefill, measured

Prefill had never been swept.  Five constants that shape it were compile-time
`#define`s with no override, and two of them interact, so each is now
separately settable -- `DS4_GLM_PREFILL_CHUNK_TOKENS`,
`DS4_GLM_FULL_ATTN_LAYER_FLUSH_TOKENS`, `DS4_GLM_FULL_ATTN_CAP`,
`DS4_GLM_FULL_ATTN_STREAMING_CAP`, `DS4_GLM_PREFILL_SCORE_SCRATCH_MB`.
Defaults are unchanged: logits with the knobs unset and with them set to the
old constants match at max|delta| = 0.

### Chunk size, with layer flushing held constant

The document previously warned that the chunk (2048) and the layer-flush
threshold (2048) are the same number against a strict `>`, so raising the chunk
also switches per-layer flushing on -- two changes, not one.  Pinning the flush
threshold separates them.  Prefill tok/s at ctx 16384:

| chunk | flush off | flush on | default |
|---:|---:|---:|---:|
| 1024 | 354.40 | 355.00 | 354.34 |
| 2048 | 394.30 | 394.57 | 394.17 |
| 4096 | 393.98 | 394.45 | 394.12 |
| 8192 | 394.05 | 394.43 | 393.97 |

Two results.  **Layer flushing does not matter at all** -- every column agrees
to 0.2%, so the coupling the doc warned about is real in the code and
immaterial in practice.  And **the default chunk of 2048 is already optimal**:
1024 costs 10%, while 4096 and 8192 buy nothing.  Confirmed at ctx 32768, where
2048/4096/8192 give 388.68/388.31/388.50.

Raising the chunk is not free elsewhere, either.  Context buffers at ctx 4096
grow 1.62 -> 3.04 -> 5.88 GiB across chunk 1024/2048/4096, so 4096 would cost
nearly 2 GiB for no throughput.  The GLM 5.2 path's 4096 is not an argument for
changing this one.

### The full-attention cap asymmetry is backwards

`glm_graph_full_attention_cap` gives the SSD-streaming path 8192 and the
fully-resident path 4096, which looked like the memory-constrained machine
getting the larger window.  Forcing each value on the resident path, ctx 16384,
interleaved, n=6:

| cap | prefill | decode |
|---:|---:|---:|
| 4096 | 394.23 (sd 0.03) | 23.17 (sd 0.02) |
| 8192 | 379.23 (sd 0.10) | 23.15 (sd 0.02) |

**The larger window costs 3.81% of prefill and nothing on decode.**  So 4096 is
not the conservative choice, it is the fast one, and the resident default is
right.  Whether 8192 pays for itself on the streaming path by reducing
re-streaming is untested here -- `DS4_GLM_FULL_ATTN_STREAMING_CAP` exists to
try it.

### The 256 MiB score scratch does bind, but not where it hurts yet

`DS4_GLM_METAL_INDEXED_PREFILL_SCORE_SCRATCH_MB` clamps rows scored per
dispatch.  It is not dead: score columns are `compact_cap / 4`, so the budget
starts biting above 131072 allocated context.  Observed, chunk 2048 throughout:

| ctx_alloc | score_rows | scratch |
|---:|---:|---:|
| 65536 | 2048 | 128 MiB |
| 131072 | 2048 | 256 MiB |
| 262144 | **1024** | 256 MiB |
| 524288 | **512** | 256 MiB |

Raising the budget to 1024 MiB restores 2048 rows at a 524288 allocation.  The
model context limit is 1048576, so this is reachable, not theoretical.

It does not currently cost anything measurable, though: holding the allocation
at 524288 and varying only the budget, a 16384-token prefill runs at 393.84
tok/s with score_rows=512 against 394.30 with 2048 -- **0.12%, noise**.  A
prefill long enough for scoring to dominate was not measured; each run at ctx
65536 with that allocation exceeds ten minutes.  So: the clamp is real, the
knob to lift it exists, and nobody has yet shown it matters.

## Untested constants noticed while reading

Recorded so the next person does not re-derive them.  None were measured.

All four prefill entries that used to sit here have been measured; see
"Prefill, measured" above.  In summary: the chunk default of 2048 is optimal,
per-layer flushing does not matter, the 8192 full-attention cap is 3.81% slower
than 4096 rather than more generous, and the 256 MiB score scratch does clamp
above 131072 allocated context but costs nothing measurable at the prefill
lengths tested.

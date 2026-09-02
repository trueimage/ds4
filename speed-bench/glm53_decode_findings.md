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

**The KDA row is not one of those measurements.**  `DS4_GLM_DECODE_ABLATE`
carries no `kda` bit, and the KDA path returns before the mask is even read
(`glm53_graph_kda_attention` is dispatched above `decode_ablate` in `ds4.c`),
so no ablation arm in this tree can produce it.  The 18.37 ms figure came from
instrumentation that was never committed and **has not been reproduced**.
Everything downstream of it -- KDA's share, its GB/s, and the ~7 ms floor
derived below -- inherits that.  Treat the row as an unverified estimate until
a committed KDA substage timer replaces it.  The other rows stand.

| component | ms/token | share | bytes/token | GB/s | % of ceiling |
|---|---:|---:|---:|---:|---:|
| KDA attention (34 layers) | 18.37 | 38.9% | 8.50 GiB | 497 | 67% |
| routed MoE (42 layers) | 8.08 | 17.1% | 4.43 GiB | 589 | **80%** |
| DSA attention core (11 layers) | 7.91 | 16.8% | -- | -- | -- |
| shared expert (42 layers) | 2.13 | 4.5% | 0.55 GiB | 279 | 38% |
| attn_output projection (11 layers) | 1.26 | 2.7% | 0.73 GiB | 622 | **84%** |
| q_path (11 layers) | 0.59 | 1.3% | -- | -- | -- |
| indexer (11 layers) | 0.35 | 0.7% | -- | -- | -- |
| norms, hyper-connections, residual, LM head | ~8.5 | ~18% | -- | -- | -- |

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

`gguf-tools/glm53-requant-kda` converts the 136 KDA tensors from BF16 to Q8_0
into a **new file**, copying every other byte verbatim, through the same
`quants.c` facade the other tools use.  It refuses an output that resolves to
the input (same path, hard link or symlink), because the input stays mmapped
for the whole run; there is no in-place mode.

    make -C gguf-tools glm53-requant-kda
    ./gguf-tools/glm53-requant-kda in.gguf out.gguf q8_0

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

### The projection was too optimistic, and why

Scaling KDA's measured 497 GB/s by the byte reduction predicts +22%.  The
measured result is +13.4%.  The difference is the useful part: only about 62%
of KDA's time was weight streaming.  The remaining **~7 ms/token** is the
conv1d, the gating, and the recurrent state update, none of which shrink when
the weights do.  That floor is the next thing to attack on this path, and it is
not a bandwidth problem.

Note that this arithmetic runs through the unverified 18.37 ms KDA row: the
+13.4% and the +22% projection are both measured, but turning their ratio into
a millisecond floor needs KDA's absolute time.  The 62% split is solid; the
"~7 ms" is only as good as the row it scales.  Re-derive it once KDA substage
timing lands.

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

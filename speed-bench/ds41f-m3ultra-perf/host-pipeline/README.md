# Review fixes and the decode host pipeline

Follow-up to the [first campaign](../README.md) and the
[GLM strategy transfer](../glm-transfer/README.md). The reference for every
number on this page is `c1c5a4a`, which already contains both. The GGUF, weight
precision, compiler math settings and 32,768-token allocation limit are
unchanged. Hardware: 80-core Apple M3 Ultra, 512 GiB.

Decode improved **10.8–12.2%** with the Engram tables warm in the page cache
and incremental prefill improved 1.1–4.2%. The separate reader trial E improved
cold-table decode **12.95%**; no combined cold-table sweep was run.
All 33,612,800 captured logits and all four serialized snapshots are
bit-identical to the reference, and the logits hash equals the one the two
earlier campaigns recorded.

## A correction to the earlier numbers

V4.1 reads 48 Engram rows per decode token (two tables, 24 rows each) from
regions of the GGUF that total 188.8 GiB. The tables are opened `F_NOCACHE`,
but macOS still serves those reads from the unified buffer cache when the pages
happen to be resident. The earlier campaigns were measured in that warm state.
On a 512 GiB machine with 294 GiB wired it does not last: after ordinary memory
pressure the same frozen `3c16a1d` binary that recorded 23.77 tokens/s at 8K
measured 21.12, with prefill unchanged. A step then spends 5.3 ms reading rows
one after another while the GPU is idle.

`DS4_METAL_V41_DECODE_HOST_PROFILE=1` prints this attribution (host time per
step, compute encodes and blit copies per step). To reproduce the warm state,
read both table ranges once through the cache; their offsets come from the
`blk.1.engram_embd.weight` and `blk.14.engram_embd.weight` tensor entries.

## Correctness fixes

| Finding | Fix | Check |
| --- | --- | --- |
| The histogram top-k finisher published its sorted result into the buffer its last staged bitonic stage was still reading. The intervening stages synchronise only inside a SIMD group, so another group could read a final value as a stage value. It never fired in any recorded run | Publish into the other staging buffer | `make test-deepseek41-topk`, `tests/test_glm53_topk_fast`, live-edge fixture |
| No test compared queued with unqueued decode, and every live-edge case paused the graph at its interposed calls | The live-edge fixture now ends with four uninterposed arms (original/production stages × original/production schedule), 16 steps each, comparing tokens, full logits and the final snapshot, and requiring that the histogram selector really ran inside the queued buffers | 61 accepted selections per production arm |
| The four fusions and the Q4 tail cull did not check quantization, and the router and tail cull also ran for vision sessions and imatrix collection | One admission rule: the graph publishes text-only, no imatrix, Q4_K routed experts in every layer; the backend adds device, quality, streaming and TP. The decode queue and both concurrent Engram reader branches also require the M3 Ultra device | `make test-deepseek41-engram-admission` exercises graph entry with simulated device responses, configuration exclusions and reader rollbacks; the GPU oracles cover quality, streaming and ownership refusal |
| A missing fused pipeline failed decode although the original chain is the same arithmetic | Router, HC and shared now fall back | — |
| `kernel_dsv41_hc_norm` divided by a compile-time `5120.0f` and the router clamped against a compile-time `INFINITY`, where the kernels they replace take runtime operands; under fast-math only the constant forms invite a rewrite | Both arrive as runtime arguments | `make test-deepseek41-fusions` |
| The three V4.1 GPU oracles were outside `make test` | They run in `make test` on Darwin | — |
| `ds4_gpu_dsv41_hc_norm` had no autorelease pool | Added | — |
| GLM: the graph-local exact paths (phased DSA decode, BF16 HC producer and the HC expands) had no device, streaming or TP scope, and the BF16 pair/trio/HC-expand kernels used a device substring test without the ownership exclusions | All share `ds4_gpu_glm53_tuning_available()`; the phased scores dispatch also checks the pipeline's threadgroup limit | `make test-glm53-kda`, `make test-glm53-fork` |
| GLM: the base KDA decode kernel had been edited for every device with no test against the kernel it replaced | The pre-edit kernel is kept verbatim as `kernel_glm53_kda_decode_reference`, selected by `DS4_METAL_GLM53_KDA_DECODE_REFERENCE`, and is arm 0 of the persistent-state test | bit-identical over 16 steps × 3 shapes |
| GLM: the phased decode attention returned NaN where the generic kernel returns zeros when no selected row is valid. The generic kernel computes the same softmax weights and skips invalid rows only when it accumulates | The phased accumulation stages a zero weight for an invalid row, as it already does for a row past the end. The softmax kernel is unchanged | new all-invalid and overflowing-score cases in `make test-glm53-kda` |
| GLM: the router/shared oracle's reference arm was itself the tuned top-eight selector | The reference arm runs with the tuning rolled back | `make test-glm53-fork` |
| `ds4.c`: a failed SSD-streaming output map was overwritten by the head-encode result | Guarded | — |
| `glm53-requant-bf16`: unchecked writes, no `ferror`/`fsync` before the rename, unbounded tensor name length | Checked; refuses names over 4096 bytes | byte-identical output to the previous build on a synthetic GGUF |

## Exactness review of this work

The changes above were then reviewed again for exactness only, by two
independent readings of the commit and by runs beyond what any campaign had
covered.

- **Against `main` itself, to 130,816 tokens.** Earlier evidence compared each
  campaign with its own base and stopped at 32,768. `origin/main` (`8db1d1d`),
  this branch, this branch with all eleven V4.1 rollback controls set, and this
  branch repeated each captured **455 full vocabulary rows (58,822,400 FP32
  values) through position 130,880**, at frontiers 2,044 to 130,816 with 64
  decode tokens each. The four logit files are the same file, all seven
  serialized snapshots (up to 848,748,596 bytes) are byte-identical, and no
  logit is non-finite. The rollback arm decodes at `main`'s rate, so the
  controls do restore the original paths. See
  [long-context-comparison.json](records/long-context-comparison.json).
- **Why 65K matters.** Layers 2–19 compress 2:1 and layers 20–39 1:1. The
  histogram selector admits 12,288–32,768 scores, so at a 32K context it ran at
  full width on layer 20 only. At 65,408 it runs at full width on all eighteen
  2:1 layers, with the decode tokens just inside the admission edge; at 32,704
  they cross layer 20's edge; at 130,816 every layer must refuse it.
- **End to end.** Greedy generations are byte-identical between `main` and this
  branch for five V4.1 prompts, up to a 63,848-token prompt with 400 generated
  tokens, and between `main`, `c1c5a4a` and this branch for three GLM 5.3 Flash
  Q4_K prompts, up to 8,192 tokens. See
  [generation-comparison.json](records/generation-comparison.json).
- **The schedule is asserted, not inferred.** The live-edge fixture now
  requires the published configuration, and per step exactly 41 command buffers
  under the original schedule and 3 under the pipeline; 89 blit copies per step
  with the stages rolled back and none otherwise.

The readings found no way for the measured configuration to produce different
bits, and led to these changes:

- The GLM invalid-row fix moved from the softmax to the accumulation (table
  above): the first version returned zeros where the generic kernel returns NaN
  if a valid row's score is itself negative infinity.
- The BF16-store matvec rounds a stored copy of the reduced value. The dispatch
  it replaces rounded words loaded from memory; a value straight from a
  fast-math reduction could in principle lose its non-finite test. The oracle
  gained NaN scales whose rounding would change their payload. The historical
  count of 59,530 non-finite outputs was taken after BF16 conversion, which
  clears the low word; it cannot establish GPU NaN canonicalization or exclude
  a rounding-sensitive raw payload. The oracle now inspects the reference
  matvec output before conversion and reports how many non-finite values
  would change their upper word under round-to-nearest-even. Exact output
  comparisons remain required regardless of that count.
- The published configuration is cleared when a graph is freed.
- The Engram step test uses two distinct tables and differing row IDs, so a
  table, ID or output mix-up cannot pass.

Known and accepted: an Engram row that fails to read or decode now surfaces
after layer 0 has been committed, so the graph is marked invalid and the
session must be restored. Before, it returned before any GPU work. Row IDs are
still validated first, and no output is ever produced from a failed read. With
the test flag set on a device whose pipelines allow fewer than 1,024 threads,
the phased GLM decode now fails cleanly where it used to issue an invalid
dispatch; the graph no longer selects it off the measured configuration.

## Performance trials

Each screen is `metal_decode_schedule_bench` at an 8,192-token prefix, 16 warmup
and 256 measured steps per arm, alternating variant order and session per token.
Every screen compared 273 complete vocabulary rows (35,293,440 floats) and 272
greedy selections with no tolerance. Gains are incremental, in the order below.
Trial B was measured again after the later exactness change to its store:
28.4443 → 28.6541 (+0.74%).

| ID | Attempt | Original → new, tokens/s | Decision |
| --- | --- | --- | --- |
| E | Issue a step's 48 Engram reads together | cold tables 22.7560 → 25.7025 (+12.95%); warm 25.6687 → 25.8332 (+0.64%) | Keep |
| P | Pipeline the decode step: no drain at layer 13 (the second table gets its own input buffer), non-blocking commits after layers 0 and 4, output head in the last buffer | 25.8300 → 27.5617 (+6.70%) | Keep |
| O | Let the Engram read arrive while layer 0, which does not use it, is encoded and started | 27.5548 → 27.7264 (+0.62%); 27.4996 → 27.7022 (+0.74%) | Keep |
| C | Small tensor copies as a dispatch in the running encoder instead of a blit, which ends it (89 per token) | 27.6742 → 28.4590 (+2.84%) | Keep |
| B | Apply the BF16 boundary where a Q8_0 projection stores its row, instead of a second dispatch | 28.4562 → 28.6892 (+0.82%); 28.3771 → 28.5901 (+0.75%) | Keep |

P, O and C change no arithmetic: the same dispatches in the same order. After P
the GPU has no gap inside a token; the step's one remaining wait is the final
one. E and O read the same bytes and decode each row with the same function. B
instantiates the same matvec template with the rounding at the store; the walk
and reduction tree are the plain kernel's.

Trial B also sizes what is left: removing about 200 tiny dispatches per token
bought under 1%, so each remaining BF16 boundary (attention heads, low
projection, the two HC expands, the routed sum) is worth roughly 0.15%.

Not attempted, with reasons:

- **Concurrent router, shared and routed experts.** Trial C measured what an
  encoder boundary costs: 178 of them were 2.84% of a token. Bracketing a
  concurrent section in every layer would spend about as much as the overlap
  returned on the Q2 model (+1.22%). Only an always-concurrent encoder with
  explicit barriers, as MLX and llama.cpp have, would win, and that is a
  hazard analysis of some 1,900 dispatches per token, not a local change.
- **Function constants for the matvec batch divisors.** Integer index
  arithmetic once per thread, against a 5,120-wide dot product.
- **`AGX_RELAX_CDM_CTXSTORE_TIMEOUT` and capped residency sets.** Robustness
  changes whose benefit could not be reproduced here.
- **MLX's Ultra GEMM tile constants.** A GLM prefill experiment; this campaign
  ran only the V4.1 model.

## Combined performance

Uninstrumented `ds4-bench` processes in reference/final/final/reference order,
Engram tables warm, four corpus frontiers with 64 greedy tokens each. Rates
aggregate total tokens over total time across the two runs of each build.

| Context | Reference → final prefill t/s | Change | Reference → final decode t/s | Change |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 273.35 → 284.03 | +3.91% | 25.89 → 28.93 | +11.73% |
| 2,048 | 335.91 → 350.14 | +4.24% | 25.93 → 29.08 | +12.15% |
| 8,192 | 355.37 → 367.92 | +3.53% | 25.97 → 28.76 | +10.75% |
| 32,703 | 662.83 → 670.40 | +1.14% | 25.17 → 28.23 | +12.16% |

The cold-table state was measured only for trial E; a combined cold sweep was
not run. These are local observations for one model, prompt and machine.

## Exactness

- Reference and final builds captured **260 full vocabulary rows (33,612,800
  FP32 values), all bit-identical**, through position 32,767, covering prefill,
  64 decode tokens at each frontier and continued prefill after restoration.
  The hash is `e022c41d…493c`, the same value both earlier campaigns recorded.
  See [capture-comparison.json](records/capture-comparison.json).
- All four serialized snapshots match byte for byte, including 220,441,392
  bytes at 32,703. See [snapshot-comparison.json](records/snapshot-comparison.json).
- `make test-deepseek41-fusions` gains 140 BF16-store matvec cases over V4.1's
  Q8_0 decode shapes (infinite, NaN and subnormal scales; exceptional
  activations; guards) and 160 copy cases (NaN payloads, signed zeros, offsets,
  guards, and the unaligned, large and same-buffer copies that keep the blit).
  The oracle disables Metal4 before initialization so both arms use the legacy
  Metal backend, including on devices that enable TensorOps by default.
  `make test-engram` compares the step reader, split and unsplit, with the
  ordinary reader, including duplicates, validation before output, and error
  propagation. All run under Metal API validation.
- The live-edge fixture passes with every new rollback in its stage and schedule
  matrices.
- `ds4_test` reports the same nine failures before and after: its default
  model link points at the V4.1 GGUF while those vectors target V4. Because
  `make test` stops there, the rest of its recipe was run directly and passes.

## Scope and rollback

Everything is automatic only in the measured configuration: resident, single
device, text-only, no imatrix, Q4_K routed experts, M3 Ultra, normal precision.
SSD streaming, TP, vision, other devices and other quantizations keep the
previous schedule and kernels. The pipeline adds one 24 KiB input buffer per graph.

| Stage | Rollback control |
| --- | --- |
| Concurrent Engram step read | `DS4_DISABLE_V41_ENGRAM_STEP_READERS` |
| Read overlapped with layer 0 | `DS4_DISABLE_V41_ENGRAM_STEP_OVERLAP` |
| Pipelined decode schedule | `DS4_METAL_DISABLE_V41_DECODE_PIPELINE` |
| In-encoder copy | `DS4_METAL_DISABLE_V41_COMPUTE_COPY` |
| BF16-store Q8_0 matvec | `DS4_METAL_DISABLE_V41_MATVEC_BF16` |

Reproduce a screen with the command in the [first campaign](../README.md#reproduction),
substituting the rollback control. [paired-results.json](records/paired-results.json)
holds the component rates and [records/](records/) the sweep CSVs.

# DeepSeek V4.1 Flash Q4 on M3 Ultra

The [current-main comparison](main-comparison/README.md) records fresh
ABBA measurements and bit-identical full-logit/session checks against the
PR base `0aaea5a` for both official GGUFs, through 32K. The
[clean-849e039 comparison](splash/vs-849e039/README.md) separately measures
the latest Splash decode and HTTP checkpoint latency gains.

The [Splash review implementation](splash/README.md) records the subsequent
DeepSeek Q4 and GLM Q4_K exactness, decode, verification and checkpoint work
within a 32K context limit.

See the subsequent [GLM strategy transfer campaign](glm-transfer/README.md) for
the follow-up fusion and indexer attempts, and the
[review fixes and decode host pipeline](host-pipeline/README.md) for the later
correctness fixes, a further 10.8–12.2% decode, and a note on the Engram
page-cache state in which the numbers below were measured.

Two of five attempts were retained: queued resident decode and Q4 expert-tile
culling. The final combined sweep shows **26.8–28.4% faster decode** and
**2.1–12.3% faster incremental prefill**, depending on context. All tested
full-model logits and saved session states are bit-identical to the original
implementation. The isolated paired measurements below distinguish the two
optimizations and include cold-prefix prefill.

Branch: `ds41f-m3ultra-perf`. Base: `76f3609347a7a613e76a0611ff95a7a26efe00c2`
(the rebased `glm53flash-metal-exact` branch). Hardware: 80-core Apple M3 Ultra,
512 GiB RAM. Model: `gguf/DeepSeek-V4.1-Flash-Q4.gguf`, 483 GiB including
SSD-backed Engram tables; 294.14 GiB of resident main weights. See
[manifest.json](manifest.json) and [hardware.txt](hardware.txt). The model prefix
hash identifies the local artifact; it is not a complete-file checksum.

## Attempts and decisions

| ID | Attempt | Exactness evidence | Measured effect | Decision |
| --- | --- | --- | --- | --- |
| Q | Queue resident decode between Engram input-reuse boundaries | Every full-vocabulary logit and greedy selection matched at short, 8K and near-32K contexts | +24.16% to +26.27% decode | **Keep** |
| E | Batch first-layer text embeddings | Eight 8K prefill runs matched full logits; existing scalar/batch embedding oracle passed | +0.14% at 8K, with additional embedding scratch | **Reject**: no convincing model-level gain |
| G4 | Four activation-quantization SIMDgroups per threadgroup | Tensor outputs and guards matched, including exceptional FP values; eight 8K model runs matched | Faster large-tensor microbenchmarks; +0.06% model prefill | **Reject**: no convincing model-level gain |
| G8 | Eight activation-quantization SIMDgroups per threadgroup | Same tensor oracle and eight exact 8K model runs | Faster large-tensor microbenchmarks; +0.20% model prefill | **Reject**: no convincing model-level gain |
| M | Skip unused Q4 matrix work in partial expert tiles | Full prefill logits, consumed F16 intermediates, F32 outputs and buffer guards matched | +12.49% at 512, +3.22% at 8K, +1.85% near 32K | **Keep** |

The rejected attempts stopped after the repeated 8K screen. They were not
promoted to long-context testing or left in production dispatch. Their source
and microbenchmarks are archived under [experiments/](experiments/).

The 8K synchronized stage profile attributed 5.45 s to shared/routed FFN and
3.92 s to attention/indexing, versus 1.31 s for attention output. This motivated
attempt M. [profile.json](profile.json) is diagnostic attribution: the extra
synchronization makes it unsuitable for throughput claims.

## Paired throughput

Each row compares variants in one engine, with one model process running.
Decode alternates both variant order and session assignment for every token,
with 16 warmup and 256 measured tokens per variant. Prefill uses fresh sessions
in ABBA/BAAB order. The 512 and 8K prefill cells have four measured runs per
variant; near 32K has two. Both variants warm the complete prefix for the 512
and near-32K prefill cells; the 8K screening cell warms 32 tokens per variant.

| Operation | Prefix tokens | Baseline tokens/s | Candidate tokens/s | Change |
| --- | ---: | ---: | ---: | ---: |
| Decode Q | 512 | 18.8297 | 23.7764 | +26.27% |
| Decode Q | 8,192 | 18.6141 | 23.4690 | +26.08% |
| Decode Q | 32,495 | 18.3509 | 22.7848 | +24.16% |
| Cold-prefix prefill M | 512 | 249.8343 | 281.0450 | +12.49% |
| Cold-prefix prefill M | 8,192 | 632.4444 | 652.8122 | +3.22% |
| Cold-prefix prefill M | 32,703 | 685.4556 | 698.1639 | +1.85% |

Raw commands and aggregates are in [paired-results.json](paired-results.json).
These are local observations, not confidence intervals or guarantees for other
hardware, quantizations, prompts or concurrent serving workloads.

The final combined benchmark runs original/final/final/original processes,
without logit capture or profiling. Its CSVs measure **incremental** prefill
intervals between 512, 2,048, 8,192 and 32,703 tokens; these are distinct from the
cold-prefix numbers above. Each frontier includes 64 greedy decode tokens and
restores the same corpus prefix before continuing. Each aggregate is total
tokens divided by total elapsed time across its two runs (the harmonic mean
of the CSV rates, which are rounded to two decimals).

| Context | Prefill interval | Original → final prefill t/s | Change | Original → final decode t/s | Change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 512 | 246.55 → 276.96 | +12.33% | 18.90 → 24.27 | +28.40% |
| 2,048 | 1,536 | 318.10 → 335.99 | +5.62% | 18.94 → 24.02 | +26.82% |
| 8,192 | 6,144 | 339.54 → 355.17 | +4.60% | 18.69 → 23.80 | +27.35% |
| 32,703 | 24,511 | 648.48 → 662.12 | +2.10% | 18.31 → 23.42 | +27.91% |

See [combined-results.json](combined-results.json) and the raw CSVs/commands in
[combined/](combined/). No additional swap was used during the campaign.

## Exactness and regression checks

- A clean checkout of the original commit and the final implementation captured
  260 complete FP32 vocabulary rows: **33,612,800 bit-identical logits** through
  position 32,767. The captures include prefill, 64 decode tokens at each
  frontier, and continued prefill after snapshot restoration. See
  [logit-comparison.json](logit-comparison.json).
- All four complete serialized session snapshots matched byte for byte at
  prefixes 512, 2,048, 8,192 and 32,703, including 220,441,392 bytes at the last
  frontier. See [snapshot-comparison.json](snapshot-comparison.json).
- The separate decode comparisons checked another 273 complete rows per
  context and identical greedy selections, with no tolerance.
- The new `make test-deepseek41-q4-tail` fixture checks hot, sparse and empty
  expert routes around 16/32-row tile boundaries, finite intermediates and
  outputs, and untouched buffer guards. A test-only dispatch counter proves
  the optimized path runs. It also proves streaming excludes that dispatch.
- The campaign's full V4.1-shaped Q4 fixture passed under Metal API validation.
  The permanent fixture uses smaller dimensions to remain suitable for routine
  testing. Both compare consumed F16 intermediates and final F32 outputs.
- Existing V4.1 Metal primitive and GGUF parser tests passed. The real Q4 SSD
  session fixture passed with Metal API validation, including prefix reuse,
  snapshot restoration, cancellation, and bounds. `make all` and the CPU
  object build passed.

Exactness means equality to this branch's original implementation; it is not
an independent assessment of the model's accuracy or its quantization recipe.
Capture timings are excluded from throughput results.
The commands, log hashes and final production-source hashes are retained in
[validation.json](validation.json).

## Retained implementation and scope

Queued decode applies automatically to resident, single-device, text-only
V4.1 Q4 on M3 Ultra, excluding quality/imatrix diagnostics. It retains the
wait before layer 14 overwrites the shared Engram input and the final wait
before publishing a token. SSD, TP, other devices and other quantizations keep
the previous decode schedule.

Q4 tile culling applies automatically to the resident V4.1 expert shape on
M3 Ultra. It retains all staging and barriers, and skips only matrix operations
for padded rows that cannot contribute an output. The existing accumulation
order stays intact. SSD and TP keep their previous dispatch. CUDA code is
unchanged. Vision and distributed workloads were not benchmarked in this
campaign; no remote GPU jobs were run.

A file whose routed experts are all native MXFP4 is now admitted to the same
paths, except the Q4_K tail cull; its exactness evidence, throughput and the
V4.1 DSpark measurements are in [mxfp4/README.md](mxfp4/README.md).

Two diagnostic rollback controls remain for reproducing the original paths:
`DS4_METAL_DISABLE_V41_RESIDENT_DECODE_QUEUE=1` and
`DS4_METAL_DISABLE_V41_Q4_TAIL_CULL=1`. Normal use requires neither.

## Reproduction

Run from the repository root. Keep other GPU workloads idle. `run.py` removes
inherited `DS4_`, `MTL_` and `ASTRA_` controls, records source hashes and commands,
and refuses to overwrite existing evidence. Use a fresh output/name for each
run. All full-model allocations and evaluated positions must stay at or below
32,768; the largest prefixes leave room for generation and the sentinel slot.
The command below uses the existing GGUF's absolute path from
[manifest.json](manifest.json), so it also works from a separate source worktree
without copying or downloading the model. Use that path for `-m` when replaying
the historical commands that use a checkout-relative model path.

```sh
make -j8 all metal-decode-schedule-bench metal-prefill-variant-bench
make test-deepseek41-q4-tail
python3 speed-bench/ds41f-m3ultra-perf/run.py \
  --out /tmp/ds41f-recheck --name decode-8k -- \
  ./speed-bench/metal_decode_schedule_bench \
  -m /Users/jw/ds4/gguf/DeepSeek-V4.1-Flash-Q4.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --candidate-env DS4_METAL_DISABLE_V41_RESIDENT_DECODE_QUEUE \
  --prefix-tokens 8192 --ctx 32768 --warmup 16 --tokens 256 --include-selection
```

With a rollback variable, the harness's **control is the optimized default**
and its **candidate is the original path**, so its reported delta is negative.
The historical table above instead labels the original path as baseline.
For prefill, use `metal_prefill_variant_bench` with the Q4 tail rollback,
`--prefix-tokens 8192 --ctx 32768 --prefill-chunk 8192 --repeats 2`.

To reproduce the exploratory enable flags in `paired-results.json`, create a
separate checkout at the base commit, apply `experiments/candidates.patch`,
using `git apply --unidiff-zero`, then build there. Do not apply that patch to
the final branch. Run every binary
from its matching checkout: Metal shaders are loaded relative to that tree.
The two C files in `experiments/` are standalone GPU fixtures for that patched
checkout and link against the same core objects and Metal frameworks as the
existing benchmarks.

`build_capture.py OUTPUT_DIR` builds a temporary capture harness from the
unchanged public `ds4-bench` session API. Set `DS4_BENCH_TRACE_LOGITS` and,
optionally, `DS4_BENCH_SNAPSHOT_DIR`; create the latter directory first. Use
`compare_capture.py CONTROL.bin CANDIDATE.bin` to require identical row headers
and every FP32 bit. Use `DS4_BENCH_FORCE_SNAPSHOT=1` for the continued sweep.

Complete local logs, captures, snapshots, build output and command metadata:
`/tmp/ds41f-m3ultra-perf-20260918/`. The unmodified reference checkout is
`/tmp/ds41f-baseline-76f3609/`. Large captures remain local; compact evidence and
reproducible commands are retained in this directory.

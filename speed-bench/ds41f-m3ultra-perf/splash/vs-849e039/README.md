# Clean 849e039 versus final Splash f768551

**Conclusion:** keep the validated changes for DeepSeek's **2.6–3.5% decode
gain** and shorter checkpoint-related HTTP TTFT. Bulk prefill is effectively
unchanged on both models; GLM decode stays within 0.10% of baseline.

The 10,240-token checkpoint continuation saves **13.9 ms for DeepSeek** and
**60.7 ms for GLM**. Near-32K cold requests save 104.0 ms and 126.3 ms respectively
in these two pairs. Those cold-request differences are small relative to total
latency and show more variation, particularly GLM's baseline (138.3 ms between
its repeats); they are workload-specific observations, not general prefill gains.

Exactness passed for **216,637,440 FP32 logit values**, **12 complete snapshots**
and **32 HTTP responses**. HTTP cache files match after masking only timestamps.

This comparison directly measures the pair requested by the user:

| Arm | Exact commit | Checkout |
| --- | --- | --- |
| Base | `849e039b4c8048d73423c12cdf201ccbfee8528b` | `/Users/jw/.cache/ds4-bench/wt-849e039b4c80` |
| Splash | `f768551d842942f5eed4b257f6c997f8eb5ec29e` | `/tmp/ds4-splash-20260926/final` |

The candidate is the complete saved final working-tree snapshot. It includes
the changes already present before Splash, as well as Splash itself. This is
therefore the total difference from clean `849e039`, rather than an isolated
measurement against the pre-Splash dirty-tree snapshot `280366e`.

At measurement time the user's branch remained at `849e039`; this comparison
did not modify it. The later PR consolidation preserves the measured runtime
files byte for byte. Large raw logit and checkpoint images stay in the local
run directory; their hashes, comparison results and commands are retained here.

<!-- generated-results:start -->
## Results

These are fresh measurements of **clean `849e039` versus Splash `f768551`**.
Positive throughput changes are faster; positive TTFT savings are shorter waits.

### DeepSeek V4.1 Flash Q4

| Frontier | New prefill tokens | Prefill tokens/s, base → Splash | Change | Decode tokens/s, base → Splash | Change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 512 | 286.12 → 286.70 | +0.20% | 29.19 → 30.07 | +3.00% |
| 1,024 | 512 | 275.22 → 274.95 | -0.10% | 29.38 → 30.23 | +2.88% |
| 2,048 | 1,024 | 336.04 → 336.63 | +0.18% | 29.30 → 30.16 | +2.94% |
| 4,096 | 2,048 | 371.50 → 371.59 | +0.02% | 29.26 → 30.03 | +2.61% |
| 8,192 | 4,096 | 366.23 → 366.57 | +0.09% | 28.99 → 29.91 | +3.15% |
| 16,384 | 8,192 | 654.16 → 654.75 | +0.09% | 28.73 → 29.73 | +3.48% |
| 32,639 | 16,255 | 613.23 → 613.24 | +0.00% | 28.45 → 29.23 | +2.74% |

Total measured incremental prefill time: 62.4238 → 62.3935 seconds (-0.049%).

| Frontier | Prefill repeat spread, base / Splash | Decode repeat spread, base / Splash |
| ---: | ---: | ---: |
| 512 | 0.101% / 0.140% | 0.308% / 0.266% |
| 1,024 | 0.018% / 0.375% | 0.102% / 0.463% |
| 2,048 | 0.009% / 0.027% | 0.410% / 0.398% |
| 4,096 | 0.092% / 0.027% | 0.376% / 0.133% |
| 8,192 | 0.112% / 0.044% | 0.103% / 0.802% |
| 16,384 | 0.133% / 0.000% | 0.487% / 0.605% |
| 32,639 | 0.052% / 0.095% | 0.351% / 0.411% |

Repeat spreads are descriptive ranges from two runs per arm, not confidence intervals.

| HTTP case | Actual input tokens | TTFT ms, base → Splash | Saved ms | Individual base runs, ms | Individual Splash runs, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10240 cold | 10,166 | 17945.32 → 17887.09 | +58.23 | 17948.14 / 17942.50 | 17901.95 / 17872.23 |
| 10240 continuation | 10,240 | 326.35 → 312.43 | +13.92 | 326.12 / 326.57 | 312.93 / 311.93 |
| 32000 cold | 31,994 | 44452.19 → 44348.17 | +104.02 | 44463.07 / 44441.32 | 44317.62 / 44378.72 |
| 32000 continuation | 32,068 | 313.54 → 310.88 | +2.66 | 313.42 / 313.66 | 311.07 / 310.69 |

**Exactness:** 116,739,840 FP32 logit values across 903 rows are bit-identical, through position 32,767. All 7 complete snapshots match byte-for-byte. All 16 HTTP responses have matching output, usage and finish reasons; all saved cache images match except their two header timestamps.

### GLM-5.3 Flash Q4_K

| Frontier | New prefill tokens | Prefill tokens/s, base → Splash | Change | Decode tokens/s, base → Splash | Change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 2,048 | 511.72 → 512.08 | +0.07% | 26.94 → 26.94 | -0.02% |
| 4,096 | 2,048 | 436.98 → 436.25 | -0.17% | 26.80 → 26.80 | +0.00% |
| 8,192 | 4,096 | 436.92 → 436.79 | -0.03% | 26.68 → 26.70 | +0.06% |
| 16,384 | 8,192 | 433.64 → 433.62 | -0.00% | 26.57 → 26.55 | -0.06% |
| 32,639 | 16,255 | 426.77 → 426.84 | +0.02% | 26.40 → 26.38 | -0.09% |

Total measured incremental prefill time: 75.0430 → 75.0453 seconds (+0.003%).

| Frontier | Prefill repeat spread, base / Splash | Decode repeat spread, base / Splash |
| ---: | ---: | ---: |
| 2,048 | 0.164% / 0.045% | 0.186% / 0.148% |
| 4,096 | 0.007% / 0.419% | 0.037% / 0.037% |
| 8,192 | 0.046% / 0.034% | 0.037% / 0.075% |
| 16,384 | 0.007% / 0.058% | 0.075% / 0.113% |
| 32,639 | 0.056% / 0.033% | 0.038% / 0.076% |

Repeat spreads are descriptive ranges from two runs per arm, not confidence intervals.

| HTTP case | Actual input tokens | TTFT ms, base → Splash | Saved ms | Individual base runs, ms | Individual Splash runs, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10240 cold | 10,166 | 22723.48 → 22677.22 | +46.26 | 22733.94 / 22713.01 | 22678.88 / 22675.56 |
| 10240 continuation | 10,240 | 387.63 → 326.97 | +60.66 | 385.78 / 389.48 | 327.53 / 326.41 |
| 32000 cold | 31,994 | 73840.29 → 73713.99 | +126.29 | 73909.44 / 73771.13 | 73716.50 / 73711.48 |
| 32000 continuation | 32,068 | 282.54 → 281.70 | +0.83 | 282.87 / 282.21 | 282.23 / 281.18 |

**Exactness:** 99,897,600 FP32 logit values across 645 rows are bit-identical, through position 32,767. All 5 complete snapshots match byte-for-byte. All 16 HTTP responses have matching output, usage and finish reasons; all saved cache images match except their two header timestamps.

### Limits

This comparison covers the two named GGUFs on resident Metal and the recorded
corpus/HTTP prompts. It does not establish universal performance or exactness
for other prompts, models, contexts above 32K, SSD streaming, CUDA or distributed
hardware. No production source changed as part of this comparison.

<!-- generated-results:end -->

## Measurement method

- Apple M3 Ultra, 80 GPU cores, 512 GiB RAM; macOS 27.0 (26A428).
- Both benchmark/server binaries rebuilt with `make -B -j8 ds4-bench ds4-server`
  and Apple clang 21.0.0. Each process runs in its own source/shader checkout.
- Only `DeepSeek-V4.1-Flash-Q4.gguf` and `GLM-5.3-Flash-Q4_K.gguf` are used.
  Normal resident Metal, default arithmetic, SSD streaming off, no drafter.
- One large model/GPU process at a time. DS4/Metal diagnostic overrides are
  removed from timing processes. Models and prompts are recorded in the manifest.
- Every allocated context is **32,768**. Throughput sweeps stop at **32,639**,
  generate 128 greedy tokens and leave the required spare context slot.
- Full-logit and checkpoint captures run before timing, warming both arms.
  Capture timings include instrumentation and are excluded from speed results.
- Timing order per model: **base / Splash / Splash / base**. Both individual
  repeats are retained; summary rates use harmonic means and show repeat spread.

DeepSeek frontiers: 512, 1,024, 2,048, 4,096, 8,192, 16,384 and 32,639.
GLM frontiers: 2,048, 4,096, 8,192, 16,384 and 32,639. Each sweep restores the
prompt checkpoint after generation and prefills the next extension. Prefill
throughput therefore counts newly added prompt tokens, not the entire frontier.

The benchmark's `gen_first_ms` is the first decode evaluation after prefill.
It is reported separately from actual HTTP time to first token.

## HTTP method

Each arm starts a fresh server and disk-cache directory. Requests use streaming,
temperature zero, 64 output tokens and a 32K allocation. TTFT starts before
the HTTP request and ends at the first nonempty content/reasoning chunk. The
server/model is already loaded. "Cold" denotes a prompt without a reusable
full prefix, not a cold model load.

Two fixed prompts are tested per server, each followed by a live continuation:

- A 10,166-token cold request whose continuation reaches the default continued
  checkpoint boundary at **10,240** tokens. The server log must confirm that
  it writes this continued checkpoint.
- A **31,994-token** cold request and its continuation, both within 32K.

The two prompt families have distinct leading text. No staged/direct rollback
switch is used: the comparison measures each commit's default behavior.
Generated output, usage and finish reason are compared across all four rounds.
Disk-cache images must match after masking only the two header timestamps at
bytes 24–39. Request and checkpoint-store timings are retained in the raw logs.

The commits' `save=` log fields have different timing boundaries: clean
`849e039` starts its timer **after** staging the payload, while Splash starts
before staging/direct serialization. Those fields must not be compared as
equivalent complete-store timings. The HTTP request-to-first-content clock
covers the complete request in both commits and is the primary TTFT evidence.

## Evidence

`manifest.json` identifies every source file, binary, command, commit, model
metadata and completed run. `exactness-v41.json` and `exactness-glm.json`
record full FP32-bit comparisons and complete checkpoint hashes. `results.json`
contains throughput, HTTP latency, repeat spread and checkpoint-store records.

Raw captures are in `exact-*-base/` and `exact-*-splash/`. Uninstrumented
throughput is in `perf-*.csv`. HTTP responses and server logs are in `http-*/`.
Build logs and `compared-runtime.diff` retain the build and source evidence.

Reproduce with the retained scripts and the same clean checkouts:

```sh
python3 run_comparison.py --model v41
python3 run_comparison.py --model glm
python3 summarize.py
```

The runner verifies the saved source/binary identity and skips only previously
successful completed runs. It does not silently reuse a failed measurement.

The `original-local-artifact-sha256.json` inventory describes the complete local
run, including large captures. `artifact-sha256.json` hashes this committed
subset; the report wording was updated only to explain the later consolidation.

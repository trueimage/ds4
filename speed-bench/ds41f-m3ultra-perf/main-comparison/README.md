# Current main versus the final Flash optimizations

Measured on 2026-09-26 against upstream `main` **`0aaea5a`**, the base of PR #1090.
The final measured snapshot is `f768551`; the consolidated implementation at
`60bf722465ec124788b7c719f205cdd6dc7e21e1` has byte-identical runtime sources and Makefile.
History consolidation changes commit boundaries, not the measured implementation.

Apple M3 Ultra, 80 GPU cores, 512 GiB, macOS 27.0. Only the upstream maintainer’s
`DeepSeek-V4.1-Flash-Q4.gguf` and `GLM-5.3-Flash-Q4_K.gguf` were used.
Resident Metal, default tuning, no SSD streaming or drafter; Engram tables warmed
before the DeepSeek runs. One model process at a time.

Each model ran **main / final / final / main**, two processes per arm, with
128 greedy decode steps per frontier and a fixed 32,768-token allocation.
The largest frontier is 32,639, leaving space for generation and the sentinel.
Rates are harmonic means of the two rounded CSV rates. Prefill is incremental
between frontiers; the table shows the work in each interval. These are local
measurements of one corpus, not confidence intervals or cold-cache guarantees.

## DeepSeek V4.1 Flash Q4

| Frontier | New prefill tokens | Prefill t/s, main → final | Change | Decode t/s, main → final | Change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 512 | 248.57 → 287.15 | +15.52% | 19.11 → 29.91 | +56.52% |
| 1,024 | 512 | 240.77 → 276.13 | +14.69% | 19.15 → 30.11 | +57.23% |
| 2,048 | 1,024 | 304.54 → 338.31 | +11.09% | 18.77 → 29.93 | +59.42% |
| 4,096 | 2,048 | 344.38 → 373.21 | +8.37% | 18.64 → 29.94 | +60.58% |
| 8,192 | 4,096 | 342.58 → 367.60 | +7.30% | 18.47 → 29.86 | +61.69% |
| 16,384 | 8,192 | 629.24 → 658.28 | +4.62% | 18.31 → 29.41 | +60.65% |
| 32,639 | 16,255 | 589.21 → 616.43 | +4.62% | 18.19 → 29.09 | +59.88% |

**Exactness against main:** 116,739,840 FP32 logit values across 903 complete
vocabulary rows match bit for bit, through position 32,767.
All 7 complete session snapshots match byte for byte.
Logit SHA-256: `da1baae28ee696678126fa7b65caf1aa388a47f2df291d2671a6c7bb2d4d7c64`.

Largest within-arm repeat spread: 0.30% prefill and 3.76% decode.

## GLM 5.3 Flash Q4_K

| Frontier | New prefill tokens | Prefill t/s, main → final | Change | Decode t/s, main → final | Change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 2,048 | 431.24 → 512.71 | +18.89% | 20.99 → 26.87 | +28.04% |
| 4,096 | 2,048 | 380.45 → 437.98 | +15.12% | 20.87 → 26.73 | +28.10% |
| 8,192 | 4,096 | 380.36 → 437.92 | +15.13% | 20.80 → 26.60 | +27.88% |
| 16,384 | 8,192 | 377.64 → 434.78 | +15.13% | 20.75 → 26.49 | +27.66% |
| 32,639 | 16,255 | 372.00 → 428.17 | +15.10% | 20.63 → 26.36 | +27.78% |

**Exactness against main:** 99,897,600 FP32 logit values across 645 complete
vocabulary rows match bit for bit, through position 32,767.
All 5 complete session snapshots match byte for byte.
Logit SHA-256: `6b8700c52a925bcfc4d393416b56be9a49c58ee2d363b23844cafdf406f6933c`.

Largest within-arm repeat spread: 0.10% prefill and 0.29% decode.

## Checkpoint latency and additional checks

The separate [clean-849e039 comparison](../splash/vs-849e039/README.md) isolates
the current changes from the previously published branch tip. At the 10,240-token
checkpoint continuation, HTTP time to first content fell by **13.9 ms for DeepSeek**
and **60.7 ms for GLM**. Those are measured HTTP TTFT savings against `849e039`,
not against main; bulk prefill stayed flat in that incremental comparison.
All 32 HTTP responses matched, as did 40 cache images after masking only timestamps.

The [Splash record](../splash/README.md) includes zero-tolerance GPU oracles,
the independent 637-million-logit GLM HC soak, verification-prefix parity,
SSD checkpoint save/load tests, bounded server tests and CPU builds.
Older campaign pages retain their original baselines and historical context limits.
This refresh did not run above 32K or test another model, CUDA or physical TP.

## Evidence and reproduction

`manifest.json` records all commands, model identities, source and binary hashes.
`results.json` retains both raw rates, aggregate rates and repeat spreads.
Raw performance CSVs, logs and exactness comparison results are adjacent.
Instrumented capture times are excluded from throughput results. The captures
compare fresh main output with the saved, identically configured final captures
from the immediately preceding clean-849e039 comparison. Large logit/snapshot
images remain local; complete hashes identify them.

Build `ds4-bench` separately at main and at the consolidated implementation, and
run each binary from its matching checkout so it loads that checkout’s shaders.
Clear diagnostic DS4/MTL/METAL/ASTRA environment overrides. Use `run.py` as the
record of the original absolute paths, or repeat this command in ABBA order:

```sh
./ds4-bench -m /path/to/DeepSeek-V4.1-Flash-Q4.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 512 --ctx-max 32639 --ctx-alloc 32768 \
  --step-mul 2 --gen-tokens 128 --csv /path/to/run.csv
```

For GLM, use its named GGUF and `--ctx-start 2048`. The public benchmark’s
`gen_first_ms` is the first decode evaluation after prefill; it is not HTTP TTFT.

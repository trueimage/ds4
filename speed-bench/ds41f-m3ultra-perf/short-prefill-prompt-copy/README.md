# Short prefill and prompt-copy results

Matched M3 Ultra measurements on 2026-10-12 compare refreshed main
`fc80bd695da76ee14bcb86820bf900a5cdfda806`, the preceding performance-branch
commit `f83ecfe3f28d919c32453ccd8f3104320aee93ac`, and three isolated candidates.
The earlier experiment percentages used **f83ecfe** as the baseline.

| Improvement | Measured workload | Over main | Over f83ecfe |
| --- | --- | ---: | ---: |
| Pack Q4 expert work into smaller GPU tiles | 128/256/512-token fresh prefill, code and prose | +16.98–26.57% | +1.03–2.84% |
| Pack native MXFP4 expert work and skip unused tile calculations | 128/512-token fresh prefill, code and prose | +16.51–25.07% | +13.51–23.97% |
| Copy likely continuations from earlier text, checking every token with the model | Greedy Python/JSON edit fixtures | +147.03–148.26% | +46.27% |

The prompt-copy prose controls were effectively unchanged versus f83ecfe
(−0.07% / +0.02%). These are separate, workload-specific measurements, not
additive gains or a benchmark of all three enabled together. Prompt-copy was
measured at the generation loop, not as HTTP request latency.

## Measurement protocol

The host was an 80-core M3 Ultra with 512 GiB, macOS 27.0.1 and Apple clang 21.
Models were DeepSeek V4.1 Flash Q4, native MXFP4, and GLM 5.3 Flash Q4_K.
Only one model process ran at a time. Builds, loading, Engram warming and
correctness capture work were outside inference timers. Ordinary desktop
activity remained present; two process blocks describe observed variation,
not statistical confidence.

Fresh-prefill order was main → f83ecfe → candidate → candidate → f83ecfe → main.
Each shape/corpus had two full-size warmups and four timed fresh sessions per
process, with corpus/shape order reversed in the second block. Each arm thus
had eight timed samples. Timing covered completed `ds4_session_sync` calls.

Prompt-copy order was main → f83ecfe → f83ecfe → main, with reversed scalar/copy
order inside the branch processes and four clean samples per arm/fixture.
All four fixtures used context 4096, placement hint 2 and a 256-token cap.
The measured loop included selection, proposal lookup, target verification,
accepted-prefix commit and bookkeeping. An untimed scalar trajectory supplied
the output budget. Python/JSON edits produced 181/225 tokens; the prose controls
produced 256/194 tokens. Those fixtures did not independently qualify EOS inside
a proposal, production cancellation, stop strings, tool turns or sampling.

The prompt-copy edit gain was about 28% in an earlier campaign and about 46%
in this one. The difference lies in verification time, despite matching
acceptance counts, timer boundaries and Metal executable text. Its cause
remains unresolved; the new ratios use new matched controls.

Fresh-prefill comparisons passed all 300 local and 270 cross-process artifact
checks across ten shapes/corpora. Prompt-copy passed all 96 artifact comparisons,
including 4,292 full-vocabulary logit row pairs. Token IDs, logits and serialized
session state were compared byte-for-byte with zero tolerance. The stock branch
also matched main through position 32,767 for Q4, native MXFP4 and GLM.

Individual rates, direct ratios and exactness coverage are retained in
[measurements.json](measurements.json). Root-level dated research reports and
large raw captures remain local, untracked artifacts.

## Integration validation

The measurements above pin the original isolated candidates. The production
expert defaults were separately checked against their diagnostic rollback in
the same build. They admit only resident, non-quality, single-device V4.1 on
M3 Ultra with the measured expert geometry and 128–512 batch rows. Other
devices, streaming, tensor parallelism, and scalar decode retain their existing
paths. `DS4_METAL_DISABLE_V41_SHORT_EXPERTS=1` disables these additions; the
established large-prefill MXFP4 paths are unchanged.

Each production fresh-prefix cell used one warm session per arm, then one
timed ABBA block: two samples per arm. Both code and prose were tested at
128/256/512 tokens for Q4 and 128/512 for native MXFP4. At each 512-token
prefix, every pass also appended 128 prompt tokens, recreated a session from
the same canonical append snapshot, checked the full roundtrip, and evaluated
128 scalar greedy tokens. Only completed sync/eval calls were timed; selection,
session lifecycle, snapshots, full-logit comparisons and I/O were outside.

| Production default vs rollback | Fresh prefill, all cells | Append 128, prose / code | Scalar 128, prose / code |
| --- | ---: | ---: | ---: |
| Q4 compact expert pair | +0.57–3.26% | +2.09% / +2.51% | +0.24% / −0.18% |
| MXFP4 compact pair + down culling | +13.14–20.26% | +20.35% / +22.03% | +0.07% / +0.16% |

All automatic component masks matched the expected kernels; rollback, snapshot
restore and scalar decode recorded zero short-expert components. Observation
used a GLM-only test bit that does not bypass V4.1 admission. Across both models,
132 snapshot/logit frontiers comprise 18 references and 114 exact comparisons.
The 3,072 scalar evaluations comprise 512 reference rows and 2,560 exact
full-vocabulary comparisons, with matching token IDs and final states. All
comparisons use zero tolerance. The small scalar timing deltas and earlier
mixed E2 continuation results do not establish a broad no-regression guarantee.

The model-free GPU oracle passed 156 cells / 624 calls, including boundary and
interior row sizes, sparse/hot/empty experts, exceptional inputs, buffer guards,
and quality/streaming/geometry exclusions. All three real SSD expert checks
(table admission, Q4, MXFP4) passed. Prompt-copy host checks passed 7,480 generated
cases plus boundary/scan-cap cases. Four live fixtures passed 544 blocks
(43 full, 2 partial), 1,201 full-logit comparisons and 357 retained-prefix checks;
the 8,192-token padded fixture passed 44 blocks (18 full, 2 partial), 144 logit
comparisons and 157 retained-prefix checks. Boundary checks cover keep counts,
output caps, cross-session reuse, snapshot restore, cancellation/reuse,
sampling/ignore-EOS fallback, rollback and context exhaustion.

Production HTTP checks matched response text, finish reasons and completion
counts with prompt-copy enabled and disabled: natural stop, caps of 1/3/7/17,
an explicit stop string, its follow-up, and a request after streaming
cancellation. The server observed both cancellations; the enabled arm recorded
83 verification cycles and 562 accepted draft tokens, versus zero when disabled.
These are correctness checks, not HTTP latency measurements.

The ordinary greedy CLI also produces byte-identical output at caps of 1/3/7/256.
At 256, its enabled run executes 20 copy verification cycles; the disabled run
executes none. The retained test exercises the normal CLI entry point:

```sh
python3 tests/test_prompt_copy_cli.py --model path/to/V4.1-Q4.gguf
make test-prompt-copy-live DS4_TEST_MODEL=path/to/V4.1-Q4.gguf
```

The normal Metal build and CPU object compilation pass. No large CPU inference
run was attempted.

`make test` is not fully green: nine assertion failures in official-vector,
SSD cache-pressure vector, local-golden and server think-context checks reproduce
identically on clean `f83ecfe`. The remaining 15 unit recipes passed when run
separately. [integration-validation.json](integration-validation.json) records
the production phase rates, exactness counts, and the exact expert-run
source/binary and capture hashes. Frontend checks carry their own build
provenance. CUDA and distributed hardware checks were not run. These checks do
not relabel the isolated percentages above as a combined production or HTTP
performance measurement.

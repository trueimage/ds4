# DeepSeek V4.1 Flash on M3 Ultra: native MXFP4 experts and DSpark

[Campaign](../README.md) · [Host pipeline](../host-pipeline/README.md)

Measured on 2026-09-20: 80-core Apple M3 Ultra, 512 GiB, source checkpoint
`deepseek-ai/DeepSeek-V4.1-Flash` at `dba1be0a40aa45a94ad051997016db3960a90277`.
That revision differs from the pinned `df42c109` only under `encoding/`; every
weight shard has the same LFS object. These are local observations for one
machine, prompt file and model.

## The recipe

The released routed experts are FP4 E2M1 codes with one UE8M0 scale per 32
weights along each row: MXFP4. The Q4 recipe dequantizes them and quantizes
again to Q4_K. `--quant mxfp4` moves the codes into GGUF blocks instead.

| | Q4 | MXFP4 |
| --- | ---: | ---: |
| Routed experts | 284.8 GiB, 4.5 bpw | 268.9 GiB, 4.25 bpw |
| Main weights / file | 294 / 483 GiB | 278 / 467 GiB |
| Expert weight error against the checkpoint | at least 5.4% relative RMS | none |

The Q4 error is a lower bound: the distance from each dequantized Q4_K block to
its nearest MXFP4 point, over 36 expert slices in four layers. The same bound
reads 5.05% on V4 Flash, where the published MXFP4 and Q4_K files give the true
figure, 5.32% (25.5 dB). Q8_0 on the FP8 attention and shared-expert tensors is
about 0.51% (45.8 dB) by the same method.

The audit compared every one of the 46,080 expert matrices byte for byte with
the checkpoint and decoded samples the way the runtime reads them. The other
924 resident tensors are byte-identical to the published Q4 file's.

## Runtime admission

Three V4.1 gates named Q4_K only. Each now admits a file whose routed experts
are all MXFP4; a mixed file is still neither configuration.

| Gate | Effect of leaving MXFP4 out |
| --- | --- |
| `ds41_measured_config` | No fusions, decode queue or pipeline: 20.7 instead of 30.5 tokens/s in the capture runs |
| `v41_decode_batch` | Batches of six to eight rows took the grouped matmul, so a continued prefill or session batch of that size was not one-token decode (row differences up to 0.8 logits) |

Exactness, with the same sweep as the campaign (`--ctx-start 512 --ctx-max
32703 --step-mul 4 --ctx-alloc 32768 --gen-tokens 64`): the unmodified source,
which runs MXFP4 on the original paths, and the final build captured 260 full
vocabulary rows, **33,612,800 bit-identical logits**, and four byte-identical
session snapshots through 32,703 tokens. Logits hash
`4d25b0a1bc8770565482ab08bb537b595643b1f55ca7d2628228445eb8a99a10`. The same
comparison on the published Q4 file is also exact, hash `8e06b20e…dc0e`.
That is not the `e022c41d…493c` in [logit-comparison.json](../logit-comparison.json);
the unmodified source at `849e039` already produces `8e06b20e`.

## Throughput

`ds4-bench`, Engram tables read through the cache before each run, order
Q4, MXFP4, MXFP4, Q4, 128 greedy tokens per frontier, means of two runs.

| Context (new tokens) | Prefill Q4 → MXFP4 t/s | Change | Decode Q4 → MXFP4 t/s | Change |
| ---: | ---: | ---: | ---: | ---: |
| 512 (512) | 286.6 → 262.3 | -8.5% | 28.99 → 29.85 | +3.0% |
| 2,048 (1,536) | 354.1 → 339.4 | -4.2% | 29.02 → 29.83 | +2.8% |
| 8,192 (6,144) | 368.5 → 379.7 | +3.0% | 28.70 → 29.56 | +3.0% |

Short chunks lose because the MXFP4 compact expert tile is admitted from 2,048
tokens, while Q4_K has its tail cull at every size. The fastest pre-M5 MXFP4
decode kernels are likewise specialized to V4 Flash's 4096/2048/256 shape; on
that model they were worth another 1.4%.

## Where a decoded token goes

Forty encodes of each stage's own kernels on one layer's real weights, best of
four, MXFP4, 1,032-token context. One-token decode: 33.1 ms.

| Stage | ms per token |
| --- | ---: |
| Router and routed experts | 9.09 |
| Attention output (`o_a`, `o_b`) | 5.19 |
| Attention projections (`q_a`, `q_b`, `kv`) | 4.55 |
| Attention core, index and window | 4.18 |
| Hyper-connection mixes and norms | 2.83 |
| Shared expert | 2.75 |
| Output head | 1.00 |

## DSpark

The checkpoint's draft module is three V4.1 blocks with 128 experts, three used.
`--dspark-support` writes it as a 7.4 GiB support GGUF, experts exact.

Verification here equals one-token decode. `ds41_graph_verify` runs a suffix's
rows through the row-exact batch path with a one-token head per row, records
each layer's replaced window slots and each even position's pooling input, and
`ds41_spec_commit` keeps any prefix. `tests/test_deepseek41_dspark
--verify-parity` checks, for two to eight rows below, across and beyond the
128-row window at odd and even frontiers: every logit of every row, the
serialized session after every possible kept prefix, and the three tokens that
follow, all bit-identical to decode.

| Prompt, 400 greedy tokens | Decode t/s | DSpark t/s | Drafts accepted | Tokens per cycle | Output |
| --- | ---: | ---: | ---: | ---: | --- |
| C hash table | 30.25 | 29.32 | 86.1% | 5.06 | identical |
| Prose explanation | 30.18 | 24.59 | 80.0% | 1.96 | identical |

It is slower because an exactly verified row costs most of a decoded token:

| Rows | Verify | In decode steps |
| ---: | ---: | ---: |
| 2 | 68.95 ms | 2.08 |
| 4 | 118.27 ms | 3.57 |
| 6 | 169.69 ms | 5.13 |
| 8 | 219.25 ms | 6.63 |

`kernel_mul_mv_q8_0_f32_rows` reads a Q8_0 matrix once for all rows while each
row keeps the one-token kernel's walk and reduction tree
(`make test-deepseek41-fusions`, 84 cases, bits equal both a launch per row and
each row's own one-token matvec). It takes the 40-layer `q_b` projection for six
rows from 13.23 to 9.63 ms, against 2.86 ms for one: about half of that kernel
is arithmetic per row, which exact verification cannot share. Rollback:
`DS4_METAL_DISABLE_V41_SHARED_ROWS=1`.

An unrelated observation from the same fixture: a continued prefill of exactly
eight rows differs from decode by about one logit on the unmodified source and
the published Q4 file too, while seven rows and an eight-row verification are
exact.

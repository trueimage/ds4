# Testing and Development

[README](../README.md)

Read [CONTRIBUTING.md](../CONTRIBUTING.md) before proposing an inference change.
The [release QA guide](../QA_BEFORE_RELEASES.md) defines the hardware matrix,
checkpoint-matched quality tests, long-context agent tasks, and speed checks.
It also records which checks were not completed. A smoke test is not full QA.

## Focused checks

Model-free checks include:

```sh
make ds4_test ds4_agent_test test-session-state
./ds4_test --server
./ds4_agent_test
```

On Metal, small GPU tensor tests are available without loading a full GGUF:

```sh
make tests/test_session_state_gpu tests/test_glm53_kda tests/test_mxfp4_metal
./tests/test_session_state_gpu
./tests/test_glm53_kda
./tests/test_mxfp4_metal
```

The GLM KDA/attention executable is also included in `make test` on Metal.
`make test-glm53-fork` runs the focused router/shared, serial top-k, and mixed
Q8/BF16 input checks. `make test-glm53-hc-pre-repeat` checks repeated HC
producer dispatches and equal output with both fallback producer variants.
These two focused targets are Metal-only and model-free.
For the GLM top-k oracle with admitted score widths capped at 32K, build
`make tests/test_glm53_topk_fast` and run
`MTL_DEBUG_LAYER=1 ./tests/test_glm53_topk_fast --32k`. Its larger synthetic
width checks host refusal only; it does not allocate a model context.

`make test-deepseek41-q4-tail` runs the model-free Metal Q4 tail-culling
fixture with Metal API validation enabled. See the campaign's
[fixture coverage](../speed-bench/ds41f-m3ultra-perf/README.md#exactness-and-regression-checks).
The follow-up's model-free `make test-deepseek41-fusions` and
`make test-deepseek41-topk` targets also enable Metal API validation; see their
[coverage](../speed-bench/ds41f-m3ultra-perf/glm-transfer/README.md#combined-exactness).
The separate [resident live-edge fixture](../speed-bench/ds41f-m3ultra-perf/glm-transfer/README.md#resident-live-edge-check)
requires the real Q4 GGUF on M3 Ultra and is invoked explicitly, outside `make test`.
On Metal, `make test` also runs the three model-free V4.1 oracles (fusions,
top-k, Q4 tail), so a shader-compiler change that alters their arithmetic is
noticed; see the [host pipeline notes](../speed-bench/ds41f-m3ultra-perf/host-pipeline/README.md).

`make test` also includes model-backed tests. Select the right GGUF and ensure
that it fits before running it; do not accidentally load a large model on a
single device during multi-GPU QA. ROCm has `make test-rocm`.

Official-vector tests must use continuations from the same checkpoint as the
GGUF. Flash 0731 and Vision Experimental are not interchangeable fixtures.
See [test vectors](../tests/test-vectors/README.md) and
[quality scoring](../gguf-tools/quality-testing/README.md).

## Investigating output

```sh
./ds4 --dump-tokens -p "..."
./ds4 --dump-logprobs /tmp/out.json --logprobs-top-k 20 --temp 0 -p "..."
./ds4 --dump-logits /tmp/logits.json --nothink --prompt-file prompt.txt
./ds4-server --trace /tmp/ds4-trace.txt
```

Token dumps catch template differences without inference. Logits and
continuations help distinguish sampling changes from graph errors. Server
traces include cache and tool-parser decisions. Keep traces private when they
contain real conversations.

## Tools and source references

- [GGUF conversion and quantization](../gguf-tools/README.md)
- [Imatrix collection](../gguf-tools/imatrix/README.md) and
  [calibration corpus](../gguf-tools/imatrix/dataset/README.md)
- [Steering vectors](../dir-steering/README.md)
- [Benchmark scripts and charts](../speed-bench/README.md)
- [Evaluation data and licenses](../EVAL_DATA.md)
- [Session payload implementation](../ds4.c) and
  [cache header definitions](../ds4_kvstore.h)
- [Pipeline protocol](../ds4_distributed.c) and [TP protocol](../ds4_tp.c)

For changes to state handling, cover rewind/replay, save/load, images, and
multiple sessions as well as a fresh prompt. For distributed changes, exercise
both ranks and failures; a local command-parser test is not physical TP QA.

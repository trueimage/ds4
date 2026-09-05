# Published GLM-5.3-Flash quantizations in DS4

`glm53_import_gguf.py` imports Unsloth UD-Q4_K_XL and UD-Q5_K_XL split
GGUFs into DS4's native GLM-5.3-Flash conventions. This is an experimental
compatibility path. It does not requantize matrices or reproduce another
engine's arithmetic. Evaluate each imported artifact independently before
choosing a deployment model.

## Numerical contract

- Quantized matrix payloads are copied byte-for-byte, including the publisher's
  layer-specific expert precision choices.
- Tensor names and model metadata are translated to DS4's schema. Structural
  and mathematical parameters must match the supported model configuration.
  Published HC epsilon is approximately `9.999999974752427e-7`: the F32
  representation of **1e-6**, matching DS4.
- Published `ssm_a` contains rounded `-exp(A_log)`. DS4 expects native `A_log`.
  The importer checks the published value against the source checkpoint, then
  restores the original F32 bytes. It does not apply an approximate inverse log.
- Unquantized F32 controls must equal the source checkpoint after exact BF16
  expansion where applicable.
- F32 indexer projection and pooling position parameters narrow to BF16 only
  when every value is exactly representable and matches the checkpoint bytes.
  This preserves weight values; it does not establish identical accumulation
  between DS4 and a publisher's F32 implementation.
- Token IDs must match both the checkpoint and a known working DS4 tokenizer
  template. The source checkpoint provides the chat template.

Q8 HC mixing and indexer pool-gate weights use existing generic Q8 matrix
operations. Existing BF16 HC producer fusion remains active for BF16 tensors.
Other numerical controls remain F32. The original PR branch is the frozen
baseline; compatibility work belongs in a separate branch/worktree.

Published Q5 can promote routed gate/up matrices to Q6_K and down matrices to
Q8_0 in selected layers. Metal dispatches these layers through the generic
routed expert path, with F32 Q5_K/Q6_K matvec variants for decode and short
batches and existing grouped matmul variants for large prefill. Q8_0 experts
also use that path. Previously supported GLM expert layouts retain their
established dispatch. This is compatibility support, not a claim that the new
formats have reached their best possible throughput.
The new high-precision fallback is available only for resident single-device
Metal execution. CPU execution, non-Metal GPU backends, SSD streaming, and GLM
tensor-parallel sessions reject these layouts at model open. The F32 expert
helpers remain internal arithmetic references for component diagnostics.

## Tools and inputs

Use Python 3.10+ with NumPy, a current `hf` CLI, the complete source checkpoint,
and a working DS4 GLM Flash GGUF as tokenizer template. Importing already
quantized matrices does not need PyTorch, an importance matrix, or the C
quantizer library. DS4 inference requires its usual Metal build tools.

Pin both Hub repositories to full commit IDs. Verify the source checkpoint
with `hf cache verify`, and obtain expected input GGUF SHA256 values from the
pinned Hub repository's LFS metadata. Supply a JSON object mapping each input
GGUF filename to its expected hash. The importer verifies every input file
before writing; a dry run checks headers and controls but skips full-file
hashing.

```sh
python gguf-tools/glm53_import_gguf.py \
  --input-dir /path/to/unsloth/UD-Q4_K_XL \
  --hf /path/to/zai-org--GLM-5.3-Flash \
  --tokenizer-template /path/to/working-GLM-5.3-Flash.gguf \
  --source-revision SOURCE_COMMIT_40_HEX \
  --quant-revision QUANT_COMMIT_40_HEX \
  --checksums /path/to/pinned-input-sha256.json \
  --out /path/to/GLM-5.3-Flash-UD-Q4_K_XL-DS4SourceAligned.gguf

python gguf-tools/glm53_import_gguf.py \
  --verify /path/to/GLM-5.3-Flash-UD-Q4_K_XL-DS4SourceAligned.gguf
```

The output directory must exist and have room for the complete output plus a
32 GiB reserve. Inputs are read-only; existing outputs are refused. A temporary
file is published only after a complete copy. The `.import.json` sidecar records
revisions, input hashes, transformations, output layout, and per-tensor payload
hashes. Readback verification checks the header and every tensor against it.
Interrupted copies may leave a clearly named `.partial` file; inspect it before
removing it or retrying.

## Validation

```sh
python gguf-tools/tests/test_glm53_import_gguf.py
python gguf-tools/tests/test_glm53_quantize.py
make -j8 all ds4_test tests/test_glm53_kda
./tests/test_glm53_kda
make test-glm-mixed-experts-metal
./ds4_test --metal-kernels
```

Require identical full-logit traces on unchanged GGUFs when comparing engine
revisions. Across different quantizations, compare task quality, target NLL,
long-context behavior, memory, prefill, and decode independently. Run performance
comparisons without competing downloads, builds, or GPU work. Token throughput
alone cannot determine the best recipe.

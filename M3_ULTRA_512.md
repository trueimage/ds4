# DS4 on Mac Studio M3 Ultra (512 GB)

This is the model choice for a 512 GB M3 Ultra plus a ranked list of
**unimplemented** optimization ideas for the Metal MXFP4 routed-expert kernels.

Nothing in section 4 is in the tree. These are proposals to be measured, not a
description of shipped behavior. Line references are against `b030961`.

## 1. Use the MXFP4 GGUF

```sh
./download_model.sh ds4f-mxfp4
```

DeepSeek ships Flash routed experts as packed FP4 with `F8_E8M0` block scales,
which is MXFP4. `repack_fp4_weight_mxfp4()` in
`gguf-tools/deepseek4-quantize.c:824` preserves those bytes exactly and verifies
every code and scale on the way out. The Q4_K file dequantizes that FP4 and
requantizes it, so the imatrix only limits damage that MXFP4 never takes.

There is no higher-fidelity Flash GGUF to build. Every other tensor family is
already at the loader's ceiling:

| Family | Accepted types | Recipe uses |
|---|---|---|
| attention proj, shared experts, output | `q8_0`, `q4_K`, `q4_0` (`ds4.c:4358`) | `q8_0` |
| HC, compressor, router | F16 or F32 (`ds4.c:4386`) | F16 |
| indexer q_b | F16 or `q8_0` | F16 |
| routed experts | see `tensor_is_routed_expert_type` (`ds4.c:4426`) | MXFP4 |

Do not build the "True Q8_K routed experts" recipe from
`gguf-tools/README.md`. It is an upcast of FP4 data at roughly twice the size,
and `tensor_is_routed_expert_type` omits `Q8_K`, so `ds4.c:5101` rejects the
file at load even though the compute paths (`ds4.c:8178`, `ds4.c:10845`) and
the Metal pipelines (`ds4_metal.m:27492`) exist.

## 2. Headroom

Per decoded token the routed experts alone are about 6 experts x 25.2M params x
41 MoE layers at 4.25 bpw, or roughly 3.3 GB. Against 819 GB/s that is a ~250
t/s ceiling; the measured Q4 figure in `README.md` is 35.50 t/s.

Even allowing generously for attention, KV, the indexer, and per-layer dispatch
overhead, the routed matvecs are far from bandwidth-saturated. They are
issue/latency-bound, which is the regime where sections 4.1 to 4.4 pay.

## 3. What the decode path already does

For Flash at `n_expert == 6` and `n_tokens == 1`, MXFP4 resolves both fusions:
`fuse_pair_swiglu` picks `kernel_mul_mv_id_mxfp4_pair_swiglu_f32`
(`ds4_metal.m:35752`) and `direct_down_sum` picks
`kernel_mul_mv_id_mxfp4_sum6_f32` (`ds4_metal.m:35765`). Prefill uses
`kernel_mul_mm_id_mxfp4_{f32,f16}` and the `addr` variants.

The TensorOps/MPP prefill path added in `532ec8b` wires only `IQ2_XXS` and
`Q2_K`, and is gated on `ds4_gpu_mpp_available()`, so it does not apply to M3
regardless.

## 4. Suggested changes, ranked

### 4.1 Vectorize the nibble loads

All three decode kernels read quants one byte at a time:

```c
device const uchar *q = xb.qs + 8 * it;
float4 acc = yl0 * float4(lut[q[0] & 15], lut[q[1] & 15],
                          lut[q[2] & 15], lut[q[3] & 15]);
acc += yl1 * float4(lut[q[0] >> 4], ...);
```

That is 8 scalar loads per row per block, 16 per iteration at
`N_R0_MXFP4 = 2`, and 32 in the pair kernel because it walks gate and up. Q4_K
reads through `device const uint16_t *` instead (`metal/moe.metal:2124`).

MXFP4 cannot use `uint16_t *` or `uchar4`: `block_mxfp4` is 17 bytes, so `qs`
lands at offset 1, 18, 35, and is never 4-byte aligned. `packed_uchar4` has
alignment 1 and is the right type:

```c
device const packed_uchar4 *q4 = (device const packed_uchar4 *)(xb.qs + 8 * it);
const uchar4 a = uchar4(q4[0]);
const uchar4 b = uchar4(q4[1]);
float4 acc = yl0 * float4(lut[a.x & 15], lut[a.y & 15],
                          lut[a.z & 15], lut[a.w & 15]);
acc += yl1 * float4(lut[a.x >> 4], lut[a.y >> 4],
                    lut[a.z >> 4], lut[a.w >> 4]);
acc += yl2 * float4(lut[b.x & 15], lut[b.y & 15],
                    lut[b.z & 15], lut[b.w & 15]);
acc += yl3 * float4(lut[b.x >> 4], lut[b.y >> 4],
                    lut[b.z >> 4], lut[b.w >> 4]);
```

Sites:

- `metal/moe.metal:3279` — `kernel_mul_mv_mxfp4_f32_impl`
- `metal/moe.metal:4720` and `4721` — `kernel_mul_mv_mxfp4_pair_swiglu_impl`
- `metal/moe.metal:6293` — `ds4_mxfp4_accumulate_rows` (sum6)
- `metal/moe.metal:2854` — `dequantize_mxfp4`, the prefill mm path; 16 byte
  loads become 4, and each block is currently read twice, once per `il`

No `packed_*` type is used anywhere in `metal/moe.metal` today, so confirm the
codegen collapses to word loads rather than assuming it.

### 4.2 Sweep `nsg` and `N_R0_MXFP4`

These were never tuned for MXFP4. `ds4_gpu_routed_mv_nsg()` is
`type == DS4_METAL_TENSOR_Q8_0 ? 4u : 2u` (`ds4_metal.m:27480`), and every fused
MXFP4 pipeline is created with a hardcoded `2` (`ds4_metal.m:7028` to `7037`).
`N_R0_MXFP4` is 2 (`metal/moe.metal:17`) while `Q2_K`, `IQ2_XXS`, and `Q5_K` use
4. MXFP4 landed last and inherited defaults rather than measurements.

`nr0 = 4` doubles activation-register reuse per loaded `y`, which compounds in
the pair kernel where that reuse is the whole point. `nsg = 4` gives more
latency hiding per threadgroup on an 80-core part.

`nsg` is a function constant at index 600 (`ds4_metal.m:2517`), so the creation
site and `ds4_gpu_routed_mv_nsg()` must change together or the dispatch geometry
desyncs from the kernel. The `if (sgitg == 0) lut[tiisg] = ...` fill stays
correct at any `nsg`.

This is the cheapest experiment of the set: a constant sweep, no logic change.

### 4.3 Register LUT via `simd_shuffle`

Each lane does 8 threadgroup loads per `float4` group, and the indices are
nibbles clustered at low values, so bank conflicts are likely:

```c
const float lane_val = ds4_metal_mxfp4_values[tiisg & 15];   // before the loop
...
float4 acc = yl0 * float4(simd_shuffle(lane_val, a.x & 15), ...);
```

This also drops the `threadgroup_barrier` and lets `ds4_gpu_routed_mv_smem()`
return 0 for MXFP4 (`ds4_metal.m:27467`), freeing threadgroup memory for
occupancy.

Genuinely uncertain: Apple threadgroup memory is fast and `simd_shuffle` is not
free. Benchmark before adopting.

### 4.4 Hoist the per-block bounds check

`metal/moe.metal:6291` evaluates `if (first_row + row < n_rows)` inside the `ib`
loop on every block, though both operands are loop-invariant. The pair kernel
does the equivalent check once after the loop, so this looks like an oversight
rather than a deliberate difference. Compute `rows_here` before the loop.

### 4.5 Structural: planar repack

If 4.1 shows loads are the bottleneck, the real fix is removing the 17-byte
block: split scales and nibbles into separate planes at load so nibbles are
16-byte aligned and `uint4` loads become legal.

The cost is real. Weights are wrapped zero-copy from the mmap with
`newBufferWithBytesNoCopy` (`ds4_metal.m:1809`, `ds4_metal.m:10399`), so this
means an actual copy of roughly 147 GB. That is affordable on a 512 GB machine
and not on a 128 GB one. It also breaks the SSD-streaming path, which assumes
expert bytes sit at their model-map offsets (`ds4_metal.m:35770`). Only worth it
after the cheap changes are measured.

## 5. Latent footgun

`ds4_metal.m:38874` selects `iq2_xxs ? iq2_pair : q4_k_pair` in the
`use_tiny_pair_mv` fallback, which would hand MXFP4 data to a Q4_K kernel.

It is unreachable today: for MXFP4, `use_tiny_pair_mv` requires the swiglu
pipeline to exist, which makes `use_tiny_pair_swiglu` true, and that branch is
tested first at `ds4_metal.m:38839`. It becomes a live bug if that pipeline ever
becomes optional. An explicit `nil` for unhandled types would close it.

## 6. Measuring

Profile first with Metal System Trace or an Xcode GPU capture on
`kernel_mul_mv_id_mxfp4_pair_swiglu_f32`. If it reports ALU-bound rather than
issue- or memory-bound, 4.1 and 4.3 are wasted effort and 4.2 is all that is
worth doing.

Correctness after every change, comparing the GPU against the independent scalar
oracle in `tests/test_mxfp4_metal.c`:

```sh
make test-mxfp4-metal && make mxfp4-dot-test
```

Speed with the standard *Promessi sposi* input, matching `speed-bench/README.md`
so results are comparable with the tracked CSVs:

```sh
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

No `speed-bench/` CSV contains MXFP4 numbers yet, and the only M3 Ultra rows in
`README.md` are older q2/q4 measurements taken with a different CLI procedure.
Capture a baseline before changing anything, and consider contributing it as
`speed-bench/m3_ultra.csv`.

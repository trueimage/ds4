// DeepSeek V4.1 keeps the RoPE tail in the quantized KV vector. Unlike V4,
// indexer Q/K have no Hadamard transform, and compressed KV has E4M3 scales.
struct ds4_metal_args_dsv41_quantize {
    uint width;
    uint rows;
    uint mode;
};

static inline float dsv41_bf16(float x) {
    uint bits = as_type<uint>(x);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return as_type<float>(bits & 0xffff0000u);
}

static inline float dsv41_pow2_ceil(float x) {
    const uint bits = as_type<uint>(x);
    return as_type<float>((bits & 0x7f800000u) +
                         ((bits & 0x7fffffu) ? 0x800000u : 0u));
}

kernel void kernel_dsv41_bf16_linear(
        constant ulong &count,
        device uint *x,
        uint gid [[thread_position_in_grid]]) {
    const ulong first = (ulong)gid * 4u;
    if (first + 4u <= count) {
        uint4 bits = *((device uint4 *)(x + first));
        const bool4 finite = (bits & 0x7f800000u) != 0x7f800000u;
        bits += select(uint4(0), uint4(0x7fffu) + ((bits >> 16u) & 1u), finite);
        *((device uint4 *)(x + first)) = bits & 0xffff0000u;
    } else {
        for (ulong i = first; i < count; i++) {
            uint bits = x[i];
            if ((bits & 0x7f800000u) != 0x7f800000u)
                bits += 0x7fffu + ((bits >> 16u) & 1u);
            x[i] = bits & 0xffff0000u;
        }
    }
}

struct ds4_metal_args_dsv41_rope {
    uint width, heads, rows, start, inverse, stride;
    float frequencies[32];
};

kernel void kernel_dsv41_rope(
        constant ds4_metal_args_dsv41_rope &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const float theta = float(args.start + group.y * args.stride) * args.frequencies[lane];
    const float c = precise::cos(theta);
    const float s = args.inverse ? -precise::sin(theta) : precise::sin(theta);
    const ulong i = ((ulong)group.y * args.heads + group.x) * args.width +
                    args.width - 64u + 2u * lane;
    const float re = x[i], im = x[i + 1u];
    x[i] = dsv41_bf16(re * c - im * s);
    x[i + 1u] = dsv41_bf16(re * s + im * c);
}

// The reduce writes F32 first, just as the unfused chain does. Its noinline
// body is shared with the reference; materialize BF16 before the RoPE pair.
static inline void dsv41_attention_epilogue(
        constant ds4_metal_args_dsv41_rope &args, device float *x,
        uint head, uint tid, uint threads) {
    device float *row = x + (ulong)head * 512u;
    for (uint i = tid; i < 448u; i += threads) row[i] = dsv41_bf16(row[i]);
    if (tid < 32u) {
        const float theta = float(args.start) * args.frequencies[tid];
        const float c = precise::cos(theta);
        const float s = args.inverse ? -precise::sin(theta) : precise::sin(theta);
        const uint i = 448u + 2u * tid;
        const float re = dsv41_bf16(row[i]), im = dsv41_bf16(row[i + 1u]);
        row[i] = dsv41_bf16(re * c - im * s);
        row[i + 1u] = dsv41_bf16(re * s + im * c);
    }
}

kernel void kernel_flash_attn_ext_vec_reduce_dsv41(
        constant ds4_metal_args_flash_attn_ext_vec_reduce &args,
        device const char *htmp, device char *dst,
        constant ds4_metal_args_dsv41_rope &rope,
        uint head [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    ds4_flash_attn_vec_reduce_row(args, htmp, dst, head, lane, sg,
        (short)FC_flash_attn_ext_vec_reduce_NWG, (short)FC_flash_attn_ext_vec_reduce_DV);
    threadgroup_barrier(mem_flags::mem_device);
    dsv41_attention_epilogue(rope, (device float *)dst, head, tid,
                             32u * FC_flash_attn_ext_vec_reduce_NWG);
}

kernel void kernel_dsv41_quantize(
        constant ds4_metal_args_dsv41_quantize &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const uint block = args.mode == 3u ? 16u : 32u;
    const uint column = group.x * block + lane;
    const bool valid = lane < block && column < args.width;
    const ulong index = (ulong)group.y * args.width + column;
    const float value = valid ? dsv41_bf16(x[index]) : 0.0f;
    const float amax = simd_max(abs(value));
    float result = value;
    if (args.mode == 1u) {
        const float scale = dsv41_pow2_ceil(max(amax, 1.0e-4f) * (1.0f / 448.0f));
        result = copysign(dsv4_e4m3fn_dequant(abs(value) / scale), value) * scale;
    } else if (args.mode == 2u || args.mode == 3u) {
        const float scale = args.mode == 3u
            ? dsv4_e4m3fn_dequant(max(amax, 0.01171875f) / 6.0f)
            : dsv41_pow2_ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
        result = copysign(dsv4_e2m1fn_dequant(abs(value) / scale), value) * scale;
    }
    if (valid) x[index] = dsv41_bf16(result);
}

struct ds4_metal_args_dsv41_engram {
    uint width;
    uint rows;
    float eps;
    uint masked;
};

kernel void kernel_dsv41_engram_add(
        constant ds4_metal_args_dsv41_engram &args,
        device float *residual,
        device const float *kv,
        device const float *q_weight,
        device const float *k_weight,
        device const uchar *mask,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    if (args.masked && !mask[group.x]) return;
    const ulong offset = ((ulong)group.x * 4u + group.y) * args.width;
    const ulong key_offset = ((ulong)group.x * 5u + group.y) * args.width;
    const ulong value_offset = ((ulong)group.x * 5u + 4u) * args.width;
    float h2 = 0.0f, k2 = 0.0f, dot = 0.0f;
    for (uint i = lane; i < args.width; i += 32u) {
        const float h = residual[offset + i];
        const float k = dsv41_bf16(kv[key_offset + i]);
        const uint wi = group.y * args.width + i;
        h2 += h * h;
        k2 += k * k;
        dot += h * (q_weight[wi] * k_weight[wi]) * k;
    }
    h2 = simd_sum(h2);
    k2 = simd_sum(k2);
    dot = simd_sum(dot) * rsqrt(h2 / args.width + args.eps) *
          rsqrt(k2 / args.width + args.eps) * rsqrt(float(args.width));
    const float gate = 1.0f / (1.0f + exp(-copysign(sqrt(max(abs(dot), 1.0e-6f)), dot)));
    for (uint i = lane; i < args.width; i += 32u)
        residual[offset + i] = dsv41_bf16(residual[offset + i] +
            gate * dsv41_bf16(kv[value_offset + i]));
}

struct ds4_metal_args_dsv41_pool {
    uint width;
    uint pairs;
    uint tail;
};

kernel void kernel_dsv41_pool2(
        constant ds4_metal_args_dsv41_pool &args,
        device float *out,
        device const float *kv,
        device const float *scores,
        device const float *previous_kv,
        device const float *previous_scores,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.pairs) return;
    const long a = (long)index.y * 2 - args.tail;
    const ulong b = (ulong)(a + 1) * args.width + index.x;
    const float ka = a < 0 ? previous_kv[index.x] : kv[(ulong)a * args.width + index.x];
    const float sa = a < 0 ? previous_scores[index.x] : scores[(ulong)a * args.width + index.x];
    const float sb = scores[b], peak = max(sa, sb);
    const float ea = exp(sa - peak), eb = exp(sb - peak);
    out[(ulong)index.y * args.width + index.x] = dsv41_bf16((ka * ea + kv[b] * eb) / (ea + eb));
}

struct ds4_metal_args_dsv41_candidates {
    uint width;
    uint rows;
    uint start;
    uint ratio;
};

kernel void kernel_dsv41_candidate_blocks(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *blocks,
        device const float *unused,
        uint2 index [[thread_position_in_grid]]) {
    (void)unused;
    const uint count = (args.width + 7u) / 8u;
    if (index.x >= count || index.y >= args.rows) return;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    float best = -INFINITY;
    for (uint i = index.x * 8u; i < min(visible, (index.x + 1u) * 8u); i++)
        best = max(best, scores[(ulong)index.y * args.width + i]);
    if (visible && index.x == (visible - 1u) / 8u) best = INFINITY;
    blocks[(ulong)index.y * count + index.x] = best;
}

kernel void kernel_dsv41_candidate_filter(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *out,
        device const float *block_mask,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.rows) return;
    const ulong offset = (ulong)index.y * args.width + index.x;
    const uint blocks = (args.width + 7u) / 8u;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    out[offset] = index.x < visible &&
        block_mask[(ulong)index.y * blocks + index.x / 8u] == 0.0f
        ? scores[offset] : -INFINITY;
}

struct ds4_metal_args_dsv41_carry {
    uint width, rows, words, format, pack;
};

kernel void kernel_dsv41_carry_copy(
        constant ds4_metal_args_dsv41_carry &args,
        device uint *packed, device float *plain,
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint col = group.x * 128u + tid;
    const ulong row = group.y;
    if (args.format == 0u) {
        if (col >= args.width) return;
        device ushort *p = (device ushort *)(packed + row * args.words);
        if (args.pack) p[col] = ushort(as_type<uint>(plain[row * args.width + col]) >> 16);
        else plain[row * args.width + col] = as_type<float>(uint(p[col]) << 16);
    } else {
        const uint word = col / 32u;
        if (args.pack) {
            const bool allowed = col < args.width && plain[row * args.width + col] == 0.0f;
            const uint bits = simd_sum(allowed ? 1u << lane : 0u);
            if (!lane && word < args.words) packed[row * args.words + word] = bits;
        } else if (col < args.width) {
            const uint bits = packed[row * args.words + word];
            plain[row * args.width + col] = bits & (1u << lane) ? 0.0f : -INFINITY;
        }
    }
}

#ifdef DS4_METAL_HAS_TENSOR
kernel void kernel_dsv41_indexer_pack(
        constant uint4 &args,
        device const float *q, device const float *keys,
        device uint *flags, device bfloat *packed_q, device bfloat *packed_keys,
        threadgroup uint *valid [[threadgroup(0)]],
        uint group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const bool query = group < args.y;
    const uint count = query ? 32u * 128u : 64u * 128u;
    const ulong offset = query ? (ulong)group * count : (ulong)(group - args.y) * count;
    bool exact = true;
    for (uint i = tid; i < count; i += 128u) {
        const float value = query ? q[offset + i] :
            offset + i < (ulong)args.x * 128u ? keys[offset + i] : 0.0f;
        const bfloat converted = bfloat(value);
        if (query) packed_q[offset + i] = converted;
        else packed_keys[offset + i] = converted;
        exact = exact && float(converted) == value;
    }
    const bool same = simd_all(exact);
    if (!(tid % 32u)) valid[tid / 32u] = same;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!tid) flags[group] = valid[0] && valid[1] && valid[2] && valid[3];
}

kernel void kernel_dsv41_indexer_scores_packed(
        constant uint4 &args, constant uint2 &range,
        device const float *q, device const float *weights,
        device const float *keys, device float *scores,
        device const uint *flags, device const bfloat *packed_q,
        device const bfloat *packed_keys,
        threadgroup float *scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int HEADS = 32, KEYS = 64, DIM = 128;
    const uint width = args.x, token = group.y, row0 = group.x * KEYS;
    const uint visible = (args.z + token + 1u) / args.w;
    if (row0 >= visible) {
        if (tid < KEYS && row0 + tid < width)
            scores[(ulong)token * width + row0 + tid] = -INFINITY;
        return;
    }
    matmul2d<matmul2d_descriptor(HEADS, KEYS, DIM, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
    auto result = tensor(scratch, dextents<int32_t, 2>(KEYS, HEADS));
    if (flags[range.y + token] && flags[range.x + group.x]) {
        auto query = tensor((device bfloat *)packed_q + (ulong)(range.y + token) * HEADS * DIM,
                             dextents<int32_t, 2>(DIM, HEADS));
        auto key = tensor((device bfloat *)packed_keys + (ulong)row0 * DIM,
                           dextents<int32_t, 2>(DIM, KEYS));
        auto dots = mm.template get_destination_cooperative_tensor<decltype(query), decltype(key), float>();
        for (uint16_t i = 0; i < dots.get_capacity(); i++)
            if (dots.is_valid_element(i)) dots[i] = 0;
        mm.run(query, key, dots);
        dots.store(result);
    } else {
        auto query = tensor((device float *)q + (ulong)token * HEADS * DIM,
                             dextents<int32_t, 2>(DIM, HEADS));
        auto all_keys = tensor((device float *)keys, dextents<int32_t, 2>(DIM, int(width)));
        auto key = all_keys.slice(0, int(row0));
        auto dots = mm.template get_destination_cooperative_tensor<decltype(query), decltype(key), float>();
        for (uint16_t i = 0; i < dots.get_capacity(); i++)
            if (dots.is_valid_element(i)) dots[i] = 0;
        mm.run(query, key, dots);
        dots.store(result);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < KEYS && row0 + tid < width) {
        float sum = 0;
        for (uint h = 0; h < HEADS; h++)
            sum += max(scratch[h * KEYS + tid] * (1.0f / 64.0f), 0.0f) * weights[token * HEADS + h];
        scores[(ulong)token * width + row0 + tid] = row0 + tid < visible ? sum : -INFINITY;
    }
}
#endif

// Fuse the reference 384-expert router without changing its sorting network,
// active-lane sum or the stored F32 boundaries between normalization stages.
// The clamp bounds arrive as runtime arguments, as they do in the unary clamp
// this replaces: a compile-time INFINITY invites fast-math to fold the clamp.
kernel void kernel_dsv41_router(
        constant float &scale, device const float *logits,
        device const float *bias, device float *probs,
        device int *selected, device float *weights,
        constant float2 &sum_bounds,
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup int ids[512];
    threadgroup volatile float scores[512];
    threadgroup volatile float scratch[40];
    ids[tid] = int(tid);
    if (tid < 384u) {
        const float x = logits[row * 384u + tid];
        // Match the reference softplus, including its small-exp polynomial.
        const float ex = exp(x);
        const float em = min(ex, 0.03125f);
        const float poly = em*(1.0f - em*(0.5f - em*(1.0f/3.0f - 0.25f*em)));
        scores[tid] = select(select(log(1.0f + ex), poly, ex < 0.03125f), x, x > 20.0f);
        const float p = sqrt(scores[tid]);
        probs[row * 384u + tid] = p;
        scores[tid] = p + bias[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    // Match kernel_argsort_f32_i32_desc, including padding and unordered ties.
    for (uint k = 2; k <= 512u; k <<= 1u) {
        for (uint j = k >> 1u; j; j >>= 1u) {
            const uint other = tid ^ j;
            if (other > tid) {
                const int a = ids[tid], b = ids[other];
                const bool swap = (tid & k) == 0u ?
                    (a >= 384 || (b < 384 && scores[a] < scores[b])) :
                    (b >= 384 || (a < 384 && scores[a] > scores[b]));
                if (swap) { ids[tid] = b; ids[other] = a; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    float p = 0.0f, sum = 0.0f;
    if (tid < 6u) {
        selected[row * 6u + tid] = ids[tid];
        p = probs[row * 384u + uint(ids[tid])];
        scratch[tid] = 0.0f;
        sum += p;
        sum = simd_sum(sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) scratch[0] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 6u) {
        sum = scratch[tid];
        sum = simd_sum(sum);
        if (tid == 0u) scratch[32] = clamp(sum, sum_bounds.x, sum_bounds.y);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 6u) scratch[33u + tid] = p / scratch[32];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 6u) weights[row * 6u + tid] = scratch[33u + tid] * scale;
}

kernel void kernel_dsv41_router_top6(
        constant float &scale, device const float *logits,
        device const float *bias, device float *probs,
        device int *selected, device float *weights,
        constant float2 &sum_bounds,
        device atomic_uint *coverage, constant uint &record,
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint network;
    threadgroup int winners[6];
    threadgroup int ids[512];
    threadgroup volatile float scores[512];
    threadgroup volatile float scratch[40];
    ids[tid] = int(tid);
    if (tid < 384u) {
        const float x = logits[row * 384u + tid];
        // Match the reference softplus, including its small-exp polynomial.
        const float ex = exp(x);
        const float em = min(ex, 0.03125f);
        const float poly = em*(1.0f - em*(0.5f - em*(1.0f/3.0f - 0.25f*em)));
        scores[tid] = select(select(log(1.0f + ex), poly, ex < 0.03125f), x, x > 20.0f);
        const float p = sqrt(scores[tid]);
        probs[row * 384u + tid] = p;
        scores[tid] = p + bias[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    // Unique finite winners have the same order in any correct sort. Reject
    // ties and exceptional stored scores before considering that shortcut.
    if (tid < 32u) {
        float values[12];
        uint bad = 0u;
        for (uint i = 0; i < 12u; i++) {
            values[i] = scores[tid + i * 32u];
            const uint bits = as_type<uint>(values[i]) & 0x7fffffffu;
            bad |= (bits >= 0x7f800000u || (bits != 0u && bits < 0x00800000u));
        }
        bad = simd_max(bad);
        float previous = INFINITY;
        for (uint rank = 0u; rank < 7u; rank++) {
            float best = -INFINITY;
            for (uint i = 0; i < 12u; i++) best = max(best,values[i]);
            best = simd_max(best);
            uint id = 512u, count = 0u;
            for (uint i = 0; i < 12u; i++) if (values[i] == best) {
                id = min(id,tid + i * 32u); count++;
            }
            id = simd_min(id); count = simd_sum(count);
            bad |= count != 1u || !(best < previous);
            if (tid == 0u && rank < 6u) winners[rank] = int(id);
            for (uint i = 0; i < 12u; i++) if (tid + i * 32u == id) values[i] = -INFINITY;
            previous = best;
        }
        if (tid == 0u) {
            network = bad;
            if (!bad && record) atomic_fetch_or_explicit(coverage,32u,memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (network) {
    // Match kernel_argsort_f32_i32_desc, including padding and unordered ties.
    for (uint k = 2; k <= 512u; k <<= 1u) {
        for (uint j = k >> 1u; j; j >>= 1u) {
            const uint other = tid ^ j;
            if (other > tid) {
                const int a = ids[tid], b = ids[other];
                const bool swap = (tid & k) == 0u ?
                    (a >= 384 || (b < 384 && scores[a] < scores[b])) :
                    (b >= 384 || (a < 384 && scores[a] > scores[b]));
                if (swap) { ids[tid] = b; ids[other] = a; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    } else if (tid < 6u) ids[tid] = winners[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float p = 0.0f, sum = 0.0f;
    if (tid < 6u) {
        selected[row * 6u + tid] = ids[tid];
        p = probs[row * 384u + uint(ids[tid])];
        scratch[tid] = 0.0f;
        sum += p;
        sum = simd_sum(sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) scratch[0] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 6u) {
        sum = scratch[tid];
        sum = simd_sum(sum);
        if (tid == 0u) scratch[32] = clamp(sum, sum_bounds.x, sum_bounds.y);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 6u) scratch[33u + tid] = p / scratch[32];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 6u) weights[row * 6u + tid] = scratch[33u + tid] * scale;
}

// Collapse consumes the previous mixer. Preserve the four ordered FMAs, both
// BF16 stores and the weighted RMSNorm's original 1024-thread reduction.
// The width arrives as the same runtime int32 the reference RMSNorm divides
// by: a compile-time 5120.0f invites fast-math to multiply by a reciprocal.
kernel void kernel_dsv41_hc_norm(
        constant float &eps, device const float *residual,
        device const float *pre, device const float4 *weight,
        device float4 *collapsed, device float4 *normalized,
        constant int32_t &width,
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float4 values[1280];
    threadgroup float sums[32];
    float sum = 0.0f;
    for (uint i = tid; i < 1280u; i += 1024u) {
        float4 v;
        for (uint c = 0; c < 4; c++) {
            volatile float acc = 0.0f;
            for (uint h = 0; h < 4; h++) acc += residual[h * 5120u + i * 4u + c] * pre[h];
            v[c] = dsv41_bf16(acc);
        }
        values[i] = v;
        collapsed[i] = v;
        sum += dot(v, v);
    }
    sum = simd_sum(sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) sums[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = simd_sum(sums[lane]);
    const float mean = sum / width;
    const float scale = 1.0f / sqrt(mean + eps);
    for (uint i = tid; i < 1280u; i += 1024u) {
        volatile float4 v = (values[i] * scale) * weight[i];
        normalized[i] = float4(dsv41_bf16(v.x), dsv41_bf16(v.y),
                               dsv41_bf16(v.z), dsv41_bf16(v.w));
    }
}

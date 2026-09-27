/* Exact Q4_K routed-MoE oracle at the V4.1 shape. Run on an otherwise idle GPU.
 * A hot expert, sparse routes, empty experts and 16/32-row tile tails all occur.
 * Compare both the consumed F16 intermediate and the final F32 output. */
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "line %d: %s\n", __LINE__, #x); exit(1); } } while (0)
typedef struct { uint16_t d, dmin; uint8_t scales[12], qs[128]; } q4_block;
static uint32_t rng = 1;
static uint32_t random_u32(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng;
}

int main(void) {
    enum { D = 5120, H = 2304, E = 384, K = 6, N = 513 };
    const uint64_t row = D / 256 * sizeof(q4_block);
    const uint64_t down_row = H / 256 * sizeof(q4_block);
    const uint64_t expert = H * row, tensor = E * expert, bytes = 3 * tensor;
    void *model = NULL;
    CHECK(posix_memalign(&model, getpagesize(), bytes) == 0);
    memset(model, 0, bytes);
    const int32_t active[] = {0, 1, 63, 127, 190, 191, 192, 193, 255, 381, 382, 383};
    for (unsigned w = 0; w < 3; w++) for (unsigned e = 0; e < 12; e++) {
        q4_block *b = (q4_block *)((char *)model + w * tensor + active[e] * expert);
        for (uint64_t j = 0; j < expert / sizeof(*b); j++) {
            b[j].d = b[j].dmin = 0x1800;
            for (unsigned i = 0; i < sizeof(b[j].scales); i++) b[j].scales[i] = random_u32();
            for (unsigned i = 0; i < sizeof(b[j].qs); i++) b[j].qs[i] = random_u32();
        }
    }
    CHECK(ds4_gpu_init() && ds4_gpu_set_model_map(model, bytes));
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc_managed(N * D * 4ull);
    ds4_gpu_tensor *ids = ds4_gpu_tensor_alloc_managed(N * K * 4ull);
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc_managed(N * K * 4ull);
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(N * K * H * 4ull);
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(N * K * H * 4ull);
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc_managed(N * K * H * 4ull + 16);
    ds4_gpu_tensor *down = ds4_gpu_tensor_alloc(N * K * D * 4ull);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc_managed(N * D * 4ull + 16);
    CHECK(x && ids && weights && gate && up && mid && down && out);
    float *input = ds4_gpu_tensor_contents(x), *route_weights = ds4_gpu_tensor_contents(weights);
    int32_t *route_ids = ds4_gpu_tensor_contents(ids);
    CHECK(input && route_weights && route_ids);
    for (unsigned i = 0; i < N * D; i++) input[i] = ((int)(random_u32() % 101) - 50) / 256.0f;
    const unsigned sizes[] = {32, 33, 47, 48, 49, 63, 64, 65, 127, 128, 129, 511, 512, 513};
    const unsigned order[] = {0, 1, 1, 0};
    void *ref_mid = malloc(N * K * H * 2ull), *ref_out = malloc(N * D * 4ull);
    CHECK(ref_mid && ref_out);
    for (unsigned pattern = 0; pattern < 2; pattern++) {
        for (unsigned r = 0; r < N; r++) for (unsigned s = 0; s < K; s++) {
            route_ids[r * K + s] = s == 0 ? 0 :
                active[1 + ((pattern ? r : r / 17) * 3 + s - 1) % 11];
            route_weights[r * K + s] = (s + 1) / 21.0f;
        }
        for (unsigned shape = 0; shape < sizeof(sizes) / sizeof(*sizes); shape++) {
            const uint32_t n = sizes[shape];
            const size_t mid_bytes = n * K * H * 2ull, out_bytes = n * D * 4ull;
            for (unsigned pass = 0; pass < 4; pass++) {
                if (order[pass]) CHECK(setenv("DS4_METAL_V41_Q4_TAIL_CULL", "1", 1) == 0);
                else CHECK(unsetenv("DS4_METAL_V41_Q4_TAIL_CULL") == 0);
                uint8_t *m = ds4_gpu_tensor_contents(mid), *o = ds4_gpu_tensor_contents(out);
                CHECK(m && o);
                memset(m, 0xa5, mid_bytes + 16); memset(o, 0xa5, out_bytes + 16);
                bool half_mid = false;
                CHECK(ds4_gpu_routed_moe_batch_tensor(out, gate, up, mid, down,
                    model, bytes, 0, tensor, 2 * tensor, 12, 12, expert, row, expert, down_row,
                    D, H, D, ids, weights, E, K, 7.0f, x, 0, n, &half_mid, true));
                CHECK(ds4_gpu_synchronize() && half_mid);
                for (unsigned i = 0; i < 16; i++) CHECK(m[mid_bytes + i] == 0xa5 && o[out_bytes + i] == 0xa5);
                for (unsigned i = 0; i < n * K * H; i++) CHECK(isfinite((float)((_Float16 *)m)[i]));
                for (unsigned i = 0; i < n * D; i++) CHECK(isfinite(((float *)o)[i]));
                if (!pass) { memcpy(ref_mid, m, mid_bytes); memcpy(ref_out, o, out_bytes); }
                else { CHECK(!memcmp(ref_mid, m, mid_bytes)); CHECK(!memcmp(ref_out, o, out_bytes)); }
            }
            printf("Q4 tails pattern=%u rows=%u intermediate/output/guards exact=yes\n", pattern, n);
            fflush(stdout);
        }
    }
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(ids); ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(up); ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(down); ds4_gpu_tensor_free(out);
    ds4_gpu_cleanup(); free(ref_mid); free(ref_out); free(model);
    return 0;
}

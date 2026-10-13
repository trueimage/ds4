/* Execute the production short-prefill selectors on a small fixture. The
 * test flag substitutes only dimensions/device/configuration; row boundaries,
 * quality and streaming exclusions remain active. Compare consumed mid and
 * final output bytes against the diagnostic rollback path. */
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "line %d: %s\n", __LINE__, #x); exit(1); } } while (0)
typedef struct { uint16_t d, dmin; uint8_t scales[12], qs[128]; } q4_block;
static uint32_t rng = 1;
static uint32_t random_u32(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng;
}

int main(int argc, char **argv) {
    if (argc == 1) {
        /* Fresh processes keep model maps, pipeline caches and test flags
         * independent while allowing the normal `make test` runner. */
        const char *families[] = {"q4", "mxfp4"};
        for (unsigned i = 0; i < 2; i++) {
            const pid_t child = fork();
            CHECK(child >= 0);
            if (!child) {
                execl(argv[0], argv[0], families[i], (char *)NULL);
                _exit(127);
            }
            int status;
            CHECK(waitpid(child, &status, 0) == child);
            CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
        }
        return 0;
    }
    CHECK(argc == 2 && (!strcmp(argv[1], "q4") || !strcmp(argv[1], "mxfp4")));
    const bool mxfp4 = !strcmp(argv[1], "mxfp4");
    const uint32_t type = mxfp4 ? 39u : 12u;
    const uint32_t block_size = mxfp4 ? 17u : sizeof(q4_block);
    const uint32_t block_n = mxfp4 ? 32u : 256u;
    const char *rollback = "DS4_METAL_DISABLE_V41_SHORT_EXPERTS";
    /* Keep the reference on the simdgroup path on M5 test hosts too. */
    CHECK(setenv("DS4_METAL_DISABLE_METAL4", "1", 1) == 0);
    CHECK(unsetenv("DS4_METAL_DISABLE_V41_Q4_TAIL_CULL") == 0);
    enum { D = 256, H = 512, E = 32, K = 6, N = 513 };
    const uint64_t row = D / block_n * block_size;
    const uint64_t down_row = H / block_n * block_size;
    const uint64_t expert = H * row, tensor = E * expert, bytes = 3 * tensor;
    void *model = NULL;
    CHECK(posix_memalign(&model, getpagesize(), bytes) == 0);
    memset(model, 0, bytes);
    const int32_t active[] = {0, 1, 3, 7, 14, 15, 16, 17, 23, 29, 30, 31};
    for (unsigned w = 0; w < 3; w++) for (unsigned e = 0; e < 12; e++) {
        uint8_t *data = (uint8_t *)model + w * tensor + active[e] * expert;
        if (mxfp4) {
            for (uint64_t j = 0; j < expert; j += 17u) {
                data[j] = 120u + random_u32() % 6u;
                for (unsigned i = 1; i < 17; i++) data[j + i] = random_u32();
            }
        } else {
            q4_block *b = (q4_block *)data;
            for (uint64_t j = 0; j < expert / sizeof(*b); j++) {
                b[j].d = b[j].dmin = 0x1800;
                for (unsigned i = 0; i < sizeof(b[j].scales); i++) b[j].scales[i] = random_u32();
                for (unsigned i = 0; i < sizeof(b[j].qs); i++) b[j].qs[i] = random_u32();
            }
        }
    }
    CHECK(ds4_gpu_init() && ds4_gpu_set_model_map(model, bytes));
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
    const unsigned sizes[] = {1, 8, 32, 127, 128, 129, 145, 255, 256, 257, 511, 512, 513};
    const unsigned order[] = {0, 1, 1, 0};
    void *ref_mid = malloc(N * K * H * 4ull), *ref_out = malloc(N * D * 4ull);
    CHECK(ref_mid && ref_out);
    for (unsigned pattern = 0; pattern < 6; pattern++) {
        /* Pattern 5 publishes an eligible graph but leaves the real geometry
         * gate active: this small non-V4.1 fixture must still be excluded. */
        ds4_gpu_test_set_flags(DS4_GPU_TEST_V41_Q4_TAIL_CULL |
            (pattern == 5 ? 0u : DS4_GPU_TEST_V41_SHORT_EXPERTS));
        ds4_gpu_dsv41_set_measured_config(pattern == 5);
        ds4_gpu_set_ssd_streaming(pattern == 2);
        ds4_gpu_set_quality(pattern == 3);
        for (unsigned i = 0; i < N * D; i++) input[i] = ((int)(random_u32() % 101) - 50) / 256.0f;
        if (pattern == 4) {
            const uint32_t exceptional[] = {0x7fc00123u, 0x7f800000u, 0xff800000u,
                0x80000000u, 1u, 0x80000001u, 0x33800000u, 0x7f7fffffu};
            for (unsigned r = 0; r < N; r++) memcpy(input + r * D + (r * 17u) % D, exceptional + r % 8u, 4);
        }
        for (unsigned r = 0; r < N; r++) for (unsigned s = 0; s < K; s++) {
            route_ids[r * K + s] = s == 0 ? 0 :
                active[1 + ((pattern ? r : r / 17) * 3 + s - 1) % 11];
            route_weights[r * K + s] = (s + 1) / 21.0f;
        }
        for (unsigned shape = 0; shape < sizeof(sizes) / sizeof(*sizes); shape++) {
            const uint32_t n = sizes[shape];
            const bool expected_half = n >= 32 && pattern != 3;
            const size_t mid_bytes = n * K * H * (expected_half ? 2ull : 4ull);
            const size_t out_bytes = n * D * 4ull;
            for (unsigned pass = 0; pass < 4; pass++) {
                if (order[pass]) CHECK(unsetenv(rollback) == 0);
                else CHECK(setenv(rollback, "1", 1) == 0);
                CHECK(ds4_gpu_test_v41_short_experts_take_dispatches() == 0);
                uint8_t *m = ds4_gpu_tensor_contents(mid), *o = ds4_gpu_tensor_contents(out);
                CHECK(m && o);
                memset(m, 0xa5, mid_bytes + 16); memset(o, 0xa5, out_bytes + 16);
                bool half_mid = !expected_half;
                CHECK(ds4_gpu_routed_moe_batch_tensor(out, gate, up, mid, down,
                    model, bytes, 0, tensor, 2 * tensor, type, type, expert, row, expert, down_row,
                    D, H, D, ids, weights, E, K, 7.0f, x, 0, n, &half_mid, true));
                CHECK(ds4_gpu_synchronize() && half_mid == expected_half);
                const bool admitted = order[pass] && pattern != 2 && pattern != 3 && pattern != 5 &&
                    n >= 128 && n <= 512;
                const uint32_t expected = !admitted ? 0u : mxfp4 ?
                    DS4_GPU_V41_SHORT_MXFP4_PAIR | DS4_GPU_V41_SHORT_MXFP4_DOWN :
                    DS4_GPU_V41_SHORT_Q4_PAIR;
                CHECK(ds4_gpu_test_v41_short_experts_take_dispatches() == expected);
                for (unsigned i = 0; i < 16; i++) CHECK(m[mid_bytes + i] == 0xa5 && o[out_bytes + i] == 0xa5);
                if (pattern != 4) {
                    for (unsigned i = 0; i < n * K * H; i++)
                        CHECK(isfinite(expected_half ? (float)((_Float16 *)m)[i] : ((float *)m)[i]));
                    for (unsigned i = 0; i < n * D; i++) CHECK(isfinite(((float *)o)[i]));
                }
                if (!pass) { memcpy(ref_mid, m, mid_bytes); memcpy(ref_out, o, out_bytes); }
                else { CHECK(!memcmp(ref_mid, m, mid_bytes)); CHECK(!memcmp(ref_out, o, out_bytes)); }
            }
            printf("short experts %s pattern=%u rows=%u intermediate/output/guards exact=yes\n", argv[1], pattern, n);
            fflush(stdout);
        }
    }
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(ids); ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(up); ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(down); ds4_gpu_tensor_free(out);
    ds4_gpu_test_set_flags(0);
    ds4_gpu_dsv41_set_measured_config(0);
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);
    unsetenv(rollback);
    ds4_gpu_cleanup(); free(ref_mid); free(ref_out); free(model);
    return 0;
}

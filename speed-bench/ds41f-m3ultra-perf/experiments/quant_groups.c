/* Campaign microbenchmark: compare launch grouping against one SIMDgroup.
 * Run separately from any model process. Timings include one command drain. */
#include "ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "line %d: %s\n", __LINE__, #x); exit(1); } } while (0)

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(void) {
    CHECK(ds4_gpu_init());
    const uint32_t special[] = {0, 0x80000000u, 1, 0x80000001u, 0x7f7fffffu,
        0xff7fffffu, 0x7f800000u, 0xff800000u, 0x7fc01234u, 0x7f801234u};
    const struct { uint32_t width, rows; } shapes[] = {
        {32, 1}, {96, 3}, {128, 33}, {5120, 1}, {5120, 512}, {5120, 8192}
    };
    const unsigned order[] = {1, 4, 8, 8, 4, 1};
    for (unsigned shape = 0; shape < sizeof(shapes) / sizeof(*shapes); shape++) {
        const size_t count = (size_t)shapes[shape].width * shapes[shape].rows;
        const size_t bytes = count * sizeof(uint32_t);
        uint32_t *input = malloc(bytes), *reference = malloc(bytes), *actual = malloc(bytes);
        ds4_gpu_tensor *storage = ds4_gpu_tensor_alloc_managed(bytes + 32);
        CHECK(input && reference && actual && storage);
        ds4_gpu_tensor *view = ds4_gpu_tensor_view(storage, 16, bytes);
        CHECK(view);
        for (unsigned mode = 0; mode < 4; mode++) for (unsigned exceptional = 0; exceptional < 2; exceptional++) {
            for (size_t i = 0; i < count; i++) {
                const float x = ((int)((i * 7919u) % 65537) - 32768) / 8192.0f;
                memcpy(input + i, &x, 4);
                if (exceptional && i % 37 < sizeof(special) / sizeof(*special))
                    input[i] = special[i % 37];
            }
            for (unsigned pass = 0; pass < sizeof(order) / sizeof(*order); pass++) {
                char value[16]; snprintf(value, sizeof(value), "%u", order[pass]);
                CHECK(setenv("DS4_METAL_V41_QUANT_GROUPS", value, 1) == 0);
                uint32_t guard[4] = {0x12345678, 0x12345678, 0x12345678, 0x12345678};
                CHECK(ds4_gpu_tensor_write(storage, 0, guard, 16));
                CHECK(ds4_gpu_tensor_write(storage, bytes + 16, guard, 16));
                CHECK(ds4_gpu_tensor_write(view, 0, input, bytes));
                const double start = now();
                CHECK(ds4_gpu_begin_commands());
                for (unsigned repeat = 0; repeat < 20; repeat++)
                    CHECK(ds4_gpu_dsv41_quantize(view, shapes[shape].width,
                        shapes[shape].rows, (ds4_v41_activation_format)mode));
                CHECK(ds4_gpu_end_commands());
                const double ms = (now() - start) * 50;
                CHECK(ds4_gpu_tensor_read(view, 0, actual, bytes));
                if (!pass) memcpy(reference, actual, bytes);
                else CHECK(!memcmp(reference, actual, bytes));
                uint32_t observed[4];
                CHECK(ds4_gpu_tensor_read(storage, 0, observed, 16));
                CHECK(!memcmp(guard, observed, 16));
                CHECK(ds4_gpu_tensor_read(storage, bytes + 16, observed, 16));
                CHECK(!memcmp(guard, observed, 16));
                printf("width=%u rows=%u mode=%u exceptional=%u pass=%u groups=%u ms=%.6f exact=yes\n",
                    shapes[shape].width, shapes[shape].rows, mode, exceptional, pass, order[pass], ms);
                fflush(stdout);
            }
        }
        ds4_gpu_tensor_free(view); ds4_gpu_tensor_free(storage);
        free(input); free(reference); free(actual);
    }
    ds4_gpu_cleanup();
    return 0;
}

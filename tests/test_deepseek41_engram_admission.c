#include "ds4_gpu.h"
#include "ds4_engram.h"
static __typeof__(ds4_gpu_device_is_m3_ultra) test_device_is_m3_ultra;
static __typeof__(ds4_engram_read_batch) test_read_batch;
static __typeof__(ds4_engram_read_step) test_read_step;
static __typeof__(ds4_engram_read_step_begin) test_read_step_begin;
#define ds4_gpu_device_is_m3_ultra test_device_is_m3_ultra
#define ds4_engram_read_batch test_read_batch
#define ds4_engram_read_step test_read_step
#define ds4_engram_read_step_begin test_read_step_begin
#include "../ds4.c"
#undef ds4_gpu_device_is_m3_ultra
#undef ds4_engram_read_batch
#undef ds4_engram_read_step
#undef ds4_engram_read_step_begin

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "line %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

enum { ORDINARY, STEP, OVERLAPPED, READERS };
static bool m3_ultra;
static unsigned calls[READERS], cases;

static int test_device_is_m3_ultra(void) { return m3_ultra; }

static bool test_read_batch(const ds4_engram_table *table, const uint32_t *rows,
                            size_t tokens, size_t stride, float *out) {
    CHECK(table && rows && out && tokens == 1 && stride == DS4_ENGRAM_COLS);
    calls[ORDINARY]++;
    return false;
}

static bool test_read_step(const ds4_engram_table tables[DS4_ENGRAM_LAYERS],
                           const uint32_t rows[DS4_ENGRAM_LAYERS][DS4_ENGRAM_COLS],
                           float *out[DS4_ENGRAM_LAYERS]) {
    CHECK(tables && rows && out && out[0] && out[1]);
    calls[STEP]++;
    return false;
}

static bool test_read_step_begin(ds4_engram_step *step,
                                 const ds4_engram_table tables[DS4_ENGRAM_LAYERS],
                                 const uint32_t rows[DS4_ENGRAM_LAYERS][DS4_ENGRAM_COLS],
                                 float *out[DS4_ENGRAM_LAYERS]) {
    CHECK(step && tables && rows && out && out[0] && out[1]);
    calls[OVERLAPPED]++;
    return false;
}

static void check_reader(ds41_gpu_graph *g, const ds4_weights *w, unsigned reader) {
    const ds4_model model = {0};
    const ds4_engram_history history = g->history;
    memset(calls, 0, sizeof(calls));
    CHECK(!ds41_graph_step(g, &model, w, 0, NULL));
    for (unsigned i = 0; i < READERS; i++) CHECK(calls[i] == (i == reader));
    CHECK(g->valid && g->pos == 0 && !memcmp(&g->history, &history, sizeof(history)));
    cases++;
}

int main(void) {
    static ds41_gpu_graph g = {.ctx = 1, .valid = true, .tp_world = 1};
    static ds4_weights w;
    ds4_tensor q4 = {.type = DS4_TENSOR_Q4_K}, q8 = {.type = DS4_TENSOR_Q8_0};
    ds4_tensor mxfp4 = {.type = DS4_TENSOR_MXFP4};
    const uint32_t token_map[] = {0};
    const char *rollbacks[] = {
        "DS4_METAL_DISABLE_V41_RESIDENT_DECODE_QUEUE",
        "DS4_METAL_DISABLE_V41_DECODE_PIPELINE",
        "DS4_DISABLE_V41_ENGRAM_STEP_OVERLAP",
        "DS4_DISABLE_V41_ENGRAM_STEP_READERS"
    };
    CHECK(unsetenv("DS4_METAL_V41_DECODE_HOST_PROFILE") == 0);
    g_ds4_shape = DS4_SHAPE_FLASH41;
    g.engram.token_map = token_map;
    g.engram.vocab_size = g.engram.compressed_vocab_size = 1;
    ds4_engram_history_reset(&g.history);
    for (unsigned i = 0; i < DS4_ENGRAM_LAYERS; i++)
        for (unsigned j = 0; j < DS4_ENGRAM_COLS; j++) g.engram.primes[i][j] = 1;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        w.layer[il].ffn_gate_exps = &q4;
        w.layer[il].ffn_up_exps = &q4;
        w.layer[il].ffn_down_exps = &q4;
    }
    for (unsigned device = 0; device < 2; device++) {
        m3_ultra = device != 0;
        for (unsigned mask = 0; mask < 16; mask++) {
            for (unsigned i = 0; i < 4; i++) {
                if (mask & (1u << i)) CHECK(setenv(rollbacks[i], "1", 1) == 0);
                else CHECK(unsetenv(rollbacks[i]) == 0);
            }
            check_reader(&g, &w, !m3_ultra || (mask & 8u) ? ORDINARY :
                         (mask & 7u) ? STEP : OVERLAPPED);
        }
    }
    for (unsigned i = 0; i < 4; i++) CHECK(unsetenv(rollbacks[i]) == 0);
    g.quality = true; check_reader(&g, &w, ORDINARY); g.quality = false;
    g.streaming = true; check_reader(&g, &w, ORDINARY); g.streaming = false;
    g.tp_world = 2; check_reader(&g, &w, ORDINARY); g.tp_world = 1;
    ds4_imatrix_collector imatrix = {0};
    g.imatrix = &imatrix; check_reader(&g, &w, ORDINARY); g.imatrix = NULL;
    const ds4_vision_span image = {.token_start = 1};
    g.images = &image; g.image_count = 1;
    check_reader(&g, &w, ORDINARY);
    g.images = NULL; g.image_count = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_tensor **experts[] = {&w.layer[il].ffn_gate_exps,
                                 &w.layer[il].ffn_up_exps, &w.layer[il].ffn_down_exps};
        for (unsigned i = 0; i < 3; i++) {
            *experts[i] = &q8; check_reader(&g, &w, ORDINARY);
            /* One native tensor in a Q4_K file is a mix, not a measured configuration. */
            *experts[i] = &mxfp4; check_reader(&g, &w, ORDINARY); *experts[i] = &q4;
        }
    }
    check_reader(&g, &w, OVERLAPPED);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++)
        w.layer[il].ffn_gate_exps = w.layer[il].ffn_up_exps = w.layer[il].ffn_down_exps = &mxfp4;
    check_reader(&g, &w, OVERLAPPED);
    w.layer[DS4_N_LAYER - 1u].ffn_down_exps = &q4; check_reader(&g, &w, ORDINARY);
    printf("Engram admission: %u graph cases, device/configuration/rollback reader selection and failure state exact (no GPU initialization)\n", cases);
    return 0;
}

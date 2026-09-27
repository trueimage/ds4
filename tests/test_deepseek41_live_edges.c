/* Resident M3 Ultra integration campaign, explicitly invoked with MODEL PROMPT.
 * The preprocessor interposes only this fixture's graph calls. The production
 * backend, shaders, GGUF and ordinary builds have no injection controls.
 * Restore one 32K session for each reference/candidate timeline, inject at the
 * same real graph boundary, then compare all logits and serialized state.
 */
#include "ds4_gpu.h"
static __typeof__(ds4_gpu_router_select_tensor) live_router;
static __typeof__(ds4_gpu_dsv41_hc_norm) live_hc;
static __typeof__(ds4_gpu_dsv41_shared) live_shared;
static __typeof__(ds4_gpu_dsv41_indexer_topk) live_topk;
#define ds4_gpu_router_select_tensor live_router
#define ds4_gpu_dsv41_hc_norm live_hc
#define ds4_gpu_dsv41_shared live_shared
#define ds4_gpu_dsv41_indexer_topk live_topk
#include "../ds4.c"
#undef ds4_gpu_router_select_tensor
#undef ds4_gpu_dsv41_hc_norm
#undef ds4_gpu_dsv41_shared
#undef ds4_gpu_dsv41_indexer_topk

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "LIVE FAIL case=%s arm=%u line=%d: %s\n", \
            case_name, arm, __LINE__, #x); exit(1); \
} } while (0)

enum edge {
    NATURAL, ROUTER_TIES, ROUTER_ZERO, HC_ZERO, HC_CANCEL, SHARED_ZERO,
    QUALITY_REFUSAL, OWNERSHIP_REFUSAL, TOPK_ACCEPT, TOPK_TIES, TOPK_CUT_TIE,
    TOPK_NAN, TOPK_INF, TOPK_OVERFLOW, TOPK_ZERO, TOPK_RECOVERY, HC_FAILURE
};
static const char *case_name = "setup";
static enum edge edge;
static unsigned arm, step;
static bool active;
static struct {
    unsigned calls[4], fused[3], fallback[3], injected;
    unsigned hist_calls, accepted, nonfinite, overflow, ties, host_skip, max_width;
} counts;

/* Executed input/output contracts, kept in memory only for the matched replay. */
static struct { void *data; size_t bytes; const char *label; } observations[256];
static unsigned observation_count, observation_cursor;
static uint64_t compared_bytes;
static void observe(const char *label, const void *data, size_t bytes) {
    CHECK(data && bytes);
    if (!arm) {
        CHECK(observation_count < 256);
        observations[observation_count].data = malloc(bytes);
        CHECK(observations[observation_count].data);
        memcpy(observations[observation_count].data, data, bytes);
        observations[observation_count].bytes = bytes;
        observations[observation_count++].label = label;
    } else {
        CHECK(observation_cursor < observation_count);
        CHECK(!strcmp(label, observations[observation_cursor].label));
        CHECK(bytes == observations[observation_cursor].bytes);
        if (memcmp(data, observations[observation_cursor].data, bytes)) {
            fprintf(stderr, "mismatched live boundary: %s (%zu bytes)\n", label, bytes);
            CHECK(false);
        }
        observation_cursor++;
        compared_bytes += bytes;
    }
}

static bool pause_graph(void) {
    bool batch = ds4_gpu_commands_active();
    if (batch) CHECK(ds4_gpu_end_commands());
    CHECK(ds4_gpu_synchronize());
    return batch;
}
static void resume_graph(bool batch) {
    if (batch) CHECK(ds4_gpu_begin_commands());
}
static bool guarded(void) {
    return step == 0 && (edge == QUALITY_REFUSAL || edge == OWNERSHIP_REFUSAL);
}
static void guard(bool enabled) {
    if (!guarded()) return;
    /* Probe admission at the real dispatch boundary. This does not change the
     * session's precision or reconfigure its resident weights as an SSD graph. */
    if (edge == QUALITY_REFUSAL) ds4_gpu_set_quality(enabled);
    else ds4_gpu_set_ssd_streaming(enabled);
}
static void dispatch(unsigned which, uint32_t bit) {
    uint32_t got = ds4_gpu_test_v41_fusions_take_dispatches();
    const uint32_t expected = (arm && !guarded()) ? bit : 0;
    /* The router's optional top-six path also reports actual fast acceptance. */
    CHECK(got == expected || (which == 0 && expected == 1u && got == 33u));
    if (got) counts.fused[which]++;
    else counts.fallback[which]++;
}
static float *contents(const ds4_gpu_tensor *t) {
    float *p = ds4_gpu_tensor_contents((ds4_gpu_tensor *)t);
    CHECK(p);
    return p;
}

static int live_router(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
        ds4_gpu_tensor *probs, const void *map, uint64_t size, uint64_t bias_offset,
        uint64_t hash_offset, uint32_t hash_rows, uint32_t token, uint32_t experts,
        uint32_t used, float scale, uint32_t groups, uint32_t group_used,
        bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits) {
    bool sample = active && counts.calls[0]++ == 0;
    bool batch = sample ? pause_graph() : false;
    if (sample) {
        CHECK(experts == 384 && used == 6 && has_bias && !hash_mode);
        float *x = contents(logits);
        observe("router live logits", x, experts * 4u);
        if (!step && (edge == ROUTER_TIES || edge == ROUTER_ZERO)) {
            const float *bias = (const float *)((const uint8_t *)map + bias_offset);
            for (unsigned i = 0; i < experts; i++) {
                x[i] = edge == ROUTER_TIES ? 1e20f : -120.0f;
                /* Real biases remain untouched; their additions round away,
                 * creating equal finite selection scores for all 384 experts. */
                if (edge == ROUTER_TIES) CHECK(sqrtf(x[i]) + bias[i] == sqrtf(x[i]));
            }
            counts.injected++;
        }
        observe("router injected logits", x, experts * 4u);
    }
    if (active) { ds4_gpu_test_v41_fusions_take_dispatches(); guard(true); }
    int rc = ds4_gpu_router_select_tensor(selected, weights, probs, map, size,
        bias_offset, hash_offset, hash_rows, token, experts, used, scale, groups,
        group_used, has_bias, hash_mode, logits);
    if (active) { guard(false); CHECK(rc); dispatch(0, 1); }
    if (sample) {
        observe("router ids", contents(selected), used * 4u);
        observe("router weights", contents(weights), used * 4u);
        observe("router probabilities", contents(probs), experts * 4u);
        if (!step && edge == ROUTER_ZERO) {
            for (unsigned i = 0; i < experts; i++) CHECK(contents(probs)[i] == 0);
            for (unsigned i = 0; i < used; i++) CHECK(contents(weights)[i] == 0);
        }
        resume_graph(batch);
    }
    return rc;
}

static int live_hc(ds4_gpu_tensor *collapsed, ds4_gpu_tensor *norm,
        const ds4_gpu_tensor *residual, const ds4_gpu_tensor *pre,
        const void *map, uint64_t size, uint64_t offset, float eps) {
    bool sample = active && counts.calls[1]++ == 0;
    bool batch = sample ? pause_graph() : false;
    if (sample) {
        float *r = contents(residual), *p = contents(pre);
        observe("HC live residual", r, 4u * 5120u * 4u);
        observe("HC live pre", p, 4u * 4u);
        if (!step && (edge == HC_ZERO || edge == HC_CANCEL)) {
            if (edge == HC_ZERO) {
                for (unsigned i = 0; i < 4u * 5120u; i++) r[i] = i & 1 ? -0.0f : 0.0f;
            } else {
                for (unsigned lane = 1; lane < 4; lane++) memcpy(r + lane * 5120u, r, 5120u * 4u);
                p[0] = p[2] = 1; p[1] = p[3] = -1;
            }
            counts.injected++;
        }
        observe("HC injected residual", r, 4u * 5120u * 4u);
        observe("HC injected pre", p, 4u * 4u);
    }
    if (active) { ds4_gpu_test_v41_fusions_take_dispatches(); guard(true); }
    int rc = ds4_gpu_dsv41_hc_norm(collapsed, norm, residual, pre, map, size,
        sample && edge == HC_FAILURE ? size : offset, eps);
    if (active) {
        guard(false);
        if (edge == HC_FAILURE) {
            CHECK(rc == -1 && !ds4_gpu_test_v41_fusions_take_dispatches());
            counts.injected++;
        } else {
            CHECK(rc == ((arm && !guarded()) ? 1 : 0));
            dispatch(1, 2);
        }
    }
    if (sample) resume_graph(batch);
    return rc; /* rc == 0 must reach the graph's original rollback sequence. */
}

static int live_shared(ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid,
        const ds4_gpu_tensor *x, const void *map, uint64_t size,
        uint64_t gate_offset, uint64_t up_offset, float clamp) {
    bool sample = active && counts.calls[2]++ == 0;
    bool batch = sample ? pause_graph() : false;
    if (sample) {
        observe("shared live norm", contents(x), 5120u * 4u);
        if (!step && edge == SHARED_ZERO) {
            memset(contents(x), 0, 5120u * 4u);
            counts.injected++;
        }
        observe("shared injected norm", contents(x), 5120u * 4u);
    }
    if (active) { ds4_gpu_test_v41_fusions_take_dispatches(); guard(true); }
    int rc = ds4_gpu_dsv41_shared(gate, up, mid, x, map, size, gate_offset, up_offset, clamp);
    if (active) {
        guard(false);
        CHECK(rc == ((arm && !guarded()) ? 1 : 0));
        dispatch(2, 4);
    }
    if (sample) resume_graph(batch);
    return rc;
}

static void inject_scores(float *x, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) x[i] = -1;
    for (uint32_t i = 0; i < 512; i++) x[n - 512 + i] = 1 + (float)i / 512;
    if (edge == TOPK_TIES) x[n - 2] = x[n - 1];
    if (edge == TOPK_CUT_TIE) x[n - 513] = x[n - 512];
    if (edge == TOPK_NAN || edge == TOPK_INF) {
        uint32_t bits = edge == TOPK_NAN ? 0x7fc01234u : 0x7f800000u;
        memcpy(x, &bits, 4);
    }
    if (edge == TOPK_OVERFLOW || edge == TOPK_ZERO || edge == TOPK_RECOVERY) {
        for (uint32_t i = 0; i < n; i++) {
            uint32_t bits = edge == TOPK_OVERFLOW ? 0x3f800000u + i :
                edge == TOPK_ZERO ? (i & 1 ? 0x80000000u : 0) :
                i < n - 512 ? 0xbf800000u : 0x3f800000u + i - (n - 512);
            memcpy(x + i, &bits, 4);
        }
    }
    counts.injected++;
}

static int live_topk(ds4_gpu_tensor *selected, const ds4_gpu_tensor *scores,
        uint32_t width, uint32_t top) {
    if (!active) return ds4_gpu_dsv41_indexer_topk(selected, scores, width, top);
    counts.calls[3]++;
    CHECK(width <= 32768 && top == 512);
    if (width > counts.max_width) counts.max_width = width;
    bool batch = pause_graph();
    observe("topk live scores", contents(scores), width * 4u);
    bool inject = !step && edge >= TOPK_ACCEPT && edge <= TOPK_RECOVERY &&
        !counts.injected && width >= 32000;
    if (inject) {
        inject_scores(contents(scores), width);
        if (edge == TOPK_RECOVERY && arm) {
            CHECK(ds4_gpu_test_glm53_topk_poison_recovery_state());
            ds4_gpu_test_invalidate_completion_counters();
        }
    }
    observe("topk injected scores", contents(scores), width * 4u);
    uint32_t before[15] = {0}, after[15] = {0};
    ds4_gpu_test_glm53_topk_stats(before, 15);
    guard(true);
    int rc = ds4_gpu_dsv41_indexer_topk(selected, scores, width, top);
    guard(false);
    CHECK(rc && ds4_gpu_synchronize());
    ds4_gpu_test_glm53_topk_stats(after, 15);
    unsigned calls = after[4] - before[4], accepted = after[5] - before[5];
    CHECK(calls == ((arm && !guarded()) ? 1u : 0u));
    counts.hist_calls += calls; counts.accepted += accepted;
    counts.nonfinite += after[8] - before[8];
    counts.overflow += after[9] - before[9];
    counts.ties += after[11] - before[11];
    counts.host_skip += !calls;
    if (inject && arm) {
        if (edge == TOPK_ACCEPT || edge == TOPK_RECOVERY) CHECK(accepted == 1);
        else CHECK(!accepted);
        if (edge == TOPK_TIES || edge == TOPK_CUT_TIE) CHECK(after[11] > before[11]);
        if (edge == TOPK_NAN || edge == TOPK_INF) CHECK(after[8] > before[8]);
        if (edge == TOPK_OVERFLOW || edge == TOPK_ZERO) CHECK(after[9] > before[9]);
    }
    observe("topk ordered ids", contents(selected), top * 4u);
    resume_graph(batch);
    return rc;
}

static void rollback(bool on) {
    const char *names[] = {"DS4_METAL_DISABLE_V41_ROUTER_FUSION", "DS4_METAL_DISABLE_V41_HC_NORM",
        "DS4_METAL_DISABLE_V41_SHARED_FUSION", "DS4_METAL_DISABLE_V41_TOPK_FAST",
        "DS4_METAL_DISABLE_V41_COMPUTE_COPY", "DS4_METAL_DISABLE_V41_MATVEC_BF16"};
    for (unsigned i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        if (on) CHECK(setenv(names[i], "1", 1) == 0);
        else CHECK(unsetenv(names[i]) == 0);
    }
}
/* Decode scheduling rollbacks: each restores an original synchronization point. */
static void schedule_rollback(bool on) {
    const char *names[] = {"DS4_METAL_DISABLE_V41_RESIDENT_DECODE_QUEUE",
        "DS4_METAL_DISABLE_V41_DECODE_PIPELINE", "DS4_DISABLE_V41_ENGRAM_STEP_READERS",
        "DS4_DISABLE_V41_ENGRAM_STEP_OVERLAP"};
    for (unsigned i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        if (on) CHECK(setenv(names[i], "1", 1) == 0);
        else CHECK(unsetenv(names[i]) == 0);
    }
}
static void clear_observations(void) {
    for (unsigned i = 0; i < observation_count; i++) free(observations[i].data);
    observation_count = observation_cursor = 0;
    compared_bytes = 0;
}
static void report(void) {
    printf("BRANCH case=%s arm=%u step=%u R=%u/%u H=%u/%u S=%u/%u "
           "hist=%u accepted=%u gpu_fallback=%u host_skip=%u nonfinite=%u overflow=%u ties=%u "
           "max_width=%u injections=%u\n", case_name, arm, step,
           counts.fused[0], counts.fallback[0], counts.fused[1], counts.fallback[1],
           counts.fused[2], counts.fallback[2], counts.hist_calls, counts.accepted,
           counts.hist_calls - counts.accepted, counts.host_skip, counts.nonfinite,
           counts.overflow, counts.ties, counts.max_width, counts.injected);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL.gguf PROMPT_FILE (resident Metal, ctx=32768)\n", argv[0]);
        return 2;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    ds4_engine_options opt = {.model_path = argv[1], .backend = DS4_BACKEND_METAL,
        .context_size = 32768, .power_percent = 100};
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_session_snapshot seed = {0}, reference[2] = {{0}}, actual = {0};
    ds4_tokens tokens = {0};
    char *prompt = NULL, err[256] = {0};
    size_t prompt_bytes;
    CHECK(imatrix_read_text_file(argv[2], &prompt, &prompt_bytes));
    CHECK(ds4_engine_open(&engine, &opt) == 0);
    CHECK(DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_DEEPSEEK41 && ds4_gpu_device_is_m3_ultra());
    CHECK(ds4_session_create(&session, engine, 32768) == 0);
    CHECK(!session->ds41_graph.streaming && !session->ds41_graph.quality &&
          session->ds41_graph.tp_world == 1 && session->ds41_graph.ctx == 32768);
    /* The test flag stands in for the device and the published configuration
     * inside the backend, but the decode schedule is chosen by the graph. */
    CHECK(ds41_measured_config(&session->ds41_graph, &engine->weights));
    ds4_encode_chat_prompt(engine, NULL, prompt, DS4_THINK_NONE, &tokens);
    CHECK(tokens.len >= 32703);
    tokens.len = 32703;
    CHECK(ds4_session_sync(session, &tokens, err, sizeof(err)) == 0);
    CHECK(ds4_session_save_snapshot(session, &seed, err, sizeof(err)) == 0);
    printf("SEED pos=%d ctx=%d vocab=%u snapshot_bytes=%llu resident=1 mmap=1\n",
        ds4_session_pos(session), ds4_session_ctx(session), DS4_N_VOCAB,
        (unsigned long long)seed.len);
    ds4_gpu_test_set_flags(DS4_GPU_TEST_V41_FUSIONS);
    const char *names[] = {"natural", "router-ties", "router-zero", "hc-zero", "hc-cancellation",
        "shared-zero", "quality-refusal", "ownership-refusal", "topk-accept", "topk-winner-tie",
        "topk-cut-tie", "topk-nan", "topk-inf", "topk-overflow", "topk-signed-zero", "topk-recovery"};
    float *logits = malloc(2u * DS4_N_VOCAB * sizeof(float));
    CHECK(logits);
    int reference_tokens[2];
    for (edge = NATURAL; edge <= TOPK_RECOVERY; edge++) {
        case_name = names[edge];
        clear_observations();
        for (arm = 0; arm < 2; arm++) {
            rollback(!arm);
            CHECK(ds4_session_load_snapshot(session, &seed, err, sizeof(err)) == 0);
            for (step = 0; step < 2; step++) {
                memset(&counts, 0, sizeof(counts));
                int token = ds4_session_argmax(session);
                active = true;
                int rc = ds4_session_eval(session, token, err, sizeof(err));
                active = false;
                if (rc) fprintf(stderr, "eval: %s\n", err);
                CHECK(rc == 0 && ds4_session_pos(session) == 32704 + (int)step);
                CHECK(ds4_session_save_snapshot(session, &actual, err, sizeof(err)) == 0);
                CHECK(counts.calls[0] && counts.calls[1] && counts.calls[2] && counts.max_width >= 32000);
                if (!step && edge != NATURAL && !guarded()) CHECK(counts.injected == 1);
                if (!arm) {
                    reference_tokens[step] = token;
                    CHECK(ds4_session_copy_logits(session, logits + step * DS4_N_VOCAB, DS4_N_VOCAB) == (int)DS4_N_VOCAB);
                    ds4_session_snapshot_free(&reference[step]);
                    reference[step] = actual; actual = (ds4_session_snapshot){0};
                } else {
                    CHECK(token == reference_tokens[step]);
                    CHECK(!memcmp(session->logits, logits + step * DS4_N_VOCAB, DS4_N_VOCAB * sizeof(float)));
                    CHECK(actual.len == reference[step].len && !memcmp(actual.ptr, reference[step].ptr, actual.len));
                    printf("EXACT case=%s step=%u token=%d next=%d logits=%u snapshot_bytes=%llu pos=%d\n",
                        case_name, step, token, ds4_session_argmax(session), DS4_N_VOCAB,
                        (unsigned long long)actual.len, ds4_session_pos(session));
                }
                report();
                ds4_session_snapshot_free(&actual);
            }
        }
        CHECK(observation_cursor == observation_count);
        printf("PASS case=%s matched_boundaries=%u compared_boundary_bytes=%llu\n",
            case_name, observation_cursor, (unsigned long long)compared_bytes);
    }
    /* Propagate a real helper error through the graph, reject the incomplete
     * session snapshot, restore, and prove normal inference recovers exactly. */
    clear_observations();
    case_name = "hc-error-recovery"; edge = HC_FAILURE; arm = 0; step = 0;
    rollback(false);
    CHECK(ds4_session_load_snapshot(session, &seed, err, sizeof(err)) == 0);
    memset(&counts, 0, sizeof(counts));
    active = true;
    CHECK(ds4_session_eval(session, ds4_session_argmax(session), err, sizeof(err)) != 0);
    active = false;
    CHECK(counts.injected == 1 && !session->ds41_graph.valid);
    CHECK(ds4_session_save_snapshot(session, &actual, err, sizeof(err)) != 0);
    clear_observations(); edge = NATURAL;
    for (arm = 0; arm < 2; arm++) {
        rollback(!arm);
        CHECK(ds4_session_load_snapshot(session, &seed, err, sizeof(err)) == 0);
        CHECK(ds4_session_eval(session, ds4_session_argmax(session), err, sizeof(err)) == 0);
        CHECK(ds4_session_save_snapshot(session, &actual, err, sizeof(err)) == 0);
        observe("recovered full logits", session->logits, DS4_N_VOCAB * sizeof(float));
        observe("recovered full snapshot", actual.ptr, actual.len);
        ds4_session_snapshot_free(&actual);
    }
    CHECK(observation_cursor == observation_count);
    puts("PASS hc-error-recovery: graph error, invalid snapshot rejection, restore, exact logits/state");
    rollback(false); clear_observations();
    /* Every case above pauses the graph at its interposed calls, so no fused
     * stage or histogram selection ran inside a multi-layer queued command
     * buffer. Nothing is interposed from here on. Arm 0 is the original
     * per-layer schedule with every exact stage rolled back; the others cross
     * the production schedule with the production stages. Each arm replays the
     * same seed and must reproduce arm 0's tokens, logits and final state. */
    case_name = "queued-schedule"; edge = NATURAL; active = false;
    enum { SCHEDULE_STEPS = 16 };
    float *schedule_logits = malloc((size_t)SCHEDULE_STEPS * DS4_N_VOCAB * sizeof(float));
    int schedule_tokens[SCHEDULE_STEPS];
    ds4_session_snapshot schedule_state = {0};
    CHECK(schedule_logits);
    for (arm = 0; arm < 4; arm++) {
        const bool original_stages = arm == 0 || arm == 2;
        const bool original_schedule = arm == 0 || arm == 1;
        rollback(original_stages);
        schedule_rollback(original_schedule);
        CHECK(ds4_session_load_snapshot(session, &seed, err, sizeof(err)) == 0);
        uint32_t before[15] = {0}, after[15] = {0};
        ds4_gpu_test_glm53_topk_stats(before, 15);
        uint64_t buffers = 0, copies = ds4_gpu_tensor_copy_count();
        for (step = 0; step < SCHEDULE_STEPS; step++) {
            const int token = ds4_session_argmax(session);
            const uint64_t buffers_before = ds4_gpu_command_buffer_count();
            CHECK(ds4_session_eval(session, token, err, sizeof(err)) == 0);
            const uint64_t used = ds4_gpu_command_buffer_count() - buffers_before;
            /* The schedule itself, not just its result: one buffer per layer
             * and one for the head, against the pipeline's three. */
            CHECK(original_schedule ? used == (uint64_t)DS4_N_LAYER + 1u : used == 3u);
            buffers += used;
            float *row = schedule_logits + (size_t)step * DS4_N_VOCAB;
            if (!arm) {
                schedule_tokens[step] = token;
                CHECK(ds4_session_copy_logits(session, row, DS4_N_VOCAB) == (int)DS4_N_VOCAB);
            } else {
                CHECK(token == schedule_tokens[step]);
                CHECK(!memcmp(session->logits, row, DS4_N_VOCAB * sizeof(float)));
            }
        }
        ds4_gpu_test_glm53_topk_stats(after, 15);
        /* The histogram selector really ran inside the queued buffers, and
         * only where it is admitted; so did the in-encoder copy. */
        CHECK((after[5] - before[5] != 0) == !original_stages);
        copies = ds4_gpu_tensor_copy_count() - copies;
        CHECK((copies == 0) == !original_stages);
        CHECK(ds4_session_save_snapshot(session, &actual, err, sizeof(err)) == 0);
        if (!arm) { schedule_state = actual; actual = (ds4_session_snapshot){0}; }
        else CHECK(actual.len == schedule_state.len &&
                   !memcmp(actual.ptr, schedule_state.ptr, actual.len));
        printf("SCHEDULE arm=%u original_stages=%d original_schedule=%d steps=%d "
               "command_buffers=%llu blit_copies=%llu histogram_accepted=%u snapshot_bytes=%llu\n",
               arm, original_stages, original_schedule, SCHEDULE_STEPS,
               (unsigned long long)buffers, (unsigned long long)copies, after[5] - before[5],
               (unsigned long long)(arm ? actual.len : schedule_state.len));
        ds4_session_snapshot_free(&actual);
    }
    rollback(false); schedule_rollback(false);
    ds4_session_snapshot_free(&schedule_state); free(schedule_logits);
    puts("PASS queued-schedule: 4 arms x 16 uninterposed steps, full logits/tokens/state exact");
    /* Reuse is valid only within one scalar sweep. Replaying from an empty
     * session also covers ratio-2 layers with no compressed row at pos 0. */
    case_name = "gather-reuse";
    for (unsigned start = 0; start < 2; ++start) {
        const unsigned steps = start ? 16 : 4;
        uint64_t reference_encodes[16];
        clear_observations();
        for (arm = 0; arm < 2; ++arm) {
            if (arm) unsetenv("DS4_METAL_DISABLE_V41_GATHER_REUSE");
            else setenv("DS4_METAL_DISABLE_V41_GATHER_REUSE", "1", 1);
            if (start) CHECK(ds4_session_load_snapshot(session, &seed, err, sizeof(err)) == 0);
            else ds4_session_invalidate(session);
            CHECK(!session->ds41_graph.gather_reuse);
            for (step = 0; step < steps; ++step) {
                const int token = start || step ? ds4_session_argmax(session) : tokens.v[0];
                const uint64_t before = ds4_gpu_encoder_count();
                if (!start && !step) {
                    const ds4_tokens one = {.v = tokens.v, .len = 1};
                    CHECK(ds4_session_sync(session, &one, err, sizeof(err)) == 0);
                } else CHECK(ds4_session_eval(session, token, err, sizeof(err)) == 0);
                const uint64_t encoded = ds4_gpu_encoder_count() - before;
                CHECK(!session->ds41_graph.gather_reuse);
                if (!arm) reference_encodes[step] = encoded;
                else CHECK(reference_encodes[step] - encoded == (!start && !step ? 15u : 30u));
                observe("gather greedy token", &token, sizeof(token));
                observe("gather logits", session->logits, DS4_N_VOCAB * sizeof(float));
            }
            CHECK(ds4_session_save_snapshot(session, &actual, err, sizeof(err)) == 0);
            observe("gather snapshot", actual.ptr, actual.len);
            ds4_session_snapshot_free(&actual);
        }
        CHECK(observation_cursor == observation_count);
        printf("PASS gather-reuse: start=%u steps=%u exact logits/tokens/state; 15/30 dispatches removed\n",
               start ? 32703u : 0u, steps);
    }
    unsetenv("DS4_METAL_DISABLE_V41_GATHER_REUSE");
    clear_observations();
    ds4_session_snapshot_free(&seed);
    for (unsigned i = 0; i < 2; i++) ds4_session_snapshot_free(&reference[i]);
    free(logits); free(prompt); ds4_tokens_free(&tokens);
    ds4_session_free(session); ds4_engine_close(engine);
    puts("PASS live edges: 16 matched cases, full logits/tokens/snapshots, error recovery, queued schedule; max_ctx=32768 max_pos=32719");
    return 0;
}

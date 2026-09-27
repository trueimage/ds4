/* V4.1 speculative verification. Resident real-model fixture; run it only on a
 * dedicated Metal host:
 *   test_deepseek41_dspark MODEL --verify-parity PROMPT   checks
 *   test_deepseek41_dspark MODEL --short-prefill PROMPT   checks
 *   test_deepseek41_dspark MODEL --verify-timing PROMPT   measures
 *   test_deepseek41_dspark MODEL --stage-timing PROMPT    measures */
#include "../ds4.c"

#define REQUIRE(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); goto done; \
} } while (0)

static char *read_text(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    const long size = ftell(fp);
    rewind(fp);
    char *text = size > 0 ? malloc((size_t)size + 1u) : NULL;
    if (text && fread(text, 1, (size_t)size, fp) != (size_t)size) { free(text); text = NULL; }
    if (text) text[size] = 0;
    fclose(fp);
    return text;
}

/* A verified suffix must be what one-token decode computes, and any kept
 * prefix must leave the session that decode would have left: every logit of
 * every row, the serialized state, and the tokens that follow. */
static int check_verify_parity(const char *model, const char *prompt_path) {
    ds4_engine *engine = NULL;
    ds4_session *scalar = NULL, *verified = NULL;
    ds4_session_snapshot base = {0}, want = {0}, got = {0};
    ds4_tokens prompt = {0}, prefix = {0};
    ds41_spec spec = {0};
    float *rows = NULL;
    char err[256] = {0};
    char *text = read_text(prompt_path);
    int rc = 1;
    const uint32_t targets[] = {37, 38, 39};
    ds4_engine_options opt = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = 4096, .power_percent = 100};
    REQUIRE(text);
    REQUIRE(ds4_engine_open(&engine, &opt) == 0);
    ds4_tokenize_text(engine, text, &prompt);
    REQUIRE(prompt.len > 1100);
    REQUIRE(ds4_session_create(&scalar, engine, 4096) == 0);
    REQUIRE(ds4_session_create(&verified, engine, 4096) == 0);
    REQUIRE(ds41_spec_alloc(&spec, targets, 3));
    rows = malloc((size_t)DS4_TP_BATCH_MAX_ROWS * DS4_N_VOCAB * sizeof(float));
    REQUIRE(rows);
    /* Below and across the 128-row window, odd and even frontiers, and a
     * frontier whose rows straddle a window wrap. */
    const struct { int prefix; uint32_t count; } cases[] = {
        {40, 2}, {40, 3}, {40, 6}, {41, 5}, {126, 6}, {250, 8}, {251, 2}, {1021, 6}, {1022, 3},
    };
    /* DS4_TEST_VERIFY_CASE=PREFIX,ROWS runs that one frontier instead. */
    int only_prefix = 0, only_count = 0;
    const char *only = getenv("DS4_TEST_VERIFY_CASE");
    if (only && sscanf(only, "%d,%d", &only_prefix, &only_count) != 2) only = NULL;
    for (size_t c = 0; c < (only ? 1u : sizeof(cases) / sizeof(cases[0])); c++) {
        const int start = only ? only_prefix : cases[c].prefix;
        const uint32_t count = only ? (uint32_t)only_count : cases[c].count;
        REQUIRE(start > 0 && count >= 2 && count <= DS4_TP_BATCH_MAX_ROWS);
        const int *suffix = prompt.v + start;
        prefix.len = 0;
        for (int i = 0; i < start; i++) ds4_tokens_push(&prefix, prompt.v[i]);
        REQUIRE(ds4_session_sync(scalar, &prefix, err, sizeof(err)) == 0);
        REQUIRE(ds4_session_save_snapshot(scalar, &base, err, sizeof(err)) == 0);
        /* One-token reference rows. */
        for (uint32_t i = 0; i < count; i++) {
            REQUIRE(ds4_session_eval(scalar, suffix[i], err, sizeof(err)) == 0);
            memcpy(rows + (size_t)i * DS4_N_VOCAB, scalar->logits, DS4_N_VOCAB * sizeof(float));
        }
        for (uint32_t keep = 1; keep <= count; keep++) {
            REQUIRE(ds4_session_load_snapshot(verified, &base, err, sizeof(err)) == 0);
            ds41_gpu_graph *g = &verified->ds41_graph;
            REQUIRE(g->pos == (uint32_t)start);
            REQUIRE(ds41_graph_verify(g, &engine->model, &engine->weights, suffix, count, &spec));
            REQUIRE(g->pos == (uint32_t)start);
            const float *logits = ds4_gpu_tensor_contents(spec.logits);
            REQUIRE(logits);
            bool exact = true;
            for (uint32_t i = 0; i < count; i++) {
                if (memcmp(logits + (size_t)i * DS4_N_VOCAB, rows + (size_t)i * DS4_N_VOCAB,
                           DS4_N_VOCAB * sizeof(float)) != 0) {
                    double worst = 0;
                    for (uint32_t v = 0; v < DS4_N_VOCAB; v++)
                        worst = fmax(worst, fabs((double)logits[(size_t)i * DS4_N_VOCAB + v] -
                                                 rows[(size_t)i * DS4_N_VOCAB + v]));
                    fprintf(stderr, "prefix %d rows %u: row %u differs from one-token decode (max %.9g)\n",
                            start, count, i, worst);
                    exact = false;
                }
            }
            if (!exact) goto done;
            REQUIRE(ds4_gpu_begin_commands() && ds41_spec_commit(g, &spec, keep) &&
                    ds4_gpu_end_commands());
            REQUIRE(g->pos == (uint32_t)start + keep);
            for (uint32_t i = 0; i < keep; i++) token_vec_push(&verified->checkpoint, suffix[i]);
            memcpy(verified->logits, logits + (size_t)(keep - 1u) * DS4_N_VOCAB,
                   DS4_N_VOCAB * sizeof(float));
            /* The reference for this prefix, and three tokens past it. */
            REQUIRE(ds4_session_load_snapshot(scalar, &base, err, sizeof(err)) == 0);
            for (uint32_t i = 0; i < keep; i++)
                REQUIRE(ds4_session_eval(scalar, suffix[i], err, sizeof(err)) == 0);
            REQUIRE(ds4_session_save_snapshot(scalar, &want, err, sizeof(err)) == 0);
            REQUIRE(ds4_session_save_snapshot(verified, &got, err, sizeof(err)) == 0);
            REQUIRE(want.len == got.len);
            if (memcmp(want.ptr, got.ptr, want.len) != 0) {
                size_t at = 0;
                while (((const uint8_t *)want.ptr)[at] == ((const uint8_t *)got.ptr)[at]) at++;
                fprintf(stderr, "prefix %d rows %u keep %u: state differs at byte %zu of %zu\n",
                        start, count, keep, at, (size_t)want.len);
                goto done;
            }
            for (uint32_t i = 0; i < 3; i++) {
                const int next = suffix[keep + i];
                REQUIRE(ds4_session_eval(scalar, next, err, sizeof(err)) == 0);
                REQUIRE(ds4_session_eval(verified, next, err, sizeof(err)) == 0);
                REQUIRE(memcmp(scalar->logits, verified->logits, DS4_N_VOCAB * sizeof(float)) == 0);
            }
        }
        fprintf(stderr, "V4.1 verify parity prefix=%d rows=%u: every row, kept prefix and continuation exact\n",
                start, count);
        REQUIRE(ds4_session_load_snapshot(scalar, &base, err, sizeof(err)) == 0);
    }
    puts("V4.1 speculative verification: PASS");
    rc = 0;
done:
    if (err[0]) fprintf(stderr, "error: %s\n", err);
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    free(rows);
    ds41_spec_free(&spec);
    ds4_session_snapshot_free(&base); ds4_session_snapshot_free(&want); ds4_session_snapshot_free(&got);
    ds4_tokens_free(&prefix); ds4_tokens_free(&prompt);
    ds4_session_free(verified); ds4_session_free(scalar);
    ds4_engine_close(engine);
    free(text);
    return rc;
}

/* The production continued prefill of two to eight rows is one-token decode:
 * every routed recipe keeps scalar reductions through eight rows. */
static int check_short_prefill(const char *model, const char *prompt_path) {
    ds4_engine *engine = NULL;
    ds4_session *scalar = NULL, *batched = NULL;
    ds4_tokens prompt = {0}, prefix = {0};
    char err[256] = {0};
    char *text = read_text(prompt_path);
    int rc = 1;
    ds4_engine_options opt = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = 4096, .power_percent = 100};
    REQUIRE(text && ds4_engine_open(&engine, &opt) == 0);
    ds4_tokenize_text(engine, text, &prompt);
    REQUIRE(ds4_session_create(&scalar, engine, 4096) == 0);
    REQUIRE(ds4_session_create(&batched, engine, 4096) == 0);
    for (int start = 40; start <= 41; start++) {
        for (int extra = 2; extra <= 8; extra += extra < 5 ? 3 : 1) {
            prefix.len = 0;
            for (int i = 0; i < start; i++) ds4_tokens_push(&prefix, prompt.v[i]);
            ds4_session_invalidate(scalar); ds4_session_invalidate(batched);
            REQUIRE(ds4_session_sync(scalar, &prefix, err, sizeof(err)) == 0);
            REQUIRE(ds4_session_sync(batched, &prefix, err, sizeof(err)) == 0);
            REQUIRE(memcmp(scalar->logits, batched->logits, DS4_N_VOCAB * sizeof(float)) == 0);
            for (int i = 0; i < extra; i++) {
                REQUIRE(ds4_session_eval(scalar, prompt.v[start + i], err, sizeof(err)) == 0);
                ds4_tokens_push(&prefix, prompt.v[start + i]);
            }
            REQUIRE(ds4_session_sync(batched, &prefix, err, sizeof(err)) == 0);
            double worst = 0;
            for (uint32_t v = 0; v < DS4_N_VOCAB; v++)
                worst = fmax(worst, fabs((double)scalar->logits[v] - batched->logits[v]));
            fprintf(stderr, "prefix %d + %d continued rows: max logit difference from decode %.9g\n",
                    start, extra, worst);
            REQUIRE(worst == 0);
        }
    }
    rc = 0;
done:
    if (err[0]) fprintf(stderr, "error: %s\n", err);
    ds4_tokens_free(&prefix); ds4_tokens_free(&prompt);
    ds4_session_free(batched); ds4_session_free(scalar);
    ds4_engine_close(engine);
    free(text);
    return rc;
}

/* Measurement, not a check: what a verified row costs next to a decoded token. */
static int time_verify(const char *model, const char *prompt_path) {
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_session_snapshot base = {0};
    ds4_tokens prompt = {0}, prefix = {0};
    ds41_spec spec = {0};
    char err[256] = {0};
    char *text = read_text(prompt_path);
    int rc = 1;
    const uint32_t targets[] = {37, 38, 39};
    ds4_engine_options opt = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = 4096, .power_percent = 100};
    REQUIRE(text && ds4_engine_open(&engine, &opt) == 0);
    ds4_tokenize_text(engine, text, &prompt);
    REQUIRE(ds4_session_create(&session, engine, 4096) == 0);
    REQUIRE(ds41_spec_alloc(&spec, targets, 3));
    const int start = 1024;
    for (int i = 0; i < start; i++) ds4_tokens_push(&prefix, prompt.v[i]);
    REQUIRE(ds4_session_sync(session, &prefix, err, sizeof(err)) == 0);
    REQUIRE(ds4_session_save_snapshot(session, &base, err, sizeof(err)) == 0);
    for (int warm = 0; warm < 8; warm++)
        REQUIRE(ds4_session_eval(session, prompt.v[start + warm], err, sizeof(err)) == 0);
    double t = now_sec();
    for (int i = 0; i < 32; i++)
        REQUIRE(ds4_session_eval(session, prompt.v[start + 8 + i], err, sizeof(err)) == 0);
    const double step_ms = (now_sec() - t) * 1000.0 / 32.0;
    fprintf(stderr, "one-token decode: %.2f ms\n", step_ms);
    for (uint32_t rows = 2; rows <= 8; rows += 2) {
        const char *ab_env = getenv("DS4_TEST_VERIFY_AB_ENV");
        if (ab_env && !strncmp(ab_env,"DS4_METAL_DISABLE_V41_",sizeof("DS4_METAL_DISABLE_V41_")-1)) {
            const size_t bytes = (size_t)rows * DS4_N_VOCAB * sizeof(float);
            float *reference = malloc(bytes), *actual = malloc(bytes);
            REQUIRE(reference && actual);
            double totals[2] = {0,0};
            for (unsigned rep=0;rep<20;rep++) {
                const unsigned slot=rep%4, cycle=rep/4;
                const unsigned rollback=(slot==1 || slot==2) ^ (cycle%2);
                if (rollback) setenv(ab_env,"1",1); else unsetenv(ab_env);
                REQUIRE(ds4_session_load_snapshot(session,&base,err,sizeof(err))==0);
                const double begin=now_sec();
                REQUIRE(ds41_graph_verify(&session->ds41_graph,&engine->model,&engine->weights,
                                          prompt.v+start,rows,&spec));
                const double ms=(now_sec()-begin)*1000.0;
                if(rep>=4) totals[rollback]+=ms;
                REQUIRE(ds4_gpu_tensor_read(spec.logits,0,actual,bytes));
                if(!rep) memcpy(reference,actual,bytes);
                else REQUIRE(!memcmp(reference,actual,bytes));
            }
            unsetenv(ab_env);
            fprintf(stderr,"verify_ab env=%s rows=%u control_ms=%.6f rollback_ms=%.6f saved_percent=%.6f exact_floats=%llu\n",
                ab_env,rows,totals[0]/8.0,totals[1]/8.0,
                100.0*(totals[1]-totals[0])/totals[1],
                (unsigned long long)rows*DS4_N_VOCAB*20u);
            free(reference);free(actual);
            continue;
        }
        double total = 0;
        for (int rep = 0; rep < 5; rep++) {
            REQUIRE(ds4_session_load_snapshot(session, &base, err, sizeof(err)) == 0);
            t = now_sec();
            REQUIRE(ds41_graph_verify(&session->ds41_graph, &engine->model, &engine->weights,
                                      prompt.v + start, rows, &spec));
            if (rep) total += (now_sec() - t) * 1000.0;
        }
        fprintf(stderr, "verify %u rows: %.2f ms = %.2f decode steps, %.2f ms per row\n",
                rows, total / 4.0, total / 4.0 / step_ms, total / 4.0 / rows);
    }
    rc = 0;
done:
    if (err[0]) fprintf(stderr, "error: %s\n", err);
    ds41_spec_free(&spec);
    ds4_session_snapshot_free(&base);
    ds4_tokens_free(&prefix); ds4_tokens_free(&prompt);
    ds4_session_free(session);
    ds4_engine_close(engine);
    free(text);
    return rc;
}

/* Measurement: where one decoded token's GPU time goes, from 40 encodes of
 * one layer's own kernels on its real weights. */
static int time_stages(const char *model, const char *prompt_path) {
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_tokens prompt = {0}, prefix = {0};
    char err[256] = {0};
    char *text = read_text(prompt_path);
    int rc = 1;
    ds4_engine_options opt = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = 4096, .power_percent = 100};
    REQUIRE(text && ds4_engine_open(&engine, &opt) == 0);
    ds4_tokenize_text(engine, text, &prompt);
    REQUIRE(ds4_session_create(&session, engine, 4096) == 0);
    for (int i = 0; i < 1024; i++) ds4_tokens_push(&prefix, prompt.v[i]);
    REQUIRE(ds4_session_sync(session, &prefix, err, sizeof(err)) == 0);
    for (int i = 0; i < 8; i++)
        REQUIRE(ds4_session_eval(session, prompt.v[1024 + i], err, sizeof(err)) == 0);
    ds41_gpu_graph *g = &session->ds41_graph;
    const ds4_model *m = &engine->model;
    const ds4_weights *w = &engine->weights;
    (void)ds41_publish_measured_config(g, w);
    const char *names[] = {"attention projections (q_a, q_b, kv)", "attention core, index and window",
        "attention output (o_a, o_b)", "shared expert", "router and routed experts",
        "hyper-connection mixes and norms", "output head (once per token)"};
    for (int stage = 0; stage < 7; stage++) {
        double best = 1e30;
        for (int rep = 0; rep < 4; rep++) {
            const uint32_t pos = g->pos;
            REQUIRE(ds4_gpu_begin_commands());
            bool ok = true;
            const int loops = stage == 6 ? 8 : 40;
            for (int i = 0; ok && i < loops; i++) {
                const uint32_t il = stage == 6 ? 0 : (uint32_t)i;
                const ds4_layer_weights *l = &w->layer[il];
                switch (stage) {
                case 0: ok = ds41_attention_project(g, m, l); break;
                case 1: ok = ds41_attention(g, m, l, il, true); break;
                case 2: ok = ds41_attention_output(g, m, l, true); break;
                case 3: ok = ds41_shared_mid(g, m, l) &&
                            ds41_matmul(g->shared, m, l->ffn_down_shexp, g->shared_mid, true); break;
                case 4: {
                    uint64_t gate_row = 0, down_row = 0;
                    ok = tensor_nbytes(l->ffn_gate_exps->type, DS4_N_EMBD, &gate_row) &&
                        tensor_nbytes(l->ffn_down_exps->type, DS4_N_FF_EXP, &down_row) &&
                        ds41_matmul(g->route_logits, m, l->ffn_gate_inp, g->norm, false) &&
                        ds4_gpu_router_select_tensor(g->selected, g->route_weights, g->route_probs,
                            m->map, m->size, l->ffn_exp_probs_b->abs_offset, 0, 0, 0,
                            DS4_N_EXPERT, DS4_N_EXPERT_USED, DS4_EXPERT_WEIGHT_SCALE, 0, 0, true, false,
                            g->route_logits) &&
                        ds4_gpu_routed_moe_one_tensor(g->routed, g->gate, g->up, g->mid, g->experts,
                            m->map, m->size, l->ffn_gate_exps->abs_offset, l->ffn_up_exps->abs_offset,
                            l->ffn_down_exps->abs_offset, l->ffn_gate_exps->type, l->ffn_down_exps->type,
                            gate_row * DS4_N_FF_EXP, gate_row, down_row * DS4_N_EMBD, down_row,
                            DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EMBD, g->selected, g->route_weights,
                            DS4_N_EXPERT, DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP, g->norm, NULL, il, true);
                    break;
                }
                case 5: ok = ds41_hc_mix(g, m, l, false) &&
                            ds41_hc_collapse_norm(g, m, l->attn_norm, g->residual, g->pre) &&
                            ds41_hc_mix(g, m, l, true) &&
                            ds41_hc_collapse_norm(g, m, l->ffn_norm, g->after_attn, g->attn_split); break;
                default: ok = ds41_graph_logits_encode(g, m, w); break;
                }
            }
            const double t = now_sec();
            REQUIRE(ok && ds4_gpu_end_commands());
            const double ms = (now_sec() - t) * 1000.0 / (stage == 6 ? 8.0 : 1.0);
            if (ms < best) best = ms;
            g->pos = pos;
        }
        fprintf(stderr, "%-44s %7.2f ms per token\n", names[stage], best);
    }
    /* How the Q8_0 row kernels scale: 40 layers' q_b projection per launch. */
    for (uint32_t rows = 1; rows <= 8; rows += rows < 2 ? 1 : 2) {
        double best = 1e30;
        for (int rep = 0; rep < 4; rep++) {
            REQUIRE(ds4_gpu_begin_commands());
            bool ok = true;
            for (uint32_t il = 0; ok && il < 40; il++)
                ok = ds41_matmul_batch(g->batch.q, m, w->layer[il].attn_q_b, g->batch.qr, rows, false);
            const double t = now_sec();
            REQUIRE(ok && ds4_gpu_end_commands());
            if ((now_sec() - t) * 1000.0 < best) best = (now_sec() - t) * 1000.0;
        }
        fprintf(stderr, "q_b projection, %u rows: %7.2f ms for 40 layers (%.2f ms per row)\n",
                rows, best, best / rows);
    }
    rc = 0;
done:
    if (err[0]) fprintf(stderr, "error: %s\n", err);
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_tokens_free(&prefix); ds4_tokens_free(&prompt);
    ds4_session_free(session);
    ds4_engine_close(engine);
    free(text);
    return rc;
}

int main(int argc, char **argv) {
    if (argc == 4 && !strcmp(argv[2], "--stage-timing"))
        return time_stages(argv[1], argv[3]);
    if (argc == 4 && !strcmp(argv[2], "--verify-timing"))
        return time_verify(argv[1], argv[3]);
    if (argc == 4 && !strcmp(argv[2], "--short-prefill"))
        return check_short_prefill(argv[1], argv[3]);
    if (argc == 4 && !strcmp(argv[2], "--verify-parity"))
        return check_verify_parity(argv[1], argv[3]);
    fprintf(stderr, "usage: %s MODEL --verify-parity PROMPT\n", argv[0]);
    return 2;
}

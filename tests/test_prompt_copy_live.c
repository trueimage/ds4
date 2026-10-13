/* Production-API prompt-copy oracle. No private graph access or test math.
 * DS4_TEST_MODEL=... ./tests/test_prompt_copy_live prompt.txt [prompt.txt ...]
 * DS4_TEST_COPY_PAD=8192 adds that many text tokens before the chat fixture.
 * The scalar session alone constructs every reference snapshot. */
#include "ds4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char err[256];
static int vocab;
static unsigned blocks, rows, rewinds, full, partial;
static ds4_session_snapshot context_base;
static int context_start;
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__, #x, err); exit(1); } } while (0)

static int top(ds4_session *s) {
    uint64_t rng = 1;
    return ds4_session_sample(s, 0, 0, 1, 0, &rng);
}
static void save(ds4_session *s, ds4_session_snapshot *out) {
    CHECK(ds4_session_save_snapshot(s, out, err, sizeof(err)) == 0);
}
static void load(ds4_session *s, const ds4_session_snapshot *snap) {
    CHECK(ds4_session_load_snapshot(s, snap, err, sizeof(err)) == 0);
}
static void equal_state(ds4_session *s, const ds4_session_snapshot *ref) {
    ds4_session_snapshot got = {0};
    save(s, &got);
    CHECK(got.len == ref->len && !memcmp(got.ptr, ref->ptr, (size_t)got.len));
    ds4_session_snapshot_free(&got);
}
static void equal_logits(ds4_session *a, ds4_session *b) {
    float *x = malloc((size_t)vocab * 2 * sizeof(float)); CHECK(x);
    CHECK(ds4_session_copy_logits(a, x, vocab) == vocab);
    CHECK(ds4_session_copy_logits(b, x + vocab, vocab) == vocab);
    CHECK(!memcmp(x, x + vocab, (size_t)vocab * sizeof(float)));
    free(x); rows++;
}
static int cycle(ds4_engine *e, ds4_session *s, int limit, int out[8]) {
    uint64_t rng = 1;
    return ds4_session_eval_speculative(s, top(s), limit, ds4_token_eos(e),
                                        0, 0, 1, 0, &rng, out, 8, err, sizeof(err));
}
static bool cancel_now(void *ud) { (void)ud; return true; }
static bool cancel_after_verify(void *ud) { return ++*(int *)ud > 1; }

/* Retained-prefix coverage at one naturally accepted full block. Each target
 * scalar trajectory is rebuilt independently; another session verifies before
 * the candidate rewinds, so shared backend scratch cannot stand in for ownership. */
static void boundaries(ds4_engine *e, ds4_session *a, ds4_session *b,
                        const ds4_session_snapshot *base) {
    load(b, base);
    const int start = ds4_session_pos(b);
    for (int keep = 0; keep <= 8; keep++) {
        int out[8], other[8];
        load(a, base); load(b, base);
        CHECK(cycle(e, a, 8, out) == 8);
        CHECK(cycle(e, b, 8, other) == 8);
        CHECK(!memcmp(out, other, sizeof(out)));
        if (keep < 8) CHECK(ds4_session_rewind_speculative(a, start + keep));
        load(b, base);
        for (int i = 0; i < keep; i++) {
            CHECK(top(b) == out[i]);
            CHECK(ds4_session_eval(b, out[i], err, sizeof(err)) == 0);
        }
        equal_logits(a, b);
        ds4_session_snapshot ref = {0}; save(b, &ref); equal_state(a, &ref);
        ds4_session_snapshot_free(&ref);
        for (int i = 0; i < 3; i++) {
            CHECK(top(a) == top(b));
            int token = top(b);
            CHECK(ds4_session_eval(a, token, err, sizeof(err)) == 0);
            CHECK(ds4_session_eval(b, token, err, sizeof(err)) == 0);
            equal_logits(a, b);
        }
        save(b, &ref); equal_state(a, &ref); ds4_session_snapshot_free(&ref);
    }
    for (int cap = 1; cap <= 8; cap++) {
        int out[8]; load(a, base); load(b, base);
        int n = cycle(e, a, cap, out);
        CHECK(n > 0 && n <= cap);
        for (int i = 0; i < n; i++) {
            CHECK(top(b) == out[i]);
            CHECK(ds4_session_eval(b, out[i], err, sizeof(err)) == 0);
        }
        equal_logits(a, b);
        ds4_session_snapshot ref = {0}; save(b, &ref); equal_state(a, &ref);
        ds4_session_snapshot_free(&ref);
    }
    for (int cap = 1; cap <= 8; cap++) {
        int guarded[10];
        for (int i = 0; i < 10; i++) guarded[i] = -12345;
        load(a, base); load(b, base);
        uint64_t rng = 1;
        int n = ds4_session_eval_speculative(a, top(a), 8, ds4_token_eos(e),
            0, 0, 1, 0, &rng, guarded + 1, cap, err, sizeof(err));
        CHECK(n > 0 && n <= cap && guarded[0] == -12345);
        for (int i = n + 1; i < 10; i++) CHECK(guarded[i] == -12345);
        for (int i = 0; i < n; i++) {
            CHECK(top(b) == guarded[i + 1]);
            CHECK(ds4_session_eval(b, guarded[i + 1], err, sizeof(err)) == 0);
        }
        equal_logits(a, b);
        ds4_session_snapshot ref = {0}; save(b, &ref); equal_state(a, &ref);
        ds4_session_snapshot_free(&ref);
    }
    /* A different snapshot at the same length must never inherit a rewind. */
    int out[8]; load(a, base); CHECK(cycle(e, a, 8, out) == 8);
    load(b, base);
    for (int i = 0; i < 8; i++) {
        int token = i ? out[i] : (out[0] + 1) % vocab;
        CHECK(ds4_session_eval(b, token, err, sizeof(err)) == 0);
    }
    ds4_session_snapshot other = {0}; save(b, &other); load(a, &other);
    CHECK(!ds4_session_rewind_speculative(a, start + 3));
    equal_state(a, &other); ds4_session_snapshot_free(&other);

    load(a, base); ds4_session_set_cancel(a, cancel_now, NULL);
    CHECK(cycle(e, a, 8, out) < 0);
    ds4_session_set_cancel(a, NULL, NULL); equal_state(a, base);
    int calls = 0; ds4_session_set_cancel(a, cancel_after_verify, &calls);
    CHECK(cycle(e, a, 8, out) < 0 && !ds4_session_checkpoint_valid(a));
    ds4_session_set_cancel(a, NULL, NULL); load(a, base);
    CHECK(cycle(e, a, 8, out) == 8);
    ds4_session_invalidate(a); load(a, base);
    CHECK(!ds4_session_rewind_speculative(a, start - 1));
    CHECK(cycle(e, a, 8, out) == 8);

    /* Sampled and ignore-EOS entry points must retain one-token behavior. */
    load(a, base); load(b, base);
    uint64_t rng = 123;
    int token = top(b);
    CHECK(ds4_session_eval_speculative(a, token, 8, ds4_token_eos(e),
            0.7f, 0, 0.9f, 0, &rng, out, 8, err, sizeof(err)) == 1);
    CHECK(ds4_session_eval(b, token, err, sizeof(err)) == 0);
    equal_logits(a, b);
    load(a, base); load(b, base);
    token = ds4_session_argmax_ignoring_eos(a, DS4_THINK_NONE);
    CHECK(ds4_session_eval_speculative_argmax_ignoring_eos(a, token, 8,
            ds4_token_eos(e), DS4_THINK_NONE, out, 8, err, sizeof(err)) == 1);
    CHECK(ds4_session_eval(b, token, err, sizeof(err)) == 0);
    equal_logits(a, b);

    load(a, base); load(b, base);
    CHECK(setenv("DS4_METAL_DISABLE_V41_PROMPT_COPY", "1", 1) == 0);
    CHECK(!ds4_session_prompt_copy_supported(a));
    token = top(b);
    CHECK(cycle(e, a, 8, out) == 1 && out[0] == token);
    CHECK(ds4_session_eval(b, token, err, sizeof(err)) == 0);
    equal_logits(a, b);
    ds4_session_snapshot ref = {0}; save(b, &ref); equal_state(a, &ref);
    ds4_session_snapshot_free(&ref);
    CHECK(unsetenv("DS4_METAL_DISABLE_V41_PROMPT_COPY") == 0);
    CHECK(ds4_session_prompt_copy_supported(a));
    printf("boundaries: PASS keep=0..8, independent max/cap=1..8, cross-session, same-length restore, cancel/reuse, sampled/ignore-EOS, rollback\n");
}

/* Snapshot payloads can move to a smaller allocation when the live prefix
 * fits. Two fresh small sessions keep the complete state headers comparable
 * and let the context limit, rather than max_tokens, constrain the cycle. */
static void context_limits(ds4_engine *e) {
    CHECK(context_base.ptr);
    for (int room = 1; room <= 7; room++) {
        ds4_session *a = NULL, *b = NULL;
        CHECK(ds4_session_create(&a, e, context_start + room) == 0);
        CHECK(ds4_session_create(&b, e, context_start + room) == 0);
        load(a, &context_base); load(b, &context_base);
        while (ds4_session_pos(a) < ds4_session_ctx(a)) {
            int out[8], remaining = ds4_session_ctx(a) - ds4_session_pos(a);
            int n = cycle(e, a, 8, out); CHECK(n > 0 && n <= remaining);
            for (int i = 0; i < n; i++) {
                CHECK(top(b) == out[i]);
                CHECK(ds4_session_eval(b, out[i], err, sizeof(err)) == 0);
            }
            equal_logits(a, b);
            ds4_session_snapshot ref = {0}; save(b, &ref); equal_state(a, &ref);
            ds4_session_snapshot_free(&ref);
        }
        int out[8]; CHECK(cycle(e, a, 8, out) == 0);
        ds4_session_free(a); ds4_session_free(b);
    }
    printf("context-end: PASS remaining room=1..7, full-context no-op\n");
}

static char *read_text(const char *path) {
    FILE *f = fopen(path, "rb"); CHECK(f);
    CHECK(fseek(f, 0, SEEK_END) == 0); long size = ftell(f);
    CHECK(size > 0 && size < 4 * 1024 * 1024); rewind(f);
    char *text = malloc((size_t)size + 1); CHECK(text);
    CHECK(fread(text, 1, (size_t)size, f) == (size_t)size); fclose(f);
    text[size] = 0; return text;
}

static void fixture(ds4_engine *e, ds4_session *a, ds4_session *b,
                     const char *path, int pad) {
    char *text = read_text(path);
    ds4_tokens prompt = {0}, chat = {0}, filler = {0};
    ds4_encode_chat_prompt(e, NULL, text, DS4_THINK_NONE, &chat); free(text);
    ds4_tokenize_text(e, "A reference paragraph about careful software testing.\n", &filler);
    CHECK(filler.len > 0);
    for (int i = 0; i < pad; i++) ds4_tokens_push(&prompt, filler.v[i % filler.len]);
    for (int i = 0; i < chat.len; i++) ds4_tokens_push(&prompt, chat.v[i]);
    CHECK(ds4_session_sync(b, &prompt, err, sizeof(err)) == 0);
    ds4_session_snapshot base = {0}; save(b, &base); load(a, &base);
    CHECK(ds4_session_prompt_copy_supported(a));
    ds4_session_snapshot full_base = {0};
    int generated = 0, local_full = 0, stop = 0;
    while (generated < 256) {
        equal_logits(a, b);
        int first = top(b); CHECK(top(a) == first);
        if (ds4_token_is_stop_for_think_mode(e, first, DS4_THINK_NONE)) { stop = 1; break; }
        ds4_session_snapshot refs[9] = {{0}};
        save(b, &refs[0]);
        const int start = ds4_session_pos(b);
        int out[8], n = cycle(e, a, 256 - generated, out);
        CHECK(n > 0 && n <= 8 && n <= 256 - generated);
        for (int i = 0; i < n; i++) {
            CHECK(!ds4_token_is_stop_for_think_mode(e, out[i], DS4_THINK_NONE));
            CHECK(top(b) == out[i]);
            CHECK(ds4_session_eval(b, out[i], err, sizeof(err)) == 0);
            save(b, &refs[i + 1]);
        }
        equal_logits(a, b); equal_state(a, &refs[n]);
        blocks++;
        if (n == 8) { full++; local_full++; }
        else if (n > 1) partial++;
        /* Inspect every returned row and its complete retained state through
         * the public rewind contract, descending without replaying the target. */
        if (n > 1) {
            for (int keep = n - 1; keep >= 0; keep--) {
                CHECK(ds4_session_rewind_speculative(a, start + keep));
                equal_state(a, &refs[keep]); rewinds++;
            }
            load(a, &refs[n]);
        }
        if (n == 8 && !full_base.ptr) { full_base = refs[0]; memset(&refs[0], 0, sizeof(refs[0])); }
        for (int i = 0; i <= n; i++) ds4_session_snapshot_free(&refs[i]);
        generated += n;
    }
    printf("fixture=%s pad=%d generated=%d natural_stop=%d full_blocks=%d: PASS\n",
           path, pad, generated, stop, local_full); fflush(stdout);
    if (full_base.ptr) boundaries(e, a, b, &full_base);
    if (full_base.ptr && !context_base.ptr) {
        load(b, &full_base);
        context_start = ds4_session_pos(b);
        context_base = full_base;
        memset(&full_base, 0, sizeof(full_base));
    }
    ds4_session_snapshot_free(&full_base); ds4_session_snapshot_free(&base);
    ds4_tokens_free(&prompt); ds4_tokens_free(&chat); ds4_tokens_free(&filler);
}

int main(int argc, char **argv) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || argc < 2) { fprintf(stderr, "DS4_TEST_MODEL=model %s prompt.txt [prompt.txt ...]\n", argv[0]); return 2; }
    int pad = getenv("DS4_TEST_COPY_PAD") ? atoi(getenv("DS4_TEST_COPY_PAD")) : 0;
    CHECK(pad >= 0 && pad <= 32000);
    ds4_engine_options opt = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .n_threads = 1, .context_size = pad ? 32768 : 4096,
        .placement_session_count_hint = 2};
    ds4_engine *e = NULL; CHECK(ds4_engine_open(&e, &opt) == 0);
    vocab = ds4_engine_vocab_size(e);
    ds4_session *a = NULL, *b = NULL;
    CHECK(ds4_session_create(&a, e, opt.context_size) == 0);
    CHECK(ds4_session_create(&b, e, opt.context_size) == 0);
    for (int i = 1; i < argc; i++) fixture(e, a, b, argv[i], pad);
    CHECK(full > 0);
    printf("prompt-copy: PASS blocks=%u full=%u partial=%u full-logit-comparisons=%u retained-prefixes=%u\n",
           blocks, full, partial, rows, rewinds);
    ds4_session_free(a); ds4_session_free(b);
    context_limits(e);
    ds4_session_snapshot_free(&context_base);
    ds4_engine_close(e);
    return 0;
}

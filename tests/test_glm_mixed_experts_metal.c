#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { IN = 512, MID = 256, TOTAL = 12, USED = 8, TOKENS = 48 };

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

static uint64_t aligned(uint64_t n, uint64_t a) { return (n + a - 1) / a * a; }
static uint64_t row_bytes(unsigned type, unsigned cols) {
    return type == 8 ? cols / 32 * 34 : cols / 256 * (type == 14 ? 210 : 176);
}

/* Encode known quantized values and record their mathematical weights while
 * packing. The reference does not reuse the GPU's dequantization indexing. */
static void matrix(unsigned char *dst, float *ref, unsigned type,
                   unsigned rows, unsigned cols, unsigned salt) {
    const unsigned block = type == 8 ? 32 : 256;
    const unsigned bytes = type == 8 ? 34 : type == 14 ? 210 : 176;
    const uint16_t scale_bits = 0x1800; /* 2^-9, exactly representable */
    const float d = 1.0f / 512.0f;
    for (unsigned e = 0; e < TOTAL; e++) for (unsigned row = 0; row < rows; row++) {
        for (unsigned b = 0; b < cols / block; b++) {
            unsigned char *p = dst + ((uint64_t)e * rows + row) * row_bytes(type, cols) + b * bytes;
            float *v = ref + ((uint64_t)e * rows + row) * cols + b * block;
            memset(p, 0, bytes);
            if (type == 8) {
                memcpy(p, &scale_bits, 2);
                for (unsigned k = 0; k < 32; k++) {
                    const int q = (int)((k * 37 + row * 19 + e * 11 + salt + b * 5) % 255) - 127;
                    p[2 + k] = (unsigned char)(int8_t)q;
                    v[k] = d * q;
                }
            } else if (type == 14) {
                memcpy(p + 208, &scale_bits, 2);
                for (unsigned g = 0; g < 16; g++) p[192 + g] = (unsigned char)(int8_t)((int)((g + row + salt) % 9) - 4);
                for (unsigned half = 0; half < 2; half++) for (unsigned l = 0; l < 32; l++) {
                    for (unsigned quarter = 0; quarter < 4; quarter++) {
                        const unsigned k = half * 128 + quarter * 32 + l;
                        const unsigned q = (k * 13 + row * 7 + e * 3 + salt + b * 17) % 64;
                        const unsigned lo = half * 64 + (quarter % 2) * 32 + l;
                        p[lo] |= (q & 15) << (quarter / 2 * 4);
                        p[128 + half * 32 + l] |= (q >> 4) << (quarter * 2);
                        v[k] = d * (int8_t)p[192 + k / 16] * ((int)q - 32);
                    }
                }
            } else {
                memcpy(p, &scale_bits, 2); memcpy(p + 2, &scale_bits, 2);
                unsigned sc[8], mn[8];
                for (unsigned g = 0; g < 8; g++) {
                    sc[g] = 1 + (g * 9 + row + salt) % 63;
                    mn[g] = (g * 7 + e + salt) % 64;
                }
                for (unsigned g = 0; g < 4; g++) {
                    p[4 + g] = sc[g] | ((sc[g + 4] >> 4) << 6);
                    p[8 + g] = mn[g] | ((mn[g + 4] >> 4) << 6);
                    p[12 + g] = (sc[g + 4] & 15) | ((mn[g + 4] & 15) << 4);
                }
                for (unsigned g = 0; g < 8; g++) for (unsigned l = 0; l < 32; l++) {
                    const unsigned q = (g * 7 + l * 13 + row + e + salt + b * 3) % 32;
                    p[48 + (g / 2) * 32 + l] |= (q & 15) << (g % 2 * 4);
                    p[16 + l] |= (q >> 4) << g;
                    v[g * 32 + l] = d * sc[g] * q - d * mn[g];
                }
            }
        }
    }
}

static float dot(const float *a, const float *b, unsigned n) {
    double sum = 0;
    for (unsigned i = 0; i < n; i++) sum += (double)a[i] * b[i];
    return (float)sum;
}

static int close_values(const char *stage, const float *a, const float *b, uint64_t n, float tol) {
    float worst = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (!isfinite(a[i]) || !isfinite(b[i])) return 0;
        float error = fabsf(a[i] - b[i]);
        if (error > worst) worst = error;
        if (error > tol * (1.0f + fabsf(b[i]))) {
            fprintf(stderr, "%s mismatch %llu: %g vs %g (tolerance %g)\n",
                    stage, (unsigned long long)i, a[i], b[i], tol);
            return 0;
        }
    }
    fprintf(stderr, "%s max_abs=%g\n", stage, worst);
    return 1;
}

static int run(unsigned type, unsigned nt, int quality) {
    const uint64_t page = getpagesize(), gr = row_bytes(type, IN), dr = row_bytes(8, MID);
    const uint64_t ge = MID * gr, de = IN * dr;
    const uint64_t uo = aligned(TOTAL * ge, page), doff = aligned(uo + TOTAL * ge, page);
    const uint64_t size = aligned(doff + TOTAL * de, page), pairs = (uint64_t)nt * USED * MID;
    unsigned char *model = NULL;
    if (posix_memalign((void **)&model, page, size)) return 0;
    memset(model, 0, size);
    float *gm = calloc((uint64_t)TOTAL * MID * IN, sizeof(float));
    float *um = calloc((uint64_t)TOTAL * MID * IN, sizeof(float));
    float *dm = calloc((uint64_t)TOTAL * IN * MID, sizeof(float));
    float *x = calloc((uint64_t)nt * IN, sizeof(float));
    int32_t *ids = calloc(nt * USED, sizeof(int32_t));
    float *weights = calloc(nt * USED, sizeof(float));
    float *g = calloc(pairs, sizeof(float)), *u = calloc(pairs, sizeof(float));
    float *mid = calloc(pairs, sizeof(float)), *ref = calloc((uint64_t)nt * IN, sizeof(float));
    float *got = calloc(pairs, sizeof(float)), *first = calloc((uint64_t)nt * IN, sizeof(float));
    if (!gm || !um || !dm || !x || !ids || !weights || !g || !u || !mid || !ref || !got || !first) abort();
    matrix(model, gm, type, MID, IN, 1);
    matrix(model + uo, um, type, MID, IN, 5);
    matrix(model + doff, dm, 8, IN, MID, 9);
    for (unsigned t = 0; t < nt; t++) {
        for (unsigned k = 0; k < IN; k++) x[t * IN + k] = ((int)((k * 17 + t * 7) % 41) - 20) / 64.0f;
        for (unsigned s = 0; s < USED; s++) {
            const unsigned e = (s * 5 + t) % TOTAL, base = (t * USED + s) * MID;
            ids[t * USED + s] = e; weights[t * USED + s] = (s + 1) / 36.0f;
            for (unsigned row = 0; row < MID; row++) {
                g[base + row] = dot(gm + ((uint64_t)e * MID + row) * IN, x + t * IN, IN);
                u[base + row] = dot(um + ((uint64_t)e * MID + row) * IN, x + t * IN, IN);
                const float gv = fminf(g[base + row], 0.25f);
                const float uv = fmaxf(-0.25f, fminf(u[base + row], 0.25f));
                mid[base + row] = gv / (1.0f + expf(-gv)) * uv * weights[t * USED + s];
            }
            for (unsigned row = 0; row < IN; row++) ref[t * IN + row] += dot(dm + ((uint64_t)e * IN + row) * MID, mid + base, MID);
        }
    }
    /* The batch quality path exposes clamped gate/up scratch. Decode and
     * fused batch activation leave those scratch arrays unmodified. */
    if (quality && nt > 1) for (uint64_t i = 0; i < pairs; i++) {
        g[i] = fminf(g[i], 0.25f);
        u[i] = fmaxf(-0.25f, fminf(u[i], 0.25f));
    }
    ds4_gpu_set_quality(quality);
    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc((uint64_t)nt * IN * 4);
    ds4_gpu_tensor *ti = ds4_gpu_tensor_alloc(nt * USED * 4), *tw = ds4_gpu_tensor_alloc(nt * USED * 4);
    ds4_gpu_tensor *tg = ds4_gpu_tensor_alloc(pairs * 4), *tu = ds4_gpu_tensor_alloc(pairs * 4);
    ds4_gpu_tensor *tm = ds4_gpu_tensor_alloc(pairs * 4), *te = ds4_gpu_tensor_alloc((uint64_t)nt * USED * IN * 4);
    ds4_gpu_tensor *to = ds4_gpu_tensor_alloc((uint64_t)nt * IN * 4);
    int ok = tx && ti && tw && tg && tu && tm && te && to && ds4_gpu_set_model_map(model, size);
    ok = ok && ds4_gpu_tensor_write(tx, 0, x, (uint64_t)nt * IN * 4);
    ok = ok && ds4_gpu_tensor_write(ti, 0, ids, nt * USED * 4) && ds4_gpu_tensor_write(tw, 0, weights, nt * USED * 4);
    for (unsigned rep = 0; rep < 2 && ok; rep++) {
        ok = ds4_gpu_tensor_fill_f32(to, -123.0f - rep, (uint64_t)nt * IN);
        bool half_mid = false;
        if (nt == 1) ok = ok && ds4_gpu_routed_moe_one_tensor(to, tg, tu, tm, te,
            model, size, 0, uo, doff, type, 8, ge, gr, de, dr, IN, MID, IN,
            ti, tw, TOTAL, USED, 0.25f, tx, NULL, 0, true);
        else ok = ok && ds4_gpu_routed_moe_batch_tensor(to, tg, tu, tm, te,
            model, size, 0, uo, doff, type, 8, ge, gr, de, dr, IN, MID, IN,
            ti, tw, TOTAL, USED, 0.25f, tx, 0, nt, &half_mid, true);
        ok = ok && ds4_gpu_tensor_read(tg, 0, got, pairs * 4) && close_values("gate", got, g, pairs, nt >= 32 ? 0.003f : 0.00001f);
        ok = ok && ds4_gpu_tensor_read(tu, 0, got, pairs * 4) && close_values("up", got, u, pairs, nt >= 32 ? 0.003f : 0.00001f);
        ok = ok && ds4_gpu_tensor_read(to, 0, got, (uint64_t)nt * IN * 4) && close_values("out", got, ref, (uint64_t)nt * IN, nt >= 32 ? 0.0003f : 0.00001f);
        if (!rep) memcpy(first, got, (uint64_t)nt * IN * 4);
        else ok = ok && memcmp(first, got, (uint64_t)nt * IN * 4) == 0;
    }
    fprintf(stderr, "mixed experts gate=%u down=8 tokens=%u quality=%d %s\n", type, nt, quality, ok ? "PASS" : "FAIL");
    ds4_gpu_tensor_free(tx); ds4_gpu_tensor_free(ti); ds4_gpu_tensor_free(tw);
    ds4_gpu_tensor_free(tg); ds4_gpu_tensor_free(tu); ds4_gpu_tensor_free(tm);
    ds4_gpu_tensor_free(te); ds4_gpu_tensor_free(to);
    ds4_gpu_set_model_map(NULL, 0);
    free(model); free(gm); free(um); free(dm); free(x); free(ids); free(weights);
    free(g); free(u); free(mid); free(ref); free(got); free(first);
    return ok;
}

int main(void) {
    if (!ds4_gpu_init()) return 1;
    ds4_gpu_set_ssd_streaming(false);
    const unsigned types[] = {14, 13, 8}, counts[] = {1, 7, TOKENS};
    for (unsigned i = 0; i < 3; i++) for (unsigned j = 0; j < 3; j++) for (int q = 0; q < 2; q++) {
        if (!run(types[i], counts[j], q)) return 1;
    }
    return 0;
}

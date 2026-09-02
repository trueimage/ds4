/*
 * Requantize GLM 5.3 KDA projections in place from BF16 to a smaller type.
 *
 * The shipped GLM-5.3-Flash Q4_K artifact stores blk.N.kda_{q,k,v,output}
 * as BF16 while its experts are Q4_K.  Those four tensors are dense -- every
 * one is read on every decoded token -- so at 34 KDA layers they account for
 * roughly 8.5 GiB of the per-token read traffic, more than all routed experts
 * combined.  glm53_quantize.py already emits Q8_0 for role="linear_attention"
 * on its q4 artifact, so this is a supported shape; this tool produces the
 * same thing from an existing GGUF without needing the source checkpoint.
 *
 * Everything other than the selected tensors is copied byte for byte, and the
 * quantization goes through the same quants.c facade the other tools use, so
 * the output differs from the input only in those tensors' type and payload.
 */

#include "quants.h"

#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    GV_U8 = 0, GV_I8 = 1, GV_U16 = 2, GV_I16 = 3, GV_U32 = 4, GV_I32 = 5,
    GV_F32 = 6, GV_BOOL = 7, GV_STR = 8, GV_ARR = 9, GV_U64 = 10,
    GV_I64 = 11, GV_F64 = 12
};

typedef struct {
    const char *name;
    uint64_t    name_len;
    uint32_t    n_dims;
    uint64_t    dims[4];
    uint64_t    ne;
    uint32_t    type;
    uint64_t    offset;
    uint32_t    new_type;
    uint64_t    new_offset;
    uint64_t    new_bytes;
} tinfo;

static const uint8_t *g_base, *g_cur, *g_end;

/* Set once the scratch output exists, so a die() anywhere below does not leave
 * a half-written file sitting next to the real one. */
static char *g_tmp_path;

static void die(const char *msg) __attribute__((noreturn));
static void die(const char *msg) {
    fprintf(stderr, "glm53-requant-kda: %s\n", msg);
    if (g_tmp_path) unlink(g_tmp_path);
    exit(1);
}

static void need(size_t n) {
    if ((size_t)(g_end - g_cur) < n) die("truncated gguf");
}

static uint32_t rd_u32(void) { need(4); uint32_t v; memcpy(&v, g_cur, 4); g_cur += 4; return v; }
static uint64_t rd_u64(void) { need(8); uint64_t v; memcpy(&v, g_cur, 8); g_cur += 8; return v; }

static const char *rd_str(uint64_t *len) {
    uint64_t n = rd_u64();
    need(n);
    const char *s = (const char *)g_cur;
    g_cur += n;
    if (len) *len = n;
    return s;
}

static size_t scalar_size(uint32_t t) {
    switch (t) {
        case GV_U8: case GV_I8: case GV_BOOL: return 1;
        case GV_U16: case GV_I16: return 2;
        case GV_U32: case GV_I32: case GV_F32: return 4;
        case GV_U64: case GV_I64: case GV_F64: return 8;
        default: return 0;
    }
}

/* Skips a metadata value, returning its u32 content when it is a plain u32
 * (used only to pick general.alignment out of the stream). */
static void skip_value(uint32_t t, int *is_u32, uint32_t *u32_out) {
    if (is_u32) *is_u32 = 0;
    if (t == GV_STR) { rd_str(NULL); return; }
    if (t == GV_ARR) {
        uint32_t et = rd_u32();
        uint64_t n  = rd_u64();
        if (et == GV_STR) {
            for (uint64_t i = 0; i < n; i++) rd_str(NULL);
        } else {
            size_t sz = scalar_size(et);
            if (!sz) die("array of unsupported element type");
            need(sz * n);
            g_cur += sz * n;
        }
        return;
    }
    size_t sz = scalar_size(t);
    if (!sz) die("unsupported metadata value type");
    if (t == GV_U32 && is_u32) { *is_u32 = 1; *u32_out = rd_u32(); return; }
    need(sz);
    g_cur += sz;
}

static int is_kda_target(const char *name, uint64_t len) {
    static const char *suffix[] = {
        ".kda_q.weight", ".kda_k.weight", ".kda_v.weight", ".kda_output.weight"
    };
    for (size_t i = 0; i < sizeof(suffix) / sizeof(suffix[0]); i++) {
        size_t sl = strlen(suffix[i]);
        if (len >= sl && memcmp(name + len - sl, suffix[i], sl) == 0) return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr,
                "usage: %s <in.gguf> <out.gguf> [q8_0|q4_K]\n"
                "  Requantizes blk.N.kda_{q,k,v,output}.weight from BF16.\n"
                "  Default target type is q8_0.\n", argv[0]);
        return 2;
    }
    const char *in_path = argv[1], *out_path = argv[2];
    const char *want = (argc == 4) ? argv[3] : "q8_0";
    ds4q_type target;
    if      (!strcmp(want, "q8_0")) target = DS4Q_TYPE_Q8_0;
    else if (!strcmp(want, "q4_K")) target = DS4Q_TYPE_Q4_K;
    else die("target type must be q8_0 or q4_K");
    if (!ds4q_can_quantize(target)) die("quantizer cannot emit that type");

    int fd = open(in_path, O_RDONLY);
    if (fd < 0) die("cannot open input");
    struct stat st;
    if (fstat(fd, &st) != 0) die("cannot stat input");
    /* The input stays mmapped for the whole run, so an output that resolves to
     * the same file would pull the source out from under every read still to
     * come.  st_dev/st_ino catches the hard link and the symlink too, which a
     * string compare of the two paths would not. */
    struct stat out_st;
    if (stat(out_path, &out_st) == 0 &&
        out_st.st_dev == st.st_dev && out_st.st_ino == st.st_ino) {
        die("output resolves to the input; write to a new path instead");
    }
    const size_t in_size = (size_t)st.st_size;
    if (in_size < 24) die("input is too small to be a gguf file");
    void *map = mmap(NULL, in_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) die("cannot mmap input");
    g_base = (const uint8_t *)map;
    g_cur  = g_base;
    g_end  = g_base + in_size;

    need(4);
    if (memcmp(g_cur, "GGUF", 4) != 0) die("not a gguf file");
    g_cur += 4;
    const uint32_t version = rd_u32();
    if (version != 3) fprintf(stderr, "glm53-requant-kda: warning: gguf version %u\n", version);
    const uint64_t n_tensors = rd_u64();
    const uint64_t n_kv      = rd_u64();
    /* A tensor-info entry costs at least 8+4+8+4+8 bytes and a kv pair at
     * least 8+4+1, so a count past these bounds is a corrupt header.  Reject
     * it here rather than at the calloc() it would otherwise size. */
    if (n_tensors > in_size / 32) die("implausible tensor count");
    if (n_kv > in_size / 13) die("implausible metadata count");

    uint32_t alignment = 32;
    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t klen; const char *key = rd_str(&klen);
        uint32_t vt = rd_u32();
        int is_u32 = 0; uint32_t v = 0;
        skip_value(vt, &is_u32, &v);
        if (is_u32 && klen == strlen("general.alignment") &&
            memcmp(key, "general.alignment", klen) == 0) {
            alignment = v ? v : 32;
        }
    }
    if (alignment == 0 || (alignment & (alignment - 1)) != 0 || alignment > 65536) {
        die("general.alignment is not a power of two in range");
    }
    const size_t kv_end = (size_t)(g_cur - g_base);

    tinfo *ts = calloc((size_t)n_tensors, sizeof(*ts));
    if (!ts) die("out of memory");
    for (uint64_t i = 0; i < n_tensors; i++) {
        tinfo *t = &ts[i];
        t->name = rd_str(&t->name_len);
        t->n_dims = rd_u32();
        if (t->n_dims == 0 || t->n_dims > 4) die("tensor with zero or more than 4 dimensions");
        t->ne = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) {
            t->dims[d] = rd_u64();
            /* Both guard the ne/dims[0] divisions below and keep the product
             * from wrapping into a small, plausible-looking byte count. */
            if (t->dims[d] == 0) die("tensor with a zero-length dimension");
            if (t->dims[d] > UINT64_MAX / t->ne) die("tensor element count overflows");
            t->ne *= t->dims[d];
        }
        t->type   = rd_u32();
        t->offset = rd_u64();
    }
    const size_t info_end  = (size_t)(g_cur - g_base);
    const size_t data_start = ds4q_pad(info_end, alignment);
    if (data_start > in_size) die("tensor data section starts past the end of the input");

    /* Plan: pick new types and lay the data section out again. */
    uint64_t cursor = 0, converted = 0, before = 0, after = 0;
    for (uint64_t i = 0; i < n_tensors; i++) {
        tinfo *t = &ts[i];
        int convert = (t->type == DS4Q_TYPE_BF16) && is_kda_target(t->name, t->name_len);
        /* row_size() returns 0 for a type this build does not know, for a row
         * that is not a whole number of blocks, and for anything out of range.
         * Treating that as "copy 0 bytes" would emit a file that still parses
         * but has quietly lost the payload, so stop instead. */
        const size_t row_bytes = ds4q_row_size((ds4q_type)t->type, (int64_t)t->dims[0]);
        if (row_bytes == 0) {
            fprintf(stderr, "glm53-requant-kda: %.*s is type %" PRIu32 ", which this build cannot size\n",
                    (int)t->name_len, t->name, t->type);
            die("refusing to copy a tensor whose layout is unknown");
        }
        const uint64_t nrows = t->ne / t->dims[0];
        const uint64_t old_bytes = (uint64_t)row_bytes * nrows;
        if (t->offset > in_size - data_start ||
            old_bytes > in_size - data_start - t->offset) {
            die("tensor data runs past the end of the input");
        }
        if (convert && (t->dims[0] % (uint64_t)ds4q_block_size(target)) != 0) {
            fprintf(stderr, "glm53-requant-kda: %.*s row %" PRIu64 " not a multiple of the block size; leaving as is\n",
                    (int)t->name_len, t->name, t->dims[0]);
            convert = 0;
        }
        t->new_type = convert ? (uint32_t)target : t->type;
        t->new_bytes = convert
            ? (uint64_t)ds4q_row_size(target, (int64_t)t->dims[0]) * nrows
            : old_bytes;
        cursor = ds4q_pad(cursor, alignment);
        t->new_offset = cursor;
        cursor += t->new_bytes;
        if (convert) { converted++; before += old_bytes; after += t->new_bytes; }
    }
    if (!converted) die("no BF16 kda tensors found -- nothing to do");
    fprintf(stderr,
            "glm53-requant-kda: %" PRIu64 " tensors -> %s, %.2f GiB -> %.2f GiB (saves %.2f GiB per full read)\n",
            converted, ds4q_type_name(target),
            before / 1073741824.0, after / 1073741824.0, (before - after) / 1073741824.0);

    /* Build the file beside its destination and rename it into place at the
     * end: out_path then either still holds whatever it held before, or holds
     * a complete result, and never a truncated one. */
    const size_t tmp_len = strlen(out_path) + 32;
    g_tmp_path = malloc(tmp_len);
    if (!g_tmp_path) die("out of memory");
    snprintf(g_tmp_path, tmp_len, "%s.requant.%ld.tmp", out_path, (long)getpid());
    FILE *out = fopen(g_tmp_path, "wb");
    if (!out) die("cannot open output");
    /* Header and metadata are copied verbatim; tensor-info entries keep their
     * width, so the data section still begins at the same offset. */
    if (fwrite(g_base, 1, kv_end, out) != kv_end) die("write failed");
    for (uint64_t i = 0; i < n_tensors; i++) {
        tinfo *t = &ts[i];
        fwrite(&t->name_len, 8, 1, out);
        fwrite(t->name, 1, t->name_len, out);
        fwrite(&t->n_dims, 4, 1, out);
        for (uint32_t d = 0; d < t->n_dims; d++) fwrite(&t->dims[d], 8, 1, out);
        fwrite(&t->new_type, 4, 1, out);
        if (fwrite(&t->new_offset, 8, 1, out) != 1) die("write failed");
    }
    static const uint8_t zeros[4096] = {0};
    size_t here = (size_t)ftello(out);
    if (here != info_end) die("tensor info section changed size unexpectedly");
    while (here < data_start) {
        size_t n = data_start - here;
        if (n > sizeof(zeros)) n = sizeof(zeros);
        fwrite(zeros, 1, n, out);
        here += n;
    }

    ds4q_quantize_init(target);
    const int64_t CHUNK = 256; /* rows per pass, keeps the f32 staging small */
    for (uint64_t i = 0; i < n_tensors; i++) {
        tinfo *t = &ts[i];
        const size_t want_at = data_start + t->new_offset;
        size_t at = (size_t)ftello(out);
        while (at < want_at) {
            size_t n = want_at - at;
            if (n > sizeof(zeros)) n = sizeof(zeros);
            fwrite(zeros, 1, n, out);
            at += n;
        }
        const uint8_t *src = g_base + data_start + t->offset;
        if (t->new_type == t->type) {
            if (fwrite(src, 1, t->new_bytes, out) != t->new_bytes) die("write failed");
            continue;
        }
        const int64_t ncols = (int64_t)t->dims[0];
        const int64_t nrows = (int64_t)(t->ne / t->dims[0]);
        float *f32 = malloc((size_t)ncols * CHUNK * sizeof(float));
        void  *qbuf = malloc((size_t)ds4q_row_size(target, ncols) * CHUNK);
        if (!f32 || !qbuf) die("out of memory");
        for (int64_t r = 0; r < nrows; r += CHUNK) {
            const int64_t rows = (r + CHUNK <= nrows) ? CHUNK : (nrows - r);
            const uint16_t *bf = (const uint16_t *)src + (size_t)r * ncols;
            for (int64_t k = 0; k < rows * ncols; k++) f32[k] = ds4q_bf16_to_f32(bf[k]);
            size_t wrote = ds4q_quantize_chunk(target, f32, qbuf, 0, rows, ncols, NULL);
            if (fwrite(qbuf, 1, wrote, out) != wrote) die("write failed");
        }
        free(f32);
        free(qbuf);
        fprintf(stderr, "  %.*s -> %s\n", (int)t->name_len, t->name, ds4q_type_name(target));
    }
    if (fclose(out) != 0) die("close failed");
    if (rename(g_tmp_path, out_path) != 0) die("cannot move the finished file into place");
    free(g_tmp_path);
    g_tmp_path = NULL;
    munmap(map, in_size);
    close(fd);
    fprintf(stderr, "glm53-requant-kda: wrote %s\n", out_path);
    return 0;
}

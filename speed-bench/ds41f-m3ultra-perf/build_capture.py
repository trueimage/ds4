#!/usr/bin/env python3
"""Build a campaign-only full-logit capture version of the public benchmark.

Run from the repository root; pass an output directory. Capture timings include
host logit reads and are correctness evidence, never throughput measurements.
"""
from pathlib import Path
import subprocess
import sys

out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=True)
source = Path("ds4_bench.c").read_text()
helper = r'''
static int campaign_snapshot(ds4_session *session) {
    const char *dir = getenv("DS4_BENCH_SNAPSHOT_DIR");
    if (!dir) return 0;
    ds4_session_snapshot snap = {0};
    char err[256] = {0}, path[4096];
    if (ds4_session_save_snapshot(session, &snap, err, sizeof(err))) return -1;
    snprintf(path, sizeof(path), "%s/%d.bin", dir, ds4_session_pos(session));
    FILE *fp = fopen(path, "wb");
    int rc = !fp || fwrite(snap.ptr, 1, snap.len, fp) != snap.len;
    if (fp && fclose(fp)) rc = 1;
    ds4_session_snapshot_free(&snap);
    return rc ? -1 : 0;
}
static int campaign_trace(ds4_engine *engine, ds4_session *session) {
    const char *path = getenv("DS4_BENCH_TRACE_LOGITS");
    if (!path) return 0;
    static FILE *fp;
    static float *logits;
    const int vocab = ds4_engine_vocab_size(engine);
    if (!fp) {
        fp = fopen(path, "wb");
        logits = malloc((size_t)vocab * sizeof(float));
    }
    if (!fp || !logits) return -1;
    memset(logits, 0xa5, (size_t)vocab * sizeof(float));
    if (ds4_session_copy_logits(session, logits, vocab) != vocab) return -1;
    const uint32_t header[] = {(uint32_t)ds4_session_pos(session), (uint32_t)vocab};
    if (fwrite(header, sizeof(header), 1, fp) != 1 ||
        fwrite(logits, sizeof(float), vocab, fp) != (size_t)vocab || fflush(fp)) return -1;
    return 0;
}
'''
source = source.replace("static double bench_now_sec(", helper + "\nstatic double bench_now_sec(", 1)
for marker in ["        const bool need_restore_after_generation =",
               "            const double token_t1 = bench_now_sec();"]:
    if source.count(marker) != 1:
        raise RuntimeError("benchmark capture insertion point changed: " + marker)
    source = source.replace(marker, '''
        if (campaign_trace(engine, session)) {
            fprintf(stderr, "campaign: full-logit capture failed\\n"); rc = 1; break;
        }
''' + marker)
marker = "        const bool need_restore_after_generation ="
source = source.replace(marker, '''
        if (campaign_snapshot(session)) {
            fprintf(stderr, "campaign: snapshot capture failed\\n"); rc = 1; break;
        }
''' + marker, 1)
capture = out / "capture.c"
capture.write_text(source)
objects = ["ds4_help.o", "ds4_gpu_args.o", "ds4.o", "ds4_image.o",
           "ds4_distributed.o", "ds4_tp.o", "ds4_ssd.o", "ds4_metal.o",
           "ds4_layer_pack.o", "ds4_engram.o"]
subprocess.run(["cc", "-O3", "-ffast-math", "-g", "-mcpu=native", "-Wall",
                "-Wextra", "-std=c99", "-I.", str(capture), *objects,
                "-o", str(out / "capture-bench"), "-lm", "-pthread",
                "-framework", "Foundation", "-framework", "Metal"], check=True)

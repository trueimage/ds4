#ifndef DS4_BENCH_VARIANT_ENV_H
#define DS4_BENCH_VARIANT_ENV_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* These switches are cached by glm53_flash_feature_enabled. Toggling them
 * after startup compares the same path twice, despite reporting two arms. */
static void bench_reject_cached_glm_env(const char *name) {
    static const char *const cached[] = {
        "DS4_METAL_DISABLE_GLM53_FLASH_TUNING",
        "DS4_METAL_DISABLE_GLM53_HC_PRODUCER_FUSE",
        "DS4_METAL_DISABLE_GLM53_KDA_GATE_PAIR",
        "DS4_METAL_DISABLE_GLM53_KDA_GATE_TRIO",
        "DS4_METAL_DISABLE_GLM53_KDA_OUT_HC_EXPAND",
        "DS4_METAL_DISABLE_GLM53_ATTN_OUT_HC_EXPAND",
        "DS4_METAL_DISABLE_GLM53_FFN_HC_EXPAND_ADD",
        "DS4_METAL_DISABLE_GLM53_SHARED_DOWN_HC_EXPAND",
        "DS4_METAL_DISABLE_GLM53_DSA_EXACT",
    };
    if (!name) return;
    for (size_t i = 0; i < sizeof(cached) / sizeof(cached[0]); ++i) {
        if (strcmp(name, cached[i])) continue;
        fprintf(stderr, "benchmark: %s is cached at engine startup; "
                "use separate processes to compare it\n", name);
        exit(2);
    }
}

#endif

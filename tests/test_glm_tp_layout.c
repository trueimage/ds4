#define DS4_TEST_HOOKS
#include "../ds4.h"

#include <stdint.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

enum {
    GGUF_Q8_0 = 8,
    GGUF_Q2_K = 10,
    GGUF_Q4_K = 12,
    GGUF_Q5_K = 13,
    GGUF_Q6_K = 14,
};

enum {
    POLICY_ALLOWED = 0,
    POLICY_UNSUPPORTED_SCOPE,
    POLICY_UNSUPPORTED_TP,
};

int ds4_test_glm_expert_layout_policy(ds4_backend backend,
                                      ds4_tp_role role,
                                      bool ssd_streaming,
                                      bool inspect_only,
                                      uint32_t gate_type,
                                      uint32_t up_type,
                                      uint32_t down_type,
                                      uint32_t *bad_layer,
                                      uint32_t *bad_type);
int ds4_test_glm_promoted_first_token_diagnostic(uint32_t gate_type,
                                                 uint32_t up_type,
                                                 uint32_t down_type);

static int failures;

static void expect_layout(const char *name,
                          ds4_backend backend,
                          ds4_tp_role role,
                          bool ssd_streaming,
                          bool inspect_only,
                          uint32_t gate_type,
                          uint32_t up_type,
                          uint32_t down_type,
                          int expected_policy,
                          uint32_t expected_bad_type) {
    uint32_t bad_layer = UINT32_MAX;
    uint32_t bad_type = UINT32_MAX;
    const int policy = ds4_test_glm_expert_layout_policy(backend,
                                                         role,
                                                         ssd_streaming,
                                                         inspect_only,
                                                         gate_type,
                                                         up_type,
                                                         down_type,
                                                         &bad_layer,
                                                         &bad_type);
    if (policy != expected_policy ||
        (expected_policy != POLICY_ALLOWED &&
         (bad_layer != 0 || bad_type != expected_bad_type))) {
        fprintf(stderr,
                "FAIL: %s policy=%d layer=%u type=%u\n",
                name,
                policy,
                bad_layer,
                bad_type);
        failures++;
    }
}

static void expect_promoted_first_token_rejection(void) {
    const pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        failures++;
        return;
    }
    if (pid == 0) {
        const int rc = ds4_test_glm_promoted_first_token_diagnostic(
                GGUF_Q6_K,
                GGUF_Q6_K,
                GGUF_Q8_0);
        _exit(rc == 1 ? 0 : 1);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) != pid ||
        !WIFEXITED(status) ||
        WEXITSTATUS(status) != 0) {
        fprintf(stderr,
                "FAIL: promoted public first-token diagnostic did not "
                "reject cleanly\n");
        failures++;
    }
}

int main(void) {
    expect_layout("CPU promoted execution", DS4_BACKEND_CPU, DS4_TP_NONE,
                  false, false,
                  GGUF_Q6_K, GGUF_Q6_K, GGUF_Q8_0,
                  POLICY_UNSUPPORTED_SCOPE, GGUF_Q6_K);
    expect_layout("CPU promoted inspection", DS4_BACKEND_CPU, DS4_TP_NONE,
                  false, true,
                  GGUF_Q6_K, GGUF_Q6_K, GGUF_Q8_0,
                  POLICY_ALLOWED, 0);
    expect_layout("resident Metal promoted layout", DS4_BACKEND_METAL,
                  DS4_TP_NONE, false, false,
                  GGUF_Q6_K, GGUF_Q6_K, GGUF_Q8_0,
                  POLICY_ALLOWED, 0);
    expect_layout("CUDA promoted layout", DS4_BACKEND_CUDA, DS4_TP_NONE,
                  false, false,
                  GGUF_Q6_K, GGUF_Q6_K, GGUF_Q8_0,
                  POLICY_UNSUPPORTED_SCOPE, GGUF_Q6_K);
    expect_layout("CUDA promoted inspection", DS4_BACKEND_CUDA, DS4_TP_NONE,
                  false, true,
                  GGUF_Q6_K, GGUF_Q6_K, GGUF_Q8_0,
                  POLICY_ALLOWED, 0);
    expect_layout("Metal streaming promoted down", DS4_BACKEND_METAL,
                  DS4_TP_NONE, true, false,
                  GGUF_Q5_K, GGUF_Q5_K, GGUF_Q8_0,
                  POLICY_UNSUPPORTED_SCOPE, GGUF_Q8_0);
    expect_layout("Metal TP promoted gate", DS4_BACKEND_METAL,
                  DS4_TP_WORKER, false, false,
                  GGUF_Q8_0, GGUF_Q8_0, GGUF_Q6_K,
                  POLICY_UNSUPPORTED_SCOPE, GGUF_Q8_0);
    expect_layout("CPU prior layout", DS4_BACKEND_CPU, DS4_TP_NONE,
                  false, false,
                  GGUF_Q5_K, GGUF_Q5_K, GGUF_Q6_K,
                  POLICY_ALLOWED, 0);
    expect_layout("CUDA prior layout", DS4_BACKEND_CUDA, DS4_TP_NONE,
                  false, false,
                  GGUF_Q5_K, GGUF_Q5_K, GGUF_Q6_K,
                  POLICY_ALLOWED, 0);
    expect_layout("Metal streaming prior layout", DS4_BACKEND_METAL,
                  DS4_TP_NONE, true, false,
                  GGUF_Q5_K, GGUF_Q5_K, GGUF_Q6_K,
                  POLICY_ALLOWED, 0);
    expect_layout("Metal streaming prior TP layout", DS4_BACKEND_METAL,
                  DS4_TP_WORKER, true, false,
                  GGUF_Q5_K, GGUF_Q5_K, GGUF_Q6_K,
                  POLICY_ALLOWED, 0);
    expect_layout("Metal resident prior TP layout", DS4_BACKEND_METAL,
                  DS4_TP_LEADER, false, false,
                  GGUF_Q4_K, GGUF_Q4_K, GGUF_Q4_K,
                  POLICY_ALLOWED, 0);
    expect_layout("Metal resident unsupported TP layout", DS4_BACKEND_METAL,
                  DS4_TP_LEADER, false, false,
                  GGUF_Q5_K, GGUF_Q5_K, GGUF_Q6_K,
                  POLICY_UNSUPPORTED_TP, GGUF_Q5_K);
    expect_layout("Metal streaming Q2 layout", DS4_BACKEND_METAL,
                  DS4_TP_LEADER, true, false,
                  GGUF_Q2_K, GGUF_Q2_K, GGUF_Q2_K,
                  POLICY_ALLOWED, 0);
    expect_promoted_first_token_rejection();
    if (failures != 0) return 1;
    puts("test_glm_tp_layout: PASS");
    return 0;
}

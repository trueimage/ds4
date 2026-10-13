#ifndef DS4_PROMPT_COPY_H
#define DS4_PROMPT_COPY_H

/* A repeated suffix suggests a continuation, never an acceptance decision.
 * Bound lookup work on long histories and prefer the newest equally long match.
 * The caller owns out[0], the already-selected target token. */
static int ds4_prompt_copy_propose(const int *ids, int history_len,
                                   int first, int *out, int room) {
    if (!ids || !out || history_len < 12 || room < 3) return 1;
    const int n = history_len + 1;
    int best = -1, best_len = 0, candidates = 0;
    const int floor = history_len - 32768 > 11 ? history_len - 32768 : 11;
    for (int end = n - 2; end >= floor && candidates < 32; end--) {
        if (ids[end] != first || ids[end - 1] != ids[n - 2] ||
            ids[end - 2] != ids[n - 3]) continue;
        candidates++;
        int len = 3, cap = end + 1 < 64 ? end + 1 : 64;
        while (len < cap && ids[end - len] == ids[n - 1 - len]) len++;
        if (len > best_len) {
            best = end; best_len = len;
            if (len == 64) break;
        }
    }
    if (best_len < 12) return 1;
    const int available = history_len - best - 1;
    int take = room - 1 < 7 ? room - 1 : 7;
    if (take > available) take = available;
    if (take < 2) return 1;
    for (int i = 0; i < take; i++) out[i + 1] = ids[best + 1 + i];
    return take + 1;
}

#endif

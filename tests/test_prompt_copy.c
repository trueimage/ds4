/* Bounded prompt lookup, independent of an inference backend. */
#include "ds4_prompt_copy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
typedef struct { int *v, len, cap; } ds4_tokens;
static void ds4_tokens_push(ds4_tokens *h, int id) {
    if (h->len == h->cap) { h->cap = h->cap ? 2 * h->cap : 32; h->v = realloc(h->v, (size_t)h->cap * sizeof(int)); CHECK(h->v); }
    h->v[h->len++] = id;
}
static void ds4_tokens_free(ds4_tokens *h) { free(h->v); memset(h, 0, sizeof(*h)); }
static int copy_propose(const ds4_tokens *h, int first, int *out, int room) {
    return ds4_prompt_copy_propose(h->v, h->len, first, out, room);
}
static void copy_selftest(void) {
    ds4_tokens h={0}; int proposed[8]={12};
    for(int i=1;i<=32;i++) ds4_tokens_push(&h,i);
    ds4_tokens_push(&h,999);
    for(int i=1;i<=11;i++) ds4_tokens_push(&h,i);
    CHECK(copy_propose(&h,12,proposed,8)==8);
    for(int i=1;i<8;i++) CHECK(proposed[i]==12+i);
    CHECK(copy_propose(&h,12,proposed,3)==3);
    CHECK(copy_propose(&h,12,proposed,2)==1);
    CHECK(copy_propose(&h,321,proposed,8)==1);
    h.len=5;CHECK(copy_propose(&h,6,proposed,8)==1);
    ds4_tokens_free(&h);
    /* The scan's oldest included endpoint, its excluded neighbor, and a
     * duplicate source with a different continuation. These affect lookup
     * opportunity; the target verifier still owns every acceptance decision. */
    int *long_ids=malloc(40000*sizeof(int));CHECK(long_ids);
    for(int outside=0;outside<2;outside++) {
        for(int i=0;i<40000;i++) long_ids[i]=10000+i;
        const int end=40000-32768-outside,first=long_ids[end];
        for(int i=0;i<11;i++) long_ids[40000-11+i]=long_ids[end-11+i];
        ds4_tokens long_history={.v=long_ids,.len=40000};proposed[0]=first;
        CHECK(copy_propose(&long_history,first,proposed,8)==(outside?1:8));
        if(!outside) for(int i=1;i<8;i++) CHECK(proposed[i]==first+i);
    }
    free(long_ids);
    for(int i=1;i<=20;i++) ds4_tokens_push(&h,i);
    ds4_tokens_push(&h,999);
    for(int i=1;i<=12;i++) ds4_tokens_push(&h,i);
    for(int i=801;i<=808;i++) ds4_tokens_push(&h,i);
    ds4_tokens_push(&h,998);
    for(int i=1;i<=11;i++) ds4_tokens_push(&h,i);
    proposed[0]=12;CHECK(copy_propose(&h,12,proposed,8)==8);
    for(int i=1;i<8;i++) CHECK(proposed[i]==800+i);
    ds4_tokens_free(&h);
    /* Generated sources check shifted matches, 12/64-token limits, context
     * room and untouched output guards. All source tokens are distinct. */
    int cases=0;
    for(int offset=0;offset<17;offset++) for(int matched=11;matched<=65;matched++) {
        for(int room=1;room<=8;room++) {
            int storage[256],out[10];
            for(int i=0;i<128;i++) storage[i]=10000+i;
            storage[128]=-1;
            for(int i=0;i<matched;i++) storage[129+i]=storage[offset+i];
            ds4_tokens sample={.v=storage,.len=129+matched};
            for(int i=0;i<10;i++) out[i]=-999;
            const int first=storage[offset+matched];out[0]=first;
            int got=copy_propose(&sample,first,out,room);
            CHECK(got==(room<3?1:room));CHECK(out[0]==first);
            for(int i=1;i<got;i++) CHECK(out[i]==storage[offset+matched+i]);
            for(int i=got;i<10;i++) CHECK(out[i]==-999);
            cases++;
        }
    }
    printf("prompt-copy host cases: PASS (basic boundaries, scan-cap/duplicate cases, %d generated cases)\n",cases);
}

int main(void) { copy_selftest(); return 0; }

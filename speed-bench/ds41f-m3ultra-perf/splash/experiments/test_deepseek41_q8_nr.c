/* Execute each row grouping against the original GPU dispatch, with guards. */
#include "ds4_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"Q8 NR FAIL line %d: %s\n",__LINE__,#x); exit(1); } } while (0)
bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }
static uint32_t rng = 37;
static uint32_t next(void) { rng ^= rng<<13; rng ^= rng>>17; rng ^= rng<<5; return rng; }
static void equal(const void *a, const void *b, size_t n, unsigned kind,
                  unsigned in, unsigned out, unsigned draw, unsigned nr) {
    if (!memcmp(a,b,n)) return;
    const uint32_t *x=a,*y=b;
    for (size_t i=0;i<n/4;i++) if(x[i]!=y[i]) {
        fprintf(stderr,"Q8 NR kind=%u %ux%u draw=%u nr=%u word=%zu old=%08x new=%08x\n",
                kind,in,out,draw,nr,i,x[i],y[i]); exit(1);
    }
}
int main(void) {
    setenv("DS4_METAL_DISABLE_METAL4","1",1);
    const size_t bytes=64u<<20;
    void *model=NULL; CHECK(posix_memalign(&model,getpagesize(),bytes)==0);
    uint8_t *w=model;
    for(size_t i=0;i+34<=bytes;i+=34) {
        uint16_t scale=0x2000u + (next()%0x800u); memcpy(w+i,&scale,2);
        for(unsigned j=0;j<32;j++) w[i+2+j]=(uint8_t)next();
    }
    CHECK(ds4_gpu_init() && ds4_gpu_set_model_map(model,bytes));
    ds4_gpu_test_set_flags(DS4_GPU_TEST_V41_FUSIONS);
    const unsigned shapes[][2]={{5120,1280},{5120,512},{5120,2304},{2304,5120},
        {8192,5120},{1280,32768},{1280,4096},{512,65544},{512,1282},{512,516}};
    const uint32_t edges[]={0x7fc00123,0x7f800000,0xff800000,0x80000000,1,0x80000001};
    unsigned cases=0;
    for(unsigned kind=0;kind<4;kind++) for(unsigned sh=0;sh<(kind<2?10u:1u);sh++) {
        const unsigned in=kind==2?5120:kind==3?4096:shapes[sh][0];
        const unsigned od=kind==2?2304:kind==3?1024:shapes[sh][1];
        if(kind==1 && od>65536) continue;
        const unsigned groups=kind==3?8:1, outputs=kind==2?3:1;
        const size_t n=(size_t)od*groups*4, xbytes=(size_t)in*groups*4;
        ds4_gpu_tensor *x=ds4_gpu_tensor_alloc_managed(xbytes);
        ds4_gpu_tensor *storage[3],*out[3]; void *ref[3];
        CHECK(x);
        for(unsigned i=0;i<outputs;i++) {
            storage[i]=ds4_gpu_tensor_alloc_managed(n+32);
            out[i]=ds4_gpu_tensor_view(storage[i],16,n);
            ref[i]=malloc(n+32); CHECK(storage[i]&&out[i]&&ref[i]);
        }
        for(unsigned draw=0;draw<10;draw++) {
            float *v=ds4_gpu_tensor_contents(x);
            for(size_t j=0;j<xbytes/4;j++) v[j]=draw==0?0.0f:
                ((int)(next()%20001)-10000)/1024.0f*(draw==8?1e18f:1.0f);
            if(draw>=2 && draw<=7) memcpy(v+61,&edges[draw-2],4);
            uint16_t saved; memcpy(&saved,w,2);
            if(draw==9) {uint16_t nan=0x7c06;memcpy(w,&nan,2);}
            for(unsigned arm=0;arm<4;arm++) {
                const unsigned nr=arm==0?2:arm==1?1:arm==2?4:8;
                char choice[16]; snprintf(choice,sizeof(choice),"all:%u",nr);
                setenv("DS4_METAL_V41_Q8_NR0",choice,1);
                if(!arm)setenv("DS4_METAL_DISABLE_V41_Q8_NR0","1",1);
                else unsetenv("DS4_METAL_DISABLE_V41_Q8_NR0");
                for(unsigned i=0;i<outputs;i++) memset(ds4_gpu_tensor_contents(storage[i]),0xa5,n+32);
                CHECK(!ds4_gpu_test_v41_fusions_take_dispatches());
                CHECK(ds4_gpu_begin_commands());
                if(kind==0) CHECK(ds4_gpu_matmul_q8_0_tensor(out[0],model,bytes,0,in,od,x,1));
                else if(kind==1) CHECK(ds4_gpu_dsv41_matmul_q8_0_bf16(out[0],model,bytes,0,in,od,x)==1);
                else if(kind==2) CHECK(ds4_gpu_dsv41_shared(out[0],out[1],out[2],x,model,bytes,0,(uint64_t)in/32*34*od,10)==1);
                else CHECK(ds4_gpu_attention_output_low_q8_tensor(out[0],model,bytes,0,in,od,groups,x));
                CHECK(ds4_gpu_end_commands());
                const unsigned bits=ds4_gpu_test_v41_fusions_take_dispatches();
                CHECK(bits==((kind==1?16u:kind==2?4u:0u) | (arm && od%nr==0?32u:0u)));
                for(unsigned i=0;i<outputs;i++) {
                    const uint8_t *p=ds4_gpu_tensor_contents(storage[i]);
                    for(unsigned j=0;j<16;j++) CHECK(p[j]==0xa5 && p[16+n+j]==0xa5);
                    if(!arm)memcpy(ref[i],p,n+32);
                    else equal(ref[i],p,n+32,kind,in,od,draw,nr);
                }
                cases++;
            }
            memcpy(w,&saved,2);
        }
        for(unsigned i=0;i<outputs;i++) {ds4_gpu_tensor_free(out[i]);ds4_gpu_tensor_free(storage[i]);free(ref[i]);}
        ds4_gpu_tensor_free(x);
    }
    unsetenv("DS4_METAL_V41_Q8_NR0");
    unsetenv("DS4_METAL_DISABLE_V41_Q8_NR0");
    ds4_gpu_cleanup();free(model);
    printf("Q8 NR PASS: %u dispatches, original versus NR0=1/4/8, all outputs and guards exact\n",cases);
    return 0;
}

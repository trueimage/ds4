/* Compare optimized dispatches with their original executable GPU paths. */
#include "ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "line %d: %s\n", __LINE__, #x); exit(1); } } while (0)
static uint32_t rng=9137;
static uint32_t random_u32(void) { rng^=rng<<13; rng^=rng>>17; rng^=rng<<5; return rng; }
static float sample(void) { return ((int)(random_u32()%20001)-10000)/1024.0f; }
static void equal(const void *a, const void *b, size_t bytes, const char *label, unsigned draw) {
    if (!memcmp(a,b,bytes)) return;
    const uint32_t *x=a,*y=b;
    for (size_t i=0;i<bytes/4;i++) if(x[i]!=y[i]) {
        fprintf(stderr,"%s draw=%u word=%zu old=%08x new=%08x\n",label,draw,i,x[i],y[i]);exit(1);
    }
}
int main(void) {
    enum {E=384,K=6,N=17,G=16};
    const size_t bytes=40960 + 2u*160u*34u*2304u;
    void *model=NULL; CHECK(posix_memalign(&model,getpagesize(),bytes)==0);
    memset(model,0,bytes);
    CHECK(ds4_gpu_init() && ds4_gpu_set_model_map(model,bytes));
    ds4_gpu_test_set_flags(DS4_GPU_TEST_V41_FUSIONS);
    ds4_gpu_tensor *logits=ds4_gpu_tensor_alloc_managed(N*E*4), *tokens=ds4_gpu_tensor_alloc_managed(N*4);
    const size_t sizes[]={N*K*4,N*K*4,N*E*4};
    ds4_gpu_tensor *out[3]; void *ref[3];
    for(unsigned i=0;i<3;i++){out[i]=ds4_gpu_tensor_alloc_managed(sizes[i]+G);ref[i]=malloc(sizes[i]);CHECK(out[i]&&ref[i]);}
    CHECK(logits&&tokens); memset(ds4_gpu_tensor_contents(tokens),0,N*4);
    const unsigned rows[]={1,2,7,17};
    const uint32_t exceptional[]={0x7fc00123,0x7f800000,0xff800000,0x80000000,1,0x80000001};
    for(unsigned draw=0;draw<240;draw++) {
        const unsigned n=rows[draw%4];
        float *x=ds4_gpu_tensor_contents(logits),*bias=model;
        for(unsigned i=0;i<N*E;i++) x[i]=draw%10==0?0:sample();
        for(unsigned i=0;i<E;i++) bias[i]=draw%10==0?0:sample()/8;
        if(draw%10==1) {for(unsigned i=0;i<N*E;i++) x[i]=sample()*12;}
        if(draw%10==2) {for(unsigned i=0;i<N*E;i++) x[i]=-120;}
        if(draw%10==3) {for(unsigned i=0;i<N*E;i++) x[i]=(float)(i%7);}
        if(draw%10>=4) {
            uint32_t u=exceptional[(draw%10)-4]; memcpy(x+(draw*13)%(n*E),&u,4);memcpy(bias+(draw*19)%E,&u,4);
        }
        for(unsigned pass=0;pass<2;pass++) {
            if(pass)unsetenv("DS4_METAL_DISABLE_V41_ROUTER_FUSION");else setenv("DS4_METAL_DISABLE_V41_ROUTER_FUSION","1",1);
            for(unsigned i=0;i<3;i++) memset(ds4_gpu_tensor_contents(out[i]),0xa5,sizes[i]+G);
            CHECK(ds4_gpu_test_v41_fusions_take_dispatches()==0);
            CHECK(ds4_gpu_router_select_batch_tensor(out[0],out[1],out[2],model,bytes,0,0,0,0,0,true,false,logits,tokens,E,K,1.5f,n));
            CHECK(ds4_gpu_synchronize());
            CHECK(ds4_gpu_test_v41_fusions_take_dispatches()==pass);
            for(unsigned i=0;i<3;i++) {
                const size_t used=n*(i==2?E:K)*4;uint8_t *p=ds4_gpu_tensor_contents(out[i]);
                for(size_t j=used;j<sizes[i]+G;j++) CHECK(p[j]==0xa5);
                if(!pass)memcpy(ref[i],p,used);else equal(ref[i],p,used,i==0?"ids":i==1?"weights":"probabilities",draw);
            }
        }
    }
    puts("router: 240 scalar/batch cases, ids/weights/probabilities/guards exact");
    ds4_gpu_set_ssd_streaming(true);
    CHECK(ds4_gpu_router_select_batch_tensor(out[0],out[1],out[2],model,bytes,0,0,0,0,0,true,false,logits,tokens,E,K,1.5f,1));
    CHECK(ds4_gpu_synchronize() && !ds4_gpu_test_v41_fusions_take_dispatches());
    ds4_gpu_set_ssd_streaming(false);

    enum {D=5120};
    ds4_gpu_tensor *res=ds4_gpu_tensor_alloc_managed(4u*D*4u), *pre=ds4_gpu_tensor_alloc_managed(24u*4u);
    ds4_gpu_tensor *collapsed=ds4_gpu_tensor_alloc_managed(D*4u), *norm=ds4_gpu_tensor_alloc_managed(D*4u);
    CHECK(res&&pre&&collapsed&&norm);
    float *ref_x=malloc(D*4u),*ref_n=malloc(D*4u); CHECK(ref_x&&ref_n);
    for(unsigned draw=0;draw<120;draw++) {
        float *r=ds4_gpu_tensor_contents(res), *w=ds4_gpu_tensor_contents(pre),*nw=model;
        for(unsigned i=0;i<4u*D;i++) r[i]=draw%10==0?0:sample();
        for(unsigned i=0;i<24;i++) w[i]=sample()/32;
        for(unsigned i=0;i<D;i++) nw[i]=1+sample()/32;
        if(draw%10==1) {w[0]=1;w[1]=-1;w[2]=1;w[3]=-1;}
        CHECK(ds4_gpu_hc_weighted_sum_split_tensor(collapsed,res,pre,D,4));
        CHECK(ds4_gpu_dsv41_quantize(collapsed,D,1,DS4_V41_BF16));
        CHECK(ds4_gpu_rms_norm_weight_tensor(norm,collapsed,model,bytes,0,D,1e-20f));
        CHECK(ds4_gpu_dsv41_quantize(norm,D,1,DS4_V41_BF16));
        CHECK(ds4_gpu_synchronize());
        memcpy(ref_x,ds4_gpu_tensor_contents(collapsed),D*4u);memcpy(ref_n,ds4_gpu_tensor_contents(norm),D*4u);
        memset(ds4_gpu_tensor_contents(collapsed),0xa5,D*4u);memset(ds4_gpu_tensor_contents(norm),0xa5,D*4u);
        CHECK(ds4_gpu_dsv41_hc_norm(collapsed,norm,res,pre,model,bytes,0,1e-20f)==1);
        CHECK(ds4_gpu_synchronize() && ds4_gpu_test_v41_fusions_take_dispatches()==2);
        equal(ref_x,ds4_gpu_tensor_contents(collapsed),D*4u,"HC collapse",draw);
        equal(ref_n,ds4_gpu_tensor_contents(norm),D*4u,"HC norm",draw);
    }
    puts("HC: 120 cases, previous-mixer collapse and normalized BF16 outputs exact");
    ds4_gpu_tensor_free(res);ds4_gpu_tensor_free(pre);ds4_gpu_tensor_free(collapsed);ds4_gpu_tensor_free(norm);free(ref_x);free(ref_n);

    enum {M=2304};
    const uint64_t wg=40960,wu=wg+160u*34u*M;
    for(unsigned a=0;a<2;a++) for(unsigned b=0;b<160u*M;b++) {
        uint8_t *q=(uint8_t *)model+(a?wu:wg)+b*34u;
        uint16_t scale=(b%7==0?0x9800:0x1800);memcpy(q,&scale,2);
        for(unsigned j=0;j<32;j++) q[2+j]=(uint8_t)(random_u32()%255-127);
    }
    ds4_gpu_tensor *sx=ds4_gpu_tensor_alloc_managed(D*4u),*sg=ds4_gpu_tensor_alloc_managed(M*4u),*su=ds4_gpu_tensor_alloc_managed(M*4u),*sm=ds4_gpu_tensor_alloc_managed(M*4u);
    CHECK(sx&&sg&&su&&sm);
    ds4_gpu_tensor *so[]={sg,su,sm};float *sr[3];for(unsigned i=0;i<3;i++){sr[i]=malloc(M*4u);CHECK(sr[i]);}
    for(unsigned draw=0;draw<120;draw++) {
        float *x=ds4_gpu_tensor_contents(sx);
        for(unsigned i=0;i<D;i++) x[i]=draw%10==0?0:sample();
        const float clamp=draw%3==0?0:draw%3==1?10:0.1f;
        CHECK(ds4_gpu_matmul_q8_0_tensor(sg,model,bytes,wg,D,M,sx,1));
        CHECK(ds4_gpu_matmul_q8_0_tensor(su,model,bytes,wu,D,M,sx,1));
        CHECK(ds4_gpu_dsv41_quantize(sg,M,1,DS4_V41_BF16));
        CHECK(ds4_gpu_dsv41_quantize(su,M,1,DS4_V41_BF16));
        CHECK(ds4_gpu_swiglu_tensor(sm,sg,su,M,clamp,1));
        CHECK(ds4_gpu_dsv41_quantize(sm,M,1,DS4_V41_BF16));
        CHECK(ds4_gpu_synchronize());
        for(unsigned i=0;i<3;i++){memcpy(sr[i],ds4_gpu_tensor_contents(so[i]),M*4u);memset(ds4_gpu_tensor_contents(so[i]),0xa5,M*4u);}
        CHECK(ds4_gpu_dsv41_shared(sg,su,sm,sx,model,bytes,wg,wu,clamp)==1);
        CHECK(ds4_gpu_synchronize() && ds4_gpu_test_v41_fusions_take_dispatches()==4);
        for(unsigned i=0;i<3;i++) equal(sr[i],ds4_gpu_tensor_contents(so[i]),M*4u,i==0?"shared gate":i==1?"shared up":"shared mid",draw);
    }
    puts("shared: 120 full-shaped Q8 cases, gate/up/mid BF16 bits exact");

    for(unsigned kind=0;kind<2;kind++) {
        if(kind) for(unsigned a=0;a<2;a++) {
            _Float16 *w=(_Float16 *)((uint8_t *)model+(a?wu:wg));
            for(unsigned i=0;i<D*512u;i++)w[i]=(_Float16)(sample()/64);
        }
        const unsigned da=kind?512:1280,db=512;
        for(unsigned draw=0;draw<60;draw++) {
            float *x=ds4_gpu_tensor_contents(sx);for(unsigned i=0;i<D;i++) x[i]=draw%10==0?0:sample();
            if(kind) {
                CHECK(ds4_gpu_matmul_f16_tensor(sg,model,bytes,wg,D,da,sx,1));
                CHECK(ds4_gpu_matmul_f16_tensor(su,model,bytes,wu,D,db,sx,1));
            } else {
                CHECK(ds4_gpu_matmul_q8_0_tensor(sg,model,bytes,wg,D,da,sx,1));
                CHECK(ds4_gpu_matmul_q8_0_tensor(su,model,bytes,wu,D,db,sx,1));
            }
            CHECK(ds4_gpu_synchronize());
            memcpy(sr[0],ds4_gpu_tensor_contents(sg),da*4u);memcpy(sr[1],ds4_gpu_tensor_contents(su),db*4u);
            memset(ds4_gpu_tensor_contents(sg),0xa5,M*4u);memset(ds4_gpu_tensor_contents(su),0xa5,M*4u);
            if(kind) CHECK(ds4_gpu_matmul_f16_pair_tensor(sg,su,model,bytes,wg,wu,D,da,sx,1));
            else CHECK(ds4_gpu_matmul_q8_0_pair_tensor(sg,su,model,bytes,wg,wu,D,da,db,sx,1));
            CHECK(ds4_gpu_synchronize());
            equal(sr[0],ds4_gpu_tensor_contents(sg),da*4u,"paired projection A",draw);
            equal(sr[1],ds4_gpu_tensor_contents(su),db*4u,"paired projection B",draw);
        }
    }
    puts("projections: 120 full-shaped Q8/F16 paired cases, raw F32 outputs exact");
    ds4_gpu_tensor_free(sx);for(unsigned i=0;i<3;i++){ds4_gpu_tensor_free(so[i]);free(sr[i]);}
    for(unsigned i=0;i<3;i++){ds4_gpu_tensor_free(out[i]);free(ref[i]);}
    ds4_gpu_tensor_free(logits);ds4_gpu_tensor_free(tokens);ds4_gpu_cleanup();free(model);return 0;
}

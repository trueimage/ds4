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
    CHECK(setenv("DS4_METAL_DISABLE_METAL4","1",1)==0);
    enum {E=384,K=6,N=17,G=16};
    const size_t bytes=48u<<20;
    void *model=NULL; CHECK(posix_memalign(&model,getpagesize(),bytes)==0);
    memset(model,0,bytes);
    CHECK(ds4_gpu_init() && ds4_gpu_set_model_map(model,bytes));
    CHECK(!ds4_gpu_dsv41_tensor_ops_available());
    ds4_gpu_test_set_flags(DS4_GPU_TEST_V41_FUSIONS);
    ds4_gpu_tensor *logits=ds4_gpu_tensor_alloc_managed(N*E*4), *tokens=ds4_gpu_tensor_alloc_managed(N*4);
    const size_t sizes[]={N*K*4,N*K*4,N*E*4};
    ds4_gpu_tensor *out[3]; void *ref[3];
    for(unsigned i=0;i<3;i++){out[i]=ds4_gpu_tensor_alloc_managed(sizes[i]+G);ref[i]=malloc(sizes[i]);CHECK(out[i]&&ref[i]);}
    CHECK(logits&&tokens); memset(ds4_gpu_tensor_contents(tokens),0,N*4);
    const unsigned rows[]={1,2,7,17};
    const uint32_t exceptional[]={0x7fc00123,0x7f800000,0xff800000,0x80000000,1,0x80000001};
    unsigned top6_accepts = 0, top6_rejects = 0;
    for(unsigned draw=0;draw<320;draw++) {
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
        // Scores with prescribed winner ties/ULP gaps and exceptional losers.
        if(draw >= 240) {
            for(unsigned i=0;i<N*E;i++) x[i]=0.0f;
            for(unsigned i=0;i<E;i++) bias[i]=(float)(E-i);
            const unsigned kind=(draw-240)%20;
            if(kind<8) bias[kind+1]=bias[kind];
            if(kind>=8 && kind<16) {
                uint32_t u; memcpy(&u,bias+kind-8,4); u--;
                memcpy(bias+kind-7,&u,4);
            }
            if(kind>=16) {
                const uint32_t u=exceptional[kind-16]; memcpy(bias+E-1,&u,4);
            }
        }
        for(unsigned pass=0;pass<3;pass++) {
            if(pass==2)unsetenv("DS4_METAL_DISABLE_V41_ROUTER_TOP6");else setenv("DS4_METAL_DISABLE_V41_ROUTER_TOP6","1",1);
            if(pass)unsetenv("DS4_METAL_DISABLE_V41_ROUTER_FUSION");else setenv("DS4_METAL_DISABLE_V41_ROUTER_FUSION","1",1);
            for(unsigned i=0;i<3;i++) memset(ds4_gpu_tensor_contents(out[i]),0xa5,sizes[i]+G);
            CHECK(ds4_gpu_test_v41_fusions_take_dispatches()==0);
            CHECK(ds4_gpu_router_select_batch_tensor(out[0],out[1],out[2],model,bytes,0,0,0,0,0,true,false,logits,tokens,E,K,1.5f,n));
            CHECK(ds4_gpu_synchronize());
            const uint32_t dispatch = ds4_gpu_test_v41_fusions_take_dispatches();
            CHECK((dispatch & ~32u)==(pass ? 1u : 0u));
            if(pass==2) {if(dispatch&32u)top6_accepts++;else top6_rejects++;}
            else CHECK(!(dispatch&32u));
            if(draw>=240 && pass==2 && (draw-240)%20<6) CHECK(!(dispatch&32u));
            for(unsigned i=0;i<3;i++) {
                const size_t used=n*(i==2?E:K)*4;uint8_t *p=ds4_gpu_tensor_contents(out[i]);
                for(size_t j=used;j<sizes[i]+G;j++) CHECK(p[j]==0xa5);
                if(!pass)memcpy(ref[i],p,used);else equal(ref[i],p,used,i==0?"ids":i==1?"weights":"probabilities",draw);
            }
        }
    }
    CHECK(top6_accepts>20 && top6_rejects>20);
    printf("router: 320 three-arm cases exact; top6 accepted=%u fallback=%u\n",top6_accepts,top6_rejects);
    unsetenv("DS4_METAL_DISABLE_V41_ROUTER_TOP6");
    ds4_gpu_set_ssd_streaming(true);
    CHECK(!ds4_gpu_dsv41_gather_reuse_admitted());
    CHECK(ds4_gpu_router_select_batch_tensor(out[0],out[1],out[2],model,bytes,0,0,0,0,0,true,false,logits,tokens,E,K,1.5f,1));
    CHECK(ds4_gpu_synchronize() && !ds4_gpu_test_v41_fusions_take_dispatches());
    ds4_gpu_set_ssd_streaming(false);
    ds4_gpu_set_quality(true);
    CHECK(!ds4_gpu_dsv41_gather_reuse_admitted());
    CHECK(ds4_gpu_router_select_batch_tensor(out[0],out[1],out[2],model,bytes,0,0,0,0,0,true,false,logits,tokens,E,K,1.5f,1));
    CHECK(ds4_gpu_synchronize() && !ds4_gpu_test_v41_fusions_take_dispatches());
    ds4_gpu_set_quality(false);
    CHECK(ds4_gpu_dsv41_gather_reuse_admitted());
    setenv("DS4_METAL_DISABLE_V41_GATHER_REUSE", "1", 1);
    CHECK(!ds4_gpu_dsv41_gather_reuse_admitted());
    unsetenv("DS4_METAL_DISABLE_V41_GATHER_REUSE");

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
        if(draw%10==2) {uint32_t u=0x7fc00123u;memcpy(r+draw,&u,4);}
        if(draw%10==3) {uint32_t u=0x7f800000u;memcpy(r+draw,&u,4);}
        if(draw%10==4) {uint32_t u=0x80000001u;memcpy(r+draw,&u,4);}
        if(draw%10==5) {uint32_t u=0x7f800000u;memcpy(nw+draw,&u,4);}

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
    ds4_gpu_set_ssd_streaming(true);
    CHECK(ds4_gpu_dsv41_hc_norm(collapsed,norm,res,pre,model,bytes,0,1e-20f)==0);
    ds4_gpu_set_ssd_streaming(false);
    ds4_gpu_set_quality(true);
    CHECK(ds4_gpu_dsv41_hc_norm(collapsed,norm,res,pre,model,bytes,0,1e-20f)==0);
    ds4_gpu_set_quality(false);
    setenv("DS4_METAL_DISABLE_V41_HC_NORM","1",1);
    CHECK(ds4_gpu_dsv41_hc_norm(collapsed,norm,res,pre,model,bytes,0,1e-20f)==0);
    unsetenv("DS4_METAL_DISABLE_V41_HC_NORM");
    CHECK(ds4_gpu_dsv41_hc_norm(collapsed,norm,res,pre,model,bytes,bytes-1,1e-20f)==-1);

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
        if(draw%10==2) {uint32_t u=0x7fc00123u;memcpy(x+draw,&u,4);}
        if(draw%10==3) {uint32_t u=0x7f800000u;memcpy(x+draw,&u,4);}
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
    ds4_gpu_set_ssd_streaming(true);
    CHECK(ds4_gpu_dsv41_shared(sg,su,sm,sx,model,bytes,wg,wu,10)==0);
    ds4_gpu_set_ssd_streaming(false);
    ds4_gpu_set_quality(true);
    CHECK(ds4_gpu_dsv41_shared(sg,su,sm,sx,model,bytes,wg,wu,10)==0);
    ds4_gpu_set_quality(false);
    setenv("DS4_METAL_DISABLE_V41_SHARED_FUSION","1",1);
    CHECK(ds4_gpu_dsv41_shared(sg,su,sm,sx,model,bytes,wg,wu,10)==0);
    unsetenv("DS4_METAL_DISABLE_V41_SHARED_FUSION");
    CHECK(!ds4_gpu_test_v41_fusions_take_dispatches());

    /* The BF16-store matvec against the matvec followed by the BF16 dispatch:
     * V4.1's real Q8_0 decode shapes, exceptional scales and activations. */
    {
        unsigned raw_nonfinite_out=0, rounding_sensitive_out=0;
        const uint32_t shapes[][2]={{5120,1536},{5120,576},{2304,5120},{8192,2048},{1536,128*64},{5120,64},{512,128},{1280,32768},{1280,4096}};
        uint8_t *wq=(uint8_t *)model;
        for(unsigned draw=0;draw<180;draw++) {
            const uint32_t in=shapes[draw%9][0], od=shapes[draw%9][1];
            const size_t wbytes=(size_t)od*(in/32u)*34u;
            CHECK(wbytes<=bytes);
            for(size_t b=0;b<wbytes;b+=34) {
                uint16_t scale=(uint16_t)(0x3000u+(random_u32()%0x1000u));
                if(draw%10==4 && b%(34*97)==0) scale=draw%30==4?0x7c00u:draw%30==14?0x7e01u:0x7c06u;
                if(draw%10==5) scale=(uint16_t)(random_u32()&1u?0x0001u:0x8001u);
                memcpy(wq+b,&scale,2);
                for(unsigned i=0;i<32;i++) wq[b+2+i]=(uint8_t)(draw%10==6?0:random_u32());
            }
            ds4_gpu_tensor *mx=ds4_gpu_tensor_alloc_managed(in*4), *mo[2];
            for(unsigned i=0;i<2;i++){mo[i]=ds4_gpu_tensor_alloc_managed(od*4+G);CHECK(mo[i]);}
            CHECK(mx);
            float *xv=ds4_gpu_tensor_contents(mx);
            for(unsigned i=0;i<in;i++) xv[i]=draw%10==7?0.0f:sample()*(draw%10==8?1e18f:1.0f);
            if(draw%10==9) for(unsigned i=0;i<6;i++) memcpy(xv+i*61,&exceptional[i],4);
            for(unsigned arm=0;arm<2;arm++) {
                memset(ds4_gpu_tensor_contents(mo[arm]),0xa5,od*4+G);
                if(!arm) setenv("DS4_METAL_DISABLE_V41_MATVEC_BF16","1",1);
                else unsetenv("DS4_METAL_DISABLE_V41_MATVEC_BF16");
                CHECK(ds4_gpu_begin_commands());
                const int rc=ds4_gpu_dsv41_matmul_q8_0_bf16(mo[arm],model,bytes,0,in,od,mx);
                CHECK(rc==(int)arm);
                if(!arm) {
                    CHECK(ds4_gpu_matmul_q8_0_tensor(mo[0],model,bytes,0,in,od,mx,1));
                    CHECK(ds4_gpu_end_commands());
                    const uint32_t *raw=ds4_gpu_tensor_contents(mo[0]);
                    for(unsigned i=0;i<od;i++) if((raw[i]&0x7f800000u)==0x7f800000u) {
                        raw_nonfinite_out++;
                        const uint32_t rounded=raw[i]+0x7fffu+((raw[i]>>16)&1u);
                        if((rounded>>16)!=(raw[i]>>16)) rounding_sensitive_out++;
                    }
                    CHECK(ds4_gpu_begin_commands());
                    CHECK(ds4_gpu_dsv41_quantize(mo[0],od,1,DS4_V41_BF16));
                }
                CHECK(ds4_gpu_end_commands());
                CHECK(ds4_gpu_test_v41_fusions_take_dispatches()==(arm?16u:0u));
            }
            equal(ds4_gpu_tensor_contents(mo[0]),ds4_gpu_tensor_contents(mo[1]),od*4+G,"matvec bf16",draw);
            ds4_gpu_tensor_free(mx);for(unsigned i=0;i<2;i++)ds4_gpu_tensor_free(mo[i]);
        }
        ds4_gpu_set_quality(true);
        CHECK(ds4_gpu_dsv41_matmul_q8_0_bf16(out[2],model,bytes,0,512,128,logits)==0);
        ds4_gpu_set_quality(false);
        ds4_gpu_set_ssd_streaming(true);
        CHECK(ds4_gpu_dsv41_matmul_q8_0_bf16(out[2],model,bytes,0,512,128,logits)==0);
        ds4_gpu_set_ssd_streaming(false);
        CHECK(!ds4_gpu_test_v41_fusions_take_dispatches());
        memset(model,0,bytes);
        CHECK(raw_nonfinite_out);
        printf("matvec: 180 Q8_0 cases, BF16-store outputs and guards equal matvec then BF16 "
               "(%u raw non-finite outputs, %u whose upper word would change under rounding)\n",
               raw_nonfinite_out,rounding_sensitive_out);
    }


    /* One weight read for every row of a small batch, against a launch per
     * row: V4.1's Q8_0 shapes, 2 to 8 rows, exceptional scales and inputs. */
    {
        const uint32_t shapes[][2]={{5120,1280},{5120,512},{5120,2304},{2304,5120},{8192,5120},{1280,32768},{15360,1024},{4096,1024},{512,65544}};
        const uint32_t counts[]={2,3,4,5,6,7,8};
        uint8_t *wq=(uint8_t *)model;
        unsigned cases=0;
        ds4_gpu_test_set_flags(DS4_GPU_TEST_V41_FUSIONS|DS4_GPU_TEST_V41_SHARED_ROWS);
        for(unsigned draw=0;draw<252;draw++) {
            const uint32_t in=shapes[draw%9][0], od=shapes[draw%9][1], rows=counts[(draw/9)%7];
            char nsg[2]={(char)('0'+(1u<<(draw/63))),0};
            setenv("DS4_METAL_Q8_MV_NSG",nsg,1);
            const size_t wbytes=(size_t)od*(in/32u)*34u;
            CHECK(wbytes<=bytes);
            for(size_t b=0;b<wbytes;b+=34) {
                uint16_t scale=(uint16_t)(0x3000u+(random_u32()%0x1000u));
                if(draw%12==4 && b%(34*97)==0) scale=draw%36==4?0x7c00u:draw%36==16?0x7e01u:0x7c06u;
                if(draw%12==5) scale=(uint16_t)(random_u32()&1u?0x0001u:0x8001u);
                memcpy(wq+b,&scale,2);
                for(unsigned i=0;i<32;i++) wq[b+2+i]=(uint8_t)(draw%12==6?0:random_u32());
            }
            ds4_gpu_tensor *mx=ds4_gpu_tensor_alloc_managed((size_t)rows*in*4), *mo[2];
            for(unsigned i=0;i<2;i++){mo[i]=ds4_gpu_tensor_alloc_managed((size_t)rows*od*4+G);CHECK(mo[i]);}
            CHECK(mx);
            float *xv=ds4_gpu_tensor_contents(mx);
            for(unsigned i=0;i<rows*in;i++) xv[i]=draw%12==7?0.0f:sample()*(draw%12==8?1e18f:1.0f);
            if(draw%12==9) for(unsigned i=0;i<6;i++) memcpy(xv+(size_t)(rows-1u)*in+i*61,&exceptional[i],4);
            for(unsigned arm=0;arm<2;arm++) {
                memset(ds4_gpu_tensor_contents(mo[arm]),0xa5,(size_t)rows*od*4+G);
                if(!arm) setenv("DS4_METAL_DISABLE_V41_SHARED_ROWS","1",1);
                else unsetenv("DS4_METAL_DISABLE_V41_SHARED_ROWS");
                CHECK(ds4_gpu_begin_commands());
                CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(mo[arm],model,bytes,0,in,od,mx,rows));
                CHECK(ds4_gpu_end_commands());
                CHECK(ds4_gpu_test_v41_shared_rows_take_dispatches()==arm);
            }
            equal(ds4_gpu_tensor_contents(mo[0]),ds4_gpu_tensor_contents(mo[1]),(size_t)rows*od*4+G,"shared rows",draw);
            /* Each row is also the one-token matvec of its own input. */
            ds4_gpu_tensor *one=ds4_gpu_tensor_alloc_managed(od*4);
            CHECK(one);
            for(uint32_t r=0;r<rows;r++) {
                ds4_gpu_tensor *xr=ds4_gpu_tensor_view(mx,(uint64_t)r*in*4,(uint64_t)in*4);
                CHECK(xr && ds4_gpu_begin_commands());
                CHECK(ds4_gpu_matmul_q8_0_tensor(one,model,bytes,0,in,od,xr,1));
                CHECK(ds4_gpu_end_commands());
                equal(ds4_gpu_tensor_contents(one),(const float *)ds4_gpu_tensor_contents(mo[1])+(size_t)r*od,od*4,"shared row vs one token",draw);
                ds4_gpu_tensor_free(xr);
            }
            ds4_gpu_tensor_free(one);
            ds4_gpu_tensor_free(mx);for(unsigned i=0;i<2;i++)ds4_gpu_tensor_free(mo[i]);
            cases++;
        }
        unsetenv("DS4_METAL_Q8_MV_NSG");
        /* Quality and streaming keep the launch per row. */
        ds4_gpu_tensor *qx=ds4_gpu_tensor_alloc_managed(2u*512u*4u), *qo=ds4_gpu_tensor_alloc_managed(2u*128u*4u);
        CHECK(qx && qo);
        memset(ds4_gpu_tensor_contents(qx),0,2u*512u*4u);
        ds4_gpu_set_quality(true);
        CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(qo,model,bytes,0,512,128,qx,2));
        ds4_gpu_set_quality(false);
        ds4_gpu_set_ssd_streaming(true);
        CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(qo,model,bytes,0,512,128,qx,2));
        ds4_gpu_set_ssd_streaming(false);
        CHECK(!ds4_gpu_test_v41_shared_rows_take_dispatches());
        ds4_gpu_tensor_free(qx); ds4_gpu_tensor_free(qo);
        ds4_gpu_test_set_flags(DS4_GPU_TEST_V41_FUSIONS);
        memset(model,0,bytes);
        printf("shared rows: %u Q8_0 cases, outputs and guards equal a launch per row and each row's one-token matvec\n",cases);
    }

    {
        enum { IN=32768, OD=8192, ROW=128*34 };
        ds4_gpu_test_set_flags(DS4_GPU_TEST_V41_SHARED_ROWS);
        for(unsigned b=0;b<8u*1024u*128u;b++) {
            uint8_t *block=(uint8_t *)model+b*34u;uint16_t d=0x1800u+(b%23u);
            memcpy(block,&d,2);for(unsigned i=2;i<34;i++)block[i]=random_u32();
        }
        for(unsigned draw=0;draw<56;draw++) {
            const unsigned rows=2+draw%7;
            ds4_gpu_tensor *x=ds4_gpu_tensor_alloc_managed((uint64_t)rows*IN*4);
            ds4_gpu_tensor *store[2],*y[2];
            CHECK(x);
            for(unsigned arm=0;arm<2;arm++) {
                store[arm]=ds4_gpu_tensor_alloc_managed((uint64_t)rows*OD*4+2*G);CHECK(store[arm]);
                memset(ds4_gpu_tensor_contents(store[arm]),0xa5,(size_t)rows*OD*4+2*G);
                y[arm]=ds4_gpu_tensor_view(store[arm],G,(uint64_t)rows*OD*4);CHECK(y[arm]);
            }
            float *v=ds4_gpu_tensor_contents(x);
            for(unsigned i=0;i<rows*IN;i++)v[i]=sample();
            if(draw>=7)for(unsigned i=0;i<6;i++)memcpy(v+(draw*131u+i*997u)%(rows*IN),&exceptional[i],4);
            CHECK(ds4_gpu_begin_commands());
            for(unsigned r=0;r<rows;r++) {
                ds4_gpu_tensor *xr=ds4_gpu_tensor_view(x,(uint64_t)r*IN*4,IN*4u);
                ds4_gpu_tensor *yr=ds4_gpu_tensor_view(y[0],(uint64_t)r*OD*4,OD*4u);
                CHECK(xr&&yr&&ds4_gpu_attention_output_low_q8_tensor(yr,model,bytes,0,4096,1024,8,xr));
                ds4_gpu_tensor_free(xr);ds4_gpu_tensor_free(yr);
            }
            CHECK(ds4_gpu_dsv41_verify_oa_rows(y[1],model,bytes,0,x,rows)==1);
            CHECK(ds4_gpu_end_commands());CHECK(ds4_gpu_test_v41_shared_rows_take_dispatches()==1);
            equal(ds4_gpu_tensor_contents(store[0]),ds4_gpu_tensor_contents(store[1]),(size_t)rows*OD*4+2*G,"verify OA",draw);
            for(unsigned arm=0;arm<2;arm++){ds4_gpu_tensor_free(y[arm]);ds4_gpu_tensor_free(store[arm]);}
            ds4_gpu_tensor_free(x);
        }
        ds4_gpu_test_set_flags(DS4_GPU_TEST_V41_FUSIONS);
        puts("verify OA: 56 grouped/strided Q8 rows cases exact against the scalar projection");
    }

    /* The in-encoder copy against the blit it replaces: every bit pattern,
     * offsets inside both tensors, untouched bytes on either side, and the
     * cases that must keep the blit. */
    {
        enum { WORDS = 20480 + 64, PAD = 32 };
        ds4_gpu_tensor *cs=ds4_gpu_tensor_alloc_managed(WORDS*4), *cd[2];
        for(unsigned i=0;i<2;i++){cd[i]=ds4_gpu_tensor_alloc_managed(WORDS*4);CHECK(cd[i]);}
        CHECK(cs);
        const uint32_t sizes_w[]={1,4,5,1023,1024,1025,5120,16384};
        const uint32_t special[]={0x7fc00123,0xffc00001,0x7f800000,0xff800000,0x80000000,1,0x80000001,0x7fffffff};
        for(unsigned draw=0;draw<160;draw++) {
            uint32_t *src=ds4_gpu_tensor_contents(cs);
            for(unsigned i=0;i<WORDS;i++) src[i]=i%11==0?special[(i+draw)%8]:random_u32();
            const uint32_t n=sizes_w[draw%8], so_w=draw%PAD, do_w=(draw*7)%PAD;
            for(unsigned arm=0;arm<2;arm++) {
                memset(ds4_gpu_tensor_contents(cd[arm]),0xa5,WORDS*4);
                if(!arm) setenv("DS4_METAL_DISABLE_V41_COMPUTE_COPY","1",1);
                else unsetenv("DS4_METAL_DISABLE_V41_COMPUTE_COPY");
                CHECK(ds4_gpu_begin_commands());
                CHECK(ds4_gpu_tensor_copy(cd[arm],do_w*4u,cs,so_w*4u,n*4u));
                CHECK(ds4_gpu_end_commands());
                CHECK(ds4_gpu_test_v41_fusions_take_dispatches()==(arm?8u:0u));
            }
            equal(ds4_gpu_tensor_contents(cd[0]),ds4_gpu_tensor_contents(cd[1]),WORDS*4,"copy",draw);
            CHECK(!memcmp((uint8_t *)ds4_gpu_tensor_contents(cd[1])+do_w*4u,src+so_w,n*4u));
        }
        /* Unaligned sizes or offsets, large and same-buffer copies keep the blit. */
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_tensor_copy(cd[1],0,cs,0,65540));
        CHECK(ds4_gpu_tensor_copy(cd[1],0,cs,0,7));
        CHECK(ds4_gpu_tensor_copy(cd[1],2,cs,0,8));
        CHECK(ds4_gpu_tensor_copy(cd[1],4096,cd[1],0,1024));
        CHECK(ds4_gpu_end_commands());
        CHECK(!ds4_gpu_test_v41_fusions_take_dispatches());
        CHECK(!memcmp((uint8_t *)ds4_gpu_tensor_contents(cd[1])+4096,ds4_gpu_tensor_contents(cd[1]),1024));
        ds4_gpu_set_quality(true);
        CHECK(ds4_gpu_begin_commands() && ds4_gpu_tensor_copy(cd[1],0,cs,0,64) && ds4_gpu_end_commands());
        ds4_gpu_set_quality(false);
        ds4_gpu_set_ssd_streaming(true);
        CHECK(ds4_gpu_begin_commands() && ds4_gpu_tensor_copy(cd[1],0,cs,0,64) && ds4_gpu_end_commands());
        ds4_gpu_set_ssd_streaming(false);
        CHECK(!ds4_gpu_test_v41_fusions_take_dispatches());
        ds4_gpu_tensor_free(cs);for(unsigned i=0;i<2;i++)ds4_gpu_tensor_free(cd[i]);
        puts("copy: 160 cases, in-encoder copy bits and guards equal the blit; fallbacks keep the blit");
    }

    ds4_gpu_tensor_free(sx);for(unsigned i=0;i<3;i++){ds4_gpu_tensor_free(so[i]);free(sr[i]);}
    for(unsigned i=0;i<3;i++){ds4_gpu_tensor_free(out[i]);free(ref[i]);}
    ds4_gpu_tensor_free(logits);ds4_gpu_tensor_free(tokens);ds4_gpu_cleanup();free(model);return 0;
}

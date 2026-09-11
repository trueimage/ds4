#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

static void require(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "GLM full FAIL: %s\n", what); exit(1); }
}
static uint32_t rng = 0x91ab8473;
static uint32_t random_u32(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng;
}
static float sample(void) { return ((int)(random_u32() % 2049) - 1024) / 128.0f; }
static void *model_alloc(size_t bytes) {
    void *p = NULL;
    require(posix_memalign(&p, getpagesize(), bytes) == 0, "model allocation");
    memset(p, 0, bytes);
    return p;
}
static ds4_gpu_tensor *tensor(size_t bytes) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    require(t != NULL, "tensor allocation"); return t;
}

static void moe_cases(void) {
    enum { D=256, H=256, O=260, E=256, K=8 };
    typedef struct { uint16_t d; uint8_t qs[64]; } iq2_block;
    const size_t row=sizeof(iq2_block), expert=H*row, down_expert=O*row;
    const size_t up=E*expert, down=2*up;
    const size_t bytes=(down+E*down_expert+16383u)&~16383u;
    unsigned char *model=model_alloc(bytes);
    for (int w=0; w<3; w++) {
        const size_t off=w==2 ? down : w*up, count=w==2 ? E*O : E*H;
        iq2_block *b=(iq2_block *)(model+off);
        for (size_t i=0; i<count; i++) {
            b[i].d=0x1400;
            for (int q=0; q<64; q++) b[i].qs[q]=random_u32();
        }
    }
    require(ds4_gpu_init() && ds4_gpu_set_model_map(model,bytes),"MoE model map");
    ds4_gpu_test_set_flags(DS4_GPU_TEST_GLM_FULL);
    const unsigned counts[]={32,33,47,64,65,129,257};
    for (int mode=0; mode<3; mode++) {
        const int quality=mode==1;
        ds4_gpu_set_quality(quality!=0);
        /* The resident escape hatch inside streaming must keep its old path. */
        ds4_gpu_set_ssd_streaming(mode==2);
        for (unsigned c=0; c<sizeof(counts)/sizeof(*counts); c++) {
            const unsigned n=counts[c], pairs=n*K;
            float *x=malloc(n*D*4), *weights=malloc(pairs*4);
            int32_t *ids=malloc(pairs*4);
            require(x && weights && ids,"MoE host inputs");
            for (unsigned i=0; i<n*D; i++) x[i]=sample()/32.0f;
            for (unsigned t=0; t<n; t++) for (unsigned k=0; k<K; k++) {
                ids[t*K+k]=k ? 1+(t*7+k*19)%239 : 0;
                weights[t*K+k]=1.0f/K;
            }
            ds4_gpu_tensor *xt=tensor(n*D*4), *it=tensor(pairs*4), *wt=tensor(pairs*4);
            require(ds4_gpu_tensor_write(xt,0,x,n*D*4) &&
                    ds4_gpu_tensor_write(it,0,ids,pairs*4) &&
                    ds4_gpu_tensor_write(wt,0,weights,pairs*4),"MoE upload");
            const size_t sizes[]={pairs*H*4,pairs*H*4,pairs*H*4,pairs*O*4,n*O*4};
            ds4_gpu_tensor *out[5]; void *ref[5], *actual[5];
            for (int i=0; i<5; i++) {
                out[i]=tensor(sizes[i]); ref[i]=malloc(sizes[i]); actual[i]=malloc(sizes[i]);
                require(ref[i] && actual[i],"MoE host outputs");
            }
            for (int arm=0; arm<3; arm++) {
                if (!arm) setenv("DS4_METAL_DISABLE_GLM_FULL_MOE_TAIL_CULL","1",1);
                else unsetenv("DS4_METAL_DISABLE_GLM_FULL_MOE_TAIL_CULL");
                for (int i=0; i<5; i++) {
                    memset(actual[i],0xa5,sizes[i]);
                    require(ds4_gpu_tensor_write(out[i],0,actual[i],sizes[i]),"MoE poison");
                }
                bool half=false;
                require(ds4_gpu_routed_moe_batch_tensor(out[4],out[0],out[1],out[2],out[3],
                    model,bytes,0,up,down,16,16,expert,row,down_expert,row,D,H,O,
                    it,wt,E,K,0.0f,xt,0,n,&half,true),"MoE dispatch");
                require(half==!quality,"MoE intermediate precision");
                for (int i=0; i<5; i++) {
                    const size_t size=i==2 && half ? sizes[i]/2 : sizes[i];
                    require(ds4_gpu_tensor_read(out[i],0,actual[i],size),"MoE read");
                    for (size_t j=0; j<size/(i==2 && half ? 2u : 4u); j++) {
                        const float v=i==2 && half ? (float)((_Float16 *)actual[i])[j] : ((float *)actual[i])[j];
                        require(isfinite(v), "finite MoE output");
                    }
                    if (!arm) memcpy(ref[i],actual[i],size);
                    else if (memcmp(ref[i],actual[i],size)) {
                        fprintf(stderr,"MoE n=%u quality=%d arm=%d output=%d\n",n,quality,arm,i);
                        require(0,"MoE exact comparison");
                    }
                }
            }
            for (int i=0; i<5; i++) { ds4_gpu_tensor_free(out[i]); free(ref[i]); free(actual[i]); }
            ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(it); ds4_gpu_tensor_free(wt);
            free(x); free(weights); free(ids);
        }
    }
    ds4_gpu_set_ssd_streaming(false);
    ds4_gpu_cleanup(); free(model);
    fprintf(stderr,"GLM full MoE: 21 exact cases with repeated poisoned outputs passed\n");
}


static void decode_cases(void) {
    const unsigned shapes[][3]={{256,256,260},{6144,2048,6144},{6144,2048,260}};
    enum { E=16, K=8 };
    typedef struct { uint16_t d; uint8_t qs[64]; } iq2_block;
    for (unsigned shape=0; shape<sizeof(shapes)/sizeof(*shapes); shape++) {
        const unsigned D=shapes[shape][0], H=shapes[shape][1], O=shapes[shape][2];
        const size_t row=(D/256)*sizeof(iq2_block), dr=(H/256)*sizeof(iq2_block);
        const size_t expert=H*row, de=O*dr, up=E*expert, down=2*up;
        const size_t bytes=(down+E*de+16383u)&~16383u;
        unsigned char *model=model_alloc(bytes);
        for (int w=0; w<3; w++) {
            const size_t off=w==2 ? down : w*up, size=w==2 ? E*de : E*expert;
            iq2_block *b=(iq2_block *)(model+off);
            for (size_t j=0; j<size/sizeof(*b); j++) {
                b[j].d=0x0800+(random_u32()%0x1000);
                for (unsigned q=0; q<64; q++) b[j].qs[q]=random_u32();
            }
        }
        require(ds4_gpu_init() && ds4_gpu_set_model_map(model,bytes),"decode map");
        ds4_gpu_test_set_flags(DS4_GPU_TEST_GLM_FULL);
        float *x=malloc(D*4), weights[K]; int32_t ids[K];
        require(x!=NULL,"decode input");
        ds4_gpu_tensor *xt=tensor(D*4), *it=tensor(K*4), *wt=tensor(K*4);
        const size_t sizes[]={K*H*4,K*H*4,K*H*4,K*O*4,O*4};
        ds4_gpu_tensor *out[5]; void *ref[5], *actual[5];
        for (int i=0; i<5; i++) {
            out[i]=tensor(sizes[i]);ref[i]=malloc(sizes[i]);actual[i]=malloc(sizes[i]);
            require(ref[i] && actual[i],"decode outputs");
        }
        for (int mode=0; mode<3; mode++) for (int trial=0; trial<4; trial++) {
            ds4_gpu_set_quality(mode==1); ds4_gpu_set_ssd_streaming(mode==2);
            for (unsigned j=0; j<D; j++) x[j]=sample()/32.0f;
            for (int j=0; j<K; j++) {
                ids[j]=trial==0 ? E-1-j : trial==1 ? 0 : random_u32()%E;
                weights[j]=trial==2 ? (j==0 ? 1.0f : 0.0f) : sample()/16.0f;
            }
            require(ds4_gpu_tensor_write(xt,0,x,D*4) && ds4_gpu_tensor_write(it,0,ids,K*4) &&
                    ds4_gpu_tensor_write(wt,0,weights,K*4),"decode upload");
            for (int arm=0; arm<3; arm++) {
                if (arm==0) setenv("DS4_METAL_DISABLE_GLM_FULL_IQ2_DOWN_SUM","1",1);
                else unsetenv("DS4_METAL_DISABLE_GLM_FULL_IQ2_DOWN_SUM");
                for (int i=0; i<5; i++) {
                    memset(actual[i],0xa5,sizes[i]);
                    require(ds4_gpu_tensor_write(out[i],0,actual[i],sizes[i]),"decode poison");
                }
                require(ds4_gpu_begin_commands(),"decode begin");
                require(ds4_gpu_routed_moe_one_tensor(out[4],out[0],out[1],out[2],out[3],
                        model,bytes,0,up,down,16,16,expert,row,de,dr,D,H,O,it,wt,E,K,
                        0.0f,xt,NULL,0,true),"decode dispatch");
                require(ds4_gpu_end_commands(),"decode end");
                for (int i=0; i<5; i++) {
                    if (i<2 && mode!=1) continue; /* gate/up are dead scratch under fused activation */
                    if (i==3 && mode==0) continue; /* fused path has no expert scratch output */
                    require(ds4_gpu_tensor_read(out[i],0,actual[i],sizes[i]),"decode read");
                    for (size_t j=0; j<sizes[i]/4; j++) require(isfinite(((float *)actual[i])[j]),"decode finite");
                    if (!arm) memcpy(ref[i],actual[i],sizes[i]);
                    else if (memcmp(ref[i],actual[i],sizes[i])) {
                        fprintf(stderr,"decode shape=%u mode=%d trial=%d arm=%d output=%d\n",shape,mode,trial,arm,i);
                        for (size_t j=0, printed=0; j<sizes[i]/4 && printed<8; j++) {
                            if (((uint32_t *)ref[i])[j]!=((uint32_t *)actual[i])[j]) {
                                fprintf(stderr," idx=%zu ref=%g actual=%g bits=%08x/%08x\n",j,
                                    ((float *)ref[i])[j],((float *)actual[i])[j],((uint32_t *)ref[i])[j],((uint32_t *)actual[i])[j]); printed++;
                            }
                        }
                        require(0,"decode exact");
                    }
                }
            }
        }
        for (int i=0; i<5; i++) {ds4_gpu_tensor_free(out[i]);free(ref[i]);free(actual[i]);}
        ds4_gpu_tensor_free(xt);ds4_gpu_tensor_free(it);ds4_gpu_tensor_free(wt);free(x);
        ds4_gpu_set_ssd_streaming(false);ds4_gpu_set_quality(false);
        ds4_gpu_cleanup();free(model);
    }
    fprintf(stderr,"GLM full decode: 36 exact cases with repeated poisoned outputs passed\n");
}

int main(void) {
    require(ds4_gpu_init(),"GPU initialization");
    ds4_gpu_test_set_flags(DS4_GPU_TEST_GLM_FULL);
    moe_cases();
    decode_cases();
    return 0;
}

/* Exact executable oracle for the V4.1 attention reduce epilogue. */
#include "ds4_gpu.h"
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define CHECK(x) do { if(!(x)){fprintf(stderr,"attention line %d: %s\n",__LINE__,#x);exit(1);} } while(0)
static uint32_t rng=0x139563;
static uint32_t random_u32(void){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static float sample(void){return ((int)(random_u32()%2049)-1024)/1024.0f;}
static void same(const void *a,const void *b,size_t bytes,unsigned draw,unsigned arm){
 if(!memcmp(a,b,bytes))return;
 const uint32_t *x=a,*y=b;
 for(size_t i=0;i<bytes/4;i++)if(x[i]!=y[i]){
  fprintf(stderr,"attention draw=%u arm=%u word=%zu old=%08x new=%08x\n",draw,arm,i,x[i],y[i]);exit(1);
 }
}
int main(void){
 CHECK(!setenv("DS4_METAL_DISABLE_METAL4","1",1));
 enum{D=512,H=64,G=16};const size_t bytes=D*H*4;
 void *model=NULL;CHECK(!posix_memalign(&model,getpagesize(),16384));memset(model,0,16384);
 CHECK(ds4_gpu_init()&&ds4_gpu_set_model_map(model,16384));
 ds4_gpu_test_set_flags(DS4_GPU_TEST_V41_FUSIONS);
 ds4_gpu_tensor *q=ds4_gpu_tensor_alloc_managed(bytes),*raw=ds4_gpu_tensor_alloc_managed(128u*D*4),*comp=ds4_gpu_tensor_alloc_managed(512u*D*4);
 ds4_gpu_tensor *storage=ds4_gpu_tensor_alloc_managed(bytes+2*G),*out=ds4_gpu_tensor_view(storage,G,bytes);
 void *ref=malloc(bytes);CHECK(q&&raw&&comp&&storage&&out&&ref);
 const uint32_t positions[]={0,1,127,128,32767,1048575};
 const uint32_t raw_counts[]={1,5,127,128},comp_counts[]={0,1,31,32,33,511,512};
 const uint32_t exceptional[]={0x7f800000,0xff800000,0x80000000,1,0x80000001,0x7f800001,0x7fc00123,0x33800000};
 unsigned reduced=0;
 for(unsigned draw=0;draw<(getenv("DS4_TEST_ATTN_SOAK")?10000u:384u);draw++){
  const uint32_t nr=draw<160?(draw<128?draw+1:128):raw_counts[draw%4];
  const uint32_t nc=draw<160?(draw<128?0:draw+1-128):comp_counts[(draw/4)%7];
  const uint32_t start=(draw*17)%128,pos=positions[draw%6];const bool compressed=draw&1;
  float *v[]={ds4_gpu_tensor_contents(q),ds4_gpu_tensor_contents(raw),ds4_gpu_tensor_contents(comp)};
  const unsigned n[]={H*D,128*D,512*D};
  for(unsigned k=0;k<3;k++)for(unsigned i=0;i<n[k];i++)v[k][i]=sample();
  float *sink=model;for(unsigned i=0;i<H;i++)sink[i]=draw%9==0?-FLT_MAX:draw%9==1?1e30f:sample();
  if(draw>=160){
   for(unsigned k=0;k<3;k++)for(unsigned j=0;j<8;j++){
    uint32_t u=exceptional[(j+draw)%8];memcpy(v[k]+((draw*193+j*397)%n[k]),&u,4);
   }
   if(draw%11==0)memcpy(sink+draw%H,&exceptional[6],4);
   if(draw%13==0)for(unsigned i=0;i<H*D;i++)v[0][i]*=1e30f;
  }
  for(unsigned arm=0;arm<2;arm++){
   if(arm==0)setenv("DS4_METAL_DISABLE_V41_ATTN_EPILOGUE","1",1);else unsetenv("DS4_METAL_DISABLE_V41_ATTN_EPILOGUE");
   memset(ds4_gpu_tensor_contents(storage),0xa5,bytes+2*G);
   ds4_gpu_dsv41_arm_attention_epilogue(pos,compressed,true);
   CHECK(ds4_gpu_begin_commands());
   CHECK(ds4_gpu_attention_decode_heads_tensor(out,model,16384,0,q,raw,nr,128,start,comp,0,nc,NULL,0,H,D));
   int used=ds4_gpu_dsv41_attention_epilogue_used();CHECK(used==(int)arm);
   if(!used){CHECK(ds4_gpu_dsv41_quantize(out,D,H,DS4_V41_BF16));CHECK(ds4_gpu_dsv41_rope(out,D,H,1,pos,compressed,true));}
   CHECK(ds4_gpu_end_commands());
   unsigned char *p=ds4_gpu_tensor_contents(storage);for(unsigned i=0;i<G;i++){CHECK(p[i]==0xa5);CHECK(p[G+bytes+i]==0xa5);}
   if(!arm)memcpy(ref,p+G,bytes);else same(ref,p+G,bytes,draw,arm);
   if(arm==1)reduced++;
  }
 }
 for(unsigned mode=0;mode<2;mode++) {
  ds4_gpu_set_quality(mode==0);ds4_gpu_set_ssd_streaming(mode==1);
  ds4_gpu_dsv41_arm_attention_epilogue(1,false,true);
  CHECK(ds4_gpu_begin_commands());
  CHECK(ds4_gpu_attention_decode_heads_tensor(out,model,16384,0,q,raw,128,128,0,comp,0,512,NULL,0,H,D));
  CHECK(!ds4_gpu_dsv41_attention_epilogue_used());
  CHECK(ds4_gpu_end_commands());
 }
 ds4_gpu_set_quality(false);ds4_gpu_set_ssd_streaming(false);
 printf("attention: %u reduce cases, every head bit and guard exact\n",reduced);
 ds4_gpu_tensor_free(out);ds4_gpu_tensor_free(storage);ds4_gpu_tensor_free(q);ds4_gpu_tensor_free(raw);ds4_gpu_tensor_free(comp);free(ref);ds4_gpu_cleanup();free(model);return 0;
}

/* Real-session checkpoint images and store latency: staged versus direct. */
#include "ds4.h"
#include "ds4_kvstore.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"KV write FAIL line %d: %s (%s)\n",__LINE__,#x,err); exit(1); } } while (0)
static double seconds(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static double resident_fraction(const char *path,size_t bytes) {
    const int fd=open(path,O_RDONLY);
    if(fd<0)return -1;
    void *map=mmap(NULL,bytes,PROT_READ,MAP_SHARED,fd,0);close(fd);
    if(map==MAP_FAILED)return -1;
    const size_t pages=(bytes+(size_t)getpagesize()-1)/(size_t)getpagesize();
    char *vec=calloc(pages,1);double resident=0;
    if(!vec || mincore(map,bytes,vec))resident=-1;
    else for(size_t i=0;i<pages;i++)resident+=(vec[i]&1)!=0;
    free(vec);munmap(map,bytes);return resident<0?resident:resident/pages;
}
int main(int argc,char **argv) {
    char err[256]={0};
    CHECK(argc==3 || argc==4);
    ds4_engine_options opt={.model_path=argv[1],.backend=DS4_BACKEND_METAL,
        .context_size=32768,.power_percent=100,.ssd_streaming=argc==4,
        .ssd_streaming_cache_bytes=16ull<<30};
    ds4_engine *engine=NULL;ds4_session *session=NULL;
    CHECK(ds4_engine_open(&engine,&opt)==0);
    FILE *fp=fopen(argv[2],"rb");CHECK(fp);
    CHECK(fseek(fp,0,SEEK_END)==0);long size=ftell(fp);CHECK(size>0);
    rewind(fp);char *prompt=malloc((size_t)size+1);CHECK(prompt);
    CHECK(fread(prompt,1,(size_t)size,fp)==(size_t)size);fclose(fp);prompt[size]=0;
    ds4_tokens corpus={0},prefix={0};ds4_tokenize_text(engine,prompt,&corpus);
    CHECK(ds4_session_create(&session,engine,32768)==0);
    const int positions[]={512,2048,8193,32703};
    for(unsigned pi=0;pi<(argc==4?2u:4u);pi++) {
        const int pos=positions[pi];CHECK(corpus.len>=pos);
        while(prefix.len<pos)ds4_tokens_push(&prefix,corpus.v[prefix.len]);
        CHECK(ds4_session_sync(session,&prefix,err,sizeof(err))==0);
        ds4_session_snapshot seed={0};CHECK(ds4_session_save_snapshot(session,&seed,err,sizeof(err))==0);
        size_t text_bytes;char *text=ds4_kvstore_render_tokens_text(engine,&prefix,&text_bytes);
        CHECK(text);
        char sha[41];ds4_kvstore_sha1_bytes_hex(text,text_bytes,sha);
        uint8_t *reference=NULL;size_t reference_bytes=0;
        for(unsigned rep=0;rep<8;rep++) {
            const bool staged=(rep%4==0 || rep%4==3);
            if(staged)setenv("DS4_KVSTORE_STAGE_PAYLOAD","1",1);
            else unsetenv("DS4_KVSTORE_STAGE_PAYLOAD");
            char dir[]="/tmp/ds4-kv-write.XXXXXX";CHECK(mkdtemp(dir));
            ds4_kvstore kc={0};ds4_kvstore_options ko=ds4_kvstore_default_options();ko.min_tokens=1;
            CHECK(ds4_kvstore_open(&kc,dir,4096,true,ko,"bench",NULL,NULL));
            const double start=seconds();
            CHECK(ds4_kvstore_store_live_prefix(&kc,engine,session,&prefix,pos,"cold",NULL,err,sizeof(err)));
            const double ms=(seconds()-start)*1000.0;
            char *path=ds4_kvstore_path_for_sha(&kc,sha);CHECK(path);
            const double residency=resident_fraction(path,52u+text_bytes+seed.len);
            fp=fopen(path,"rb");CHECK(fp);CHECK(fseek(fp,0,SEEK_END)==0);
            long n=ftell(fp);CHECK(n>48);rewind(fp);
            uint8_t *image=malloc((size_t)n);CHECK(image);
            CHECK(fread(image,1,(size_t)n,fp)==(size_t)n);fclose(fp);
            memset(image+24,0,16); /* Only wall-clock creation/touch timestamps differ. */
            if(!rep){reference=image;reference_bytes=(size_t)n;}
            else {CHECK(reference_bytes==(size_t)n && !memcmp(reference,image,(size_t)n));free(image);}
            ds4_session_snapshot actual={0};
            const double copy_start=seconds();
            CHECK(ds4_session_save_snapshot(session,&actual,err,sizeof(err))==0);
            const double copy_ms=(seconds()-copy_start)*1000.0;
            CHECK(seed.len==actual.len && !memcmp(seed.ptr,actual.ptr,seed.len));
            ds4_session_snapshot_free(&actual);
            ds4_session_invalidate(session);
            ds4_tokens loaded={0};ds4_kvstore_load_result hit={0};
            const double load_start=seconds();
            CHECK(ds4_kvstore_try_load_text(&kc,engine,session,text,&loaded,&hit,NULL,false)>0);
            const double load_ms=(seconds()-load_start)*1000.0;
            CHECK(hit.tokens==pos && loaded.len==prefix.len && !memcmp(loaded.v,prefix.v,(size_t)pos*sizeof(int)));
            CHECK(ds4_session_save_snapshot(session,&actual,err,sizeof(err))==0);
            CHECK(seed.len==actual.len && !memcmp(seed.ptr,actual.ptr,seed.len));
            ds4_session_snapshot_free(&actual);ds4_tokens_free(&loaded);ds4_kvstore_load_result_free(&hit);
            printf("{\"pos\":%d,\"staged\":%s,\"cached_io\":%s,\"load_ms\":%.6f,\"resident_fraction\":%.6f,\"rep\":%u,\"ms\":%.6f,\"copy_ms\":%.6f,\"bytes\":%ld,\"exact\":true}\n",
                pos,staged?"true":"false","true",load_ms,residency,rep,ms,copy_ms,n);fflush(stdout);
            CHECK(unlink(path)==0);free(path);ds4_kvstore_close(&kc);CHECK(rmdir(dir)==0);
        }
        free(reference);free(text);ds4_session_snapshot_free(&seed);
    }
    ds4_tokens_free(&corpus);ds4_tokens_free(&prefix);free(prompt);
    ds4_session_free(session);ds4_engine_close(engine);return 0;
}

#include "ds4_host_memory.h"
#include <assert.h>
#include <stdio.h>
int main(void) {
    ds4_host_memory_pages p = {100,20,200,10,80,10,15};
    assert(ds4_host_memory_estimate(p,16384,false)==290u*16384u);
    assert(ds4_host_memory_estimate(p,1,true)==317);
    p.compressed=20;assert(ds4_host_memory_estimate(p,1,true)==330);
    p.compressed=10;assert(ds4_host_memory_estimate(p,1,true)==290);
    p.compressor=0;assert(ds4_host_memory_estimate(p,1,true)==330);
    p.speculative=101;assert(!ds4_host_memory_estimate(p,1,true));
    p=(ds4_host_memory_pages){.free=UINT64_MAX,.external=1};
    assert(!ds4_host_memory_estimate(p,1,false));
    p.external=0;assert(!ds4_host_memory_estimate(p,2,false));
    assert(!ds4_host_memory_estimate(p,0,false));
    p=(ds4_host_memory_pages){.anonymous=UINT64_MAX,.compressor=UINT64_MAX-1,.compressed=UINT64_MAX};
    assert(ds4_host_memory_estimate(p,1,true)==1);
    p=(ds4_host_memory_pages){.anonymous=1,.compressor=2,.compressed=3};
    assert(ds4_host_memory_estimate(p,1,true)==0);
    ds4_host_pressure level=ds4_host_memory_pressure();
    assert(level>=DS4_HOST_PRESSURE_UNKNOWN && level<=DS4_HOST_PRESSURE_CRITICAL);
    printf("host memory: estimates/overflow exact; pressure=%d available=%llu\n",level,(unsigned long long)ds4_host_available_bytes());
    return 0;
}

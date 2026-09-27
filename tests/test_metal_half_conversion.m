#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
static void check(bool ok,const char *what){if(!ok){fprintf(stderr,"half conversion: %s\n",what);exit(1);}}
int main(void){@autoreleasepool{
 id<MTLDevice>d=MTLCreateSystemDefaultDevice();check(d!=nil,"device");
 NSError *err=nil;
 NSString *copy=[NSString stringWithContentsOfFile:@"metal/cpy.metal" encoding:NSUTF8StringEncoding error:&err];
 NSString *oracle=[NSString stringWithContentsOfFile:@"tests/metal_half_conversion.metal" encoding:NSUTF8StringEncoding error:&err];
 check(copy&&oracle,"shader sources");NSRange end=[copy rangeOfString:@"kernel void kernel_cpy_contig_f16_f32_4"];
 check(end.location!=NSNotFound,"production conversion kernel");
 NSString *source=[NSString stringWithFormat:@"#include <metal_stdlib>\nusing namespace metal;\n%@\n%@",[copy substringToIndex:end.location],oracle];
 id<MTLLibrary>lib=[d newLibraryWithSource:source options:[MTLCompileOptions new] error:&err];
 if(!lib)fprintf(stderr,"%s\n",err.localizedDescription.UTF8String);check(lib!=nil,"compile");
 NSString *names[]={@"half_oracle_fill",@"kernel_cpy_contig_f32_f16_4",@"half_oracle_compare"};
 id<MTLComputePipelineState>p[3];for(unsigned i=0;i<3;i++){p[i]=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:names[i]] error:&err];check(p[i]!=nil,"pipeline");}
 const uint32_t n=1u<<22;id<MTLBuffer>x=[d newBufferWithLength:(NSUInteger)n*4 options:MTLResourceStorageModeShared];
 id<MTLBuffer>y=[d newBufferWithLength:(NSUInteger)n*2 options:MTLResourceStorageModeShared];
 id<MTLBuffer>r=[d newBufferWithLength:8 options:MTLResourceStorageModeShared];check(x&&y&&r,"buffers");
 id<MTLCommandQueue>q=[d newCommandQueue];double seconds=0;
 for(uint64_t first=0;first<(1ull<<32);first+=n)@autoreleasepool{
  uint32_t base=(uint32_t)first;((uint32_t*)r.contents)[0]=0;((uint32_t*)r.contents)[1]=UINT32_MAX;
  id<MTLCommandBuffer>cb=[q commandBuffer];id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];
  [e setComputePipelineState:p[0]];[e setBytes:&base length:4 atIndex:0];[e setBuffer:x offset:0 atIndex:1];
  [e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  [e setComputePipelineState:p[1]];[e setBytes:&n length:4 atIndex:0];[e setBuffer:x offset:0 atIndex:1];[e setBuffer:y offset:0 atIndex:2];
  [e dispatchThreads:MTLSizeMake(n/4u,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  [e setComputePipelineState:p[2]];[e setBuffer:x offset:0 atIndex:0];[e setBuffer:y offset:0 atIndex:1];[e setBuffer:r offset:0 atIndex:2];[e setBytes:&base length:4 atIndex:3];
  [e dispatchThreadgroups:MTLSizeMake(n/1024u,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
  [e endEncoding];[cb commit];[cb waitUntilCompleted];check(cb.status==MTLCommandBufferStatusCompleted,"completion");seconds+=cb.GPUEndTime-cb.GPUStartTime;
  if(((uint32_t*)r.contents)[0]){fprintf(stderr,"mismatches=%u first=%08x\n",((uint32_t*)r.contents)[0],((uint32_t*)r.contents)[1]);return 1;}
 }
 printf("{\"patterns\":4294967296,\"mismatches\":0,\"gpu_seconds\":%.6f}\n",seconds);
}return 0;}

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void require(bool ok, const char *message) {
    if (!ok) { fprintf(stderr, "coherence FAIL: %s\n", message); exit(1); }
}

int main(int argc, char **argv) {
    const unsigned repeats = argc == 2 ? (unsigned)strtoul(argv[1], NULL, 10) : 100000u;
    require(repeats > 0 && repeats <= 100000u, "repeats must be 1..100000");
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "Metal device");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError *error = nil;
        NSString *source = [NSString stringWithContentsOfFile:@"tests/metal_coherence.metal"
            encoding:NSUTF8StringEncoding error:&error];
        require(source != nil, "read test shader");
        const unsigned groups[] = {1, 16, 160, 640}, producers[] = {2, 4, 6};
        const char *names[] = {"plain", "volatile", "atomic", "coherent"};
        for (unsigned mode = 0; mode < 4; ++mode) {
            MTLCompileOptions *options = [MTLCompileOptions new];
            options.preprocessorMacros = @{@"MODE": @(mode)};
            id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
            if (!library) fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
            require(library != nil, "compile handoff");
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:
                [library newFunctionWithName:@"handoff"] error:&error];
            require(pipeline != nil, "handoff pipeline");
            for (unsigned gi = 0; gi < 4; ++gi) for (unsigned si = 0; si < 3; ++si) {
                const unsigned g = groups[gi], s = producers[si];
                id<MTLBuffer> data = [device newBufferWithLength:g * s * 128u * 4u
                    options:MTLResourceStorageModeShared];
                id<MTLBuffer> counters = [device newBufferWithLength:g * 4u
                    options:MTLResourceStorageModeShared];
                id<MTLBuffer> results = [device newBufferWithLength:8
                    options:MTLResourceStorageModeShared];
                require(data && counters && results, "buffers");
                memset(data.contents, 0, data.length);
                memset(counters.contents, 0, counters.length);
                memset(results.contents, 0, results.length);
                double seconds = 0;
                for (unsigned first = 0; first < repeats; first += 100u) @autoreleasepool {
                    id<MTLCommandBuffer> cb = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pipeline];
                    [enc setBuffer:data offset:0 atIndex:0];
                    [enc setBuffer:counters offset:0 atIndex:1];
                    [enc setBuffer:results offset:0 atIndex:2];
                    [enc setBytes:&s length:4 atIndex:3];
                    for (unsigned i = first; i < repeats && i < first + 100u; ++i) {
                        [enc setBytes:&i length:4 atIndex:4];
                        [enc dispatchThreadgroups:MTLSizeMake(g * s, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    }
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    require(cb.status == MTLCommandBufferStatusCompleted, "command completion");
                    seconds += cb.GPUEndTime - cb.GPUStartTime;
                    for (unsigned i = 0; i < g; ++i)
                        require(((uint32_t *)counters.contents)[i] == 0, "counter reset");
                }
                const uint32_t *r = results.contents;
                printf("{\"mode\":\"%s\",\"groups\":%u,\"producers\":%u,\"dispatches\":%u,"
                       "\"stale\":%u,\"completions\":%u,\"gpu_seconds\":%.6f}\n",
                       names[mode], g, s, repeats, r[0], r[1], seconds);
                fflush(stdout);
                require(r[1] == repeats * g, "every consumer completed");
                if (mode >= 2) require(r[0] == 0, "coherent handoff has no stale words");
            }
        }
    }
    return 0;
}

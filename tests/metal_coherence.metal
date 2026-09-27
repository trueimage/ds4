#include <metal_stdlib>
using namespace metal;

// Same last-arriver protocol as the HC producer. There is no spinning: the
// last producer checks every published word, then resets its own counter.
#if MODE == 2
using published = atomic_uint;
#elif MODE == 3
#if __METAL_VERSION__ < 320
#error coherent device memory requires MSL 3.2
#endif
using published = coherent(device) uint;
#elif MODE == 1
using published = volatile uint;
#else
using published = uint;
#endif

static uint value(uint index, uint iteration) {
    return (index * 1664525u + iteration * 1013904223u) ^ 0xa5c379bdu;
}

kernel void handoff(device published *data [[buffer(0)]],
                    device atomic_uint *arrivals [[buffer(1)]],
                    device atomic_uint *results [[buffer(2)]],
                    constant uint &producers [[buffer(3)]],
                    constant uint &iteration [[buffer(4)]],
                    uint group [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]]) {
    const uint consumer = group / producers;
    const uint i = group * 128u + tid;
#if MODE == 2
    atomic_store_explicit(data + i, value(i, iteration), memory_order_relaxed);
#else
    data[i] = value(i, iteration);
#endif
    threadgroup_barrier(mem_flags::mem_device);
    threadgroup uint last;
    if (tid == 0u) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                            thread_scope_device);
        last = atomic_fetch_add_explicit(arrivals + consumer, 1u,
                                         memory_order_relaxed) + 1u == producers;
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                            thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!last) return;
    uint stale = 0;
    for (uint p = 0; p < producers; ++p) {
        const uint j = (consumer * producers + p) * 128u + tid;
#if MODE == 2
        const uint got = atomic_load_explicit(data + j, memory_order_relaxed);
#else
        const uint got = data[j];
#endif
        stale += got != value(j, iteration);
    }
    if (stale) atomic_fetch_add_explicit(results, stale, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0u) {
        atomic_store_explicit(arrivals + consumer, 0u, memory_order_relaxed);
        atomic_fetch_add_explicit(results + 1, 1u, memory_order_relaxed);
    }
}

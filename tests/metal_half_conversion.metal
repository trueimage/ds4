kernel void half_oracle_fill(constant uint &base, device uint *input,
                            uint gid [[thread_position_in_grid]]) {
    input[gid] = base + gid;
}
// The matrix cast and shared-memory layout are the unmodified pair kernel's
// staging conversion. Read a neighbouring thread's tile after the barrier.
kernel void half_oracle_compare(device const float *input, device const ushort *candidate,
        device atomic_uint *result, constant uint &base,
        uint group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup half sb[1024];
    const short sx_b = tid % 4, sy_b = (tid/4)/8, ly_b = (tid/4)%8;
    const short ib_b = 4*sx_b + sy_b;
    device const float *y = input + group*1024u + tid*8u;
    *(threadgroup half2x4 *)(sb + 64*ib_b + 8*ly_b) =
        (half2x4)(*((device float2x4 *)y));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint p = (tid + 1u) % 128u;
    const uint offset = 64u*(4u*(p%4u)+(p/4u)/8u)+8u*((p/4u)%8u);
    threadgroup ushort *bits = (threadgroup ushort *)sb;
    for(uint i=0;i<8u;i++) {
        const uint index=group*1024u+p*8u+i;
        if(bits[offset+i]!=candidate[index]) {
            atomic_fetch_add_explicit(result,1u,memory_order_relaxed);
            atomic_fetch_min_explicit(result+1,base+index,memory_order_relaxed);
        }
    }
}

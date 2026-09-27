#ifndef DS4_HOST_MEMORY_H
#define DS4_HOST_MEMORY_H
#include <stdbool.h>
#include <stdint.h>
#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

typedef enum {
    DS4_HOST_PRESSURE_UNKNOWN, DS4_HOST_PRESSURE_NORMAL,
    DS4_HOST_PRESSURE_WARN, DS4_HOST_PRESSURE_CRITICAL
} ds4_host_pressure;

typedef struct {
    uint64_t free, speculative, external, purgeable;
    uint64_t anonymous, compressor, compressed;
} ds4_host_memory_pages;

/* free includes speculative, already counted in external. Estimate at most
 * half the anonymous pages as reclaimable through compression; wired model
 * pages are absent from these counts. Zero denotes an unavailable estimate. */
static inline uint64_t ds4_host_memory_estimate(ds4_host_memory_pages p,
                                               uint64_t page_size,
                                               bool compression) {
    if (!page_size || p.speculative > p.free) return 0;
    uint64_t savings = 0;
    if (compression) {
        if (!p.compressor || p.compressed / 2 >= p.compressor) savings = p.anonymous / 2;
        else if (p.compressed > p.compressor)
            savings = p.anonymous - (uint64_t)((__uint128_t)p.anonymous * p.compressor / p.compressed);
    }
    if (savings > p.anonymous / 2) savings = p.anonymous / 2;
    uint64_t pages = p.free - p.speculative;
    const uint64_t reclaimable[] = {p.external, p.purgeable, savings};
    for (unsigned i = 0; i < 3; i++) {
        if (reclaimable[i] > UINT64_MAX - pages) return 0;
        pages += reclaimable[i];
    }
    return pages <= UINT64_MAX / page_size ? pages * page_size : 0;
}

static inline ds4_host_pressure ds4_host_memory_pressure(void) {
#ifdef __APPLE__
    uint32_t level = 0;
    size_t size = sizeof(level);
    if (sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &size, NULL, 0) == 0 && size == sizeof(level)) {
        if (level == 1) return DS4_HOST_PRESSURE_NORMAL;
        if (level == 2) return DS4_HOST_PRESSURE_WARN;
        if (level == 4) return DS4_HOST_PRESSURE_CRITICAL;
    }
#endif
    return DS4_HOST_PRESSURE_UNKNOWN;
}

static inline uint64_t ds4_host_available_bytes(void) {
#ifdef __APPLE__
    mach_port_t host = mach_host_self();
    vm_size_t page_size = 0;
    vm_statistics64_data_t v = {0};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    kern_return_t rc = host_page_size(host, &page_size);
    if (rc == KERN_SUCCESS) rc = host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&v, &count);
    mach_port_deallocate(mach_task_self(), host);
    if (rc != KERN_SUCCESS) return 0;
    const ds4_host_memory_pages p = {v.free_count, v.speculative_count,
        v.external_page_count, v.purgeable_count, v.internal_page_count,
        v.compressor_page_count, v.total_uncompressed_pages_in_compressor};
    const ds4_host_pressure pressure = ds4_host_memory_pressure();
    return ds4_host_memory_estimate(p, page_size,
        pressure == DS4_HOST_PRESSURE_NORMAL || pressure == DS4_HOST_PRESSURE_WARN);
#else
    return 0;
#endif
}
#endif

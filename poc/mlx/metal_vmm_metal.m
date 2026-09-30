/*
 * metal_vmm_metal.m - Metal buffer integration for the VMM arena
 *
 * This Objective-C file provides the Metal-specific functionality:
 *   - Creating a Metal buffer that wraps the arena's VA range
 *   - The buffer is created ONCE and stays valid as pages are mapped/unmapped
 *
 * Build: clang -c metal_vmm_metal.m -framework Metal -framework Foundation
 *
 * This file is only compiled on macOS. On Linux, the Metal buffer
 * functionality is stubbed out.
 */

#ifdef __APPLE__

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_vmm.h"

/* Access arena internals - in production this would be a friend or accessor */
struct mvmm_arena {
    void   *base;
    size_t  total_size;
    size_t  page_size;
    int     num_pages;
    int    *mapped;
    int    *page_fds;
    id<MTLBuffer> metal_buffer;
};

id<MTLBuffer> mvmm_arena_metal_buffer(mvmm_arena_t *arena,
                                       id<MTLDevice> device) {
    if (arena->metal_buffer)
        return arena->metal_buffer;

    /*
     * Create a Metal buffer that wraps the arena's virtual address range.
     *
     * Key properties:
     *   - MTLResourceStorageModeShared: both CPU and GPU access (UMA)
     *   - MTLResourceHazardTrackingModeUntracked: we manage synchronization
     *   - bytesNoCopy: Metal directly references our VA range, no memcpy
     *   - deallocator: nil — we manage the memory ourselves
     *
     * Requirements:
     *   - arena->base must be page-aligned (guaranteed by mmap)
     *   - arena->total_size must be a multiple of page size (guaranteed)
     *
     * Behavior with page remapping:
     *   - When we mmap(MAP_FIXED) a new page into the VA range, Metal
     *     sees the change because Apple Silicon uses shared page tables
     *   - No Metal API call is needed after remapping
     *   - BUT: must not remap while a command buffer using this buffer is
     *     in flight (undefined behavior / GPU page fault)
     */
    arena->metal_buffer = [device
        newBufferWithBytesNoCopy:arena->base
                         length:arena->total_size
                        options:MTLResourceStorageModeShared |
                                MTLResourceHazardTrackingModeUntracked
                    deallocator:nil];

    if (!arena->metal_buffer) {
        NSLog(@"[mvmm] Failed to create Metal buffer (size=%zu, base=%p)",
              arena->total_size, arena->base);
        return nil;
    }

    NSLog(@"[mvmm] Created Metal buffer: %zu bytes at %p",
          arena->total_size, arena->base);
    return arena->metal_buffer;
}

#endif /* __APPLE__ */

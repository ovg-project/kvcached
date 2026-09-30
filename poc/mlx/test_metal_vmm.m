/*
 * test_metal_vmm.m - Test Metal buffer + VMM page remapping on macOS
 *
 * This test validates the critical assumption:
 *   After mmap(MAP_FIXED) remaps pages within a Metal buffer's VA range,
 *   GPU compute via Metal sees the remapped data.
 *
 * Build:
 *   clang -O2 -framework Metal -framework Foundation \
 *         metal_vmm.c metal_vmm_metal.m test_metal_vmm.m \
 *         -o test_metal_vmm
 *
 * Run:  ./test_metal_vmm
 *
 * This ONLY runs on macOS with Apple Silicon (or Intel with UMA).
 */

#ifdef __APPLE__

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "metal_vmm.h"

/* A trivial Metal compute shader that reads from the buffer and writes
   the sum of a page's first 4 uint32 elements to an output buffer. */
static NSString *shader_source = @R"(
#include <metal_stdlib>
using namespace metal;

kernel void read_page(
    device const uint *input [[buffer(0)]],
    device uint *output      [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    // Each thread reads 4 consecutive uint32s from input and sums them
    uint base = tid * 4;
    output[tid] = input[base] + input[base+1] + input[base+2] + input[base+3];
}
)";

static void test_gpu_sees_mapped_data(void) {
    printf("\n=== Test: GPU reads mapped page data via Metal ===\n");

    /* Setup Metal */
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    assert(device && "No Metal device found");
    printf("  Metal device: %s\n", [[device name] UTF8String]);

    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:shader_source
                                                  options:nil
                                                    error:&error];
    assert(library && "Failed to compile shader");
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:
            [library newFunctionWithName:@"read_page"] error:&error];
    assert(pipeline && "Failed to create pipeline");

    /* Create arena and Metal buffer */
    size_t page_size = 16 * 1024;  /* 16 KB */
    size_t total = page_size * 4;  /* 4 pages */
    mvmm_arena_t *arena = mvmm_arena_create(total, page_size);
    assert(arena);

    id<MTLBuffer> buffer = mvmm_arena_metal_buffer(arena, device);
    assert(buffer && "Failed to create Metal buffer from arena");
    printf("  Arena: %zu bytes, Metal buffer at %p\n", total, [buffer contents]);

    /* Verify Metal buffer pointer matches arena base */
    assert([buffer contents] == mvmm_arena_base(arena));
    printf("  [OK] Metal buffer pointer == arena base\n");

    /* Output buffer for GPU results */
    size_t num_threads = page_size / (4 * sizeof(uint32_t));
    id<MTLBuffer> output = [device newBufferWithLength:num_threads * sizeof(uint32_t)
                                               options:MTLResourceStorageModeShared];

    /* --- Phase 1: Read unmapped (zero) page via GPU --- */
    {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buffer offset:0 atIndex:0];
        [enc setBuffer:output offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(num_threads, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(MIN(num_threads, 256), 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        uint32_t *results = (uint32_t *)[output contents];
        int all_zero = 1;
        for (size_t i = 0; i < num_threads; i++) {
            if (results[i] != 0) { all_zero = 0; break; }
        }
        assert(all_zero && "Unmapped page should read as zeros on GPU");
        printf("  [OK] GPU reads zeros from unmapped page\n");
    }

    /* --- Phase 2: Map a page with known data, GPU should see it --- */
    mvmm_page_t *page = mvmm_page_create(page_size);
    assert(page);

    /* Fill page with known pattern: every uint32 = 0x42 */
    uint32_t *pdata = (uint32_t *)mvmm_page_data(page);
    for (size_t i = 0; i < page_size / sizeof(uint32_t); i++)
        pdata[i] = 0x42;

    assert(mvmm_map(arena, 0, page) == MVMM_OK);
    printf("  Mapped page at slot 0 (pattern: 0x42)\n");

    {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buffer offset:0 atIndex:0];
        [enc setBuffer:output offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(num_threads, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(MIN(num_threads, 256), 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        uint32_t *results = (uint32_t *)[output contents];
        uint32_t expected = 0x42 * 4;  /* sum of 4 x 0x42 */
        int correct = 1;
        for (size_t i = 0; i < num_threads; i++) {
            if (results[i] != expected) {
                printf("  FAIL: results[%zu] = %u, expected %u\n",
                       i, results[i], expected);
                correct = 0;
                break;
            }
        }
        assert(correct && "GPU should see mapped page data");
        printf("  [OK] GPU reads 0x42 pattern from mapped page\n");
    }

    /* --- Phase 3: Unmap and verify GPU sees zeros again --- */
    assert(mvmm_unmap(arena, 0) == MVMM_OK);
    printf("  Unmapped page at slot 0\n");

    {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buffer offset:0 atIndex:0];
        [enc setBuffer:output offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(num_threads, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(MIN(num_threads, 256), 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        uint32_t *results = (uint32_t *)[output contents];
        int all_zero = 1;
        for (size_t i = 0; i < num_threads; i++) {
            if (results[i] != 0) { all_zero = 0; break; }
        }
        assert(all_zero && "Unmapped page should read as zeros again");
        printf("  [OK] GPU reads zeros after unmap\n");
    }

    /* --- Phase 4: Remap with different data --- */
    for (size_t i = 0; i < page_size / sizeof(uint32_t); i++)
        pdata[i] = 0xFF;

    assert(mvmm_map(arena, 0, page) == MVMM_OK);
    printf("  Remapped page at slot 0 (new pattern: 0xFF)\n");

    {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buffer offset:0 atIndex:0];
        [enc setBuffer:output offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(num_threads, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(MIN(num_threads, 256), 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        uint32_t *results = (uint32_t *)[output contents];
        uint32_t expected = 0xFF * 4;
        int correct = 1;
        for (size_t i = 0; i < num_threads; i++) {
            if (results[i] != expected) { correct = 0; break; }
        }
        assert(correct && "GPU should see remapped data");
        printf("  [OK] GPU reads 0xFF pattern from remapped page\n");
    }

    /* Cleanup */
    mvmm_unmap(arena, 0);
    mvmm_page_destroy(page);
    mvmm_arena_destroy(arena);
    printf("  PASSED\n");
}

static void test_gpu_multi_page(void) {
    printf("\n=== Test: GPU reads from multiple mapped pages ===\n");

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:shader_source
                                                  options:nil error:&error];
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:
            [library newFunctionWithName:@"read_page"] error:&error];

    size_t page_size = 16 * 1024;
    size_t total = page_size * 8;
    mvmm_arena_t *arena = mvmm_arena_create(total, page_size);
    id<MTLBuffer> buffer = mvmm_arena_metal_buffer(arena, device);

    /* Map pages 0,2,4,6 with distinct tags; leave 1,3,5,7 unmapped */
    mvmm_page_t *pages[4];
    for (int i = 0; i < 4; i++) {
        pages[i] = mvmm_page_create(page_size);
        uint32_t *pd = (uint32_t *)mvmm_page_data(pages[i]);
        for (size_t j = 0; j < page_size / sizeof(uint32_t); j++)
            pd[j] = (i + 1) * 100;  /* 100, 200, 300, 400 */
        mvmm_map(arena, i * 2, pages[i]);
    }

    /* Read each page via GPU and verify */
    size_t threads_per_page = page_size / (4 * sizeof(uint32_t));
    id<MTLBuffer> output = [device newBufferWithLength:threads_per_page * sizeof(uint32_t)
                                               options:MTLResourceStorageModeShared];

    for (int slot = 0; slot < 8; slot++) {
        size_t byte_offset = (size_t)slot * page_size;

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buffer offset:byte_offset atIndex:0];
        [enc setBuffer:output offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(threads_per_page, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(MIN(threads_per_page, 256), 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        uint32_t *results = (uint32_t *)[output contents];
        uint32_t expected;
        if (slot % 2 == 0) {
            expected = ((slot / 2) + 1) * 100 * 4;  /* sum of 4 identical values */
        } else {
            expected = 0;  /* unmapped */
        }

        int ok = 1;
        for (size_t j = 0; j < threads_per_page; j++) {
            if (results[j] != expected) { ok = 0; break; }
        }
        printf("  Slot %d: expected=%u, %s\n", slot, expected, ok ? "OK" : "FAIL");
        assert(ok);
    }

    for (int i = 0; i < 4; i++) {
        mvmm_unmap(arena, i * 2);
        mvmm_page_destroy(pages[i]);
    }
    mvmm_arena_destroy(arena);
    printf("  PASSED\n");
}

int main(void) {
    @autoreleasepool {
        printf("KVCached MLX PoC — Metal VMM Test\n");
        printf("==================================\n");

        test_gpu_sees_mapped_data();
        test_gpu_multi_page();

        printf("\n========================================\n");
        printf("ALL METAL TESTS PASSED\n");
        printf("========================================\n");
        printf("\nConclusion: mmap(MAP_FIXED) page remapping IS visible\n");
        printf("to the GPU through a Metal newBufferWithBytesNoCopy buffer.\n");
        printf("This validates the core KVCached-on-MLX approach.\n");
    }
    return 0;
}

#else /* not __APPLE__ */

#include <stdio.h>
int main(void) {
    printf("This test requires macOS with Metal support.\n");
    printf("Run test_vmm_remap instead for the platform-independent VMM test.\n");
    return 1;
}

#endif

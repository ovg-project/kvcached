/*
 * metal_vmm.h - Virtual Memory Manager for Metal/MLX on Apple Silicon
 *
 * This is the macOS equivalent of KVCached's CUDA VMM layer.
 * It uses mmap + Metal's newBufferWithBytesNoCopy to provide:
 *   - Virtual address space reservation
 *   - On-demand physical page mapping/unmapping
 *   - Metal buffer backed by the managed VA range
 *   - Zero-copy access from MLX arrays
 *
 * Apple Silicon UMA guarantees:
 *   - CPU and GPU share page tables → remapped pages visible to GPU immediately
 *   - Hardware cache coherence → no manual flushes needed
 *   - StorageModeShared → both CPU and GPU can read/write
 *
 * IMPORTANT: Page remapping MUST NOT occur while a Metal command buffer
 * referencing this buffer is in flight. Synchronize via:
 *   - MLX: call mx.eval() / mx.synchronize() before remapping
 *   - Metal: waitUntilCompleted on command buffer before remapping
 */

#ifndef KVCACHED_METAL_VMM_H
#define KVCACHED_METAL_VMM_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- Page size ---------- */

/*
 * ARM64 macOS hardware page size is 16 KB.
 * We use this as the minimum granularity. For better amortization with
 * large KV caches, users can configure larger "compound pages" (multiples
 * of 16 KB) just like KVCached uses multiples of 2 MB on CUDA.
 *
 * Recommended compound page sizes:
 *   - 16 KB: fine-grained, good for small models / tight memory
 *   - 64 KB: balanced
 *   - 256 KB: good for large models (70B+)
 *   - 2 MB: matches CUDA granularity (for feature parity)
 */
#define METAL_VMM_HW_PAGE_SIZE   (16 * 1024)
#define METAL_VMM_DEFAULT_PAGE   (64 * 1024)

/* ---------- Error codes ---------- */

typedef enum {
    MVMM_OK = 0,
    MVMM_ERR_MMAP_FAILED,
    MVMM_ERR_METAL_BUFFER_FAILED,
    MVMM_ERR_ALREADY_MAPPED,
    MVMM_ERR_NOT_MAPPED,
    MVMM_ERR_ALIGNMENT,
    MVMM_ERR_OUT_OF_RANGE,
    MVMM_ERR_GPU_IN_FLIGHT,
} mvmm_error_t;

/* ---------- Opaque types ---------- */

typedef struct mvmm_arena  mvmm_arena_t;   /* virtual address space manager  */
typedef struct mvmm_page   mvmm_page_t;    /* a "physical page" (memfd/shm)  */

/* ---------- Arena API ---------- */

/*
 * Create an arena: reserve `total_size` bytes of virtual address space.
 * `page_size` must be a multiple of METAL_VMM_HW_PAGE_SIZE.
 * Returns NULL on failure.
 */
mvmm_arena_t *mvmm_arena_create(size_t total_size, size_t page_size);

/*
 * Destroy the arena and release all resources.
 * All mapped pages are automatically unmapped.
 */
void mvmm_arena_destroy(mvmm_arena_t *arena);

/* Get the base pointer of the reserved VA range. */
void *mvmm_arena_base(const mvmm_arena_t *arena);

/* Get total size. */
size_t mvmm_arena_size(const mvmm_arena_t *arena);

/* Get page size. */
size_t mvmm_arena_page_size(const mvmm_arena_t *arena);

/* Number of page slots. */
int mvmm_arena_num_pages(const mvmm_arena_t *arena);

#ifdef __OBJC__
/*
 * Get a Metal buffer wrapping the entire arena VA range.
 * Created lazily on first call. The buffer pointer equals arena_base().
 * The buffer remains valid across map/unmap operations.
 */
id<MTLBuffer> mvmm_arena_metal_buffer(mvmm_arena_t *arena, id<MTLDevice> device);
#endif

/* ---------- Page API ---------- */

/*
 * Allocate a new physical page backed by shared memory.
 * The page can be mapped into multiple arenas simultaneously.
 */
mvmm_page_t *mvmm_page_create(size_t page_size);

/* Destroy a physical page. Must be unmapped from all arenas first. */
void mvmm_page_destroy(mvmm_page_t *page);

/* Get the page's own memory pointer (for direct writes). */
void *mvmm_page_data(const mvmm_page_t *page);

/*
 * Map a physical page into the arena at `page_index`.
 *
 * This uses mmap(MAP_FIXED | MAP_SHARED) to atomically place the physical
 * page's backing memory at the corresponding slot in the arena's VA range.
 *
 * After this call, reads through arena_base() + page_index * page_size
 * will return the physical page's data.
 *
 * IMPORTANT: Ensure no Metal command buffer is currently accessing this
 * region. Call mx.synchronize() or [commandBuffer waitUntilCompleted] first.
 */
mvmm_error_t mvmm_map(mvmm_arena_t *arena, int page_index,
                       mvmm_page_t *page);

/*
 * Unmap a page slot, replacing it with zeros.
 * Same synchronization requirements as mvmm_map.
 */
mvmm_error_t mvmm_unmap(mvmm_arena_t *arena, int page_index);

/* Check if a slot is mapped. */
bool mvmm_is_mapped(const mvmm_arena_t *arena, int page_index);

/* ---------- Batch operations ---------- */

/*
 * Map/unmap multiple pages atomically (all-or-nothing).
 * `indices` and `pages` arrays must have `count` elements.
 * For unmap_batch, `pages` is ignored (can be NULL).
 */
mvmm_error_t mvmm_map_batch(mvmm_arena_t *arena, const int *indices,
                             mvmm_page_t *const *pages, int count);

mvmm_error_t mvmm_unmap_batch(mvmm_arena_t *arena, const int *indices,
                               int count);

#ifdef __cplusplus
}
#endif

#endif /* KVCACHED_METAL_VMM_H */

/*
 * test_vmm_remap.c - Validate mmap-based virtual memory remapping for KV cache
 *
 * This tests the core VMM concept that KVCached-on-MLX would use:
 *   1. Reserve a large virtual address range (like cuMemAddressReserve)
 *   2. Create independent "physical pages" via anonymous mmap
 *   3. Remap pages into the reserved range using MAP_FIXED (like cuMemMap)
 *   4. Unmap pages by mapping a zero page over them (like cuMemUnmap)
 *   5. Verify data integrity through all operations
 *
 * This runs on Linux. On macOS ARM64, the same approach works but:
 *   - Page size is 16KB (vs 4KB on x86 Linux)
 *   - The remapped memory can be wrapped as a Metal buffer
 *   - Metal on Apple Silicon UMA sees remapped pages immediately
 *
 * Build: gcc -O2 -o test_vmm_remap test_vmm_remap.c
 * Run:   ./test_vmm_remap
 */

#define _GNU_SOURCE
#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

/* ---------- configuration ---------- */

/* On ARM64 macOS the HW page is 16 KB; on x86-64 Linux it is 4 KB.
   We use 16 KB throughout so the same logic transfers to Apple Silicon. */
#define PAGE_SIZE      (16 * 1024)

/* KV cache "block" parameters (mimicking a small LLM). */
#define NUM_HEADS      8
#define HEAD_DIM       128
#define BLOCK_SIZE     16     /* tokens per block */
#define DTYPE_BYTES    2      /* float16 */

/* Derived. */
#define CELL_BYTES     (NUM_HEADS * HEAD_DIM * DTYPE_BYTES)            /* 2 KB   */
#define BLOCK_BYTES    ((size_t)BLOCK_SIZE * CELL_BYTES)               /* 32 KB  */
#define BLOCKS_PER_PAGE (PAGE_SIZE / BLOCK_BYTES > 0 ? PAGE_SIZE / BLOCK_BYTES : 1)

/* Total virtual space to reserve (simulates large KV capacity). */
#define NUM_VIRTUAL_PAGES 64
#define VIRTUAL_SIZE   ((size_t)NUM_VIRTUAL_PAGES * PAGE_SIZE)

/* ---------- helpers ---------- */

static long page_size_os;

static void die(const char *msg) {
    perror(msg);
    exit(1);
}

/* Allocate an independent anonymous "physical page". */
static void *alloc_physical_page(void) {
    void *p = mmap(NULL, PAGE_SIZE, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) die("alloc_physical_page mmap");
    return p;
}

/* Free a physical page. */
static void free_physical_page(void *p) {
    if (munmap(p, PAGE_SIZE) != 0) die("free_physical_page munmap");
}

/* Fill a page with a recognisable pattern (page_id written to every uint64). */
static void fill_page(void *p, uint64_t tag) {
    uint64_t *arr = (uint64_t *)p;
    size_t n = PAGE_SIZE / sizeof(uint64_t);
    for (size_t i = 0; i < n; i++) arr[i] = tag;
}

/* Verify every uint64 in a page equals the expected tag. */
static int verify_page(const void *p, uint64_t expected_tag) {
    const uint64_t *arr = (const uint64_t *)p;
    size_t n = PAGE_SIZE / sizeof(uint64_t);
    for (size_t i = 0; i < n; i++) {
        if (arr[i] != expected_tag) return 0;
    }
    return 1;
}

/* ---------- virtual address space manager ---------- */

typedef struct {
    void   *base;          /* start of reserved VA range                     */
    size_t  total_size;    /* VIRTUAL_SIZE                                   */
    void   *zero_page;     /* shared read-only zero page                     */
    void  **page_table;    /* physical page ptr per slot (NULL = zero-mapped) */
    int     num_pages;
} va_manager_t;

/* Reserve a contiguous virtual range filled with zeros (zero page). */
static va_manager_t *va_create(void) {
    va_manager_t *vm = calloc(1, sizeof(*vm));
    vm->num_pages  = NUM_VIRTUAL_PAGES;
    vm->total_size = VIRTUAL_SIZE;

    /* 1. Allocate the shared zero page. */
    vm->zero_page = alloc_physical_page();
    memset(vm->zero_page, 0, PAGE_SIZE);

    /* 2. Reserve the full VA range with an anonymous mapping (all zeros). */
    vm->base = mmap(NULL, vm->total_size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (vm->base == MAP_FAILED) die("va_create: reserve mmap");

    /* Page table tracks which slots have real physical pages. */
    vm->page_table = calloc(vm->num_pages, sizeof(void *));

    printf("[va_create] reserved %zu bytes at %p  (%d virtual pages of %d bytes)\n",
           vm->total_size, vm->base, vm->num_pages, PAGE_SIZE);
    return vm;
}

/*
 * Map a physical page into slot `page_idx`.
 * Equivalent to cuMemMap on CUDA, or what MetalPage::map would do on macOS.
 *
 * Uses mmap(MAP_FIXED) to atomically replace the zero-filled anonymous page
 * at that slot with a file-backed or anonymous mapping of the physical page.
 *
 * On macOS this is how we'd remap pages under a newBufferWithBytesNoCopy
 * Metal buffer.  The GPU sees the change immediately because Apple Silicon
 * UMA uses shared page tables.
 */
static int va_map(va_manager_t *vm, int page_idx, void *phys_page) {
    assert(page_idx >= 0 && page_idx < vm->num_pages);
    assert(vm->page_table[page_idx] == NULL && "double map");

    void *target = (char *)vm->base + (size_t)page_idx * PAGE_SIZE;

    /*
     * Strategy: We use mremap (Linux) or mmap(MAP_FIXED) to place the
     * physical page's content at `target`.
     *
     * On Linux, the cleanest approach is:
     *   - Create a memfd for the physical page
     *   - mmap(MAP_FIXED) the memfd at target
     * But for this PoC we just copy the data after a MAP_FIXED anonymous map,
     * since the real macOS implementation uses mach_vm_remap or mmap(MAP_FIXED)
     * on shared memory objects.
     *
     * The semantics we're validating:
     *   - MAP_FIXED atomically replaces the old mapping
     *   - Reading from `target` yields the physical page's data
     *   - The pointer `vm->base` (which would be the Metal buffer pointer)
     *     stays valid throughout
     */

    /* Overwrite the slot with a fresh anonymous page, then memcpy. */
    void *p = mmap(target, PAGE_SIZE, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    if (p == MAP_FAILED) { perror("va_map mmap"); return -1; }
    memcpy(p, phys_page, PAGE_SIZE);

    vm->page_table[page_idx] = phys_page;
    return 0;
}

/*
 * Unmap a page slot — replace with zero page.
 * Equivalent to cuMemUnmap + mapping the shared zero page.
 */
static int va_unmap(va_manager_t *vm, int page_idx) {
    assert(page_idx >= 0 && page_idx < vm->num_pages);
    assert(vm->page_table[page_idx] != NULL && "not mapped");

    void *target = (char *)vm->base + (size_t)page_idx * PAGE_SIZE;

    /* Overwrite with a fresh zero-filled anonymous mapping. */
    void *p = mmap(target, PAGE_SIZE, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    if (p == MAP_FAILED) { perror("va_unmap mmap"); return -1; }

    vm->page_table[page_idx] = NULL;
    return 0;
}

static void va_destroy(va_manager_t *vm) {
    munmap(vm->base, vm->total_size);
    free_physical_page(vm->zero_page);
    free(vm->page_table);
    free(vm);
}

/* ---------- shared-memory variant (true page sharing without copy) ---------- */

/*
 * This variant uses a memfd to back each physical page.
 * mmap(MAP_FIXED | MAP_SHARED, fd) remaps without any memcpy — the virtual
 * slot and the "physical page" reference the same underlying memory object.
 *
 * This is closer to what macOS mach_vm_remap achieves and is what we'd
 * actually use on macOS (via shm_open or vm_allocate + vm_remap).
 */

#ifdef __linux__
#include <sys/syscall.h>

static int create_memfd(const char *name, size_t size) {
    int fd = syscall(SYS_memfd_create, name, 0);
    if (fd < 0) die("memfd_create");
    if (ftruncate(fd, size) != 0) die("ftruncate memfd");
    return fd;
}

typedef struct {
    int    fd;       /* memfd backing this page       */
    void  *origin;   /* original mmap (for writes)    */
} shared_page_t;

static shared_page_t *shared_page_create(uint64_t tag) {
    shared_page_t *sp = malloc(sizeof(*sp));
    char name[64];
    snprintf(name, sizeof(name), "kvcached_page_%lu", tag);
    sp->fd = create_memfd(name, PAGE_SIZE);
    sp->origin = mmap(NULL, PAGE_SIZE, PROT_READ | PROT_WRITE,
                      MAP_SHARED, sp->fd, 0);
    if (sp->origin == MAP_FAILED) die("shared_page mmap");
    fill_page(sp->origin, tag);
    return sp;
}

static void shared_page_destroy(shared_page_t *sp) {
    munmap(sp->origin, PAGE_SIZE);
    close(sp->fd);
    free(sp);
}

/*
 * Map a shared page into a virtual slot — TRUE zero-copy remap.
 * Both `sp->origin` and `target` now reference the same physical frames.
 */
static int va_map_shared(va_manager_t *vm, int page_idx, shared_page_t *sp) {
    void *target = (char *)vm->base + (size_t)page_idx * PAGE_SIZE;
    void *p = mmap(target, PAGE_SIZE, PROT_READ | PROT_WRITE,
                   MAP_SHARED | MAP_FIXED, sp->fd, 0);
    if (p == MAP_FAILED) { perror("va_map_shared mmap"); return -1; }
    vm->page_table[page_idx] = sp->origin;
    return 0;
}
#endif /* __linux__ */

/* ---------- tests ---------- */

static void test_basic_map_unmap(void) {
    printf("\n=== Test 1: Basic map / read / unmap ===\n");

    va_manager_t *vm = va_create();

    /* Initially every slot is zero. */
    uint64_t *slot0 = (uint64_t *)vm->base;
    assert(slot0[0] == 0 && "slot 0 should be zero initially");
    printf("  [OK] initial read: zeros\n");

    /* Create a physical page with tag=0xCAFE and map it to slot 0. */
    void *phys = alloc_physical_page();
    fill_page(phys, 0xCAFE);
    assert(va_map(vm, 0, phys) == 0);
    assert(verify_page(vm->base, 0xCAFE));
    printf("  [OK] mapped page with tag 0xCAFE, verified through VA\n");

    /* Unmap slot 0 — should be zeros again. */
    assert(va_unmap(vm, 0) == 0);
    assert(slot0[0] == 0 && "slot 0 should be zero after unmap");
    printf("  [OK] unmapped, slot is zeros\n");

    free_physical_page(phys);
    va_destroy(vm);
    printf("  PASSED\n");
}

static void test_multi_page_map(void) {
    printf("\n=== Test 2: Multiple pages mapped simultaneously ===\n");

    va_manager_t *vm = va_create();
    void *pages[8];

    /* Map 8 pages with distinct tags. */
    for (int i = 0; i < 8; i++) {
        pages[i] = alloc_physical_page();
        fill_page(pages[i], 0xA000 + i);
        assert(va_map(vm, i, pages[i]) == 0);
    }

    /* Verify each slot independently. */
    for (int i = 0; i < 8; i++) {
        void *slot = (char *)vm->base + (size_t)i * PAGE_SIZE;
        assert(verify_page(slot, 0xA000 + i));
    }
    printf("  [OK] 8 pages mapped and verified\n");

    /* Unmap odd pages. */
    for (int i = 1; i < 8; i += 2) {
        assert(va_unmap(vm, i) == 0);
    }

    /* Even pages still correct, odd pages are zeros. */
    for (int i = 0; i < 8; i++) {
        void *slot = (char *)vm->base + (size_t)i * PAGE_SIZE;
        if (i % 2 == 0) {
            assert(verify_page(slot, 0xA000 + i));
        } else {
            assert(verify_page(slot, 0));  /* zero page */
        }
    }
    printf("  [OK] selective unmap correct\n");

    for (int i = 0; i < 8; i++) {
        if (i % 2 == 0) va_unmap(vm, i);
        free_physical_page(pages[i]);
    }
    va_destroy(vm);
    printf("  PASSED\n");
}

static void test_remap_different_page(void) {
    printf("\n=== Test 3: Remap slot with a different physical page ===\n");

    va_manager_t *vm = va_create();

    void *phys_a = alloc_physical_page();
    void *phys_b = alloc_physical_page();
    fill_page(phys_a, 0xAAAA);
    fill_page(phys_b, 0xBBBB);

    /* Map page A at slot 0. */
    va_map(vm, 0, phys_a);
    assert(verify_page(vm->base, 0xAAAA));
    printf("  [OK] slot 0 = page A (0xAAAA)\n");

    /* Unmap, then map page B at slot 0. */
    va_unmap(vm, 0);
    va_map(vm, 0, phys_b);
    assert(verify_page(vm->base, 0xBBBB));
    printf("  [OK] slot 0 remapped to page B (0xBBBB)\n");

    va_unmap(vm, 0);
    free_physical_page(phys_a);
    free_physical_page(phys_b);
    va_destroy(vm);
    printf("  PASSED\n");
}

#ifdef __linux__
static void test_shared_zero_copy(void) {
    printf("\n=== Test 4: Shared-memory zero-copy remap (memfd) ===\n");

    va_manager_t *vm = va_create();

    shared_page_t *sp = shared_page_create(0xDEAD);
    assert(va_map_shared(vm, 0, sp) == 0);
    assert(verify_page(vm->base, 0xDEAD));
    printf("  [OK] shared page mapped, data visible through VA\n");

    /* Write through the original pointer — should be visible through VA too. */
    fill_page(sp->origin, 0xBEEF);
    assert(verify_page(vm->base, 0xBEEF));
    printf("  [OK] write through origin visible through VA (true zero-copy)\n");

    /* Write through the VA — should be visible through origin. */
    fill_page(vm->base, 0xF00D);
    assert(verify_page(sp->origin, 0xF00D));
    printf("  [OK] write through VA visible through origin (bidirectional)\n");

    va_unmap(vm, 0);
    shared_page_destroy(sp);
    va_destroy(vm);
    printf("  PASSED\n");
}
#endif

static void test_pointer_stability(void) {
    printf("\n=== Test 5: Base pointer stability across map/unmap ===\n");
    printf("  (Critical: Metal buffer wraps base pointer once; must stay valid)\n");

    va_manager_t *vm = va_create();
    void *original_base = vm->base;

    void *pages[4];
    for (int i = 0; i < 4; i++) {
        pages[i] = alloc_physical_page();
        fill_page(pages[i], 0x1000 + i);
    }

    /* Rapid map/unmap cycles. */
    for (int cycle = 0; cycle < 100; cycle++) {
        int idx = cycle % 4;
        va_map(vm, idx, pages[idx]);
        assert(vm->base == original_base && "base pointer must not move");
        va_unmap(vm, idx);
        assert(vm->base == original_base && "base pointer must not move");
    }
    printf("  [OK] 100 map/unmap cycles, base pointer stable at %p\n", original_base);

    for (int i = 0; i < 4; i++) free_physical_page(pages[i]);
    va_destroy(vm);
    printf("  PASSED\n");
}

static void test_perf_map_unmap(void) {
    printf("\n=== Test 6: Map/unmap throughput ===\n");

    va_manager_t *vm = va_create();
    void *phys = alloc_physical_page();
    fill_page(phys, 0x42);

    int num_ops = 10000;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < num_ops; i++) {
        va_map(vm, 0, phys);
        va_unmap(vm, 0);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    double ops_per_sec = (2.0 * num_ops) / elapsed;  /* map + unmap = 2 ops */

    printf("  %d map+unmap pairs in %.3f s  (%.0f ops/sec, %.1f us/op)\n",
           num_ops, elapsed, ops_per_sec, 1e6 / ops_per_sec);

    free_physical_page(phys);
    va_destroy(vm);
    printf("  PASSED\n");
}

static void test_kv_cache_simulation(void) {
    printf("\n=== Test 7: Simulated KV cache elastic allocation ===\n");
    printf("  Config: %d heads x %d dim x %d tokens/block x %d-byte dtype\n",
           NUM_HEADS, HEAD_DIM, BLOCK_SIZE, DTYPE_BYTES);
    printf("  Cell: %d bytes, Block: %d bytes, Blocks/page: %zu\n",
           CELL_BYTES, (int)BLOCK_BYTES, BLOCKS_PER_PAGE > 0 ? BLOCKS_PER_PAGE : 1);

    va_manager_t *vm = va_create();

    /* Simulate: 3 requests arrive needing 4, 2, and 6 pages respectively. */
    int req_pages[] = {4, 2, 6};
    int req_start[] = {0, 4, 6};  /* starting page index */
    void *phys_pool[64] = {0};

    for (int r = 0; r < 3; r++) {
        printf("  Request %d: allocating %d pages at offset %d\n",
               r, req_pages[r], req_start[r]);
        for (int p = 0; p < req_pages[r]; p++) {
            int idx = req_start[r] + p;
            phys_pool[idx] = alloc_physical_page();
            fill_page(phys_pool[idx], 0xC000 + idx);
            va_map(vm, idx, phys_pool[idx]);
        }
    }

    /* Verify all mapped pages. */
    for (int i = 0; i < 12; i++) {
        void *slot = (char *)vm->base + (size_t)i * PAGE_SIZE;
        assert(verify_page(slot, 0xC000 + i));
    }
    printf("  [OK] all 12 pages mapped and verified\n");

    /* Request 1 finishes — free its 4 pages. */
    printf("  Request 0 done: freeing 4 pages\n");
    for (int p = 0; p < 4; p++) {
        va_unmap(vm, p);
        free_physical_page(phys_pool[p]);
        phys_pool[p] = NULL;
    }

    /* Verify: slots 0-3 are zeros, slots 4-11 still have data. */
    for (int i = 0; i < 12; i++) {
        void *slot = (char *)vm->base + (size_t)i * PAGE_SIZE;
        if (i < 4) {
            assert(verify_page(slot, 0));
        } else {
            assert(verify_page(slot, 0xC000 + i));
        }
    }
    printf("  [OK] freed pages are zeros, active pages intact\n");

    /* New request reuses freed slots. */
    printf("  Request 3: reusing freed slots 0-3\n");
    for (int p = 0; p < 4; p++) {
        phys_pool[p] = alloc_physical_page();
        fill_page(phys_pool[p], 0xD000 + p);
        va_map(vm, p, phys_pool[p]);
    }

    for (int i = 0; i < 4; i++) {
        void *slot = (char *)vm->base + (size_t)i * PAGE_SIZE;
        assert(verify_page(slot, 0xD000 + i));
    }
    printf("  [OK] reused slots have new data\n");

    /* Cleanup. */
    for (int i = 0; i < 12; i++) {
        if (phys_pool[i]) {
            va_unmap(vm, i);
            free_physical_page(phys_pool[i]);
        }
    }
    va_destroy(vm);
    printf("  PASSED\n");
}

/* ---------- main ---------- */

int main(void) {
    page_size_os = sysconf(_SC_PAGESIZE);
    printf("KVCached MLX PoC — mmap-based Virtual Memory Manager\n");
    printf("OS page size: %ld bytes, KV page size: %d bytes\n",
           page_size_os, PAGE_SIZE);
    printf("Virtual capacity: %d pages x %d bytes = %zu bytes\n",
           NUM_VIRTUAL_PAGES, PAGE_SIZE, (size_t)VIRTUAL_SIZE);

    test_basic_map_unmap();
    test_multi_page_map();
    test_remap_different_page();
#ifdef __linux__
    test_shared_zero_copy();
#endif
    test_pointer_stability();
    test_perf_map_unmap();
    test_kv_cache_simulation();

    printf("\n========================================\n");
    printf("ALL TESTS PASSED\n");
    printf("========================================\n");
    printf("\nConclusion: mmap(MAP_FIXED) based VMM works correctly.\n");
    printf("The base pointer stays stable across map/unmap cycles,\n");
    printf("which means a Metal buffer wrapping this pointer (via\n");
    printf("newBufferWithBytesNoCopy) would remain valid.\n");
    return 0;
}

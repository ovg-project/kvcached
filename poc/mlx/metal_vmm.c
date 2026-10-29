/*
 * metal_vmm.c - Virtual Memory Manager implementation (portable C core)
 *
 * This file contains the platform-independent mmap logic.
 * The Metal/Objective-C parts live in metal_vmm_metal.m.
 *
 * Build on Linux:  gcc -c metal_vmm.c
 * Build on macOS:  clang -c metal_vmm.c
 */

#include "metal_vmm.h"

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

/* macOS uses shm_open; Linux has memfd_create */
#ifdef __linux__
#include <sys/syscall.h>
static int portable_memfd(const char *name, size_t size) {
    int fd = syscall(SYS_memfd_create, name, 0);
    if (fd < 0) return -1;
    if (ftruncate(fd, size) != 0) { close(fd); return -1; }
    return fd;
}
#else /* macOS */
static int portable_memfd(const char *name, size_t size) {
    /* Use shm_open for cross-process shared memory on macOS */
    char shm_name[256];
    snprintf(shm_name, sizeof(shm_name), "/kvcached_%s_%d", name, getpid());
    int fd = shm_open(shm_name, O_RDWR | O_CREAT | O_EXCL, 0600);
    if (fd < 0) return -1;
    shm_unlink(shm_name);  /* unlink immediately; fd keeps it alive */
    if (ftruncate(fd, size) != 0) { close(fd); return -1; }
    return fd;
}
#endif

/* ---------- arena internals ---------- */

struct mvmm_arena {
    void   *base;           /* reserved VA range start                  */
    size_t  total_size;     /* total bytes reserved                     */
    size_t  page_size;      /* bytes per page slot                      */
    int     num_pages;      /* total_size / page_size                   */
    int    *mapped;         /* 1 if slot has a real page, 0 otherwise   */
    int    *page_fds;       /* fd of the shared memory object per slot  */
#ifdef __OBJC__
    id      metal_buffer;   /* MTLBuffer (created lazily)               */
#else
    void   *metal_buffer;   /* placeholder on non-Apple                 */
#endif
};

struct mvmm_page {
    int     fd;             /* shared memory fd backing this page       */
    void   *data;           /* mmap'd pointer for direct access         */
    size_t  size;           /* page size in bytes                       */
};

/* ---------- arena implementation ---------- */

mvmm_arena_t *mvmm_arena_create(size_t total_size, size_t page_size) {
    if (page_size == 0)
        page_size = METAL_VMM_DEFAULT_PAGE;

    /* Validate alignment */
    if (page_size % METAL_VMM_HW_PAGE_SIZE != 0) {
        fprintf(stderr, "mvmm: page_size %zu not aligned to %d\n",
                page_size, METAL_VMM_HW_PAGE_SIZE);
        return NULL;
    }
    if (total_size % page_size != 0) {
        /* Round up */
        total_size = ((total_size + page_size - 1) / page_size) * page_size;
    }

    mvmm_arena_t *a = calloc(1, sizeof(*a));
    if (!a) return NULL;

    a->page_size  = page_size;
    a->total_size = total_size;
    a->num_pages  = total_size / page_size;

    /* Reserve VA range — all zeros initially. */
    a->base = mmap(NULL, total_size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (a->base == MAP_FAILED) {
        perror("mvmm_arena_create: mmap reserve");
        free(a);
        return NULL;
    }

    a->mapped   = calloc(a->num_pages, sizeof(int));
    a->page_fds = calloc(a->num_pages, sizeof(int));
    for (int i = 0; i < a->num_pages; i++)
        a->page_fds[i] = -1;

    return a;
}

void mvmm_arena_destroy(mvmm_arena_t *arena) {
    if (!arena) return;

    /* Unmap all mapped pages first */
    for (int i = 0; i < arena->num_pages; i++) {
        if (arena->mapped[i]) {
            mvmm_unmap(arena, i);
        }
    }

    if (arena->base)
        munmap(arena->base, arena->total_size);

    free(arena->mapped);
    free(arena->page_fds);
    free(arena);
}

void *mvmm_arena_base(const mvmm_arena_t *arena) {
    return arena->base;
}

size_t mvmm_arena_size(const mvmm_arena_t *arena) {
    return arena->total_size;
}

size_t mvmm_arena_page_size(const mvmm_arena_t *arena) {
    return arena->page_size;
}

int mvmm_arena_num_pages(const mvmm_arena_t *arena) {
    return arena->num_pages;
}

bool mvmm_is_mapped(const mvmm_arena_t *arena, int page_index) {
    if (page_index < 0 || page_index >= arena->num_pages) return false;
    return arena->mapped[page_index] != 0;
}

/* ---------- page implementation ---------- */

mvmm_page_t *mvmm_page_create(size_t page_size) {
    if (page_size == 0) page_size = METAL_VMM_DEFAULT_PAGE;

    mvmm_page_t *p = calloc(1, sizeof(*p));
    if (!p) return NULL;

    p->size = page_size;

    /* Create shared-memory fd so multiple mmap calls share the same memory */
    static int page_counter = 0;
    char name[64];
    snprintf(name, sizeof(name), "page_%d", __sync_fetch_and_add(&page_counter, 1));
    p->fd = portable_memfd(name, page_size);
    if (p->fd < 0) {
        perror("mvmm_page_create: memfd");
        free(p);
        return NULL;
    }

    /* Map for direct access */
    p->data = mmap(NULL, page_size, PROT_READ | PROT_WRITE,
                   MAP_SHARED, p->fd, 0);
    if (p->data == MAP_FAILED) {
        perror("mvmm_page_create: mmap");
        close(p->fd);
        free(p);
        return NULL;
    }

    /* Zero-initialize */
    memset(p->data, 0, page_size);
    return p;
}

void mvmm_page_destroy(mvmm_page_t *page) {
    if (!page) return;
    if (page->data && page->data != MAP_FAILED)
        munmap(page->data, page->size);
    if (page->fd >= 0)
        close(page->fd);
    free(page);
}

void *mvmm_page_data(const mvmm_page_t *page) {
    return page->data;
}

/* ---------- map / unmap ---------- */

mvmm_error_t mvmm_map(mvmm_arena_t *arena, int page_index,
                       mvmm_page_t *page) {
    if (page_index < 0 || page_index >= arena->num_pages)
        return MVMM_ERR_OUT_OF_RANGE;
    if (arena->mapped[page_index])
        return MVMM_ERR_ALREADY_MAPPED;
    if (page->size != arena->page_size)
        return MVMM_ERR_ALIGNMENT;

    void *target = (char *)arena->base + (size_t)page_index * arena->page_size;

    /*
     * MAP_FIXED atomically replaces the existing anonymous mapping at
     * `target` with a shared mapping of the page's fd.
     *
     * On macOS Apple Silicon (UMA):
     *   - The GPU accesses memory through the same page tables
     *   - This remap is immediately visible to Metal command buffers
     *     (as long as no command buffer is currently in flight)
     *   - Hardware cache coherence handles the rest
     *
     * On Linux (for testing):
     *   - Same mmap(MAP_FIXED) semantics, just no GPU involved
     */
    void *p = mmap(target, arena->page_size, PROT_READ | PROT_WRITE,
                   MAP_SHARED | MAP_FIXED, page->fd, 0);
    if (p == MAP_FAILED)
        return MVMM_ERR_MMAP_FAILED;

    arena->mapped[page_index] = 1;
    arena->page_fds[page_index] = page->fd;
    return MVMM_OK;
}

mvmm_error_t mvmm_unmap(mvmm_arena_t *arena, int page_index) {
    if (page_index < 0 || page_index >= arena->num_pages)
        return MVMM_ERR_OUT_OF_RANGE;
    if (!arena->mapped[page_index])
        return MVMM_ERR_NOT_MAPPED;

    void *target = (char *)arena->base + (size_t)page_index * arena->page_size;

    /* Replace with a fresh zero-filled anonymous mapping. */
    void *p = mmap(target, arena->page_size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    if (p == MAP_FAILED)
        return MVMM_ERR_MMAP_FAILED;

    arena->mapped[page_index] = 0;
    arena->page_fds[page_index] = -1;
    return MVMM_OK;
}

/* ---------- batch operations ---------- */

mvmm_error_t mvmm_map_batch(mvmm_arena_t *arena, const int *indices,
                             mvmm_page_t *const *pages, int count) {
    /* Validate all first (all-or-nothing semantics). */
    for (int i = 0; i < count; i++) {
        if (indices[i] < 0 || indices[i] >= arena->num_pages)
            return MVMM_ERR_OUT_OF_RANGE;
        if (arena->mapped[indices[i]])
            return MVMM_ERR_ALREADY_MAPPED;
    }

    /* Map all. On failure, roll back. */
    for (int i = 0; i < count; i++) {
        mvmm_error_t err = mvmm_map(arena, indices[i], pages[i]);
        if (err != MVMM_OK) {
            /* Rollback previously mapped in this batch */
            for (int j = 0; j < i; j++)
                mvmm_unmap(arena, indices[j]);
            return err;
        }
    }
    return MVMM_OK;
}

mvmm_error_t mvmm_unmap_batch(mvmm_arena_t *arena, const int *indices,
                               int count) {
    for (int i = 0; i < count; i++) {
        mvmm_error_t err = mvmm_unmap(arena, indices[i]);
        if (err != MVMM_OK) return err;
    }
    return MVMM_OK;
}

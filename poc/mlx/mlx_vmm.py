"""
mlx_vmm.py - Python interface for KVCached's MLX Virtual Memory Manager

This module provides the Python-side VMM for MLX, analogous to:
  - kvcached.vmm_ops (CUDA VMM)
  - kvcached.kv_cache_manager (block management)

Architecture:
  mmap-based VA reservation
    → shm_open / memfd_create for physical pages
    → mmap(MAP_FIXED) for page mapping
    → Metal buffer wrapping via ctypes/objc bridge
    → MLX array view of the Metal buffer (via numpy + memoryview)

Usage:
    arena = MLXArena(total_size=64*1024*1024, page_size=64*1024)
    page = arena.alloc_page()
    arena.map(page_index=0, page=page)
    # ... arena.as_mlx_array(shape, dtype) returns an mx.array view
    arena.unmap(page_index=0)
"""

import ctypes
import mmap
import os
import struct
import sys
from typing import List, Optional, Tuple

import numpy as np

# Conditionally import MLX (only available on macOS Apple Silicon)
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

# Page sizes
HW_PAGE_SIZE = 16 * 1024  # ARM64 macOS = 16KB
DEFAULT_PAGE_SIZE = 64 * 1024  # 64KB default compound page


class PhysicalPage:
    """
    A 'physical page' backed by shared memory (shm or memfd).

    On macOS: uses shm_open for POSIX shared memory
    On Linux: uses memfd_create (for testing)

    The page can be mapped into an arena via mmap(MAP_FIXED, fd).
    This is zero-copy: the arena VA and this page's VA point to the
    same physical memory frames.
    """

    _counter = 0

    def __init__(self, size: int, tag: int = 0):
        self.size = size
        self.tag = tag
        self._fd: Optional[int] = None
        self._mm: Optional[mmap.mmap] = None

        PhysicalPage._counter += 1
        name = f"kvcached_mlx_{os.getpid()}_{PhysicalPage._counter}"

        if sys.platform == "linux":
            # memfd_create (Linux 3.17+)
            import ctypes.util

            libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
            MFD_CLOEXEC = 0x0001
            self._fd = libc.memfd_create(name.encode(), MFD_CLOEXEC)
            if self._fd < 0:
                raise OSError(f"memfd_create failed: {os.strerror(ctypes.get_errno())}")
            os.ftruncate(self._fd, size)
        elif sys.platform == "darwin":
            # POSIX shared memory (macOS)
            import posix_ipc

            shm = posix_ipc.SharedMemory(
                f"/{name}", posix_ipc.O_CREX, size=size
            )
            self._fd = os.dup(shm.fd)
            shm.close_fd()
            shm.unlink()  # Unlink immediately; fd keeps it alive
        else:
            raise RuntimeError(f"Unsupported platform: {sys.platform}")

        # mmap for direct access
        self._mm = mmap.mmap(self._fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

        # Zero-initialize
        self._mm[:] = b"\x00" * size

    @property
    def fd(self) -> int:
        """File descriptor for the shared memory backing."""
        return self._fd

    def as_numpy(self, dtype=np.float16) -> np.ndarray:
        """View this page's memory as a numpy array (zero-copy)."""
        return np.frombuffer(self._mm, dtype=dtype)

    def fill_pattern(self, value: int):
        """Fill with a uint64 pattern (for testing)."""
        arr = np.frombuffer(self._mm, dtype=np.uint64)
        arr[:] = value

    def close(self):
        if self._mm:
            try:
                self._mm.close()
            except BufferError:
                pass
            self._mm = None
        if self._fd is not None and self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class MLXArena:
    """
    Virtual address space arena for KV cache tensors.

    Reserves a contiguous VA range via mmap, then maps/unmaps physical
    pages into it using mmap(MAP_FIXED). On macOS, the VA range can be
    wrapped as a Metal buffer via newBufferWithBytesNoCopy, and MLX
    arrays can view it.

    This is the MLX equivalent of FTensorAllocator + FTensor.
    """

    def __init__(self, total_size: int, page_size: int = DEFAULT_PAGE_SIZE):
        assert page_size >= HW_PAGE_SIZE
        assert page_size % HW_PAGE_SIZE == 0
        # Round up total_size
        if total_size % page_size != 0:
            total_size = ((total_size + page_size - 1) // page_size) * page_size

        self.total_size = total_size
        self.page_size = page_size
        self.num_pages = total_size // page_size

        # Reserve VA range with anonymous mmap (all zeros)
        self._fd = -1
        self._mm = mmap.mmap(
            -1,
            total_size,
            mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
            mmap.PROT_READ | mmap.PROT_WRITE,
        )

        # Get the base address via ctypes
        self._base_addr = ctypes.addressof(ctypes.c_char.from_buffer(self._mm))

        # Track mapped pages
        self._mapped: List[Optional[PhysicalPage]] = [None] * self.num_pages

        # Metal buffer handle (macOS only, created lazily)
        self._metal_buffer = None

    @property
    def base_ptr(self) -> int:
        """Base virtual address of the arena."""
        return self._base_addr

    def map(self, page_index: int, page: PhysicalPage):
        """
        Map a physical page into the arena at the given slot.

        Uses mmap(MAP_FIXED | MAP_SHARED) with the page's fd to
        atomically replace the anonymous mapping at that slot.

        IMPORTANT on macOS: call mx.synchronize() before this if any
        MLX computation is using this arena's data.
        """
        assert 0 <= page_index < self.num_pages, f"page_index {page_index} out of range"
        assert self._mapped[page_index] is None, f"slot {page_index} already mapped"
        assert page.size == self.page_size

        offset = page_index * self.page_size
        target_addr = self._base_addr + offset

        # On Linux, use ctypes to call mmap directly (Python's mmap module
        # doesn't support MAP_FIXED well). On macOS, same approach.
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [
            ctypes.c_void_p,  # addr
            ctypes.c_size_t,  # length
            ctypes.c_int,     # prot
            ctypes.c_int,     # flags
            ctypes.c_int,     # fd
            ctypes.c_long,    # offset
        ]

        MAP_FIXED = 0x10
        MAP_SHARED = 0x01
        PROT_RW = 0x01 | 0x02  # PROT_READ | PROT_WRITE

        result = libc.mmap(
            ctypes.c_void_p(target_addr),
            ctypes.c_size_t(self.page_size),
            ctypes.c_int(PROT_RW),
            ctypes.c_int(MAP_SHARED | MAP_FIXED),
            ctypes.c_int(page.fd),
            ctypes.c_long(0),
        )

        if result == ctypes.c_void_p(-1).value:
            errno = ctypes.get_errno()
            raise OSError(f"mmap MAP_FIXED failed: {os.strerror(errno)}")

        self._mapped[page_index] = page

    def unmap(self, page_index: int):
        """
        Unmap a page slot, replacing with zeros (anonymous mapping).

        Same synchronization requirements as map().
        """
        assert 0 <= page_index < self.num_pages
        assert self._mapped[page_index] is not None, f"slot {page_index} not mapped"

        offset = page_index * self.page_size
        target_addr = self._base_addr + offset

        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_long,
        ]

        MAP_FIXED = 0x10
        MAP_PRIVATE = 0x02
        MAP_ANONYMOUS = 0x20
        PROT_RW = 0x01 | 0x02

        result = libc.mmap(
            ctypes.c_void_p(target_addr),
            ctypes.c_size_t(self.page_size),
            ctypes.c_int(PROT_RW),
            ctypes.c_int(MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED),
            ctypes.c_int(-1),
            ctypes.c_long(0),
        )

        if result == ctypes.c_void_p(-1).value:
            errno = ctypes.get_errno()
            raise OSError(f"mmap unmap failed: {os.strerror(errno)}")

        self._mapped[page_index] = None

    def is_mapped(self, page_index: int) -> bool:
        return self._mapped[page_index] is not None

    def as_numpy(self, dtype=np.float16) -> np.ndarray:
        """
        View the entire arena as a flat numpy array (zero-copy).

        The returned array directly references the arena's VA range.
        Mapped pages contain their data; unmapped pages contain zeros.

        NOTE: Uses ctypes to avoid holding a reference to the mmap object,
        which would prevent mmap.close().
        """
        count = self.total_size // np.dtype(dtype).itemsize
        buf = (ctypes.c_char * self.total_size).from_address(self._base_addr)
        return np.frombuffer(buf, dtype=dtype, count=count)

    def as_numpy_page(self, page_index: int, dtype=np.float16) -> np.ndarray:
        """View a single page slot as a numpy array."""
        offset = page_index * self.page_size
        count = self.page_size // np.dtype(dtype).itemsize
        buf = (ctypes.c_char * self.page_size).from_address(
            self._base_addr + offset
        )
        return np.frombuffer(buf, dtype=dtype, count=count)

    def as_mlx_array(self, shape: Tuple[int, ...], dtype=None):
        """
        View the arena as an MLX array.

        On Apple Silicon, this is effectively zero-copy because MLX
        creates the array from numpy which shares the same physical
        memory (UMA).

        NOTE: In the production implementation, we'd use:
          1. Metal buffer wrapping (newBufferWithBytesNoCopy)
          2. MLX C++ API to create array from Metal buffer
        This numpy bridge is the PoC approach.
        """
        if not HAS_MLX:
            raise RuntimeError("MLX not available (requires macOS Apple Silicon)")

        if dtype is None:
            dtype = mx.float16

        # Get numpy view and reshape
        np_dtype = {
            mx.float16: np.float16,
            mx.float32: np.float32,
            mx.bfloat16: np.float16,  # numpy doesn't have bfloat16
        }.get(dtype, np.float16)

        np_arr = self.as_numpy(np_dtype).reshape(shape)

        # mx.array from numpy — on Apple Silicon UMA, this should be
        # a lightweight operation since CPU and GPU share memory.
        # In production, we'd bypass this and go Metal buffer → MLX array.
        return mx.array(np_arr)

    def close(self):
        if self._mm:
            # Unmap all pages first
            for i in range(self.num_pages):
                if self._mapped[i] is not None:
                    self.unmap(i)
            try:
                self._mm.close()
            except BufferError:
                # numpy views may hold references; the OS reclaims on exit
                pass
            self._mm = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        mapped_count = sum(1 for p in self._mapped if p is not None)
        return (
            f"MLXArena(total={self.total_size}, page_size={self.page_size}, "
            f"pages={self.num_pages}, mapped={mapped_count})"
        )


class MLXKVCacheManager:
    """
    Block-level KV cache manager for MLX, analogous to KVCacheManager.

    Maps the KV cache memory hierarchy:
      Blocks (logical, per-token-group) →
        Pages (physical, mmap'd shared memory) →
          Arena (virtual, contiguous VA range → Metal buffer → MLX array)

    This manages elastic allocation/deallocation of KV cache blocks
    for an MLX-based inference engine (e.g., mlx-lm).
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,  # tokens per block
        num_heads: int,
        head_dim: int,
        num_layers: int,
        dtype_bytes: int = 2,  # float16
        page_size: int = DEFAULT_PAGE_SIZE,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.dtype_bytes = dtype_bytes
        self.page_size = page_size

        # Cell = memory for one block in one layer (K or V)
        self.cell_size = block_size * num_heads * head_dim * dtype_bytes
        # KV pair per block per layer
        self.kv_cell_size = self.cell_size * 2  # K + V

        # Total memory per block across all layers
        self.block_mem = self.kv_cell_size * num_layers

        # Blocks per page
        self.blocks_per_page = max(1, page_size // self.block_mem)
        if self.blocks_per_page * self.block_mem > page_size:
            # Adjust page size to be a multiple of block memory
            effective_page_size = (
                (page_size + self.block_mem - 1) // self.block_mem
            ) * self.block_mem
            # Round up to HW page alignment
            if effective_page_size % HW_PAGE_SIZE != 0:
                effective_page_size = (
                    (effective_page_size + HW_PAGE_SIZE - 1) // HW_PAGE_SIZE
                ) * HW_PAGE_SIZE
            self.page_size = effective_page_size
            self.blocks_per_page = self.page_size // self.block_mem

        # Total pages needed
        self.num_pages = (num_blocks + self.blocks_per_page - 1) // self.blocks_per_page
        total_size = self.num_pages * self.page_size

        # Create arena
        self.arena = MLXArena(total_size, self.page_size)

        # Block tracking
        self._free_blocks = list(range(num_blocks))
        self._allocated_blocks = set()
        self._page_ref_count = [0] * self.num_pages  # blocks allocated per page
        self._pages: List[Optional[PhysicalPage]] = [None] * self.num_pages

    def alloc(self, count: int) -> List[int]:
        """Allocate `count` blocks. Returns list of block IDs."""
        if count > len(self._free_blocks):
            raise MemoryError(
                f"Cannot allocate {count} blocks, only {len(self._free_blocks)} free"
            )

        block_ids = self._free_blocks[:count]
        self._free_blocks = self._free_blocks[count:]

        for bid in block_ids:
            self._allocated_blocks.add(bid)
            page_idx = bid // self.blocks_per_page
            self._page_ref_count[page_idx] += 1

            # Map page on first block allocation
            if self._page_ref_count[page_idx] == 1:
                self._ensure_page_mapped(page_idx)

        return block_ids

    def free(self, block_ids: List[int]):
        """Free blocks. Unmaps pages when all their blocks are freed."""
        for bid in block_ids:
            assert bid in self._allocated_blocks, f"block {bid} not allocated"
            self._allocated_blocks.discard(bid)
            self._free_blocks.append(bid)

            page_idx = bid // self.blocks_per_page
            self._page_ref_count[page_idx] -= 1

            # Unmap page when last block is freed
            if self._page_ref_count[page_idx] == 0:
                self._ensure_page_unmapped(page_idx)

    def _ensure_page_mapped(self, page_idx: int):
        if self.arena.is_mapped(page_idx):
            return
        if self._pages[page_idx] is None:
            self._pages[page_idx] = PhysicalPage(self.page_size)
        self.arena.map(page_idx, self._pages[page_idx])

    def _ensure_page_unmapped(self, page_idx: int):
        if not self.arena.is_mapped(page_idx):
            return
        self.arena.unmap(page_idx)
        # Optionally free physical page to reclaim memory
        if self._pages[page_idx] is not None:
            self._pages[page_idx].close()
            self._pages[page_idx] = None

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    @property
    def num_mapped_pages(self) -> int:
        return sum(1 for p in self._page_ref_count if p > 0)

    @property
    def physical_memory_used(self) -> int:
        """Bytes of physical memory currently mapped."""
        return self.num_mapped_pages * self.page_size

    def __repr__(self):
        return (
            f"MLXKVCacheManager(blocks={self.num_blocks}, "
            f"allocated={len(self._allocated_blocks)}, "
            f"pages_mapped={self.num_mapped_pages}/{self.num_pages}, "
            f"phys_mem={self.physical_memory_used / 1024 / 1024:.1f}MB)"
        )

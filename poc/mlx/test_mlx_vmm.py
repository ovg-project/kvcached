#!/usr/bin/env python3
"""
test_mlx_vmm.py - Test the Python MLX VMM layer

Tests:
  1. Arena creation and basic properties
  2. Page map/unmap with data verification through arena's numpy view
  3. Zero-copy shared memory: writes through page visible through arena
  4. Block-level KV cache manager: alloc/free with automatic page management
  5. KV cache simulation: multiple concurrent requests with elastic memory

Runs on Linux (mmap/memfd) and macOS (mmap/shm_open).
MLX-specific tests are skipped if MLX is not available.
"""

import sys
import os
import time
import numpy as np

# Add parent to path for import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlx_vmm import PhysicalPage, MLXArena, MLXKVCacheManager, HW_PAGE_SIZE


def test_arena_creation():
    print("\n=== Test 1: Arena creation ===")

    page_size = 16 * 1024  # 16KB
    total = page_size * 8  # 8 pages
    arena = MLXArena(total, page_size)

    assert arena.total_size == total
    assert arena.page_size == page_size
    assert arena.num_pages == 8
    assert arena.base_ptr != 0
    print(f"  Arena: {arena}")
    print(f"  Base ptr: 0x{arena.base_ptr:x}")
    print("  [OK] Arena created")

    # Initial content should be all zeros
    np_view = arena.as_numpy(np.uint64)
    assert np.all(np_view == 0), "Arena should be zero-initialized"
    print("  [OK] Zero-initialized")

    arena.close()
    print("  PASSED")


def test_page_map_unmap():
    print("\n=== Test 2: Page map/unmap with data verification ===")

    page_size = 16 * 1024
    arena = MLXArena(page_size * 4, page_size)

    # Create a page with known data
    page = PhysicalPage(page_size)
    np_page = page.as_numpy(np.uint64)
    np_page[:] = 0xDEADBEEF

    # Map page at slot 0
    arena.map(0, page)
    assert arena.is_mapped(0)
    print("  [OK] Page mapped at slot 0")

    # Read through arena — should see the page's data
    arena_view = arena.as_numpy_page(0, np.uint64)
    assert np.all(arena_view == 0xDEADBEEF), "Arena should reflect page data"
    print("  [OK] Arena reads page data (0xDEADBEEF)")

    # Slots 1-3 should still be zeros
    for i in range(1, 4):
        slot_view = arena.as_numpy_page(i, np.uint64)
        assert np.all(slot_view == 0), f"Slot {i} should be zeros"
    print("  [OK] Unmapped slots are zeros")

    # Unmap
    arena.unmap(0)
    assert not arena.is_mapped(0)
    arena_view = arena.as_numpy_page(0, np.uint64)
    assert np.all(arena_view == 0), "Slot 0 should be zeros after unmap"
    print("  [OK] Unmapped, slot is zeros")

    page.close()
    arena.close()
    print("  PASSED")


def test_zero_copy_shared_memory():
    print("\n=== Test 3: Zero-copy shared memory (bidirectional) ===")

    page_size = 16 * 1024
    arena = MLXArena(page_size * 2, page_size)
    page = PhysicalPage(page_size)

    # Map
    arena.map(0, page)

    # Write through page's numpy view
    np_page = page.as_numpy(np.float32)
    np_page[:] = 3.14

    # Read through arena — should see 3.14 (zero-copy!)
    arena_view = arena.as_numpy_page(0, np.float32)
    assert np.allclose(arena_view, 3.14), "Arena should see writes through page"
    print("  [OK] Write through page → visible through arena")

    # Write through arena
    arena_view[:] = 2.718

    # Read through page — should see 2.718
    np_page_check = page.as_numpy(np.float32)
    assert np.allclose(np_page_check, 2.718), "Page should see writes through arena"
    print("  [OK] Write through arena → visible through page (bidirectional)")

    arena.unmap(0)
    page.close()
    arena.close()
    print("  PASSED")


def test_remap_different_data():
    print("\n=== Test 4: Remap slot with different page ===")

    page_size = 16 * 1024
    arena = MLXArena(page_size * 2, page_size)

    page_a = PhysicalPage(page_size)
    page_b = PhysicalPage(page_size)

    page_a.as_numpy(np.uint64)[:] = 0xAAAA
    page_b.as_numpy(np.uint64)[:] = 0xBBBB

    # Map page A
    arena.map(0, page_a)
    assert np.all(arena.as_numpy_page(0, np.uint64) == 0xAAAA)
    print("  [OK] Slot 0 = page A (0xAAAA)")

    # Unmap and map page B
    arena.unmap(0)
    arena.map(0, page_b)
    assert np.all(arena.as_numpy_page(0, np.uint64) == 0xBBBB)
    print("  [OK] Slot 0 remapped to page B (0xBBBB)")

    arena.unmap(0)
    page_a.close()
    page_b.close()
    arena.close()
    print("  PASSED")


def test_pointer_stability():
    print("\n=== Test 5: Base pointer stability ===")
    print("  (Metal buffer wraps this pointer; it must not change)")

    page_size = 16 * 1024
    arena = MLXArena(page_size * 4, page_size)
    original_base = arena.base_ptr

    page = PhysicalPage(page_size)
    page.as_numpy(np.uint64)[:] = 42

    for cycle in range(50):
        idx = cycle % 4
        arena.map(idx, page)
        assert arena.base_ptr == original_base
        arena.unmap(idx)
        assert arena.base_ptr == original_base

    print(f"  [OK] 50 map/unmap cycles, base stable at 0x{original_base:x}")

    page.close()
    arena.close()
    print("  PASSED")


def test_kv_cache_manager_basic():
    print("\n=== Test 6: KV Cache Manager — basic alloc/free ===")

    mgr = MLXKVCacheManager(
        num_blocks=64,
        block_size=16,
        num_heads=8,
        head_dim=128,
        num_layers=32,
        dtype_bytes=2,
        page_size=64 * 1024,
    )
    print(f"  Config: {mgr}")
    print(f"  Cell size: {mgr.cell_size} bytes")
    print(f"  Block mem (all layers, KV): {mgr.block_mem} bytes")
    print(f"  Blocks per page: {mgr.blocks_per_page}")

    # Allocate some blocks
    blocks_a = mgr.alloc(8)
    assert len(blocks_a) == 8
    print(f"  Allocated 8 blocks: {blocks_a}")
    print(f"  Physical memory: {mgr.physical_memory_used / 1024:.0f} KB ({mgr.num_mapped_pages} pages)")

    blocks_b = mgr.alloc(4)
    assert len(blocks_b) == 4
    print(f"  Allocated 4 more: {blocks_b}")
    print(f"  Physical memory: {mgr.physical_memory_used / 1024:.0f} KB ({mgr.num_mapped_pages} pages)")

    # Free first batch
    mgr.free(blocks_a)
    print(f"  Freed first 8 blocks")
    print(f"  Physical memory: {mgr.physical_memory_used / 1024:.0f} KB ({mgr.num_mapped_pages} pages)")

    # Free second batch
    mgr.free(blocks_b)
    print(f"  Freed remaining blocks")
    print(f"  Physical memory: {mgr.physical_memory_used / 1024:.0f} KB ({mgr.num_mapped_pages} pages)")
    assert mgr.physical_memory_used == 0, "All physical memory should be freed"

    mgr.arena.close()
    print("  PASSED")


def test_kv_cache_elastic_simulation():
    print("\n=== Test 7: Elastic KV cache — multi-request simulation ===")

    # Simulate Llama-3.1-8B parameters
    mgr = MLXKVCacheManager(
        num_blocks=256,      # total capacity
        block_size=16,       # tokens per block
        num_heads=8,         # GQA heads for KV
        head_dim=128,
        num_layers=32,
        dtype_bytes=2,       # float16
        page_size=256 * 1024,  # 256KB pages
    )
    print(f"  Simulating Llama-3.1-8B-like KV cache")
    print(f"  {mgr}")

    # Request 1: 2048 tokens → 128 blocks
    print("\n  --- Request 1: 2048 tokens ---")
    req1_blocks = mgr.alloc(128)
    print(f"  Allocated {len(req1_blocks)} blocks")
    print(f"  {mgr}")

    # Request 2: 512 tokens → 32 blocks
    print("\n  --- Request 2: 512 tokens ---")
    req2_blocks = mgr.alloc(32)
    print(f"  Allocated {len(req2_blocks)} blocks")
    print(f"  {mgr}")

    # Request 1 finishes → memory reclaimed
    print("\n  --- Request 1 done ---")
    mgr.free(req1_blocks)
    print(f"  Freed {len(req1_blocks)} blocks")
    print(f"  {mgr}")

    # Request 3: 1024 tokens → 64 blocks (reuses freed memory)
    print("\n  --- Request 3: 1024 tokens (reuses freed pages) ---")
    req3_blocks = mgr.alloc(64)
    print(f"  Allocated {len(req3_blocks)} blocks")
    print(f"  {mgr}")

    # All done
    mgr.free(req2_blocks)
    mgr.free(req3_blocks)
    print(f"\n  Final: {mgr}")
    assert mgr.physical_memory_used == 0

    mgr.arena.close()
    print("  PASSED")


def test_map_unmap_throughput():
    print("\n=== Test 8: Map/unmap throughput ===")

    page_size = 64 * 1024
    arena = MLXArena(page_size * 4, page_size)
    page = PhysicalPage(page_size)

    num_ops = 1000
    t0 = time.monotonic()
    for _ in range(num_ops):
        arena.map(0, page)
        arena.unmap(0)
    elapsed = time.monotonic() - t0

    ops_per_sec = (2 * num_ops) / elapsed
    us_per_op = 1e6 / ops_per_sec

    print(f"  {num_ops} map+unmap pairs in {elapsed:.3f}s")
    print(f"  {ops_per_sec:.0f} ops/sec, {us_per_op:.1f} us/op")

    page.close()
    arena.close()
    print("  PASSED")


def test_mlx_integration():
    """Test MLX array creation from arena (macOS only)."""
    try:
        import mlx.core as mx
    except ImportError:
        print("\n=== Test 9: MLX integration — SKIPPED (MLX not available) ===")
        return

    print("\n=== Test 9: MLX array from arena ===")

    page_size = 16 * 1024
    arena = MLXArena(page_size * 4, page_size)

    # Map a page with known float16 data
    page = PhysicalPage(page_size)
    np_data = page.as_numpy(np.float16)
    np_data[:] = 1.5

    arena.map(0, page)

    # Create MLX array from arena
    num_elements = arena.total_size // 2  # float16 = 2 bytes
    mlx_arr = arena.as_mlx_array(shape=(num_elements,), dtype=mx.float16)

    # First page should be 1.5, rest should be 0
    page_elements = page_size // 2
    mx.eval(mlx_arr)

    first_page = mlx_arr[:page_elements]
    rest = mlx_arr[page_elements:]

    assert mx.allclose(first_page, mx.full((page_elements,), 1.5, dtype=mx.float16))
    print("  [OK] MLX reads mapped page data (1.5)")

    assert mx.allclose(rest, mx.zeros((num_elements - page_elements,), dtype=mx.float16))
    print("  [OK] MLX reads zeros from unmapped pages")

    arena.unmap(0)
    page.close()
    arena.close()
    print("  PASSED")


if __name__ == "__main__":
    print("KVCached MLX PoC — Python VMM Tests")
    print("=" * 50)
    print(f"Platform: {sys.platform}")
    print(f"Python: {sys.version}")

    test_arena_creation()
    test_page_map_unmap()
    test_zero_copy_shared_memory()
    test_remap_different_data()
    test_pointer_stability()
    test_kv_cache_manager_basic()
    test_kv_cache_elastic_simulation()
    test_map_unmap_throughput()
    test_mlx_integration()

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)

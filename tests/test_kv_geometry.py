# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""KV block / physical page geometry.

Hybrid models pad a mamba state into one attention block (Qwen3.8-27B:
784 tokens x 4096 B = 3,211,264 B). With KVCACHED_PAGE_SIZE_MB=4 two thirds of
the pages held no whole block; _alloc parked each such page after mapping it
and free() never visited it again, so KV memory grew with every request.
"""

import math
import sys
import threading
import types
from unittest import mock

import pytest

from kvcached.kv_geometry import (
    MIB,
    aligned_block_size,
    check_page_geometry,
    has_zero_capacity_pages,
    recommend_page_geometry,
)

QWEN38_27B = (784, 4096)   # vLLM block size, attention bytes per token and layer
QWEN35_4B = (528, 4096)


def _block_range(page_id, page_size, unit):
    # InternalPage::get_block_range
    return (page_id * page_size + unit - 1) // unit, ((page_id + 1) * page_size) // unit


def _brute_zero_capacity(unit, page_size):
    period = unit // math.gcd(unit, page_size) + 1
    return any(_block_range(p, page_size, unit)[1] <= _block_range(p, page_size, unit)[0]
               for p in range(period))


@pytest.mark.parametrize("unit,page_mb", [
    (3211264, 4), (3211264, 8), (3211264, 16), (2162688, 4), (2162688, 16),
    (4 * MIB, 4), (2 * MIB, 4), (18432, 2), (32768, 2), (131072, 2),
    (3 * MIB, 4), (int(2.5 * MIB), 4), (MIB + 4096, 2), (MIB, 2),
    (4 * MIB, 6), (6 * MIB, 10), (5 * MIB, 6),
])
def test_zero_capacity_criterion_matches_page_ranges(unit, page_mb):
    page = page_mb * MIB
    assert has_zero_capacity_pages(unit, page) == _brute_zero_capacity(unit, page)


def test_zero_capacity_criterion_matches_all_small_integer_geometries():
    for page in range(1, 65):
        for unit in range(1, 65):
            assert has_zero_capacity_pages(unit, page) == _brute_zero_capacity(unit, page), (
                unit, page,
            )


def test_real_hybrid_units():
    assert has_zero_capacity_pages(3211264, 4 * MIB)       # 27B, the leaking run
    assert not has_zero_capacity_pages(3211264, 16 * MIB)  # straddling, but every page used
    assert has_zero_capacity_pages(2162688, 4 * MIB)       # Qwen3.5-4B
    assert not has_zero_capacity_pages(4 * MIB, 4 * MIB)   # aligned geometry


@pytest.mark.parametrize("model,page_mb,expected", [
    (QWEN38_27B, 4, 1024), (QWEN38_27B, 8, 1024), (QWEN38_27B, 16, 1024),
    (QWEN38_27B, 2, None), (QWEN35_4B, 4, 1024), (QWEN35_4B, 2, None),
])
def test_aligned_block_size(model, page_mb, expected):
    block_size, per_token = model
    got = aligned_block_size(block_size, per_token, page_mb * MIB)
    assert got == expected
    if got is not None:
        assert (page_mb * MIB) % (got * per_token) == 0
        assert got % (block_size & -block_size) == 0  # keeps vLLM's kernel alignment


def test_aligned_block_size_keeps_growth_bounded():
    # 1000 tokens x 3000 B cannot tile a 4 MiB page within 2x the block size.
    assert aligned_block_size(1000, 3000, 4 * MIB) is None


def test_recommendation_for_27b():
    assert recommend_page_geometry(*QWEN38_27B) == (4, 1024)


def test_check_page_geometry_messages():
    assert check_page_geometry(32768, 2 * MIB, 16) is None        # attention: tiles
    assert check_page_geometry(18432, 2 * MIB, 16) is None        # MLA: straddles, no empty page
    assert check_page_geometry(3211264, 16 * MIB, 784) is None    # straddles, no empty page
    assert check_page_geometry(4 * MIB, 4 * MIB, 1024) is None
    assert check_page_geometry(4 * MIB, 6 * MIB, 1024) is None
    assert check_page_geometry(6 * MIB, 10 * MIB, 1536) is None

    msg = check_page_geometry(3211264, 4 * MIB, 784)
    assert msg is not None
    assert "does not tile" in msg and "--block-size 1024" in msg
    msg = check_page_geometry(3211264, 2 * MIB, 784)
    assert msg is not None
    assert "larger than the page" in msg
    assert "KVCACHED_PAGE_SIZE_MB=4 with --block-size 1024" in msg
    # Without the token count the advice falls back to a page size.
    msg = check_page_geometry(3211264, 4 * MIB)
    assert msg is not None and "KVCACHED_PAGE_SIZE_MB=" in msg


# --------------------------------------------------------------- KVCacheManager

def _stub_extension(monkeypatch):
    torch = mock.MagicMock()
    torch.__version__ = "2.6.0"
    for name, mod in (("torch", torch), ("torch.cuda", torch.cuda), ("torch.utils", torch.utils),
                      ("torch.utils.cpp_extension", torch.utils.cpp_extension),
                      ("posix_ipc", mock.MagicMock()), ("kvcached.vmm_ops", mock.MagicMock())):
        monkeypatch.setitem(sys.modules, name, mod)


def test_manager_rejects_zero_capacity_geometry(monkeypatch):
    _stub_extension(monkeypatch)
    import kvcached.kv_cache_manager as kcm
    from kvcached.utils import KVCachedConfigError

    monkeypatch.setattr(kcm, "PAGE_SIZE", 4 * MIB)
    allocator = mock.Mock()
    monkeypatch.setattr(kcm, "PageAllocator", allocator)
    with pytest.raises(KVCachedConfigError, match="--block-size 1024"):
        kcm.KVCacheManager(num_blocks=416, block_size=784, cell_size=4096, num_layers=16)
    allocator.assert_not_called()


class _EmptyPage:
    page_id = 1

    def init(self, block_mem_size):
        pass

    def num_free_blocks(self):
        return 0


def test_alloc_fails_loud_instead_of_parking_an_empty_page(monkeypatch):
    _stub_extension(monkeypatch)
    import kvcached.kv_cache_manager as kcm
    from kvcached.errors import StateConsistencyError
    from kvcached.locks import NoOpLock

    manager = object.__new__(kcm.KVCacheManager)
    manager.page_size = 4 * MIB
    manager.block_mem_size = 3211264
    manager.page_allocator = types.SimpleNamespace(
        alloc_page=lambda: _EmptyPage(), get_resize_target=lambda: 0,
        get_num_free_pages=lambda: 10, get_avail_physical_pages=lambda: 10,
        get_num_reserved_pages=lambda: 0)
    manager.num_avail_blocks = 0
    manager.avail_pages = {}
    manager.full_pages = {}
    manager.reserved_blocks = []
    manager.in_shrink = False
    manager.target_num_blocks = None
    manager._lock = NoOpLock()
    manager._post_init_done = threading.Event()
    manager._post_init_done.set()
    monkeypatch.setattr(kcm.KVCacheManager, "available_size", lambda self: 10)

    with pytest.raises(StateConsistencyError, match="holds no whole"):
        manager.alloc(1)
    assert manager.full_pages == {}

# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Layer-major compressed index layout and physical ownership, without CUDA."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from kvcached.integration import sglang
from kvcached.integration.sglang import patches, qsa


def test_shared_index_tiles_are_mapped_and_zeroed_once(monkeypatch):
    raw = torch.full((320,), 255, dtype=torch.uint8)
    manager = Mock(permanent_block_ids={2})
    manager.page_allocator.get_num_reserved_pages.return_value = 1
    manager.exact_recovery_quote.return_value = None
    manager.page_allocator.get_avail_physical_bytes.return_value = 0
    full = Mock()
    full.exact_recovery_quote.return_value = (0, [], 0)
    full.page_allocator.get_num_reserved_pages.return_value = 0
    manager.available_size.return_value = 3
    manager.alloc_exact.side_effect = [None, [1, 3], [1, 3]]
    api = SimpleNamespace(create_kv_tensors=Mock(return_value=[raw]),
                          get_kv_cache_manager=Mock(return_value=manager))
    monkeypatch.setattr(sglang, "interfaces", api, raising=False)
    monkeypatch.setattr(qsa, "PAGE_SIZE", 64)
    current_stream = Mock()
    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    fence = Mock()
    monkeypatch.setattr(torch.cuda, "synchronize", fence)

    pool = SimpleNamespace(qsa_compress_ratio=4, qsa_index_kv_heads=1, qsa_index_head_dim=4,
                           full_kv_pool=SimpleNamespace(kvcached_allocator=full))
    backing = qsa.QSAIndexBacking(pool, (2, 72), torch.bfloat16, "cpu", 4, 18)
    # Two 8-byte native padding slots occupy the permanently mapped tiles.
    assert backing.leases.physical_demand(15) == 1
    assert api.get_kv_cache_manager.call_args.kwargs["permanent_block_ids"] == (2,)
    assert not raw[:64].any() and not raw[128:192].any()
    assert (raw[64:128] == 255).all() and (raw[256:] == 255).all()
    with pytest.raises(qsa.KVCachedConfigError):
        backing.acquire([18])
    assert not backing.acquire([8]) and not backing.leases.live
    assert backing.acquire([8])
    manager.alloc_exact.assert_called_with([1, 3])
    rows = backing.flat.view(2, 18, 4)
    assert not rows[:, 8:10].any()
    rows[:, 8] = 42
    assert backing.acquire([9]) and (rows[:, 8] == 42).all()
    rows[:, 9] = 17
    backing.release([8])
    assert (rows[:, 9] == 17).all()
    fence.assert_not_called()
    manager.free.assert_not_called()
    backing.clear()
    manager.trim.assert_called_once()
    fence.assert_not_called()
    manager.free.assert_called_once_with([1, 3])
    manager.trim_selected_reserves.assert_not_called()
    assert backing.acquire([8]) and not rows[:, 8:10].any()
    assert (raw[256:] == 255).all()

    full.exact_recovery_quote.return_value = (64, [], 0)
    manager.exact_recovery_quote.return_value = (64, [], 0)
    manager.page_allocator.get_avail_physical_bytes.return_value = 64
    assert not backing.can_acquire([16])
    manager.page_allocator.get_avail_physical_bytes.return_value = 128
    assert backing.can_acquire([16])


def test_first_decode_miss_survives_smaller_successful_reservation():
    backing = object.__new__(qsa.QSAIndexBacking)
    backing.full_manager = Mock()
    backing.failed_decode = ()

    def reserve(count, *, packed):
        assert packed
        backing._decode_attempt = tuple(range(1, count + 1))
        return count == 1

    backing.full_manager.try_to_reserve.side_effect = reserve
    assert not backing.reserve_decode(3, packed=True)
    assert not backing.reserve_decode(2, packed=True)
    assert backing.reserve_decode(1, packed=True)
    assert backing.failed_decode == (1, 2, 3)
    assert backing._decode_attempt is None


@pytest.mark.parametrize(("prefill_bytes", "decode_bytes"), [(0, 64), (96, 32)])
def test_fallback_decode_and_prefill_publish_the_larger_shortfall(monkeypatch, prefill_bytes, decode_bytes):
    backing = object.__new__(qsa.QSAIndexBacking)
    req = object()
    backing.full_manager = Mock(group_id=4)
    backing.failed_prefill = ((id(req),), (8,))
    backing.failed_decode = (12, 13)
    backing.posted_bytes = 0
    prices = Mock(side_effect=[0, prefill_bytes, decode_bytes])
    monkeypatch.setattr(backing, "_demand_bytes", prices)
    publish = Mock()
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", SimpleNamespace(set_kv_demand=publish))
    failed = SimpleNamespace(requests=(req,), requested=1)
    backing.post_demand(5, allocated=1, failed=failed, prefill_need=4)
    assert [c.args for c in prices.call_args_list] == [(5,), (4, (8,)), (3, (12, 13))]
    publish.assert_called_once_with(max(prefill_bytes, decode_bytes), 4)
    assert not backing.failed_decode and backing.failed_prefill == ((), ())


@pytest.mark.parametrize(("all_reused", "matching_batch", "admission_blocks", "expected"), [
    (False, True, 0, 128), (True, True, 0, 32), (False, False, 0, 0),
    (False, True, 5, 256),
])
def test_report_prices_only_the_native_failed_batch(monkeypatch, all_reused, matching_batch, admission_blocks, expected):
    req = object()
    full = Mock(group_id=4, reserved_blocks=[], _priority_reserve_debt=0)
    full.demand_pages.return_value = 0
    full.take_step_alloc_blocks.return_value = 1
    full._memory_limit_page_bundle_bytes.return_value = 64
    full.page_allocator.get_num_reserved_pages.return_value = 0
    full.exact_recovery_quote.return_value = None
    index = Mock(permanent_block_ids={3})
    index.page_allocator.get_num_reserved_pages.return_value = 0
    raw = torch.empty(448, dtype=torch.uint8)
    api = SimpleNamespace(create_kv_tensors=Mock(return_value=[raw]), get_kv_cache_manager=Mock(return_value=index))
    monkeypatch.setattr(sglang, "interfaces", api, raising=False)
    monkeypatch.setattr(qsa, "PAGE_SIZE", 64)
    monkeypatch.setattr(torch.cuda, "current_stream", Mock())
    pool = SimpleNamespace(qsa_compress_ratio=4, qsa_index_kv_heads=1, qsa_index_head_dim=4,
                           full_kv_pool=SimpleNamespace(kvcached_allocator=full))
    backing = qsa.QSAIndexBacking(pool, (2, 104), torch.bfloat16, "cpu", 4, 26)
    full._block_backing = backing
    index.alloc_exact.side_effect = [None, [1, 4], [2, 5], [6]]
    monkeypatch.setattr(patches, "_sglang_demand_manager", lambda _: full)
    monkeypatch.setattr(patches, "_sglang_dsv4_demand_allocator", lambda _: None)
    monkeypatch.setattr(patches._SGLANG_DEMAND, "open", True, raising=False)
    monkeypatch.setattr(patches._SGLANG_DEMAND, "allocation_miss", None, raising=False)
    monkeypatch.setattr(patches._SGLANG_DEMAND, "prefill_miss", admission_blocks or None, raising=False)
    monkeypatch.setattr(patches._SGLANG_DEMAND, "prefill_input_blocks", 1, raising=False)
    monkeypatch.setattr(patches._SGLANG_DEMAND, "need_overlap", 1, raising=False)
    with patches._sglang_prefill_requests([req]):
        assert not backing.acquire([8, 16])
    with patches._sglang_prefill_requests([req if matching_batch else object()]):
        patches._sglang_capacity_miss(None, 2, "pages", full)

    # Decode reused one of the failed IDs. Only ID 16 still needs index tiles.
    assert backing.acquire([8])
    full.page_allocator.get_num_reserved_pages.return_value = 1
    index.exact_recovery_quote.return_value = (128, [], 0)
    index.page_allocator.get_avail_physical_bytes.return_value = -64
    if all_reused:
        # Both prior IDs are live; replacing one full page at negative
        # headroom still needs a grant, even after returning idle tile 6.
        assert backing.acquire([16])
        assert backing.acquire([24])
        backing.release([24])
        index.free.assert_called_with([6])
        assert 6 not in backing.leases.refs and 6 not in backing.leases.permanent_pages
        full.demand_pages.return_value = 1
        full.page_allocator.get_num_reserved_pages.return_value = 0
        index.page_allocator.get_num_reserved_pages.return_value = 1
        index.exact_recovery_quote.return_value = (0, [6], 0)
        index.page_allocator.get_avail_physical_bytes.return_value = -32

    publish = Mock()
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", SimpleNamespace(set_kv_demand=publish))
    if admission_blocks:
        # A second request hit NO_TOKEN, so its forecast includes the selected
        # inputs above. It needs a full bundle as well as their exact tiles.
        full.demand_pages.side_effect = lambda n, **_: 2 if n >= admission_blocks else 0
    patches._sglang_post_demand(SimpleNamespace(), admission_blocks)
    full.demand_pages.assert_called_with(max(3, admission_blocks), include_reserves=False)
    if matching_batch:
        index.exact_recovery_quote.assert_called_with([] if all_reused else [2, 5])
        publish.assert_called_once_with(expected, 4)
    else:
        index.exact_recovery_quote.assert_not_called()
        publish.assert_not_called()
    assert backing.failed_prefill == ((), ())
    # An idle poll does not price the old batch or evict on its behalf.
    full.demand_pages.return_value = 0
    index.exact_recovery_quote.reset_mock()
    backing.post_demand(0)
    index.exact_recovery_quote.assert_not_called()
    if expected:
        publish.assert_called_with(0, 4)

# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Native token-page demand uses physical backing units and native retraction."""

from types import SimpleNamespace as NS

import pytest
from test_sglang_dsv4_native_backing import MODULE, Composite, DecodeComposite, Manager
from test_sglang_dsv4_runtime_reservation_patch import _load_sglang_modules


class PriorityManager(Manager):
    def __init__(self):
        super().__init__()
        self.step = 0
        self.demands = []
        self.forecasts = []
        self.events = []
        self.occupied = 10

    def take_step_alloc_blocks(self):
        result, self.step = self.step, 0
        self.events.append(("take", result))
        return result

    def demand_pages(self, physical):
        self.forecasts.append(physical)
        return physical

    def post_demand(self, physical):
        self.demands.append(physical)

    def post_reclaimable_levels(self):
        self.events.append(("levels", None))

    def reclaim_for_priority(self):
        self.events.append(("reclaim", None))
        return 0

    def occupied_pages(self):
        return self.occupied

    def _refund_priority_reserve(self):
        self.events.append(("refund", None))


@pytest.mark.parametrize("hisparse", [False, True])
def test_native_demand_uses_backing_slack_and_actual_physical_allocations(monkeypatch, hisparse):
    _, patches, _ = _load_sglang_modules(monkeypatch)
    manager = PriorityManager()
    backing = MODULE.SWABacking(manager, native_pages=12, native_page_bytes=2,
                                physical_page_bytes=8)
    assert backing.acquire({1})
    allocator = Composite(4)
    MODULE.bind_native_swa(allocator, backing)
    owner = NS(logical_attn_allocator=allocator) if hisparse else allocator
    req = NS(origin_input_ids=list(range(12)), prefix_indices=[], output_ids=[],
             sampling_params=NS(max_new_tokens=4))
    scheduler = NS(token_to_kv_pool_allocator=owner, waiting_queue=[req], chunked_req=None)
    assert patches._sglang_demand_manager(scheduler) is manager
    # Physical managers must not enter the old native-token budget round.
    assert patches._sglang_budget_managers(scheduler) == []
    manager.step = 7
    with patches._sglang_demand_round(scheduler):
        assert manager.step == 0
        manager.step = 2
    # Four native pages need eight bytes; four bytes of backing slack remain.
    # One new physical page plus two actually allocated this step is three.
    assert manager.forecasts == [3]
    assert manager.demands == [3]
    assert manager.events[:2] == [("take", 7), ("reclaim", None)]


def test_native_decode_gate_honors_priority_without_aborting_last_request(monkeypatch):
    _, patches, _ = _load_sglang_modules(monkeypatch)
    manager = PriorityManager()
    backing = MODULE.SWABacking(manager, native_pages=12, native_page_bytes=6,
                                physical_page_bytes=8)
    assert backing.acquire({1})
    allocator = DecodeComposite((3, 7, 9))
    MODULE.bind_native_swa(
        allocator, backing,
        on_priority_miss=lambda requests: patches._sglang_priority_decode_miss(manager, requests),
    )
    scheduler = NS(token_to_kv_pool_allocator=allocator)
    patches._sglang_begin_preemption(scheduler, 2)
    try:
        assert not allocator.check_decode_capacity(num_tokens=4, requests=[NS(), NS()])
        assert backing.live == {1}
        assert allocator.check_decode_capacity(num_tokens=4, requests=[NS()])
        allocator._kvcached_release_decode_reservation()
        manager.occupied = 8
        assert allocator.check_decode_capacity(num_tokens=4, requests=[NS(), NS()])
    finally:
        patches._sglang_end_preemption()


def test_failed_sparse_prefix_posts_its_exact_physical_cost(monkeypatch):
    _load_sglang_modules(monkeypatch)
    manager = PriorityManager()
    backing = MODULE.SWABacking(manager, native_pages=12, native_page_bytes=6,
                                physical_page_bytes=8)
    assert backing.acquire({1, 2})
    allocator = DecodeComposite((10,))
    allocator.swa_attn_allocator.live.add(2)
    manager.budget = len(manager.live)
    MODULE.bind_native_swa(allocator, backing)
    assert not allocator.check_decode_capacity(num_tokens=4)
    # Aggregate slack can hold one native page, but ID 10 needs pages 7 and 8.
    assert backing.physical_demand(1) == 0
    assert allocator._kvcached_backing_demand(1) == [(manager, 2)]

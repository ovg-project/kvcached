# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Reservation ownership with real GPU buffers, not full DeepSeek-V4 serving."""

import gc
import types

import pytest


@pytest.mark.parametrize("split_indexer", [False, True])
def test_gpu_reservations_survive_failed_peer_pool(monkeypatch, split_indexer):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    from kvcached.integration.sglang import interfaces, patches

    class Pool:
        _kvcached_runtime_reservation_breakdown: dict[str, int]

        def __init__(self, size, fail=False):
            self.device = "cuda:0"
            self.swa_kv_pool = types.SimpleNamespace(
                kv_buffer=[torch.full((size,), 7, dtype=torch.uint8, device=self.device)]
            )
            if fail:
                # A real allocator OOM after one buffer succeeds must not
                # publish the incomplete pool or change another pool's account.
                total = torch.cuda.get_device_properties(self.device).total_memory
                torch.empty((total * 2,), dtype=torch.uint8, device=self.device)
                raise AssertionError("expected CUDA allocation to fail")
            if split_indexer:
                self.c4_indexer_kv_pool = types.SimpleNamespace(
                    index_k_with_scale_buffer=None,
                    index_k_payload_buffer=[torch.zeros(size, dtype=torch.uint8, device=self.device)],
                    index_k_scale_buffer=[torch.zeros(size // 4, dtype=torch.uint8, device=self.device)],
                )
            else:
                self.c4_indexer_kv_pool = types.SimpleNamespace(
                    index_k_with_scale_buffer=[torch.zeros(size, dtype=torch.uint8, device=self.device)]
                )

    monkeypatch.setattr(
        patches.DeepSeekV4RuntimeReservationPatch, "initialize_version_info", lambda self: True
    )
    module = types.ModuleType("fake_dsv4_pool")
    setattr(module, "DeepSeekV4TokenToKVPool", Pool)
    assert patches.DeepSeekV4RuntimeReservationPatch().apply(module)
    before = interfaces.get_runtime_owned_reservation_bytes("cuda:0")
    size = 2 * 1024 * 1024
    target, draft = Pool(size), Pool(size // 2)
    target_bytes = sum(target._kvcached_runtime_reservation_breakdown.values())
    draft_bytes = sum(draft._kvcached_runtime_reservation_breakdown.values())
    assert target_bytes == size * 2 + (size // 4 if split_indexer else 0)
    expected = before + target_bytes + draft_bytes
    assert interfaces.get_runtime_owned_reservation_bytes("cuda:0") == expected
    for _ in range(2):
        with pytest.raises(torch.OutOfMemoryError):
            Pool(size, fail=True)
        gc.collect()
        assert interfaces.get_runtime_owned_reservation_bytes("cuda:0") == expected
        assert bool((target.swa_kv_pool.kv_buffer[0] == 7).all())
        assert bool((draft.swa_kv_pool.kv_buffer[0] == 7).all())
    del target
    gc.collect()
    assert interfaces.get_runtime_owned_reservation_bytes("cuda:0") == before + draft_bytes
    del draft
    gc.collect()
    assert interfaces.get_runtime_owned_reservation_bytes("cuda:0") == before
    recovery = Pool(size)
    assert bool((recovery.swa_kv_pool.kv_buffer[0] == 7).all())
    del recovery
    gc.collect()
    assert interfaces.get_runtime_owned_reservation_bytes("cuda:0") == before
    torch.cuda.synchronize()

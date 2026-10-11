# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""SGLang pools of different sizes get their own /dev/shm segments.

Regression for SGLang hybrid models: the C++ tracker adds the _g<id> suffix
only when it is given no name, and KVCacheManager always passes one, so the
full-attention and Mamba (or SWA) pools shared one segment. Each pool then
derived its resize target from the other pool's limit; on Qwen3.8-27B the
17-slot Mamba pool grew to 86 pages and mapped outside its reservation.
vLLM pools keep sharing the instance segment.
"""

import types

import pytest
from test_kvcache_manager_post_init import _import_kv_cache_manager


@pytest.mark.parametrize("own_segment,group_id,suffix", [
    (True, 0, ""), (True, 1, "_g1"), (True, 1000, "_g1000"),
    (False, 1, ""),  # default (vLLM): one segment per instance
])
def test_segment_name(monkeypatch, own_segment, group_id, suffix):
    kv_cache_manager = _import_kv_cache_manager(monkeypatch)
    seen = {}

    class FakePageAllocator:
        def __init__(self, *args, **kwargs):
            seen["ipc_name"] = kwargs["ipc_name"]

        def set_use_worker_ipc(self, enabled):
            pass

        def start_prealloc_thread(self):
            pass

    monkeypatch.setattr(kv_cache_manager, "PageAllocator", FakePageAllocator)
    monkeypatch.setattr(kv_cache_manager, "DEFAULT_IPC_NAME", "SGLANG")
    monkeypatch.setattr(
        kv_cache_manager.threading, "Thread",
        lambda *args, **kwargs: types.SimpleNamespace(start=lambda: None))

    manager = kv_cache_manager.KVCacheManager(
        num_blocks=4, block_size=1, cell_size=1, num_layers=1,
        async_sched=True, group_id=group_id, own_segment=own_segment)

    assert seen["ipc_name"] == manager.ipc_name == "SGLANG" + suffix

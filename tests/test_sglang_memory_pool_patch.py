# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import sys
import types
from typing import Any

from kvcached.integration.sglang.patches import ElasticMemoryPoolPatch


class FakeMHATokenToKVPool:
    _post_capture_owner: Any

    def _build_kv_buffer_descs(self):
        return ["k0", "v0"]

    def _init_data_ptrs_and_strides(self):
        self.data_ptrs_initialized = True

    def finalize_backing(self, config):
        self._finalize_backing_tokens(config.max_total_num_tokens)

    def _finalize_backing_tokens(self, final_num_tokens):
        self._post_capture_owner.finalize(final_num_tokens)
        self.size = int(final_num_tokens)


def test_elastic_mha_post_capture_finalization_uses_kvcached_backing(monkeypatch):
    distributed: Any = types.ModuleType("sglang.srt.distributed")
    distributed.get_pipeline_model_parallel_rank = lambda: 0
    distributed.get_tensor_model_parallel_rank = lambda: 0
    distributed.get_tensor_model_parallel_world_size = lambda: 1
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed", distributed)

    calls = []
    interfaces: Any = types.ModuleType("kvcached.integration.sglang.interfaces")
    interfaces.init_kvcached = lambda **kwargs: calls.append(("init", kwargs))
    interfaces.alloc_kv_cache = lambda **kwargs: (["k"], ["v"])
    monkeypatch.setitem(
        sys.modules, "kvcached.integration.sglang.interfaces", interfaces
    )
    import kvcached.integration.sglang as sglang_integration

    monkeypatch.setattr(sglang_integration, "interfaces", interfaces, raising=False)

    memory_pool: Any = types.ModuleType("sglang.srt.mem_cache.memory_pool")
    memory_pool.MHATokenToKVPool = FakeMHATokenToKVPool
    assert ElasticMemoryPoolPatch().inject_elastic_mem_pool(memory_pool) is True

    pool = memory_pool.ElasticMHATokenToKVPool.__new__(
        memory_pool.ElasticMHATokenToKVPool
    )
    pool.device = "cuda:0"
    pool.size = 1024
    pool.page_size = 16
    pool.head_num = 8
    pool.head_dim = 128
    pool.dtype = types.SimpleNamespace(itemsize=2)
    pool.layer_num = 1
    pool._group_id = 0
    pool.post_capture_active = True
    pool._post_capture_owner = None

    pool._create_buffers()
    pool.finalize_backing(types.SimpleNamespace(max_total_num_tokens=768))

    assert pool.k_buffer == ["k"]
    assert pool.v_buffer == ["v"]
    assert pool._kv_buffer_descs == ["k0", "v0"]
    assert pool.data_ptrs_initialized is True
    assert pool._post_capture_owner is None
    assert pool.size == 768
    assert calls == [
        (
            "init",
            {"tp_rank": 0, "world_size": 1, "pp_rank": 0, "async_sched": True},
        )
    ]

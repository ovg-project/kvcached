# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Native CPU offload with rank-private buffers and a common chunk capacity."""

from dataclasses import replace
from typing import Any

from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

from kvcached.integration.vllm.offload_config import PLAN_KEY, positive_int
from kvcached.utils import KVCachedConfigError


class KVCachedCPUOffloadingSpec(CPUOffloadingSpec):
    def __init__(self, config: Any):
        plan = config.extra_config.get(PLAN_KEY)
        if not isinstance(plan, dict):
            raise KVCachedConfigError("CPU offload capacity must be configured before KV initialization")
        super().__init__(config)
        count = positive_int(plan["num_chunks"], "CPU chunk count")
        if self.blocks_per_chunk != plan["blocks_per_chunk"]:
            raise KVCachedConfigError("Native CPU chunk granularity differs from the capacity plan")
        if config.parallel.world_size != len(plan["payloads"]):
            raise KVCachedConfigError("Native CPU worker count differs from the capacity plan")
        if hasattr(self, "num_chunks"):
            self.num_chunks = count
        elif hasattr(self, "num_blocks"):
            self.num_blocks = count
        else:
            raise KVCachedConfigError("Unsupported native CPU offload capacity API")
        self._kvcached_plan = plan

    def _uses_shared_region(self) -> bool:
        # Reuse native pinned tensors and disable shared-layout writer suppression.
        return False

    def create_worker(self, kv_caches: Any) -> Any:
        rank = self.config.parallel.rank
        plan = self._kvcached_plan
        if not isinstance(rank, int) or not 0 <= rank < len(plan["payloads"]):
            raise KVCachedConfigError("Invalid CPU offload worker rank")
        actual = 0
        for cache in kv_caches.tensors:
            page = positive_int(cache.page_size_bytes, "canonical KV page size")
            if cache.tensor.numel() * cache.tensor.element_size() != plan["gpu_blocks"][rank] * page:
                raise KVCachedConfigError("Canonical KV tensor differs from its logical descriptor")
            actual += page
        if actual != plan["payloads"][rank]:
            raise KVCachedConfigError("Canonical KV payload differs from the CPU offload capacity plan")
        local_groups = plan["groups"][rank]
        if len(kv_caches.group_data_refs) != len(local_groups):
            raise KVCachedConfigError("Canonical KV group indices differ from the CPU offload plan")
        # Packed registration emits a reference even for a stage-local empty group.
        kv_caches = replace(kv_caches, group_data_refs=[
            refs if present else []
            for refs, present in zip(kv_caches.group_data_refs, local_groups)
        ])
        return super().create_worker(kv_caches)

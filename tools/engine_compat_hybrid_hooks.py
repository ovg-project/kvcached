# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Trusted probe hooks; never installed into a candidate or normal serving."""

import functools
import json
import os

from wrapt.importer import when_imported

_armed = False
_injected = 0


@when_imported("vllm.v1.worker.gpu_worker")
def register_evidence_method(module):
    from engine_compat_hybrid_probe import worker_evidence

    # String-method RPC retains vLLM's safe serialization defaults.
    module.Worker.hybrid_probe_evidence = worker_evidence


def marker(event, **data):
    record = dict(event=event, pid=os.getpid(), **data)
    with open(os.environ["HYBRID_MARKER"], "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print("HYBRID_MARKER " + json.dumps(record), flush=True)


@when_imported("vllm.v1.core.single_type_kv_cache_manager")
def observe_partial_hit(module):
    original = module.MambaManager.allocate_new_blocks

    @functools.wraps(original)
    def allocate(self, request_id, *args, **kwargs):
        global _armed
        previous = _armed
        _armed = request_id in self._partial_hit_reqs
        if _armed:
            marker(
                "partial_hit",
                request_id=request_id,
                hash_block_size=self.block_pool.hash_block_size,
                allocation_block_size=self.block_size,
            )
        try:
            return original(self, request_id, *args, **kwargs)
        finally:
            _armed = previous

    module.MambaManager.allocate_new_blocks = allocate


@when_imported("kvcached.kv_cache_manager")
def inject_capacity_failure(module):
    original = module.KVCacheManager.alloc

    @functools.wraps(original)
    def allocate(self, *args, **kwargs):
        global _injected
        if _armed and _injected < 2 and os.getenv("HYBRID_FAULT") == "cow-admission":
            _injected += 1
            marker("cow_admission_miss", hit=_injected)
            return None
        return original(self, *args, **kwargs)

    module.KVCacheManager.alloc = allocate


@when_imported("kvcached.integration.vllm.interfaces")
def observe_v1_copy(module):
    original = module._copy_kv_cache_blocks

    @functools.wraps(original)
    def copy(caches, num_blocks, pairs, *args, **kwargs):
        result = original(caches, num_blocks, pairs, *args, **kwargs)
        if pairs:
            marker("copy_submitted", runner="v1", count=len(pairs))
        return result

    module._copy_kv_cache_blocks = copy


@when_imported("vllm.v1.worker.gpu.model_runner")
def observe_v2_copy(module):
    original = module.copy_kv_cache_blocks_inplace

    @functools.wraps(original)
    def copy(caches, num_blocks, pairs, *args, **kwargs):
        result = original(caches, num_blocks, pairs, *args, **kwargs)
        if pairs:
            marker("copy_submitted", runner="v2", count=len(pairs))
        return result

    module.copy_kv_cache_blocks_inplace = copy

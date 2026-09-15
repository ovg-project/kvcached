# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""vLLM 0.28-only V2 cache initialization using the legacy storage helpers."""

from functools import wraps
from types import ModuleType, SimpleNamespace
from typing import Any

from kvcached.integration.patch_base import enable_kvcached
from kvcached.integration.version_utils import VersionRange
from kvcached.integration.vllm.patches import (
    GPUModelRunnerPatch,
    _is_attention_spec,
)

VLLM_028_MRV2_RANGE = ">=0.28.0,<0.29.0.dev0"


def _initialize_kvcached_worker(runner: Any) -> None:
    from vllm.distributed.parallel_state import (
        get_pp_group,
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    from kvcached.integration.vllm import interfaces as kvi
    from kvcached.integration.vllm.patches import _should_enable_async_sched

    kvi.init_kvcached(
        tp_rank=get_tensor_model_parallel_rank(),
        world_size=get_tensor_model_parallel_world_size(),
        pp_rank=get_pp_group().rank_in_group,
        is_worker=True,
        device=str(runner.device),
        async_sched=_should_enable_async_sched(runner.vllm_config),
    )


class GPUModelRunnerV2Patch(GPUModelRunnerPatch):
    target_module = "vllm.v1.worker.gpu.model_runner"
    patch_name = "gpu_model_runner_v2"

    def apply(self, target_module: ModuleType) -> bool:
        if not self.initialize_version_info():
            return False
        if self.detected_version is None or not VersionRange(VLLM_028_MRV2_RANGE).contains(
            self.detected_version
        ):
            return False
        runner_class = self._get_target_class(target_module)
        original_cache_init = getattr(target_module, "init_kv_cache", None)
        if runner_class is None or original_cache_init is None:
            return False
        if self._is_already_patched(original_cache_init, "v2_init_kv_cache"):
            return True

        # V2 passes cache inputs to a module-level function, not runner methods.
        # Keep one context per invocation so temporary/final caches cannot share
        # layout metadata, while reusing the V1 allocation and view implementation.
        class CacheContext(SimpleNamespace):
            pass

        if not (self.add_kvcache_allocator(CacheContext)
                and self.add_reshape_methods(CacheContext)):
            return False

        original_runner_init = runner_class.__init__

        @wraps(original_runner_init)
        def _runner_init(runner: Any, *args: Any, **kwargs: Any) -> None:
            original_runner_init(runner, *args, **kwargs)
            if enable_kvcached():
                _initialize_kvcached_worker(runner)

        @wraps(original_cache_init)
        def _cache_init(
            runner_kv_caches: Any,
            forward_context: Any,
            kv_cache_config: Any,
            attn_groups: Any,
            device: Any,
            cache_dtype: str,
            kernel_block_sizes: Any,
            vllm_config: Any,
        ) -> Any:
            if not enable_kvcached() or not kv_cache_config.kv_cache_groups:
                return original_cache_init(
                    runner_kv_caches, forward_context, kv_cache_config,
                    attn_groups, device, cache_dtype, kernel_block_sizes,
                    vllm_config,
                )

            from vllm.v1.worker.gpu.attn_utils import (
                bind_kv_cache,
                get_shared_kv_cache_layers,
            )

            if any(not _is_attention_spec(group.kv_cache_spec)
                   for group in kv_cache_config.kv_cache_groups):
                raise NotImplementedError(
                    "KVCached Model Runner V2 currently supports attention-only "
                    "KV groups; hybrid/Mamba lifecycle support is not enabled."
                )
            if get_shared_kv_cache_layers(vllm_config):
                raise NotImplementedError(
                    "KVCached Model Runner V2 does not yet support cross-layer "
                    "KV sharing."
                )

            context = CacheContext(
                device=device,
                cache_config=SimpleNamespace(cache_dtype=cache_dtype),
                attn_groups=attn_groups,
                _kernel_block_sizes=kernel_block_sizes,
            )
            raw = context._allocate_kv_cache_from_kvcached(kv_cache_config)
            views = context._reshape_kv_cache_tensors_from_kvcached(
                kv_cache_config, raw,
            )
            num_attn_module = (
                2 if vllm_config.model_config.hf_config.model_type
                in ("longcat_flash", "longcat_flash_ngram") else 1
            )
            bind_kv_cache(views, forward_context, runner_kv_caches, num_attn_module)
            return views

        self._mark_as_patched(_cache_init, "v2_init_kv_cache")
        runner_class.__init__ = _runner_init
        setattr(target_module, "init_kv_cache", _cache_init)
        return True

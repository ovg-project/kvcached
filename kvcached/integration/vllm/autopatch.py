# SPDX-FileCopyrightText: 2025 contributors to the kvcached project
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import os
import types
from typing import Any, Callable, Iterable, Optional

from kvcached.utils import get_kvcached_logger

try:
    from wrapt.importer import when_imported  # type: ignore
except Exception:

    def when_imported(module_name: str):  # type: ignore

        def decorator(func: Callable[[types.ModuleType], None]):
            try:
                mod = importlib.import_module(module_name)
            except Exception:
                return func
            func(mod)
            return func

        return decorator


logger = get_kvcached_logger()


def _env_enabled() -> bool:
    return os.getenv("KVCACHED_AUTOPATCH", "false").lower() in ("true", "1")


def _enable_kvcached() -> bool:
    return os.getenv("ENABLE_KVCACHED", "false").lower() in ("true", "1")


@when_imported("vllm")
def _patch_vllm(_vllm: types.ModuleType) -> None:
    if not _env_enabled():
        logger.debug("Disabled by KVCACHED_AUTOPATCH")
        return

    # Track patch success status
    patch_status = {
        "elastic_block_pool": False,
        "engine_core": False,
        "kv_cache_coordinator": False,
        "gpu_model_runner": False,
        "gpu_worker": False,
    }

    # Lazily import submodules we need to patch to avoid import-order issues
    logger.info("Patching vllm with kvcached support")
    try:
        block_pool_mod = importlib.import_module("vllm.v1.core.block_pool")
        engine_mod = importlib.import_module("vllm.v1.engine.core")
        kvcoord_mod = importlib.import_module(
            "vllm.v1.core.kv_cache_coordinator")
        gpumr_mod = importlib.import_module("vllm.v1.worker.gpu_model_runner")
    except Exception:
        logger.error("Failed to import vllm modules, check if a supported "
                     "vllm version is installed.")
        return

    # Apply patches and track success
    patch_status["elastic_block_pool"] = _inject_elastic_block_pool(
        block_pool_mod)
    patch_status["kv_cache_coordinator"] = _patch_kv_cache_coordinator(
        kvcoord_mod, block_pool_mod)
    patch_status["engine_core"] = _patch_engine_core(engine_mod)
    patch_status["gpu_model_runner"] = _patch_gpu_model_runner(gpumr_mod)
    patch_status["gpu_worker"] = _patch_gpu_worker(importlib.import_module("vllm.v1.worker.gpu_worker"))

    # Log overall status
    successful_patches = [name for name, succ in patch_status.items() if succ]
    failed_patches = [name for name, succ in patch_status.items() if not succ]

    if successful_patches:
        logger.info("Successfully patched %s", ", ".join(successful_patches))
    if failed_patches:
        logger.warning("Failed to patch %s", ", ".join(failed_patches))


def _inject_elastic_block_pool(block_pool_mod: types.ModuleType) -> bool:
    if hasattr(block_pool_mod, "ElasticBlockPool"):
        return True

    BlockPool = getattr(block_pool_mod, "BlockPool")
    KVCacheBlock = getattr(block_pool_mod, "KVCacheBlock")
    BlockHash = getattr(block_pool_mod, "BlockHash")

    class ElasticBlockPool(BlockPool):  # type: ignore
        """ElasticBlockPool that manages KVCacheBlocks using kvcached."""

        def __init__(
            self,
            num_gpu_blocks: int,
            block_size: int,
            cell_size: int,
            num_layers: int,
            enable_caching: bool,
            enable_kv_cache_events: bool = False,
        ) -> None:
            assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
            assert not enable_caching, "Caching is not supported in ElasticBlockPool"
            assert not enable_kv_cache_events, (
                "KV cache events are not supported in ElasticBlockPool")

            self.num_gpu_blocks = num_gpu_blocks
            self.enable_kv_cache_events = enable_kv_cache_events
            self.kv_event_queue = []  # type: ignore[var-annotated]

            from kvcached.integration.vllm.interfaces import get_kv_cache_manager
            self.kv_cache_manager = get_kv_cache_manager(
                num_gpu_blocks, block_size, cell_size, num_layers)

            self.null_block = None  # type: ignore

        def get_cached_block(
            self,
            block_hash: BlockHash,  # type: ignore[valid-type]
            kv_cache_group_ids: list[int]
        ) -> Optional[list[KVCacheBlock]]:  # type: ignore[valid-type]
            return None

        def cache_full_blocks(
                self,
                request: "Request",  # type: ignore[name-defined] # noqa: F821
                blocks: list[KVCacheBlock],  # type: ignore[valid-type]
                block_hashes: list[BlockHash],  # type: ignore[valid-type]
                num_cached_blocks: int,
                num_full_blocks: int,
                block_size: int,
                kv_cache_group_id: int,
                hash_fn: Callable) -> None:
            raise NotImplementedError(
                "Caching is not supported in ElasticBlockPool")

        def get_new_blocks(
                self, num_blocks: int
        ) -> list[KVCacheBlock]:  # type: ignore[valid-type]
            if num_blocks > self.get_num_free_blocks():
                raise ValueError(
                    f"Cannot get {num_blocks} free blocks from the pool")

            block_ids = self.kv_cache_manager.alloc(num_blocks)
            assert block_ids is not None and len(block_ids) == num_blocks

            return [KVCacheBlock(bid) for bid in block_ids]

        def touch(
            self,
            blocks: tuple[list[KVCacheBlock], ...]  # type: ignore[valid-type]
        ) -> None:
            raise NotImplementedError("Not supported in ElasticBlockPool")

        def free_blocks(
            self,
            ordered_blocks: Iterable[KVCacheBlock]  # type: ignore[valid-type]
        ) -> None:  # type: ignore[valid-type]
            block_ids = [
                block.block_id  # type: ignore[attr-defined]
                for block in ordered_blocks
            ]
            if len(block_ids) > 0:
                self.kv_cache_manager.free(block_ids)

        def reset_prefix_cache(self) -> bool:
            raise NotImplementedError("Not supported in ElasticBlockPool")

        def get_num_free_blocks(self) -> int:
            return self.kv_cache_manager.available_size()

        def get_usage(self) -> float:
            return 1.0 - (self.get_num_free_blocks() / self.num_gpu_blocks)

        def take_events(
            self,
        ) -> list["KVCacheEvent"]:  # type: ignore[name-defined] # noqa: F821
            return []

    setattr(block_pool_mod, "ElasticBlockPool", ElasticBlockPool)
    return True


def _patch_engine_core(engine_mod: types.ModuleType) -> bool:
    EngineCore = getattr(engine_mod, "EngineCore", None)
    if EngineCore is None:
        return False

    original_init = EngineCore.__init__

    def _patched_engine_init(self, vllm_config, *args: Any, **kwargs: Any):
        import os
        enable_kvcached = os.getenv("ENABLE_KVCACHED",
                                    "false").lower() == "true"
        if enable_kvcached:
            try:
                from kvcached.integration.vllm.interfaces import init_kvcached
                init_kvcached(
                    tp_rank=0,
                    tp_size=vllm_config.parallel_config.tensor_parallel_size,
                    is_worker=False,
                )
            except Exception:
                pass
        return original_init(self, vllm_config, *args, **kwargs)

    if getattr(EngineCore.__init__, "__kvcached_patched__", False) is not True:
        _patched_engine_init.__kvcached_patched__ = True  # type: ignore[attr-defined]
        EngineCore.__init__ = _patched_engine_init  # type: ignore[assignment]
    return True


def _patch_kv_cache_coordinator(kvcoord_mod: types.ModuleType,
                                block_pool_mod: types.ModuleType) -> bool:
    KVCacheCoordinator = getattr(kvcoord_mod, "KVCacheCoordinator", None)
    if KVCacheCoordinator is None:
        return False

    original_init = KVCacheCoordinator.__init__

    def _patched_init(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)

        if not _enable_kvcached():
            return

        try:
            enable_caching = getattr(self, "enable_caching", False)
            if enable_caching:
                raise ValueError("Caching is not supported for kvcached")

            kv_cache_config = getattr(self, "kv_cache_config")
            kv_groups = kv_cache_config.kv_cache_groups
            if len(kv_groups) != 1:
                raise ValueError(
                    "Only one kv cache group is supported for kvcached")

            kv_cache_group = kv_groups[0]
            kv_cache_spec = kv_cache_group.kv_cache_spec
            block_size = kv_cache_spec.block_size
            cell_size = kv_cache_spec.page_size_bytes // block_size // 2

            try:
                from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
                tp_size = int(get_tensor_model_parallel_world_size())
            except Exception:
                tp_size = 1

            from kvcached.integration.vllm import interfaces as kvi
            kvi.init_kvcached(tp_rank=0, tp_size=tp_size, is_worker=False)

            ElasticBlockPool = getattr(block_pool_mod, "ElasticBlockPool")
            num_layers = len(getattr(kv_cache_config, "kv_cache_tensors"))
            self.block_pool = ElasticBlockPool(
                kv_cache_config.num_blocks,
                block_size,
                cell_size=cell_size,
                num_layers=num_layers,
                enable_caching=getattr(self, "enable_caching", False),
            )
        except Exception:
            logger.warning("Failed to patch kv_cache_coordinator")
            return

    if getattr(KVCacheCoordinator.__init__, "__kvcached_patched__",
               False) is not True:
        _patched_init.__kvcached_patched__ = True  # type: ignore[attr-defined]
        KVCacheCoordinator.__init__ = _patched_init  # type: ignore[assignment]

    return True


def _patch_gpu_model_runner(gpumr_mod: types.ModuleType) -> bool:
    GPUModelRunner = getattr(gpumr_mod, "GPUModelRunner", None)
    if GPUModelRunner is None:
        return False

    # Patch __init__ to initialize kvcached in workers if enabled
    original_init = GPUModelRunner.__init__

    def _patched_mr_init(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)

        self.enable_kvcached = _enable_kvcached()
        if not self.enable_kvcached:
            return

        try:
            from vllm.distributed.parallel_state import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )
            tp_rank = int(get_tensor_model_parallel_rank())
            tp_size = int(get_tensor_model_parallel_world_size())
        except Exception:
            tp_rank, tp_size = 0, 1

        try:
            device_str = str(getattr(self, "device", "cuda"))
        except Exception:
            device_str = "cuda"

        try:
            from kvcached.integration.vllm import interfaces as kvi
            kvi.init_kvcached(tp_rank=tp_rank,
                              tp_size=tp_size,
                              is_worker=True,
                              device=device_str)
        except Exception as e:
            # Fail open
            logger.warning("Failed to initialize kvcached, disabling: %s", e)
            self.enable_kvcached = False

    if getattr(GPUModelRunner.__init__, "__kvcached_patched__",
               False) is not True:
        _patched_mr_init.__kvcached_patched__ = True  # type: ignore[attr-defined]
        GPUModelRunner.__init__ = _patched_mr_init  # type: ignore[assignment]

    # Add our allocator method to the class if not present
    if not hasattr(GPUModelRunner, "_allocate_kv_cache_from_kvcached"):

        def _allocate_kv_cache_from_kvcached(self, kv_cache_config):
            import torch
            from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheTensor

            if len(kv_cache_config.kv_cache_groups) > 1:
                raise NotImplementedError(
                    "Hybrid models with more than one KV cache type are not supported yet."
                )

            kv_cache_group = kv_cache_config.kv_cache_groups[0]
            kv_cache_spec = kv_cache_group.kv_cache_spec
            if not isinstance(kv_cache_spec, FullAttentionSpec):
                raise ValueError(
                    "kvcached only supports FullAttentionSpec layers")

            layer_to_tensor_cfg: dict[str, KVCacheTensor] = {}
            for tensor_cfg in kv_cache_config.kv_cache_tensors:
                for ln in tensor_cfg.shared_by:
                    layer_to_tensor_cfg[ln] = tensor_cfg

            for layer_name in kv_cache_group.layer_names:
                tensor_cfg = layer_to_tensor_cfg[layer_name]
                assert (
                    tensor_cfg.size % kv_cache_spec.page_size_bytes == 0
                ), (f"Tensor size for layer {layer_name} ({tensor_cfg.size}) "
                    "is not a multiple of page size "
                    f"{kv_cache_spec.page_size_bytes}.")
                num_blocks = tensor_cfg.size // kv_cache_spec.page_size_bytes
                assert num_blocks >= kv_cache_config.num_blocks, (
                    "Number of blocks derived from tensor size is smaller than "
                    "kv_cache_config.num_blocks")

            first_layer_name = kv_cache_group.layer_names[0]
            rep_tensor_cfg = layer_to_tensor_cfg[first_layer_name]
            num_blocks = rep_tensor_cfg.size // kv_cache_spec.page_size_bytes

            attn_backend_cls = self.attn_backends[0]
            kv_cache_shape = attn_backend_cls.get_kv_cache_shape(
                num_blocks,
                kv_cache_spec.block_size,
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.head_size,
            )

            num_layers = len(kv_cache_group.layer_names)
            dtype = kv_cache_spec.dtype

            from kvcached.integration.vllm import interfaces as kvi
            kv_cache_raw_tensors = kvi.alloc_kv_cache(
                kv_cache_shape,
                kv_cache_spec.block_size,
                dtype,
                getattr(self, "device", torch.device("cuda")).type,
                num_layers,
                attention_type="MHA",
                kv_layout="NHD",
            )
            return kv_cache_raw_tensors

        setattr(GPUModelRunner, "_allocate_kv_cache_from_kvcached",
                _allocate_kv_cache_from_kvcached)

    # Find the method that allocates and returns kv_caches and wrap it
    if hasattr(GPUModelRunner, "_allocate_kv_cache_tensors"):
        original_method = getattr(GPUModelRunner, "_allocate_kv_cache_tensors")

        def _patched_alloc_kv(self, kv_cache_config, *args: Any,
                              **kwargs: Any):
            if getattr(self, "enable_kvcached", False):
                return self._allocate_kv_cache_from_kvcached(kv_cache_config)
            return original_method(self, kv_cache_config, *args, **kwargs)

        if getattr(original_method, "__kvcached_patched__", False) is not True:
            _patched_alloc_kv.__kvcached_patched__ = True  # type: ignore[attr-defined]
            setattr(GPUModelRunner, "_allocate_kv_cache_tensors",
                    _patched_alloc_kv)

    if not hasattr(GPUModelRunner, "_reshape_kv_cache_tensors_from_kvcached"):

        def _reshape_kv_cache_tensors_from_kvcached(self, kv_cache_config,
                                                    kv_cache_raw_tensors):
            import torch
            kv_caches: dict[str, torch.Tensor] = {}
            kv_cache_group = kv_cache_config.kv_cache_groups[0]
            for idx, layer_name in enumerate(kv_cache_group.layer_names):
                kv_caches[layer_name] = kv_cache_raw_tensors[idx]
            return kv_caches

        setattr(GPUModelRunner, "_reshape_kv_cache_tensors_from_kvcached",
                _reshape_kv_cache_tensors_from_kvcached)

    if hasattr(GPUModelRunner, "_reshape_kv_cache_tensors"):
        original_method = getattr(GPUModelRunner, "_reshape_kv_cache_tensors")

        def _patched_reshape_kv(self, kv_cache_config, kv_cache_raw_tensors):
            if getattr(self, "enable_kvcached", False):
                return self._reshape_kv_cache_tensors_from_kvcached(
                    kv_cache_config, kv_cache_raw_tensors)
            return original_method(self, kv_cache_config, kv_cache_raw_tensors)

        if getattr(original_method, "__kvcached_patched__", False) is not True:
            _patched_reshape_kv.__kvcached_patched__ = True  # type: ignore[attr-defined]
            setattr(GPUModelRunner, "_reshape_kv_cache_tensors",
                    _patched_reshape_kv)

    return True


def _patch_gpu_worker(gpuworker_mod: types.ModuleType) -> bool:
    """Patch Worker.init_device to ignore GPU free-memory check when kvcached
    is enabled.
    """

    Worker = getattr(gpuworker_mod, "Worker", None)
    if Worker is None:
        return False

    original_init_device = Worker.init_device

    def _patched_init_device(self, *args: Any, **kwargs: Any):  # type: ignore[no-self-use]
        self.enable_kvcached = _enable_kvcached()
        if not self.enable_kvcached:
            return

        try:
            return original_init_device(self, *args, **kwargs)
        except ValueError as e:
            # If the original impl still raises due to insufficient memory,
            # replicate the remainder of its logic while skipping the guard.
            logger.warning("Ignoring GPU free-memory check: %s", e)

            # The steps below mirror the tail of vLLM's Worker.init_device
            # after the memory-utilization check.
            try:
                from vllm.model_executor import set_random_seed  # type: ignore
                from vllm.v1.utils import report_usage_stats  # type: ignore
                from vllm.v1.worker.gpu_model_runner import GPUModelRunner
                from vllm.v1.worker.gpu_worker import (
                    init_worker_distributed_environment as _init_dist_env,
                )
            except Exception:
                logger.warning("Unable to import vLLM helpers; re-raising OOM")
                raise

            _init_dist_env(self.vllm_config, self.rank,
                           self.distributed_init_method, self.local_rank)
            set_random_seed(self.model_config.seed)
            self.model_runner = GPUModelRunner(self.vllm_config, self.device)  # type: ignore[attr-defined]
            if getattr(self, "rank", None) == 0:
                report_usage_stats(self.vllm_config)

            return None

    if getattr(Worker.init_device, "__kvcached_patched__", False) is not True:  # type: ignore[attr-defined]
        _patched_init_device.__kvcached_patched__ = True  # type: ignore[attr-defined]
        Worker.init_device = _patched_init_device  # type: ignore[assignment]

    return True

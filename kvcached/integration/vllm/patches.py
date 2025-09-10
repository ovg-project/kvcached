"""
vLLM-specific patches using unified patch infrastructure.
"""

import types
from typing import Any, Callable, Iterable, Optional

from kvcached.integration.patch_base import BasePatch, enable_kvcached


class ElasticBlockPoolPatch(BasePatch):
    """Inject ElasticBlockPool into vLLM's block pool module"""

    library = "vllm"
    target_module = "vllm.v1.core.block_pool"
    patch_name = "elastic_block_pool"

    def apply(self, block_pool_mod: types.ModuleType) -> bool:
        if hasattr(block_pool_mod, "ElasticBlockPool"):
            self.logger.debug("ElasticBlockPool already exists")
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


class EngineCorePatch(BasePatch):
    """Patch EngineCore.__init__ to initialize kvcached"""

    library = "vllm"
    target_module = "vllm.v1.engine.core"
    target_class = "EngineCore"
    patch_name = "engine_core"

    def apply(self, engine_mod: types.ModuleType) -> bool:
        EngineCore = self._get_target_class(engine_mod)
        if EngineCore is None:
            return False

        if self._is_already_patched(EngineCore.__init__):
            self.logger.debug("EngineCore.__init__ already patched")
            return True

        original_init = EngineCore.__init__

        def _patched_engine_init(self, vllm_config, *args: Any, **kwargs: Any):
            if enable_kvcached():
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

        self._mark_as_patched(_patched_engine_init)
        EngineCore.__init__ = _patched_engine_init  # type: ignore[assignment]
        return True


class KVCacheCoordinatorPatch(BasePatch):
    """Patch KVCacheCoordinator to use ElasticBlockPool"""

    library = "vllm"
    target_module = "vllm.v1.core.kv_cache_coordinator"
    target_class = "KVCacheCoordinator"
    patch_name = "kv_cache_coordinator"

    def apply(self, kvcoord_mod: types.ModuleType) -> bool:
        KVCacheCoordinator = self._get_target_class(kvcoord_mod)
        if KVCacheCoordinator is None:
            return False

        if self._is_already_patched(KVCacheCoordinator.__init__):
            self.logger.debug("KVCacheCoordinator.__init__ already patched")
            return True

        original_init = KVCacheCoordinator.__init__
        logger = self.logger  # Capture logger in closure

        def _patched_init(self, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)

            if not enable_kvcached():
                return

            try:
                self._setup_kvcached_coordinator()
            except Exception:
                logger.warning("Failed to patch kv_cache_coordinator")
                return

        def _setup_kvcached_coordinator(self) -> None:
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

            # Import ElasticBlockPool from the patched module
            import importlib
            block_pool_mod = importlib.import_module("vllm.v1.core.block_pool")
            ElasticBlockPool = getattr(block_pool_mod, "ElasticBlockPool")

            num_layers = len(getattr(kv_cache_config, "kv_cache_tensors"))
            self.block_pool = ElasticBlockPool(
                kv_cache_config.num_blocks,
                block_size,
                cell_size=cell_size,
                num_layers=num_layers,
                enable_caching=getattr(self, "enable_caching", False),
            )

        # Add helper methods to the instance
        KVCacheCoordinator._setup_kvcached_coordinator = _setup_kvcached_coordinator

        self._mark_as_patched(_patched_init)
        KVCacheCoordinator.__init__ = _patched_init  # type: ignore[assignment]
        return True


class GPUModelRunnerPatch(BasePatch):
    """Patch GPUModelRunner for kvcached integration"""

    library = "vllm"
    target_module = "vllm.v1.worker.gpu_model_runner"
    target_class = "GPUModelRunner"
    patch_name = "gpu_model_runner"

    def apply(self, gpumr_mod: types.ModuleType) -> bool:
        GPUModelRunner = self._get_target_class(gpumr_mod)
        if GPUModelRunner is None:
            return False

        success = True
        success &= self._patch_init(GPUModelRunner)
        success &= self._add_kvcache_allocator(GPUModelRunner)
        success &= self._patch_allocation_methods(GPUModelRunner)
        success &= self._add_reshape_methods(GPUModelRunner)
        success &= self._patch_reshape_methods(GPUModelRunner)

        return success

    def _patch_init(self, GPUModelRunner) -> bool:
        """Patch __init__ to initialize kvcached in workers if enabled"""
        if self._is_already_patched(GPUModelRunner.__init__, "__kvcached_patched_mr_init__"):
            return True

        original_init = GPUModelRunner.__init__
        logger = self.logger  # Capture logger in closure

        def _patched_mr_init(self, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)

            if not enable_kvcached():
                return

            try:
                self._init_kvcached()
            except Exception as e:
                logger.warning("Failed to initialize kvcached, disabling: %s", e)

        def _init_kvcached(self) -> None:
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

            from kvcached.integration.vllm import interfaces as kvi
            kvi.init_kvcached(tp_rank=tp_rank,
                              tp_size=tp_size,
                              is_worker=True,
                              device=device_str)

        # Add helper methods to the class
        GPUModelRunner._init_kvcached = _init_kvcached

        self._mark_as_patched(_patched_mr_init, "__kvcached_patched_mr_init__")
        GPUModelRunner.__init__ = _patched_mr_init  # type: ignore[assignment]
        return True

    def _add_kvcache_allocator(self, GPUModelRunner) -> bool:
        """Add kvcache allocation method to the class"""
        if hasattr(GPUModelRunner, "_allocate_kv_cache_from_kvcached"):
            return True

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
        return True

    def _patch_allocation_methods(self, GPUModelRunner) -> bool:
        """Patch the allocation methods to use kvcached when enabled"""
        if not hasattr(GPUModelRunner, "_allocate_kv_cache_tensors"):
            return True

        original_method = getattr(GPUModelRunner, "_allocate_kv_cache_tensors")
        if self._is_already_patched(original_method, "__kvcached_patched_alloc__"):
            return True

        def _patched_alloc_kv(self, kv_cache_config, *args: Any, **kwargs: Any):
            if enable_kvcached():
                return self._allocate_kv_cache_from_kvcached(kv_cache_config)
            return original_method(self, kv_cache_config, *args, **kwargs)

        self._mark_as_patched(_patched_alloc_kv, "__kvcached_patched_alloc__")
        setattr(GPUModelRunner, "_allocate_kv_cache_tensors", _patched_alloc_kv)
        return True

    def _add_reshape_methods(self, GPUModelRunner) -> bool:
        """Add kvcache reshape method to the class"""
        if hasattr(GPUModelRunner, "_reshape_kv_cache_tensors_from_kvcached"):
            return True

        def _reshape_kv_cache_tensors_from_kvcached(self, kv_cache_config, kv_cache_raw_tensors):
            import torch
            kv_caches: dict[str, torch.Tensor] = {}
            kv_cache_group = kv_cache_config.kv_cache_groups[0]
            for idx, layer_name in enumerate(kv_cache_group.layer_names):
                kv_caches[layer_name] = kv_cache_raw_tensors[idx]
            return kv_caches

        setattr(GPUModelRunner, "_reshape_kv_cache_tensors_from_kvcached",
                _reshape_kv_cache_tensors_from_kvcached)
        return True

    def _patch_reshape_methods(self, GPUModelRunner) -> bool:
        """Patch the reshape methods to use kvcached when enabled"""
        if not hasattr(GPUModelRunner, "_reshape_kv_cache_tensors"):
            return True

        original_method = getattr(GPUModelRunner, "_reshape_kv_cache_tensors")
        if self._is_already_patched(original_method, "__kvcached_patched_reshape__"):
            return True

        def _patched_reshape_kv(self, kv_cache_config, kv_cache_raw_tensors):
            if enable_kvcached():
                return self._reshape_kv_cache_tensors_from_kvcached(
                    kv_cache_config, kv_cache_raw_tensors)
            return original_method(self, kv_cache_config, kv_cache_raw_tensors)

        self._mark_as_patched(_patched_reshape_kv, "__kvcached_patched_reshape__")
        setattr(GPUModelRunner, "_reshape_kv_cache_tensors", _patched_reshape_kv)
        return True


class GPUWorkerPatch(BasePatch):
    """Patch Worker.init_device to ignore GPU free-memory check when kvcached is enabled"""

    library = "vllm"
    target_module = "vllm.v1.worker.gpu_worker"
    target_class = "Worker"
    patch_name = "gpu_worker"

    def apply(self, gpuworker_mod: types.ModuleType) -> bool:
        Worker = self._get_target_class(gpuworker_mod)
        if Worker is None:
            return False

        if self._is_already_patched(Worker.init_device):
            self.logger.debug("Worker.init_device already patched")
            return True

        original_init_device = Worker.init_device
        logger = self.logger  # Capture logger in closure

        def _patched_init_device(self, *args: Any, **kwargs: Any):  # type: ignore[no-self-use]
            if not enable_kvcached():
                return original_init_device(self, *args, **kwargs)

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

        self._mark_as_patched(_patched_init_device)
        Worker.init_device = _patched_init_device  # type: ignore[assignment]
        return True
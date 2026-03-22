# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
vLLM-specific patches using unified patch infrastructure.
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any, Iterable, Optional

from kvcached.integration.patch_base import BasePatch, enable_kvcached
from kvcached.integration.version_utils import VersionAwarePatch, VersionRange, version_range

if TYPE_CHECKING:
    # These types are imported from vLLM at runtime via getattr()
    # Import them here for type checking only
    try:
        from vllm.v1.core.block_pool import KVCacheBlock  # type: ignore[import-untyped]
        from vllm.v1.core.block_pool import KVCacheEvent  # type: ignore[import-untyped]
        from vllm.v1.core.scheduler import Request  # type: ignore[import-untyped]
    except ImportError:
        # Fallback if vLLM is not available during type checking
        KVCacheBlock = Any  # type: ignore[misc,assignment]
        KVCacheEvent = Any  # type: ignore[misc,assignment]
        Request = Any  # type: ignore[misc,assignment]


def _validate_kv_cache_groups(kv_cache_config: Any) -> None:
    """Validate KV cache groups for kvcached compatibility.

    Checks that all groups use supported spec types (FullAttentionSpec,
    SlidingWindowSpec, or MLAAttentionSpec) and that all groups share the
    same block geometry (block_size and cell_size).  Raises ValueError on
    mismatch.
    """
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec, SlidingWindowSpec

    supported = (FullAttentionSpec, SlidingWindowSpec, MLAAttentionSpec)
    kv_groups = kv_cache_config.kv_cache_groups

    first_spec = kv_groups[0].kv_cache_spec
    block_size = first_spec.block_size
    cell_size, _ = _get_kv_cache_params(first_spec, block_size)

    for idx, grp in enumerate(kv_groups):
        grp_spec = grp.kv_cache_spec
        if not isinstance(grp_spec, supported):
            raise ValueError(
                f"kvcached only supports FullAttentionSpec, SlidingWindowSpec, "
                f"and MLAAttentionSpec, got {type(grp_spec).__name__} in group {idx}"
            )
        grp_block_size = grp_spec.block_size
        grp_cell_size, _ = _get_kv_cache_params(grp_spec, grp_block_size)
        if grp_block_size != block_size or grp_cell_size != cell_size:
            raise ValueError(
                "kvcached requires all KV cache groups to have the "
                f"same block geometry. Group 0: block_size={block_size},"
                f" cell_size={cell_size}; group {idx}: "
                f"block_size={grp_block_size}, cell_size={grp_cell_size}"
            )


def _count_kv_cache_layers(kv_cache_config: Any) -> int:
    """Return the total number of KV cache layers across all groups."""
    return sum(len(g.layer_names) for g in kv_cache_config.kv_cache_groups)


# Version ranges for vLLM support
VLLM_V8_RANGE = ">=0.8.4,<0.9.0"  # vLLM 0.8.x versions, need to cover 0.8.5.post1
VLLM_V9_PLUS_RANGE = ">=0.9.0"  # vLLM 0.9.x and 0.9+.x versions
VLLM_V9_RANGE = ">=0.9.0,<=0.9.2"  # vLLM 0.9.x versions
VLLM_V10_RANGE = ">0.9.2"  # vLLM 0.10.x+ versions, need to cover 0.10.0rc1
VLLM_ALL_RANGE = ">=0.8.4"  # All supported versions


def _get_kv_cache_params(kv_cache_spec: Any, block_size: int) -> tuple:
    """Determine cell_size and num_kv_buffers from a KV cache spec.

    Returns:
        (cell_size, num_kv_buffers)
    """
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    if isinstance(kv_cache_spec, MLAAttentionSpec):
        # MLA: single combined KV buffer per layer
        # page_size_bytes = block_size * num_kv_heads * head_size * dtype_size
        cell_size = kv_cache_spec.page_size_bytes // block_size
        num_kv_buffers = 1
    else:
        # MHA/GQA: separate K and V buffers
        # page_size_bytes = 2 * block_size * num_kv_heads * head_size * dtype_size
        cell_size = kv_cache_spec.page_size_bytes // block_size // 2
        num_kv_buffers = 2
    return cell_size, num_kv_buffers


class ElasticBlockPoolPatch(VersionAwarePatch, BasePatch):
    """Inject ElasticBlockPool into vLLM's block pool module"""

    library = "vllm"
    target_module = "vllm.v1.core.block_pool"
    patch_name = "elastic_block_pool"

    def apply(self, block_pool_mod: types.ModuleType) -> bool:
        # Initialize version info
        if not self.initialize_version_info():
            return False

        # Apply version-specific patches
        return self.inject_elastic_block_pool(block_pool_mod)

    @version_range(VLLM_ALL_RANGE)
    def inject_elastic_block_pool(self,
                                  block_pool_mod: types.ModuleType) -> bool:
        """Inject ElasticBlockPool"""
        if hasattr(block_pool_mod, "ElasticBlockPool"):
            self.logger.debug("ElasticBlockPool already exists")
            return True

        BlockPool = getattr(block_pool_mod, "BlockPool")
        KVCacheBlockClass = getattr(block_pool_mod, "KVCacheBlock")

        logger = self.logger  # Capture logger in closure

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
                num_kv_buffers: int = 2,
            ) -> None:
                assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
                self.enable_prefix_cache = enable_caching
                if enable_caching:
                    logger.info("Prefix caching enabled for ElasticBlockPool")

                assert not enable_kv_cache_events, (
                    "KV cache events are not supported in ElasticBlockPool")

                self.num_gpu_blocks = num_gpu_blocks
                self.enable_kv_cache_events = enable_kv_cache_events
                self.kv_event_queue = []  # type: ignore[var-annotated]

                from kvcached.integration.vllm.interfaces import get_kv_cache_manager

                self.kv_cache_manager = get_kv_cache_manager(
                    num_gpu_blocks, block_size, cell_size, num_layers,
                    num_kv_buffers=num_kv_buffers)

                self.null_block = None  # type: ignore

                # Prefix cache: hash -> KVCacheBlock object (direct lookup)
                self._cached_blocks: dict[Any, "KVCacheBlock"] = {}
                # Reverse index: block_id -> hash for O(1) eviction
                self._block_id_to_hash: dict[int, Any] = {}
                self._warned_multi_group = False

            def get_cached_block(
                self,
                block_hash: Any,
                kv_cache_group_ids: list[int]
            ) -> Optional[list["KVCacheBlock"]]:
                if not self.enable_prefix_cache:
                    return None

                if len(kv_cache_group_ids) > 1:
                    if not self._warned_multi_group:
                        logger.warning("ElasticBlockPool only supports single KV cache group, "
                                      f"got {len(kv_cache_group_ids)} groups")
                        self._warned_multi_group = True
                    return None

                block = self._cached_blocks.get(block_hash)
                if block is None:
                    return None

                return [block]

            def cache_full_blocks(
                self,
                request: "Request",
                blocks: list["KVCacheBlock"],
                num_cached_blocks: int,
                num_full_blocks: int,
                block_size: int,
                kv_cache_group_id: int,
            ) -> None:
                if not self.enable_prefix_cache:
                    return

                if num_cached_blocks >= num_full_blocks:
                    return

                new_full_blocks = blocks[num_cached_blocks:num_full_blocks]

                assert hasattr(request, 'block_hashes'), "Request missing block_hashes attribute"
                assert len(request.block_hashes) >= num_full_blocks, \
                    f"Request has {len(request.block_hashes)} hashes but need {num_full_blocks}"

                for i, block in enumerate(new_full_blocks):
                    if hasattr(block, 'is_null') and block.is_null:
                        continue

                    block_idx = num_cached_blocks + i
                    block_hash = request.block_hashes[block_idx]

                    # Already cached, idempotent
                    if block_hash in self._cached_blocks:
                        continue

                    self._cached_blocks[block_hash] = block
                    self._block_id_to_hash[block.block_id] = block_hash

            def get_new_blocks(
                self, num_blocks: int
            ) -> list["KVCacheBlock"]:
                if num_blocks > self.get_num_free_blocks():
                    raise ValueError(
                        f"Cannot get {num_blocks} free blocks from the pool")

                block_ids = self.kv_cache_manager.alloc(num_blocks)
                assert block_ids is not None and len(block_ids) == num_blocks

                return [KVCacheBlockClass(bid, ref_cnt=1) for bid in block_ids]

            def touch(
                self, blocks: list["KVCacheBlock"] | tuple[list["KVCacheBlock"], ...]
            ) -> None:
                if not self.enable_prefix_cache:
                    return
                if isinstance(blocks, tuple):
                    for block_list in blocks:
                        for block in block_list:
                            block.ref_cnt += 1
                else:
                    for block in blocks:
                        block.ref_cnt += 1

            def free_blocks(
                self,
                ordered_blocks: Iterable["KVCacheBlock"],
            ) -> None:
                if not self.enable_prefix_cache:
                    block_ids = [
                        block.block_id
                        for block in ordered_blocks
                        if block is not None
                    ]
                    if block_ids:
                        self.kv_cache_manager.free(block_ids)
                    return

                blocks_to_free = []
                for block in ordered_blocks:
                    if block is None:
                        continue
                    block.ref_cnt -= 1
                    if block.ref_cnt == 0:
                        block_id = block.block_id
                        blocks_to_free.append(block_id)
                if blocks_to_free:
                    # Remove freed blocks from cache via reverse index (O(1) per block)
                    for bid in blocks_to_free:
                        h = self._block_id_to_hash.pop(bid, None)
                        if h is not None:
                            self._cached_blocks.pop(h, None)
                    self.kv_cache_manager.free(blocks_to_free)

            def evict_blocks(self, block_ids: set[int]) -> None:
                if not self.enable_prefix_cache:
                    return

                # Remove from cache via reverse index (O(1) per block)
                removed = 0
                for bid in block_ids:
                    h = self._block_id_to_hash.pop(bid, None)
                    if h is not None:
                        self._cached_blocks.pop(h, None)
                        removed += 1

                if removed:
                    logger.debug(f"Evicted {removed} blocks from prefix cache")

            def reset_prefix_cache(self) -> bool:
                if not self.enable_prefix_cache:
                    return True

                self._cached_blocks.clear()
                self._block_id_to_hash.clear()
                logger.info("Prefix cache reset")
                return True

            def get_num_free_blocks(self) -> int:
                return self.kv_cache_manager.available_size()

            def get_usage(self) -> float:
                return 1.0 - (self.get_num_free_blocks() / self.num_gpu_blocks)

            def take_events(
                self,
            ) -> list["KVCacheEvent"]:
                return []

        setattr(block_pool_mod, "ElasticBlockPool", ElasticBlockPool)
        return True


class EngineCorePatch(VersionAwarePatch, BasePatch):
    """Patch EngineCore.__init__ to initialize kvcached"""

    library = "vllm"
    target_module = "vllm.v1.engine.core"
    target_class = "EngineCore"
    patch_name = "engine_core"

    def apply(self, engine_mod: types.ModuleType) -> bool:
        # Initialize version info
        if not self.initialize_version_info():
            return False

        # Apply version-specific patches
        return self.patch_engine_init(engine_mod)

    @version_range(VLLM_ALL_RANGE)
    def patch_engine_init(self, engine_mod: types.ModuleType) -> bool:
        """Patch EngineCore.__init__"""
        EngineCore = self._get_target_class(engine_mod)
        if EngineCore is None:
            return False

        if self._is_already_patched(EngineCore.__init__, "init"):
            self.logger.debug("EngineCore.__init__ already patched")
            return True

        original_init = EngineCore.__init__

        def _patched_engine_init(self, vllm_config, *args: Any, **kwargs: Any):
            if enable_kvcached():
                try:
                    from kvcached.integration.vllm.interfaces import init_kvcached

                    # IMPORTANT: use tp_size only, NOT tp_size * pp_size.
                    # The kvcached IPC mechanism coordinates KV tensor readiness
                    # within a single PP stage's TP group (w0.sock … w(tp-1).sock).
                    # Each PP stage manages its own KV memory independently, so
                    # cross-stage IPC is neither needed nor correct.
                    init_kvcached(
                        tp_rank=0,
                        world_size=vllm_config.parallel_config.tensor_parallel_size,
                        is_worker=False,
                    )
                except Exception:
                    pass
            return original_init(self, vllm_config, *args, **kwargs)

        self._mark_as_patched(_patched_engine_init, "init")
        EngineCore.__init__ = _patched_engine_init  # type: ignore[assignment]
        return True


class KVCacheCoordinatorPatch(VersionAwarePatch, BasePatch):
    """Patch KVCacheCoordinator to use ElasticBlockPool"""

    library = "vllm"
    target_module = "vllm.v1.core.kv_cache_coordinator"
    target_class = "KVCacheCoordinator"
    patch_name = "kv_cache_coordinator"

    def apply(self, kvcoord_mod: types.ModuleType) -> bool:
        # Initialize version info
        if not self.initialize_version_info():
            return False

        # Apply version-specific patches
        return self.patch_coordinator(kvcoord_mod)

    @version_range(VLLM_V9_PLUS_RANGE)
    def patch_coordinator(self, kvcoord_mod: types.ModuleType) -> bool:
        """Patch KVCacheCoordinator"""
        KVCacheCoordinator = self._get_target_class(kvcoord_mod)
        if KVCacheCoordinator is None:
            return False

        if self._is_already_patched(KVCacheCoordinator.__init__, "init"):
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
                logger.info("Prefix caching enabled for kvcached")

            kv_cache_config = getattr(self, "kv_cache_config")

            _validate_kv_cache_groups(kv_cache_config)

            # All groups validated to share the same block geometry,
            # so group 0's spec is representative.
            kv_cache_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
            block_size = kv_cache_spec.block_size

            cell_size, num_kv_buffers = _get_kv_cache_params(kv_cache_spec, block_size)

            try:
                from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size

                tp_size = int(get_tensor_model_parallel_world_size())
            except Exception:
                tp_size = 1

            from kvcached.integration.vllm import interfaces as kvi

            # Use tp_size (not TP*PP global world size) for the KVCacheManager world_size.
            # Each PP stage manages its own KV tensors independently. The IPC sockets
            # are registered per TP rank within each stage (w0.sock … w(tp_size-1).sock).
            kvi.init_kvcached(tp_rank=0, world_size=tp_size, is_worker=False)

            # Import ElasticBlockPool from the patched module
            import importlib

            block_pool_mod = importlib.import_module("vllm.v1.core.block_pool")
            ElasticBlockPool = getattr(block_pool_mod, "ElasticBlockPool")

            num_layers = _count_kv_cache_layers(kv_cache_config)
            self.block_pool = ElasticBlockPool(
                kv_cache_config.num_blocks,
                block_size,
                cell_size=cell_size,
                num_layers=num_layers,
                enable_caching=getattr(self, "enable_caching", False),
                num_kv_buffers=num_kv_buffers,
            )
            for manager in self.single_type_managers:
                manager.block_pool = self.block_pool
                manager._null_block = self.block_pool.null_block

        # Add helper methods to the instance
        KVCacheCoordinator._setup_kvcached_coordinator = _setup_kvcached_coordinator

        self._mark_as_patched(_patched_init, "init")
        KVCacheCoordinator.__init__ = _patched_init  # type: ignore[assignment]
        return True


class KVCacheManagerPatch(VersionAwarePatch, BasePatch):
    """Patch KVCacheManager to use ElasticBlockPool.

    Note: this patch targets vLLM 0.8.x only, which does not support hybrid
    models (multiple KV cache groups / SlidingWindowSpec).  Hybrid model
    support is handled by KVCacheCoordinatorPatch (v0.9+).
    """

    library = "vllm"
    target_module = "vllm.v1.core.kv_cache_manager"
    target_class = "KVCacheManager"
    patch_name = "kv_cache_manager"

    def apply(self, kvcache_manager_mod: types.ModuleType) -> bool:
        # Initialize version info
        if not self.initialize_version_info():
            return False

        # Apply version-specific patches
        return self.patch_kvcache_manager(kvcache_manager_mod)

    @version_range(VLLM_V8_RANGE)
    def patch_kvcache_manager(self, kvcache_manager_mod: types.ModuleType) -> bool:
        """Patch KVCacheManager"""
        import inspect

        KVCacheManager = self._get_target_class(kvcache_manager_mod)
        if KVCacheManager is None:
            return False

        if self._is_already_patched(KVCacheManager.__init__, "init"):
            self.logger.debug("KVCacheManager.__init__ already patched")
            return True

        original_init = KVCacheManager.__init__
        sig = inspect.signature(original_init)
        logger = self.logger  # Capture logger in closure

        def _patched_init(self, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)

            if not enable_kvcached():
                return

            try:
                bound_args = sig.bind(self, *args, **kwargs)
                bound_args.apply_defaults()
                kv_cache_config = bound_args.arguments.get("kv_cache_config")
                if kv_cache_config is None:
                    raise ValueError("kv_cache_config is required")

                self._setup_kvcached_manager(kv_cache_config)
            except Exception as e:
                logger.warning("Failed to patch kv_cache_manager: %s", e)
                return

        def _setup_kvcached_manager(self, kv_cache_config: Any) -> None:
            enable_caching = getattr(self, "enable_caching", False)
            if enable_caching:
                # v0.8 scheduler may not wire cache_full_blocks / block_hashes
                # the same way as v0.9+; disable until verified.
                logger.warning(
                    "Prefix caching not yet supported for kvcached on vLLM v0.8.x, disabling")
                enable_caching = False

            # Import ElasticBlockPool from the patched module
            import importlib

            block_pool_mod = importlib.import_module("vllm.v1.core.block_pool")
            ElasticBlockPool = getattr(block_pool_mod, "ElasticBlockPool")

            # Get required attributes from the manager instance
            # This is a bit hacky but simplest
            if hasattr(self, "get_kv_cache_spec"):
                kv_cache_spec = getattr(self, "get_kv_cache_spec")().items()[0][1]
            elif hasattr(self, "specialized_manager"):
                kv_cache_spec = getattr(self, "specialized_manager").kv_cache_spec
            else:
                raise ValueError(
                    "Unable to determine kv_cache_spec: expected get_kv_cache_spec or specialized_manager"
                )

            block_size = getattr(self, "block_size")
            num_gpu_blocks = getattr(self, "num_gpu_blocks")

            cell_size, num_kv_buffers = _get_kv_cache_params(kv_cache_spec, block_size)

            # Determine number of layers
            if hasattr(kv_cache_config, "tensors"):
                num_layers = len(kv_cache_config.tensors)
            elif hasattr(kv_cache_config, "kv_cache_tensors"):
                num_layers = len(kv_cache_config.kv_cache_tensors)
            else:
                raise ValueError(
                    "Unable to determine num_layers: expected tensors or kv_cache_tensors in kv_cache_config"
                )

            # Replace the block pool with ElasticBlockPool
            self.block_pool = ElasticBlockPool(
                num_gpu_blocks,
                block_size,
                cell_size=cell_size,
                num_layers=num_layers,
                enable_caching=enable_caching,
                num_kv_buffers=num_kv_buffers,
            )
            if hasattr(self, "specialized_manager"):
                self.specialized_manager.block_pool = self.block_pool
                if hasattr(self.specialized_manager, "_null_block"):
                    self.specialized_manager._null_block = self.block_pool.null_block

        # Add helper methods to the class
        KVCacheManager._setup_kvcached_manager = _setup_kvcached_manager

        self._mark_as_patched(_patched_init, "init")
        KVCacheManager.__init__ = _patched_init  # type: ignore[assignment]
        return True


class GPUModelRunnerPatch(VersionAwarePatch, BasePatch):
    """Patch GPUModelRunner for kvcached integration"""

    library = "vllm"
    target_module = "vllm.v1.worker.gpu_model_runner"
    target_class = "GPUModelRunner"
    patch_name = "gpu_model_runner"

    def apply(self, gpumr_mod: types.ModuleType) -> bool:
        # Initialize version info
        if not self.initialize_version_info():
            return False

        GPUModelRunner = self._get_target_class(gpumr_mod)
        if GPUModelRunner is None:
            return False

        # Apply all applicable version-specific patches
        success = True

        # Execute all applicable methods for this version
        for method in self.applicable_methods:
            try:
                method_success = method(GPUModelRunner)
                success &= method_success
                if method_success:
                    self.logger.debug(f"Applied {method.__name__}")
                else:
                    self.logger.warning(f"Failed to apply {method.__name__}")
            except Exception as e:
                self.logger.error(f"Error applying {method.__name__}: {e}")
                success = False

        return success

    @version_range(VLLM_ALL_RANGE)
    def patch_model_runner_init(self, GPUModelRunner) -> bool:
        """Patch __init__ to initialize kvcached in workers if enabled"""
        if self._is_already_patched(GPUModelRunner.__init__, "init"):
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
            # Get TP rank/size: these are always available at model runner init time.
            try:
                from vllm.distributed.parallel_state import (
                    get_tensor_model_parallel_rank,
                    get_tensor_model_parallel_world_size,
                )
                tp_rank = int(get_tensor_model_parallel_rank())
                tp_size = int(get_tensor_model_parallel_world_size())
            except (ImportError, AttributeError):
                tp_rank, tp_size = 0, 1

            # Try to get PP rank; it may not be available if PP process groups
            # initialise later, so default to 0 (works for PP=1 and PP stage 0).
            try:
                from vllm.distributed.parallel_state import (
                    get_pp_group,
                )
                pp_rank = int(get_pp_group().rank_in_group)
            except Exception:
                pp_rank = 0

            try:
                device_str = str(getattr(self, "device", "cuda"))
            except Exception:
                device_str = "cuda"

            from kvcached.integration.vllm import interfaces as kvi

            # Register this worker's IPC socket using tp_rank so all TP workers
            # within this PP stage listen on w0.sock … w(tp_size-1).sock.
            kvi.init_kvcached(tp_rank=tp_rank, world_size=tp_size, pp_rank=pp_rank,
                              is_worker=True, device=device_str)

        # Add helper methods to the class
        GPUModelRunner._init_kvcached = _init_kvcached

        self._mark_as_patched(_patched_mr_init, "init")
        GPUModelRunner.__init__ = _patched_mr_init  # type: ignore[assignment]
        return True

    @version_range(VLLM_V8_RANGE)
    def patch_initialize_kv_cache(self, GPUModelRunner) -> bool:
        """Patch __init__ to initialize kvcached in workers if enabled"""
        if self._is_already_patched(GPUModelRunner.initialize_kv_cache, "init_kv_cache"):
            return True

        original_initialize_kv_cache = GPUModelRunner.initialize_kv_cache

        def _patched_initialize_kv_cache(self, kv_cache_config: Any) -> None:
            import torch
            from vllm.v1.kv_cache_interface import MLAAttentionSpec
            from vllm.v1.utils import bind_kv_cache

            from kvcached.integration.vllm import interfaces as kvi

            if not enable_kvcached():
                return original_initialize_kv_cache(self, kv_cache_config)

            _validate_kv_cache_groups(kv_cache_config)

            kv_caches: dict[str, torch.Tensor] = {}
            for kv_cache_group in kv_cache_config.kv_cache_groups:
                kv_cache_spec = kv_cache_group.kv_cache_spec
                for layer_name in kv_cache_group.layer_names:
                    tensor_config = kv_cache_config.tensors[layer_name]
                    assert tensor_config.size % kv_cache_spec.page_size_bytes == 0
                    num_blocks = tensor_config.size // kv_cache_spec.page_size_bytes
                    assert num_blocks >= kv_cache_config.num_blocks

            num_layers = _count_kv_cache_layers(kv_cache_config)
            layer_name = list(kv_cache_config.tensors.keys())[0]
            # All groups validated to share the same block geometry by
            # _validate_kv_cache_groups, so group 0's spec is representative.
            kv_cache_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
            tensor_config = kv_cache_config.tensors[layer_name]

            is_mla = isinstance(kv_cache_spec, MLAAttentionSpec)
            dtype = kv_cache_spec.dtype
            num_blocks = tensor_config.size // kv_cache_spec.page_size_bytes
            assert num_blocks >= kv_cache_config.num_blocks
            kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                num_blocks,
                kv_cache_spec.block_size,
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.head_size,
            )

            kv_cache_buffers = kvi.alloc_kv_cache(
                kv_cache_shape,
                kv_cache_spec.block_size,
                dtype,
                self.device.type,
                num_layers,
                attention_type="MLA" if is_mla else "MHA",
                kv_layout="NHD",
            )
            layer_id = 0
            for kv_cache_group in kv_cache_config.kv_cache_groups:
                for layer_name in kv_cache_group.layer_names:
                    kv_caches[layer_name] = kv_cache_buffers[layer_id]
                    layer_id += 1

            bind_kv_cache(
                kv_caches,
                self.vllm_config.compilation_config.static_forward_context,
                self.kv_caches,
            )

        self._mark_as_patched(_patched_initialize_kv_cache, "init_kv_cache")
        GPUModelRunner.initialize_kv_cache = _patched_initialize_kv_cache  # type: ignore[assignment]
        return True

    @version_range(VLLM_V9_PLUS_RANGE)
    def add_kvcache_allocator(self, GPUModelRunner) -> bool:
        """Add kvcache allocation method to the class"""
        if hasattr(GPUModelRunner, "_allocate_kv_cache_from_kvcached"):
            return True

        # Capture patch instance for version-aware access
        patch_instance = self

        def _allocate_kv_cache_from_kvcached(self, kv_cache_config):
            import torch
            from vllm.v1.kv_cache_interface import KVCacheTensor, MLAAttentionSpec

            from kvcached.integration.vllm import interfaces as kvi

            _validate_kv_cache_groups(kv_cache_config)

            # All groups validated to share the same block geometry by
            # _validate_kv_cache_groups, so group 0's spec is representative.
            first_kv_cache_group = kv_cache_config.kv_cache_groups[0]
            kv_cache_spec = first_kv_cache_group.kv_cache_spec

            is_mla = isinstance(kv_cache_spec, MLAAttentionSpec)

            layer_to_tensor_cfg: dict[str, KVCacheTensor] = {}
            for tensor_cfg in kv_cache_config.kv_cache_tensors:
                for ln in tensor_cfg.shared_by:
                    layer_to_tensor_cfg[ln] = tensor_cfg

            for grp in kv_cache_config.kv_cache_groups:
                layer_spec = grp.kv_cache_spec
                for layer_name in grp.layer_names:
                    tensor_cfg = layer_to_tensor_cfg[layer_name]
                    assert tensor_cfg.size % layer_spec.page_size_bytes == 0, (
                        f"Tensor size for layer {layer_name} ({tensor_cfg.size}) "
                        "is not a multiple of page size "
                        f"{layer_spec.page_size_bytes}."
                    )
                    num_blocks = tensor_cfg.size // layer_spec.page_size_bytes
                    assert num_blocks >= kv_cache_config.num_blocks, (
                        "Number of blocks derived from tensor size is smaller than "
                        "kv_cache_config.num_blocks"
                    )

            first_layer_name = first_kv_cache_group.layer_names[0]
            rep_tensor_cfg = layer_to_tensor_cfg[first_layer_name]
            num_blocks = rep_tensor_cfg.size // kv_cache_spec.page_size_bytes

            # Use version-aware attention backend access
            attn_backend_cls = patch_instance._get_version_specific_attention_backend(self)
            kv_cache_shape = attn_backend_cls.get_kv_cache_shape(
                num_blocks,
                kv_cache_spec.block_size,
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.head_size,
            )

            num_layers = _count_kv_cache_layers(kv_cache_config)
            dtype = kv_cache_spec.dtype
            device_type = getattr(self, "device", torch.device("cuda")).type

            kv_cache_raw_tensors = kvi.alloc_kv_cache(
                kv_cache_shape,
                kv_cache_spec.block_size,
                dtype,
                device_type,
                num_layers,
                attention_type="MLA" if is_mla else "MHA",
                kv_layout="NHD",
            )
            return kv_cache_raw_tensors

        setattr(
            GPUModelRunner, "_allocate_kv_cache_from_kvcached", _allocate_kv_cache_from_kvcached
        )
        return True

    @version_range(VLLM_V9_PLUS_RANGE)
    def patch_allocation_methods(self, GPUModelRunner) -> bool:
        """Patch the allocation methods to use kvcached when enabled"""
        if not hasattr(GPUModelRunner, "_allocate_kv_cache_tensors"):
            return False

        original_method = getattr(GPUModelRunner, "_allocate_kv_cache_tensors")
        if self._is_already_patched(original_method, "alloc_kv_cache_tensors"):
            return True

        def _patched_alloc_kv(self, kv_cache_config, *args: Any, **kwargs: Any):
            if enable_kvcached():
                return self._allocate_kv_cache_from_kvcached(kv_cache_config)
            return original_method(self, kv_cache_config, *args, **kwargs)

        self._mark_as_patched(_patched_alloc_kv, "alloc_kv_cache_tensors")
        setattr(GPUModelRunner, "_allocate_kv_cache_tensors", _patched_alloc_kv)
        return True

    @version_range(VLLM_V9_PLUS_RANGE)
    def add_reshape_methods(self, GPUModelRunner) -> bool:
        """Add kvcache reshape method to the class"""
        if hasattr(GPUModelRunner, "_reshape_kv_cache_tensors_from_kvcached"):
            return True

        def _reshape_kv_cache_tensors_from_kvcached(
            self, kv_cache_config, kv_cache_raw_tensors, *args: Any, **kwargs: Any
        ):
            import torch

            kv_caches: dict[str, torch.Tensor] = {}
            kv_cache_group = kv_cache_config.kv_cache_groups[0]
            for idx, layer_name in enumerate(kv_cache_group.layer_names):
                kv_caches[layer_name] = kv_cache_raw_tensors[idx]
            return kv_caches

        setattr(
            GPUModelRunner,
            "_reshape_kv_cache_tensors_from_kvcached",
            _reshape_kv_cache_tensors_from_kvcached,
        )
        return True

    @version_range(VLLM_V9_PLUS_RANGE)
    def patch_reshape_methods(self, GPUModelRunner) -> bool:
        """Patch the reshape methods to use kvcached when enabled"""
        if not hasattr(GPUModelRunner, "_reshape_kv_cache_tensors"):
            return False

        original_method = getattr(GPUModelRunner, "_reshape_kv_cache_tensors")
        if self._is_already_patched(original_method, "reshape_kv_cache_tensors"):
            return True

        def _patched_reshape_kv(self, kv_cache_config, kv_cache_raw_tensors, *args: Any, **kwargs: Any):
            if enable_kvcached():
                return self._reshape_kv_cache_tensors_from_kvcached(
                    kv_cache_config, kv_cache_raw_tensors, *args, **kwargs
                )
            return original_method(self, kv_cache_config, kv_cache_raw_tensors, *args, **kwargs)

        self._mark_as_patched(_patched_reshape_kv, "reshape_kv_cache_tensors")
        setattr(GPUModelRunner, "_reshape_kv_cache_tensors", _patched_reshape_kv)
        return True

    # Version-specific helper methods for attention backend access
    def get_attention_backend_v8(self, model_runner_instance):
        """Get attention backend for vLLM 0.8.x versions"""
        return model_runner_instance.attn_backend

    def get_attention_backend_v9(self, model_runner_instance):
        """Get attention backend for vLLM 0.9.x versions"""
        return model_runner_instance.attn_backends[0]

    def get_attention_backend_v10(self, model_runner_instance):
        """Get attention backend for vLLM 0.10.x+ versions"""
        return model_runner_instance.attn_groups[0][0].backend

    def _get_version_specific_attention_backend(self, model_runner_instance):
        """Get the appropriate attention backend based on detected version"""
        if not self.detected_version:
            raise ValueError("vLLM version not detected")

        # Use the version range infrastructure to check version compatibility
        v8_range = VersionRange(VLLM_V8_RANGE)
        v9_range = VersionRange(VLLM_V9_RANGE)
        v10_range = VersionRange(VLLM_V10_RANGE)

        if v10_range.contains(self.detected_version):
            return self.get_attention_backend_v10(model_runner_instance)
        elif v9_range.contains(self.detected_version):
            return self.get_attention_backend_v9(model_runner_instance)
        elif v8_range.contains(self.detected_version):
            return self.get_attention_backend_v8(model_runner_instance)
        else:
            raise ValueError(f"Unsupported vLLM version: {self.detected_version}")


class GPUWorkerPatch(VersionAwarePatch, BasePatch):
    """Patch Worker.init_device to ignore GPU free-memory check when kvcached is enabled"""

    library = "vllm"
    target_module = "vllm.v1.worker.gpu_worker"
    target_class = "Worker"
    patch_name = "gpu_worker"

    def apply(self, gpuworker_mod: types.ModuleType) -> bool:
        # Initialize version info
        if not self.initialize_version_info():
            return False

        # Apply version-specific patches
        return self.patch_worker_init_device(gpuworker_mod)

    @version_range(VLLM_ALL_RANGE)
    def patch_worker_init_device(self, gpuworker_mod: types.ModuleType) -> bool:
        """Patch Worker.init_device"""
        Worker = self._get_target_class(gpuworker_mod)
        if Worker is None:
            return False

        if self._is_already_patched(Worker.init_device, "init_device"):
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
                    from vllm.utils.mem_utils import MemorySnapshot
                    from vllm.utils.torch_utils import set_random_seed  # type: ignore
                    from vllm.v1.utils import report_usage_stats  # type: ignore
                    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
                    from vllm.v1.worker.gpu_worker import (
                        init_worker_distributed_environment as _init_dist_env,
                    )
                    from vllm.v1.worker.workspace import init_workspace_manager
                except Exception:
                    logger.warning("Unable to import vLLM helpers; re-raising OOM")
                    raise

                _init_dist_env(
                    self.vllm_config, self.rank, self.distributed_init_method, self.local_rank
                )
                set_random_seed(self.model_config.seed)

                # Set init_snapshot and requested_memory so later code
                # (e.g. determine_available_memory) can access them.
                if not hasattr(self, "init_snapshot"):
                    self.init_snapshot = MemorySnapshot(device=self.device)
                if not hasattr(self, "requested_memory"):
                    # With kvcached, claim all available free memory.
                    self.requested_memory = self.init_snapshot.free_memory

                # Initialize workspace manager
                try:
                    enable_dbo = getattr(
                        self.vllm_config.parallel_config, "enable_dbo", False)
                    num_ubatches = 2 if enable_dbo else 1
                    init_workspace_manager(self.device, num_ubatches)
                except Exception:
                    pass

                self.model_runner = GPUModelRunner(self.vllm_config, self.device)  # type: ignore[attr-defined]
                if getattr(self, "rank", None) == 0:
                    report_usage_stats(self.vllm_config)

                return None

        self._mark_as_patched(_patched_init_device, "init_device")
        Worker.init_device = _patched_init_device  # type: ignore[assignment]
        return True
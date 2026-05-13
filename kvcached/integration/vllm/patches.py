# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
vLLM-specific patches using unified patch infrastructure.
"""

from __future__ import annotations

import types
from collections import OrderedDict
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


def _is_attention_spec(spec: Any) -> bool:
    """Check if a KV cache spec is an attention-type spec.

    MLAAttentionSpec only exists in vLLM >=0.11.0. Older versions express MLA
    as FullAttentionSpec(use_mla=True), which still matches FullAttentionSpec
    here, so resolving MLAAttentionSpec dynamically is sufficient.
    """
    from vllm.v1 import kv_cache_interface

    candidates = tuple(
        cls for cls in (
            getattr(kv_cache_interface, name, None)
            for name in ("FullAttentionSpec", "SlidingWindowSpec", "MLAAttentionSpec")
        )
        if isinstance(cls, type)
    )
    return isinstance(spec, candidates)


def _is_mamba_spec(spec: Any) -> bool:
    """Check if a KV cache spec is a MambaSpec."""
    try:
        from vllm.v1.kv_cache_interface import MambaSpec

        return isinstance(spec, MambaSpec)
    except ImportError:
        return False


def _get_first_attention_group(kv_cache_config: Any) -> Any:
    """Return the first attention-type KV cache group, or None."""
    for grp in kv_cache_config.kv_cache_groups:
        if _is_attention_spec(grp.kv_cache_spec):
            return grp
    return None


def _get_group_size(kv_cache_config: Any) -> int:
    """Return the maximum number of layers across all KV cache groups.

    This matches vLLM's shared memory pool count: ``group_size`` pools
    are created, each shared by one layer from every group.
    """
    return max(len(g.layer_names) for g in kv_cache_config.kv_cache_groups)


def _validate_kv_cache_groups(kv_cache_config: Any) -> None:
    """Validate KV cache groups for kvcached compatibility.

    Checks that all groups use supported spec types and that all attention
    groups share the same block geometry (block_size and cell_size).
    MambaSpec groups are accepted but not managed by kvcached.
    Raises ValueError on mismatch.
    """
    kv_groups = kv_cache_config.kv_cache_groups

    for idx, grp in enumerate(kv_groups):
        grp_spec = grp.kv_cache_spec
        if not _is_attention_spec(grp_spec) and not _is_mamba_spec(grp_spec):
            raise ValueError(
                f"kvcached only supports FullAttentionSpec, SlidingWindowSpec, "
                f"MLAAttentionSpec, and MambaSpec, got {type(grp_spec).__name__} in group {idx}"
            )

    first_attn_group = _get_first_attention_group(kv_cache_config)
    if first_attn_group is None:
        return

    first_spec = first_attn_group.kv_cache_spec
    block_size = first_spec.block_size
    cell_size, _ = _get_kv_cache_params(first_spec, block_size)

    for idx, grp in enumerate(kv_groups):
        grp_spec = grp.kv_cache_spec
        if not _is_attention_spec(grp_spec):
            continue
        grp_block_size = grp_spec.block_size
        grp_cell_size, _ = _get_kv_cache_params(grp_spec, grp_block_size)
        if grp_block_size != block_size or grp_cell_size != cell_size:
            raise ValueError(
                "kvcached requires all attention KV cache groups to have the "
                f"same block geometry. First attention group: block_size={block_size},"
                f" cell_size={cell_size}; group {idx}: "
                f"block_size={grp_block_size}, cell_size={grp_cell_size}"
            )


def _count_kv_cache_layers(kv_cache_config: Any) -> int:
    """Return the total number of KV cache layers across all groups."""
    return sum(len(g.layer_names) for g in kv_cache_config.kv_cache_groups)


def _infer_attention_type(kv_cache_config: Any) -> str:
    """Pick the kvcached attention_type for this KV cache config.

    Returns one of: "MLA", "HYBRID_LINEAR", "MHA". HYBRID_LINEAR
    requires both a FullAttentionSpec group and a linear-attention
    (mamba/SSM) group to be present.

    Uses `_is_mla_kv_cache_spec` for MLA detection so this works on vLLM
    versions that express MLA via `use_mla=True` on FullAttentionSpec
    (pre-0.11.0) as well as via the dedicated MLAAttentionSpec class
    (0.11.0+).
    """
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    has_full_attn = False
    has_mla = False
    has_mamba = False
    for grp in kv_cache_config.kv_cache_groups:
        spec = grp.kv_cache_spec
        if _is_mla_kv_cache_spec(spec):
            has_mla = True
        elif isinstance(spec, FullAttentionSpec):
            has_full_attn = True
        elif _is_mamba_spec(spec):
            has_mamba = True

    if has_mla:
        return "MLA"
    if has_full_attn and has_mamba:
        return "HYBRID_LINEAR"
    return "MHA"


def _should_enable_async_sched(vllm_config: Any) -> bool:
    """Enable kvcached async scheduling whenever vLLM async scheduling is on."""
    if vllm_config is None:
        return False
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    return bool(getattr(scheduler_config, "async_scheduling", False))


def _reshape_mamba_non_contiguous(
    raw_int8: Any, kv_cache_spec: Any, get_dtype_size: Any,
) -> list:
    """Create strided mamba state views from a per-pool flat int8 buffer.

    Mirrors vLLM's native ``_reshape_kv_cache_tensors`` for MambaSpec:
    the raw int8 buffer is reinterpreted via ``torch.as_strided`` into
    the shapes/dtypes declared by the spec.
    """
    import torch

    raw_tensor = raw_int8
    num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
    state_tensors: list = []
    storage_offset_bytes = 0
    for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
        dtype_size = get_dtype_size(dtype)
        num_element_per_page = kv_cache_spec.page_size_bytes // dtype_size
        target_shape = (num_blocks, *shape)
        stride = torch.empty(target_shape).stride()
        target_stride = (num_element_per_page, *stride[1:])
        assert storage_offset_bytes % dtype_size == 0
        tensor = torch.as_strided(
            raw_tensor.view(dtype),
            size=target_shape,
            stride=target_stride,
            storage_offset=storage_offset_bytes // dtype_size,
        )
        state_tensors.append(tensor)
        storage_offset_bytes += stride[0] * dtype_size
    return state_tensors


# Version ranges for vLLM support
VLLM_V8_RANGE = ">=0.8.4,<0.9.0"  # vLLM 0.8.x versions, need to cover 0.8.5.post1
VLLM_V9_PLUS_RANGE = ">=0.9.0"  # vLLM 0.9.x and 0.9+.x versions
VLLM_V9_RANGE = ">=0.9.0,<=0.9.2"  # vLLM 0.9.x versions
VLLM_V10_RANGE = ">0.9.2"  # vLLM 0.10.x+ versions, need to cover 0.10.0rc1
VLLM_ALL_RANGE = ">=0.8.4"  # All supported versions


def _get_kv_cache_params(
    kv_cache_spec: Any,
    block_size: int,
    attention_type: str = "MHA",
) -> tuple:
    """Determine cell_size and num_kv_buffers from a KV cache spec.

    Returns:
        (cell_size, num_kv_buffers)
    """
    if attention_type in ("MLA", "HYBRID_LINEAR") or _is_mla_kv_cache_spec(kv_cache_spec):
        # MLA: single combined KV buffer per layer
        # HYBRID_LINEAR (full attention + linear attention): K and V are
        # interleaved into one buffer per layer, so it shares MLA's
        # single-buffer math.
        # page_size_bytes = block_size * num_kv_heads * head_size * dtype_size
        cell_size = kv_cache_spec.page_size_bytes // block_size
        num_kv_buffers = 1
    else:
        # MHA/GQA: separate K and V buffers
        # page_size_bytes = 2 * block_size * num_kv_heads * head_size * dtype_size
        cell_size = kv_cache_spec.page_size_bytes // block_size // 2
        num_kv_buffers = 2
    return cell_size, num_kv_buffers


def _is_mla_kv_cache_spec(kv_cache_spec: Any) -> bool:
    """Return whether this KV cache spec should use MLA layout.

    Some vLLM versions mark MLA via `use_mla` on generic attention specs,
    while others expose `MLAAttentionSpec`.
    """
    if getattr(kv_cache_spec, "use_mla", False):
        return True
    try:
        from vllm.v1.kv_cache_interface import MLAAttentionSpec
    except ImportError:
        return False
    return isinstance(kv_cache_spec, MLAAttentionSpec)


def _get_max_cached_blocks(block_size: int) -> int:
    """Derive max cached blocks from the unified MAX_CACHED_TOKENS config.

    Returns 0 (unlimited) when MAX_CACHED_TOKENS is 0.
    """
    from kvcached.utils import MAX_CACHED_TOKENS
    if MAX_CACHED_TOKENS <= 0:
        return 0
    return MAX_CACHED_TOKENS // block_size

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
        # NOTE: use a different local name than ``KVCacheBlock`` so the stub
        # class declared in the TYPE_CHECKING block above stays visible for
        # type annotations inside the nested ``ElasticBlockPool`` class.
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
                max_cached_blocks: int = 1000
            ) -> None:
                assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
                self.enable_prefix_cache = enable_caching
                # 0 means unlimited
                self.max_cached_blocks = max_cached_blocks
                if enable_caching:
                    logger.info("Prefix caching enabled for ElasticBlockPool")

                assert not enable_kv_cache_events, (
                    "KV cache events are not supported in ElasticBlockPool")

                self.num_gpu_blocks = num_gpu_blocks
                self.enable_kv_cache_events = enable_kv_cache_events
                self.kv_event_queue = []  # type: ignore[var-annotated]
                self.kv_block_pool = [KVCacheBlockClass(i) for i in range(num_gpu_blocks)]

                from kvcached.integration.vllm.interfaces import get_kv_cache_manager

                self.kv_cache_manager = get_kv_cache_manager(
                    num_gpu_blocks, block_size, cell_size, num_layers,
                    num_kv_buffers=num_kv_buffers)

                # Allocate a dedicated null block – a placeholder for skipped
                # positions (e.g. sliding-window / chunked-local attention).
                # The original vLLM BlockPool pops block 0 from its free queue;
                # we mirror that by allocating one real block from kvcached so
                # the block_id is valid on the GPU (the attention kernel may
                # read from it, but results are masked out).
                _null_ids = self.kv_cache_manager.alloc(1)
                assert _null_ids is not None and len(_null_ids) == 1
                self.null_block = self.kv_block_pool[_null_ids[0]]
                self.null_block.is_null = True

                # Prefix cache: (block_hash, group_id) -> KVCacheBlock
                # The key embeds group_id to support hybrid attention
                # (multiple KV cache groups with different attention types).
                self._cached_blocks: dict[Any, KVCacheBlock] = {}
                # Reverse index: block_id -> cache key for O(1) eviction.
                # Each block_id belongs to exactly one group.
                self._block_id_to_key: dict[int, Any] = {}
                # LRU evictable pool: blocks with ref_cnt==0 retained for
                # cross-request prefix reuse. Insertion order = LRU order.
                self._evictable_blocks: OrderedDict[int, KVCacheBlock] = OrderedDict()

            @staticmethod
            def _make_cache_key(block_hash: Any, group_id: int) -> bytes:
                """Pack block_hash + group_id into a composite cache key.

                Mirrors vLLM's make_block_hash_with_group_id: appends 4-byte
                big-endian group_id so the same content hash is distinct across
                KV cache groups (e.g. full attention vs sliding window).
                """
                return bytes(block_hash) + group_id.to_bytes(4, "big", signed=False)

            def get_cached_block(
                self,
                block_hash: Any,
                kv_cache_group_ids: Optional[Iterable[int]] = None,
            ) -> Optional[Any]:
                if not self.enable_prefix_cache:
                    return None

                # Backward compatibility:
                # - Older vLLM versions call get_cached_block(block_hash)
                #   and expect a single KVCacheBlock.
                # - Newer hybrid-attention paths pass multiple group ids and
                #   expect one block per group.
                if kv_cache_group_ids is None:
                    key = self._make_cache_key(block_hash, 0)
                    return self._cached_blocks.get(key)
                if isinstance(kv_cache_group_ids, int):
                    kv_cache_group_ids = [int(kv_cache_group_ids)]

                cached_blocks: list[KVCacheBlock] = []
                for group_id in kv_cache_group_ids:
                    key = self._make_cache_key(block_hash, int(group_id))
                    block = self._cached_blocks.get(key)
                    if block is None:
                        # Atomic: all groups must hit or return None
                        return None
                    cached_blocks.append(block)
                if not cached_blocks:
                    return None
                return cached_blocks

            def cache_full_blocks(
                self,
                request: "Request",
                blocks: list[KVCacheBlock],
                *args: Any,
                **kwargs: Any,
            ) -> None:
                if not self.enable_prefix_cache:
                    return

                # Compatibility with vLLM call signatures across versions:
                # - (request, blocks, num_cached_blocks, num_full_blocks, block_size[, kv_cache_group_id])
                # - (request, blocks, block_hashes, num_cached_blocks, num_full_blocks, block_size[, kv_cache_group_id], hash_fn)
                # - keyword variants containing block_hashes/hash_fn.
                block_hashes = kwargs.pop("block_hashes", None)
                num_cached_blocks = kwargs.pop("num_cached_blocks", None)
                num_full_blocks = kwargs.pop("num_full_blocks", None)
                _block_size = kwargs.pop("block_size", None)
                kv_cache_group_id = kwargs.pop("kv_cache_group_id", 0)
                _hash_fn = kwargs.pop("hash_fn", None)

                remaining_args = list(args)
                if block_hashes is None and remaining_args and isinstance(remaining_args[0], (list, tuple)):
                    block_hashes = remaining_args.pop(0)

                if num_cached_blocks is None and remaining_args:
                    num_cached_blocks = remaining_args.pop(0)
                if num_full_blocks is None and remaining_args:
                    num_full_blocks = remaining_args.pop(0)
                if _block_size is None and remaining_args:
                    _block_size = remaining_args.pop(0)
                if remaining_args and isinstance(remaining_args[0], int):
                    kv_cache_group_id = remaining_args.pop(0)
                if remaining_args:
                    # Final positional argument is typically hash_fn; ignored.
                    _hash_fn = remaining_args.pop(0)

                if num_cached_blocks is None or num_full_blocks is None:
                    raise TypeError(
                        "cache_full_blocks requires num_cached_blocks and num_full_blocks"
                    )
                num_cached_blocks = int(num_cached_blocks)
                num_full_blocks = int(num_full_blocks)
                kv_cache_group_id = int(kv_cache_group_id)

                if num_cached_blocks >= num_full_blocks:
                    return

                new_full_blocks = blocks[num_cached_blocks:num_full_blocks]

                if block_hashes is None:
                    assert hasattr(request, "block_hashes"), "Request missing block_hashes attribute"
                    block_hashes = request.block_hashes
                assert len(block_hashes) >= num_full_blocks, \
                    f"Request has {len(block_hashes)} hashes but need {num_full_blocks}"

                for i, block in enumerate(new_full_blocks):
                    if getattr(block, "is_null", False):
                        continue

                    block_idx = num_cached_blocks + i
                    block_hash = block_hashes[block_idx]
                    key = self._make_cache_key(block_hash, kv_cache_group_id)

                    # Already cached, idempotent
                    if key in self._cached_blocks:
                        continue

                    self._cached_blocks[key] = block
                    self._block_id_to_key[block.block_id] = key

            def _evict_blocks_from_pool(self, num_to_evict: int) -> int:
                """Evict oldest blocks from evictable pool, free to kvcached.

                Returns the number of blocks actually evicted.
                """
                ids_to_free: list[int] = []
                for _ in range(min(num_to_evict, len(self._evictable_blocks))):
                    bid, _block = self._evictable_blocks.popitem(last=False)
                    key = self._block_id_to_key.pop(bid, None)
                    if key is not None:
                        self._cached_blocks.pop(key, None)
                    ids_to_free.append(bid)
                if ids_to_free:
                    self.kv_cache_manager.free(ids_to_free)
                return len(ids_to_free)

            def get_new_blocks(
                self, num_blocks: int
            ) -> list[KVCacheBlock]:
                if num_blocks > self.get_num_free_blocks():
                    raise ValueError(
                        f"Cannot get {num_blocks} free blocks from the pool")

                if self.enable_prefix_cache:
                    # Evict cached blocks if kvcached doesn't have enough free space
                    kvcached_free = self.kv_cache_manager.available_size()
                    if kvcached_free < num_blocks and self._evictable_blocks:
                        self._evict_blocks_from_pool(num_blocks - kvcached_free)

                block_ids = self.kv_cache_manager.alloc(num_blocks)
                assert block_ids is not None and len(block_ids) == num_blocks

                blocks = []
                for bid in block_ids:
                    block = self.kv_block_pool[bid]
                    block.ref_cnt = 1
                    blocks.append(block)
                return blocks

            def touch(
                self, blocks: list[KVCacheBlock] | tuple[list[KVCacheBlock], ...]
            ) -> None:
                if not self.enable_prefix_cache:
                    return
                if isinstance(blocks, tuple):
                    for block_list in blocks:
                        for block in block_list:
                            block.ref_cnt += 1
                            # Reactivate: remove from evictable pool
                            self._evictable_blocks.pop(block.block_id, None)
                else:
                    for block in blocks:
                        block.ref_cnt += 1
                        self._evictable_blocks.pop(block.block_id, None)

            def free_blocks(
                self,
                ordered_blocks: Iterable[KVCacheBlock],
            ) -> None:
                if not self.enable_prefix_cache:
                    block_ids = [
                        block.block_id
                        for block in ordered_blocks
                        if block is not None and not getattr(block, "is_null", False)
                    ]
                    if block_ids:
                        self.kv_cache_manager.free(block_ids)
                    return

                uncached_to_free: list[int] = []
                for block in ordered_blocks:
                    if block is None or getattr(block, "is_null", False):
                        continue
                    block.ref_cnt -= 1
                    if block.ref_cnt == 0:
                        if block.block_id in self._block_id_to_key:
                            # Cached block: retain for cross-request reuse
                            self._evictable_blocks[block.block_id] = block
                        else:
                            # Uncached block (e.g. partial): free immediately
                            uncached_to_free.append(block.block_id)
                if uncached_to_free:
                    self.kv_cache_manager.free(uncached_to_free)

                if (self.max_cached_blocks > 0
                        and len(self._evictable_blocks) > self.max_cached_blocks):
                    excess = len(self._evictable_blocks) - self.max_cached_blocks
                    self._evict_blocks_from_pool(excess)


            def evict_blocks(self, block_ids: set[int]) -> None:
                if not self.enable_prefix_cache:
                    return

                removed = 0
                ids_to_free: list[int] = []
                for bid in block_ids:
                    key = self._block_id_to_key.pop(bid, None)
                    if key is not None:
                        self._cached_blocks.pop(key, None)
                        removed += 1
                    if bid in self._evictable_blocks:
                        self._evictable_blocks.pop(bid)
                        ids_to_free.append(bid)

                if ids_to_free:
                    self.kv_cache_manager.free(ids_to_free)
                if removed:
                    logger.debug(f"Evicted {removed} blocks from prefix cache")

            def reset_prefix_cache(self) -> bool:
                if not self.enable_prefix_cache:
                    return True

                # Free all evictable blocks back to kvcached
                if self._evictable_blocks:
                    ids_to_free = list(self._evictable_blocks.keys())
                    self._evictable_blocks.clear()
                    self.kv_cache_manager.free(ids_to_free)

                self._cached_blocks.clear()
                self._block_id_to_key.clear()
                logger.info("Prefix cache reset")
                return True

            def get_num_free_blocks(self) -> int:
                return (self.kv_cache_manager.available_size() + len(self._evictable_blocks)) if self.enable_prefix_cache else self.kv_cache_manager.available_size()

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
                        async_sched=_should_enable_async_sched(vllm_config),
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

            first_attn_group = _get_first_attention_group(kv_cache_config)
            if first_attn_group is None:
                raise RuntimeError(
                    "kvcached is enabled but the KV cache config contains no "
                    "attention groups; nothing to manage."
                )

            kv_cache_spec = first_attn_group.kv_cache_spec
            block_size = kv_cache_spec.block_size

            attention_type = _infer_attention_type(kv_cache_config)

            cell_size, num_kv_buffers = _get_kv_cache_params(
                kv_cache_spec, block_size, attention_type=attention_type)

            try:
                from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size

                tp_size = int(get_tensor_model_parallel_world_size())
            except Exception:
                tp_size = 1

            from kvcached.integration.vllm import interfaces as kvi

            # Use tp_size (not TP*PP global world size) for the KVCacheManager world_size.
            # Each PP stage manages its own KV tensors independently. The IPC sockets
            # are registered per TP rank within each stage (w0.sock … w(tp_size-1).sock).
            kvi.init_kvcached(
                tp_rank=0,
                world_size=tp_size,
                is_worker=False,
                async_sched=_should_enable_async_sched(getattr(self, "vllm_config", None)),
            )

            # Import ElasticBlockPool from the patched module
            import importlib

            block_pool_mod = importlib.import_module("vllm.v1.core.block_pool")
            ElasticBlockPool = getattr(block_pool_mod, "ElasticBlockPool")

            group_size = _get_group_size(kv_cache_config)
            self.block_pool = ElasticBlockPool(
                kv_cache_config.num_blocks,
                block_size,
                cell_size=cell_size,
                num_layers=group_size,
                enable_caching=getattr(self, "enable_caching", False),
                num_kv_buffers=num_kv_buffers,
                max_cached_blocks=_get_max_cached_blocks(block_size)
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
                max_cached_blocks=_get_max_cached_blocks(block_size)
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
            kvi.init_kvcached(
                tp_rank=tp_rank,
                world_size=tp_size,
                pp_rank=pp_rank,
                is_worker=True,
                device=device_str,
                async_sched=_should_enable_async_sched(self.vllm_config),
            )

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

            attention_type = _infer_attention_type(kv_cache_config)
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
                attention_type=attention_type,
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
        logger = self.logger  # Capture logger in closure

        def _allocate_kv_cache_from_kvcached(self, kv_cache_config):
            import torch
            from vllm.v1.kv_cache_interface import KVCacheTensor

            from kvcached.integration.vllm import interfaces as kvi

            _validate_kv_cache_groups(kv_cache_config)

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

            first_attn_group_id = None
            first_attn_group = None
            for idx, grp in enumerate(kv_cache_config.kv_cache_groups):
                if _is_attention_spec(grp.kv_cache_spec):
                    first_attn_group_id = idx
                    first_attn_group = grp
                    break

            if first_attn_group is None or first_attn_group_id is None:
                raise RuntimeError(
                    "kvcached is enabled but the KV cache config contains no "
                    "attention groups; nothing to allocate."
                )

            kv_cache_spec = first_attn_group.kv_cache_spec
            attention_type = _infer_attention_type(kv_cache_config)

            first_layer_name = first_attn_group.layer_names[0]
            rep_tensor_cfg = layer_to_tensor_cfg[first_layer_name]
            num_blocks = rep_tensor_cfg.size // kv_cache_spec.page_size_bytes

            # Use version-aware attention backend access
            attn_backend_cls = patch_instance._get_version_specific_attention_backend(
                self, kv_cache_group_id=first_attn_group_id
            )

            backend_name = (
                attn_backend_cls.get_name() if hasattr(attn_backend_cls, "get_name")
                else str(attn_backend_cls)
            ).upper()
            if backend_name == "FLASHINFER":
                required_layout = None
                if hasattr(attn_backend_cls, "get_required_kv_cache_layout"):
                    required_layout = attn_backend_cls.get_required_kv_cache_layout()

                selected_layout = required_layout or "NHD"
                if selected_layout != "NHD":
                    raise RuntimeError(
                        "kvcached currently supports NHD KV layout only, but "
                        f"{backend_name} requires {selected_layout}."
                    )

                from vllm.v1.attention.backends.utils import set_kv_cache_layout

                set_kv_cache_layout(selected_layout)

            kv_cache_shape = attn_backend_cls.get_kv_cache_shape(
                num_blocks,
                kv_cache_spec.block_size,
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.head_size,
            )

            # Allocate group_size shared VM-backed pools, mirroring vLLM's
            # KVCacheTensor sharing: pool i is shared by layer i from each
            # group, and different groups use different block IDs within the
            # same pool.
            group_size = _get_group_size(kv_cache_config)
            dtype = kv_cache_spec.dtype
            device_type = getattr(self, "device", torch.device("cuda")).type

            # vLLM may split a virtual block (spec.block_size tokens) into
            # ``ratio`` kernel-sized blocks; the attention zero kernel indexes
            # by kernel-block stride. Forward kernel_block_size so we build the
            # per-layer tensor at kernel granularity.
            kernel_block_sizes = getattr(self, "_kernel_block_sizes", None)
            kernel_block_size = (
                kernel_block_sizes[first_attn_group_id]
                if kernel_block_sizes is not None
                and first_attn_group_id < len(kernel_block_sizes)
                else None)

            if kernel_block_size is None:
                logger.warning("kernel_block_size is None, try to compute from the attention backend's supported block sizes")
                try:
                    # Try to compute kernel_block_size from the attention backend's
                    # supported block sizes, mirroring vLLM's
                    # prepare_kernel_block_sizes / select_common_block_size logic.
                    kv_manager_block_size = kv_cache_spec.block_size
                    supported_sizes = None
                    if hasattr(attn_backend_cls, 'get_supported_kernel_block_sizes'):
                        supported_sizes = attn_backend_cls.get_supported_kernel_block_sizes()
                    elif hasattr(attn_backend_cls, 'get_supported_block_sizes'):
                        supported_sizes = attn_backend_cls.get_supported_block_sizes()

                    if supported_sizes:
                        # Import MultipleOf for isinstance checks
                        try:
                            from vllm.v1.attention.backend import MultipleOf as _MultipleOf
                        except ImportError:
                            _MultipleOf = None  # type: ignore[misc]

                        # Check if the kv_manager_block_size is directly supported
                        directly_supported = False
                        for s in supported_sizes:
                            if isinstance(s, int) and s == kv_manager_block_size:
                                directly_supported = True
                                break
                            elif isinstance(s, range) and kv_manager_block_size in s:
                                directly_supported = True
                                break
                            elif (_MultipleOf is not None
                                and isinstance(s, _MultipleOf)
                                and kv_manager_block_size % s.base == 0):
                                directly_supported = True
                                break
                        if not directly_supported:
                            # Pick the largest supported size that divides kv_manager_block_size
                            candidates = []
                            for s in supported_sizes:
                                if isinstance(s, int):
                                    if kv_manager_block_size % s == 0:
                                        candidates.append(s)
                                elif isinstance(s, range):
                                    for v in s:
                                        if kv_manager_block_size % v == 0:
                                            candidates.append(v)
                                elif (_MultipleOf is not None
                                    and isinstance(s, _MultipleOf)):
                                    if kv_manager_block_size % s.base == 0:
                                        candidates.append(kv_manager_block_size)
                            if candidates:
                                kernel_block_size = max(candidates)
                except Exception:
                    pass
            # Fallback: try prepare_kernel_block_sizes if available
            if kernel_block_size is None:
                try:
                    from vllm.v1.worker.utils import prepare_kernel_block_sizes as _prep_kbs
                    _attn_groups = getattr(self, "attn_groups", None)
                    if _attn_groups is not None:
                        _kbs_list = _prep_kbs(kv_cache_config, _attn_groups)
                        if first_attn_group_id < len(_kbs_list):
                            kernel_block_size = _kbs_list[first_attn_group_id]
                except Exception:
                    pass

            alloc_result = kvi.alloc_kv_cache(
                kv_cache_shape,
                kv_cache_spec.block_size,
                dtype,
                device_type,
                group_size,
                attention_type=attention_type,
                kv_layout="NHD",
                kernel_block_size=kernel_block_size,
            )

            if attention_type == "HYBRID_LINEAR":
                kv_cache_raw_tensors, raw_info = alloc_result
                self._kvcached_mamba_raw_info = raw_info
            else:
                kv_cache_raw_tensors = alloc_result

            # Return the list of pool tensors directly; the layer-name
            # mapping is done in _reshape_kv_cache_tensors_from_kvcached.
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
            try:
                from vllm.utils.torch_utils import get_dtype_size
            except ImportError:
                from vllm.utils import get_dtype_size  # type: ignore[attr-defined]

            kv_caches: dict[str, torch.Tensor] = {}

            mamba_info = getattr(self, "_kvcached_mamba_raw_info", None)

            for kv_cache_group in kv_cache_config.kv_cache_groups:
                kv_cache_spec = kv_cache_group.kv_cache_spec

                if _is_mamba_spec(kv_cache_spec):
                    if mamba_info is None:
                        raise RuntimeError(
                            "Mamba layers found but no raw buffer info "
                            "available from kvcached"
                        )
                    for pool_idx, layer_name in enumerate(kv_cache_group.layer_names):
                        state_tensors = _reshape_mamba_non_contiguous(
                            mamba_info["buffers"][pool_idx],
                            kv_cache_spec, get_dtype_size,
                        )
                        kv_caches[layer_name] = state_tensors  # type: ignore[assignment]
                else:
                    for pool_idx, layer_name in enumerate(kv_cache_group.layer_names):
                        kv_caches[layer_name] = kv_cache_raw_tensors[pool_idx]

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
    def get_attention_backend_v8(self, model_runner_instance, kv_cache_group_id=0):
        """Get attention backend for vLLM 0.8.x versions"""
        return model_runner_instance.attn_backend

    def get_attention_backend_v9(self, model_runner_instance, kv_cache_group_id=0):
        """Get attention backend for vLLM 0.9.x versions"""
        return model_runner_instance.attn_backends[kv_cache_group_id]

    def get_attention_backend_v10(self, model_runner_instance, kv_cache_group_id=0):
        """Get attention backend for vLLM 0.10.x+ versions"""
        return model_runner_instance.attn_groups[kv_cache_group_id][0].backend

    def _get_version_specific_attention_backend(
        self, model_runner_instance, kv_cache_group_id=0
    ):
        """Get the appropriate attention backend based on detected version"""
        if not self.detected_version:
            raise ValueError("vLLM version not detected")

        # Use the version range infrastructure to check version compatibility
        v8_range = VersionRange(VLLM_V8_RANGE)
        v9_range = VersionRange(VLLM_V9_RANGE)
        v10_range = VersionRange(VLLM_V10_RANGE)

        if v10_range.contains(self.detected_version):
            return self.get_attention_backend_v10(model_runner_instance, kv_cache_group_id)
        elif v9_range.contains(self.detected_version):
            return self.get_attention_backend_v9(model_runner_instance, kv_cache_group_id)
        elif v8_range.contains(self.detected_version):
            return self.get_attention_backend_v8(model_runner_instance, kv_cache_group_id)
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

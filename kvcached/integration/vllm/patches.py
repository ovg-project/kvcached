# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
vLLM-specific patches using unified patch infrastructure.
"""

from __future__ import annotations

import inspect
import math
import types
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Iterable, Optional

from kvcached.integration.patch_base import BasePatch, enable_kvcached
from kvcached.integration.version_utils import VersionAwarePatch, VersionRange, version_range
from kvcached.utils import KVCachedConfigError, KVCachePoolExhausted, get_kvcached_logger

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


logger = get_kvcached_logger()


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
    cell_size, num_kv_buffers = _get_kv_cache_params(first_spec, block_size)
    block_mem_size = block_size * cell_size

    for idx, grp in enumerate(kv_groups):
        grp_spec = grp.kv_cache_spec
        if not _is_attention_spec(grp_spec):
            continue
        grp_block_size = grp_spec.block_size
        grp_cell_size, grp_num_kv_buffers = _get_kv_cache_params(grp_spec, grp_block_size)
        grp_block_mem_size = grp_block_size * grp_cell_size
        # kvcached needs one uniform physical block stride (block_mem_size =
        # bytes/block per K-or-V) and one K/V buffer count. It does NOT require
        # identical block_size/cell_size individually: attention groups that split
        # a block into different token counts (e.g. Gemma's sliding-window
        # block_size=64/cell=1024 vs full-attention block_size=16/cell=4096, both
        # block_mem_size=65536) share one physical pool and get a per-group
        # as_strided view. Reject only when the physical block stride or the K/V
        # buffer count actually differ (a genuine single-pool violation, e.g.
        # mixing MLA num_kv_buffers=1 with MHA num_kv_buffers=2).
        if grp_block_mem_size != block_mem_size or grp_num_kv_buffers != num_kv_buffers:
            raise ValueError(
                "kvcached requires all attention KV cache groups to share one "
                "physical block geometry (block_mem_size and num_kv_buffers). "
                f"First attention group: block_mem_size={block_mem_size}, "
                f"num_kv_buffers={num_kv_buffers}; group {idx}: "
                f"block_mem_size={grp_block_mem_size}, "
                f"num_kv_buffers={grp_num_kv_buffers}"
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


def _reshape_mamba_contiguous(
    mamba_info: dict, kv_cache_spec: Any, pool_idx: int, get_dtype_size: Any,
) -> list:
    """Create strided mamba state views from a contiguous interleaved buffer.

    In contiguous layout there is a single base buffer shared by all pools.
    Block N for pool L sits at byte offset
    ``(N * num_pools + L) * page_size_bytes`` inside that base buffer, so the
    inter-block stride is ``block_stride_bytes == num_pools * page_size_bytes``
    (not ``page_size_bytes`` as in the non-contiguous per-pool case), and the
    per-pool base offset is ``pool_idx * page_size_bytes``. This aliases exactly
    the same cell the contiguous attention view (contiguous_tensor[:, L]) reads,
    so a hybrid model's attention and mamba layers share one physical block.
    """
    import torch

    base_buffer = mamba_info["buffers"][0]  # flat int8 buffer
    num_blocks = mamba_info["num_blocks"]
    page_size_bytes = mamba_info["page_size_bytes"]
    block_stride_bytes = mamba_info["block_stride_bytes"]

    layer_offset_bytes = pool_idx * page_size_bytes

    state_tensors: list = []
    inner_offset_bytes = 0
    for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
        dtype_size = get_dtype_size(dtype)
        block_stride_elems = block_stride_bytes // dtype_size
        target_shape = (num_blocks, *shape)
        inner_stride = torch.empty(target_shape).stride()
        target_stride = (block_stride_elems, *inner_stride[1:])
        assert (layer_offset_bytes + inner_offset_bytes) % dtype_size == 0
        storage_offset = (layer_offset_bytes + inner_offset_bytes) // dtype_size
        tensor = torch.as_strided(
            base_buffer.view(dtype),
            size=target_shape,
            stride=target_stride,
            storage_offset=storage_offset,
        )
        state_tensors.append(tensor)
        inner_offset_bytes += inner_stride[0] * dtype_size
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

    Returns -1 (unlimited) when MAX_CACHED_TOKENS < 0.
    Returns 0  (disabled — evict on free) when MAX_CACHED_TOKENS == 0.
    Otherwise returns ``max(1, MAX_CACHED_TOKENS // block_size)``.

    The floor matters: a *positive* ``MAX_CACHED_TOKENS`` smaller than
    ``block_size`` (e.g. 8 tokens with a 16-token block) integer-divides to
    ``0``, which is indistinguishable from the ``== 0`` "disabled" sentinel and
    would silently turn prefix caching off. Flooring at one block keeps caching
    enabled for the smallest non-zero budget, matching the user's intent; a
    warning is logged so the effective granularity is not silent.
    """
    from kvcached.utils import MAX_CACHED_TOKENS
    if MAX_CACHED_TOKENS < 0:
        return -1
    if MAX_CACHED_TOKENS == 0:
        return 0
    max_cached_blocks = MAX_CACHED_TOKENS // block_size
    if max_cached_blocks == 0:
        logger.warning(
            "KVCACHED_MAX_CACHED_TOKENS=%d is smaller than the KV block size "
            "(%d tokens); flooring max cached blocks to 1 so prefix caching "
            "stays enabled. Set KVCACHED_MAX_CACHED_TOKENS=0 to disable "
            "caching explicitly.",
            MAX_CACHED_TOKENS,
            block_size,
        )
        return 1
    return max_cached_blocks


def _cache_dtype_str(model_runner: Any) -> Optional[str]:
    """Extract the KV cache dtype string from a GPUModelRunner, if available."""
    cache_config = getattr(model_runner, "cache_config", None)
    if cache_config is None:
        vllm_config = getattr(model_runner, "vllm_config", None)
        cache_config = getattr(vllm_config, "cache_config", None)
    return getattr(cache_config, "cache_dtype", None)


def _get_kv_cache_shape_compat(attn_backend: Any, num_blocks: int,
                               block_size: int, num_kv_heads: int,
                               head_size: int,
                               cache_dtype_str: Optional[str]) -> Any:
    """Call ``get_kv_cache_shape``, forwarding ``cache_dtype_str`` when the
    backend's signature accepts it.

    Per-token-head quantization modes (fp8_per_token_head,
    int8_per_token_head, nvfp4) widen ``head_size`` by a few elements to
    inline per-head scales into the KV page. Omitting ``cache_dtype_str``
    makes the backend compute the un-widened shape, so every page stride is
    wrong and output is garbled (#424). Older vLLM versions do not take the
    parameter, so it is only forwarded when declared.

    Module-level so it is unit-testable without an installed vLLM or a GPU.
    """
    if cache_dtype_str is not None:
        try:
            declares_dtype = "cache_dtype_str" in inspect.signature(
                attn_backend.get_kv_cache_shape).parameters
        except (TypeError, ValueError):
            declares_dtype = False
        if declares_dtype:
            return attn_backend.get_kv_cache_shape(
                num_blocks,
                block_size,
                num_kv_heads,
                head_size,
                cache_dtype_str=cache_dtype_str,
            )
    return attn_backend.get_kv_cache_shape(num_blocks, block_size,
                                           num_kv_heads, head_size)


def _make_cache_key(block_hash: Any, group_id: int) -> bytes:
    """Pack block_hash + group_id into a composite cache key.

    Mirrors vLLM's make_block_hash_with_group_id: appends a 4-byte big-endian
    group_id so the same content hash is distinct across KV cache groups
    (e.g. full attention vs sliding window). ``block_hash`` may arrive as
    ``bytes`` (the common case) or ``str`` (some vLLM versions pass a hex
    digest); a str is encoded to bytes so both inputs produce identical keys.

    Module-level (rather than a nested staticmethod) so it is unit-testable
    without applying the vLLM patch or holding a GPU.
    """
    if isinstance(block_hash, str):
        block_hash = block_hash.encode()
    return bytes(block_hash) + group_id.to_bytes(4, "big", signed=False)


def _reset_block_hash(block: Any) -> None:
    """Clear vLLM's cached block hash before returning a block to kvcached."""
    reset_hash = getattr(block, "reset_hash", None)
    if callable(reset_hash):
        reset_hash()
        return
    if hasattr(block, "_block_hash"):
        block._block_hash = None
    if hasattr(block, "_block_hash_num_tokens"):
        block._block_hash_num_tokens = None


def _set_block_hash(block: Any, key: Any) -> None:
    """Set vLLM's cached block hash across API versions.

    vLLM 0.24 and later expose a read-only property plus set_block_hash().
    Older supported versions expose a writable block_hash property instead.
    """
    set_block_hash = getattr(block, "set_block_hash", None)
    if callable(set_block_hash):
        set_block_hash(key)
    else:
        block.block_hash = key


def _convert_block_hashes(
    block_hashes: Any,
    hash_block_size: int,
    target_block_size: int,
) -> Any:
    if target_block_size == hash_block_size:
        return block_hashes

    import importlib

    kv_cache_utils = importlib.import_module("vllm.v1.core.kv_cache_utils")
    converter = getattr(kv_cache_utils, "BlockHashListWithBlockSize", None)
    if converter is None:
        raise RuntimeError(
            "This vLLM version does not support heterogeneous block-hash conversion"
        )
    return converter(block_hashes, hash_block_size, target_block_size)


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
                max_cached_blocks: int = 1000,
                hash_block_size: Optional[int] = None,
            ) -> None:
                assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
                self.enable_prefix_cache = enable_caching
                # -1 = unlimited, 0 = disabled (evict on free), >0 = cap
                self.max_cached_blocks = max_cached_blocks
                if enable_caching:
                    logger.info("Prefix caching enabled for ElasticBlockPool")

                assert not enable_kv_cache_events, (
                    "KV cache events are not supported in ElasticBlockPool")

                self.num_gpu_blocks = num_gpu_blocks
                # Request.block_hashes are computed at hash_block_size, which
                # can be smaller than a heterogeneous KV group's physical
                # block_size. Keep this distinct from block_size, which is also
                # used to configure kvcached's physical allocation geometry.
                self.hash_block_size = (
                    block_size if hash_block_size is None else int(hash_block_size)
                )
                self.enable_kv_cache_events = enable_kv_cache_events
                self.kv_event_queue = []  # type: ignore[var-annotated]
                self.kv_block_pool = [KVCacheBlockClass(i) for i in range(num_gpu_blocks)]

                from kvcached.integration.vllm.interfaces import get_kv_cache_manager

                self.kv_cache_manager = get_kv_cache_manager(
                    num_gpu_blocks, block_size, cell_size, num_layers,
                    num_kv_buffers=num_kv_buffers,
                    pool_name="block_pool")

                # Allocate a dedicated null block – a placeholder for skipped
                # positions (e.g. sliding-window / chunked-local attention).
                # The original vLLM BlockPool pops block 0 from its free queue;
                # we mirror that by allocating one real block from kvcached so
                # the block_id is valid on the GPU (the attention kernel may
                # read from it, but results are masked out).
                # vLLM hard-codes null == block 0: native BlockPool pops
                # block 0 as the null block, NULL_BLOCK_ID = 0, block tables
                # are fill_(0) so padded/unused slots read 0, and mamba/GDN
                # state kernels skip index 0 on both read and write. If any
                # real request owns block 0, its GDN state is silently
                # skipped as "null" and its output garbles. The manager is
                # created with reserve_null_block=True (see get_kv_cache_manager),
                # which reserves and maps block 0 synchronously before the
                # page-prealloc thread starts (and fails loud if it cannot),
                # so block 0 never enters circulation. Just wrap it here.
                self.null_block = self.kv_block_pool[0]
                self.null_block.is_null = True

                # Prefix cache: (block_hash, group_id) -> {block_id: block}.
                # Multiple in-flight requests can materialize the same prefix
                # before either block becomes reusable. vLLM preserves every
                # such block because request block tables are append-only.
                # The key embeds group_id to support hybrid attention
                # (multiple KV cache groups with different attention types).
                self._cached_blocks: dict[Any, dict[int, KVCacheBlock]] = {}
                # Reverse index: block_id -> cache key for O(1) eviction.
                # Each block_id belongs to exactly one group.
                self._block_id_to_key: dict[int, Any] = {}
                # LRU evictable pool: blocks with ref_cnt==0 retained for
                # cross-request prefix reuse. Insertion order = LRU order.
                self._evictable_blocks: OrderedDict[int, KVCacheBlock] = OrderedDict()

            def _get_one_cached_block(self, key: Any) -> Optional[KVCacheBlock]:
                blocks = self._cached_blocks.get(key)
                if not blocks:
                    return None
                return next(iter(blocks.values()))

            def _remove_cached_block(
                self, key: Any, block_id: int
            ) -> Optional[KVCacheBlock]:
                blocks = self._cached_blocks.get(key)
                if not blocks:
                    return None
                block = blocks.pop(block_id, None)
                if not blocks:
                    self._cached_blocks.pop(key, None)
                return block

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
                    key = _make_cache_key(block_hash, 0)
                    return self._get_one_cached_block(key)
                if isinstance(kv_cache_group_ids, int):
                    kv_cache_group_ids = [int(kv_cache_group_ids)]

                cached_blocks: list[KVCacheBlock] = []
                for group_id in kv_cache_group_ids:
                    key = _make_cache_key(block_hash, int(group_id))
                    block = self._get_one_cached_block(key)
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
                block_mask = kwargs.pop("block_mask", None)
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
                if (
                    block_mask is None
                    and remaining_args
                    and isinstance(remaining_args[0], (list, tuple))
                ):
                    block_mask = remaining_args.pop(0)
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
                assert block_mask is None or len(block_mask) == len(new_full_blocks)

                if block_hashes is None:
                    assert hasattr(request, "block_hashes"), "Request missing block_hashes attribute"
                    target_block_size = (
                        self.hash_block_size
                        if _block_size is None
                        else int(_block_size)
                    )
                    block_hashes = _convert_block_hashes(
                        request.block_hashes,
                        self.hash_block_size,
                        target_block_size,
                    )
                assert len(block_hashes) >= num_full_blocks, \
                    f"Request has {len(block_hashes)} hashes but need {num_full_blocks}"

                for i, block in enumerate(new_full_blocks):
                    if (
                        getattr(block, "is_null", False)
                        or (block_mask is not None and not block_mask[i])
                    ):
                        continue

                    block_idx = num_cached_blocks + i
                    block_hash = block_hashes[block_idx]
                    key = _make_cache_key(block_hash, kv_cache_group_id)

                    # ElasticBlockPool tracks cached blocks through its own maps,
                    # but vLLM manager code may still read KVCacheBlock.block_hash
                    # after cache_full_blocks. Preserve that metadata contract and
                    # clear it before the block is evicted or reused.
                    previous_key = self._block_id_to_key.get(block.block_id)
                    if previous_key is not None and previous_key != key:
                        self._remove_cached_block(previous_key, block.block_id)
                        _reset_block_hash(block)
                    if getattr(block, "block_hash", None) is None:
                        _set_block_hash(block, key)
                    self._cached_blocks.setdefault(key, {})[block.block_id] = block
                    self._block_id_to_key[block.block_id] = key

            def _page_aligned_victims(self, num_to_evict: int) -> list[int]:
                """Order evictable blocks so whole pages come free.

                kvcached only returns physical memory once every block on a page
                is freed, so evicting in LRU order can free no memory at all when
                the survivors stay scattered across pages. Prefer pages this pool
                can empty outright, cheapest first; draining a page only part of
                the way costs hit rate and frees nothing.
                """
                mgr = self.kv_cache_manager
                allocator = getattr(mgr, "page_allocator", None)
                if allocator is None:
                    return []

                bids = list(self._evictable_blocks)
                by_page = allocator.group_indices_by_page(
                    bids, mgr.block_mem_size)
                # Blocks held by running requests are absent from
                # _evictable_blocks, so a page whose occupancy exceeds its
                # evictable count cannot be emptied here -- skip it.
                occupancy = mgr.get_page_occupancy(list(by_page))
                lru_rank = {bid: i for i, bid in enumerate(bids)}

                pages = [(len(ids), max(lru_rank[b] for b in ids), ids)
                         for page_id, ids in by_page.items()
                         if len(ids) >= occupancy.get(page_id, 0)]
                # Cheapest page first; break ties on the page whose most
                # recently used block is oldest, so hot pages are kept.
                pages.sort(key=lambda page: (page[0], page[1]))

                victims: list[int] = []
                for cost, _rank, ids in pages:
                    if len(victims) + cost > num_to_evict:
                        break
                    victims.extend(ids)
                return victims

            def _evict_blocks_from_pool(self,
                                        num_to_evict: int,
                                        page_aware: bool = True) -> int:
                """Evict blocks from evictable pool, free to kvcached.

                With page_aware set, prefers victims that empty whole pages so
                freeing them returns physical memory, then falls back to LRU
                order for the remainder. Use it only when the goal is physical
                release (cap trimming): a page returns memory only once every
                block on it is free.

                With page_aware clear, evicts in pure LRU order. Callers that
                reuse the freed logical slot immediately (allocation shortage)
                get no page benefit -- the page is neither unmapped nor
                remapped -- so reordering victims by page only trades away a
                newer prefix for an older one and pays for the full evictable
                scan and page sort.

                Returns the number of blocks actually evicted.
                """
                num_to_evict = min(num_to_evict, len(self._evictable_blocks))
                if num_to_evict <= 0:
                    return 0

                if page_aware:
                    ordered = self._page_aligned_victims(num_to_evict)
                    chosen = set(ordered)
                    # Top up in LRU order: page alignment is best-effort, but
                    # the caller still needs the count it asked for.
                    ordered.extend(bid for bid in self._evictable_blocks
                                   if bid not in chosen)
                else:
                    ordered = list(self._evictable_blocks)

                ids_to_free: list[int] = []
                for bid in ordered[:num_to_evict]:
                    block = self._evictable_blocks.pop(bid, None)
                    key = self._block_id_to_key.pop(bid, None)
                    if key is not None:
                        self._remove_cached_block(key, bid)
                    if block is not None:
                        _reset_block_hash(block)
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

                block_ids: Optional[list[int]] = None
                for _ in range(2):
                    if self.enable_prefix_cache:
                        kvcached_free = self.kv_cache_manager.available_size()
                        if kvcached_free < num_blocks and self._evictable_blocks:
                            # Allocation shortage: the freed slot is reused
                            # immediately, so page-aware selection buys no
                            # memory and would evict a newer prefix over the
                            # LRU victim. Keep pure LRU here.
                            self._evict_blocks_from_pool(
                                num_blocks - kvcached_free, page_aware=False)
                    block_ids = self.kv_cache_manager.alloc(num_blocks)
                    if block_ids is not None:
                        break

                if block_ids is None:
                    # Transient, not a defect: a colocated engine took the last
                    # physical pages. KVCacheManagerAllocateSlotsPatch turns
                    # this into the scheduler's own "cannot allocate now"
                    # signal, so keep it a distinct type the patch can catch
                    # without also swallowing real contract violations.
                    raise KVCachePoolExhausted(
                        "Unable to allocate KV cache blocks from physical pool; "
                        f"requested={num_blocks}, available={self.kv_cache_manager.available_size()}"
                    )

                # alloc() returns either None or exactly num_blocks ids
                # (see KVCacheManager._alloc), so a different length is a
                # contract violation rather than a recoverable runtime state.
                assert len(block_ids) == num_blocks, (
                    f"alloc returned {len(block_ids)} blocks, expected {num_blocks}")

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
                prepend: bool = False,
            ) -> None:
                # vLLM >= 0.23 passes ``prepend`` to put freed blocks at the
                # front of its free queue for reuse priority. kvcached has no
                # linear free queue: reuse order is governed by
                # KVCacheManager's page-affine allocation, so the hint is
                # accepted for signature compatibility and unused (#438).
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
                            _reset_block_hash(block)
                            uncached_to_free.append(block.block_id)
                if uncached_to_free:
                    self.kv_cache_manager.free(uncached_to_free)

                if (self.max_cached_blocks >= 0
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
                        block = self._remove_cached_block(key, bid)
                        if block is not None:
                            _reset_block_hash(block)
                        removed += 1
                    if bid in self._evictable_blocks:
                        block = self._evictable_blocks.pop(bid)
                        _reset_block_hash(block)
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
                    for block in self._evictable_blocks.values():
                        _reset_block_hash(block)
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
            except (KVCachedConfigError, RuntimeError):
                # User-fixable misconfiguration (e.g. KV block larger than the
                # page size), or a broken kvcached invariant such as
                # get_world_size() finding kvcached uninitialized. Abort loudly
                # instead of silently disabling kvcached and falling back to
                # vanilla allocation: a half-applied coordinator patch changes
                # KV behaviour while leaving only a warning in the log.
                raise
            except Exception as e:
                logger.warning("Failed to patch kv_cache_coordinator: %s", e)
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

            from kvcached.integration.vllm import interfaces as kvi

            # EngineCore records tensor_parallel_size before constructing this
            # coordinator. parallel_state is not authoritative here: depending
            # on startup timing it can either raise or still report world size 1.
            tp_size = int(kvi.get_world_size())

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
            # vLLM computes Request.block_hashes at a shared fine-grained size
            # (normally the GCD of heterogeneous group block sizes). Preserve
            # the value from the native pool before replacing it.
            native_block_pool = getattr(self, "block_pool", None)
            hash_block_size = getattr(
                native_block_pool,
                "hash_block_size",
                getattr(self, "hash_block_size", block_size),
            )
            self.block_pool = ElasticBlockPool(
                kv_cache_config.num_blocks,
                block_size,
                cell_size=cell_size,
                num_layers=group_size,
                enable_caching=getattr(self, "enable_caching", False),
                num_kv_buffers=num_kv_buffers,
                max_cached_blocks=_get_max_cached_blocks(block_size),
                hash_block_size=hash_block_size,
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
            kv_cache_shape = _get_kv_cache_shape_compat(
                self.attn_backend,
                num_blocks,
                kv_cache_spec.block_size,
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.head_size,
                _cache_dtype_str(self),
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

            cache_dtype = _cache_dtype_str(self)
            kv_cache_shape = _get_kv_cache_shape_compat(
                attn_backend_cls,
                num_blocks,
                kv_cache_spec.block_size,
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.head_size,
                cache_dtype,
            )

            if attention_type == "HYBRID_LINEAR":
                # The unified-pool layout math (both layouts; load-bearing for
                # contiguous ratio>1 linearity) assumes the spec's page is
                # EXACTLY the geometric K+V bytes of one block:
                #   page_size_bytes == 2 * block_size * H * D * itemsize.
                # Quantized KV modes that inline per-token scales into the
                # page, or a padded attention page (page_size_padded), break
                # that identity silently -- fail loud instead of garbling.
                import math as _math
                _geom_page = (_math.prod(kv_cache_shape) // num_blocks *
                              kv_cache_spec.dtype.itemsize)
                if kv_cache_spec.page_size_bytes != _geom_page:
                    raise NotImplementedError(
                        "kvcached hybrid-linear requires the attention page "
                        "size to equal the geometric K+V block bytes, but "
                        f"page_size_bytes={kv_cache_spec.page_size_bytes} != "
                        f"{_geom_page}. This typically means a quantized KV "
                        "cache dtype with inline scales (e.g. "
                        "fp8_per_token_head) or a padded attention page, "
                        "which the unified hybrid pool does not support yet.")

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
            has_kernel_block_size_source = kernel_block_sizes is not None
            if kernel_block_sizes is None:
                prepare_kernel_block_sizes_method = getattr(
                    self, "_prepare_kernel_block_sizes", None
                )
                if prepare_kernel_block_sizes_method is not None:
                    has_kernel_block_size_source = True
                    kernel_block_sizes = prepare_kernel_block_sizes_method(
                        kv_cache_config
                    )

            if kernel_block_sizes is None:
                try:
                    from vllm.v1.worker.utils import (
                        prepare_kernel_block_sizes as prepare_kernel_block_sizes_fn,
                    )
                except ImportError:
                    pass
                else:
                    attn_groups = getattr(self, "attn_groups", None)
                    if attn_groups is not None:
                        has_kernel_block_size_source = True
                        kernel_block_sizes = prepare_kernel_block_sizes_fn(
                            kv_cache_config, attn_groups
                        )

            kernel_block_size = (
                kernel_block_sizes[first_attn_group_id]
                if kernel_block_sizes is not None
                and first_attn_group_id < len(kernel_block_sizes)
                else None)
            if kernel_block_size is None and has_kernel_block_size_source:
                raise RuntimeError(
                    "kvcached could not determine the vLLM kernel block size. "
                    "This value is required when vLLM splits a virtual KV block "
                    "into smaller kernel blocks; falling back to block_size can "
                    "produce invalid KV cache strides."
                )

            # Detect heterogeneous attention groups (Gemma-style: sliding-window
            # + full-attention groups with different (block_size, num_kv_heads,
            # head_size) but identical block_mem_size, validated above). For
            # those we allocate ONE uniform physical pool set and build a
            # per-group as_strided view; homogeneous / single-group models keep
            # the original fast path byte-for-byte.
            def _attn_geom(grp):
                s = grp.kv_cache_spec
                return (s.block_size, s.num_kv_heads, s.head_size)

            attn_group_list = [
                (gid, grp)
                for gid, grp in enumerate(kv_cache_config.kv_cache_groups)
                if _is_attention_spec(grp.kv_cache_spec)
            ]
            distinct_attn_geoms = {_attn_geom(grp) for _, grp in attn_group_list}
            # The relaxed validate gate admits attention groups with differing
            # geometry as long as block_mem_size matches. Per-group views are only
            # built for the pure-attention path; HYBRID_LINEAR (full attn + mamba)
            # reshape still binds all attention layers to the first group's
            # geometry, so heterogeneous attention geometry there would be wrong.
            # No known hybrid-linear model has that, but fail loud rather than
            # silently mis-stride.
            if attention_type == "HYBRID_LINEAR" and len(distinct_attn_geoms) > 1:
                raise NotImplementedError(
                    "kvcached does not support hybrid-linear (attention + mamba) "
                    "models with heterogeneous attention-group geometry "
                    f"({sorted(distinct_attn_geoms)})."
                )
            is_hetero = (attention_type != "HYBRID_LINEAR"
                         and len(distinct_attn_geoms) > 1)
            self._kvcached_attn_layer_views = None

            alloc_result = kvi.alloc_kv_cache(
                kv_cache_shape,
                kv_cache_spec.block_size,
                dtype,
                device_type,
                group_size,
                attention_type=attention_type,
                kv_layout="NHD",
                kernel_block_size=kernel_block_size,
                return_meta=is_hetero,
            )

            if attention_type == "HYBRID_LINEAR":
                kv_cache_raw_tensors, raw_info = alloc_result
                self._kvcached_mamba_raw_info = raw_info
            elif is_hetero:
                kv_cache_raw_tensors, meta = alloc_result
                # Build a per-group view over the shared physical pools using each
                # group's own (block_size, num_kv_heads, head_size). Both layouts
                # work at kernel_block_size == block_size: every group has the
                # same block_mem_size (validated above), so block N sits at the
                # same byte offset whichever group's view addresses it, and
                # build_kv_views derives the per-group shape/stride from that one
                # uniform block stride. build_kv_views still fails loud for
                # contiguous + kernel_block_size != block_size.
                layer_views: dict = {}
                for gid, grp in attn_group_list:
                    gspec = grp.kv_cache_spec
                    gbackend = patch_instance._get_version_specific_attention_backend(
                        self, kv_cache_group_id=gid
                    )
                    gshape = _get_kv_cache_shape_compat(
                        gbackend, num_blocks, gspec.block_size,
                        gspec.num_kv_heads, gspec.head_size,
                        _cache_dtype_str(self),
                    )
                    gkbs = (kernel_block_sizes[gid]
                            if kernel_block_sizes is not None
                            and gid < len(kernel_block_sizes) else None)
                    gviews, _ = kvi.build_kv_views(
                        meta["raw_kv_tensors"], gshape, gspec.block_size, dtype,
                        attention_type, meta["num_blocks_per_layer"],
                        meta["gpu_mem_bytes_per_layer_k_or_v"], meta["num_layers"],
                        kernel_block_size=gkbs,
                    )
                    for pool_idx, layer_name in enumerate(grp.layer_names):
                        layer_views[layer_name] = gviews[pool_idx]
                self._kvcached_attn_layer_views = layer_views
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
            # Per-group attention views for heterogeneous hybrids (Gemma). None
            # for homogeneous / single-group models, which use the raw-tensor
            # index mapping below unchanged.
            attn_layer_views = getattr(self, "_kvcached_attn_layer_views", None)

            for kv_cache_group in kv_cache_config.kv_cache_groups:
                kv_cache_spec = kv_cache_group.kv_cache_spec

                if _is_mamba_spec(kv_cache_spec):
                    if mamba_info is None:
                        raise RuntimeError(
                            "Mamba layers found but no raw buffer info "
                            "available from kvcached"
                        )
                    for pool_idx, layer_name in enumerate(kv_cache_group.layer_names):
                        if mamba_info.get("is_contiguous"):
                            state_tensors = _reshape_mamba_contiguous(
                                mamba_info, kv_cache_spec, pool_idx,
                                get_dtype_size,
                            )
                        else:
                            state_tensors = _reshape_mamba_non_contiguous(
                                mamba_info["buffers"][pool_idx],
                                kv_cache_spec, get_dtype_size,
                            )
                        kv_caches[layer_name] = state_tensors  # type: ignore[assignment]
                else:
                    for pool_idx, layer_name in enumerate(kv_cache_group.layer_names):
                        if attn_layer_views is not None and layer_name in attn_layer_views:
                            kv_caches[layer_name] = attn_layer_views[layer_name]
                        else:
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

        def _patched_reshape_kv(self, *args: Any, **kwargs: Any):
            if enable_kvcached():
                # vLLM <0.20:  _reshape_kv_cache_tensors(self, kv_cache_config, kv_cache_raw_tensors, ...)
                # vLLM >=0.20: _reshape_kv_cache_tensors(self, kv_cache_raw_tensors, kernel_block_sizes)
                #   -> the kv_cache_config arg was dropped; pull it from self.kv_cache_config.
                if args and hasattr(args[0], "kv_cache_groups"):
                    kv_cache_config, kv_cache_raw_tensors = args[0], args[1]
                else:
                    kv_cache_config = getattr(self, "kv_cache_config", None)
                    kv_cache_raw_tensors = args[0]
                return self._reshape_kv_cache_tensors_from_kvcached(
                    kv_cache_config, kv_cache_raw_tensors
                )
            return original_method(self, *args, **kwargs)

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


def _is_vllm_startup_memory_guard(error: ValueError) -> bool:
    """Return whether *error* is vLLM's initial whole-device memory guard."""
    message = str(error)
    return (
        "Free memory on device" in message
        and "on startup is less than desired GPU memory utilization" in message
    )


def _get_virtual_kv_capacity_bytes(init_snapshot: Any, cache_config: Any) -> int:
    """Derive the scheduler-visible KV capacity from virtual pool geometry.

    kvcached reserves a full-device-sized KV virtual address range and backs it
    lazily. Preserve vLLM's gpu_memory_utilization setting as the logical upper
    bound, but do not reduce it based on physical memory consumed by peers.
    """
    return math.ceil(
        init_snapshot.total_memory * cache_config.gpu_memory_utilization
    )


def _get_worker_total_memory_bytes(worker: Any) -> int:
    """Read device geometry without using mutable whole-device free memory."""
    import torch

    try:
        properties = torch.cuda.get_device_properties(worker.device)
        return int(properties.total_memory)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return int(torch.cuda.mem_get_info()[1])


def _should_profile_cudagraph_memory(worker: Any) -> bool:
    profile_cudagraph = getattr(
        worker.model_runner, "profile_cudagraph_memory", None
    )
    if not callable(profile_cudagraph):
        return False
    model_config = getattr(worker, "model_config", None)
    if bool(getattr(model_config, "enforce_eager", False)):
        return False

    vllm_config = getattr(worker, "vllm_config", None)
    compilation_config = getattr(vllm_config, "compilation_config", None)
    if compilation_config is not None and hasattr(
        compilation_config, "cudagraph_mode"
    ):
        cudagraph_mode = compilation_config.cudagraph_mode
        mode_name = str(
            getattr(cudagraph_mode, "name", cudagraph_mode)
        ).upper()
        if mode_name == "NONE":
            return False

    try:
        from vllm.platforms import current_platform

        is_cuda = getattr(current_platform, "is_cuda", None)
        if callable(is_cuda) and not bool(is_cuda()):
            return False
        is_rocm = getattr(current_platform, "is_rocm", None)
        if callable(is_rocm) and bool(is_rocm()):
            return False
    except ImportError:
        pass
    return True


def _get_process_local_torch_peak_bytes(device: Any) -> int:
    import torch

    accelerator = getattr(torch, "accelerator", None)
    memory_stats = getattr(accelerator, "memory_stats", None)
    if callable(memory_stats):
        return int(memory_stats(device).get("allocated_bytes.all.peak", 0))
    return int(torch.cuda.memory_stats()["allocated_bytes.all.peak"])


class GPUWorkerPatch(VersionAwarePatch, BasePatch):
    """Decouple kvcached virtual KV capacity from whole-device free memory."""

    library = "vllm"
    target_module = "vllm.v1.worker.gpu_worker"
    target_class = "Worker"
    patch_name = "gpu_worker"

    def apply(self, gpuworker_mod: types.ModuleType) -> bool:
        # Initialize version info
        if not self.initialize_version_info():
            return False

        # Apply version-specific patches
        init_device_patched = self.patch_worker_init_device(gpuworker_mod)
        memory_profile_patched = self.patch_worker_determine_available_memory(
            gpuworker_mod
        )
        return init_device_patched and memory_profile_patched

    @version_range(VLLM_ALL_RANGE)
    def patch_worker_init_device(self, gpuworker_mod: types.ModuleType) -> bool:
        """Patch Worker.init_device"""
        original_request_memory = getattr(gpuworker_mod, "request_memory", None)
        if original_request_memory is not None:
            if self._is_already_patched(original_request_memory, "request_memory"):
                self.logger.debug("gpu_worker.request_memory already patched")
                return True

            logger = self.logger

            def _patched_request_memory(init_snapshot: Any, cache_config: Any) -> int:
                if not enable_kvcached():
                    return int(original_request_memory(init_snapshot, cache_config))

                requested_memory = _get_virtual_kv_capacity_bytes(
                    init_snapshot, cache_config
                )
                if int(init_snapshot.free_memory) < requested_memory:
                    logger.warning(
                        "Ignoring vLLM's whole-device startup memory guard "
                        "because kvcached provides virtual KV capacity: free=%d "
                        "bytes, requested=%d bytes",
                        int(init_snapshot.free_memory),
                        requested_memory,
                    )
                return requested_memory

            self._mark_as_patched(_patched_request_memory, "request_memory")
            setattr(gpuworker_mod, "request_memory", _patched_request_memory)
            return True

        # vLLM releases before request_memory() was factored out perform the
        # same guard inside Worker.init_device(). Keep a narrow compatibility
        # fallback for those releases and never swallow unrelated ValueErrors.
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
                result = original_init_device(self, *args, **kwargs)
            except ValueError as e:
                if not _is_vllm_startup_memory_guard(e):
                    raise
                # If the original impl still raises due to insufficient memory,
                # replicate the remainder of its logic while skipping the guard.
                logger.warning(
                    "Ignoring vLLM's whole-device startup memory guard because "
                    "kvcached provides virtual KV capacity: %s",
                    e,
                )

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

                # Set init_snapshot and requested_memory so later vLLM code can
                # access a coherent logical budget. Do not use current free
                # memory: it includes allocations made by colocated engines.
                if not hasattr(self, "init_snapshot"):
                    self.init_snapshot = MemorySnapshot(device=self.device)
                if not hasattr(self, "requested_memory"):
                    self.requested_memory = _get_virtual_kv_capacity_bytes(
                        self.init_snapshot, self.cache_config
                    )

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

            # vLLM 0.8.x predates MemorySnapshot/request_memory but is still
            # supported by kvcached. Persist the same logical budget after its
            # original init_device() succeeds so the patched profiler can use
            # process-local accounting.
            if not hasattr(self, "requested_memory"):
                total_memory = _get_worker_total_memory_bytes(self)
                snapshot = types.SimpleNamespace(total_memory=total_memory)
                self.requested_memory = _get_virtual_kv_capacity_bytes(
                    snapshot, self.cache_config
                )
            return result

        self._mark_as_patched(_patched_init_device, "init_device")
        Worker.init_device = _patched_init_device  # type: ignore[assignment]
        return True

    @version_range(VLLM_ALL_RANGE)
    def patch_worker_determine_available_memory(
        self, gpuworker_mod: types.ModuleType
    ) -> bool:
        """Use vLLM's explicit-capacity branch with an automatic virtual budget."""
        Worker = self._get_target_class(gpuworker_mod)
        if Worker is None:
            return False

        original_determine = getattr(Worker, "determine_available_memory", None)
        if original_determine is None:
            self.logger.error("Worker.determine_available_memory was not found")
            return False
        if self._is_already_patched(original_determine, "determine_available_memory"):
            self.logger.debug("Worker.determine_available_memory already patched")
            return True

        import torch

        logger = self.logger

        @torch.inference_mode()
        def _patched_determine_available_memory(
            self, *args: Any, **kwargs: Any
        ) -> int:
            if not enable_kvcached():
                return original_determine(self, *args, **kwargs)

            cache_config = self.cache_config
            configured_budget = getattr(cache_config, "kv_cache_memory_bytes", None)
            if configured_budget is not None:
                return original_determine(self, *args, **kwargs)

            virtual_budget = int(self.requested_memory)
            init_snapshot = getattr(self, "init_snapshot", None)
            if init_snapshot is None:
                # vLLM 0.8.x has no MemorySnapshot. Resetting peak stats after
                # model load makes this peak process-local and includes both
                # resident model weights and the profiling activation peak.
                import torch

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                self.model_runner.profile_run()
                weights_memory = 0
                torch_peak_increase = int(
                    torch.cuda.memory_stats()["allocated_bytes.all.peak"]
                )
                cudagraph_memory_estimate = 0
            else:
                from vllm.utils.mem_utils import memory_profiling

                weights_memory = int(self.model_runner.model_memory_usage)
                profile_torch_peak = None
                cudagraph_memory_estimate = 0
                with memory_profiling(
                    init_snapshot, weights_memory=weights_memory
                ) as profile_result:
                    self.model_runner.profile_run()
                    if _should_profile_cudagraph_memory(self):
                        profile_torch_peak = _get_process_local_torch_peak_bytes(
                            self.device
                        )
                        cudagraph_memory_estimate = int(
                            self.model_runner.profile_cudagraph_memory()
                        )

                torch_peak_increase = int(profile_result.torch_peak_increase)
                if profile_torch_peak is not None:
                    before_profile = getattr(profile_result, "before_profile", None)
                    before_torch_peak = getattr(
                        before_profile, "torch_peak", None
                    )
                    if before_torch_peak is not None:
                        torch_peak_increase = max(
                            0, profile_torch_peak - int(before_torch_peak)
                        )
            available_memory = (
                virtual_budget
                - weights_memory
                - torch_peak_increase
            )

            self.available_kv_cache_memory_bytes = available_memory
            # vLLM 0.24 reads this field during compile_or_warm_up_model().
            # Keep the worker contract without reintroducing the device-wide
            # non-torch delta that colocated processes can corrupt.
            self.non_torch_memory = 0
            self.peak_activation_memory = torch_peak_increase
            self.cudagraph_memory_estimate = cudagraph_memory_estimate
            logger.warning(
                "Using kvcached process-local KV capacity: budget=%d bytes, "
                "weights=%d bytes, torch_peak=%d bytes, cudagraph=%d bytes "
                "(ignored), available=%d bytes; "
                "whole-device non-torch and CUDA Graph deltas are ignored",
                virtual_budget,
                weights_memory,
                torch_peak_increase,
                cudagraph_memory_estimate,
                available_memory,
            )
            return available_memory

        self._mark_as_patched(
            _patched_determine_available_memory, "determine_available_memory"
        )
        Worker.determine_available_memory = _patched_determine_available_memory
        return True


class KVCacheManagerAllocateSlotsPatch(VersionAwarePatch, BasePatch):
    """Report an exhausted physical KV pool the way vLLM's scheduler expects.

    vLLM's own block pool can raise from `get_new_blocks()` because its free
    count is process-local and authoritative: if the count says the blocks are
    there, the allocation cannot fail, so the raise is an invariant guard that
    never fires. Under kvcached the same count reads device-wide state shared
    with colocated engines, so it is a snapshot rather than a reservation, and
    a peer can take the last pages before they are claimed. The guard becomes
    reachable.

    The scheduler already handles this exact situation -- `allocate_slots()`
    returning None means "not now", and it preempts a running request and
    retries on the next step, which incidentally releases physical pages back
    to the shared pool. What it does not handle is an exception: `schedule()`
    catches nothing and EngineCore's own handler wraps only `execute_model`,
    so the exception terminates the engine and every in-flight request with
    it.

    Translate only `KVCachePoolExhausted`. A plain ValueError from the pool
    (asking for more blocks than were just reported free) is a contract
    violation and must stay fail-loud.
    """

    library = "vllm"
    target_module = "vllm.v1.core.kv_cache_manager"
    target_class = "KVCacheManager"
    patch_name = "allocate_slots"

    def apply(self, kvcache_manager_mod: types.ModuleType) -> bool:
        if not self.initialize_version_info():
            return False
        return self.patch_allocate_slots(kvcache_manager_mod)

    @version_range(VLLM_ALL_RANGE)
    def patch_allocate_slots(self, kvcache_manager_mod: types.ModuleType) -> bool:
        KVCacheManager = self._get_target_class(kvcache_manager_mod)
        if KVCacheManager is None:
            return False

        original_allocate_slots = getattr(KVCacheManager, "allocate_slots", None)
        if original_allocate_slots is None:
            self.logger.warning(
                "KVCacheManager.allocate_slots was not found; an exhausted "
                "physical KV pool will terminate EngineCore")
            return False
        if self._is_already_patched(original_allocate_slots, "allocate_slots"):
            self.logger.debug("KVCacheManager.allocate_slots already patched")
            return True

        logger = self.logger

        def _patched_allocate_slots(self, *args: Any, **kwargs: Any) -> Any:
            if not enable_kvcached():
                return original_allocate_slots(self, *args, **kwargs)
            try:
                return original_allocate_slots(self, *args, **kwargs)
            except KVCachePoolExhausted as exhausted:
                # None is the scheduler's own "cannot schedule this request
                # now" path. Partially allocated blocks are released when the
                # scheduler preempts or frees the request, so returning here
                # does not strand them.
                logger.warning(
                    "Shared physical KV pool is exhausted; reporting a "
                    "scheduling miss so the engine can preempt and retry: %s",
                    exhausted)
                return None

        self._mark_as_patched(_patched_allocate_slots, "allocate_slots")
        KVCacheManager.allocate_slots = _patched_allocate_slots  # type: ignore[assignment]
        return True


class TritonAttentionPatch(VersionAwarePatch, BasePatch):
    """Build the per-token-head scale views from the KV tensor, not raw storage.

    ``TritonAttentionImpl._ensure_scale_caches`` carves the per-head scale
    planes out of ``kv_cache.untyped_storage()`` under a hard-coded dense
    layout, ignoring both ``stride()`` and ``storage_offset()``. Every kvcached
    KV tensor is a strided view: K and V occupy separate halves of the layer
    buffer, and in the contiguous layout each layer is a slice of one shared
    buffer at a non-zero offset. The computed addresses are therefore wrong --
    below the K/V split they land on some other block's scale padding (a
    consistent, harmless relabeling), above it they land on KV data and corrupt
    it, and in the contiguous layout every layer's scale plane collapses onto
    layer 0's.

    Slicing the tensor carries the strides and the storage offset along, so the
    addresses follow whatever layout is actually in use. For vLLM's own dense
    tensors this yields exactly the same views as upstream. See #424 / #434.
    """

    library = "vllm"
    target_module = "vllm.v1.attention.backends.triton_attn"
    target_class = "TritonAttentionImpl"
    patch_name = "triton_attention_scales"

    def apply(self, triton_attn_mod: types.ModuleType) -> bool:
        if not self.initialize_version_info():
            return False
        return self.patch_ensure_scale_caches(triton_attn_mod)

    @version_range(VLLM_V9_PLUS_RANGE)
    def patch_ensure_scale_caches(self,
                                  triton_attn_mod: types.ModuleType) -> bool:
        impl_cls = self._get_target_class(triton_attn_mod)
        if impl_cls is None:
            return False
        if not hasattr(impl_cls, "_ensure_scale_caches"):
            # vLLM without per-token-head KV quantization: nothing to patch.
            self.logger.debug("TritonAttentionImpl has no _ensure_scale_caches")
            return True
        if self._is_already_patched(impl_cls, "ensure_scale_caches"):
            return True

        def _ensure_scale_caches(self, kv_cache: Any) -> None:
            import torch

            if self._k_scale_cache is not None:
                return
            # kv_cache is (num_blocks, 2, block_size, num_kv_heads, padded_hs);
            # the last ``scale_pad`` elements of each head hold one float32.
            padded_hs = kv_cache.shape[-1]
            head_size = padded_hs - 4 // kv_cache.element_size()
            self._k_scale_cache = (kv_cache[:, 0, :, :, head_size:].view(
                torch.float32).squeeze(-1))
            self._v_scale_cache = (kv_cache[:, 1, :, :, head_size:].view(
                torch.float32).squeeze(-1))
            self._k_scale_cache.fill_(1.0)
            self._v_scale_cache.fill_(1.0)

        self._mark_as_patched(impl_cls, "ensure_scale_caches")
        impl_cls._ensure_scale_caches = _ensure_scale_caches
        return True

# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
SGLang-specific patches using unified patch infrastructure.
"""

import inspect
import math
import types
from typing import Any, List, Optional, Tuple, Union, cast

from kvcached.integration.patch_base import BasePatch, enable_kvcached
from kvcached.integration.version_utils import VersionAwarePatch, version_range
from kvcached.utils import MAX_CACHED_TOKENS, get_kvcached_logger

BYTES_PER_GB = 1024**3

# Version ranges for SGLang support
SGLANG_ALL_RANGE = ">=0.4.9"  # All supported versions

logger = get_kvcached_logger()


class ElasticAllocatorPatch(VersionAwarePatch, BasePatch):
    """Inject ElasticTokenToKVPoolAllocator into SGLang's allocator module"""

    library = "sglang"
    target_module = "sglang.srt.mem_cache.allocator"
    patch_name = "elastic_allocator"

    def apply(self, alloc_mod: types.ModuleType) -> bool:
        # Initialize version info
        if not self.initialize_version_info():
            return False

        # Apply version-specific patches
        success = self.inject_elastic_allocator(alloc_mod)
        if success:
            success &= self.alias_allocator_to_elastic(alloc_mod)

        # Also inject and alias the paged variant for page_size > 1
        paged_success = self.inject_elastic_paged_allocator(alloc_mod)
        if paged_success:
            paged_success &= self.alias_paged_allocator_to_elastic(alloc_mod)
        success &= paged_success

        if success:
            logger.info(
                "Elastic allocators patched (TokenToKVPool + PagedTokenToKVPool)"
            )

        return success

    @version_range(SGLANG_ALL_RANGE)
    def inject_elastic_allocator(self, alloc_mod: types.ModuleType) -> bool:
        """Inject ElasticTokenToKVPoolAllocator"""
        if hasattr(alloc_mod, "ElasticTokenToKVPoolAllocator"):
            self.logger.debug("ElasticTokenToKVPoolAllocator already exists")
            return True

        try:
            import torch

            BaseTokenToKVPoolAllocator = getattr(alloc_mod, "BaseTokenToKVPoolAllocator")

            class ElasticTokenToKVPoolAllocator(
                BaseTokenToKVPoolAllocator  # type: ignore[misc, valid-type]
            ):
                def __init__(self, size: int, dtype, device: str, kvcache, *args, **kwargs) -> None:
                    super().__init__(size, 1, dtype, device, kvcache, *args, **kwargs)
                    if not hasattr(kvcache, "kvcached_allocator"):
                        raise ValueError("ElasticTokenToKVPoolAllocator requires elastic MHA pool")
                    if "cuda" not in device:
                        raise ValueError("ElasticTokenToKVPoolAllocator only supports cuda device")
                    self.kvcached_allocator = kvcache.kvcached_allocator
                    logger.info(
                        f"[kvcached] ElasticTokenToKVPoolAllocator in use: size={size} "
                        "(page_size=1 path)"
                    )

                def available_size(self):
                    # Cap at the pool's token capacity.  KVCacheManager
                    # internally manages size+1 blocks (the extra slot is
                    # the null/padding block) and considers physical GPU
                    # memory availability, so its reported available_size
                    # can slightly exceed the pool's declared size.
                    # SGLang's SWATokenToKVPoolAllocator asserts
                    # available <= size.
                    return min(self.kvcached_allocator.available_size(),
                               self.size)

                def alloc(self, need_size: int):
                    indices: list[int] = self.kvcached_allocator.alloc(need_size)
                    return torch.tensor(indices, dtype=torch.int64, device=self.device)

                def free(self, free_index):
                    if self.is_not_in_free_group:
                        try:
                            indices: list[int] = free_index.cpu().numpy().tolist()
                        except Exception:
                            indices = list(free_index)
                        return self.kvcached_allocator.free(indices)
                    else:
                        self.free_group.append(free_index)

                def clear(self):
                    if hasattr(self, "kvcached_allocator"):
                        self.kvcached_allocator.clear()

            setattr(alloc_mod, "ElasticTokenToKVPoolAllocator", ElasticTokenToKVPoolAllocator)
            return True
        except Exception as e:
            self.logger.error(f"Failed to inject ElasticTokenToKVPoolAllocator: {e}")
            return False

    @version_range(SGLANG_ALL_RANGE)
    def alias_allocator_to_elastic(self, alloc_mod: types.ModuleType) -> bool:
        """Alias TokenToKVPoolAllocator to ElasticTokenToKVPoolAllocator"""
        if self._is_already_patched(alloc_mod, "__kvcached_allocator_aliased__"):
            return True

        try:
            ElasticTokenToKVPoolAllocator = getattr(alloc_mod, "ElasticTokenToKVPoolAllocator")
            if ElasticTokenToKVPoolAllocator is None:
                return False
            alloc_mod.TokenToKVPoolAllocator = ElasticTokenToKVPoolAllocator  # type: ignore
            self._mark_as_patched(alloc_mod, "__kvcached_allocator_aliased__")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to alias allocator to elastic one: {e}")
            return False

    @version_range(SGLANG_ALL_RANGE)
    def inject_elastic_paged_allocator(self, alloc_mod: types.ModuleType) -> bool:
        """Inject ElasticPagedTokenToKVPoolAllocator for page_size > 1"""
        if hasattr(alloc_mod, "ElasticPagedTokenToKVPoolAllocator"):
            self.logger.debug("ElasticPagedTokenToKVPoolAllocator already exists")
            return True

        try:
            import torch

            BaseTokenToKVPoolAllocator = getattr(alloc_mod, "BaseTokenToKVPoolAllocator")
            alloc_extend_kernel = getattr(alloc_mod, "alloc_extend_kernel")
            alloc_decode_kernel = getattr(alloc_mod, "alloc_decode_kernel")

            from sglang.srt.utils import get_num_new_pages, next_power_of_2

            class ElasticPagedTokenToKVPoolAllocator(
                BaseTokenToKVPoolAllocator  # type: ignore[misc, valid-type]
            ):
                def __init__(
                    self, size: int, page_size: int, dtype, device: str, kvcache, *args, **kwargs
                ) -> None:
                    super().__init__(size, page_size, dtype, device, kvcache, *args, **kwargs)
                    if not hasattr(kvcache, "kvcached_allocator"):
                        raise ValueError(
                            "ElasticPagedTokenToKVPoolAllocator requires elastic MHA pool"
                        )
                    if "cuda" not in device:
                        raise ValueError(
                            "ElasticPagedTokenToKVPoolAllocator only supports cuda device"
                        )
                    self.kvcached_allocator = kvcache.kvcached_allocator
                    self.num_pages = size // page_size
                    self.seen_max_num_extend_tokens_next_power_of_2 = 1
                    logger.info(
                        f"[kvcached] ElasticPagedTokenToKVPoolAllocator in use: size={size}, "
                        f"page_size={page_size}"
                    )
                    # Base class expects these tensors for backup_state / free_group_end
                    self.free_pages = torch.empty((0,), dtype=torch.int64, device=self.device)
                    self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)

                def available_size(self):
                    return self.kvcached_allocator.available_size() * self.page_size

                def alloc(self, need_size: int):
                    num_pages = need_size // self.page_size
                    block_ids = self.kvcached_allocator.alloc(num_pages)
                    if block_ids is None:
                        return None
                    page_ids = torch.tensor(block_ids, dtype=torch.int64, device=self.device)
                    out_indices = (
                        page_ids[:, None] * self.page_size
                        + torch.arange(self.page_size, device=self.device)
                    ).reshape(-1)
                    return out_indices

                def alloc_extend(
                    self,
                    prefix_lens: torch.Tensor,
                    prefix_lens_cpu: torch.Tensor,
                    seq_lens: torch.Tensor,
                    seq_lens_cpu: torch.Tensor,
                    last_loc: torch.Tensor,
                    extend_num_tokens: int,
                ):
                    self.seen_max_num_extend_tokens_next_power_of_2 = max(
                        self.seen_max_num_extend_tokens_next_power_of_2,
                        next_power_of_2(extend_num_tokens),
                    )
                    bs = len(prefix_lens)

                    num_new_pages = get_num_new_pages(
                        seq_lens=seq_lens_cpu,
                        page_size=self.page_size,
                        prefix_lens=prefix_lens_cpu,
                    )

                    if num_new_pages > 0:
                        block_ids = self.kvcached_allocator.alloc(num_new_pages)
                        if block_ids is None:
                            return None
                        free_pages = torch.tensor(
                            block_ids, dtype=torch.int64, device=self.device
                        )
                    else:
                        free_pages = torch.empty((0,), dtype=torch.int64, device=self.device)

                    out_indices = torch.empty(
                        (extend_num_tokens,), dtype=torch.int64, device=self.device
                    )
                    alloc_extend_kernel[(bs,)](
                        prefix_lens,
                        seq_lens,
                        last_loc,
                        free_pages,
                        out_indices,
                        next_power_of_2(bs),
                        self.page_size,
                        self.seen_max_num_extend_tokens_next_power_of_2,
                    )
                    return out_indices

                def alloc_decode(
                    self,
                    seq_lens: torch.Tensor,
                    seq_lens_cpu: torch.Tensor,
                    last_loc: torch.Tensor,
                ):
                    bs = len(seq_lens)

                    num_new_pages = get_num_new_pages(
                        seq_lens=seq_lens_cpu,
                        page_size=self.page_size,
                        decode=True,
                    )

                    if num_new_pages > 0:
                        block_ids = self.kvcached_allocator.alloc(num_new_pages)
                        if block_ids is None:
                            return None
                        free_pages = torch.tensor(
                            block_ids, dtype=torch.int64, device=self.device
                        )
                    else:
                        free_pages = torch.empty((0,), dtype=torch.int64, device=self.device)

                    out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
                    alloc_decode_kernel[(bs,)](
                        seq_lens,
                        last_loc,
                        free_pages,
                        out_indices,
                        next_power_of_2(bs),
                        self.page_size,
                    )
                    return out_indices

                def free(self, free_index):
                    if free_index.numel() == 0:
                        return

                    if self.is_not_in_free_group:
                        page_ids = torch.unique(free_index // self.page_size)
                        try:
                            indices: list[int] = page_ids.cpu().numpy().tolist()
                        except Exception:
                            indices = list(page_ids)
                        return self.kvcached_allocator.free(indices)
                    else:
                        self.free_group.append(free_index)

                def clear(self):
                    if hasattr(self, "kvcached_allocator"):
                        self.kvcached_allocator.clear()
                    self.free_pages = torch.empty(
                        (0,), dtype=torch.int64, device=self.device
                    )
                    self.release_pages = torch.empty(
                        (0,), dtype=torch.int64, device=self.device
                    )
                    self.is_not_in_free_group = True
                    self.free_group = []

                def merge_and_sort_free(self):
                    pass  # No-op: kvcached manages the free list

            setattr(
                alloc_mod,
                "ElasticPagedTokenToKVPoolAllocator",
                ElasticPagedTokenToKVPoolAllocator,
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to inject ElasticPagedTokenToKVPoolAllocator: {e}")
            return False

    @version_range(SGLANG_ALL_RANGE)
    def alias_paged_allocator_to_elastic(self, alloc_mod: types.ModuleType) -> bool:
        """Alias PagedTokenToKVPoolAllocator to ElasticPagedTokenToKVPoolAllocator"""
        if self._is_already_patched(alloc_mod, "__kvcached_paged_allocator_aliased__"):
            return True

        try:
            ElasticPagedTokenToKVPoolAllocator = getattr(
                alloc_mod, "ElasticPagedTokenToKVPoolAllocator"
            )
            if ElasticPagedTokenToKVPoolAllocator is None:
                return False
            alloc_mod.PagedTokenToKVPoolAllocator = ElasticPagedTokenToKVPoolAllocator  # type: ignore
            self._mark_as_patched(alloc_mod, "__kvcached_paged_allocator_aliased__")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to alias paged allocator to elastic one: {e}")
            return False


class ElasticMemoryPoolPatch(VersionAwarePatch, BasePatch):
    """Inject ElasticMHATokenToKVPool into SGLang's memory pool module"""

    library = "sglang"
    target_module = "sglang.srt.mem_cache.memory_pool"
    patch_name = "elastic_memory_pool"

    def apply(self, mem_pool_mod: types.ModuleType) -> bool:
        # Initialize version info
        if not self.initialize_version_info():
            return False

        # Apply version-specific patches
        success = self.inject_elastic_mem_pool(mem_pool_mod)
        if success:
            success &= self.alias_mem_pool_to_elastic(mem_pool_mod)
        return success

    @version_range(SGLANG_ALL_RANGE)
    def inject_elastic_mem_pool(self, mem_pool_mod: types.ModuleType) -> bool:
        """Inject ElasticMHATokenToKVPool"""
        if hasattr(mem_pool_mod, "ElasticMHATokenToKVPool"):
            self.logger.debug("ElasticMHATokenToKVPool already exists")
            return True

        try:
            MHATokenToKVPool = getattr(mem_pool_mod, "MHATokenToKVPool")

            class ElasticMHATokenToKVPool(MHATokenToKVPool):  # type: ignore
                # Auto-incrementing group_id so that each pool instance
                # (e.g., full-attention pool and SWA pool in SWAKVPool)
                # gets independent FTensors and page spaces in the C++
                # FTensorAllocator.
                _next_group_id = 0

                def __init__(
                    self,
                    size: int,
                    page_size: int,
                    dtype,
                    head_num: int,
                    head_dim: int,
                    layer_num: int,
                    device: str,
                    enable_memory_saver: bool,
                    start_layer: Union[int, None] = None,
                    end_layer: Union[int, None] = None,
                    *args,
                    **kwargs,
                ) -> None:
                    # Assign group_id BEFORE super().__init__() because it
                    # calls _create_buffers() which needs self._group_id.
                    self._group_id = ElasticMHATokenToKVPool._next_group_id
                    ElasticMHATokenToKVPool._next_group_id += 1

                    super().__init__(
                        size=size,
                        page_size=page_size,
                        dtype=dtype,
                        head_num=head_num,
                        head_dim=head_dim,
                        layer_num=layer_num,
                        device=device,
                        enable_memory_saver=enable_memory_saver,
                        start_layer=start_layer,
                        end_layer=end_layer,
                        *args,
                        **kwargs,
                    )
                    import kvcached.integration.sglang.interfaces as kvi

                    self.cell_size = self.head_num * self.head_dim * dtype.itemsize
                    self.kvcached_allocator = kvi.get_kv_cache_manager(
                        math.ceil(size / page_size) + 1, page_size, self.cell_size, layer_num,
                        group_id=self._group_id,
                    )

                    k_size, v_size = self.get_kv_size_bytes()
                    k_size_phy, v_size_phy = self.get_kv_size_bytes_phy()

                    logger.info(
                        f"VirtualKV Cache is allocated (group_id={self._group_id}). "
                        f"#tokens: {size}, #layers: {layer_num}, K size: "
                        f"{k_size / BYTES_PER_GB:.2f} GB, V size: {v_size / BYTES_PER_GB:.2f} GB"
                    )
                    logger.info(
                        f"Physical KV Cache limits by --mem-fraction-static: "
                        f"#tokens: {size}, K size: "
                        f"{k_size_phy / BYTES_PER_GB:.2f} GB, V size: {v_size_phy / BYTES_PER_GB:.2f} GB"
                    )

                    self.mem_usage = (k_size + v_size) / BYTES_PER_GB

                def _create_buffers(self):
                    import kvcached.integration.sglang.interfaces as kvi

                    # Resolve TP rank and size for IPC socket registration.
                    # SGLang workers each call _create_buffers() independently,
                    # so we query the distributed state at this point (which is
                    # guaranteed to be initialised by the time buffers are created).
                    try:
                        from sglang.srt.distributed import (
                            get_pipeline_model_parallel_rank,
                            get_tensor_model_parallel_rank,
                            get_tensor_model_parallel_world_size,
                        )
                        tp_rank = int(get_tensor_model_parallel_rank())
                        tp_size = int(get_tensor_model_parallel_world_size())
                        pp_rank = int(get_pipeline_model_parallel_rank())
                    except (ImportError, AttributeError):
                        try:
                            import torch.distributed as dist
                            tp_rank = dist.get_rank() if dist.is_initialized() else 0
                            tp_size = dist.get_world_size() if dist.is_initialized() else 1
                            pp_rank = 0
                        except (ImportError, AttributeError, RuntimeError, ValueError, TypeError):
                            tp_rank, tp_size, pp_rank = 0, 1, 0

                    # Initialize kvcached with overlap scheduling to be conservative
                    kvi.init_kvcached(tp_rank=tp_rank, world_size=tp_size, pp_rank=pp_rank, async_sched=True)

                    if "cuda" not in self.device:
                        raise ValueError("ElasticMHATokenToKVPool only supports cuda device")
                    _kv_mha: Tuple[List[Any], List[Any]] = cast(
                        Tuple[List[Any], List[Any]],
                        kvi.alloc_kv_cache(
                            kvcache_shape=(
                                self.size + self.page_size,
                                self.head_num,
                                self.head_dim,
                            ),
                            dtype=self.dtype,
                            device=self.device,
                            num_layers=self.layer_num,
                            page_size=self.page_size,
                            attention_type="MHA",
                            kv_layout="NHD",
                            group_id=self._group_id,
                        ),
                    )
                    self.k_buffer, self.v_buffer = _kv_mha

                def get_kv_size_bytes_phy(self):
                    """Return the physical memory limits of the K/V buffers.

                    This limit is enforced by `--mem-fraction-static` option.
                    """
                    total_tokens = self.size + self.page_size
                    elems_per_token = self.head_num * self.head_dim
                    bytes_per_elem = self.dtype.itemsize

                    k_size_bytes = self.layer_num * total_tokens * elems_per_token * bytes_per_elem
                    v_size_bytes = k_size_bytes

                    return k_size_bytes, v_size_bytes

            setattr(mem_pool_mod, "ElasticMHATokenToKVPool", ElasticMHATokenToKVPool)
            return True
        except Exception as e:
            self.logger.error(f"Failed to inject ElasticMHATokenToKVPool: {e}")
            return False

    @version_range(SGLANG_ALL_RANGE)
    def alias_mem_pool_to_elastic(self, mem_pool_mod: types.ModuleType) -> bool:
        """Alias MHATokenToKVPool to ElasticMHATokenToKVPool"""
        if self._is_already_patched(mem_pool_mod, "__kvcached_mempool_aliased__"):
            return True

        try:
            ElasticMHATokenToKVPool = getattr(mem_pool_mod, "ElasticMHATokenToKVPool")
            if ElasticMHATokenToKVPool is None:
                return False
            # Alias defaults so core code will use elastic variants
            mem_pool_mod.MHATokenToKVPool = ElasticMHATokenToKVPool  # type: ignore
            self._mark_as_patched(mem_pool_mod, "__kvcached_mempool_aliased__")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to alias memory_pool to elastic one: {e}")
            return False


class ElasticMLAMemoryPoolPatch(VersionAwarePatch, BasePatch):
    """Inject ElasticMLATokenToKVPool into SGLang's memory pool module"""

    library = "sglang"
    target_module = "sglang.srt.mem_cache.memory_pool"
    patch_name = "elastic_mla_memory_pool"

    def apply(self, mem_pool_mod: types.ModuleType) -> bool:
        # Initialize version info
        if not self.initialize_version_info():
            return False

        # Apply version-specific patches
        success = self.inject_elastic_mla_mem_pool(mem_pool_mod)
        if success:
            success &= self.alias_mla_mem_pool_to_elastic(mem_pool_mod)
        return success

    @version_range(SGLANG_ALL_RANGE)
    def inject_elastic_mla_mem_pool(self, mem_pool_mod: types.ModuleType) -> bool:
        """Inject ElasticMLATokenToKVPool"""
        if hasattr(mem_pool_mod, "ElasticMLATokenToKVPool"):
            self.logger.debug("ElasticMLATokenToKVPool already exists")
            return True

        try:
            import torch

            MLATokenToKVPool = getattr(mem_pool_mod, "MLATokenToKVPool")
            KVCache = getattr(mem_pool_mod, "KVCache")

            class ElasticMLATokenToKVPool(MLATokenToKVPool):  # type: ignore
                def __init__(
                    self,
                    size: int,
                    page_size: int,
                    dtype,
                    kv_lora_rank: int,
                    qk_rope_head_dim: int,
                    layer_num: int,
                    device: str,
                    enable_memory_saver: bool,
                    start_layer: Union[int, None] = None,
                    end_layer: Union[int, None] = None,
                    *args,
                    **kwargs,
                ) -> None:
                    # Skip MLATokenToKVPool.__init__ which inlines torch.zeros
                    # buffer allocation. Call KVCache.__init__ directly and set
                    # MLA-specific attributes ourselves.
                    KVCache.__init__(
                        self,
                        size,
                        page_size,
                        dtype,
                        layer_num,
                        device,
                        enable_memory_saver,
                        start_layer,
                        end_layer,
                    )

                    # MLA-specific attributes (mirroring MLATokenToKVPool)
                    self.kv_lora_rank = kv_lora_rank
                    self.qk_rope_head_dim = qk_rope_head_dim
                    self.use_nsa = kwargs.get("use_nsa", False)
                    self.nsa_kv_cache_store_fp8 = (
                        self.use_nsa and dtype == torch.float8_e4m3fn
                    )
                    override_kv_cache_dim = kwargs.get("override_kv_cache_dim", None)
                    self.kv_cache_dim = (
                        override_kv_cache_dim
                        if self.use_nsa and self.nsa_kv_cache_store_fp8
                        else (kv_lora_rank + qk_rope_head_dim)
                    )
                    # Attributes from parent that we skip but inherited methods may need
                    self.custom_mem_pool = None

                    import kvcached.integration.sglang.interfaces as kvi

                    # Initialize kvcached and create virtual memory buffers.
                    # Resolve TP rank and size so IPC sockets are registered correctly
                    # across all TP workers in this SGLang instance.
                    try:
                        from sglang.srt.distributed import (
                            get_pipeline_model_parallel_rank,
                            get_tensor_model_parallel_rank,
                            get_tensor_model_parallel_world_size,
                        )
                        tp_rank = int(get_tensor_model_parallel_rank())
                        tp_size = int(get_tensor_model_parallel_world_size())
                        pp_rank = int(get_pipeline_model_parallel_rank())
                    except Exception:
                        try:
                            import torch.distributed as dist
                            tp_rank = dist.get_rank() if dist.is_initialized() else 0
                            tp_size = dist.get_world_size() if dist.is_initialized() else 1
                            pp_rank = 0
                        except Exception:
                            tp_rank, tp_size, pp_rank = 0, 1, 0

                    kvi.init_kvcached(tp_rank=tp_rank, world_size=tp_size, pp_rank=pp_rank, async_sched=True)

                    if "cuda" not in device:
                        raise ValueError("ElasticMLATokenToKVPool only supports cuda device")
                    self.kv_buffer = cast(
                        List[torch.Tensor],
                        kvi.alloc_kv_cache(
                            kvcache_shape=(
                                size + page_size,
                                1,
                                self.kv_cache_dim,
                            ),
                            dtype=dtype,
                            device=device,
                            num_layers=layer_num,
                            page_size=page_size,
                            attention_type="MLA",
                        ),
                    )

                    self.data_ptrs = torch.tensor(
                        [x.data_ptr() for x in self.kv_buffer],
                        dtype=torch.uint64,
                        device=self.device,
                    )

                    self.cell_size = (kv_lora_rank + qk_rope_head_dim) * dtype.itemsize
                    self.kvcached_allocator = kvi.get_kv_cache_manager(
                        size + page_size, page_size, self.cell_size, layer_num,
                        num_kv_buffers=1,
                    )

                    kv_size = self.get_kv_size_bytes()
                    kv_size_phy = self.get_kv_size_bytes_phy()

                    logger.info(
                        f"VirtualKV Cache is allocated. #tokens: {size}, "
                        f"KV size: {kv_size / BYTES_PER_GB:.2f} GB"
                    )
                    logger.info(
                        f"Physical KV Cache limits by --mem-fraction-static: "
                        f"#tokens: {size}, KV size: {kv_size_phy / BYTES_PER_GB:.2f} GB"
                    )

                    self.mem_usage = kv_size / BYTES_PER_GB

                def get_kv_size_bytes_phy(self):
                    """Return the physical memory limits of the KV buffer."""
                    total_tokens = self.size + self.page_size
                    elems_per_token = self.kv_cache_dim
                    bytes_per_elem = self.dtype.itemsize

                    return self.layer_num * total_tokens * elems_per_token * bytes_per_elem

            setattr(mem_pool_mod, "ElasticMLATokenToKVPool", ElasticMLATokenToKVPool)
            return True
        except Exception as e:
            self.logger.error(f"Failed to inject ElasticMLATokenToKVPool: {e}")
            return False

    @version_range(SGLANG_ALL_RANGE)
    def alias_mla_mem_pool_to_elastic(self, mem_pool_mod: types.ModuleType) -> bool:
        """Alias MLATokenToKVPool to ElasticMLATokenToKVPool"""
        if self._is_already_patched(mem_pool_mod, "__kvcached_mla_mempool_aliased__"):
            return True

        try:
            ElasticMLATokenToKVPool = getattr(mem_pool_mod, "ElasticMLATokenToKVPool")
            if ElasticMLATokenToKVPool is None:
                return False
            # Alias defaults so core code will use elastic variants
            mem_pool_mod.MLATokenToKVPool = ElasticMLATokenToKVPool  # type: ignore
            self._mark_as_patched(mem_pool_mod, "__kvcached_mla_mempool_aliased__")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to alias MLA memory_pool to elastic one: {e}")
            return False


class ElasticMambaPoolPatch(VersionAwarePatch, BasePatch):
    """Inject ElasticMambaPool with kvcached-backed conv+temporal state.

    Packs each slot's conv[i] and temporal state into a single super-cell and
    uses one kvcached group (contiguous layout) so one ``map_to_kv_tensors``
    call backs every state kind for that slot across all mamba layers.  Slot
    0 is reserved as the padded dummy slot via kvcached's null-block
    mechanism.

    Speculative-decoding intermediate buffers (``intermediate_ssm`` /
    ``intermediate_conv_window``) have a different slot count and remain
    eager torch allocations in this first cut.
    """

    library = "sglang"
    target_module = "sglang.srt.mem_cache.memory_pool"
    patch_name = "elastic_mamba_pool"

    def apply(self, mem_pool_mod: types.ModuleType) -> bool:
        if not self.initialize_version_info():
            return False

        success = self.inject_elastic_mamba_pool(mem_pool_mod)
        if success:
            success &= self.alias_mamba_pool_to_elastic(mem_pool_mod)
        return success

    @version_range(SGLANG_ALL_RANGE)
    def inject_elastic_mamba_pool(self, mem_pool_mod: types.ModuleType) -> bool:
        if hasattr(mem_pool_mod, "ElasticMambaPool"):
            self.logger.debug("ElasticMambaPool already exists")
            return True

        MambaPool = getattr(mem_pool_mod, "MambaPool", None)
        if MambaPool is None:
            # Older SGLang versions don't ship MambaPool.
            self.logger.debug(
                "MambaPool not found in memory_pool module; "
                "skipping ElasticMambaPool injection")
            return True

        try:
            import torch

            class ElasticMambaPool(MambaPool):  # type: ignore[misc, valid-type]
                """MambaPool variant whose conv + temporal state tensors are
                backed by kvcached virtual memory.

                One group_id per instance (independent VM reservation).  Each
                allocated block corresponds to one mamba slot; block_size=1,
                num_kv_buffers=1, cell_size = sum of per-slot bytes across
                all state kinds.
                """

                # Kept separate from attention-pool group IDs to make logs
                # easier to read; the C++ allocator lazily constructs a
                # fresh FTensorAllocator per group_id on first access.
                _next_group_id = 1000

                def __init__(
                    self,
                    *,
                    size: int,
                    spec_state_size: int,
                    cache_params: Any,
                    device: str,
                    mamba_layer_ids: Optional[List[int]] = None,
                    enable_memory_saver: bool = False,
                    speculative_num_draft_tokens: Optional[int] = None,
                ) -> None:
                    import kvcached.integration.sglang.interfaces as kvi

                    # Resolve TP/PP rank the same way ElasticMHATokenToKVPool
                    # does so the IPC socket naming matches.
                    try:
                        from sglang.srt.distributed import (
                            get_pipeline_model_parallel_rank,
                            get_tensor_model_parallel_rank,
                            get_tensor_model_parallel_world_size,
                        )
                        tp_rank = int(get_tensor_model_parallel_rank())
                        tp_size = int(get_tensor_model_parallel_world_size())
                        pp_rank = int(get_pipeline_model_parallel_rank())
                    except (ImportError, AttributeError):
                        try:
                            import torch.distributed as dist
                            tp_rank = dist.get_rank() if dist.is_initialized() else 0
                            tp_size = dist.get_world_size() if dist.is_initialized() else 1
                            pp_rank = 0
                        except (ImportError, AttributeError, RuntimeError, ValueError, TypeError):
                            tp_rank, tp_size, pp_rank = 0, 1, 0

                    kvi.init_kvcached(
                        tp_rank=tp_rank, world_size=tp_size,
                        pp_rank=pp_rank, async_sched=True,
                    )

                    if "cuda" not in device:
                        raise ValueError(
                            "ElasticMambaPool only supports cuda device")

                    self._group_id = ElasticMambaPool._next_group_id
                    ElasticMambaPool._next_group_id += 1

                    self.size = size
                    self.device = device
                    # SGLang passes the layer list as either a mamba_layer_ids
                    # kwarg or cache_params.layers, depending on version.
                    if mamba_layer_ids is not None:
                        layer_ids = list(mamba_layer_ids)
                    else:
                        layer_ids = list(getattr(cache_params, "layers", []))
                    num_mamba_layers = len(layer_ids)
                    if num_mamba_layers == 0:
                        raise ValueError(
                            "ElasticMambaPool could not determine mamba layer "
                            "count: pass mamba_layer_ids or ensure "
                            "cache_params.layers is set.")
                    self.num_mamba_layers = num_mamba_layers

                    # Slot 0 is the padded dummy slot; kvcached reserves it
                    # via reserve_null_block.
                    num_slots = size + 1

                    conv_state, temporal_state, layout = kvi.alloc_mamba_states(
                        num_slots=num_slots,
                        num_mamba_layers=num_mamba_layers,
                        cache_params=cache_params,
                        device=device,
                        group_id=self._group_id,
                    )
                    self._kvcached_layout = layout

                    if speculative_num_draft_tokens is not None:
                        conv_state_shape = cache_params.shape.conv
                        temporal_state_shape = cache_params.shape.temporal
                        intermediate_ssm_state_cache = torch.zeros(
                            size=(
                                num_mamba_layers,
                                spec_state_size + 1,
                                speculative_num_draft_tokens,
                                temporal_state_shape[0],
                                temporal_state_shape[1],
                                temporal_state_shape[2],
                            ),
                            dtype=cache_params.dtype.temporal,
                            device="cuda",
                        )
                        intermediate_conv_window_cache = [
                            torch.zeros(
                                size=(
                                    num_mamba_layers,
                                    spec_state_size + 1,
                                    speculative_num_draft_tokens,
                                    conv_shape[0],
                                    conv_shape[1],
                                ),
                                dtype=cache_params.dtype.conv,
                                device="cuda",
                            )
                            for conv_shape in conv_state_shape
                        ]
                        self.mamba_cache = self.SpeculativeState(
                            conv=conv_state,
                            temporal=temporal_state,
                            intermediate_ssm=intermediate_ssm_state_cache,
                            intermediate_conv_window=intermediate_conv_window_cache,
                        )
                    else:
                        self.mamba_cache = self.State(
                            conv=conv_state, temporal=temporal_state,
                        )

                    # block_size=1 → one block == one mamba slot.
                    # num_kv_buffers=1 → single super-cell per slot per layer.
                    self.kvcached_allocator = kvi.get_kv_cache_manager(
                        num_blocks=num_slots,
                        block_size=1,
                        cell_size=layout["cell_size"],
                        num_layers=num_mamba_layers,
                        reserve_null_block=True,
                        num_kv_buffers=1,
                        group_id=self._group_id,
                    )

                    # Placeholder so code that touches self.free_slots in
                    # error paths (e.g. suppressed leak checks) doesn't
                    # AttributeError.  Never used for allocation.
                    self.free_slots = torch.empty(
                        0, dtype=torch.int64, device=self.device)

                    self.mem_usage = (
                        self.mamba_cache.mem_usage_bytes() / BYTES_PER_GB
                    )
                    logger.info(
                        f"Elastic MambaPool (group_id={self._group_id}) "
                        f"#slots={num_slots}, "
                        f"#mamba_layers={num_mamba_layers}, "
                        f"super_cell_bytes={layout['cell_size']}, "
                        f"virtual_mem={self.mem_usage:.2f} GB"
                    )

                def available_size(self) -> int:
                    return self.kvcached_allocator.available_size()

                def alloc(
                    self, need_size: int,
                ):
                    block_ids = self.kvcached_allocator.alloc(need_size)
                    if block_ids is None:
                        return None
                    select_index = torch.tensor(
                        block_ids, dtype=torch.int64, device=self.device,
                    )
                    for i in range(len(self.mamba_cache.conv)):
                        self.mamba_cache.conv[i][:, select_index] = 0
                    self.mamba_cache.temporal[:, select_index] = 0
                    return select_index

                def free(self, free_index: Any) -> None:
                    if free_index.numel() == 0:
                        return
                    self.kvcached_allocator.free(free_index.tolist())

                def clear(self) -> None:
                    self.kvcached_allocator.clear()

            setattr(mem_pool_mod, "ElasticMambaPool", ElasticMambaPool)
            return True
        except Exception as e:
            self.logger.error(f"Failed to inject ElasticMambaPool: {e}")
            return False

    @version_range(SGLANG_ALL_RANGE)
    def alias_mamba_pool_to_elastic(self, mem_pool_mod: types.ModuleType) -> bool:
        if self._is_already_patched(mem_pool_mod, "__kvcached_mamba_pool_aliased__"):
            return True

        ElasticMambaPool = getattr(mem_pool_mod, "ElasticMambaPool", None)
        if ElasticMambaPool is None:
            return True

        try:
            mem_pool_mod.MambaPool = ElasticMambaPool  # type: ignore
            self._mark_as_patched(mem_pool_mod, "__kvcached_mamba_pool_aliased__")
            return True
        except Exception as e:
            self.logger.warning(
                f"Failed to alias MambaPool to elastic one: {e}")
            return False


class ElasticHybridLinearKVPoolPatch(VersionAwarePatch, BasePatch):
    """Inject ElasticHybridLinearKVPool into SGLang's memory pool module.

    Hybrid linear-attention models (e.g. Qwen3-Next, Nemotron-H, Bamba, LFM2,
    Jamba) wrap a full-attention ``MHATokenToKVPool`` / ``MLATokenToKVPool``
    and a separate ``MambaPool`` inside ``HybridLinearKVPool``.  SGLang keeps
    the two pools in distinct memory — unlike vLLM, there is no shared
    buffer — so kvcached needs to manage the full-attention pool and mamba
    pool separately.

    The outer ``PagedTokenToKVPoolAllocator`` (already aliased to
    ``ElasticPagedTokenToKVPoolAllocator``) requires the kvcache object to
    expose ``kvcached_allocator``.  The inner ``full_kv_pool`` has it (since
    ``MHATokenToKVPool``/``MLATokenToKVPool`` are aliased to their elastic
    variants), so we expose a delegating property on the hybrid pool.
    """

    library = "sglang"
    target_module = "sglang.srt.mem_cache.memory_pool"
    patch_name = "elastic_hybrid_linear_memory_pool"

    def apply(self, mem_pool_mod: types.ModuleType) -> bool:
        if not self.initialize_version_info():
            return False

        success = self.inject_elastic_hybrid_linear_pool(mem_pool_mod)
        if success:
            success &= self.alias_hybrid_linear_pool_to_elastic(mem_pool_mod)
        return success

    @version_range(SGLANG_ALL_RANGE)
    def inject_elastic_hybrid_linear_pool(self, mem_pool_mod: types.ModuleType) -> bool:
        """Inject ElasticHybridLinearKVPool."""
        if hasattr(mem_pool_mod, "ElasticHybridLinearKVPool"):
            self.logger.debug("ElasticHybridLinearKVPool already exists")
            return True

        HybridLinearKVPool = getattr(mem_pool_mod, "HybridLinearKVPool", None)
        if HybridLinearKVPool is None:
            # Older SGLang versions don't ship hybrid linear-attention support.
            self.logger.debug(
                "HybridLinearKVPool not found in memory_pool module; "
                "skipping ElasticHybridLinearKVPool injection"
            )
            return True

        try:
            class ElasticHybridLinearKVPool(HybridLinearKVPool):  # type: ignore[misc, valid-type]
                """Hybrid linear-attention KV pool backed by kvcached for
                the full-attention layers.

                Both sub-pools are kvcached-backed via class aliasing done
                earlier in this patch module: ``MHATokenToKVPool`` /
                ``MLATokenToKVPool`` are aliased to their elastic variants, so
                the inner ``full_kv_pool`` constructed by
                ``HybridLinearKVPool.__init__`` is an elastic attention pool;
                ``MambaPool`` is aliased to ``ElasticMambaPool`` (see
                ``ElasticMambaPoolPatch``), so the inner mamba pool allocates
                conv + temporal state through kvcached as well. Each sub-pool
                gets its own ``group_id`` so their IPC mem-info segments don't
                collide.
                """

                @property
                def kvcached_allocator(self):
                    # Delegate to the attention pool's kvcached allocator so
                    # the outer ElasticPagedTokenToKVPoolAllocator can reach it
                    # through kvcache.kvcached_allocator.
                    return self.full_kv_pool.kvcached_allocator

            setattr(mem_pool_mod, "ElasticHybridLinearKVPool", ElasticHybridLinearKVPool)
            return True
        except Exception as e:
            self.logger.error(f"Failed to inject ElasticHybridLinearKVPool: {e}")
            return False

    @version_range(SGLANG_ALL_RANGE)
    def alias_hybrid_linear_pool_to_elastic(self, mem_pool_mod: types.ModuleType) -> bool:
        """Alias HybridLinearKVPool to ElasticHybridLinearKVPool."""
        if self._is_already_patched(mem_pool_mod, "__kvcached_hybrid_linear_mempool_aliased__"):
            return True

        ElasticHybridLinearKVPool = getattr(mem_pool_mod, "ElasticHybridLinearKVPool", None)
        if ElasticHybridLinearKVPool is None:
            # Injection skipped (older SGLang without HybridLinearKVPool).
            return True

        try:
            mem_pool_mod.HybridLinearKVPool = ElasticHybridLinearKVPool  # type: ignore
            self._mark_as_patched(mem_pool_mod, "__kvcached_hybrid_linear_mempool_aliased__")
            return True
        except Exception as e:
            self.logger.warning(
                f"Failed to alias HybridLinearKVPool to elastic one: {e}")
            return False


class SchedulerMemoryLeakPatch(VersionAwarePatch, BasePatch):
    """Patch SGLang scheduler to suppress memory leak check when kvcached is enabled"""

    library = "sglang"
    target_module = "sglang.srt.managers.scheduler"
    target_class = "Scheduler"
    patch_name = "scheduler_memory_leak"

    def apply(self, sched_mod: types.ModuleType) -> bool:
        # Initialize version info
        if not self.initialize_version_info():
            return False

        # Apply version-specific patches
        return self.patch_scheduler_memory_leak(sched_mod)

    @version_range(SGLANG_ALL_RANGE)
    def patch_scheduler_memory_leak(self, sched_mod: types.ModuleType) -> bool:
        """Patch scheduler to suppress memory leak check when kvcached is enabled"""
        Scheduler = self._get_target_class(sched_mod)
        if Scheduler is None:
            return False

        target_method_name: Union[str, None] = None
        for name, fn in inspect.getmembers(Scheduler, predicate=inspect.isfunction):
            try:
                src = inspect.getsource(fn)
            except Exception:
                continue
            if "token_to_kv_pool_allocator memory leak detected!" in src or (
                "memory leak detected" in src and "token_to_kv_pool_allocator" in src
            ):
                target_method_name = name
                break

        if target_method_name is None:
            self.logger.debug("No memory leak detection method found in Scheduler")
            return False

        original = getattr(Scheduler, target_method_name)
        if self._is_already_patched(original):
            self.logger.debug("Scheduler memory leak check already patched")
            return True

        def _wrapped(self, *args: Any, **kwargs: Any):
            # Disable memory leak detection when ENABLE_KVCACHED is set
            if enable_kvcached():
                return
            return original(self, *args, **kwargs)

        self._mark_as_patched(_wrapped)
        setattr(Scheduler, target_method_name, _wrapped)
        return True


class RadixCacheLimitPatch(VersionAwarePatch, BasePatch):
    """Enforce KVCACHED_MAX_CACHED_TOKENS limit on SGLang's RadixCache.

    After each finished request is inserted into the radix cache, if the
    evictable (cached) size exceeds the configured limit, immediately evict
    down to that limit.  This prevents the cache from consuming all KV pool
    capacity and leaves headroom for running requests.

    Set KVCACHED_MAX_CACHED_TOKENS to a positive integer to enable.
    0 (default) means unlimited — this patch is a no-op in that case.
    """

    library = "sglang"
    target_module = "sglang.srt.mem_cache.radix_cache"
    target_class = "RadixCache"
    patch_name = "radix_cache_limit"

    def apply(self, radix_cache_mod: types.ModuleType) -> bool:
        if MAX_CACHED_TOKENS <= 0:
            logger.debug(
                "KVCACHED_MAX_CACHED_TOKENS not set — RadixCacheLimitPatch skipped"
            )
            return True  # Not an error; just not needed.

        if not self.initialize_version_info():
            return False

        return self.patch_radix_cache_limit(radix_cache_mod)

    @version_range(SGLANG_ALL_RANGE)
    def patch_radix_cache_limit(self, radix_cache_mod: types.ModuleType) -> bool:
        RadixCache = self._get_target_class(radix_cache_mod)
        if RadixCache is None:
            return False

        original_cache_finished = getattr(RadixCache, "cache_finished_req", None)
        if original_cache_finished is None:
            self.logger.warning("RadixCache.cache_finished_req not found")
            return False

        if self._is_already_patched(original_cache_finished):
            self.logger.debug("RadixCache.cache_finished_req already patched")
            return True

        max_cached = MAX_CACHED_TOKENS

        # SGLang >= ~0.5.9 uses EvictParams; older releases pass token count as int.
        try:
            from sglang.srt.mem_cache.radix_cache import EvictParams as _EvictParams
        except ImportError:
            _EvictParams = None  # type: ignore[misc, assignment]

        def _wrapped(self_rc, *args: Any, **kwargs: Any):
            original_cache_finished(self_rc, *args, **kwargs)
            excess = self_rc.evictable_size_ - max_cached
            if excess > 0:
                if _EvictParams is not None:
                    self_rc.evict(_EvictParams(num_tokens=excess))
                else:
                    self_rc.evict(excess)

        self._mark_as_patched(_wrapped)
        RadixCache.cache_finished_req = _wrapped  # type: ignore

        logger.info(
            f"[kvcached] RadixCache evictable size capped at "
            f"{max_cached} tokens (KVCACHED_MAX_CACHED_TOKENS)"
        )
        return True


class HttpServerPatch(VersionAwarePatch, BasePatch):
    """Patch SGLang's HTTP server to inject kvcached KV cache management endpoints.

    SGLang creates a module-level ``app = FastAPI(...)`` in
    ``sglang.srt.entrypoints.http_server``.  We wrap
    ``_setup_and_run_http_server`` so that the kvcache router is attached
    to ``app`` before uvicorn starts listening.

    This adds ``/kvcache/status``, ``/kvcache/limit``, and
    ``/kvcache/limit_percent`` to every SGLang instance.
    """

    library = "sglang"
    target_module = "sglang.srt.entrypoints.http_server"
    target_class = None  # Module-level function, not a class
    patch_name = "http_server_kvcache"

    def can_apply(self, target_module: types.ModuleType) -> bool:
        return hasattr(target_module, "_setup_and_run_http_server")

    def apply(self, http_server_mod: types.ModuleType) -> bool:
        if not self.initialize_version_info():
            return False
        return self.patch_setup_and_run(http_server_mod)

    @version_range(SGLANG_ALL_RANGE)
    def patch_setup_and_run(self, http_server_mod: types.ModuleType) -> bool:
        """Wrap _setup_and_run_http_server to attach kvcache routes."""
        original_fn = getattr(http_server_mod, "_setup_and_run_http_server", None)
        if original_fn is None:
            self.logger.warning(
                "_setup_and_run_http_server not found in http_server module")
            return False

        if self._is_already_patched(original_fn, "_setup_and_run"):
            self.logger.debug("_setup_and_run_http_server already patched")
            return True

        _logger = self.logger

        def _patched_setup_and_run(*args: Any, **kwargs: Any):
            if enable_kvcached():
                try:
                    # The module-level ``app`` is the FastAPI instance.
                    fastapi_app = getattr(http_server_mod, "app", None)
                    if fastapi_app is not None:
                        from kvcached.integration.vllm.api_router import (
                            attach_to_app,
                        )
                        attach_to_app(fastapi_app)
                except Exception as exc:
                    _logger.warning(
                        "Failed to attach kvcache router to SGLang: %s", exc)

            return original_fn(*args, **kwargs)

        self._mark_as_patched(_patched_setup_and_run, "_setup_and_run")
        http_server_mod._setup_and_run_http_server = _patched_setup_and_run
        return True

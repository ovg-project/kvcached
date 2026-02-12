# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
SGLang-specific patches using unified patch infrastructure.
"""

import inspect
import math
import types
from typing import Any, Union

from kvcached.integration.patch_base import BasePatch, enable_kvcached
from kvcached.integration.version_utils import VersionAwarePatch, version_range
from kvcached.utils import get_kvcached_logger

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
                    return self.kvcached_allocator.available_size()

                def alloc(self, need_size: int):
                    indices: list[int] = self.kvcached_allocator.alloc(need_size)
                    return torch.tensor(indices, dtype=torch.int32, device="cuda")

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

                def get_cpu_copy(self, indices):
                    return self._kvcache.get_cpu_copy(indices)

                def load_cpu_copy(self, kv_cache_cpu, indices):
                    return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)

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
                        math.ceil(size / page_size) + 1, page_size, self.cell_size, layer_num
                    )

                    k_size, v_size = self.get_kv_size_bytes()
                    GB = 1024**3
                    k_size_phy, v_size_phy = self.get_kv_size_bytes_phy()

                    logger.info(
                        f"VirtualKV Cache is allocated. #tokens: {size}, K size: "
                        f"{k_size / GB:.2f} GB, V size: {v_size / GB:.2f} GB"
                    )
                    logger.info(
                        f"Physical KV Cache limits by --mem-fraction-static: "
                        f"#tokens: {size}, K size: "
                        f"{k_size_phy / GB:.2f} GB, V size: {v_size_phy / GB:.2f} GB"
                    )

                    self.mem_usage = (k_size + v_size) / GB

                def __del__(self):  # best-effort cleanup
                    try:
                        import kvcached.integration.sglang.interfaces as kvi

                        kvi.shutdown_kvcached()
                    except Exception:
                        pass

                def _create_buffers(self):
                    import kvcached.integration.sglang.interfaces as kvi

                    # Initialize kvcached with overlap scheduling to be conservative
                    kvi.init_kvcached(async_sched=True)

                    if "cuda" not in self.device:
                        raise ValueError("ElasticMHATokenToKVPool only supports cuda device")
                    self.k_buffer, self.v_buffer = kvi.alloc_kv_cache(
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
                    )

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

# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
SGLang-specific patches using unified patch infrastructure.
"""

import functools
import inspect
import math
import os
import types
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

from kvcached.integration.patch_base import BasePatch, enable_kvcached
from kvcached.integration.version_utils import (
    VersionAwarePatch,
    version_range,
)
from kvcached.utils import MAX_CACHED_TOKENS, get_kvcached_logger

BYTES_PER_GB = 1024**3
_CAPACITY_QUERY_FAILED = -(1 << 63)

# Version ranges for SGLang support
SGLANG_ALL_RANGE = ">=0.4.9"  # All supported versions

logger = get_kvcached_logger()


def _is_supported_gpu_device(device: str) -> bool:
    device_str = str(device).lower()
    return device_str.startswith("cuda") or device_str.startswith("hip")


def _sglang_free_immediately(allocator: Any) -> bool:
    """True when a free should release memory now instead of joining a group.

    SGLang up to 0.5.18 tracks free-group state with an
    ``is_not_in_free_group`` flag next to an always-present ``free_group``
    list; 0.5.19 removed the flag and made ``free_group is None`` the
    not-in-group sentinel.
    """
    flag = getattr(allocator, "is_not_in_free_group", None)
    if flag is not None:
        return bool(flag)
    return allocator.free_group is None


def _sglang_free_group_copy(allocator: Any, free_index: Any) -> Any:
    """Deferred tensors are cloned on SGLang 0.5.19+ so a caller cannot
    mutate a queued view before the group flushes; older releases defer the
    tensor as passed."""
    copy_for_free_group = getattr(allocator, "_copy_for_free_group", None)
    if copy_for_free_group is not None:
        return copy_for_free_group(free_index)
    return free_index


def _sglang_reset_free_group(allocator: Any) -> None:
    """Reset free-group state to not-in-group under either protocol."""
    if hasattr(allocator, "is_not_in_free_group"):
        allocator.is_not_in_free_group = True
        allocator.free_group = []
    else:
        allocator.free_group = None


def _reduce_sglang_world_min_bytes(torch: Any, local_bytes: int) -> int:
    """Return one capacity shared by every rank in the SGLang world group."""
    from sglang.srt.distributed.parallel_state import get_world_group

    world_group = get_world_group()
    if int(world_group.world_size) <= 1:
        return local_bytes

    capacity = torch.tensor(local_bytes, dtype=torch.int64)
    torch.distributed.all_reduce(
        capacity,
        op=torch.distributed.ReduceOp.MIN,
        group=world_group.cpu_group,
    )
    return int(capacity.item())


def _import_sglang_allocator_kernels() -> types.ModuleType:
    try:
        from sglang.kernels.ops.memory import allocator as allocator_kernels

        return allocator_kernels
    except ModuleNotFoundError as exc:
        if exc.name is None or not exc.name.startswith("sglang.kernels"):
            raise

    from sglang.srt.mem_cache.triton_ops import allocator as allocator_kernels

    return allocator_kernels


def _resolve_sglang_allocator_kernels(
    alloc_mod: types.ModuleType,
) -> Tuple[Any, Any]:
    try:
        return alloc_mod.alloc_extend_kernel, alloc_mod.alloc_decode_kernel
    except AttributeError:
        allocator_kernels = _import_sglang_allocator_kernels()
        return (
            allocator_kernels.alloc_extend_kernel,
            allocator_kernels.alloc_decode_kernel,
        )


class _SGLangVirtualKVCapacityPatchBase(VersionAwarePatch, BasePatch):
    """Keep SGLang's logical KV capacity independent of peer processes."""

    library = "sglang"
    patch_name = "virtual_kv_capacity"

    def apply(self, target_module: types.ModuleType) -> bool:
        if not self.initialize_version_info():
            return False
        return self.patch_profile_available_bytes(target_module)

    def _get_mem_fraction_static(self, owner: Any) -> float:
        raise NotImplementedError

    def _handle_max_mamba_cache(self, owner: Any, capacity_gib: float) -> float:
        raise NotImplementedError

    def _adjust_logical_budget(
        self, *, owner: Any, total_memory: int, logical_budget: int
    ) -> int:
        return logical_budget

    @version_range(SGLANG_ALL_RANGE)
    def patch_profile_available_bytes(self, target_module: types.ModuleType) -> bool:
        target_class = self._get_target_class(target_module)
        if target_class is None:
            return False

        original_profile = getattr(target_class, "_profile_available_bytes", None)
        if original_profile is None:
            self.logger.warning(
                "SGLang %s does not expose _profile_available_bytes",
                self.target_class,
            )
            return False
        if self._is_already_patched(original_profile, "virtual_kv_capacity"):
            return True

        @functools.wraps(original_profile)
        def _patched_profile_available_bytes(owner, pre_model_load_memory: int) -> int:
            if not enable_kvcached() or not _is_supported_gpu_device(owner.device):
                return original_profile(owner, pre_model_load_memory)

            if getattr(owner, "post_capture_kv_active", False):
                raise RuntimeError(
                    "SGLang post-capture KV sizing is not supported with "
                    "kvcached elastic pools"
                )

            import torch

            query_error = None
            try:
                total_memory = int(
                    torch.cuda.get_device_properties(owner.gpu_id).total_memory
                )
                mem_fraction_static = self._get_mem_fraction_static(owner)
                logical_budget = math.ceil(total_memory * mem_fraction_static)
                logical_budget = self._adjust_logical_budget(
                    owner=owner,
                    total_memory=total_memory,
                    logical_budget=logical_budget,
                )
                process_local_reserved = int(
                    torch.cuda.memory_reserved(owner.gpu_id)
                )
                local_available_bytes = logical_budget - process_local_reserved
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                query_error = exc
                total_memory = 0
                logical_budget = 0
                process_local_reserved = 0
                local_available_bytes = _CAPACITY_QUERY_FAILED

            try:
                available_bytes = _reduce_sglang_world_min_bytes(
                    torch, local_available_bytes
                )
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                logger.warning(
                    "Unable to synchronize SGLang virtual KV capacity; "
                    "falling back to SGLang profiling: %s",
                    exc,
                )
                return original_profile(owner, pre_model_load_memory)

            if available_bytes == _CAPACITY_QUERY_FAILED:
                logger.warning(
                    "Unable to derive stable SGLang virtual KV capacity on "
                    "at least one rank; falling back to SGLang profiling: %s",
                    query_error or "peer rank query failed",
                )
                return original_profile(owner, pre_model_load_memory)

            if owner.mambaish_config is not None:
                available_gib = available_bytes / BYTES_PER_GB
                available_bytes = int(
                    self._handle_max_mamba_cache(owner, available_gib) * BYTES_PER_GB
                )

            logger.info(
                "Using kvcached process-local KV capacity for SGLang: "
                "budget=%d bytes, pytorch_reserved=%d bytes, "
                "world_min_available=%d bytes (device_total=%d, "
                "mem_fraction_static=%.4f)",
                logical_budget,
                process_local_reserved,
                available_bytes,
                total_memory,
                mem_fraction_static,
            )
            return available_bytes

        self._mark_as_patched(_patched_profile_available_bytes, "virtual_kv_capacity")
        target_class._profile_available_bytes = _patched_profile_available_bytes
        return True


class SGLangVirtualKVCapacityPatch(_SGLangVirtualKVCapacityPatchBase):
    target_module = "sglang.srt.mem_cache.kv_cache_configurator"
    target_class = "KVCacheConfigurator"

    def _get_mem_fraction_static(self, configurator: Any) -> float:
        return float(configurator.server_args.mem_fraction_static)

    def _handle_max_mamba_cache(
        self, configurator: Any, capacity_gib: float
    ) -> float:
        return configurator._handle_max_mamba_cache(capacity_gib)


class SGLangLegacyVirtualKVCapacityPatch(_SGLangVirtualKVCapacityPatchBase):
    target_module = "sglang.srt.model_executor.model_runner"
    target_class = "ModelRunner"

    def _get_mem_fraction_static(self, runner: Any) -> float:
        return float(runner.mem_fraction_static)

    def _handle_max_mamba_cache(self, runner: Any, capacity_gib: float) -> float:
        return runner.handle_max_mamba_cache(capacity_gib)


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
                    if not _is_supported_gpu_device(device):
                        raise ValueError(
                            "ElasticTokenToKVPoolAllocator only supports GPU "
                            "devices (cuda/hip)"
                        )
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
                    if _sglang_free_immediately(self):
                        try:
                            indices: list[int] = free_index.cpu().numpy().tolist()
                        except Exception:
                            indices = list(free_index)
                        return self.kvcached_allocator.free(indices)
                    else:
                        self.free_group.append(
                            _sglang_free_group_copy(self, free_index)
                        )

                def free_page_ids(self, page_ids):
                    # SGLang 0.5.20's SWA composite frees sub-allocator pages
                    # through free_page_ids().  With page_size == 1 page ids
                    # are token ids, mirroring the native token allocator.
                    if page_ids.numel() == 0:
                        return
                    self.free(page_ids)

                def clear(self):
                    if hasattr(self, "kvcached_allocator"):
                        self.kvcached_allocator.clear()
                    _sglang_reset_free_group(self)

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
            alloc_extend_kernel, alloc_decode_kernel = (
                _resolve_sglang_allocator_kernels(alloc_mod)
            )

            alloc_extend_kernel_fn = getattr(
                alloc_extend_kernel, "fn", alloc_extend_kernel
            )
            alloc_extend_param_names = tuple(
                inspect.signature(alloc_extend_kernel_fn).parameters
            )

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
                    if not _is_supported_gpu_device(device):
                        raise ValueError(
                            "ElasticPagedTokenToKVPoolAllocator only supports GPU "
                            "devices (cuda/hip)"
                        )
                    self.kvcached_allocator = kvcache.kvcached_allocator
                    self.num_pages = size // page_size
                    self.seen_max_num_extend_tokens_next_power_of_2 = 1
                    # The native PagedTokenToKVPoolAllocator init sets this, and
                    # 0.5.20's SWA allocator reads it on every free at
                    # page_size > 1.
                    self.debug_mode = os.getenv(
                        "SGLANG_DEBUG_MEMORY_POOL", "false").lower() in ("true", "1")
                    logger.info(
                        f"[kvcached] ElasticPagedTokenToKVPoolAllocator in use: size={size}, "
                        f"page_size={page_size}"
                    )
                    # Base class expects these tensors for backup_state / free_group_end
                    self.free_pages = torch.empty((0,), dtype=torch.int64, device=self.device)
                    self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)
                    # SGLang 0.5.20 defers page-id frees in a group separate
                    # from token-index frees; see free_page_ids().
                    self.free_page_ids_group: List[Any] = []

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
                    num_new_pages: Optional[int] = None,
                ):
                    self.seen_max_num_extend_tokens_next_power_of_2 = max(
                        self.seen_max_num_extend_tokens_next_power_of_2,
                        next_power_of_2(extend_num_tokens),
                    )
                    bs = len(prefix_lens)

                    if num_new_pages is None:
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
                    kernel_kwargs: dict[str, Any] = {
                        "pre_lens_ptr": prefix_lens,
                        "seq_lens_ptr": seq_lens,
                        "last_loc_ptr": last_loc,
                        "free_page_ptr": free_pages,
                        "out_indices": out_indices,
                        "bs_upper": next_power_of_2(bs),
                        "page_size": self.page_size,
                    }
                    if "ret_values" in alloc_extend_param_names:
                        kernel_kwargs["ret_values"] = torch.empty(
                            (), dtype=torch.int64, device=self.device
                        )
                    if "max_num_extend_tokens" in alloc_extend_param_names:
                        kernel_kwargs["max_num_extend_tokens"] = (
                            self.seen_max_num_extend_tokens_next_power_of_2
                        )

                    alloc_extend_kernel[(bs,)](**kernel_kwargs)
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
                        block_ids = self.kvcached_allocator.alloc_packed(num_new_pages)
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

                    if _sglang_free_immediately(self):
                        page_ids = torch.unique(free_index // self.page_size)
                        try:
                            indices: list[int] = page_ids.cpu().numpy().tolist()
                        except Exception:
                            indices = list(page_ids)
                        return self.kvcached_allocator.free(indices)
                    else:
                        self.free_group.append(
                            _sglang_free_group_copy(self, free_index)
                        )

                def free_page_ids(self, page_ids):
                    # SGLang 0.5.20's paged allocator and SWA composite free
                    # exact page ids through this method, with no dedup and
                    # no index-to-page reduction.  kvcached block ids equal
                    # SGLang page ids, so the ids release directly; inside a
                    # free group they wait in free_page_ids_group, mirroring
                    # the native deferral.
                    if page_ids.numel() == 0:
                        return
                    if _sglang_free_immediately(self):
                        try:
                            ids: list[int] = page_ids.cpu().numpy().tolist()
                        except Exception:
                            ids = list(page_ids)
                        return self.kvcached_allocator.free(ids)
                    else:
                        self.free_page_ids_group.append(
                            _sglang_free_group_copy(self, page_ids)
                        )

                def free_group_begin(self):
                    super().free_group_begin()
                    self.free_page_ids_group = []

                def free_group_end(self):
                    super().free_group_end()
                    if self.free_page_ids_group:
                        page_ids_group = self.free_page_ids_group
                        self.free_page_ids_group = []
                        self.free_page_ids(torch.cat(page_ids_group))

                def clear(self):
                    if hasattr(self, "kvcached_allocator"):
                        self.kvcached_allocator.clear()
                    self.free_pages = torch.empty(
                        (0,), dtype=torch.int64, device=self.device
                    )
                    self.release_pages = torch.empty(
                        (0,), dtype=torch.int64, device=self.device
                    )
                    self.free_page_ids_group = []
                    _sglang_reset_free_group(self)

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


class ElasticSWAAllocatorPatch(VersionAwarePatch, BasePatch):
    """Make SGLang's composite SWA allocator use elastic sub-allocators.

    SGLang's ``allocator.swa`` module imports the token and paged allocator
    classes directly from their implementation modules.  Replacing only the
    re-exports on ``sglang.srt.mem_cache.allocator`` therefore does not affect
    the classes captured by ``SWATokenToKVPoolAllocator``.
    """

    library = "sglang"
    target_module = "sglang.srt.mem_cache.allocator.swa"
    patch_name = "elastic_swa_allocator"

    def apply(self, swa_alloc_mod: types.ModuleType) -> bool:
        if not self.initialize_version_info():
            return False
        return self.alias_swa_sub_allocators(swa_alloc_mod)

    @version_range(">=0.5.13")
    def alias_swa_sub_allocators(self, swa_alloc_mod: types.ModuleType) -> bool:
        marker = "__kvcached_swa_sub_allocators_aliased__"
        if self._is_already_patched(swa_alloc_mod, marker):
            return True

        try:
            from sglang.srt.mem_cache import allocator as alloc_mod

            elastic_token_allocator = getattr(
                alloc_mod, "ElasticTokenToKVPoolAllocator"
            )
            elastic_paged_allocator = getattr(
                alloc_mod, "ElasticPagedTokenToKVPoolAllocator"
            )
        except (ImportError, AttributeError) as exc:
            self.logger.warning(
                "Failed to resolve elastic allocators for SGLang SWA: %s", exc
            )
            return False

        setattr(swa_alloc_mod, "TokenToKVPoolAllocator", elastic_token_allocator)
        setattr(swa_alloc_mod, "PagedTokenToKVPoolAllocator", elastic_paged_allocator)
        self._mark_as_patched(swa_alloc_mod, marker)
        return True


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

            # SGLang 0.5.16 split _create_buffers() into
            # _create_buffers_normal() (the plain allocation stage) plus a
            # tail that builds _kv_buffer_descs, which PD transfer
            # (prefill-decode disaggregation) registers buffers from, and
            # the data_ptrs/data_strides tensors the speculative-decode kv
            # copy reads. Replacing _create_buffers() wholesale skips that
            # tail, so on the split layout we override the inner stage and
            # let the native tail run over the elastic buffers. Detect the
            # split by presence rather than version so source builds
            # without version metadata route the same way.
            has_buffer_seam = hasattr(MHATokenToKVPool, "_create_buffers_normal")

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
                    if kwargs.get("post_capture_active"):
                        # SGLang 0.5.16+ post-capture sizing reserves VA-only
                        # buffers and later finalizes backing through its own
                        # VMM owner, which the elastic buffer override never
                        # creates.  Refuse instead of half-running.
                        raise NotImplementedError(
                            "ElasticMHATokenToKVPool does not support SGLang "
                            "post-capture KV sizing. Unset "
                            "SGLANG_ENABLE_POST_CAPTURE_KV_SIZING or disable "
                            "kvcached (ENABLE_KVCACHED=false)."
                        )
                    # Assign group_id BEFORE super().__init__() because it
                    # calls _create_buffers() which needs self._group_id.
                    self._group_id = ElasticMHATokenToKVPool._next_group_id
                    ElasticMHATokenToKVPool._next_group_id += 1
                    # Older SGLang pools do not expose a separate physical
                    # storage dtype. The parent may call our _create_buffers()
                    # before returning, so install the logical fallback first.
                    self.store_dtype = dtype

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

                    self.cell_size = (
                        self.head_num * self.head_dim * self.store_dtype.itemsize
                    )
                    self.kvcached_allocator = kvi.get_kv_cache_manager(
                        math.ceil(size / page_size) + 1, page_size, self.cell_size, layer_num,
                        group_id=self._group_id,
                        pool_name="mha",
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

                def _create_buffers_elastic(self):
                    import kvcached.integration.sglang.interfaces as kvi

                    # kvcached backs NHD rows, one (head_num, head_dim) row
                    # per token slot. HND and the ROCm vectorized layouts
                    # reshape the buffers, so refuse them instead of serving
                    # NHD-shaped memory under another layout's label.
                    kv_cache_layout = getattr(self, "kv_cache_layout", "nhd")
                    if getattr(self, "use_hnd", False) or kv_cache_layout != "nhd":
                        raise NotImplementedError(
                            "ElasticMHATokenToKVPool only supports the NHD "
                            f"KV cache layout, got {kv_cache_layout!r}. Unset "
                            "SGLANG_USE_HND_KVCACHE or the kv_cache_layout "
                            "override, or disable kvcached "
                            "(ENABLE_KVCACHED=false).")

                    # Resolve TP rank and size for IPC socket registration.
                    # SGLang workers each call this independently,
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

                    if not _is_supported_gpu_device(self.device):
                        raise ValueError(
                            "ElasticMHATokenToKVPool only supports GPU devices "
                            "(cuda/hip)")
                    _kv_mha: Tuple[List[Any], List[Any]] = cast(
                        Tuple[List[Any], List[Any]],
                        kvi.alloc_kv_cache(
                            kvcache_shape=(
                                self.size + self.page_size,
                                self.head_num,
                                self.head_dim,
                            ),
                            dtype=self.store_dtype,
                            device=self.device,
                            num_layers=self.layer_num,
                            page_size=self.page_size,
                            attention_type="MHA",
                            kv_layout="NHD",
                            group_id=self._group_id,
                        ),
                    )
                    self.k_buffer, self.v_buffer = _kv_mha

                if has_buffer_seam:
                    # 0.5.16+: native _create_buffers() keeps running. Its
                    # non-quantized branch pins k/v_scale_buffer and
                    # dq_k/dq_v_buffer to None before dispatching here, and
                    # its tail builds _kv_buffer_descs from the elastic
                    # buffers. The pointer-table part of the tail is
                    # guarded below: it is only valid when every K/V view
                    # is independently contiguous.
                    def _create_buffers_normal(self):
                        self._create_buffers_elastic()

                    def _create_quantized_buffers(self):
                        # The native dispatch routes here when a quantized
                        # KV cache recipe (quant_method) is configured. The
                        # recipe would allocate native torch buffers outside
                        # kvcached while the elastic allocator keeps
                        # tracking the pool, so refuse instead of
                        # half-running.
                        raise NotImplementedError(
                            "ElasticMHATokenToKVPool does not support "
                            "quantized KV cache recipes (quant_method). "
                            "Disable kvcached (ENABLE_KVCACHED=false) to "
                            "use a quantized KV cache.")

                    def _kv_buffers_independently_contiguous(self):
                        # The per-layer FTensor layout
                        # (KVCACHED_CONTIGUOUS_LAYOUT=false) hands out
                        # dense per-layer views, while the default
                        # contiguous layout interleaves all layers and K/V
                        # in one (tokens, layers, 2, heads, dim) buffer,
                        # so each view's token stride spans every layer.
                        return all(
                            t.is_contiguous()
                            for t in (*self.k_buffer, *self.v_buffer))

                    def _init_data_ptrs_and_strides(self):
                        # The native tables store one scalar per buffer,
                        # prod(shape[1:]) * itemsize, and every consumer
                        # (the Triton copy kernel in
                        # kernels/ops/kvcache/cache_move.py) uses that
                        # scalar both as the token-address pitch and as
                        # the bytes to copy. That only holds when rows are
                        # independently contiguous; publishing the tables
                        # for interleaved views would make the kernel walk
                        # and overwrite unrelated bytes. Leave them unset
                        # instead so an unexpected consumer fails with
                        # AttributeError, matching pre-0.5.16 elastic
                        # pools, which never had them.
                        if self._kv_buffers_independently_contiguous():
                            super()._init_data_ptrs_and_strides()

                    def _init_kv_copy_and_warmup(self):
                        # The warmup launches the copy kernel over the
                        # pointer tables, which the interleaved layout
                        # does not publish; move_kv_cache below covers
                        # that layout without the kernel.
                        if self._kv_buffers_independently_contiguous():
                            super()._init_kv_copy_and_warmup()
                        else:
                            self._kv_copy_config = None

                    def _move_kv_cache_impl(self, tgt_loc, src_loc):
                        # Native move strategy hook (the base
                        # move_kv_cache already did the OOB checks). For
                        # interleaved views, advanced indexing on the
                        # views walks the real strides, mirroring the
                        # native move_kv_cache_native fallback; scale
                        # buffers cannot exist here because quantized
                        # recipes are refused above.
                        if self._kv_buffers_independently_contiguous():
                            super()._move_kv_cache_impl(tgt_loc, src_loc)
                            return
                        if tgt_loc.numel() == 0:
                            return
                        tgt = tgt_loc.view(-1).long()
                        src = src_loc.view(-1).long()
                        for k_cache, v_cache in zip(self.k_buffer,
                                                    self.v_buffer):
                            k_cache[tgt] = k_cache[src]
                            v_cache[tgt] = v_cache[src]

                    def get_contiguous_buf_infos(self):
                        # PD transfer registers [ptr, ptr + len) per layer
                        # and walks pages at ptr + page * item_len, which
                        # has no valid answer when the layers interleave
                        # within every token row.
                        if self._kv_buffers_independently_contiguous():
                            return super().get_contiguous_buf_infos()
                        raise NotImplementedError(
                            "the interleaved elastic KV layout has no "
                            "per-layer contiguous regions, so PD transfer "
                            "cannot register it. Use the per-layer layout "
                            "(KVCACHED_CONTIGUOUS_LAYOUT=false) or "
                            "disable kvcached (ENABLE_KVCACHED=false) "
                            "for disaggregation.")
                else:
                    # Before 0.5.16 _create_buffers() is a single stage
                    # (any pointer derivation is inlined after allocation),
                    # so it is replaced whole, as before.
                    def _create_buffers(self):
                        self._create_buffers_elastic()

                def get_kv_size_bytes_phy(self):
                    """Return the physical memory limits of the K/V buffers.

                    This limit is enforced by `--mem-fraction-static` option.
                    """
                    total_tokens = self.size + self.page_size
                    elems_per_token = self.head_num * self.head_dim
                    bytes_per_elem = self.store_dtype.itemsize

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
                    self.store_dtype = getattr(
                        self, "store_dtype", getattr(self, "dtype", dtype)
                    )

                    # MLA-specific attributes (mirroring MLATokenToKVPool).
                    # SGLang 0.5.13 renamed the sparse-attention kwarg and
                    # attributes from use_nsa/nsa_kv_cache_store_fp8 to
                    # use_dsa/dsa_kv_cache_store_fp8 (DSA is DeepSeek sparse
                    # attention). Inherited write paths such as
                    # set_kv_buffer read the current spelling on every
                    # call, so accept either kwarg and set both spellings.
                    # The fp8 flag also requires override_kv_cache_dim,
                    # matching the native derivation on every version since
                    # 0.5.9.
                    self.kv_lora_rank = kv_lora_rank
                    self.qk_rope_head_dim = qk_rope_head_dim
                    use_dsa = kwargs.get("use_dsa", kwargs.get("use_nsa", False))
                    override_kv_cache_dim = kwargs.get("override_kv_cache_dim", None)
                    self.use_dsa = self.use_nsa = use_dsa
                    self.dsa_kv_cache_store_fp8 = self.nsa_kv_cache_store_fp8 = (
                        use_dsa
                        and dtype == torch.float8_e4m3fn
                        and override_kv_cache_dim is not None
                    )
                    self.kv_cache_dim = (
                        override_kv_cache_dim
                        if self.dsa_kv_cache_store_fp8
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

                    if not _is_supported_gpu_device(device):
                        raise ValueError(
                            "ElasticMLATokenToKVPool only supports GPU devices "
                            "(cuda/hip)")
                    self.kv_buffer = cast(
                        List[torch.Tensor],
                        kvi.alloc_kv_cache(
                            kvcache_shape=(
                                size + page_size,
                                1,
                                self.kv_cache_dim,
                            ),
                            dtype=self.store_dtype,
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

                    self.cell_size = (
                        self.kv_cache_dim * self.store_dtype.itemsize
                    )
                    self.kvcached_allocator = kvi.get_kv_cache_manager(
                        size + page_size, page_size, self.cell_size, layer_num,
                        num_kv_buffers=1,
                        pool_name="mla",
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
                    bytes_per_elem = self.store_dtype.itemsize

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
        if success and self.rebind_hybrid_mamba_pool_cls in self.applicable_methods:
            success &= self.rebind_hybrid_mamba_pool_cls(mem_pool_mod)
        if success and self.patch_mamba_slot_allocator in self.applicable_methods:
            success &= self.patch_mamba_slot_allocator(mem_pool_mod)
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

            State = getattr(MambaPool, "State")

            class _PerLayerMambaState:
                """Mamba state holder for kvcached non-contiguous layout.

                In non-contiguous layout each mamba layer has its own VM
                reservation, so a single ``(num_mamba_layers, num_slots,
                *shape)`` tensor that spans all layers is impossible.  We
                therefore keep per-layer 2D state tensors and expose the
                same ``at_layer_idx`` API the rest of SGLang uses — every
                hot-path consumer (e.g. ``Mamba2AttnBackend.forward``) goes
                through ``mamba2_layer_cache(layer_id)`` which calls
                ``at_layer_idx`` and only ever sees per-layer slices.

                Multi-layer code paths that index the state directly as a
                3D tensor — currently only
                ``HybridLinearAttnBackend.update_mamba_state_after_mtp_verify``
                in the speculative-decode flow — are not supported in
                non-contiguous layout (caught at __init__ time).
                """

                def __init__(
                    self,
                    conv_per_layer: List[List["torch.Tensor"]],
                    temporal_per_layer: List["torch.Tensor"],
                ) -> None:
                    # conv_per_layer[shape_idx][layer_idx] -> (slots, *shape)
                    self.conv_per_layer = conv_per_layer
                    # temporal_per_layer[layer_idx] -> (slots, *temporal_shape)
                    self.temporal_per_layer = temporal_per_layer

                # GDNAttnBackend (and similar) introspect
                # ``mamba_cache.conv[0].shape`` / ``.temporal`` at attention-
                # backend init time to learn the per-slot state shape.  We
                # expose layer-0's per-layer tensor as a stand-in: trailing
                # dims (slots, *shape) are identical across layers, and
                # consumers only read .shape[-1] / .shape from it.  Any
                # multi-layer 3D indexing would silently mis-target layer 0
                # — but in this codebase those paths only run during
                # speculative decode, which we already block at __init__.
                @property
                def conv(self) -> List["torch.Tensor"]:
                    return [c[0] for c in self.conv_per_layer]

                @property
                def temporal(self) -> "torch.Tensor":
                    return self.temporal_per_layer[0]

                def at_layer_idx(self, layer: int) -> Any:
                    return State(
                        conv=[c[layer] for c in self.conv_per_layer],
                        temporal=self.temporal_per_layer[layer],
                    )

                def mem_usage_bytes(self) -> int:
                    total = 0
                    for shape_list in self.conv_per_layer:
                        for t in shape_list:
                            total += t.numel() * t.element_size()
                    for t in self.temporal_per_layer:
                        total += t.numel() * t.element_size()
                    return total

            class ElasticMambaPool(MambaPool):  # type: ignore[misc, valid-type]
                """MambaPool variant whose conv + temporal state tensors are
                backed by kvcached virtual memory.

                One group_id per instance (independent VM reservation).  Each
                allocated block corresponds to one mamba slot; block_size=1,
                num_kv_buffers=1, cell_size = sum of per-slot bytes across
                all state kinds.

                Two layouts are supported:

                * Contiguous: a single VM reservation backs all layers, so
                  ``mamba_cache.conv[i]`` and ``mamba_cache.temporal`` are
                  ``(num_mamba_layers, slots, *)`` tensors — identical in
                  shape to the native MambaPool buffers.
                * Non-contiguous: each layer has its own VM reservation,
                  so ``mamba_cache`` is a ``_PerLayerMambaState`` that
                  satisfies the ``at_layer_idx`` contract used by the
                  hybrid linear attention backend.  Speculative decoding
                  is not supported in this layout.
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
                    speculative_eagle_topk: Optional[int] = None,
                    enable_linear_replayssm: bool = False,
                    linear_replayssm_cache_len: int = 16,
                    envelope_layout: bool = False,
                    enable_gdn_replayssm_spec: bool = False,
                    enable_linear_replayssm_spec: bool = False,
                ) -> None:
                    import kvcached.integration.sglang.interfaces as kvi

                    if enable_linear_replayssm:
                        raise NotImplementedError(
                            "ElasticMambaPool does not support SGLang "
                            "linear ReplaySSM buffers yet."
                        )
                    if envelope_layout:
                        raise NotImplementedError(
                            "ElasticMambaPool uses the kvcached mamba state "
                            "layout and does not support SGLang envelope_layout."
                        )
                    # SGLang 0.5.16 passes enable_gdn_replayssm_spec and
                    # 0.5.17 renamed it to enable_linear_replayssm_spec.  The
                    # spec-verify replay ring is allocated by the native init
                    # this class skips, so accept the kwargs but refuse the
                    # feature.
                    if enable_gdn_replayssm_spec or enable_linear_replayssm_spec:
                        raise NotImplementedError(
                            "ElasticMambaPool does not support SGLang "
                            "ReplaySSM speculative verification buffers yet."
                        )

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

                    if not _is_supported_gpu_device(device):
                        raise ValueError(
                            "ElasticMambaPool only supports GPU devices (cuda/hip)")

                    self._group_id = ElasticMambaPool._next_group_id
                    ElasticMambaPool._next_group_id += 1

                    self.size = size
                    self.device = device
                    self.enable_linear_replayssm = False
                    self.linear_replayssm_cache_len = linear_replayssm_cache_len
                    self.replayssm_is_kda = False
                    self.replayssm_write_pos = None
                    # The native init leaves the spec-verify ring as None when
                    # the feature is off, and HybridReqToTokenPool.alloc on
                    # 0.5.16-0.5.19 reads it for every new request.
                    self.replayssm_cache_base = None
                    self.replayssm_is_flush = None
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
                    # Attributes the native init sets and inherited methods
                    # read on 0.5.16+: the transfer iterator walks
                    # mamba_layer_ids with conv_slice_axis /
                    # conv_shard_groups, copy_from checks debug_memory_pool,
                    # and the replayssm-spec flags mirror the refusals above.
                    self.mamba_layer_ids = layer_ids
                    self.debug_memory_pool = False
                    self.enable_linear_replayssm_spec = False
                    self.replayssm_spec_fold = False
                    shape_params = getattr(cache_params, "shape", None)
                    self.conv_shard_groups = getattr(
                        shape_params, "conv_shard_groups", None
                    )
                    self.conv_slice_axis = getattr(
                        shape_params, "conv_slice_axis", 0
                    )

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
                    self._is_contiguous = bool(layout.get("is_contiguous", True))

                    if not self._is_contiguous and speculative_num_draft_tokens is not None:
                        # Spec-decode's `update_mamba_state_after_mtp_verify`
                        # uses fused_mamba_state_scatter_with_mask, which
                        # needs a single contiguous (num_layers, slots, *)
                        # tensor.  Per-layer tensors can't satisfy that.
                        raise NotImplementedError(
                            "ElasticMambaPool does not support speculative "
                            "decoding with non-contiguous kvcached layout. "
                            "Re-launch with KVCACHED_CONTIGUOUS_LAYOUT=true "
                            "if you need spec decode for hybrid linear models."
                        )

                    if self._is_contiguous:
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
                    else:
                        # Non-contiguous: conv_state is List[List[Tensor]]
                        # (outer=conv shape, inner=layer); temporal_state is
                        # List[Tensor] (one per layer).
                        self.mamba_cache = _PerLayerMambaState(
                            conv_per_layer=conv_state,
                            temporal_per_layer=temporal_state,
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
                        pool_name="mamba",
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
                        f"layout={'contig' if self._is_contiguous else 'non-contig'}, "
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
                    if self._is_contiguous:
                        for i in range(len(self.mamba_cache.conv)):
                            self.mamba_cache.conv[i][:, select_index] = 0
                        self.mamba_cache.temporal[:, select_index] = 0
                    else:
                        # Per-layer zeroing: each layer's state tensor is
                        # (slots, *), so [select_index] slices on the slot
                        # dim directly.
                        for shape_list in self.mamba_cache.conv_per_layer:
                            for t in shape_list:
                                t[select_index] = 0
                        for t in self.mamba_cache.temporal_per_layer:
                            t[select_index] = 0
                    return select_index

                def free(self, free_index: Any) -> None:
                    if free_index.numel() == 0:
                        return
                    self.kvcached_allocator.free(free_index.tolist())

                def clear(self) -> None:
                    self.kvcached_allocator.clear()

                def register_slot_state(self, state: Any) -> None:
                    # SGLang 0.5.20 attaches Qwen4-Exp PLE side states
                    # (ShortConvPool / NGramPool) that must follow every slot
                    # clear, copy, and host round-trip.  The elastic pool
                    # does not implement that ride-along yet, so refuse
                    # instead of dropping sibling state silently.
                    raise NotImplementedError(
                        "ElasticMambaPool does not support SGLang PLE "
                        "slot-sibling states (register_slot_state) yet."
                    )

                def clear_slots(self, indices: "torch.Tensor") -> None:
                    # 0.5.20's deferred COW/clear on the extend path
                    # (ModelRunner._maybe_execute_deferred_mamba_cow_and_clear)
                    # calls this; the native body indexes (layers, slots, *)
                    # tensors, which on per-layer state would zero the wrong
                    # axis of layer 0 and miss every other layer.
                    if self._is_contiguous:
                        if hasattr(MambaPool, "clear_slots"):
                            super().clear_slots(indices)
                    else:
                        # Per-layer state is (slots, *shape); index the slot
                        # dim directly.  No _slot_siblings pass here:
                        # register_slot_state refuses, so none can exist.
                        for shape_list in self.mamba_cache.conv_per_layer:
                            for t in shape_list:
                                t[indices] = 0
                        for t in self.mamba_cache.temporal_per_layer:
                            t[indices] = 0

                def copy_from(
                    self, src_index: "torch.Tensor", dst_index: "torch.Tensor"
                ) -> None:
                    if self._is_contiguous:
                        if hasattr(MambaPool, "copy_from"):
                            super().copy_from(src_index, dst_index)
                    else:
                        for shape_list in self.mamba_cache.conv_per_layer:
                            for t in shape_list:
                                t[dst_index] = t[src_index]
                        for t in self.mamba_cache.temporal_per_layer:
                            t[dst_index] = t[src_index]

                def _iter_transfer_state_entries(self):
                    # The 0.5.20 PD-transfer readers (get_state_layer_ids,
                    # get_state_slice_outer_counts,
                    # get_state_conv_shard_groups) all walk this iterator,
                    # whose native body expects (layers, slots, *) tensors
                    # in vars(mamba_cache) and chokes on the nested
                    # conv_per_layer list.
                    if self._is_contiguous:
                        if hasattr(MambaPool, "_iter_transfer_state_entries"):
                            yield from super()._iter_transfer_state_entries()
                    else:
                        # Same flattening as the contiguous iterator: conv
                        # shape groups outer, layer inner, temporal last.
                        for shape_list in self.mamba_cache.conv_per_layer:
                            if shape_list[0].numel() == 0:
                                continue
                            for layer_index, layer_id in enumerate(
                                    self.mamba_layer_ids):
                                yield (
                                    "conv",
                                    shape_list[layer_index],
                                    self.conv_slice_axis,
                                    layer_id,
                                )
                        temporal_list = self.mamba_cache.temporal_per_layer
                        if temporal_list[0].numel() > 0:
                            for layer_index, layer_id in enumerate(
                                    self.mamba_layer_ids):
                                yield (
                                    "temporal",
                                    temporal_list[layer_index],
                                    0,
                                    layer_id,
                                )

                def get_contiguous_buf_infos(self):
                    if self._is_contiguous:
                        if hasattr(MambaPool, "get_contiguous_buf_infos"):
                            return super().get_contiguous_buf_infos()
                    else:
                        # Non-contiguous: per-layer pointer/length triples,
                        # aligned with the transfer iterator by sharing it.
                        data_ptrs: List[int] = []
                        data_lens: List[int] = []
                        item_lens: List[int] = []
                        entries = self._iter_transfer_state_entries()
                        for _, layer_t, _, _ in entries:
                            data_ptrs.append(layer_t.data_ptr())
                            data_lens.append(layer_t.nbytes)
                            item_lens.append(layer_t[0].nbytes)
                        return data_ptrs, data_lens, item_lens

                def get_state_dim_per_tensor(self):
                    if self._is_contiguous:
                        if hasattr(MambaPool, "get_state_dim_per_tensor"):
                            return super().get_state_dim_per_tensor()
                    else:
                        # Per-layer state shape is (slots, ...); the native
                        # reader takes shape[1 + slice_axis] (Kimi conv state
                        # slices its second per-slot axis) and 0 marks a
                        # replicated tensor that PD copies whole.
                        dim_per_tensor: List[int] = []
                        entries = self._iter_transfer_state_entries()
                        for _, layer_t, slice_axis, _ in entries:
                            if slice_axis is None:
                                dim_per_tensor.append(0)
                                continue
                            dim_per_tensor.append(
                                layer_t.shape[1 + slice_axis]
                            )
                        return dim_per_tensor

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

    @version_range(">=0.5.16")
    def rebind_hybrid_mamba_pool_cls(self, mem_pool_mod: types.ModuleType) -> bool:
        """Route ``HybridReqToTokenPool``'s Mamba pool construction to the
        elastic class.

        SGLang 0.5.16 made the pool class a ``mamba_pool_cls`` class
        attribute, which captured the native ``MambaPool`` when the module
        executed, before any aliasing ran.  The module-attribute alias no
        longer routes construction there: without the rebind, mamba state
        silently reverts to static native allocation and the slot-allocator
        wrap below no-ops because the pool lacks ``kvcached_allocator``.
        """
        HybridReqToTokenPool = getattr(mem_pool_mod, "HybridReqToTokenPool", None)
        if HybridReqToTokenPool is None:
            self.logger.debug(
                "HybridReqToTokenPool not found; skipping mamba_pool_cls rebind"
            )
            return True
        if not hasattr(HybridReqToTokenPool, "mamba_pool_cls"):
            self.logger.debug(
                "HybridReqToTokenPool has no mamba_pool_cls; skipping rebind"
            )
            return True

        ElasticMambaPool = getattr(mem_pool_mod, "ElasticMambaPool", None)
        if ElasticMambaPool is None:
            # Injection was skipped (no native MambaPool to subclass).
            return True

        HybridReqToTokenPool.mamba_pool_cls = ElasticMambaPool
        return True

    @version_range(">=0.5.13")
    def patch_mamba_slot_allocator(self, mem_pool_mod: types.ModuleType) -> bool:
        """Route SGLang's request-level Mamba slots through kvcached.

        SGLang 0.5.13 split slot ownership out of ``MambaPool`` into a
        separate ``MambaSlotAllocator``.  Merely aliasing ``MambaPool`` then
        leaves ``ElasticMambaPool.alloc/free`` dead and never maps the VMM
        pages for request slots.  Wrap ``HybridReqToTokenPool._init_mamba_pool``
        so the allocator installed after pool construction owns the same IDs
        through the pool's ``KVCacheManager``.
        """
        HybridReqToTokenPool = getattr(mem_pool_mod, "HybridReqToTokenPool", None)
        if HybridReqToTokenPool is None:
            self.logger.debug(
                "HybridReqToTokenPool not found; skipping Mamba slot allocator patch"
            )
            return True

        original_init = getattr(HybridReqToTokenPool, "_init_mamba_pool", None)
        if original_init is None:
            self.logger.debug(
                "HybridReqToTokenPool._init_mamba_pool not found; skipping"
            )
            return True
        if "MambaSlotAllocator" not in getattr(original_init, "__globals__", {}):
            # SGLang <=0.5.12 keeps allocation on MambaPool.alloc/free, which
            # ElasticMambaPool already overrides directly.
            self.logger.debug(
                "SGLang uses MambaPool-owned slots; no separate allocator patch needed"
            )
            return True
        marker = "__kvcached_mamba_slot_allocator_patched__"
        if self._is_already_patched(original_init, marker):
            return True

        import torch

        class ElasticMambaSlotAllocator:
            """SGLang Mamba-slot interface backed by ``ElasticMambaPool``."""

            def __init__(self, size: int, device: str, mamba_pool: Any) -> None:
                self.size = size
                self.device = device
                self.mamba_pool = mamba_pool
                self._alloc_iter = None
                self._free_ids = set(range(1, size + 1))

            @property
            def free_slots(self):
                # Compatibility for SGLang's debug invariant checker. Normal
                # scheduling uses available_size() and does not materialize it.
                return torch.tensor(
                    sorted(self._free_ids), dtype=torch.int64, device=self.device
                )

            def available_size(self) -> int:
                return self.mamba_pool.available_size()

            def schedulable_available_size(self) -> int:
                return self.available_size()

            def _do_alloc(self, need_size: int):
                slots = self.mamba_pool.alloc(need_size)
                if slots is None:
                    return None
                self._free_ids.difference_update(slots.tolist())
                return slots

            def alloc(self, need_size: int):
                if self._alloc_iter is not None and need_size == 1:
                    slot = next(self._alloc_iter, None)
                    if slot is not None:
                        return slot
                return self._do_alloc(need_size)

            def free(self, free_index) -> None:
                if free_index.numel() == 0:
                    return
                block_ids = free_index.tolist()
                self.mamba_pool.free(free_index)
                self._free_ids.update(block_ids)

            def alloc_group_begin(self, num_reqs: int) -> None:
                self._alloc_iter = None
                if num_reqs > 0:
                    result = self._do_alloc(num_reqs)
                    if result is not None:
                        self._alloc_iter = iter(result.split(1))

            def alloc_group_end(self) -> None:
                if self._alloc_iter is not None:
                    remaining = list(self._alloc_iter)
                    if remaining:
                        self.free(torch.cat(remaining))
                self._alloc_iter = None

            def clear(self) -> None:
                self.mamba_pool.clear()
                self._alloc_iter = None
                self._free_ids = set(range(1, self.size + 1))

        def _patched_init_mamba_pool(self, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            mamba_pool = getattr(self, "mamba_pool", None)
            if mamba_pool is None or not hasattr(mamba_pool, "kvcached_allocator"):
                return
            self.mamba_allocator = ElasticMambaSlotAllocator(
                size=mamba_pool.size,
                device=mamba_pool.device,
                mamba_pool=mamba_pool,
            )
            logger.info(
                "[kvcached] ElasticMambaSlotAllocator in use: size=%d",
                mamba_pool.size,
            )

        self._mark_as_patched(_patched_init_mamba_pool, marker)
        HybridReqToTokenPool._init_mamba_pool = _patched_init_mamba_pool
        setattr(mem_pool_mod, "ElasticMambaSlotAllocator", ElasticMambaSlotAllocator)
        return True


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
        """Patch scheduler to suppress memory leak check when kvcached is enabled.

        kvcached maps physical KV pages lazily, so SGLang's static-pool
        invariant (total == available + in-use) does not hold and its leak
        detector would raise spuriously.  We neutralize the leak *raisers*.

        Older SGLang keeps the whole check in a single Scheduler method whose
        source mentions ``token_to_kv_pool_allocator``.  Newer SGLang
        moved it into helpers such as ``SchedulerRuntimeCheckerMixin`` or
        ``SchedulerInvariantChecker`` (e.g. ``_check_req_pool`` raises directly,
        ``_report_leak`` is the generic choke point for *token/KV* pool leaks).

        We suppress only the leak checks for pools kvcached actually manages
        (the KV / token pools).  A check that is specific to
        ``req_to_token_pool`` is deliberately left intact -- kvcached does not
        manage the request pool, its invariant still holds, and silencing it
        would hide a genuine request-pool leak.  The old single-method layout
        (which names ``token_to_kv_pool_allocator``) and the new generic
        reporter (which names no pool, and is only ever called for the token
        pools) are both kept; only the req-pool-specific check is skipped.
        """
        Scheduler = self._get_target_class(sched_mod)
        if Scheduler is None:
            return False

        target_classes: List[Tuple[str, Any]] = [("Scheduler", Scheduler)]
        InvariantChecker = getattr(sched_mod, "SchedulerInvariantChecker", None)
        if InvariantChecker is not None:
            target_classes.append(("SchedulerInvariantChecker", InvariantChecker))

        target_methods: List[Tuple[str, Any, str]] = []
        for class_name, cls in target_classes:
            for name, fn in inspect.getmembers(cls, predicate=inspect.isfunction):
                try:
                    src = inspect.getsource(fn)
                except Exception:
                    continue
                if "memory leak detected" not in src:
                    continue
                # Skip a check that is specific to the request pool, which kvcached
                # does not manage.  The generic reporter names no pool (so it is not
                # excluded) and the legacy combined check names the KV allocator.
                if "req_to_token_pool" in src and "token_to_kv_pool" not in src:
                    continue
                target_methods.append((class_name, cls, name))

        if not target_methods:
            self.logger.debug(
                "No memory leak detection method found in SGLang scheduler"
            )
            return False

        def _make_wrapped(original: Callable[..., Any]) -> Callable[..., Any]:
            import functools

            @functools.wraps(original)
            def _wrapped(sched_self, *args: Any, **kwargs: Any):
                # Disable memory leak detection when ENABLE_KVCACHED is set
                if enable_kvcached():
                    return
                return original(sched_self, *args, **kwargs)

            return _wrapped

        patched_any = False
        for class_name, cls, target_method_name in target_methods:
            original = getattr(cls, target_method_name)
            if self._is_already_patched(original):
                self.logger.debug(
                    f"{class_name}.{target_method_name} leak check already patched"
                )
                patched_any = True
                continue

            wrapped = _make_wrapped(original)
            self._mark_as_patched(wrapped)
            setattr(cls, target_method_name, wrapped)
            patched_any = True

        return patched_any


class _RadixBlockIndex:
    def __init__(self, root_node: Any, logical_page_size: int) -> None:
        self.root_node = root_node
        self.logical_page_size = logical_page_size
        self.block_owners: Dict[int, Any] = {}
        self.node_blocks: Dict[Any, Tuple[int, ...]] = {}

    def sync(self, nodes: Set[Any]) -> None:
        changed_nodes = [
            node
            for node in nodes
            if node not in self.node_blocks
            or self.node_blocks[node]
            is not getattr(node.value, "_kvcached_block_ids", None)
        ]
        removed_nodes = self.node_blocks.keys() - nodes
        for node in removed_nodes:
            for block_id in self.node_blocks.pop(node, ()):
                if self.block_owners.get(block_id) is node:
                    del self.block_owners[block_id]
        for node in changed_nodes:
            if node not in self.node_blocks:
                continue
            for block_id in self.node_blocks.pop(node):
                if self.block_owners.get(block_id) is node:
                    del self.block_owners[block_id]

        uncached = [
            node
            for node in changed_nodes
            if getattr(node.value, "_kvcached_block_ids", None) is None
        ]
        if uncached:
            import torch

            values = [node.value for node in uncached]
            lengths = [int(value.numel()) // self.logical_page_size for value in values]
            if self.logical_page_size > 1:
                flattened = torch.cat(
                    [value[:: self.logical_page_size] for value in values]
                )
                flattened = flattened // self.logical_page_size
            else:
                flattened = torch.cat(values)
            new_block_ids = cast(List[int], cast(Any, flattened).tolist())
            offset = 0
            for node, length in zip(uncached, lengths):
                node.value._kvcached_block_ids = tuple(
                    new_block_ids[offset : offset + length]
                )
                offset += length

        for node in changed_nodes:
            block_ids = cast(Tuple[int, ...], node.value._kvcached_block_ids)
            self.node_blocks[node] = block_ids
            for block_id in block_ids:
                self.block_owners[block_id] = node

    def count_tokens(self, nodes: Dict[Any, int]) -> int:
        return sum(
            int(node.value.numel()) - split_len
            for node, split_len in nodes.items()
        )


def _evictable_radix_nodes(radix_cache: Any) -> Set[Any]:
    """Return every node reachable by repeatedly evicting current leaves."""
    root = radix_cache.root_node
    nodes: Set[Any] = set()
    for leaf in radix_cache.evictable_leaves:
        node = leaf
        while node is not root and node.lock_ref == 0:
            if node in nodes:
                break
            nodes.add(node)
            node = node.parent
    return nodes


_RadixSuffixPlan = Tuple[Any, int]
_RadixCandidate = Tuple[Tuple[Any, ...], Tuple[_RadixSuffixPlan, ...]]


class _RadixEvictionSelection(NamedTuple):
    suffix_plans: List[_RadixSuffixPlan]
    token_count: int


def _align_radix_eviction_budget(radix_cache: Any, num_tokens: int) -> int:
    """Round eviction up because radix nodes split only on logical pages.

    This is equivalent to rounding the effective cache cap down. It may evict
    up to one page minus one extra token, but avoids an unsafe partial-page
    split; the cache's evictable size remains the upper bound.
    """
    logical_page_size = max(1, int(radix_cache.page_size))
    aligned_tokens = (
        num_tokens + logical_page_size - 1
    ) // logical_page_size * logical_page_size
    return min(aligned_tokens, int(radix_cache.evictable_size_))


def _radix_suffix_closure(
    index: _RadixBlockIndex,
    plans: Sequence[_RadixSuffixPlan],
) -> Optional[Tuple[List[_RadixSuffixPlan], Dict[Any, int]]]:
    """Normalize suffix roots and expand them to their descendant closure."""
    split_by_node: Dict[Any, int] = {}
    for node, split_len in plans:
        previous = split_by_node.get(node)
        split_by_node[node] = (
            split_len if previous is None else min(previous, split_len)
        )

    planned_ids = {id(node) for node in split_by_node}
    roots: List[_RadixSuffixPlan] = []
    for node, split_len in split_by_node.items():
        ancestor = node.parent
        while ancestor is not None and id(ancestor) not in planned_ids:
            ancestor = ancestor.parent
        if ancestor is None:
            roots.append((node, split_len))

    closure: Dict[Any, int] = {}
    stack = list(roots)
    while stack:
        node, retained_prefix = stack.pop()
        if node not in index.node_blocks or node.lock_ref != 0:
            return None
        closure[node] = retained_prefix
        stack.extend((child, 0) for child in node.children.values())
    return roots, closure


def _select_page_aware_plan(
    radix_cache: Any,
    token_budget: int,
) -> _RadixEvictionSelection:
    empty = _RadixEvictionSelection([], 0)
    if token_budget <= 0:
        return empty

    allocator = getattr(radix_cache, "token_to_kv_pool_allocator", None)
    manager = getattr(allocator, "kvcached_allocator", None)
    if manager is None or getattr(manager, "page_allocator", None) is None:
        return empty

    logical_page_size = max(1, int(getattr(radix_cache, "page_size", 1)))
    if int(getattr(allocator, "page_size", logical_page_size)) != logical_page_size:
        return empty
    can_split = callable(getattr(radix_cache, "_split_node", None))

    evictable_nodes = _evictable_radix_nodes(radix_cache)
    if not evictable_nodes:
        return empty

    index = getattr(radix_cache, "_kvcached_radix_block_index", None)
    if (
        index is None
        or index.root_node is not radix_cache.root_node
        or index.logical_page_size != logical_page_size
    ):
        index = _RadixBlockIndex(radix_cache.root_node, logical_page_size)
        radix_cache._kvcached_radix_block_index = index
    index.sync(evictable_nodes)

    by_page = manager.page_allocator.group_indices_by_page(
        list(index.block_owners), manager.block_mem_size
    )
    occupancy = manager.get_page_occupancy(list(by_page))
    reclaimable_pages = [
        block_ids
        for page_id, block_ids in by_page.items()
        if len(block_ids) == occupancy.get(page_id, 0)
    ]
    if not reclaimable_pages:
        return empty

    block_offsets = {
        block_id: offset
        for node in index.node_blocks
        for offset, block_id in enumerate(index.node_blocks[node])
    }
    candidates: List[_RadixCandidate] = []
    seen_plans: Set[frozenset[Tuple[Any, int]]] = set()

    for target_blocks in reclaimable_pages:
        split_by_node: Dict[Any, int] = {}
        for block_id in target_blocks:
            owner = index.block_owners[block_id]
            block_offset = block_offsets[block_id]
            previous = split_by_node.get(owner)
            split_by_node[owner] = (
                block_offset if previous is None else min(previous, block_offset)
            )
        plan = _radix_suffix_closure(
            index,
            [
                (node, block_offset * logical_page_size)
                for node, block_offset in split_by_node.items()
            ],
        )
        if plan is None:
            continue
        suffix_plans, closure = plan
        if not can_split and any(split_len > 0 for _, split_len in suffix_plans):
            continue
        plan_key = frozenset(suffix_plans)
        if plan_key in seen_plans:
            continue
        seen_plans.add(plan_key)

        token_cost = index.count_tokens(closure)
        if token_cost > token_budget:
            continue

        selected_blocks: Dict[int, int] = {}
        for node, split_len in closure.items():
            block_offset = split_len // logical_page_size
            suffix_blocks = index.node_blocks[node][block_offset:]
            for block_id in suffix_blocks:
                page_id = block_id * manager.block_mem_size // manager.page_size
                selected_blocks[page_id] = selected_blocks.get(page_id, 0) + 1

        completed_pages = [
            page_id
            for page_id, count in selected_blocks.items()
            if count == occupancy.get(page_id, 0)
        ]
        priority = max(
            radix_cache.eviction_strategy.get_priority(node) for node in closure
        )
        sort_key = (
            token_cost / len(completed_pages),
            -len(completed_pages),
            token_cost,
            priority,
            min(completed_pages),
        )
        candidates.append((sort_key, tuple(suffix_plans)))

    selected_plans: List[_RadixSuffixPlan] = []
    selected_tokens = 0
    for _sort_key, candidate_plans in sorted(candidates, key=lambda item: item[0]):
        plan = _radix_suffix_closure(index, [*selected_plans, *candidate_plans])
        assert plan is not None
        merged_plans, closure = plan
        merged_tokens = index.count_tokens(closure)
        if merged_tokens <= selected_tokens or merged_tokens > token_budget:
            continue
        selected_plans = merged_plans
        selected_tokens = merged_tokens
        if selected_tokens == token_budget:
            break

    return _RadixEvictionSelection(selected_plans, selected_tokens)


def _split_eviction_suffixes(
    radix_cache: Any,
    plans: Sequence[_RadixSuffixPlan],
) -> Set[Any]:
    for node, split_len in plans:
        if split_len > 0:
            radix_cache._split_node(node.key, node, split_len)

    suffix_closures: Set[Any] = set()
    stack = [node for node, _split_len in plans]
    while stack:
        current = stack.pop()
        if current in suffix_closures:
            continue
        suffix_closures.add(current)
        stack.extend(current.children.values())
    return suffix_closures


def _make_sglang_evict_arg(
    num_tokens: int, evict_params_cls: Optional[Callable[..., Any]]
) -> Any:
    """Build the eviction argument used by the installed SGLang version."""
    if evict_params_cls is None:
        return num_tokens
    return evict_params_cls(num_tokens=num_tokens)


def _evict_radix_cache_page_aware(
    radix_cache: Any,
    num_tokens: int,
    evict_params_cls: Callable[..., Any],
) -> Any:
    """Evict at least ``num_tokens`` in legal logical-page units."""
    eviction_budget = _align_radix_eviction_budget(radix_cache, num_tokens)
    if eviction_budget >= radix_cache.evictable_size_:
        try:
            return radix_cache.evict(
                _make_sglang_evict_arg(
                    num_tokens=eviction_budget,
                    evict_params_cls=evict_params_cls,
                )
            )
        finally:
            index = getattr(radix_cache, "_kvcached_radix_block_index", None)
            if index is not None:
                index.sync(_evictable_radix_nodes(radix_cache))

    original_strategy = radix_cache.eviction_strategy

    def evict_selected(selected: Set[Any], token_count: int) -> Any:
        radix_cache.eviction_strategy = types.SimpleNamespace(
            get_priority=lambda node: (
                node not in selected,
                original_strategy.get_priority(node),
            )
        )
        try:
            result = radix_cache.evict(
                _make_sglang_evict_arg(token_count, evict_params_cls)
            )
        finally:
            radix_cache.eviction_strategy = original_strategy
        if result is not None and result.num_tokens_evicted != token_count:
            raise RuntimeError(
                "SGLang radix eviction did not honor the exact selected-node budget"
            )
        return result

    selection = _select_page_aware_plan(radix_cache, eviction_budget)
    result = None
    try:
        if selection.suffix_plans:
            selected = _split_eviction_suffixes(radix_cache, selection.suffix_plans)
            result = evict_selected(selected, selection.token_count)

        remaining = eviction_budget - selection.token_count
        if remaining > 0:
            # Let SGLang choose the remaining victims. Native eviction removes
            # complete radix nodes, so the actual count may exceed `remaining`.
            native_result = radix_cache.evict(
                _make_sglang_evict_arg(remaining, evict_params_cls)
            )
            if native_result is not None:
                native_result.num_tokens_evicted += selection.token_count
                result = native_result
        return result
    finally:
        index = getattr(radix_cache, "_kvcached_radix_block_index", None)
        if index is not None:
            index.sync(_evictable_radix_nodes(radix_cache))


class RadixCacheLimitPatch(VersionAwarePatch, BasePatch):
    """Enforce KVCACHED_MAX_CACHED_TOKENS limit on SGLang's RadixCache.

    After each finished request is inserted into the radix cache, if the
    evictable (cached) size exceeds the configured limit, immediately evict
    down to that limit.  This prevents the cache from consuming all KV pool
    capacity and leaves headroom for running requests.

    KVCACHED_MAX_CACHED_TOKENS semantics:
      < 0  → unlimited; this patch is a no-op.
      == 0 → disabled; every cached entry is evicted immediately on insert.
      > 0  → cap evictable size at this many tokens.
    """

    library = "sglang"
    target_module = "sglang.srt.mem_cache.radix_cache"
    target_class = "RadixCache"
    patch_name = "radix_cache_limit"

    def apply(self, radix_cache_mod: types.ModuleType) -> bool:
        if MAX_CACHED_TOKENS < 0:
            logger.debug(
                "KVCACHED_MAX_CACHED_TOKENS < 0 (unlimited) — "
                "RadixCacheLimitPatch skipped"
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
        evict_params_cls = getattr(radix_cache_mod, "EvictParams", None)
        # SGLang < 0.5.9 computes leaves on demand and does not expose the
        # persistent evictable_leaves set required by the page-aware planner.

        def _wrapped(self_rc: Any, *args: Any, **kwargs: Any) -> None:
            original_cache_finished(self_rc, *args, **kwargs)
            excess = self_rc.evictable_size_ - max_cached
            if excess > 0:
                if type(self_rc) is RadixCache and evict_params_cls is not None:
                    _evict_radix_cache_page_aware(
                        radix_cache=self_rc,
                        num_tokens=excess,
                        evict_params_cls=evict_params_cls,
                    )
                else:
                    self_rc.evict(
                        _make_sglang_evict_arg(
                            num_tokens=excess,
                            evict_params_cls=evict_params_cls,
                        )
                    )

        self._mark_as_patched(_wrapped)
        RadixCache.cache_finished_req = _wrapped  # type: ignore

        if evict_params_cls is not None:
            original_reset = RadixCache.reset

            @functools.wraps(original_reset)
            def _wrapped_reset(
                self_rc: Any, *args: Any, **kwargs: Any
            ) -> Any:
                result = original_reset(self_rc, *args, **kwargs)
                self_rc._kvcached_radix_block_index = None
                return result

            self._mark_as_patched(
                _wrapped_reset, "__kvcached_radix_block_index_reset__"
            )
            RadixCache.reset = _wrapped_reset  # type: ignore

        logger.info(
            f"[kvcached] RadixCache evictable size capped at "
            f"{max_cached} tokens (KVCACHED_MAX_CACHED_TOKENS)"
        )
        return True

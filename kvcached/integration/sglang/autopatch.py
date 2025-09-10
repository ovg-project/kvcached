# SPDX-FileCopyrightText: 2025 contributors to the kvcached project
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import os
import types
from typing import Any, Union

from wrapt.importer import when_imported

from kvcached.utils import get_kvcached_logger

logger = get_kvcached_logger(__name__)


def _env_enabled() -> bool:
    return os.getenv("KVCACHED_AUTOPATCH", "false").lower() in ("true", "1")


def _enable_kvcached() -> bool:
    return os.getenv("ENABLE_KVCACHED", "false").lower() in ("true", "1")


@when_imported("sglang.srt.mem_cache.allocator")
def _on_allocator_import(alloc_mod: types.ModuleType) -> None:
    if not _env_enabled():
        logger.debug("Disabled by KVCACHED_AUTOPATCH")
        return
    injected = _inject_elastic_allocator(alloc_mod)
    aliased = _alias_allocator_to_elastic(alloc_mod)
    if injected or aliased:
        logger.info("Patched sglang allocator (elastic=%s, alias=%s)",
                    injected, aliased)


@when_imported("sglang.srt.mem_cache.memory_pool")
def _on_memory_pool_import(mem_pool_mod: types.ModuleType) -> None:
    if not _env_enabled():
        logger.debug("Disabled by KVCACHED_AUTOPATCH")
        return
    injected = _inject_elastic_mem_pool(mem_pool_mod)
    aliased = _alias_mem_pool_to_elastic(mem_pool_mod)
    if injected or aliased:
        logger.info("Patched sglang memory_pool (elastic=%s, alias=%s)",
                    injected, aliased)


@when_imported("sglang.srt.managers.scheduler")
def _on_scheduler_import(sched_mod: types.ModuleType) -> None:
    if not _env_enabled():
        logger.debug("Disabled by KVCACHED_AUTOPATCH")
        return
    if _patch_scheduler_memory_leak(sched_mod):
        logger.info(
            "Patched sglang scheduler (suppress memory leak check when elastic)"
        )


def _inject_elastic_allocator(alloc_mod: types.ModuleType) -> bool:
    if hasattr(alloc_mod, "ElasticTokenToKVPoolAllocator"):
        return True

    import torch
    BaseTokenToKVPoolAllocator = getattr(alloc_mod,
                                         "BaseTokenToKVPoolAllocator")

    class ElasticTokenToKVPoolAllocator(
            BaseTokenToKVPoolAllocator  # type: ignore[misc, valid-type]
    ):

        def __init__(self, size: int, dtype, device: str, kvcache) -> None:
            super().__init__(size, 1, dtype, device, kvcache)
            if not hasattr(kvcache, "kvcached_allocator"):
                raise ValueError(
                    "ElasticTokenToKVPoolAllocator requires elastic MHA pool")
            if "cuda" not in device:
                raise ValueError(
                    "ElasticTokenToKVPoolAllocator only supports cuda device")
            self.kvcached_allocator = kvcache.kvcached_allocator

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

    setattr(alloc_mod, "ElasticTokenToKVPoolAllocator",
            ElasticTokenToKVPoolAllocator)
    return True


def _inject_elastic_mem_pool(mem_pool_mod: types.ModuleType) -> bool:
    if hasattr(mem_pool_mod, "ElasticMHATokenToKVPool"):
        return True

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
            enable_overlap_schedule: bool = True,
        ) -> None:
            # Call grandparent (KVCache) initializer because we redefine
            # all member variables.
            super(MHATokenToKVPool, self).__init__(
                size=size,
                page_size=page_size,
                dtype=dtype,
                layer_num=layer_num,
                device=device,
                enable_memory_saver=enable_memory_saver,
                start_layer=start_layer,
                end_layer=end_layer,
            )
            self.head_num = head_num
            self.head_dim = head_dim
            try:
                import torch
                from sglang.srt.utils import is_cuda

                import kvcached.integration.sglang.interfaces as kvi

                # Initialize kvcached with overlap scheduling if requested
                kvi.init_kvcached(async_sched=enable_overlap_schedule)
                # Per-token KV bytes
                self.cell_size = self.head_num * self.head_dim * dtype.itemsize
                # Elastic allocator
                self.kvcached_allocator = kvi.get_kv_cache_manager(
                    size + page_size, page_size, self.cell_size, layer_num)
            except Exception:
                raise

            # Allocate K/V buffers via kvcached
            self._create_buffers()

            self.layer_transfer_counter = None
            self.device_module = torch.get_device_module(self.device)
            self.alt_stream = self.device_module.Stream() if is_cuda(
            ) else None

            k_size, v_size = self.get_kv_size_bytes()
            GB = 1024**3
            k_size_phy, v_size_phy = self.get_kv_size_bytes_phy()

            logger.info(f"VirtualKV Cache is allocated. #tokens: {size}, K size: "
                        f"{k_size / GB:.2f} GB, V size: {v_size / GB:.2f} GB")
            logger.info(f"Physical KV Cache limits by --mem-fraction-static: "
                        f"#tokens: {size}, K size: "
                        f"{k_size_phy / GB:.2f} GB, V size: {v_size_phy / GB:.2f} GB")
            
            self.mem_usage = (k_size + v_size) / GB

        def __del__(self):  # best-effort cleanup
            try:
                import kvcached.integration.sglang.interfaces as kvi

                kvi.shutdown_kvcached()
            except Exception:
                pass

        def _create_buffers(self):
            import kvcached.integration.sglang.interfaces as kvi

            if "cuda" not in self.device:
                raise ValueError(
                    "ElasticMHATokenToKVPool only supports cuda device")
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


def _alias_allocator_to_elastic(alloc_mod: types.ModuleType) -> bool:
    if getattr(alloc_mod, "__kvcached_patched__", False) is not True:
        alloc_mod.__kvcached_patched__ = True  # type: ignore[attr-defined]
        try:
            ElasticTokenToKVPoolAllocator = getattr(
                alloc_mod, "ElasticTokenToKVPoolAllocator")
            if ElasticTokenToKVPoolAllocator is None:
                return False
            alloc_mod.TokenToKVPoolAllocator = (  # type: ignore
                ElasticTokenToKVPoolAllocator)
            return True
        except Exception:
            logger.warning("Failed to alias allocator to elastic one")
            return False
    return True


def _alias_mem_pool_to_elastic(mem_pool_mod: types.ModuleType) -> bool:
    if getattr(mem_pool_mod, "__kvcached_patched__", False) is not True:
        mem_pool_mod.__kvcached_patched__ = True  # type: ignore[attr-defined]
        try:
            ElasticMHATokenToKVPool = getattr(mem_pool_mod,
                                              "ElasticMHATokenToKVPool")
            if ElasticMHATokenToKVPool is None:
                return False
            # Alias defaults so core code will use elastic variants
            mem_pool_mod.MHATokenToKVPool = ElasticMHATokenToKVPool  # type: ignore
            return True
        except Exception:
            logger.warning("Failed to alias memory_pool to elastic one")
            return False
    return True


def _patch_scheduler_memory_leak(sched_mod: types.ModuleType) -> bool:
    Scheduler = getattr(sched_mod, "Scheduler")

    target_method_name: Union[str, None] = None
    for name, fn in inspect.getmembers(Scheduler,
                                       predicate=inspect.isfunction):
        try:
            src = inspect.getsource(fn)
        except Exception:
            continue
        if "token_to_kv_pool_allocator memory leak detected!" in src or (
                "memory leak detected" in src
                and "token_to_kv_pool_allocator" in src):
            target_method_name = name
            break

    if target_method_name is None:
        return False

    original = getattr(Scheduler, target_method_name)

    def _wrapped(self, *args: Any, **kwargs: Any):
        # TODO: (YIFAN) This is a simple but coarse-grained workaround that
        # disables memory leak detection when ENABLE_KVCACHED is set.
        if _enable_kvcached():
            return
        return original(self, *args, **kwargs)

    if getattr(original, "__kvcached_patched__", False) is not True:
        _wrapped.__kvcached_patched__ = True  # type: ignore[attr-defined]
        setattr(Scheduler, target_method_name, _wrapped)
    return True

from typing import List, Optional, Tuple

import torch

from kvcached.kv_cache_manager import KVCacheManager
from kvcached.utils import PAGE_SIZE
from kvcached.vmm_ops import create_kv_tensors
from kvcached.vmm_ops import init_kvcached as _init_kvcached_impl
from kvcached.vmm_ops import shutdown_kvcached as _shutdown_kvcached_impl

_kvcached_initialized: bool = False
_kvcached_device = None


def init_kvcached(tp_rank: int = 0,
                  tp_size: int = 1,
                  device: Optional[str] = None) -> None:
    global _kvcached_initialized, _kvcached_device
    if _kvcached_initialized:
        return

    if device is None:
        device = f"cuda:{torch.cuda.current_device()}"

    _init_kvcached_impl(device)
    _kvcached_initialized = True
    _kvcached_device = device


def shutdown_kvcached() -> None:
    global _kvcached_initialized, _kvcached_device
    if not _kvcached_initialized:
        return

    _shutdown_kvcached_impl()
    _kvcached_initialized = False
    _kvcached_device = None


def alloc_kv_cache(
    num_tokens: int,
    head_num: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
    num_layers: int,
    page_size: int = 1,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    if not _kvcached_initialized:
        raise RuntimeError(
            "kvcached is not initialized. Please call init_kvcached() first.")

    assert torch.cuda.is_available(), "CUDA is not available."
    if page_size != 1:
        print(
            "Warning: kvcached is only tested with page_size = 1 for SGLang.")

    # SGLang named it "page" to be consistent with PagedAttention. But we call
    # it "block" to distinguish a KV cache block and a physical memory page.
    block_size = page_size
    block_mem_size = head_num * head_dim * dtype.itemsize * block_size
    blocks_per_page = PAGE_SIZE // block_mem_size

    gpu_mem_size = torch.cuda.get_device_properties(device).total_memory
    num_pages = gpu_mem_size // num_layers // 2 // PAGE_SIZE
    virtual_mem_size = num_pages * PAGE_SIZE * 2

    raw_kv_tensors = create_kv_tensors(virtual_mem_size, dtype.itemsize,
                                       device, num_layers)

    assert block_size * blocks_per_page * num_pages >= num_tokens, \
        "Not enough memory to allocate KV cache."
    num_tokens = block_size * blocks_per_page * num_pages

    kv_shape = (num_tokens, head_num, head_dim)
    k_tensors, v_tensors = [], []
    for t in raw_kv_tensors:
        t = t.view(2, *kv_shape).view(dtype=dtype)
        k_tensors.append(t.narrow(0, 0, 1).view(kv_shape))
        v_tensors.append(t.narrow(0, 1, 1).view(kv_shape))

    return k_tensors, v_tensors


def get_kv_cache_manager(num_blocks: int, block_size: int, cell_size: int,
                         num_layers: int) -> KVCacheManager:
    if not _kvcached_initialized:
        raise RuntimeError(
            "kvcached is not initialized. Please call init_kvcached() first.")

    return KVCacheManager(num_blocks, block_size, cell_size, num_layers)

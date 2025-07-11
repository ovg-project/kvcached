import math
from typing import List, Optional, Tuple

import torch

from kvcached.kv_cache_manager import KVCacheManager
from kvcached.utils import PAGE_SIZE
from kvcached.vmm_ops import create_kv_tensors
from kvcached.vmm_ops import init_kvcached as _init_kvcached_impl
from kvcached.vmm_ops import shutdown_kvcached as _shutdown_kvcached_impl


def init_kvcached(tp_rank: int = 0,
                  tp_size: int = 1,
                  device: Optional[str] = None) -> None:
    if device is None:
        device = f"cuda:{torch.cuda.current_device()}"
    _init_kvcached_impl(device)


def shutdown_kvcached() -> None:
    _shutdown_kvcached_impl()


def alloc_kv_cache(
    kvcache_shape: Tuple[int, ...],
    block_size: int,
    dtype: torch.dtype,
    device: str,
    num_layers: int,
) -> List[torch.Tensor]:
    assert (len(kvcache_shape) > 2 and kvcache_shape[0]
            == 2), "Only supports stacked kv cache at 1st dim."
    assert torch.cuda.is_available(), "CUDA is not available."

    kvcache_shape = list(kvcache_shape)
    block_mem_size = math.prod(kvcache_shape[2:]) * dtype.itemsize
    blocks_per_page = PAGE_SIZE // block_mem_size

    gpu_mem_size = torch.cuda.get_device_properties(device).total_memory

    num_pages = gpu_mem_size // num_layers // 2 // PAGE_SIZE
    num_blocks = num_pages * blocks_per_page
    kvcache_shape[1] = num_blocks
    virtual_mem_size = math.prod(kvcache_shape) * dtype.itemsize

    raw_kv_tensors = create_kv_tensors(virtual_mem_size, dtype.itemsize,
                                       device, num_layers)

    kv_tensors = [
        t.view(kvcache_shape).view(dtype=dtype) for t in raw_kv_tensors
    ]
    return kv_tensors


def get_kv_cache_manager(num_blocks: int, block_size: int, cell_size: int,
                         num_layers: int) -> KVCacheManager:
    return KVCacheManager(num_blocks, block_size, cell_size, num_layers)

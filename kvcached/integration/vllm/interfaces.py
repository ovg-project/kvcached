import math
from typing import List, Optional, Tuple

import torch

from kvcached.kv_cache_manager import KVCacheManager
from kvcached.tp_ipc_util import start_worker_listerner_thread
from kvcached.utils import PAGE_SIZE
from kvcached.vmm_ops import create_kv_tensors
from kvcached.vmm_ops import init_kvcached as _init_kvcached_impl
from kvcached.vmm_ops import shutdown_kvcached as _shutdown_kvcached_impl

_kvcached_initialized: bool = False
_kvcached_device = None
_tp_size: int = 1


def init_kvcached(
    tp_rank: int = 0,
    tp_size: int = 1,
    is_worker: bool = False,
    device: Optional[str] = None,
) -> None:
    global _kvcached_initialized, _kvcached_device, _tp_size

    if _kvcached_initialized:
        return

    if device is None:
        device = f"cuda:{torch.cuda.current_device()}"

    _init_kvcached_impl(device)
    _kvcached_initialized = True
    _kvcached_device = device
    _tp_size = tp_size

    if tp_size > 1 and is_worker:
        # start the listener thread for tensor parallel kv cache management
        start_worker_listerner_thread(tp_rank)


def shutdown_kvcached() -> None:
    global _kvcached_initialized, _kvcached_device
    if not _kvcached_initialized:
        return

    _shutdown_kvcached_impl()
    _kvcached_initialized = False
    _kvcached_device = None


def alloc_kv_cache(
    kvcache_shape: Tuple[int, ...],
    block_size: int,
    dtype: torch.dtype,
    device: str,
    num_layers: int,
) -> List[torch.Tensor]:
    if not _kvcached_initialized:
        raise RuntimeError(
            "kvcached is not initialized. Please call init_kvcached() first.")

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
    if not _kvcached_initialized:
        raise RuntimeError(
            "kvcached is not initialized. Please call init_kvcached() first.")

    return KVCacheManager(num_blocks, block_size, cell_size, num_layers,
                          _tp_size)

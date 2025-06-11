import math
from typing import List, Tuple

import torch

from kvcached.slab_allocator import PAGE_SIZE
from kvcached.vmm_ops import create_kv_tensors
from kvcached.vmm_ops import init_kvcached as _init_kvcached_impl
from kvcached.vmm_ops import shutdown_kvcached as _shutdown_kvcached_impl


def init_kvcached() -> None:
    device = f"cuda:{torch.cuda.current_device()}"
    _init_kvcached_impl(device)


def shutdown_kvcached() -> None:
    _shutdown_kvcached_impl()


def vllm_alloc_kv_cache(
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


def sgl_alloc_kv_cache(
    num_tokens: int,
    head_num: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
    num_layers: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    assert torch.cuda.is_available(), "CUDA is not available."
    gpu_mem_size = torch.cuda.get_device_properties(device).total_memory
    virtual_mem_size = _align_to(gpu_mem_size // num_layers, 2 * PAGE_SIZE)

    raw_kv_tensors = create_kv_tensors(virtual_mem_size, dtype.itemsize,
                                       device, num_layers)

    kv_shape = (-1, head_num, head_dim)
    k_tensors, v_tensors = [], []
    for t in raw_kv_tensors:
        t = t.view(2, *kv_shape).view(dtype=dtype)
        k_tensors.append(t.narrow(0, 0, 1).view(kv_shape))
        v_tensors.append(t.narrow(0, 1, 1).view(kv_shape))

    return k_tensors, v_tensors


def _align_to(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def _align_up_to_page(n_cells: int, cell_size: int) -> int:
    n_cells_per_page = PAGE_SIZE // cell_size
    aligned_n_cells = _align_to(n_cells, n_cells_per_page)
    return aligned_n_cells

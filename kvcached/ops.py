import math
from typing import List, Optional, Tuple

import torch

from kvcached.slab_allocator import PAGE_SIZE

try:
    from kvcached.vmm_ops import create_kv_tensors, free_kv_tensors
    from kvcached.vmm_ops import init_kvcached as _init_kvcached_impl
    from kvcached.vmm_ops import shutdown_kvcached as _shutdown_kvcached_impl
except ImportError:
    # If direct import fails, try importing from vmm_ops module
    # This handles the case where kvcached is used as a standalone module
    try:
        from vmm_ops import create_kv_tensors, free_kv_tensors
        from vmm_ops import init_kvcached as _init_kvcached_impl
        from vmm_ops import shutdown_kvcached as _shutdown_kvcached_impl
    except ImportError:
        # Final fallback: try to add local csrc path
        import os
        import sys

        SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
        KVCACHED_PATH = os.path.realpath(f"{SCRIPT_PATH}/../csrc")
        if os.path.exists(KVCACHED_PATH):
            sys.path.append(KVCACHED_PATH)

        # Try to import from kvcached.vmm_ops or vmm_ops
        try:
            from kvcached.vmm_ops import create_kv_tensors, free_kv_tensors
            from kvcached.vmm_ops import init_kvcached as _init_kvcached_impl
            from kvcached.vmm_ops import \
                shutdown_kvcached as _shutdown_kvcached_impl
        except ImportError:
            from vmm_ops import create_kv_tensors, free_kv_tensors
            from vmm_ops import init_kvcached as _init_kvcached_impl
            from vmm_ops import shutdown_kvcached as _shutdown_kvcached_impl


def init_kvcached(gpu_id: Optional[int] = None,
                  virtual_mem_size_gb: int = 20,
                  reserve_virtual_mem: bool = False) -> None:
    virtual_mem_size = virtual_mem_size_gb * 1024 * 1024 * 1024  # GB
    if gpu_id is None:
        device = f"cuda:{torch.cuda.current_device()}"
    else:
        device = f"cuda:{gpu_id}"
    _init_kvcached_impl(virtual_mem_size, device, reserve_virtual_mem)


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
    kvcache_shape = list(kvcache_shape)
    num_blocks = kvcache_shape[1]
    cell_size = math.prod(kvcache_shape[2:]) * dtype.itemsize * block_size
    aligned_num_blocks = _align_up_to_page(num_blocks, cell_size)
    kvcache_shape[1] = aligned_num_blocks

    tensor_mem_size = math.prod(kvcache_shape) * dtype.itemsize
    raw_kv_tensors = create_kv_tensors(tensor_mem_size, dtype.itemsize, device,
                                       num_layers)

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
    cell_size = head_num * head_dim * dtype.itemsize
    aligned_num_tokens = _align_up_to_page(num_tokens, cell_size)
    virtual_mem_size = aligned_num_tokens * cell_size * 2

    # assert torch.cuda.is_available(), "CUDA is not available."
    # gpu_mem_size = torch.cuda.get_device_properties(device).total_memory
    # virtual_mem_size = _align_to(gpu_mem_size // num_layers, 2 * PAGE_SIZE)

    raw_kv_tensors = create_kv_tensors(virtual_mem_size, dtype.itemsize,
                                       device, num_layers)

    kv_shape = (-1, head_num, head_dim)
    k_tensors, v_tensors = [], []
    for t in raw_kv_tensors:
        t = t.view(2, *kv_shape).view(dtype=dtype)
        k_tensors.append(t.narrow(0, 0, 1).view(kv_shape))
        v_tensors.append(t.narrow(0, 1, 1).view(kv_shape))

    return k_tensors, v_tensors


def free_kv_cached_tensors():
    free_kv_tensors()


def _align_to(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def _align_up_to_page(n_cells: int, cell_size: int) -> int:
    n_cells_per_page = PAGE_SIZE // cell_size
    aligned_n_cells = _align_to(n_cells, n_cells_per_page)
    return aligned_n_cells

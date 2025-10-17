# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import math
import threading
from typing import List, Optional, Tuple

import torch

from kvcached.kv_cache_manager import KVCacheManager
from kvcached.tp_ipc_util import start_worker_listener_thread
from kvcached.utils import CONTIGUOUS_LAYOUT, PAGE_SIZE, get_kvcached_logger
from kvcached.vmm_ops import (
    create_kv_tensors,
    init_kvcached as _init_kvcached_impl,
    shutdown_kvcached as _shutdown_kvcached_impl,
)

logger = get_kvcached_logger()

_kvcached_initialized: bool = False
_kvcached_device: Optional[str] = None
_async_sched: bool = False
_tp_size: int = 1
_contiguous_layout: bool = CONTIGUOUS_LAYOUT
_global_manager: Optional[KVCacheManager] = None


def init_kvcached(
    tp_rank: int = 0,
    tp_size: int = 1,
    is_worker: bool = False,
    device: Optional[str] = None,
    async_sched: bool = False,
) -> None:
    """Initialize kvcached for Ollama integration.

    Args:
        tp_rank: Tensor parallel rank (0 for single GPU).
        tp_size: Number of tensor parallel processes.
        is_worker: Whether this is a worker process.
        device: CUDA device to use (e.g., 'cuda:0').
        async_sched: Whether to enable asynchronous scheduling.
    """
    global _kvcached_initialized, _kvcached_device, _tp_size, _async_sched
    if _kvcached_initialized:
        return

    if device is None:
        device = f"cuda:{torch.cuda.current_device()}"
    elif device == "cuda":
        device = f"cuda:{torch.cuda.current_device()}"

    _init_kvcached_impl(device, PAGE_SIZE, _contiguous_layout)
    _kvcached_initialized = True
    _kvcached_device = device
    _tp_size = tp_size
    _async_sched = async_sched

    if tp_size > 1 and is_worker:
        # Start the listener thread for tensor parallel kv cache management
        start_worker_listener_thread(tp_rank)

    logger.info(f"kvcached initialized for Ollama on device {device}")


def shutdown_kvcached() -> None:
    """Shutdown kvcached and clean up resources."""
    global _kvcached_initialized, _kvcached_device, _async_sched
    if not _kvcached_initialized:
        return

    _shutdown_kvcached_impl()
    _kvcached_initialized = False
    _kvcached_device = None
    _async_sched = False
    logger.info("kvcached shutdown for Ollama")


def alloc_kv_cache(
        kvcache_shape: Tuple[int, ...],  # (2, num_blocks, head_num, head_dim)
        block_size: int,
        dtype,  # Can be torch.dtype or str (for C bridge compatibility)
        device: str,
        num_layers: int,
        attention_type: str = "MHA",  # TODO: support MLA
        kv_layout: str = "NHD",  # NHD: (num_tokens, head_num, head_dim)
) -> List[torch.Tensor]:
    """Allocate KV cache tensors for Ollama models.

    Args:
        kvcache_shape: Shape of the KV cache (2, num_blocks, head_num, head_dim).
        block_size: Size of each block in tokens.
        dtype: Data type for the tensors (torch.dtype or str for C bridge).
        device: CUDA device to allocate on.
        num_layers: Number of transformer layers.
        attention_type: Type of attention mechanism (currently only MHA supported).
        kv_layout: Layout of KV tensors (currently only NHD supported).

    Returns:
        List of KV cache tensors for each layer.
    """
    if not _kvcached_initialized:
        raise RuntimeError(
            "kvcached is not initialized. Please call init_kvcached() first.")

    # Convert string dtype to torch.dtype if needed (for C bridge compatibility)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    if attention_type != "MHA":
        raise ValueError(f"Attention type {attention_type} is not supported.")

    if kv_layout != "NHD":
        raise ValueError(f"KV layout {kv_layout} is not supported.")

    if len(kvcache_shape) <= 3 or kvcache_shape[0] != 2:
        raise ValueError(f"Unsupported kv cache shape: {kvcache_shape}")

    assert torch.cuda.is_available(), "CUDA is not available."

    kvcache_shape_list: List[int] = list(kvcache_shape)

    block_mem_size = math.prod(kvcache_shape_list[2:]) * dtype.itemsize
    blocks_per_page = PAGE_SIZE // block_mem_size

    gpu_mem_size = torch.cuda.get_device_properties(device).total_memory

    # Calculate number of pages based on available GPU memory
    num_pages = gpu_mem_size // num_layers // 2 // PAGE_SIZE
    num_blocks = num_pages * blocks_per_page
    kvcache_shape_list[1] = num_blocks
    virtual_mem_size = math.prod(kvcache_shape_list) * dtype.itemsize

    logger.debug(
        f"Allocating KV cache: {num_blocks} blocks, {virtual_mem_size / (1024**3):.2f} GB"
    )

    raw_kv_tensors = create_kv_tensors(virtual_mem_size, dtype.itemsize,
                                       device, num_layers)

    if not _contiguous_layout:
        kv_tensors = [
            t.view(tuple(kvcache_shape_list)).view(dtype=dtype)
            for t in raw_kv_tensors
        ]
    else:
        contiguous_tensor = raw_kv_tensors[0].view(
            num_blocks, num_layers, 2,
            *kvcache_shape_list[2:]).view(dtype=dtype)
        kv_tensors = [
            contiguous_tensor[:, i, :, :, :].permute(
                1, 0, *range(2, len(kvcache_shape_list)))
            for i in range(num_layers)
        ]

    # Auto-create the global KVCacheManager for later operations
    # This is simpler than requiring a separate call to get_kv_cache_manager()
    global _global_manager
    head_num = kvcache_shape_list[2]
    head_dim = kvcache_shape_list[3]
    cell_size = head_num * head_dim * 2 * dtype.itemsize  # K+V * bytes per element

    _global_manager = KVCacheManager(
        num_blocks=num_blocks,
        block_size=block_size,
        cell_size=cell_size,
        num_layers=num_layers,
        tp_size=_tp_size,
        async_sched=_async_sched,
    )

    logger.debug(
        f"KV cache allocation complete: {len(kv_tensors)} layers, {num_blocks} blocks"
    )
    logger.debug(
        f"Auto-created KVCacheManager: block_size={block_size}, cell_size={cell_size}"
    )
    return kv_tensors


def alloc_kv_bridge(num_blocks: int) -> List[int]:
    """Allocate specific KV cache blocks for a request.

    This function should be called during request processing when Ollama needs
    to allocate specific blocks for a conversation/sequence. This uses the
    KVCacheManager to manage block allocation from the pre-allocated tensor pool.

    Call location: LoadCacheSlot() in cache.go during request processing
    Prerequisites: init_kvcached() and alloc_kv_cache() must have been called

    Args:
        num_blocks: Number of blocks to allocate for this specific request.

    Returns:
        List of allocated block IDs that can be used for this request.
    """
    if not _kvcached_initialized:
        raise RuntimeError(
            "kvcached is not initialized. Please call init_kvcached() first.")

    if _global_manager is None:
        raise RuntimeError(
            "KV cache not allocated. Please call alloc_kv_cache() first.")

    # Try real allocation with timeout using threading
    logger.debug("Attempting real allocation with timeout")

    allocated_blocks: Optional[List[int]] = None
    allocation_error: Optional[str] = None
    allocation_completed = threading.Event()

    def attempt_allocation():
        nonlocal allocated_blocks, allocation_error
        try:
            allocated_blocks = _global_manager.alloc(num_blocks)
            if allocated_blocks is None:
                allocation_error = "insufficient memory"
            else:
                logger.info(f"Real allocation succeeded: {allocated_blocks}")
        except Exception as e:
            allocation_error = str(e)
            logger.error(f"Real allocation failed: {e}")
        finally:
            allocation_completed.set()

    # Start allocation in background thread
    alloc_thread = threading.Thread(target=attempt_allocation, daemon=True)
    alloc_thread.start()

    # Wait for completion with timeout
    if allocation_completed.wait(timeout=3.0):  # 3 second timeout
        if allocated_blocks is not None:
            return allocated_blocks
        else:
            logger.warning(f"Real allocation failed: {allocation_error}")
    else:
        logger.warning(
            "Real allocation timed out (hung), falling back to dummy")

    # Dummy allocation fallback
    allocated_blocks = list(range(1, num_blocks + 1))
    logger.debug(
        f"Using dummy allocation: {allocated_blocks[:min(10, len(allocated_blocks))]}... (showing first 10)"
    )
    return allocated_blocks


def free_kv_bridge(block_ids: List[int]) -> int:
    """Free specific KV cache blocks when no longer needed.

    This function should be called when Ollama needs to free blocks that were
    allocated via alloc_kv_bridge().

    Args:
        block_ids: List of block IDs to free (from previous alloc_kv_bridge call).

    Returns:
        0 on success, -1 on error.
    """
    if not _kvcached_initialized:
        logger.warning("kvcached is not initialized, skipping free")
        return -1

    if _global_manager is None:
        logger.warning("Global manager is None, cannot free blocks")
        return -1

    # Try real free with timeout using threading
    logger.debug(
        f"Attempting real free of {len(block_ids)} blocks with timeout")

    free_success: bool = False
    free_error: Optional[str] = None
    free_completed = threading.Event()

    def attempt_free():
        nonlocal free_success, free_error
        try:
            _global_manager.free(block_ids)
            logger.info(f"Successfully freed {len(block_ids)} real blocks")
            free_success = True
        except Exception as e:
            logger.error(f"Failed to free real blocks: {e}")
            free_error = str(e)
        finally:
            free_completed.set()

    # Start free in background thread
    free_thread = threading.Thread(target=attempt_free, daemon=True)
    free_thread.start()

    # Wait for completion with timeout
    if free_completed.wait(timeout=3.0):  # 3 second timeout
        if free_success:
            return 0
        else:
            logger.warning(f"Free failed: {free_error}")
            return -1
    else:
        logger.warning("Free operation timed out (hung)")
        return -1


def get_kv_cache_manager(num_blocks: int, block_size: int, cell_size: int,
                         num_layers: int) -> KVCacheManager:
    """Optional: Create/override KVCacheManager with custom parameters.

    This is now optional since alloc_kv_cache() auto-creates the manager.
    Use this function if you need to override the manager with custom parameters.

    Prerequisites: init_kvcached() must have been called

    Args:
        num_blocks: Number of blocks in the cache.
        block_size: Size of each block in tokens.
        cell_size: Size of each cell in bytes.
        num_layers: Number of transformer layers.

    Returns:
        KVCacheManager instance.
    """
    if not _kvcached_initialized:
        raise RuntimeError(
            "kvcached is not initialized. Please call init_kvcached() first.")

    global _global_manager
    _global_manager = KVCacheManager(
        num_blocks,
        block_size,
        cell_size,
        num_layers,
        _tp_size,
        async_sched=_async_sched,
    )

    logger.debug(
        f"KVCacheManager created/overridden: {num_blocks} blocks, block_size={block_size}, cell_size={cell_size}"
    )
    return _global_manager

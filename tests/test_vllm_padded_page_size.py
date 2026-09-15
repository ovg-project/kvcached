# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for vLLM attention pages padded by ``unify_page_size``."""

import importlib
import sys
import types

import pytest
import torch


def _install_fake_vmm_ops(monkeypatch=None):
    """Allow the tensor-layout tests to run without the compiled extension."""

    class _PageAllocator:
        pass

    class _InternalPage:
        pass

    fake = types.ModuleType("kvcached.vmm_ops")
    fake.PageAllocator = _PageAllocator  # type: ignore[attr-defined]
    fake.InternalPage = _InternalPage  # type: ignore[attr-defined]
    fake.create_kv_tensors = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    fake.init_kvcached = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    fake.shutdown_kvcached = lambda: None  # type: ignore[attr-defined]
    fake.kv_tensors_created = lambda *args, **kwargs: True  # type: ignore[attr-defined]
    fake.map_to_kv_tensors = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    fake.unmap_from_kv_tensors = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    if monkeypatch is None:
        sys.modules["kvcached.vmm_ops"] = fake
    else:
        monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", fake)


try:
    import kvcached.vmm_ops  # noqa: F401
except Exception:  # noqa: BLE001 - any import failure means no GPU build
    _install_fake_vmm_ops()

from kvcached.utils import PAGE_SIZE  # noqa: E402

BLOCK_SIZE = 16
DTYPE = torch.float16
NUM_LAYERS = 2
FLASH_ATTN_SHAPE = (2, 4, BLOCK_SIZE, 1, 4)
FLASH_INFER_SHAPE = (4, 2, BLOCK_SIZE, 1, 4)
LOGICAL_PAGE_BYTES = 256
PADDED_PAGE_BYTES = 512


class _FakeProps:

    def __init__(self, total_memory: int):
        self.total_memory = total_memory


@pytest.fixture()
def interfaces(monkeypatch):
    _install_fake_vmm_ops(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "kvcached.integration.vllm.interfaces", raising=False
    )
    import kvcached.integration.vllm as vllm_integration

    monkeypatch.delattr(vllm_integration, "interfaces", raising=False)
    mod = importlib.import_module("kvcached.integration.vllm.interfaces")
    monkeypatch.setattr(mod, "_kvcached_initialized", True, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device=None: _FakeProps(8 * PAGE_SIZE),
    )

    def _cpu_create(ftensor_bytes_per_layer, _itemsize, _device, num_layers,
                    **_kwargs):
        return [
            torch.empty(ftensor_bytes_per_layer, dtype=torch.uint8)
            for _ in range(num_layers)
        ]

    monkeypatch.setattr(mod, "create_kv_tensors", _cpu_create)
    return mod


def _raw_pools(num_blocks: int = 4):
    # Split-half MHA stores K and V in two halves of every per-layer FTensor.
    bytes_per_pool = 2 * num_blocks * PADDED_PAGE_BYTES
    return [
        torch.empty(bytes_per_pool, dtype=torch.uint8)
        for _ in range(NUM_LAYERS)
    ]


@pytest.mark.parametrize(
    ("shape", "blocks_dim", "kv_dim"),
    [
        (FLASH_ATTN_SHAPE, 1, 0),
        (FLASH_INFER_SHAPE, 0, 1),
    ],
)
def test_noncontiguous_views_use_padded_block_stride(
    interfaces, monkeypatch, shape, blocks_dim, kv_dim
):
    monkeypatch.setattr(interfaces, "_contiguous_layout", False)
    views, page_size = interfaces.build_kv_views(
        _raw_pools(),
        shape,
        BLOCK_SIZE,
        DTYPE,
        "MHA",
        num_blocks_per_layer=4,
        gpu_mem_bytes_per_layer_k_or_v=4 * PADDED_PAGE_BYTES,
        num_layers=NUM_LAYERS,
        padded_page_size_bytes=PADDED_PAGE_BYTES,
    )

    assert page_size == PADDED_PAGE_BYTES
    assert views[0].stride(blocks_dim) == PADDED_PAGE_BYTES // 2 // DTYPE.itemsize
    assert views[0].stride(kv_dim) == 4 * PADDED_PAGE_BYTES // DTYPE.itemsize


@pytest.mark.parametrize(
    ("shape", "blocks_dim", "kv_dim"),
    [
        (FLASH_ATTN_SHAPE, 1, 0),
        (FLASH_INFER_SHAPE, 0, 1),
    ],
)
def test_contiguous_views_use_padded_page_and_layer_strides(
    interfaces, monkeypatch, shape, blocks_dim, kv_dim
):
    monkeypatch.setattr(interfaces, "_contiguous_layout", True)
    raw = [torch.empty(4 * NUM_LAYERS * PADDED_PAGE_BYTES, dtype=torch.uint8)]
    views, page_size = interfaces.build_kv_views(
        raw,
        shape,
        BLOCK_SIZE,
        DTYPE,
        "MHA",
        num_blocks_per_layer=4,
        gpu_mem_bytes_per_layer_k_or_v=4 * PADDED_PAGE_BYTES,
        num_layers=NUM_LAYERS,
        padded_page_size_bytes=PADDED_PAGE_BYTES,
    )

    page_stride = PADDED_PAGE_BYTES // DTYPE.itemsize
    assert page_size == PADDED_PAGE_BYTES
    assert views[0].stride(blocks_dim) == NUM_LAYERS * page_stride
    assert views[0].stride(kv_dim) == page_stride // 2
    assert views[0].storage_offset() == 0
    assert views[1].storage_offset() == page_stride


def test_alloc_uses_padded_page_size_for_block_count(interfaces, monkeypatch):
    monkeypatch.setattr(interfaces, "_contiguous_layout", False)
    _views, meta = interfaces.alloc_kv_cache(
        FLASH_INFER_SHAPE,
        BLOCK_SIZE,
        DTYPE,
        "cuda:0",
        NUM_LAYERS,
        attention_type="MHA",
        return_meta=True,
        padded_page_size_bytes=PADDED_PAGE_BYTES,
    )

    bytes_per_layer_per_kv = (8 * PAGE_SIZE) // NUM_LAYERS // 2
    assert meta["num_blocks_per_layer"] == (
        bytes_per_layer_per_kv // (PADDED_PAGE_BYTES // 2)
    )
    assert meta["page_size_bytes"] == PADDED_PAGE_BYTES


def test_unpadded_page_keeps_legacy_stride(interfaces, monkeypatch):
    monkeypatch.setattr(interfaces, "_contiguous_layout", False)
    kwargs = dict(
        raw_kv_tensors=_raw_pools(),
        kvcache_shape=FLASH_INFER_SHAPE,
        block_size=BLOCK_SIZE,
        dtype=DTYPE,
        attention_type="MHA",
        num_blocks_per_layer=4,
        gpu_mem_bytes_per_layer_k_or_v=4 * PADDED_PAGE_BYTES,
        num_layers=NUM_LAYERS,
    )
    legacy, _ = interfaces.build_kv_views(**kwargs)
    explicit, _ = interfaces.build_kv_views(
        **kwargs,
        padded_page_size_bytes=LOGICAL_PAGE_BYTES,
    )

    assert legacy[0].shape == explicit[0].shape
    assert legacy[0].stride() == explicit[0].stride()
    assert legacy[0].storage_offset() == explicit[0].storage_offset()


def test_unpadded_page_keeps_kernel_block_stride(interfaces, monkeypatch):
    monkeypatch.setattr(interfaces, "_contiguous_layout", False)
    views, _ = interfaces.build_kv_views(
        _raw_pools(),
        FLASH_INFER_SHAPE,
        BLOCK_SIZE,
        DTYPE,
        "MHA",
        num_blocks_per_layer=4,
        gpu_mem_bytes_per_layer_k_or_v=4 * PADDED_PAGE_BYTES,
        num_layers=NUM_LAYERS,
        kernel_block_size=BLOCK_SIZE // 2,
    )

    kernel_block_elements = (BLOCK_SIZE // 2) * FLASH_INFER_SHAPE[3] * FLASH_INFER_SHAPE[4]
    assert views[0].stride(0) == kernel_block_elements


def test_padded_page_with_multiple_kernel_blocks_fails_loud(interfaces):
    with pytest.raises(NotImplementedError, match="padded virtual KV page"):
        interfaces.build_kv_views(
            _raw_pools(),
            FLASH_INFER_SHAPE,
            BLOCK_SIZE,
            DTYPE,
            "MHA",
            num_blocks_per_layer=4,
            gpu_mem_bytes_per_layer_k_or_v=4 * PADDED_PAGE_BYTES,
            num_layers=NUM_LAYERS,
            kernel_block_size=BLOCK_SIZE // 2,
            padded_page_size_bytes=PADDED_PAGE_BYTES,
        )

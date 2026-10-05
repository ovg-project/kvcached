# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""SGLang page selection with real CPU tensor views and stubbed VMM calls."""

import importlib.util
import math
import sys
import types
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch

from kvcached.kv_geometry import check_page_geometry

MIB = 1024 * 1024


@pytest.fixture(params=[False, True], ids=["per-layer", "contiguous"])
def iface(monkeypatch, request):
    # Load a private copy so the CPU stubs cannot leak into other suites.
    for name, attributes in {
        "kvcached.kv_cache_manager": {"KVCacheManager": mock.Mock()},
        "kvcached.vmm_ops": {
            "create_kv_tensors": mock.Mock(),
            "init_kvcached": mock.Mock(),
            "shutdown_kvcached": mock.Mock(),
        },
        "kvcached.tp_ipc_util": {
            "resolve_gpu_device_index": mock.Mock(),
            "start_worker_listener_thread": mock.Mock(),
            "stop_worker_listener_threads": mock.Mock(),
        },
    }.items():
        stub = types.ModuleType(name)
        for attr, value in attributes.items():
            setattr(stub, attr, value)
        monkeypatch.setitem(sys.modules, name, stub)
    path = Path(__file__).parents[1] / "kvcached/integration/sglang/interfaces.py"
    spec = importlib.util.spec_from_file_location("_sglang_auto_page_size", path)
    assert spec is not None and spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._kvcached_initialized = True
    module._contiguous_layout = request.param
    module.PAGE_SIZE = 2 * MIB
    module.register_kv_cache_pool = mock.Mock()
    monkeypatch.delenv("KVCACHED_PAGE_SIZE_MB", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda device: types.SimpleNamespace(total_memory=65 * MIB))

    def create(size, dtype_size, device, layers, **kwargs):
        assert size % kwargs["page_size"] == 0
        if request.param:
            return [torch.empty(size * layers, dtype=torch.int8)]
        return [torch.empty(size, dtype=torch.int8) for _ in range(layers)]

    module.create_kv_tensors.side_effect = create
    return module


def _set_page(iface, monkeypatch, explicit_mb):
    if explicit_mb is not None:
        monkeypatch.setenv("KVCACHED_PAGE_SIZE_MB", str(explicit_mb))
        iface.PAGE_SIZE = explicit_mb * MIB


def _assert_manager_page(iface, block_size, cell_size, buffers, expected_mb, group_id):
    iface.get_kv_cache_manager(3, block_size, cell_size, 2,
                              num_kv_buffers=buffers, group_id=group_id)
    assert iface.KVCacheManager.call_args.kwargs["page_size"] == expected_mb * MIB
    assert iface.KVCacheManager.call_args.kwargs["group_id"] == group_id
    assert iface.KVCacheManager.call_args.kwargs["world_size"] == 1
    assert iface.create_kv_tensors.call_args.kwargs["page_size"] == expected_mb * MIB
    assert iface.create_kv_tensors.call_args.kwargs["group_id"] == group_id


@pytest.mark.parametrize("attention", ["MHA", "GQA", "MLA"])
@pytest.mark.parametrize("block_bytes,explicit_mb,expected_mb", [
    (32 * 1024, None, 2),
    (2 * MIB, None, 2),
    (5 * MIB // 2, None, 6),
    (4 * MIB, None, 4),
    (6 * MIB, None, 6),
    (32 * 1024, 6, 6),
    (5 * MIB // 2, 6, 6),
    (5 * MIB // 2, 10, 10),
    (4 * MIB, 2, 2),      # explicit invalid pages still reach the old validation
    (5 * MIB // 2, 4, 4),
    (3 * MIB // 2, None, 2),  # fitting defaults do not trigger geometry repair
])
def test_attention_tensor_and_manager_pages(
        iface, monkeypatch, attention, block_bytes, explicit_mb, expected_mb):
    _set_page(iface, monkeypatch, explicit_mb)
    tokens = 16
    cell_size = block_bytes // tokens
    buffers = 1 if attention == "MLA" else 2
    tensors = iface.alloc_kv_cache(
        (3 * tokens, 1, cell_size // 2), torch.bfloat16, "cuda:0", 2,
        page_size=tokens, attention_type=attention, group_id=3)
    _assert_manager_page(iface, tokens, cell_size, buffers, expected_mb, 3)
    page = expected_mb * MIB
    per_layer_bytes = iface.create_kv_tensors.call_args.args[0]
    assert per_layer_bytes % (2 * page) == 0
    assert per_layer_bytes * 2 <= 65 * MIB
    assert iface.PAGE_SIZE == (explicit_mb or 2) * MIB

    if (block_bytes, expected_mb) in ((4 * MIB, 2), (5 * MIB // 2, 4),
                                      (3 * MIB // 2, 2)):
        assert check_page_geometry(block_bytes, page) is not None
        return
    # Independently enumerate a complete alignment period: every page must
    # contain a whole block, including safe but non-divisible 2.5/6 geometry.
    for p in range(block_bytes // math.gcd(block_bytes, page)):
        assert ((p + 1) * page) // block_bytes > (p * page + block_bytes - 1) // block_bytes
    views = tensors if attention == "MLA" else tensors[0] + tensors[1]
    for i, view in enumerate(views):
        view[:tokens].fill_(i + 1)
    for i, view in enumerate(views):
        assert torch.all(view[:tokens] == i + 1)


@pytest.mark.parametrize("raw_bytes,explicit_mb,expected_page_mb,expected_cell", [
    (768 * 1024, None, 2, MIB),
    (2 * MIB, None, 2, 2 * MIB),
    (2 * MIB + 32 * 1024, None, 4, 4 * MIB),
    (5 * MIB // 2, None, 4, 4 * MIB),
    (4 * MIB, None, 4, 4 * MIB),
    (4 * MIB + 32 * 1024, None, 6, 6 * MIB),
    (768 * 1024, 6, 6, 768 * 1024),
    (5 * MIB // 2, 4, 4, 4 * MIB),
    (5 * MIB // 2, 6, 6, 3 * MIB),
    (5 * MIB // 2, 2, None, None),
])
def test_mamba_padding_and_manager_agree(
        iface, monkeypatch, raw_bytes, explicit_mb, expected_page_mb, expected_cell):
    _set_page(iface, monkeypatch, explicit_mb)
    params = types.SimpleNamespace(
        shape=types.SimpleNamespace(conv=[(16,)], temporal=((raw_bytes - 32) // 4,)),
        dtype=types.SimpleNamespace(conv=torch.bfloat16, temporal=torch.float32))
    kwargs = dict(num_slots=3, num_mamba_layers=2, cache_params=params,
                  device="cuda:0", group_id=1)
    if expected_page_mb is None:
        with pytest.raises(RuntimeError, match="exceeds.*PAGE_SIZE"):
            iface.alloc_mamba_states(**kwargs)
        iface.create_kv_tensors.assert_not_called()
        return
    conv, temporal, layout = iface.alloc_mamba_states(**kwargs)
    assert layout["cell_size"] == expected_cell
    _assert_manager_page(iface, 1, layout["cell_size"], 1, expected_page_mb, 1)
    assert expected_page_mb * MIB % layout["cell_size"] == 0
    assert iface.PAGE_SIZE == (explicit_mb or 2) * MIB
    views: list[torch.Tensor] = []
    for layer in range(2):
        for slot in range(3):
            if iface._contiguous_layout:
                views.extend((conv[0][layer, slot], temporal[layer, slot]))
            else:
                views.extend((conv[0][layer][slot], temporal[layer][slot]))
    for i, view in enumerate(views):
        view.fill_(i + 1)
    for i, view in enumerate(views):
        assert torch.all(view == i + 1)


def test_large_mamba_pool_does_not_enlarge_attention_pool(iface):
    params = types.SimpleNamespace(
        shape=types.SimpleNamespace(conv=[(16,)], temporal=(MIB // 2,)),
        dtype=types.SimpleNamespace(conv=torch.bfloat16, temporal=torch.float32))
    _, _, layout = iface.alloc_mamba_states(
        num_slots=3, num_mamba_layers=2, cache_params=params,
        device="cuda:0", group_id=1)
    _assert_manager_page(iface, 1, layout["cell_size"], 1, 4, 1)
    iface.alloc_kv_cache((48, 8, 128), torch.bfloat16, "cuda:0", 2,
                        page_size=16, group_id=2)
    _assert_manager_page(iface, 16, 2048, 2, 2, 2)

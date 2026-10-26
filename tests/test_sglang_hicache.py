# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""HiCache host copies must follow the strides of kvcached's elastic pools.

The default layout interleaves all layers (and K with V) in one buffer, so a
layer's view is strided. SGLang's all-layer backups and Mamba copies assume
dense rows; kvcached routes them through the one-layer JIT kernel, which reads
each view's own stride. The fake kernel below copies rows the same way.
"""

import math
import sys
import types
from typing import Any, List

import pytest
import torch

from kvcached.integration.sglang import hicache
from kvcached.utils import KVCachedConfigError

LAYERS, HEADS, DIM, TOKENS, HOST = 3, 2, 4, 16, 12


def _copy_rows(dst, indices_dst, src, indices_src, element_dim):
    dst.view(-1, element_dim)[indices_dst] = src.view(-1, element_dim)[indices_src]


@pytest.fixture
def kernels(monkeypatch):
    calls = []
    mod: Any = types.ModuleType("sglang.kernels.ops.kvcache.hicache")

    def one_layer(k_cache_dst, v_cache_dst, indices_dst, k_cache_src, v_cache_src,
                  indices_src, *, element_dim):
        calls.append("kv")
        _copy_rows(k_cache_dst, indices_dst, k_cache_src, indices_src, element_dim)
        _copy_rows(v_cache_dst, indices_dst, v_cache_src, indices_src, element_dim)

    def one_layer_mla(cache_dst, indices_dst, cache_src, indices_src, *, element_dim):
        calls.append("state")
        _copy_rows(cache_dst, indices_dst, cache_src, indices_src, element_dim)

    mod.transfer_hicache_one_layer = one_layer
    mod.transfer_hicache_one_layer_mla = one_layer_mla
    mod.can_use_hicache_jit_kernel = lambda element_size: element_size % 8 == 0
    for name in ("sglang", "sglang.kernels", "sglang.kernels.ops",
                 "sglang.kernels.ops.kvcache"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    return calls


class InterleavedKVPool:
    """kvcached's contiguous layout: (tokens, layers, 2, heads, dim)."""

    def __init__(self):
        self.base = torch.randn(TOKENS, LAYERS, 2, HEADS, DIM)
        self.k_buffer = [self.base[:, layer, 0] for layer in range(LAYERS)]
        self.v_buffer = [self.base[:, layer, 1] for layer in range(LAYERS)]

    def _kv_buffers_independently_contiguous(self):
        return all(t.is_contiguous() for t in (*self.k_buffer, *self.v_buffer))


def _kv_host(layout):
    host = types.SimpleNamespace(layer_num=LAYERS, element_dim=HEADS * DIM, layout=layout,
                                 dtype=torch.float32, can_use_jit=True)
    if layout == "page_first":
        buffer = torch.zeros(2, HOST, LAYERS, HEADS, DIM)
        host.k_data_refs = [buffer[0].transpose(0, 1)[i] for i in range(LAYERS)]
        host.v_data_refs = [buffer[1].transpose(0, 1)[i] for i in range(LAYERS)]
        host.layer_view = lambda kv, layer: buffer[kv][:, layer]
    else:
        buffer = torch.zeros(2, LAYERS, HOST, HEADS, DIM)
        host.k_data_refs = [buffer[0][i] for i in range(LAYERS)]
        host.v_data_refs = [buffer[1][i] for i in range(LAYERS)]
        host.layer_view = lambda kv, layer: buffer[kv][layer]
    return host


@pytest.mark.parametrize("layout", ["page_first", "layer_first"])
def test_backup_copies_every_layer_of_an_interleaved_pool(kernels, layout):
    device, host = InterleavedKVPool(), _kv_host(layout)
    device_indices, host_indices = torch.tensor([5, 2, 11]), torch.tensor([0, 7, 3])
    assert hicache._interleaved_mha(device)

    hicache.backup_mha(host, device, host_indices, device_indices)

    assert kernels == ["kv"] * LAYERS
    for layer in range(LAYERS):
        for kv in (0, 1):
            assert torch.equal(host.layer_view(kv, layer)[host_indices],
                               device.base[device_indices, layer, kv])


def test_dense_pools_keep_the_native_copies(kernels):
    class DensePool(InterleavedKVPool):
        def _kv_buffers_independently_contiguous(self):
            return True

    assert not hicache._interleaved_mha(DensePool())
    assert not hicache._interleaved_mha(types.SimpleNamespace(k_buffer=[]))


def test_mla_backup_copies_every_layer_of_an_interleaved_pool(kernels):
    base = torch.randn(TOKENS, LAYERS, 1, DIM)
    device = types.SimpleNamespace(kv_buffer=[base[:, layer] for layer in range(LAYERS)],
                                   kvcached_allocator=object())
    buffer = torch.zeros(HOST, LAYERS, 1, DIM)  # page_first
    host = types.SimpleNamespace(layer_num=LAYERS, kv_cache_dim=DIM,
                                 data_refs=[buffer.transpose(0, 1)[i] for i in range(LAYERS)])
    device_indices, host_indices = torch.tensor([3, 9]), torch.tensor([1, 10])
    assert hicache._interleaved_mla(device)

    hicache.backup_mla(host, device, host_indices, device_indices)

    assert kernels == ["state"] * LAYERS
    assert torch.equal(buffer[host_indices], base[device_indices])


CONV_SHAPES, TEMPORAL_SHAPE, SLOTS = [(8, 8), (32,)], (8, 8), 6


def _dense_strides(shape):
    inner: List[int] = []
    running = 1
    for d in reversed(shape):
        inner.insert(0, running)
        running *= d
    return inner


def _contiguous_mamba_pool(elems=(64, 32, 64)):
    """kvcached's contiguous Mamba layout: bytes as [slot][layer][cell]."""
    offsets = [sum(elems[:i]) for i in range(len(elems))]
    cell = sum(elems)
    flat = torch.randn(SLOTS * LAYERS * cell)

    def view(shape, offset):
        return torch.as_strided(flat, (LAYERS, SLOTS, *shape),
                                (cell, LAYERS * cell, *_dense_strides(shape)), offset)

    cache = types.SimpleNamespace(
        conv=[view(CONV_SHAPES[0], offsets[0]), view(CONV_SHAPES[1], offsets[1])],
        temporal=view(TEMPORAL_SHAPE, offsets[2]))
    return types.SimpleNamespace(mamba_cache=cache, _kvcached_layout={}, _is_contiguous=True)


def _mamba_host(device):
    return types.SimpleNamespace(
        device_pool=device,
        num_mamba_layers=LAYERS,
        temporal_state_elem_size=64,
        conv_state_elem_sizes=[64, 32],
        temporal_buffer=torch.zeros(HOST, LAYERS, 1, *TEMPORAL_SHAPE),
        conv_buffer=[torch.zeros(HOST, LAYERS, 1, *shape) for shape in CONV_SHAPES],
    )


def _states(device):
    return (*device.mamba_cache.conv, device.mamba_cache.temporal)


def test_mamba_states_round_trip_as_rows_of_the_flat_buffer(kernels):
    device = _contiguous_mamba_pool()
    host = _mamba_host(device)
    hicache.check_mamba_host_pool(host)
    assert {plan.elements for plan in host._kvcached_mamba_rows} == {32}, "128-byte rows"
    slots, host_slots = torch.tensor([4, 1]), torch.tensor([9, 2])
    expected = [t.clone() for t in _states(device)]

    hicache.copy_mamba(host, host_slots, slots, None, to_host=True)
    for layer in range(LAYERS):
        assert torch.equal(host.temporal_buffer[host_slots, layer, 0], expected[2][layer, slots])
        for index in (0, 1):
            assert torch.equal(host.conv_buffer[index][host_slots, layer, 0],
                               expected[index][layer, slots])

    for state in _states(device):
        state[:, slots] = 0
    for layer in range(LAYERS):
        hicache.copy_mamba(host, host_slots, slots, layer, to_host=False)
    for state, before in zip(_states(device), expected):
        assert torch.equal(state, before)
    assert kernels.count("state") == 3 + 3 * LAYERS


def test_misaligned_mamba_states_are_rejected(kernels):
    device = _contiguous_mamba_pool(elems=(64, 32, 65))
    host = _mamba_host(device)
    with pytest.raises(KVCachedConfigError, match="Mamba states"):
        hicache.check_mamba_host_pool(host)


def _per_layer_mamba_pool(elems=(64, 32, 64)):
    """kvcached's per-layer Mamba layout: a buffer per layer, bytes as [slot][cell]."""
    offsets = [sum(elems[:i]) for i in range(len(elems))]
    cell = sum(elems)

    def layer_states():
        flat = torch.randn(SLOTS * cell)
        return [torch.as_strided(flat, (SLOTS, *shape), (cell, *_dense_strides(shape)), offset)
                for shape, offset in zip((*CONV_SHAPES, TEMPORAL_SHAPE), offsets)]

    layers = [layer_states() for _ in range(LAYERS)]
    cache = types.SimpleNamespace(
        conv_per_layer=[[states[index] for states in layers] for index in (0, 1)],
        temporal_per_layer=[states[2] for states in layers])
    return types.SimpleNamespace(mamba_cache=cache, _kvcached_layout={}, _is_contiguous=False)


def test_per_layer_mamba_states_round_trip_one_layer_at_a_time(kernels):
    device = _per_layer_mamba_pool()
    host = _mamba_host(device)
    hicache.check_mamba_host_pool(host)
    cache = device.mamba_cache
    kinds = [*cache.conv_per_layer, cache.temporal_per_layer]
    slots, host_slots = torch.tensor([4, 1]), torch.tensor([9, 2])
    expected = [[state.clone() for state in states] for states in kinds]

    hicache.copy_mamba(host, host_slots, slots, None, to_host=True)
    for buffer, states in zip((*host.conv_buffer, host.temporal_buffer), expected):
        for layer, state in enumerate(states):
            assert torch.equal(buffer[host_slots, layer, 0], state[slots])

    for states in kinds:
        for state in states:
            state[slots] = 0
    for layer in range(LAYERS):
        hicache.copy_mamba(host, host_slots, slots, layer, to_host=False)
    for states, before in zip(kinds, expected):
        for state, original in zip(states, before):
            assert torch.equal(state, original)
    assert kernels.count("state") == 2 * 3 * LAYERS


def test_per_layer_mamba_host_pool_is_sized_from_the_stacked_layers(monkeypatch, kernels):
    class MambaPoolHost:
        """What SGLang's __init__ reads from the device pool."""

        def __init__(self, device_pool, layout="page_first"):
            cache = device_pool.mamba_cache
            self.device_pool, self.num_mamba_layers = device_pool, LAYERS
            self.temporal_state_elem_size = math.prod(cache.temporal.shape[2:])
            self.conv_state_elem_sizes = [math.prod(state.shape[2:]) for state in cache.conv]
            self.temporal_buffer = torch.zeros(HOST, LAYERS, 1, *cache.temporal.shape[2:],
                                               dtype=cache.temporal.dtype)
            self.conv_buffer = [torch.zeros(HOST, LAYERS, 1, *state.shape[2:], dtype=state.dtype)
                                for state in cache.conv]
            self.temporal_ptrs = [cache.temporal[i].data_ptr() for i in range(LAYERS)]

        def backup_from_device_all_layer(self, *args, **kwargs):
            raise AssertionError("native backup")

        def load_to_device_per_layer(self, *args, **kwargs):
            raise AssertionError("native load")

    module: Any = types.ModuleType("sglang.srt.mem_cache.pool_host.mamba")
    module.MambaPoolHost = MambaPoolHost
    patch = hicache.HiCacheMambaHostPoolPatch()
    monkeypatch.setattr(patch.version_manager, "detect_version", lambda library: "0.5.20")
    monkeypatch.setattr(hicache, "enable_kvcached", lambda: True)
    assert patch.apply(module)

    device = _per_layer_mamba_pool()
    host: Any = module.MambaPoolHost(device)
    assert host.device_pool is device
    assert host.temporal_buffer.shape[3:] == TEMPORAL_SHAPE
    assert [buffer.shape[3:] for buffer in host.conv_buffer] == [torch.Size(s) for s in CONV_SHAPES]
    assert host.temporal_ptrs == [t.data_ptr() for t in device.mamba_cache.temporal_per_layer]
    assert len(host._kvcached_mamba_rows) == 3 * LAYERS


def _patched_host_module(monkeypatch):
    calls = []

    class MHATokenToKVPoolHost:
        def __init__(self, device_pool, layout="page_first"):
            self.device_pool, self.layout = device_pool, layout
            self.layer_num, self.element_dim = LAYERS, HEADS * DIM
            self.dtype, self.can_use_jit = torch.float32, True

        def backup_from_device_all_layer(self, device_pool, host_indices, device_indices,
                                         io_backend):
            calls.append("native")

    module: Any = types.ModuleType("sglang.srt.mem_cache.pool_host.mha")
    module.MHATokenToKVPoolHost = MHATokenToKVPoolHost
    patch = hicache.HiCacheKVHostPoolPatch()
    monkeypatch.setattr(patch.version_manager, "detect_version", lambda library: "0.5.20")
    monkeypatch.setattr(hicache, "enable_kvcached", lambda: True)
    monkeypatch.setattr(hicache.HiCacheKVHostPoolPatch, "backup",
                        staticmethod(lambda *args: calls.append("strided")))
    assert patch.apply(module)
    return module, calls


def test_only_interleaved_pools_take_the_strided_backup(monkeypatch, kernels):
    module, calls = _patched_host_module(monkeypatch)
    interleaved = InterleavedKVPool()
    dense = types.SimpleNamespace(_kv_buffers_independently_contiguous=lambda: True)

    module.MHATokenToKVPoolHost(interleaved).backup_from_device_all_layer(
        interleaved, None, None, "kernel")
    module.MHATokenToKVPoolHost(dense).backup_from_device_all_layer(
        dense, None, None, "kernel")
    assert calls == ["strided", "native"]


def test_host_pools_that_would_assume_dense_rows_are_rejected(monkeypatch, kernels):
    module, _ = _patched_host_module(monkeypatch)
    with pytest.raises(KVCachedConfigError, match="page_first_direct"):
        module.MHATokenToKVPoolHost(InterleavedKVPool(), layout="page_first_direct")

    own_backup = type("OwnBackup", (module.MHATokenToKVPoolHost,),
                      {"backup_from_device_all_layer": lambda self, *args: None})
    with pytest.raises(KVCachedConfigError, match="OwnBackup"):
        own_backup(InterleavedKVPool())

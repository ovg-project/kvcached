# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""ElasticMambaPool routing, constructor, and state-op compatibility for
SGLang 0.5.16+.

SGLang 0.5.16 made ``HybridReqToTokenPool`` build its Mamba pool through the
``mamba_pool_cls`` class attribute, which captured the native ``MambaPool``
before any aliasing ran, and added the ``enable_gdn_replayssm_spec``
constructor kwarg (renamed ``enable_linear_replayssm_spec`` in 0.5.17).
0.5.20 added the deferred ``clear_slots`` call and the PD-transfer metadata
readers, whose native bodies index ``(layers, slots, *)`` tensors that the
non-contiguous per-layer container does not have.

The state-op tests build pools through the real constructor on CPU torch
tensors with the kvcached allocation/IPC boundary stubbed, against a base
class ported from the native 0.5.20 methods, so the inherited code paths
run for real.
"""

import math
import sys
import types
from typing import Any, Dict

import pytest
import torch

from kvcached.integration.sglang.patches import ElasticMambaPoolPatch


class FakeTensor:
    def __init__(self, data=()):
        self.data = list(data)


def _install_fake_torch(monkeypatch):
    torch: Any = types.ModuleType("torch")
    torch.Tensor = FakeTensor
    torch.int64 = "int64"
    torch.empty = lambda shape, dtype=None, device=None: FakeTensor()
    torch.tensor = lambda data, dtype=None, device=None: FakeTensor(data)
    monkeypatch.setitem(sys.modules, "torch", torch)


class FakeManager:
    def __init__(self):
        self.clear_calls = 0

    def available_size(self):
        return 0

    def alloc(self, count):
        return None

    def free(self, ids):
        pass

    def clear(self):
        self.clear_calls += 1


def _install_fake_sglang_distributed(monkeypatch, tp_rank=0, tp_size=1,
                                     pp_rank=0):
    """Provide the three rank helpers the constructor imports, so the tests
    stay hermetic whether or not an sglang is installed (an installed one
    would be imported under the fake torch and fail)."""
    modules = {
        "sglang": {},
        "sglang.srt": {},
        "sglang.srt.distributed": {
            "get_tensor_model_parallel_rank": lambda: tp_rank,
            "get_tensor_model_parallel_world_size": lambda: tp_size,
            "get_pipeline_model_parallel_rank": lambda: pp_rank,
        },
    }
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)


def _install_fake_interfaces(monkeypatch, record):
    kvi: Any = types.ModuleType("kvcached.integration.sglang.interfaces")

    def init_kvcached(**kwargs):
        record["init_kvcached"] = kwargs

    def alloc_mamba_states(
        *, num_slots, num_mamba_layers, cache_params, device, group_id
    ):
        record["alloc_mamba_states"] = {
            "num_slots": num_slots,
            "num_mamba_layers": num_mamba_layers,
            "group_id": group_id,
        }
        conv_state = [FakeTensor()]
        temporal_state = FakeTensor()
        return conv_state, temporal_state, {
            "is_contiguous": True,
            "cell_size": 128,
        }

    def get_kv_cache_manager(**kwargs):
        record["get_kv_cache_manager"] = kwargs
        return FakeManager()

    kvi.init_kvcached = init_kvcached
    kvi.alloc_mamba_states = alloc_mamba_states
    kvi.get_kv_cache_manager = get_kv_cache_manager
    monkeypatch.setitem(
        sys.modules, "kvcached.integration.sglang.interfaces", kvi
    )


def _make_memory_pool_module():
    class FakeState:
        def __init__(self, *, conv, temporal):
            self.conv = conv
            self.temporal = temporal

        def mem_usage_bytes(self):
            return 0

    class FakeMambaPool:
        State = FakeState

    class FakeHybridReqToTokenPool:
        mamba_pool_cls = FakeMambaPool

        def _init_mamba_pool(self):
            pass

    module: Any = types.ModuleType("sglang.srt.mem_cache.memory_pool")
    module.MambaPool = FakeMambaPool
    module.HybridReqToTokenPool = FakeHybridReqToTokenPool
    return module


def _apply_patch(monkeypatch, module, version):
    patch = ElasticMambaPoolPatch()
    monkeypatch.setattr(
        patch.version_manager, "detect_version", lambda library: version
    )
    assert patch.apply(module) is True
    return patch


def _make_cache_params():
    return types.SimpleNamespace(
        layers=[3, 7],
        shape=types.SimpleNamespace(
            conv=[(4, 3)],
            temporal=(2, 2),
            conv_shard_groups=(4, 4, 8),
            conv_slice_axis=1,
        ),
        dtype=types.SimpleNamespace(conv="conv-dtype", temporal="ssm-dtype"),
    )


def _mamba_pool_kwargs(**overrides):
    kwargs = {
        "size": 8,
        "spec_state_size": 0,
        "cache_params": _make_cache_params(),
        "mamba_layer_ids": [3, 7],
        "device": "cuda:0",
        "enable_memory_saver": False,
        "speculative_num_draft_tokens": None,
        "speculative_eagle_topk": None,
        "enable_linear_replayssm": False,
        "linear_replayssm_cache_len": 16,
        "envelope_layout": False,
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize("version", ["0.5.16", "0.5.20"])
def test_mamba_pool_cls_rebound_to_elastic(monkeypatch, version):
    """0.5.16+ constructs the pool from the class attribute, so the module
    alias alone leaves hybrid models on the native, static MambaPool."""
    _install_fake_torch(monkeypatch)
    module = _make_memory_pool_module()

    _apply_patch(monkeypatch, module, version)

    assert module.MambaPool is module.ElasticMambaPool
    assert (
        module.HybridReqToTokenPool.mamba_pool_cls is module.ElasticMambaPool
    )


def test_mamba_pool_cls_untouched_before_0516(monkeypatch):
    _install_fake_torch(monkeypatch)
    module = _make_memory_pool_module()
    native_pool_cls = module.HybridReqToTokenPool.mamba_pool_cls

    _apply_patch(monkeypatch, module, "0.5.15")

    # Pre-0.5.16 construction goes through the module attribute, which the
    # alias covers; the class attribute (absent there) stays untouched.
    assert module.MambaPool is module.ElasticMambaPool
    assert module.HybridReqToTokenPool.mamba_pool_cls is native_pool_cls


@pytest.fixture
def elastic_mamba_pool_cls(monkeypatch):
    _install_fake_torch(monkeypatch)
    _install_fake_sglang_distributed(monkeypatch)
    record: Dict[str, Any] = {}
    _install_fake_interfaces(monkeypatch, record)
    module = _make_memory_pool_module()
    _apply_patch(monkeypatch, module, "0.5.20")
    cls = module.ElasticMambaPool
    cls._kvcached_test_record = record
    return cls


def test_ctor_accepts_0517_and_0520_kwargs(elastic_mamba_pool_cls):
    """0.5.17+ _init_mamba_pool always passes enable_linear_replayssm_spec;
    a constructor that rejects it fails hybrid startup with a TypeError."""
    pool = elastic_mamba_pool_cls(
        **_mamba_pool_kwargs(enable_linear_replayssm_spec=False)
    )
    record = elastic_mamba_pool_cls._kvcached_test_record
    assert record["alloc_mamba_states"]["num_slots"] == 9
    assert record["get_kv_cache_manager"]["num_layers"] == 2
    assert pool.available_size() == 0


def test_ctor_accepts_0516_kwarg_name(elastic_mamba_pool_cls):
    pool = elastic_mamba_pool_cls(
        **_mamba_pool_kwargs(enable_gdn_replayssm_spec=False)
    )
    assert pool.num_mamba_layers == 2


@pytest.mark.parametrize(
    "kwarg", ["enable_gdn_replayssm_spec", "enable_linear_replayssm_spec"]
)
def test_ctor_refuses_replayssm_spec(elastic_mamba_pool_cls, kwarg):
    with pytest.raises(NotImplementedError, match="ReplaySSM"):
        elastic_mamba_pool_cls(**_mamba_pool_kwargs(**{kwarg: True}))


def test_ctor_sets_attributes_inherited_methods_read(elastic_mamba_pool_cls):
    """The 0.5.20 transfer iterator and copy path read these attributes,
    which only the (skipped) native init used to set."""
    pool = elastic_mamba_pool_cls(**_mamba_pool_kwargs())

    assert pool.mamba_layer_ids == [3, 7]
    assert pool.conv_slice_axis == 1
    assert pool.conv_shard_groups == (4, 4, 8)
    assert pool.debug_memory_pool is False
    assert pool.enable_linear_replayssm_spec is False
    assert pool.replayssm_spec_fold is False
    assert pool.replayssm_write_pos is None


def test_register_slot_state_refused(elastic_mamba_pool_cls):
    """0.5.20 registers Qwen4-Exp PLE sibling states that must follow slot
    clears and copies; the elastic pool refuses rather than dropping them."""
    pool = elastic_mamba_pool_cls(**_mamba_pool_kwargs())

    with pytest.raises(NotImplementedError, match="slot-sibling"):
        pool.register_slot_state(object())


def test_ctor_is_isolated_from_installed_sglang(monkeypatch):
    """The rank helpers must come from the test's own fakes, never from
    whatever sglang the environment has (importing a real one under the
    fake torch fails the constructor tests)."""
    def _boom():
        raise RuntimeError("ctor reached the environment's sglang")

    for name in ("sglang", "sglang.srt", "sglang.srt.distributed"):
        module = types.ModuleType(name)
        for helper in (
            "get_tensor_model_parallel_rank",
            "get_tensor_model_parallel_world_size",
            "get_pipeline_model_parallel_rank",
        ):
            setattr(module, helper, _boom)
        monkeypatch.setitem(sys.modules, name, module)

    _install_fake_torch(monkeypatch)
    _install_fake_sglang_distributed(monkeypatch)
    record: Dict[str, Any] = {}
    _install_fake_interfaces(monkeypatch, record)
    module = _make_memory_pool_module()
    _apply_patch(monkeypatch, module, "0.5.20")

    pool = module.ElasticMambaPool(**_mamba_pool_kwargs())

    assert pool.num_mamba_layers == 2
    assert record["init_kvcached"] == {
        "tp_rank": 0,
        "world_size": 1,
        "pp_rank": 0,
        "async_sched": True,
    }


class _Native0520MambaPool:
    """CPU port of the SGLang 0.5.20 native MambaPool methods the elastic
    class inherits (memory_pool.py at v0.5.20): clear_slots' non-fused
    branch, _iter_transfer_state_entries, and the transfer metadata
    readers.  The fused clear branch is CUDA-only upstream, so the port
    keeps only the Python fallback.  Serving as the patch's base class
    makes the tests exercise the true inherited code paths."""

    _slot_siblings: tuple = ()
    _NON_TRANSFER_STATE_FIELDS = frozenset({
        "intermediate_ssm",
        "intermediate_conv_window",
        "replayssm_d",
        "replayssm_k",
        "replayssm_g",
        "replayssm_rawv",
        "replayssm_rawk",
        "replayssm_beta",
    })

    # Set by the pool subclass constructor, as with the native class.
    mamba_cache: Any
    mamba_layer_ids: Any
    conv_slice_axis: Any
    conv_shard_groups: Any

    class State:

        def __init__(self, *, conv, temporal):
            self.conv = conv
            self.temporal = temporal

        def mem_usage_bytes(self):
            return 0

    def clear_slots(self, indices):
        for sibling in self._slot_siblings:
            sibling.reset_slots(indices)
        need_size = len(indices)
        for i in range(len(self.mamba_cache.conv)):
            t = self.mamba_cache.conv[i]
            z = torch.zeros(1, dtype=t.dtype, device=t.device).expand(
                t.shape[0], need_size, *t.shape[2:]
            )
            t[:, indices] = z
        t = self.mamba_cache.temporal
        z = torch.zeros(1, dtype=t.dtype, device=t.device).expand(
            t.shape[0], need_size, *t.shape[2:]
        )
        t[:, indices] = z

    def _iter_transfer_state_entries(self):
        for field, value in vars(self.mamba_cache).items():
            if field in self._NON_TRANSFER_STATE_FIELDS or value is None:
                continue
            tensors = value if isinstance(value, list) else [value]
            slice_axis = self.conv_slice_axis if field == "conv" else 0
            for state_tensor in tensors:
                if state_tensor.numel() == 0:
                    continue
                for layer_index, layer_id in enumerate(self.mamba_layer_ids):
                    yield (
                        field, state_tensor[layer_index], slice_axis, layer_id
                    )
        for sibling in self._slot_siblings:
            yield from sibling.iter_transfer_state_entries()

    def get_contiguous_buf_infos(self):
        data_ptrs, data_lens, item_lens = [], [], []
        for _, state_tensor, _, _ in self._iter_transfer_state_entries():
            data_ptrs.append(state_tensor.data_ptr())
            data_lens.append(state_tensor.nbytes)
            item_lens.append(state_tensor[0].nbytes)
        return data_ptrs, data_lens, item_lens

    def get_state_dim_per_tensor(self):
        dim_per_tensor = []
        entries = self._iter_transfer_state_entries()
        for _, state_tensor, slice_axis, _ in entries:
            if slice_axis is None:
                dim_per_tensor.append(0)
                continue
            dim_per_tensor.append(state_tensor.shape[1 + slice_axis])
        return dim_per_tensor

    def get_state_layer_ids(self):
        return [
            layer_id
            for _, _, _, layer_id in self._iter_transfer_state_entries()
        ]

    def get_state_slice_outer_counts(self):
        outer_counts = []
        entries = self._iter_transfer_state_entries()
        for _, state_tensor, slice_axis, _ in entries:
            outer_count = (
                1
                if slice_axis is None
                else math.prod(state_tensor.shape[1:1 + slice_axis])
            )
            outer_counts.append(outer_count)
        return outer_counts

    def get_state_conv_shard_groups(self):
        subdims_per_tensor = []
        for field, _, _, _ in self._iter_transfer_state_entries():
            subdims = (
                list(self.conv_shard_groups)
                if field == "conv" and self.conv_shard_groups is not None
                else None
            )
            subdims_per_tensor.append(subdims)
        return subdims_per_tensor


@pytest.fixture
def elastic_state_pool_factory(monkeypatch):
    """ElasticMambaPool built through the real constructor on CPU torch
    tensors, both layouts, every element initialized to 7 — the CPU mirror
    of the T4 state-layout probes from review, with the kvcached
    allocation/IPC boundary stubbed."""
    _install_fake_sglang_distributed(monkeypatch)
    monkeypatch.setattr(
        "kvcached.integration.sglang.patches._is_supported_gpu_device",
        lambda device: True,
    )

    def factory(*, contiguous):
        kvi: Any = types.ModuleType("kvcached.integration.sglang.interfaces")
        kvi.init_kvcached = lambda **kwargs: None

        def alloc_mamba_states(*, num_slots, num_mamba_layers, cache_params,
                               device, group_id):
            conv_shapes = [tuple(s) for s in cache_params.shape.conv]
            temporal_shape = tuple(cache_params.shape.temporal)
            if contiguous:
                conv = [
                    torch.full((num_mamba_layers, num_slots) + s, 7.0)
                    for s in conv_shapes
                ]
                temporal = torch.full(
                    (num_mamba_layers, num_slots) + temporal_shape, 7.0
                )
            else:
                conv = [[
                    torch.full((num_slots,) + s, 7.0)
                    for _ in range(num_mamba_layers)
                ] for s in conv_shapes]
                temporal = [
                    torch.full((num_slots,) + temporal_shape, 7.0)
                    for _ in range(num_mamba_layers)
                ]
            return conv, temporal, {
                "is_contiguous": contiguous,
                "cell_size": 128,
            }

        kvi.alloc_mamba_states = alloc_mamba_states
        kvi.get_kv_cache_manager = lambda **kwargs: FakeManager()
        monkeypatch.setitem(
            sys.modules, "kvcached.integration.sglang.interfaces", kvi
        )

        module: Any = types.ModuleType("sglang.srt.mem_cache.memory_pool")
        module.MambaPool = _Native0520MambaPool
        patch = ElasticMambaPoolPatch()
        monkeypatch.setattr(
            patch.version_manager, "detect_version", lambda library: "0.5.20"
        )
        assert patch.apply(module) is True
        return module.ElasticMambaPool(**_mamba_pool_kwargs(device="cpu"))

    return factory


def test_clear_slots_per_layer_clears_target_and_preserves_rest(
        elastic_state_pool_factory):
    """The 0.5.20 deferred clear calls mamba_pool.clear_slots(); the native
    body indexes (layers, slots, ...) tensors, which on per-layer state
    zeroes the wrong axis of layer 0 and misses every other layer."""
    pool = elastic_state_pool_factory(contiguous=False)

    pool.clear_slots(torch.tensor([1], dtype=torch.int64))

    state_lists = list(pool.mamba_cache.conv_per_layer) + [
        pool.mamba_cache.temporal_per_layer
    ]
    for state_list in state_lists:
        for layer_t in state_list:
            assert torch.all(layer_t[1] == 0)
            assert torch.all(layer_t[0] == 7)
            assert torch.all(layer_t[2:] == 7)


def test_clear_slots_matches_contiguous_elementwise(
        elastic_state_pool_factory):
    """Stacking the per-layer state after a clear must equal the contiguous
    state after the same clear, element for element."""
    contig = elastic_state_pool_factory(contiguous=True)
    per_layer = elastic_state_pool_factory(contiguous=False)
    indices = torch.tensor([1, 4], dtype=torch.int64)

    contig.clear_slots(indices)
    per_layer.clear_slots(indices)

    for group, shape_list in enumerate(per_layer.mamba_cache.conv_per_layer):
        assert torch.equal(
            torch.stack(shape_list), contig.mamba_cache.conv[group]
        )
    assert torch.equal(
        torch.stack(per_layer.mamba_cache.temporal_per_layer),
        contig.mamba_cache.temporal,
    )


def test_transfer_metadata_readers_agree_across_layouts(
        elastic_state_pool_factory):
    """SGLang's PD setup consumes these readers; the per-layer container
    must produce the same metadata as the contiguous layout."""
    contig = elastic_state_pool_factory(contiguous=True)
    per_layer = elastic_state_pool_factory(contiguous=False)

    assert contig.get_state_layer_ids() == [3, 7, 3, 7]
    assert per_layer.get_state_layer_ids() == contig.get_state_layer_ids()
    assert (per_layer.get_state_slice_outer_counts()
            == contig.get_state_slice_outer_counts())
    assert (per_layer.get_state_conv_shard_groups()
            == contig.get_state_conv_shard_groups())
    assert (per_layer.get_state_dim_per_tensor()
            == contig.get_state_dim_per_tensor())


@pytest.mark.parametrize("contiguous", [True, False])
def test_transfer_iterator_orders_like_buf_infos(elastic_state_pool_factory,
                                                 contiguous):
    """get_contiguous_buf_infos() and the transfer iterator must flatten the
    state in the same order, or PD transfer matches metadata to the wrong
    buffers."""
    pool = elastic_state_pool_factory(contiguous=contiguous)

    entries = list(pool._iter_transfer_state_entries())
    data_ptrs, data_lens, item_lens = pool.get_contiguous_buf_infos()

    assert [t.data_ptr() for _, t, _, _ in entries] == data_ptrs
    assert [t.nbytes for _, t, _, _ in entries] == data_lens
    assert [t[0].nbytes for _, t, _, _ in entries] == item_lens
    assert [f for f, _, _, _ in entries] == [
        "conv", "conv", "temporal", "temporal"
    ]

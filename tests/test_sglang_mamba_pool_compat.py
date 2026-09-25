# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""ElasticMambaPool routing and constructor compatibility for SGLang 0.5.16+.

SGLang 0.5.16 made ``HybridReqToTokenPool`` build its Mamba pool through the
``mamba_pool_cls`` class attribute, which captured the native ``MambaPool``
before any aliasing ran, and added the ``enable_gdn_replayssm_spec``
constructor kwarg (renamed ``enable_linear_replayssm_spec`` in 0.5.17).
"""

import sys
import types
from typing import Any, Dict

import pytest

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

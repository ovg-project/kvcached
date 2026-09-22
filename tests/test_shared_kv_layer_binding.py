# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for cross-layer KV sharing support in the vLLM patches (#417).

GPU-free: exercises the module-level helpers in
``kvcached.integration.vllm.patches`` that the KV cache allocation and
reshape closures use to handle layers vLLM registers in
``kv_cache_groups[*].layer_names`` without a backing ``KVCacheTensor``
(cross-layer KV sharing, e.g. gemma E2B). vLLM records those names in the
runner's ``runner_only_attn_layers`` and never adds them to any tensor's
``shared_by``; direct-indexing them against the layer-to-tensor map was the
KeyError reported in issue #417.
"""
from types import SimpleNamespace

import pytest

from kvcached.integration.vllm.patches import (
    _alias_shared_kv_layers,
    _get_group_size,
    _get_runner_only_attn_layers,
    _tensor_backed_layer_names,
)


def _group(layer_names):
    return SimpleNamespace(layer_names=list(layer_names), kv_cache_spec=None)


def _config(groups):
    return SimpleNamespace(kv_cache_groups=list(groups))


class TestGetRunnerOnlyAttnLayers:

    def test_missing_attribute_is_empty(self):
        assert _get_runner_only_attn_layers(SimpleNamespace()) == frozenset()

    def test_none_attribute_is_empty(self):
        runner = SimpleNamespace(runner_only_attn_layers=None)
        assert _get_runner_only_attn_layers(runner) == frozenset()

    def test_populated_set_is_returned(self):
        runner = SimpleNamespace(runner_only_attn_layers={"a", "b"})
        assert _get_runner_only_attn_layers(runner) == frozenset({"a", "b"})


class TestTensorBackedLayerNames:

    def test_no_runner_only_layers_keeps_all_names_in_order(self):
        grp = _group(["l0", "l1", "l2"])
        assert _tensor_backed_layer_names(grp) == ["l0", "l1", "l2"]
        assert _tensor_backed_layer_names(grp, frozenset()) == ["l0", "l1", "l2"]

    def test_appended_sharing_layers_are_skipped(self):
        # vLLM appends sharing layers after the tensor-backed ones.
        grp = _group(["l0", "l1", "shared0", "shared1"])
        assert _tensor_backed_layer_names(grp, {"shared0", "shared1"}) == ["l0", "l1"]

    def test_group_with_only_runner_only_layers_is_empty(self):
        grp = _group(["enc0", "enc1"])
        assert _tensor_backed_layer_names(grp, {"enc0", "enc1"}) == []


class TestGetGroupSize:

    def test_without_runner_only_matches_raw_max(self):
        cfg = _config([_group(["a0", "a1", "a2"]), _group(["b0", "b1"])])
        assert _get_group_size(cfg) == 3

    def test_sharing_layers_do_not_inflate_pool_count(self):
        # gemma-E2B-style config: sharing layers appended to their target
        # group must not increase the number of allocated pools, which has
        # to stay equal to the scheduler-side num_layers (the scheduler's
        # config never contains the appended names).
        cfg = _config([
            _group(["a0", "a1", "shared0", "shared1"]),
            _group(["b0", "b1"]),
        ])
        assert _get_group_size(cfg) == 4
        assert _get_group_size(cfg, {"shared0", "shared1"}) == 2


class TestAliasSharedKvLayers:

    def test_aliases_to_the_same_object(self):
        target = object()
        kv_caches = {"tgt": target}
        _alias_shared_kv_layers(kv_caches, {"shared": "tgt"})
        assert kv_caches["shared"] is target

    def test_multiple_layers_share_one_target(self):
        target = object()
        kv_caches = {"tgt": target}
        _alias_shared_kv_layers(kv_caches, {"s0": "tgt", "s1": "tgt"})
        assert kv_caches["s0"] is target
        assert kv_caches["s1"] is target

    def test_empty_mapping_is_a_no_op(self):
        kv_caches = {"tgt": object()}
        _alias_shared_kv_layers(kv_caches, {})
        assert set(kv_caches) == {"tgt"}

    def test_missing_target_fails_loud(self):
        with pytest.raises(RuntimeError, match="'gone'.*'shared'"):
            _alias_shared_kv_layers({"tgt": object()}, {"shared": "gone"})


class TestSharingScenario:
    """End-to-end over the helpers, shaped like the #417 report.

    Two attention groups of two tensor-backed layers each; vLLM appended two
    sharing layers to group 0's layer_names and registered them as
    runner-only, without touching any tensor's shared_by.
    """

    def setup_method(self):
        self.tensors = [
            SimpleNamespace(size=1024, shared_by=["a0", "b0"]),
            SimpleNamespace(size=1024, shared_by=["a1", "b1"]),
        ]
        self.cfg = _config([
            _group(["a0", "a1", "s0", "s1"]),
            _group(["b0", "b1"]),
        ])
        self.runner = SimpleNamespace(
            runner_only_attn_layers={"s0", "s1"},
            shared_kv_cache_layers={"s0": "a0", "s1": "a1"},
        )

    def test_every_tensor_backed_layer_resolves(self):
        # The exact lookup pattern that raised KeyError in #417: build the
        # layer-to-tensor map from shared_by, then resolve every group layer.
        layer_to_tensor_cfg = {}
        for tensor_cfg in self.tensors:
            for ln in tensor_cfg.shared_by:
                layer_to_tensor_cfg[ln] = tensor_cfg

        runner_only = _get_runner_only_attn_layers(self.runner)
        resolved = []
        for grp in self.cfg.kv_cache_groups:
            for layer_name in _tensor_backed_layer_names(grp, runner_only):
                resolved.append(layer_to_tensor_cfg[layer_name])  # must not raise
        assert len(resolved) == 4
        # Without the filter the same loop raises KeyError on the appended name.
        with pytest.raises(KeyError):
            for grp in self.cfg.kv_cache_groups:
                for layer_name in grp.layer_names:
                    layer_to_tensor_cfg[layer_name]

    def test_pool_binding_and_aliasing_cover_all_layers(self):
        runner_only = _get_runner_only_attn_layers(self.runner)
        pools = [object(), object()]
        assert _get_group_size(self.cfg, runner_only) == len(pools)

        kv_caches = {}
        for grp in self.cfg.kv_cache_groups:
            for pool_idx, layer_name in enumerate(
                    _tensor_backed_layer_names(grp, runner_only)):
                kv_caches[layer_name] = pools[pool_idx]
        _alias_shared_kv_layers(kv_caches, self.runner.shared_kv_cache_layers)

        assert set(kv_caches) == {"a0", "a1", "b0", "b1", "s0", "s1"}
        assert kv_caches["s0"] is kv_caches["a0"]
        assert kv_caches["s1"] is kv_caches["a1"]
        # Pool i is shared by layer i of each group.
        assert kv_caches["a0"] is pools[0] and kv_caches["b0"] is pools[0]
        assert kv_caches["a1"] is pools[1] and kv_caches["b1"] is pools[1]

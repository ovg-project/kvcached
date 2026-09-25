# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Exercise native 0.29 attention discovery without allocating CUDA storage."""

import copy
import importlib.metadata
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
if not importlib.metadata.version("vllm").startswith("0.29."):
    pytest.skip("requires the vLLM 0.29 attention discovery contract", allow_module_level=True)

import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu import attn_utils

from kvcached.integration.vllm import model_runner_v2 as adapter


@pytest.fixture
def discovery(monkeypatch):
    full = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=64, dtype=torch.float16)
    sliding = SlidingWindowSpec(
        block_size=16, num_kv_heads=1, head_size=32, dtype=torch.float16, sliding_window=64,
    )
    config = KVCacheConfig(num_blocks=4, kv_cache_tensors=[], kv_cache_groups=[
        KVCacheGroupSpec(["full"], UniformTypeKVCacheSpecs(16, {"full": full})),
        KVCacheGroupSpec(["sliding"], UniformTypeKVCacheSpecs(16, {"sliding": sliding})),
    ])
    sharing = {"full_user": "full", "sliding_user": "sliding"}
    snapshots = []

    class Backend:
        @staticmethod
        def full_cls_name():
            return ("test", "Backend")

    class Group:
        def __init__(self, backend, names, spec, group_id):
            self.backend, self.layer_names, self.kv_cache_spec = backend, names, spec

        def create_metadata_builders(self, **kwargs):
            pass

        def get_metadata_builder(self, index):
            return SimpleNamespace()

    layers = {name: SimpleNamespace(get_attn_backend=lambda: Backend, num_heads=2)
              for name in ("full", "sliding", *sharing)}
    monkeypatch.setattr(attn_utils, "get_shared_kv_cache_layers", lambda _: sharing)
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *args: layers)
    monkeypatch.setattr(attn_utils, "AttentionGroup", Group)
    monkeypatch.setattr(attn_utils, "get_attn_cg_support", lambda *args: "cg")

    def prepare(cfg, groups):
        snapshots.append([
            (list(g.kv_cache_spec.kv_cache_specs), g.kv_cache_spec.page_size_bytes)
            for g in cfg.kv_cache_groups
        ])
        return [16, 16]

    monkeypatch.setattr(attn_utils, "prepare_kernel_block_sizes", prepare)
    # apply() installs the existing allocation hook; restore it after each test.
    monkeypatch.setattr(attn_utils, "allocate_kv_cache", attn_utils.allocate_kv_cache)
    monkeypatch.setenv("ENABLE_KVCACHED", "true")

    class Runner:
        def initialize_kv_cache(self, *args, **kwargs):
            pass

    module = SimpleNamespace(GPUModelRunner=Runner, init_attn_backend=attn_utils.init_attn_backend)
    patch = adapter.ModelRunnerV2Patch()
    assert patch.apply(module)
    return SimpleNamespace(config=config, module=module, sharing=sharing,
                           snapshots=snapshots, layers=layers, patch=patch)


def owner_state(config):
    return [(list(g.kv_cache_spec.kv_cache_specs), g.kv_cache_spec.page_size_bytes)
            for g in config.kv_cache_groups]


@pytest.mark.parametrize("filtered", [False, True])
def test_native_discovery_resolves_consumers_without_changing_accounting(discovery, filtered):
    config = discovery.config
    specs = [g.kv_cache_spec for g in config.kv_cache_groups]
    before = owner_state(config)
    kwargs = {"active_layer_names": set(discovery.sharing)} if filtered else {}
    groups, cg, blocks = discovery.module.init_attn_backend(config, object(), "cpu", **kwargs)
    assert cg == "cg" and blocks == [16, 16]
    for index, (owner, consumer) in enumerate((("full", "full_user"), ("sliding", "sliding_user"))):
        assert len(groups[index]) == 1
        assert groups[index][0].layer_names == ([consumer] if filtered else [owner, consumer])
        assert groups[index][0].kv_cache_spec is specs[index].kv_cache_specs[owner]
        assert config.kv_cache_groups[index].kv_cache_spec is specs[index]
        assert type(specs[index].kv_cache_specs) is dict
        # Native discovery must still register consumers for later cache binding.
        assert config.kv_cache_groups[index].layer_names == [owner, consumer]
    assert discovery.snapshots == [before]
    assert owner_state(config) == before


def test_disabled_patch_retains_native_missing_consumer_error(monkeypatch, discovery):
    monkeypatch.setenv("ENABLE_KVCACHED", "false")
    with pytest.raises(KeyError, match="full_user"):
        discovery.module.init_attn_backend(discovery.config, object(), "cpu")


def test_metadata_failure_leaves_owner_specs_unchanged(monkeypatch, discovery):
    before = owner_state(discovery.config)
    specs = [g.kv_cache_spec for g in discovery.config.kv_cache_groups]

    def fail(*args):
        raise RuntimeError("injected metadata failure")

    monkeypatch.setattr(attn_utils, "prepare_kernel_block_sizes", fail)
    with pytest.raises(RuntimeError, match="injected metadata failure"):
        discovery.module.init_attn_backend(discovery.config, object(), "cpu")
    assert owner_state(discovery.config) == before
    assert all(g.kv_cache_spec is spec for g, spec in zip(discovery.config.kv_cache_groups, specs))


def test_missing_owner_is_not_silently_replaced(discovery):
    discovery.sharing["full_user"] = "missing_owner"
    with pytest.raises(KeyError, match="missing_owner"):
        discovery.module.init_attn_backend(discovery.config, object(), "cpu")


def test_unknown_layer_still_raises(discovery):
    discovery.config.kv_cache_groups[0].layer_names.append("unknown")
    discovery.layers["unknown"] = discovery.layers["full"]
    with pytest.raises(KeyError, match="unknown"):
        discovery.module.init_attn_backend(discovery.config, object(), "cpu")


def test_shared_storage_does_not_merge_different_query_head_groups(discovery):
    discovery.layers["full_user"].num_heads = 4
    groups, _, _ = discovery.module.init_attn_backend(discovery.config, object(), "cpu")
    assert [g.layer_names for g in groups[0]] == [["full"], ["full_user"]]
    assert groups[0][0].kv_cache_spec is groups[0][1].kv_cache_spec


def test_patch_is_idempotent_and_does_not_replace_global_discovery(discovery):
    wrapped = discovery.module.init_attn_backend
    assert discovery.patch.apply(discovery.module)
    assert discovery.module.init_attn_backend is wrapped
    # The unpatched native function remains available for controls/other callers.
    with pytest.raises(KeyError, match="full_user"):
        attn_utils.init_attn_backend(copy.deepcopy(discovery.config), object(), "cpu")


def test_lookup_is_group_local_and_unknown_owner_cannot_use_another_group(discovery):
    config = discovery.config
    # A consumer registered in the wrong group must not borrow that group's
    # arbitrary first spec or a spec belonging to a different physical pool.
    config.kv_cache_groups[0].layer_names.append("sliding_user")
    with pytest.raises(KeyError, match="sliding_user"):
        discovery.module.init_attn_backend(config, object(), "cpu")


def test_nonsharing_and_nonuniform_configs_use_original_objects(discovery):
    config = discovery.config
    assert adapter._with_owner_spec_lookup(config, {}) is config
    for group in config.kv_cache_groups:
        group.kv_cache_spec = next(iter(group.kv_cache_spec.kv_cache_specs.values()))
    assert adapter._with_owner_spec_lookup(config, discovery.sharing) is config


def test_lookup_copy_preserves_owner_enumeration_and_descriptor_identity(discovery):
    config = discovery.config
    scoped = adapter._with_owner_spec_lookup(config, discovery.sharing)
    assert scoped is not config
    assert scoped.kv_cache_tensors is config.kv_cache_tensors
    assert scoped.num_blocks == config.num_blocks
    for original, local in zip(config.kv_cache_groups, scoped.kv_cache_groups):
        owners = original.kv_cache_spec.kv_cache_specs
        lookup = local.kv_cache_spec.kv_cache_specs
        assert original.layer_names is local.layer_names
        assert lookup == owners and len(lookup) == len(owners)
        assert list(lookup.items()) == list(owners.items())
        assert list(lookup.values()) == list(owners.values())
        assert dict(lookup) == owners
        assert not set(discovery.sharing).intersection(lookup)
        assert local.kv_cache_spec.page_size_bytes == original.kv_cache_spec.page_size_bytes

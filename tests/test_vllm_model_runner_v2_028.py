# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from kvcached.integration.vllm import model_runner_v2_028 as v2


@pytest.fixture
def adapter(monkeypatch):
    target = ModuleType("vllm.v1.worker.gpu.model_runner")
    original_init = Mock()

    class Runner:
        def __init__(self, config, device):
            original_init(config, device)
            self.vllm_config = config
            self.device = device

    setattr(target, "GPUModelRunner", Runner)
    native = Mock(return_value={"native": object()})
    setattr(target, "init_kv_cache", native)
    bind = Mock()
    sharing = Mock(return_value={})
    utils = ModuleType("vllm.v1.worker.gpu.attn_utils")
    setattr(utils, "bind_kv_cache", bind)
    setattr(utils, "get_shared_kv_cache_layers", sharing)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    worker_init = Mock()
    monkeypatch.setattr(v2, "_initialize_kvcached_worker", worker_init)
    monkeypatch.setattr(v2, "enable_kvcached", lambda: True)
    monkeypatch.setattr(v2, "_is_attention_spec", lambda spec: spec == "attention")

    patch = v2.GPUModelRunnerV2Patch()
    monkeypatch.setattr(patch, "initialize_version_info", lambda: True)
    patch.detected_version = "0.28.0"
    contexts = []
    raw = object()
    views = {"layer.0": object()}
    allocate = Mock(return_value=raw)

    def install_allocate(context_class):
        def alloc(context, config):
            contexts.append(context)
            return allocate(config)
        context_class._allocate_kv_cache_from_kvcached = alloc
        return True

    reshape = Mock(return_value=views)

    def install_reshape(context_class):
        context_class._reshape_kv_cache_tensors_from_kvcached = (
            lambda context, config, tensors: reshape(config, tensors)
        )
        return True

    monkeypatch.setattr(patch, "add_kvcache_allocator", install_allocate)
    monkeypatch.setattr(patch, "add_reshape_methods", install_reshape)
    assert patch.apply(target)
    return SimpleNamespace(
        target=target, native=native, patch=patch, original_init=original_init,
        worker_init=worker_init, bind=bind, sharing=sharing, contexts=contexts,
        allocate=allocate, reshape=reshape, raw=raw, views=views,
    )


def cache_args(model_type="qwen3"):
    return (
        [], {}, SimpleNamespace(kv_cache_groups=[
            SimpleNamespace(kv_cache_spec="attention")]),
        [[SimpleNamespace(backend=object())]], "cuda:1", "auto", [16],
        SimpleNamespace(model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type=model_type))),
    )


def test_actual_v2_entry_point():
    assert v2.GPUModelRunnerV2Patch.target_module == "vllm.v1.worker.gpu.model_runner"
    assert v2.GPUModelRunnerV2Patch.patch_name == "gpu_model_runner_v2"


def test_worker_init_preserves_native_then_initializes_ipc(adapter):
    config = object()
    runner = adapter.target.GPUModelRunner(config, "cuda:1")
    adapter.original_init.assert_called_once_with(config, "cuda:1")
    adapter.worker_init.assert_called_once_with(runner)


def test_disabled_path_is_native(monkeypatch, adapter):
    monkeypatch.setattr(v2, "enable_kvcached", lambda: False)
    args = cache_args()
    assert adapter.target.init_kv_cache(*args) is adapter.native.return_value
    adapter.native.assert_called_once_with(*args)
    adapter.target.GPUModelRunner(object(), "cuda:0")
    adapter.worker_init.assert_not_called()
    adapter.allocate.assert_not_called()


def test_worker_ipc_failure_propagates(adapter):
    adapter.worker_init.side_effect = RuntimeError("IPC unavailable")
    with pytest.raises(RuntimeError, match="IPC unavailable"):
        adapter.target.GPUModelRunner(object(), "cuda:1")


@pytest.mark.parametrize("model_type,module_count", [
    ("qwen3", 1), ("longcat_flash", 2), ("longcat_flash_ngram", 2),
])
def test_views_bound_by_native_helper(adapter, model_type, module_count):
    args = cache_args(model_type)
    assert adapter.target.init_kv_cache(*args) is adapter.views
    adapter.allocate.assert_called_once_with(args[2])
    adapter.reshape.assert_called_once_with(args[2], adapter.raw)
    adapter.bind.assert_called_once_with(adapter.views, args[1], args[0], module_count)
    context = adapter.contexts[0]
    assert context.device == args[4]
    assert context.attn_groups is args[3]
    assert context._kernel_block_sizes is args[6]
    assert context.cache_config.cache_dtype == args[5]
    adapter.native.assert_not_called()


def test_each_cache_initialization_has_its_own_layout_context(adapter):
    first = cache_args()
    second = list(cache_args())
    second[6] = [8]
    adapter.target.init_kv_cache(*first)
    adapter.target.init_kv_cache(*second)
    assert adapter.contexts[0] is not adapter.contexts[1]
    assert adapter.contexts[0]._kernel_block_sizes == [16]
    assert adapter.contexts[1]._kernel_block_sizes == [8]


def test_empty_cache_preserves_native_initialization(adapter):
    args = cache_args()
    args[2].kv_cache_groups.clear()
    assert adapter.target.init_kv_cache(*args) is adapter.native.return_value
    adapter.allocate.assert_not_called()


@pytest.mark.parametrize("unsupported", ["hybrid", "sharing"])
def test_unsupported_lifecycle_rejected_before_allocation(adapter, unsupported):
    args = cache_args()
    if unsupported == "hybrid":
        args[2].kv_cache_groups.append(SimpleNamespace(kv_cache_spec="mamba"))
    else:
        adapter.sharing.return_value = {"layer.1": "layer.0"}
    with pytest.raises(NotImplementedError):
        adapter.target.init_kv_cache(*args)
    adapter.allocate.assert_not_called()
    adapter.bind.assert_not_called()


@pytest.mark.parametrize("stage", ["allocate", "reshape"])
def test_allocation_or_view_failure_does_not_bind_partial_state(adapter, stage):
    getattr(adapter, stage).side_effect = RuntimeError("cache initialization failed")
    with pytest.raises(RuntimeError, match="cache initialization failed"):
        adapter.target.init_kv_cache(*cache_args())
    adapter.bind.assert_not_called()
    adapter.native.assert_not_called()


def test_patch_is_idempotent(adapter):
    cache_init = adapter.target.init_kv_cache
    runner_init = adapter.target.GPUModelRunner.__init__
    assert adapter.patch.apply(adapter.target)
    assert adapter.target.init_kv_cache is cache_init
    assert adapter.target.GPUModelRunner.__init__ is runner_init


def test_missing_v2_entry_point_is_not_patched(monkeypatch):
    patch = v2.GPUModelRunnerV2Patch()
    monkeypatch.setattr(patch, "initialize_version_info", lambda: True)
    patch.detected_version = "0.28.0"
    assert not patch.apply(ModuleType("missing"))


@pytest.mark.parametrize("version", [None, "0.27.2", "0.29.0.dev0", "0.29.0rc1", "0.29.0", "0.30.0"])
def test_version_guard_leaves_other_runners_untouched(monkeypatch, version):
    patch = v2.GPUModelRunnerV2Patch()
    monkeypatch.setattr(patch, "initialize_version_info", lambda: True)
    patch.detected_version = version
    target = ModuleType("vllm.v1.worker.gpu.model_runner")
    setattr(target, "GPUModelRunner", type("Runner", (), {}))
    setattr(target, "init_kv_cache", Mock())
    before = vars(target).copy()
    assert not patch.apply(target)
    assert vars(target) == before


@pytest.mark.parametrize("version,expected", [
    ("0.27.2", []),
    ("0.28.0", ["gpu_model_runner_v2"]),
    ("0.28.1+cu129", ["gpu_model_runner_v2"]),
    ("0.29.0", ["model_runner_v2", "kv_layout_v2"]),
])
def test_autopatch_routes_v2_versions_without_overlap(monkeypatch, version, expected):
    import wrapt.importer

    from kvcached.integration import patch_base
    from kvcached.integration.version_utils import VersionRange

    registered: list[tuple[Any, str]] = []
    manager = Mock()
    manager.register_patches_with_versions.side_effect = registered.extend
    manager.apply_all_patches.return_value = []
    monkeypatch.setattr(patch_base, "PatchManager", lambda _: manager)
    monkeypatch.setattr(patch_base, "log_patch_results", lambda *_: None)
    monkeypatch.setattr(wrapt.importer, "when_imported", lambda _: lambda fn: fn)
    monkeypatch.setenv("KVCACHED_AUTOPATCH", "true")
    spec = importlib.util.spec_from_file_location(
        "_autopatch_version_routing", Path(v2.__file__).with_name("autopatch.py"),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._patch_vllm(ModuleType("vllm"))
    selected = [
        patch.patch_name for patch, range_spec in registered
        if patch.patch_name in ("gpu_model_runner_v2", "model_runner_v2", "kv_layout_v2")
        and VersionRange(range_spec).contains(version)
    ]
    assert selected == expected


def test_worker_registration_uses_actual_tp_pp_and_device(monkeypatch):
    parallel = ModuleType("vllm.distributed.parallel_state")
    setattr(parallel, "get_tensor_model_parallel_rank", lambda: 1)
    setattr(parallel, "get_tensor_model_parallel_world_size", lambda: 2)
    setattr(parallel, "get_pp_group", lambda: SimpleNamespace(rank_in_group=3))
    monkeypatch.setitem(sys.modules, parallel.__name__, parallel)
    interfaces = ModuleType("kvcached.integration.vllm.interfaces")
    initialize = Mock()
    setattr(interfaces, "init_kvcached", initialize)
    monkeypatch.setitem(sys.modules, interfaces.__name__, interfaces)
    from kvcached.integration import vllm as package

    monkeypatch.setattr(package, "interfaces", interfaces, raising=False)
    runner = SimpleNamespace(
        device="cuda:1", vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(async_scheduling=True)),
    )
    v2._initialize_kvcached_worker(runner)
    initialize.assert_called_once_with(
        tp_rank=1, world_size=2, pp_rank=3, is_worker=True,
        device="cuda:1", async_sched=True,
    )
    initialize.reset_mock()
    setattr(parallel, "get_pp_group", Mock(side_effect=RuntimeError("group unavailable")))
    with pytest.raises(RuntimeError, match="group unavailable"):
        v2._initialize_kvcached_worker(runner)
    initialize.assert_not_called()

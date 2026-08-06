# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
import types
from importlib.machinery import ModuleSpec
from types import SimpleNamespace
from unittest import mock


def test_engine_core_marks_multi_pp_for_fanout(monkeypatch, request):
    module_names = (
        "kvcached.integration.vllm.interfaces",
        "kvcached.integration.vllm.patches",
    )
    original_modules = {
        name: sys.modules[name] for name in module_names if name in sys.modules
    }
    for name in module_names:
        sys.modules.pop(name, None)

    def restore_modules():
        for name in module_names:
            sys.modules.pop(name, None)
        sys.modules.update(original_modules)

    request.addfinalizer(restore_modules)

    torch = mock.MagicMock()
    torch.__version__ = "2.6.0"
    torch.__spec__ = ModuleSpec("torch", loader=None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", mock.MagicMock())
    monkeypatch.setenv("KVCACHED_PP_SIZE", "1")

    interfaces = importlib.import_module("kvcached.integration.vllm.interfaces")
    patches = importlib.import_module("kvcached.integration.vllm.patches")

    init_kvcached = mock.Mock()
    monkeypatch.setattr(interfaces, "init_kvcached", init_kvcached)
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)

    engine_mod = types.ModuleType("vllm.v1.engine.core")

    class EngineCore:
        def __init__(self, vllm_config):
            self.vllm_config = vllm_config

    setattr(engine_mod, "EngineCore", EngineCore)
    assert patches.EngineCorePatch().patch_engine_init(engine_mod)

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )
    EngineCore(config)

    assert init_kvcached.call_args.kwargs["world_size"] == 4
    assert init_kvcached.call_args.kwargs["pp_rank"] == -1
    assert init_kvcached.call_args.kwargs["is_worker"] is False
    assert patches.os.environ["KVCACHED_PP_SIZE"] == "2"

# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import sys
import types
from importlib.machinery import ModuleSpec

import pytest


class FakeTensor:
    def __init__(self, shape):
        self.shape = shape


def _install_package_hierarchy(monkeypatch, package_names):
    parent = None
    for name in package_names:
        module = types.ModuleType(name)
        module.__path__ = []
        module.__spec__ = ModuleSpec(name, loader=None, is_package=True)
        monkeypatch.setitem(sys.modules, name, module)
        if parent is not None:
            setattr(parent, name.rsplit(".", 1)[-1], module)
        parent = module
    return parent


def _install_fake_nixl_module(monkeypatch, module_style="legacy"):
    package_names = [
        "vllm",
        "vllm.distributed",
        "vllm.distributed.kv_transfer",
        "vllm.distributed.kv_transfer.kv_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1",
    ]
    parent = _install_package_hierarchy(monkeypatch, package_names)

    class NixlConnector:
        @classmethod
        def get_required_kvcache_layout(cls, vllm_config):
            return "HND"

    class NixlConnectorWorker:
        def __init__(self, num_blocks=0, backend_name="", use_mla=False):
            self.num_blocks = num_blocks
            self.calls = 0
            # Mirror real vLLM: the backend is reported via backend_name
            # (= attn_backends[0].get_name()), not a `_use_flashinfer` attr.
            self.backend_name = backend_name
            self.use_mla = use_mla

        def register_kv_caches(self, kv_caches, *args, **kwargs):
            self.calls += 1
            return self.num_blocks

    if module_style == "legacy":
        module = types.ModuleType(
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector"
        )
        module.__spec__ = ModuleSpec(module.__name__, loader=None)
        setattr(module, "NixlConnector", NixlConnector)
        setattr(module, "NixlConnectorWorker", NixlConnectorWorker)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        setattr(parent, "nixl_connector", module)
    elif module_style == "split":
        nixl_parent = types.ModuleType(
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl"
        )
        nixl_parent.__path__ = []
        nixl_parent.__spec__ = ModuleSpec(
            nixl_parent.__name__, loader=None, is_package=True
        )
        monkeypatch.setitem(sys.modules, nixl_parent.__name__, nixl_parent)
        setattr(parent, "nixl", nixl_parent)

        connector_module = types.ModuleType(
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector"
        )
        connector_module.__spec__ = ModuleSpec(connector_module.__name__, loader=None)
        setattr(connector_module, "NixlConnector", NixlConnector)
        monkeypatch.setitem(sys.modules, connector_module.__name__, connector_module)
        setattr(nixl_parent, "connector", connector_module)

        worker_module = types.ModuleType(
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker"
        )
        worker_module.__spec__ = ModuleSpec(worker_module.__name__, loader=None)
        setattr(worker_module, "NixlConnectorWorker", NixlConnectorWorker)
        monkeypatch.setitem(sys.modules, worker_module.__name__, worker_module)
        setattr(nixl_parent, "worker", worker_module)
    else:
        raise ValueError(f"Unsupported fake Nixl module style: {module_style}")

    return NixlConnector, NixlConnectorWorker


def _install_fake_torch(monkeypatch):
    torch = types.ModuleType("torch")
    torch.__spec__ = ModuleSpec("torch", loader=None)
    monkeypatch.setitem(sys.modules, "torch", torch)


def _enable_kvcached_nixl(monkeypatch, nixl_compat):
    monkeypatch.setattr(nixl_compat, "enable_kvcached", lambda: True)
    monkeypatch.setattr(nixl_compat, "CONTIGUOUS_LAYOUT", False)


def test_nixl_connector_patch_is_idempotent(monkeypatch):
    _install_fake_torch(monkeypatch)
    NixlConnector, NixlConnectorWorker = _install_fake_nixl_module(monkeypatch)

    from kvcached.integration.vllm import nixl_compat

    monkeypatch.setattr(nixl_compat, "enable_kvcached", lambda: False)

    assert nixl_compat.patch_nixl_connector() is True
    assert nixl_compat.patch_nixl_connector() is True

    assert NixlConnector.get_required_kvcache_layout("config") == "HND"

    worker = NixlConnectorWorker(num_blocks=17)
    assert worker.register_kv_caches({"layer": FakeTensor((2, 33, 4, 5, 6))}) == 17
    assert worker.calls == 1


def test_nixl_connector_patch_uses_registered_tensor_block_count(monkeypatch):
    _install_fake_torch(monkeypatch)
    NixlConnector, NixlConnectorWorker = _install_fake_nixl_module(monkeypatch)

    from kvcached.integration.vllm import nixl_compat

    _enable_kvcached_nixl(monkeypatch, nixl_compat)

    assert nixl_compat.patch_nixl_connector() is True

    assert NixlConnector.get_required_kvcache_layout("config") is None

    worker = NixlConnectorWorker(num_blocks=17)
    assert worker.register_kv_caches({"layer": FakeTensor((2, 33, 4, 5, 6))}) == 33
    assert worker.num_blocks == 33
    assert worker.calls == 1


def test_nixl_connector_patch_uses_physical_tensor_block_count_for_ratio(monkeypatch):
    _install_fake_torch(monkeypatch)
    _, NixlConnectorWorker = _install_fake_nixl_module(monkeypatch)

    from kvcached.integration.vllm import nixl_compat

    _enable_kvcached_nixl(monkeypatch, nixl_compat)

    assert nixl_compat.patch_nixl_connector() is True

    worker = NixlConnectorWorker(num_blocks=17)
    # Simulate block_size/kernel_block_size == 2: NIXL metadata must use the
    # physical post-ratio block count exposed by the KV tensor, not the logical
    # pre-ratio scheduler count.
    assert worker.register_kv_caches({"layer": FakeTensor((2, 34, 4, 5, 6))}) == 34
    assert worker.num_blocks == 34


def test_nixl_connector_patch_rejects_inconsistent_block_counts(monkeypatch):
    _install_fake_torch(monkeypatch)
    _, NixlConnectorWorker = _install_fake_nixl_module(monkeypatch)

    from kvcached.integration.vllm import nixl_compat

    _enable_kvcached_nixl(monkeypatch, nixl_compat)

    assert nixl_compat.patch_nixl_connector() is True

    worker = NixlConnectorWorker(num_blocks=17)
    with pytest.raises(RuntimeError, match="inconsistent KV block counts"):
        worker.register_kv_caches(
            {
                "layer_a": FakeTensor((2, 33, 4, 5, 6)),
                "layer_b": FakeTensor((2, 44, 4, 5, 6)),
            }
        )
    assert worker.num_blocks == 17
    assert worker.calls == 0


def test_nixl_connector_patch_does_not_share_block_state(monkeypatch):
    _install_fake_torch(monkeypatch)
    _, NixlConnectorWorker = _install_fake_nixl_module(monkeypatch)

    from kvcached.integration.vllm import nixl_compat

    _enable_kvcached_nixl(monkeypatch, nixl_compat)

    assert nixl_compat.patch_nixl_connector() is True

    first_worker = NixlConnectorWorker(num_blocks=17)
    second_worker = NixlConnectorWorker(num_blocks=17)

    assert first_worker.register_kv_caches({"layer": FakeTensor((2, 33, 4, 5, 6))}) == 33
    assert second_worker.register_kv_caches({"layer": FakeTensor((2, 44, 4, 5, 6))}) == 44
    assert first_worker.register_kv_caches({"layer": FakeTensor((2, 33, 4, 5, 6))}) == 33


def test_nixl_connector_patch_supports_split_vllm_modules(monkeypatch):
    _install_fake_torch(monkeypatch)
    NixlConnector, NixlConnectorWorker = _install_fake_nixl_module(
        monkeypatch, module_style="split"
    )

    from kvcached.integration.vllm import nixl_compat

    _enable_kvcached_nixl(monkeypatch, nixl_compat)

    assert nixl_compat.patch_nixl_connector() is True

    assert NixlConnector.get_required_kvcache_layout("config") is None

    # FlashInfer layout keeps blocks in dim 0; detected via backend_name.
    worker = NixlConnectorWorker(num_blocks=17, backend_name="FLASHINFER")
    assert worker.register_kv_caches({"layer": FakeTensor((55, 2, 4, 5, 6))}) == 55


def test_nixl_connector_patch_uses_dim0_for_mla(monkeypatch):
    """MLA tensors expose blocks in dim 0; detected via worker.use_mla."""
    _install_fake_torch(monkeypatch)
    _, NixlConnectorWorker = _install_fake_nixl_module(monkeypatch)

    from kvcached.integration.vllm import nixl_compat

    _enable_kvcached_nixl(monkeypatch, nixl_compat)

    assert nixl_compat.patch_nixl_connector() is True

    worker = NixlConnectorWorker(num_blocks=17, use_mla=True)
    # MLA shape (num_blocks, block_size, head_size); blocks are in dim 0.
    assert worker.register_kv_caches({"layer": FakeTensor((48, 16, 576))}) == 48
    assert worker.num_blocks == 48


def test_nixl_connector_patch_handles_missing_layout_method(monkeypatch):
    """Older vLLM where NixlConnector has no get_required_kvcache_layout.

    The patch must not AttributeError; it skips the layout override and still
    patches register_kv_caches.
    """
    _install_fake_torch(monkeypatch)
    NixlConnector, NixlConnectorWorker = _install_fake_nixl_module(monkeypatch)
    delattr(NixlConnector, "get_required_kvcache_layout")

    from kvcached.integration.vllm import nixl_compat

    _enable_kvcached_nixl(monkeypatch, nixl_compat)

    # Must not raise despite the missing layout classmethod.
    assert nixl_compat.patch_nixl_connector() is True
    assert not hasattr(NixlConnector, "get_required_kvcache_layout")

    # register_kv_caches is still patched and rewrites num_blocks.
    worker = NixlConnectorWorker(num_blocks=17)
    assert worker.register_kv_caches({"layer": FakeTensor((2, 33, 4, 5, 6))}) == 33
    assert worker.num_blocks == 33


def test_nixl_connector_patch_rejects_contiguous_kvcached_layout(monkeypatch):
    _install_fake_torch(monkeypatch)
    NixlConnector, NixlConnectorWorker = _install_fake_nixl_module(monkeypatch)

    from kvcached.integration.vllm import nixl_compat

    monkeypatch.setattr(nixl_compat, "enable_kvcached", lambda: True)
    monkeypatch.setattr(nixl_compat, "CONTIGUOUS_LAYOUT", True)

    assert nixl_compat.patch_nixl_connector() is True

    with pytest.raises(RuntimeError, match="KVCACHED_CONTIGUOUS_LAYOUT=false"):
        NixlConnector.get_required_kvcache_layout("config")

    worker = NixlConnectorWorker(num_blocks=17)
    with pytest.raises(RuntimeError, match="KVCACHED_CONTIGUOUS_LAYOUT=false"):
        worker.register_kv_caches({"layer": FakeTensor((2, 33, 4, 5, 6))})
    assert worker.num_blocks == 17
    assert worker.calls == 0


def test_nixl_connector_base_patch_adapter(monkeypatch):
    _install_fake_torch(monkeypatch)
    NixlConnector, NixlConnectorWorker = _install_fake_nixl_module(monkeypatch)

    from kvcached.integration.vllm import nixl_compat

    _enable_kvcached_nixl(monkeypatch, nixl_compat)

    patch = nixl_compat.NixlConnectorPatch()
    assert patch.library == "vllm"
    assert patch.target_module == "vllm"
    assert patch.patch_name == "nixl_connector_compat"
    assert patch.apply(sys.modules["vllm"]) is True
    assert NixlConnector.get_required_kvcache_layout("config") is None

    worker = NixlConnectorWorker(num_blocks=17)
    assert worker.register_kv_caches({"layer": FakeTensor((2, 66, 4, 5, 6))}) == 66

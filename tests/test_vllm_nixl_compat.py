# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import sys
import types
from importlib.machinery import ModuleSpec

import pytest


@pytest.fixture(autouse=True)
def _reset_force_split_signal():
    # NixlConnectorPatch._pending_force_split is a process-global class attribute;
    # reset it after every test so state can't leak between tests.
    yield
    mod = sys.modules.get("kvcached.integration.vllm.nixl_compat")
    if mod is not None:
        mod.NixlConnectorPatch._pending_force_split = False


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
            self.transfer_topo = None

        def register_kv_caches(self, kv_caches, *args, **kwargs):
            self.calls += 1
            # Mirror real vLLM base register_kv_caches: build a TransferTopology.
            # Only when a test has installed a fake utils.TransferTopology, so the
            # existing tests (which don't) are unaffected. This lets tests drive the
            # full reconcile->build-topology->consume-signal chain end to end.
            utils = sys.modules.get(
                "vllm.distributed.kv_transfer.kv_connector.utils"
            )
            topo_cls = getattr(utils, "TransferTopology", None) if utils else None
            if topo_cls is not None:
                self.transfer_topo = topo_cls()
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


# ---------------------------------------------------------------------------
# vLLM >=0.23 blocks-first split-half layout reconciliation
# ---------------------------------------------------------------------------


class FakeStridedTensor:
    """Minimal tensor exposing the surface _reconcile_blocks_first_layout needs:
    shape, dim(), stride(), transpose(). transpose() returns a view with the two
    dims/strides swapped and records that it happened."""

    def __init__(self, shape, stride, transposed=False):
        self.shape = tuple(shape)
        self._stride = tuple(stride)
        self.transposed = transposed
        self.is_null = False

    def dim(self):
        return len(self.shape)

    def stride(self):
        return self._stride

    def transpose(self, a, b):
        shp = list(self.shape)
        strd = list(self._stride)
        shp[a], shp[b] = shp[b], shp[a]
        strd[a], strd[b] = strd[b], strd[a]
        return FakeStridedTensor(shp, strd, transposed=True)


def _split_half_tensor():
    # (N, 2, block, H, D) with block_stride == hidden (= block*H*D) -> split-half.
    hidden = 4 * 5 * 6  # 120
    return FakeStridedTensor((8, 2, 4, 5, 6), (hidden, 10_000_000, 30, 6, 1))


def _interleaved_tensor():
    # block_stride == 2*hidden -> K/V interleaved within block (matches NIXL).
    hidden = 4 * 5 * 6
    return FakeStridedTensor((8, 2, 4, 5, 6), (2 * hidden, hidden, 30, 6, 1))


def _make_patch(monkeypatch):
    _install_fake_torch(monkeypatch)
    _install_fake_nixl_module(monkeypatch)
    from kvcached.integration.vllm import nixl_compat

    _enable_kvcached_nixl(monkeypatch, nixl_compat)
    nixl_compat.NixlConnectorPatch._pending_force_split = False
    return nixl_compat, nixl_compat.NixlConnectorPatch()


def test_reconcile_transposes_split_half_and_signals(monkeypatch):
    nixl_compat, patch = _make_patch(monkeypatch)
    worker = types.SimpleNamespace()  # no _has_mamba
    out = patch._reconcile_blocks_first_layout(worker, {"layer": _split_half_tensor()})

    # (N, 2, ...) presented to NIXL as (2, N, ...) so K/V is the outer dim.
    assert out["layer"].shape == (2, 8, 4, 5, 6)
    assert out["layer"].transposed is True
    # signalled the topology hook to force split_k_and_v.
    assert nixl_compat.NixlConnectorPatch._pending_force_split is True


def test_reconcile_leaves_interleaved_untouched(monkeypatch):
    nixl_compat, patch = _make_patch(monkeypatch)
    t = _interleaved_tensor()
    out = patch._reconcile_blocks_first_layout(types.SimpleNamespace(), {"layer": t})

    assert out["layer"] is t  # not transposed
    assert nixl_compat.NixlConnectorPatch._pending_force_split is False


def test_reconcile_rejects_unknown_stride(monkeypatch):
    nixl_compat, patch = _make_patch(monkeypatch)
    bad = FakeStridedTensor((8, 2, 4, 5, 6), (999, 10_000_000, 30, 6, 1))
    with pytest.raises(RuntimeError, match="unexpected attention KV block stride"):
        patch._reconcile_blocks_first_layout(types.SimpleNamespace(), {"layer": bad})


def test_reconcile_rejects_hybrid_mamba(monkeypatch):
    nixl_compat, patch = _make_patch(monkeypatch)
    worker = types.SimpleNamespace(_has_mamba=True)
    with pytest.raises(RuntimeError, match="hybrid/mamba"):
        patch._reconcile_blocks_first_layout(worker, {"layer": _split_half_tensor()})
    # signal must not stay set after bailing out.
    assert nixl_compat.NixlConnectorPatch._pending_force_split is False


def _install_fake_transfer_topology(monkeypatch):
    utils = types.ModuleType(
        "vllm.distributed.kv_transfer.kv_connector.utils"
    )

    class TransferTopology:
        def __init__(self, is_mla=False, is_mamba=False, blocks_first=True):
            self.is_mla = is_mla
            self.is_mamba = is_mamba
            self._is_kv_layout_blocks_first = blocks_first

    setattr(utils, "TransferTopology", TransferTopology)
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.kv_transfer.kv_connector.utils",
        utils,
    )
    parent = sys.modules["vllm.distributed.kv_transfer.kv_connector"]
    monkeypatch.setattr(parent, "utils", utils, raising=False)
    return TransferTopology


def test_topology_flip_only_when_signalled(monkeypatch):
    nixl_compat, _ = _make_patch(monkeypatch)
    TransferTopology = _install_fake_transfer_topology(monkeypatch)
    assert nixl_compat.patch_nixl_connector() is True

    # Signalled (wrapper transposed a split-half tensor): flip + consume.
    nixl_compat.NixlConnectorPatch._pending_force_split = True
    t = TransferTopology()
    assert t._is_kv_layout_blocks_first is False
    assert nixl_compat.NixlConnectorPatch._pending_force_split is False

    # Not signalled (interleaved / native): keep NIXL's default path.
    t2 = TransferTopology()
    assert t2._is_kv_layout_blocks_first is True

    # Signalled but MLA/mamba topology: never flip (but still consume).
    nixl_compat.NixlConnectorPatch._pending_force_split = True
    t3 = TransferTopology(is_mla=True)
    assert t3._is_kv_layout_blocks_first is True
    assert nixl_compat.NixlConnectorPatch._pending_force_split is False


def test_register_to_topology_flip_end_to_end(monkeypatch):
    """Drive the full chain and the idempotency regression together.

    register wrapper -> _reconcile transposes a split-half tensor and signals ->
    base register builds a TransferTopology -> its patched __init__ consumes the
    signal and flips. Also runs after a SECOND patch_nixl_connector() call: that is
    the exact scenario the reviewer flagged, and this asserts the wrapper's signal
    still reaches the topology hook (it would NOT with a per-instance, unguarded
    signal, where the second call rewraps register onto a new instance).
    """
    _install_fake_torch(monkeypatch)
    _, NixlConnectorWorker = _install_fake_nixl_module(monkeypatch)
    from kvcached.integration.vllm import nixl_compat

    _enable_kvcached_nixl(monkeypatch, nixl_compat)
    nixl_compat.NixlConnectorPatch._pending_force_split = False
    _install_fake_transfer_topology(monkeypatch)

    assert nixl_compat.patch_nixl_connector() is True
    assert nixl_compat.patch_nixl_connector() is True  # second patch (the regression)

    # split-half -> full chain flips the topology built inside register.
    worker = NixlConnectorWorker(num_blocks=8)
    worker.register_kv_caches({"layer": _split_half_tensor()})
    assert worker.transfer_topo is not None
    assert worker.transfer_topo._is_kv_layout_blocks_first is False

    # interleaved -> no transpose, no signal -> topology keeps NIXL's default path.
    worker2 = NixlConnectorWorker(num_blocks=8)
    worker2.register_kv_caches({"layer": _interleaved_tensor()})
    assert worker2.transfer_topo._is_kv_layout_blocks_first is True


def _install_fake_nixl_with_push(monkeypatch):
    """Install a base/pull/push worker hierarchy mirroring real vLLM: the base
    defines register_kv_caches, the pull subclass inherits it, and the push
    subclass overrides it but delegates via super()."""
    _install_package_hierarchy(
        monkeypatch,
        [
            "vllm",
            "vllm.distributed",
            "vllm.distributed.kv_transfer",
            "vllm.distributed.kv_transfer.kv_connector",
            "vllm.distributed.kv_transfer.kv_connector.v1",
        ],
    )

    class NixlConnector:
        @classmethod
        def get_required_kvcache_layout(cls, vllm_config):
            return "HND"

    class NixlBaseConnectorWorker:
        def __init__(self, num_blocks=0, backend_name="", use_mla=False):
            self.num_blocks = num_blocks
            self.backend_name = backend_name
            self.use_mla = use_mla
            self.transfer_topo = None

        def register_kv_caches(self, kv_caches, *args, **kwargs):
            utils = sys.modules.get(
                "vllm.distributed.kv_transfer.kv_connector.utils"
            )
            topo_cls = getattr(utils, "TransferTopology", None) if utils else None
            if topo_cls is not None:
                self.transfer_topo = topo_cls()
            return self.num_blocks

    class NixlPullConnectorWorker(NixlBaseConnectorWorker):
        pass  # inherits register_kv_caches from the base

    class NixlPushConnectorWorker(NixlBaseConnectorWorker):
        def register_kv_caches(self, kv_caches, *args, **kwargs):
            return super().register_kv_caches(kv_caches, *args, **kwargs)

    module = types.ModuleType(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector"
    )
    setattr(module, "NixlConnector", NixlConnector)
    # _import_nixl_connector_classes resolves NixlConnectorWorker to the pull class.
    setattr(module, "NixlConnectorWorker", NixlPullConnectorWorker)
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector",
        module,
    )
    return NixlPullConnectorWorker, NixlPushConnectorWorker


def test_patch_covers_both_pull_and_push_workers(monkeypatch):
    """The wrapper must run for BOTH pull and push workers. Push overrides
    register_kv_caches and delegates via super(); patching the base worker (the
    class that defines register) covers it, whereas patching only the pull
    subclass would leave push silently unpatched -> silent corruption."""
    _install_fake_torch(monkeypatch)
    Pull, Push = _install_fake_nixl_with_push(monkeypatch)
    from kvcached.integration.vllm import nixl_compat

    _enable_kvcached_nixl(monkeypatch, nixl_compat)
    nixl_compat.NixlConnectorPatch._pending_force_split = False
    _install_fake_transfer_topology(monkeypatch)

    assert nixl_compat.patch_nixl_connector() is True

    # Pull: inherits the (patched) base register -> reconcile runs -> flip.
    pull = Pull(num_blocks=8)
    pull.register_kv_caches({"layer": _split_half_tensor()})
    assert pull.transfer_topo._is_kv_layout_blocks_first is False

    # Push: its super().register_kv_caches() hits the patched base -> also covered.
    push = Push(num_blocks=8)
    push.register_kv_caches({"layer": _split_half_tensor()})
    assert push.transfer_topo._is_kv_layout_blocks_first is False

import importlib
import sys
import types
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
KVCACHED_PATH = str(REPO_ROOT / "kvcached")
KVCACHED_INTEGRATION_PATH = str(Path(KVCACHED_PATH) / "integration")


def _prepare_kvcached_package(monkeypatch):
    package = types.ModuleType("kvcached")
    package.__path__ = [KVCACHED_PATH]
    integration_package = types.ModuleType("kvcached.integration")
    integration_package.__path__ = [KVCACHED_INTEGRATION_PATH]
    monkeypatch.setitem(sys.modules, "kvcached", package)
    monkeypatch.setitem(sys.modules, "kvcached.integration", integration_package)


def _make_vllm_package(monkeypatch):
    import importlib.util

    pkg = types.ModuleType("vllm")
    pkg.__path__ = []
    pkg.__spec__ = importlib.util.spec_from_loader("vllm", loader=None, origin="vllm")
    v2_pkg = types.ModuleType("vllm.v2")
    v2_pkg.__path__ = []
    mod = types.ModuleType("vllm.v2.foo")
    setattr(mod, "marker", True)
    monkeypatch.setitem(sys.modules, "vllm", pkg)
    monkeypatch.setitem(sys.modules, "vllm.v2", v2_pkg)
    monkeypatch.setitem(sys.modules, "vllm.v2.foo", mod)
    return mod


def test_import_vllm_module_falls_back_to_v2(monkeypatch):
    _prepare_kvcached_package(monkeypatch)
    mod = _make_vllm_package(monkeypatch)
    patches = importlib.import_module("kvcached.integration.vllm.patches")
    imported = patches._import_vllm_module("vllm.v1.foo")
    assert imported is mod


def test_patch_manager_uses_first_available_target_module(monkeypatch):
    _prepare_kvcached_package(monkeypatch)
    _make_vllm_package(monkeypatch)
    patch_base = importlib.import_module("kvcached.integration.patch_base")

    BasePatch = patch_base.BasePatch  # type: ignore[attr-defined]
    PatchManager = patch_base.PatchManager  # type: ignore[attr-defined]

    class DummyPatch(BasePatch):  # type: ignore[valid-type,misc]
        library = "testlib"
        target_module = ["vllm.v1.foo", "vllm.v2.foo"]

        def apply(self, target_module: Any) -> bool:
            return getattr(target_module, "marker", False)

    manager = PatchManager("testlib")
    patch = DummyPatch()
    assert manager._apply_single_patch(patch) is True

import os
import sys
import types
from pathlib import Path


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
    pkg = types.ModuleType("vllm")
    pkg.__path__ = []
    v2_pkg = types.ModuleType("vllm.v2")
    v2_pkg.__path__ = []
    mod = types.ModuleType("vllm.v2.foo")
    mod.marker = True
    monkeypatch.setitem(sys.modules, "vllm", pkg)
    monkeypatch.setitem(sys.modules, "vllm.v2", v2_pkg)
    monkeypatch.setitem(sys.modules, "vllm.v2.foo", mod)
    return mod


def test_import_vllm_module_falls_back_to_v2(monkeypatch):
    _prepare_kvcached_package(monkeypatch)
    mod = _make_vllm_package(monkeypatch)
    patches = __import__("kvcached.integration.vllm.patches", fromlist=["*"])
    imported = patches._import_vllm_module("vllm.v1.foo")
    assert imported is mod


def test_patch_manager_uses_first_available_target_module(monkeypatch):
    _prepare_kvcached_package(monkeypatch)
    _make_vllm_package(monkeypatch)
    patch_base = __import__("kvcached.integration.patch_base", fromlist=["*"])

    class DummyPatch(patch_base.BasePatch):
        library = "testlib"
        target_module = ["vllm.v1.foo", "vllm.v2.foo"]

        def apply(self, target_module):
            return getattr(target_module, "marker", False)

    manager = patch_base.PatchManager("testlib")
    patch = DummyPatch()
    assert manager._apply_single_patch(patch) is True

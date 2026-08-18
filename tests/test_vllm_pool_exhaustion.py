# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""An exhausted shared KV pool must reach vLLM as a scheduling miss.

vLLM's scheduler has one channel for "this request cannot be served right
now": ``allocate_slots()`` returns None, and it preempts a running request and
retries on the next step. It has no channel for an exception -- ``schedule()``
contains no exception handler, and EngineCore's own handler wraps only
``execute_model`` -- so an exception from the block pool terminates the engine
along with every in-flight request.

Under kvcached the pool can legitimately fail to back an allocation: colocated
engines share one physical pool, and a peer can take the last pages between the
moment availability is observed and the moment they are claimed. These tests
pin the translation, and pin that it stays narrow.
"""
from __future__ import annotations

import importlib
import sys
import types
from typing import Any
from unittest import mock

import pytest


@pytest.fixture
def vllm_patches(monkeypatch):
    torch = mock.MagicMock()
    torch.__version__ = "2.6.0"
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.cuda", torch.cuda)
    monkeypatch.setitem(sys.modules, "torch.utils", torch.utils)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension",
                        torch.utils.cpp_extension)
    monkeypatch.setitem(sys.modules, "posix_ipc", mock.MagicMock())
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", mock.MagicMock())
    monkeypatch.delitem(sys.modules, "kvcached.integration.vllm.patches",
                        raising=False)
    return importlib.import_module("kvcached.integration.vllm.patches")


def _module_with_manager(raises: BaseException | None):
    """Build a stand-in ``vllm.v1.core.kv_cache_manager`` module.

    ``allocate_slots`` either raises what the block pool would raise, or
    returns a sentinel so the success path stays observable.
    """

    class KVCacheManager:
        def allocate_slots(self, *args: Any, **kwargs: Any) -> Any:
            if raises is not None:
                raise raises
            return ("blocks", args, kwargs)

    module = types.ModuleType("vllm.v1.core.kv_cache_manager")
    module.KVCacheManager = KVCacheManager  # type: ignore[attr-defined]
    return module


def _apply(vllm_patches, monkeypatch, module, *, kvcached_enabled=True):
    patch = vllm_patches.KVCacheManagerAllocateSlotsPatch()
    monkeypatch.setattr(patch, "initialize_version_info", lambda: True)
    monkeypatch.setattr(vllm_patches, "enable_kvcached",
                        lambda: kvcached_enabled)
    assert patch.patch_allocate_slots(module) is True
    return module.KVCacheManager()


def test_pool_exhaustion_becomes_a_scheduling_miss(vllm_patches, monkeypatch):
    from kvcached.utils import KVCachePoolExhausted

    manager = _apply(
        vllm_patches, monkeypatch,
        _module_with_manager(KVCachePoolExhausted("physical pool empty")))

    assert manager.allocate_slots("request", 8) is None


def test_contract_violations_still_terminate(vllm_patches, monkeypatch):
    """Only exhaustion is transient.

    Asking the pool for more blocks than it just reported free is a defect in
    the caller. Downgrading it to a scheduling miss would turn a loud bug into
    a request that is quietly never scheduled.
    """
    manager = _apply(
        vllm_patches, monkeypatch,
        _module_with_manager(ValueError("Cannot get 999 free blocks")))

    with pytest.raises(ValueError, match="Cannot get 999 free blocks"):
        manager.allocate_slots("request", 8)

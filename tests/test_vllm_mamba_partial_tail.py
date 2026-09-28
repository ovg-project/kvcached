# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Version, activation and delegation contracts for the narrow Mamba guard."""
# ruff: noqa: F811
import sys
import types
from unittest import mock

import pytest
from test_vllm_pool_exhaustion import vllm_patches  # noqa: F401


def apply(vllm_patches, monkeypatch, version="0.29.0", enabled=True):
    original = mock.Mock(return_value="native hash")
    manager = type("MambaManager", (), {"_cache_partial_tail_block": original})
    target = types.ModuleType("single_type")
    setattr(target, "MambaManager", manager)
    status = types.SimpleNamespace(RUNNING="running")
    request_module = types.ModuleType("vllm.v1.request")
    setattr(request_module, "RequestStatus", status)
    monkeypatch.setitem(sys.modules, request_module.__name__, request_module)
    patch = vllm_patches.MambaPartialTailPatch()
    monkeypatch.setattr(patch.version_manager, "detect_version", lambda _name: version)
    monkeypatch.setattr(vllm_patches, "enable_kvcached", lambda: enabled)
    applied = patch.apply(target)
    return manager(), original, patch, target, applied


@pytest.mark.parametrize("version", ["0.28.0", "0.28.1", "0.29.0"])
@pytest.mark.parametrize("computed", [8, 9])
def test_running_stale_boundary_is_not_delegated(vllm_patches, monkeypatch, version, computed):
    manager, original, _, _, applied = apply(vllm_patches, monkeypatch, version)
    assert applied
    req = types.SimpleNamespace(status="running", num_computed_tokens=computed)
    assert manager._cache_partial_tail_block(request=req, num_tokens=8) is None
    original.assert_not_called()


@pytest.mark.parametrize("status,computed", [
    ("running", 0), ("running", 4), ("remote", 8), ("preempted", 0), ("waiting", 8),
])
def test_native_first_publication_is_preserved(vllm_patches, monkeypatch, status, computed):
    manager, original, _, _, applied = apply(vllm_patches, monkeypatch)
    assert applied
    req = types.SimpleNamespace(status=status, num_computed_tokens=computed)
    assert manager._cache_partial_tail_block(req, 8) == "native hash"
    original.assert_called_once_with(manager, req, 8)


def test_disabled_patch_passes_through_and_propagates_errors(vllm_patches, monkeypatch):
    manager, original, _, _, applied = apply(vllm_patches, monkeypatch, enabled=False)
    assert applied
    original.side_effect = ValueError("native error")
    req = types.SimpleNamespace(status="running", num_computed_tokens=8)
    with pytest.raises(ValueError, match="native error"):
        manager._cache_partial_tail_block(req, 8)


@pytest.mark.parametrize("version", [None, "0.27.0", "0.30.0"])
def test_other_versions_are_not_modified(vllm_patches, monkeypatch, version):
    manager, original, _, _, applied = apply(vllm_patches, monkeypatch, version)
    assert not applied
    assert manager._cache_partial_tail_block is original


def test_patch_installation_is_idempotent(vllm_patches, monkeypatch):
    manager, _, patch, target, applied = apply(vllm_patches, monkeypatch)
    assert applied
    wrapped = manager._cache_partial_tail_block.__func__
    assert patch.apply(target)
    assert manager._cache_partial_tail_block.__func__ is wrapped

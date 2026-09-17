# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Default KV-cache layout per backend.

The contiguous (compound-page) layout hands the attention backend strided
per-layer KV tensors. CUDA's FlashAttention/FlashInfer tolerate that; ROCm's
paged kernels read it incorrectly, so HIP defaults to the per-layer layout. XPU
takes the same conservative default until it is measured on Intel hardware.

Runs without PyTorch; see test_xpu_device_detection.py for the stub approach.
"""

import importlib
import sys
import types

import pytest


def _load_utils(monkeypatch, *, hip=None, cuda=None, xpu=None, explicit=None):
    torch_stub = types.ModuleType("torch")
    version = types.ModuleType("torch.version")
    version.hip = hip  # type: ignore[attr-defined]
    version.cuda = cuda  # type: ignore[attr-defined]
    version.xpu = xpu  # type: ignore[attr-defined]
    torch_stub.version = version  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    monkeypatch.delenv("KVCACHED_BACKEND", raising=False)
    if explicit is None:
        monkeypatch.delenv("KVCACHED_CONTIGUOUS_LAYOUT", raising=False)
    else:
        monkeypatch.setenv("KVCACHED_CONTIGUOUS_LAYOUT", explicit)

    return importlib.reload(importlib.import_module("kvcached.utils"))


@pytest.fixture(autouse=True)
def _restore_utils():
    yield
    importlib.reload(importlib.import_module("kvcached.utils"))


def test_xpu_defaults_to_non_contiguous(monkeypatch):
    utils = _load_utils(monkeypatch, xpu="20250302")
    assert utils.CONTIGUOUS_LAYOUT is False


def test_hip_still_defaults_to_non_contiguous(monkeypatch):
    utils = _load_utils(monkeypatch, hip="6.2.41133-dd7f95766")
    assert utils.CONTIGUOUS_LAYOUT is False


def test_cuda_still_defaults_to_contiguous(monkeypatch):
    """Guards against the XPU arm regressing the CUDA default."""
    utils = _load_utils(monkeypatch, cuda="12.4")
    assert utils.CONTIGUOUS_LAYOUT is True


@pytest.mark.parametrize(
    "explicit, expected",
    [("true", True), ("True", True), ("false", False), ("FALSE", False)],
)
def test_explicit_env_var_overrides_xpu_default(monkeypatch, explicit, expected):
    """An operator must be able to opt into contiguous layout on XPU to measure
    it, which is how the default gets revisited."""
    utils = _load_utils(monkeypatch, xpu="20250302", explicit=explicit)
    assert utils.CONTIGUOUS_LAYOUT is expected


def test_explicit_env_var_overrides_cuda_default(monkeypatch):
    utils = _load_utils(monkeypatch, cuda="12.4", explicit="false")
    assert utils.CONTIGUOUS_LAYOUT is False

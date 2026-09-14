# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Backend detection and device-string handling for CUDA, HIP and XPU.

Runs without PyTorch: a stub ``torch`` module supplies ``torch.version`` so each
backend can be exercised on a CPU runner. ``kvcached.utils`` computes
``ACCELERATOR_BACKEND`` at import time, so every case reloads the module.
"""

import importlib
import sys
import types

import pytest


def _load_utils(monkeypatch, *, hip=None, cuda=None, xpu=None, forced=None):
    """Reload kvcached.utils against a stub torch reporting these versions."""
    torch_stub = types.ModuleType("torch")
    version = types.ModuleType("torch.version")
    version.hip = hip  # type: ignore[attr-defined]
    version.cuda = cuda  # type: ignore[attr-defined]
    version.xpu = xpu  # type: ignore[attr-defined]
    torch_stub.version = version  # type: ignore[attr-defined]
    # Only ever consulted through get_device_module(); a sentinel is enough to
    # prove which submodule was selected without a real torch.
    torch_stub.cuda = "torch.cuda"  # type: ignore[attr-defined]
    torch_stub.xpu = "torch.xpu"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    if forced is None:
        monkeypatch.delenv("KVCACHED_BACKEND", raising=False)
    else:
        monkeypatch.setenv("KVCACHED_BACKEND", forced)

    return importlib.reload(importlib.import_module("kvcached.utils"))


@pytest.fixture(autouse=True)
def _restore_utils():
    """Leave kvcached.utils reloaded against the real environment."""
    yield
    importlib.reload(importlib.import_module("kvcached.utils"))


def test_xpu_torch_build_detected_as_xpu(monkeypatch):
    utils = _load_utils(monkeypatch, xpu="20250302")
    assert utils.ACCELERATOR_BACKEND == "xpu"
    assert utils.IS_XPU_BACKEND is True
    assert utils.get_device_type() == "xpu"
    assert utils.get_device_module() == "torch.xpu"


def test_cuda_and_hip_builds_are_unaffected(monkeypatch):
    """The XPU arm must not perturb the two existing backends."""
    utils = _load_utils(monkeypatch, cuda="12.4")
    assert utils.ACCELERATOR_BACKEND == "cuda"
    assert utils.IS_XPU_BACKEND is False
    assert utils.get_device_type() == "cuda"
    assert utils.get_device_module() == "torch.cuda"

    utils = _load_utils(monkeypatch, hip="6.2.41133-dd7f95766")
    assert utils.ACCELERATOR_BACKEND == "hip"
    assert utils.IS_XPU_BACKEND is False
    # PyTorch-ROCm presents AMD GPUs as CUDA devices, so HIP addresses them
    # through torch.cuda and the "cuda" device prefix.
    assert utils.get_device_type() == "cuda"
    assert utils.get_device_module() == "torch.cuda"


@pytest.mark.parametrize("forced", ["xpu", "cuda", "hip"])
def test_kvcached_backend_env_overrides_torch(monkeypatch, forced):
    """KVCACHED_BACKEND wins, so the Python layer matches a forced build."""
    utils = _load_utils(monkeypatch, cuda="12.4", forced=forced)
    assert utils.ACCELERATOR_BACKEND == forced


def test_invalid_backend_override_is_ignored(monkeypatch):
    """A typo must fall back to torch detection, not crash the import."""
    utils = _load_utils(monkeypatch, cuda="12.4", forced="rocm")
    assert utils.ACCELERATOR_BACKEND == "cuda"


def test_detection_survives_missing_torch(monkeypatch):
    """utils must stay importable without torch for the CPU test suite."""
    monkeypatch.delenv("KVCACHED_BACKEND", raising=False)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    utils = importlib.reload(importlib.import_module("kvcached.utils"))
    assert utils.ACCELERATOR_BACKEND == "cuda"


def test_device_argument_overrides_the_process_backend(monkeypatch):
    """A caller that knows where it is allocating must win over the build guess.

    This is what lets one process drive two accelerator families, and what keeps
    a test that hands in a ``cuda`` device from being routed to ``torch.xpu``
    just because kvcached happens to be built for XPU.
    """
    utils = _load_utils(monkeypatch, xpu="20250302")
    assert utils.get_device_module("cuda:0") == "torch.cuda"
    assert utils.get_device_type("cuda:0") == "cuda"
    # The process-wide answer is still the build, for callers with no device.
    assert utils.ACCELERATOR_BACKEND == "xpu"
    assert utils.get_device_module() == "torch.xpu"

    utils = _load_utils(monkeypatch, cuda="12.4")
    assert utils.get_device_module("xpu:1") == "torch.xpu"
    assert utils.get_device_type("xpu:1") == "xpu"
    assert utils.get_device_module() == "torch.cuda"


def test_cuda_device_on_a_rocm_build_stays_hip(monkeypatch):
    """The one thing a device string cannot decide.

    PyTorch-ROCm spells AMD devices ``cuda``, so ``cuda:0`` must not demote a
    ROCm build to plain CUDA -- ``_default_contiguous_layout()`` reads the
    hip/cuda distinction and gets the layout wrong if it is lost.
    """
    utils = _load_utils(monkeypatch, hip="6.2.41133-dd7f95766")
    assert utils._detect_accelerator_backend("cuda:0") == "hip"
    assert utils.get_device_module("cuda:0") == "torch.cuda"
    assert utils.get_device_type("cuda:0") == "cuda"


@pytest.mark.parametrize("device", [
    "xpu",
    "xpu:0",
    "XPU:2",
    types.SimpleNamespace(type="xpu"),  # stands in for torch.device("xpu:0")
])
def test_xpu_device_forms_all_route_to_torch_xpu(monkeypatch, device):
    utils = _load_utils(monkeypatch, cuda="12.4")
    assert utils.get_device_module(device) == "torch.xpu"


@pytest.mark.parametrize("device", [None, 0, 3, "cpu", "meta", ""])
def test_devices_without_accelerator_type_fall_back_to_the_build(monkeypatch,
                                                                 device):
    """``None`` and a bare index carry no type; ``cpu``/``meta`` name no
    accelerator. All of them mean "use the process backend" rather than guessing."""
    utils = _load_utils(monkeypatch, xpu="20250302")
    assert utils.get_device_module(device) == "torch.xpu"

    utils = _load_utils(monkeypatch, cuda="12.4")
    assert utils.get_device_module(device) == "torch.cuda"


def test_hip_device_string_routes_to_torch_cuda(monkeypatch):
    """``hip:0`` reaches get_device_module() when a caller skips
    normalize_gpu_device(); it must still resolve to torch.cuda."""
    utils = _load_utils(monkeypatch, hip="6.2.41133-dd7f95766")
    assert utils.get_device_module("hip:0") == "torch.cuda"
    assert utils.get_device_type("hip:0") == "cuda"


def test_normalize_gpu_device_passes_xpu_through():
    """Rewriting xpu to cuda would misroute the allocation to the wrong device
    type; only hip is an alias for cuda."""
    from kvcached.utils import normalize_gpu_device

    assert normalize_gpu_device("xpu") == "xpu"
    assert normalize_gpu_device("xpu:0") == "xpu:0"
    assert normalize_gpu_device("xpu:3") == "xpu:3"
    # Existing behavior, unchanged.
    assert normalize_gpu_device("hip:1") == "cuda:1"
    assert normalize_gpu_device("cuda:2") == "cuda:2"


def test_is_gpu_device_str_accepts_all_backends():
    from kvcached.utils import is_gpu_device_str

    for device in ("cuda", "cuda:0", "hip", "hip:1", "xpu", "xpu:0", "XPU:0",
                   "xpu:11"):
        assert is_gpu_device_str(device), device
    for device in ("cpu", "cpu:0", "meta", "mps", ""):
        assert not is_gpu_device_str(device), device


def test_is_gpu_device_str_rejects_malformed_device_strings():
    """The device type is matched exactly. A prefix test would accept these and
    let the SGLang patch claim a device it cannot serve, failing later rather
    than declining up front."""
    from kvcached.utils import is_gpu_device_str

    for device in ("cudafoo", "cuda-device", "xpufoo", "hipster", "xpu_1",
                   "cuda:", "cuda:abc", "xpu:-1"):
        assert not is_gpu_device_str(device), device


def test_sglang_patch_accepts_xpu_device():
    """_is_supported_gpu_device gates the SGLang allocator patch: an
    unrecognized prefix makes it decline silently and SGLang keeps its own
    non-elastic allocator."""
    patches = pytest.importorskip("kvcached.integration.sglang.patches")

    assert patches._is_supported_gpu_device("xpu:0")
    assert patches._is_supported_gpu_device("cuda:0")
    assert patches._is_supported_gpu_device("hip:0")
    assert not patches._is_supported_gpu_device("cpu")

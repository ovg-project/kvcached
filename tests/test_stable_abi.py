# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
#
# Guard against accidental use of the unstable libtorch C++ API/ABI.

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
from pathlib import Path

import pytest

# Itanium-mangled unstable-namespace prefixes (no nm -C needed).
_UNSTABLE_SYMBOL = re.compile(
    r"""
    ^_ZN                          # start of a mangled nested name
    [rVK]*                        # optional cv-qualifiers (const/volatile/restrict)
    (?:
        2at                       # at::
      | 3c10                      # c10::
      | 5torch                    # torch:: ...
        (?!6stable|10headeronly)  # ... but not torch::stable / torch::headeronly
    )
    """,
    re.VERBOSE,
)

# Stable-ABI C shims added after the target version (setup.py STABLE_ABI_TARGET
# = 2.10). extern "C", so unmangled. Importing any means the binary won't load
# on 2.10. Canaries, not exhaustive; see torch/csrc/stable/c/shim.h.
_POST_MIN_STABLE_SHIMS = frozenset({
    "torch_from_blob",              # 2.11
    "torch_library_def_with_tags",  # 2.12
    "torch_exception_get_what",     # 2.13
    "torch_new_stable_ivalue",      # 2.13
    "torch_tensor_from_pyobject",   # 2.14
    "torch_tensor_to_pyobject",     # 2.14
})


@pytest.fixture(scope="session")
def extension_so() -> Path:
    """Locate kvcached/_C*.so without importing it; skip if not built."""
    matches = sorted((Path(__file__).resolve().parent.parent / "kvcached").glob("_C*.so"))
    if matches:
        return matches[0]
    try:
        spec = importlib.util.find_spec("kvcached._C")
    except ImportError:
        spec = None
    if spec and spec.origin and Path(spec.origin).exists():
        return Path(spec.origin)
    raise pytest.skip.Exception("kvcached._C extension is not built")


@pytest.fixture(scope="session")
def imported_symbols(extension_so) -> frozenset[str]:
    """Dynamic symbols the extension imports (nm -D -u), left mangled.

    Skips when nm is unavailable.
    """
    if shutil.which("nm") is None:
        pytest.skip("nm (binutils) not available")
    result = subprocess.run(
        ["nm", "-D", "-u", str(extension_so)],
        check=True,
        capture_output=True,
        text=True,
    )
    # Undefined line: "<blank addr> U <symbol>"; take the symbol.
    return frozenset(
        line.split()[-1] for line in result.stdout.splitlines() if line.split()
    )


def test_c_extension_exposes_pybind_classes(extension_so):
    pytest.importorskip("torch")
    from kvcached._C import InternalPage, PageAllocator  # noqa: F401


def test_extension_uses_only_stable_torch_abi(imported_symbols):
    offenders = sorted(s for s in imported_symbols if _UNSTABLE_SYMBOL.search(s))
    assert not offenders, "unstable libtorch symbols: " + ", ".join(offenders)


def test_extension_targets_minimum_torch_abi(imported_symbols):
    # A binary built with newer torch headers must not silently require stable
    # ABI symbols newer than the minimum targeted by TORCH_TARGET_VERSION.
    newer = sorted(imported_symbols & _POST_MIN_STABLE_SHIMS)
    assert not newer, (
        "extension imports stable-ABI symbols newer than the targeted minimum "
        "(setup.py STABLE_ABI_TARGET); rebuild with a matching "
        "TORCH_TARGET_VERSION: " + ", ".join(newer)
    )

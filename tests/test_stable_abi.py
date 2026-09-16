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


@pytest.fixture
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


def test_c_extension_exposes_pybind_classes(extension_so):
    pytest.importorskip("torch")
    from kvcached._C import InternalPage, PageAllocator  # noqa: F401


@pytest.mark.skipif(shutil.which("nm") is None, reason="nm (binutils) not available")
def test_extension_uses_only_stable_torch_abi(extension_so):
    # -D dynamic, -u undefined only (imported); names left mangled for portability.
    result = subprocess.run(
        ["nm", "-D", "-u", str(extension_so)],
        check=True,
        capture_output=True,
        text=True,
    )

    offenders = []
    for line in result.stdout.splitlines():
        tokens = line.split()  # undefined line: "<blank addr> U <symbol>"
        if tokens and _UNSTABLE_SYMBOL.search(tokens[-1]):
            offenders.append(tokens[-1])

    assert not offenders, "unstable libtorch symbols: " + ", ".join(offenders)

# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import os
import site
import warnings

# Version information
from importlib.metadata import version

try:
    __version__ = version("kvcached")
except Exception:
    # Fallback for development installations
    __version__ = "unknown"

# Ensure PyTorch is imported first before importing kvcached.vmm_ops
try:
    import torch  # noqa: F401
except ImportError as e:
    if "torch" in str(e):
        raise ImportError(
            "PyTorch is required for kvcached. Please install PyTorch first:\n"
            "  pip install torch>=2.6.0")
    else:
        raise

AUTOPATCH_PTH = "kvcached_autopatch.pth"


def _autopatch_requested() -> bool:
    return any(
        os.getenv(name, "false").lower() in ("true", "1")
        for name in ("ENABLE_KVCACHED", "KVCACHED_AUTOPATCH"))


def _autopatch_pth_installed() -> bool:
    """True if the .pth that registers the engine import hooks is in a
    directory whose .pth files the interpreter executes at startup."""
    site_dirs = []
    try:
        site_dirs.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        site_dirs.append(site.getusersitepackages())
    except Exception:
        pass
    return any(os.path.isfile(os.path.join(d, AUTOPATCH_PTH)) for d in site_dirs)


def _warn_if_autopatch_pth_missing() -> None:
    """The engines are patched by import hooks that kvcached_autopatch.pth
    registers at interpreter startup. Without it, ENABLE_KVCACHED and
    KVCACHED_AUTOPATCH do nothing and vLLM/SGLang run vanilla without any
    error (issue #470). Say so loudly whenever kvcached is imported with the
    knobs set but the .pth absent."""
    if not _autopatch_requested() or _autopatch_pth_installed():
        return
    warnings.warn(
        "ENABLE_KVCACHED/KVCACHED_AUTOPATCH is set but kvcached_autopatch.pth "
        "is not installed in site-packages, so vLLM/SGLang will not be patched "
        "and will run without kvcached. Reinstall kvcached (pip install -e . or "
        "a wheel) or run `python tools/dev_copy_pth.py` from a kvcached checkout.",
        RuntimeWarning,
        stacklevel=2,
    )


_warn_if_autopatch_pth_missing()

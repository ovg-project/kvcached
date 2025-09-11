# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

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

# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Helpers for loading an externally installed TriAttention package.

kvcached intentionally does not vendor TriAttention source.  Users should
install TriAttention separately and apply the small kvcached compatibility
patch shipped in ``engine_integration/patches/kvcached-triattention-main.patch``.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Callable

PATCH_HINT = (
    "Install TriAttention from https://github.com/WeianMao/triattention and "
    "apply engine_integration/patches/kvcached-triattention-main.patch from "
    "the TriAttention repository root."
)


def _import_external_module(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(
            "ENABLE_TRIATTENTION=1 requires the external 'triattention' "
            f"package. {PATCH_HINT}"
        ) from exc


def _require_attr(module: ModuleType, attr: str) -> None:
    if not hasattr(module, attr):
        raise RuntimeError(
            "The installed TriAttention package is missing kvcached "
            f"compatibility marker {module.__name__}.{attr}. {PATCH_HINT}"
        )


def install_external_vllm_triattention() -> None:
    """Install external TriAttention's vLLM hooks after patch validation."""

    integration = _import_external_module(
        "triattention.vllm.runtime.integration_monkeypatch"
    )
    hook_context = _import_external_module(
        "triattention.vllm.runtime.hook_runtime_context"
    )
    v1_backend = _import_external_module(
        "triattention.vllm.runtime.input_patch_vllm_v1_backend"
    )

    _require_attr(hook_context, "compression_anchored_effective_len")
    _require_attr(v1_backend, "_to_numpy")
    _require_attr(integration, "_assert_apc_disabled")

    install = getattr(integration, "install_vllm_integration_monkeypatches", None)
    if not callable(install):
        raise RuntimeError(
            "The installed TriAttention package does not expose "
            "install_vllm_integration_monkeypatches. "
            f"{PATCH_HINT}"
        )

    install(patch_scheduler=True, patch_worker=True)


def get_external_sglang_installer() -> Callable[[], None]:
    """Return external TriAttention's SGLang installer."""

    sglang = _import_external_module("triattention.sglang")
    install = getattr(sglang, "install_sglang_integration", None)
    if not callable(install):
        raise RuntimeError(
            "The installed TriAttention package does not expose "
            f"install_sglang_integration. {PATCH_HINT}"
        )
    return install

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

from kvcached.integration.version_utils import VersionRange
from kvcached.utils import get_kvcached_logger

logger = get_kvcached_logger()

PATCH_HINT = (
    "Install TriAttention from https://github.com/WeianMao/triattention and "
    "apply engine_integration/patches/kvcached-triattention-main.patch from "
    "the TriAttention repository root."
)

# TriAttention's vLLM integration unconditionally replaces
# ``GPUModelRunner._prepare_inputs`` with a wrapper that expects vLLM to pass
# ``num_scheduled_tokens`` as a positional argument. That calling convention
# only exists in vLLM >= 0.13.0. On vLLM 0.11.x and earlier the engine instead
# calls ``_prepare_inputs(scheduler_output)`` and TriAttention crashes on the
# first decode step with a cryptic
# ``TypeError: ... missing 1 required positional argument: 'num_scheduled_tokens'``.
# We gate activation on the version so users get a clear startup error instead.
_TRIATTENTION_MIN_VLLM = "0.13.0"
_TRIATTENTION_TESTED_VLLM = "0.21.0"


def _check_vllm_version_supported() -> None:
    """Fail fast (with a clear message) on a vLLM that TriAttention cannot patch."""

    try:
        import vllm

        detected = str(getattr(vllm, "__version__", "") or "")
    except Exception:
        # vLLM not importable here; the activation imports below will surface it.
        return
    if not detected:
        return

    if not VersionRange(f">={_TRIATTENTION_MIN_VLLM}").contains(detected):
        raise RuntimeError(
            f"TriAttention requires vLLM >= {_TRIATTENTION_MIN_VLLM}, but vLLM "
            f"{detected} is installed. vLLM 0.11.x and earlier call "
            "GPUModelRunner._prepare_inputs(scheduler_output) without "
            "'num_scheduled_tokens', so TriAttention's input patch crashes on "
            "the first decode step. Upgrade vLLM (validated through "
            f"{_TRIATTENTION_TESTED_VLLM}) or unset ENABLE_TRIATTENTION."
        )
    if not VersionRange(f"<={_TRIATTENTION_TESTED_VLLM}").contains(detected):
        logger.warning(
            "[TriAttention] vLLM %s is newer than the last validated version "
            "(%s); the TriAttention integration is untested here and may break.",
            detected,
            _TRIATTENTION_TESTED_VLLM,
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

    _check_vllm_version_supported()

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

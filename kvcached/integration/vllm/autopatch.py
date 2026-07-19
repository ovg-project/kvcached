# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import os
import types

from wrapt.importer import when_imported

from kvcached.integration.patch_base import PatchManager, log_patch_results
from kvcached.integration.vllm.nixl_compat import NixlConnectorPatch
from kvcached.integration.vllm.patches import (
    VLLM_ALL_RANGE,
    VLLM_V8_RANGE,
    VLLM_V9_PLUS_RANGE,
    ElasticBlockPoolPatch,
    EngineCorePatch,
    GPUModelRunnerPatch,
    GPUWorkerPatch,
    KVCacheCoordinatorPatch,
    KVCacheManagerPatch,
)
from kvcached.utils import get_kvcached_logger

logger = get_kvcached_logger()


def _env_enabled() -> bool:
    return os.getenv("KVCACHED_AUTOPATCH", "false").lower() in ("true", "1")


@when_imported("vllm")
def _patch_vllm(_vllm: types.ModuleType) -> None:
    if not _env_enabled():
        logger.debug("Disabled by KVCACHED_AUTOPATCH")
        return

    # Create patch manager and register version-specific vLLM patches
    patch_manager = PatchManager("vllm")

    patch_manager.register_patches_with_versions(
        [
            (NixlConnectorPatch(), VLLM_ALL_RANGE),
            (ElasticBlockPoolPatch(), VLLM_ALL_RANGE),
            (EngineCorePatch(), VLLM_ALL_RANGE),
            (GPUModelRunnerPatch(), VLLM_ALL_RANGE),
            (GPUWorkerPatch(), VLLM_ALL_RANGE),
            (KVCacheCoordinatorPatch(), VLLM_V9_PLUS_RANGE),
            (KVCacheManagerPatch(), VLLM_V8_RANGE),
        ]
    )

    # Apply all patches
    results = patch_manager.apply_all_patches()

    # Log results
    log_patch_results("vllm", results)

    # Optional TriAttention layer. Install after kvcached patches so
    # TriAttention sees kvcached-backed vLLM scheduler and block pool state.
    if os.getenv("ENABLE_TRIATTENTION", "0").lower() in ("true", "1"):
        try:
            from kvcached.integration.triattention_external import (
                install_external_vllm_triattention,
            )

            install_external_vllm_triattention()
            logger.info(
                "[TriAttention] Runtime plugin activated on top of kvcached "
                "(ENABLE_TRIATTENTION=1)."
            )
        except Exception as exc:
            logger.error(
                "[TriAttention] Activation failed: %s: %s",
                type(exc).__name__, exc,
            )
            raise RuntimeError(
                "ENABLE_TRIATTENTION=1 was requested, but TriAttention "
                "vLLM activation failed. Check the external triattention "
                "package, PYTHONPATH/--triattention-root, and the kvcached "
                "compatibility patch."
            ) from exc

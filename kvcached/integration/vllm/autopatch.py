# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import os
import types

from wrapt.importer import when_imported

from kvcached.integration.patch_base import PatchManager, log_patch_results
from kvcached.integration.vllm.model_runner_v2 import KVLayoutV2Patch, ModelRunnerV2Patch
from kvcached.integration.vllm.nixl_compat import NixlConnectorPatch
from kvcached.integration.vllm.patches import (
    VLLM_ALL_RANGE,
    VLLM_MRV2_RANGE,
    VLLM_V8_RANGE,
    VLLM_V9_PLUS_RANGE,
    CoreEngineProcManagerPatch,
    ElasticBlockPoolPatch,
    EngineCorePatch,
    GPUModelRunnerPatch,
    GPUWorkerPatch,
    KVCacheCoordinatorPatch,
    KVCacheManagerAllocateSlotsPatch,
    KVCacheManagerPatch,
    MPClientPatch,
    TritonAttentionPatch,
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
            (MPClientPatch(), VLLM_ALL_RANGE),
            # CoreEngineProcManager lives in vllm.v1.engine.utils from 0.10
            (CoreEngineProcManagerPatch(), ">=0.10.0"),
            (GPUModelRunnerPatch(), ">=0.8.4,<0.29.0"),
            (ModelRunnerV2Patch(), VLLM_MRV2_RANGE),
            (KVLayoutV2Patch(), VLLM_MRV2_RANGE),
            (GPUWorkerPatch(), VLLM_ALL_RANGE),
            (KVCacheCoordinatorPatch(), VLLM_V9_PLUS_RANGE),
            (KVCacheManagerPatch(), VLLM_V8_RANGE),
            (KVCacheManagerAllocateSlotsPatch(), VLLM_ALL_RANGE),
            (TritonAttentionPatch(), ">=0.9.0,<0.29.0"),
        ]
    )

    # Apply all patches
    results = patch_manager.apply_all_patches()

    # Log results
    log_patch_results("vllm", results)

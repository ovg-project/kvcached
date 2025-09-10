# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import importlib
import os
import types
from typing import Callable

from kvcached.integration.patch_base import PatchManager, log_patch_results
from kvcached.integration.vllm.patches import (
    ElasticBlockPoolPatch,
    EngineCorePatch,
    GPUModelRunnerPatch,
    GPUWorkerPatch,
    KVCacheCoordinatorPatch,
)
from kvcached.utils import get_kvcached_logger

try:
    from wrapt.importer import when_imported  # type: ignore
except Exception:

    def when_imported(module_name: str):  # type: ignore

        def decorator(func: Callable[[types.ModuleType], None]):
            try:
                import importlib
                mod = importlib.import_module(module_name)
            except Exception:
                return func
            func(mod)
            return func

        return decorator


logger = get_kvcached_logger()


def _env_enabled() -> bool:
    return os.getenv("KVCACHED_AUTOPATCH", "false").lower() in ("true", "1")


@when_imported("vllm")
def _patch_vllm(_vllm: types.ModuleType) -> None:
    if not _env_enabled():
        logger.debug("Disabled by KVCACHED_AUTOPATCH")
        return

    # Create patch manager and register all vLLM patches
    patch_manager = PatchManager("vllm")
    patch_manager.register_patches([
        ElasticBlockPoolPatch(),
        EngineCorePatch(),
        KVCacheCoordinatorPatch(),
        GPUModelRunnerPatch(),
        GPUWorkerPatch(),
    ])

    # Apply all patches
    results = patch_manager.apply_all_patches()

    # Log results
    log_patch_results("vllm", results)

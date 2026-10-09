# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import os
import types

from wrapt.importer import when_imported

from kvcached.integration.patch_base import (
    PatchManager,
    is_integration_version_supported,
    log_patch_results,
)
from kvcached.integration.sglang.patches import (
    SGLANG_ALL_RANGE,
    ElasticAllocatorPatch,
    ElasticHybridLinearKVPoolPatch,
    ElasticMambaPoolPatch,
    ElasticMemoryPoolPatch,
    ElasticMLAMemoryPoolPatch,
    ElasticSWAAllocatorPatch,
    MambaRadixCacheLimitPatch,
    RadixCacheLimitPatch,
    ScheduleBatchCapacityMissPatch,
    SchedulerCapacityMissPatch,
    SchedulerMemoryLeakPatch,
    SGLangLegacyVirtualKVCapacityPatch,
    SGLangVirtualKVCapacityPatch,
    SWARadixCacheLimitPatch,
    UnifiedRadixCacheLimitPatch,
)
from kvcached.utils import get_kvcached_logger

logger = get_kvcached_logger()


def _env_enabled() -> bool:
    return os.getenv("KVCACHED_AUTOPATCH", "false").lower() in ("true", "1")


@when_imported("sglang")
def _patch_sglang(_sglang: types.ModuleType) -> None:
    if not _env_enabled():
        logger.debug("Disabled by KVCACHED_AUTOPATCH")
        return

    if not is_integration_version_supported("sglang", SGLANG_ALL_RANGE):
        return

    # Create patch manager and register version-specific SGLang patches
    patch_manager = PatchManager("sglang")

    patch_manager.register_patches_with_versions(
        [
            (ElasticAllocatorPatch(), SGLANG_ALL_RANGE),
            # SWATokenToKVPoolAllocator captures allocator classes from its
            # implementation modules, not from the package aliases above.
            (ElasticSWAAllocatorPatch(), ">=0.5.13"),
            (ElasticMemoryPoolPatch(), SGLANG_ALL_RANGE),
            (ElasticMLAMemoryPoolPatch(), SGLANG_ALL_RANGE),
            (ElasticMambaPoolPatch(), SGLANG_ALL_RANGE),
            (ElasticHybridLinearKVPoolPatch(), SGLANG_ALL_RANGE),
            # Importing the capacity owner captures memory-pool classes in
            # module globals, so apply these only after every pool alias.
            (SGLangLegacyVirtualKVCapacityPatch(), ">=0.5.11,<0.5.16"),
            (SGLangVirtualKVCapacityPatch(), ">=0.5.16"),
            (SchedulerMemoryLeakPatch(), SGLANG_ALL_RANGE),
            # A kvcached pool can miss after admission. These turn the
            # miss into a retried prefill instead of a scheduler exit
            # (#547); they read the 0.5.20 batch and request shapes.
            (ScheduleBatchCapacityMissPatch(), ">=0.5.20"),
            (SchedulerCapacityMissPatch(), ">=0.5.20"),
            (RadixCacheLimitPatch(), SGLANG_ALL_RANGE),
            # Prefix caches that are not RadixCache subclasses.
            (UnifiedRadixCacheLimitPatch(), ">=0.5.13"),
            (SWARadixCacheLimitPatch(), ">=0.5.13"),
            (MambaRadixCacheLimitPatch(), ">=0.5.13"),
        ]
    )

    # Apply all patches
    results = patch_manager.apply_all_patches()

    # Log results
    log_patch_results("sglang", results)

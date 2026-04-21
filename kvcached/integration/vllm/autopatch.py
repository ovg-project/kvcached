# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import os
import types

from wrapt.importer import when_imported

from kvcached.integration.patch_base import PatchManager, log_patch_results
from kvcached.integration.patch_base import enable_kvcached
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

    # Patch NixlConnector for kvcached compatibility (two bugs).
    # Done eagerly here because get_required_kvcache_layout() is a classmethod
    # called during create_engine_config(), before deferred module patches fire.
    _patch_nixl_connector()

    # Log results
    log_patch_results("vllm", results)


def _patch_nixl_connector() -> None:
    """Patch NixlConnector for kvcached PD disaggregation compatibility.

    Bug 1: NixlConnector forces HND layout, but kvcached's from_blob tensors
    don't support set_stride (needed for NHD->HND permutation).
    Fix: override get_required_kvcache_layout() to return None (use NHD).

    Bug 2: NixlConnectorWorker.register_kv_caches asserts
    tensor.shape[blocks_dim] == self.num_blocks. kvcached allocates a larger
    virtual blocks dimension (_num_blocks_per_layer) than vLLM's physical
    budget (self.num_blocks), so the assertion fires.
    Fix: rewrite self.num_blocks to match _num_blocks_per_layer before the
    original register_kv_caches runs.
    """
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
            NixlConnector,
            NixlConnectorWorker,
        )
    except ImportError:
        return  # NIXL not installed

    # Bug 1: force NHD layout
    @classmethod  # type: ignore[misc]
    def _kvcached_layout(cls, *args, **kwargs):
        if not enable_kvcached():
            return NixlConnector._original_get_layout(*args, **kwargs)
        logger.info("kvcached: NixlConnector layout overridden to NHD")
        return None

    NixlConnector._original_get_layout = NixlConnector.get_required_kvcache_layout
    NixlConnector.get_required_kvcache_layout = _kvcached_layout

    # Bug 2: relax block count assertion
    _original_register = NixlConnectorWorker.register_kv_caches

    def _patched_register(self, kv_caches, *args, **kwargs):
        if not enable_kvcached():
            return _original_register(self, kv_caches, *args, **kwargs)

        from kvcached.integration.vllm.interfaces import _num_blocks_per_layer
        if _num_blocks_per_layer > self.num_blocks:
            logger.info(
                "kvcached: NixlConnector num_blocks %d -> %d",
                self.num_blocks, _num_blocks_per_layer,
            )
            self.num_blocks = _num_blocks_per_layer

        return _original_register(self, kv_caches, *args, **kwargs)

    NixlConnectorWorker.register_kv_caches = _patched_register
    logger.info("Patched NixlConnector for kvcached PD disagg compatibility")

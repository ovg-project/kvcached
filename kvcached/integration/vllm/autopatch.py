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

    Bug 1: NixlConnector asserts tensor shape[0] == num_blocks. kvcached's
    tensors have a larger blocks dimension (virtual capacity vs physical budget).
    Fix: set num_blocks to match the tensor's blocks dimension before the
    assertion runs.

    Bug 2: NixlConnector forces HND layout, but kvcached's from_blob tensors
    don't support set_stride (needed for NHD->HND permutation).
    Fix: override get_required_kvcache_layout() to return None (use NHD).
    """
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
            NixlConnector,
            NixlConnectorWorker,
        )
    except ImportError:
        return  # NIXL not installed

    # Bug 2: force NHD layout
    @classmethod  # type: ignore[misc]
    def _kvcached_layout(cls, *args, **kwargs):
        if not enable_kvcached():
            return NixlConnector._original_get_layout(*args, **kwargs)
        logger.info("kvcached: NixlConnector layout overridden to NHD")
        return None

    NixlConnector._original_get_layout = NixlConnector.get_required_kvcache_layout
    NixlConnector.get_required_kvcache_layout = _kvcached_layout

    # Bug 1: relax block count assertion
    _original_register = NixlConnectorWorker.register_kv_caches

    def _patched_register(self, kv_caches, *args, **kwargs):
        if not enable_kvcached():
            return _original_register(self, kv_caches, *args, **kwargs)

        from kvcached.integration.vllm.interfaces import _num_blocks_per_layer
        num_blocks_before = self.num_blocks
        if _num_blocks_per_layer > self.num_blocks:
            logger.info(
                "kvcached: NixlConnector num_blocks %d -> %d",
                self.num_blocks, _num_blocks_per_layer,
            )
            self.num_blocks = _num_blocks_per_layer

        from kvcached.integration.vllm.debug_layout import dump_kv_layout
        tensors_for_dump = list(kv_caches.values()) if hasattr(kv_caches, "values") else list(kv_caches)
        dump_kv_layout(
            tag="nixl_register_pre",
            kv_tensors=tensors_for_dump,
            extra={
                "num_blocks_before": num_blocks_before,
                "num_blocks_after": self.num_blocks,
            },
        )
        self._kvcached_kv_caches = tensors_for_dump
        result = _original_register(self, kv_caches, *args, **kwargs)
        dump_kv_layout(
            tag="nixl_register_post",
            kv_tensors=tensors_for_dump,
            blocks_dim_idx=_guess_blocks_dim_idx(tensors_for_dump),
            extra={"num_blocks_after": self.num_blocks},
            sha_blocks=(0, 2),
        )
        return result

    NixlConnectorWorker.register_kv_caches = _patched_register

    _wrap_nixl_lifecycle_hooks(NixlConnectorWorker)

    logger.info("Patched NixlConnector for kvcached PD disagg compatibility")


def _guess_blocks_dim_idx(kv_tensors) -> int:
    """Best-effort recovery of blocks_dim_idx from tensor shape.

    Mirrors the logic in interfaces.alloc_kv_cache: FlashAttn puts K/V
    count at dim 0 (blocks at dim 1); FlashInfer/MLA puts blocks at dim 0.
    """
    if not kv_tensors:
        return 0
    shape = list(kv_tensors[0].shape)
    # FlashAttn: first dim is 2 (or 1 for MLA K/V-combined) and small.
    if len(shape) >= 2 and shape[0] <= 2:
        return 1
    return 0


def _wrap_nixl_lifecycle_hooks(worker_cls) -> None:
    """Introspect NixlConnectorWorker and wrap known lifecycle methods
    with content-SHA dumps. Gated at runtime on KVCACHED_DUMP_LAYOUT=1.

    We don't know which methods exist on every vLLM version, so we log
    the full list of save/recv/load/send/wait/start/finish methods
    we find, then wrap each with a generic around-hook that logs SHAs
    of layer-0 block 0 and block 1 at entry and exit. Runtime logs then
    reveal which hooks fired for prefill vs decode.
    """
    import json as _json

    # get_finished is excluded: it's polled every scheduler tick and would
    # spam hundreds of redundant SHA lines per request.
    candidate_patterns = (
        "save_kv_layer",
        "save_kv_caches",
        "send_kv_caches",
        "recv_kv_caches",
        "load_kv_caches",
        "start_load_kv",
        "wait_for_save",
        "wait_for_load",
    )

    found = sorted(
        name for name in dir(worker_cls)
        if any(pat in name for pat in candidate_patterns)
        and callable(getattr(worker_cls, name, None))
        and not name.startswith("_")
    )
    logger.info("KVCACHED_NIXL_INTROSPECT=%s", _json.dumps({"methods": found}))

    for name in found:
        _install_content_hook(worker_cls, name)


def _install_content_hook(worker_cls, method_name: str) -> None:
    original = getattr(worker_cls, method_name)

    def _hooked(self, *args, **kwargs):
        from kvcached.integration.vllm.debug_layout import (
            _enabled, dump_active_blocks_sha,
        )
        kv_tensors = getattr(self, "_kvcached_kv_caches", None)
        if _enabled() and kv_tensors:
            blocks_dim_idx = _guess_blocks_dim_idx(kv_tensors)
            dump_active_blocks_sha(
                tag=f"{method_name}_enter",
                kv_tensors=kv_tensors,
                blocks_dim_idx=blocks_dim_idx,
                block_ids=(0, 1),
            )
        result = original(self, *args, **kwargs)
        if _enabled() and kv_tensors:
            blocks_dim_idx = _guess_blocks_dim_idx(kv_tensors)
            dump_active_blocks_sha(
                tag=f"{method_name}_exit",
                kv_tensors=kv_tensors,
                blocks_dim_idx=blocks_dim_idx,
                block_ids=(0, 1),
            )
        return result

    _hooked.__name__ = method_name
    setattr(worker_cls, method_name, _hooked)

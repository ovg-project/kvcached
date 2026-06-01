# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Compatibility patches for vLLM's NixlConnector."""

import importlib
import types
from typing import Any, Iterator, Optional, Tuple

from kvcached.integration.patch_base import BasePatch, enable_kvcached
from kvcached.utils import CONTIGUOUS_LAYOUT


class NixlConnectorPatch(BasePatch):
    """Eager NixlConnector compatibility patch.

    The target is the already-imported vLLM module because NixlConnector's
    layout classmethod can be consulted during engine config creation, before a
    deferred module-specific import hook would be reliable.
    """

    library = "vllm"
    target_module = "vllm"
    patch_name = "nixl_connector_compat"

    # vLLM moved NixlConnectorWorker from the single nixl_connector module into
    # a split nixl.connector / nixl.worker package layout. Try both so this
    # eager patch keeps working across the supported open-ended vLLM range.
    _CONNECTOR_MODULES = (
        (
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector",
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector",
        ),
        (
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector",
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker",
        ),
    )

    def apply(self, target_module: types.ModuleType) -> bool:
        return self.patch_connector()

    def patch_connector(self) -> bool:
        """Patch NixlConnector for kvcached PD disaggregation compatibility.

        Bug 1: NixlConnector forces HND layout, but kvcached's from_blob tensors
        don't support set_stride (needed for NHD->HND permutation).
        Fix: override get_required_kvcache_layout() to return None (use NHD).

        Bug 1a: kvcached's default contiguous layout interleaves physical pages
        across layers/K/V buffers, while vLLM 0.10.2's NixlConnector assumes
        each layer's K and V block regions are independently block-contiguous.
        Fix: require KVCACHED_CONTIGUOUS_LAYOUT=false for NIXL until the worker
        descriptor builder learns kvcached's interleaved page stride.

        Bug 2: NixlConnectorWorker.register_kv_caches expects self.num_blocks to
        match the registered tensor block dimension. kvcached can allocate a
        different block count from vLLM's initial physical budget, and vLLM may
        split logical blocks into smaller kernel blocks.
        Fix: rewrite self.num_blocks to match the KV tensors about to be
        registered before the original register_kv_caches runs. If registered
        tensors disagree on block count, fail before NIXL publishes inconsistent
        transfer metadata.
        """
        nixl_classes = self._import_nixl_connector_classes()
        if nixl_classes is None:
            self.logger.debug(
                "Skipping NixlConnector patch: NIXL connector not installed"
            )
            return True

        NixlConnector, NixlConnectorWorker = nixl_classes
        patch = self

        @classmethod  # type: ignore[misc]
        def _kvcached_layout(cls, *args, **kwargs):
            if not enable_kvcached():
                return NixlConnector._original_get_layout(*args, **kwargs)
            patch._ensure_supported_kvcached_layout()
            patch.logger.info("kvcached: NixlConnector layout overridden to NHD")
            return None

        # get_required_kvcache_layout only exists on newer vLLM (added to the
        # connector base ~v0.10.1). On older versions NixlConnector does not
        # force a layout, so NHD is already used and there is nothing to
        # override; guard with hasattr so the patch does not AttributeError.
        if hasattr(NixlConnector, "get_required_kvcache_layout"):
            if not hasattr(NixlConnector, "_original_get_layout"):
                NixlConnector._original_get_layout = (
                    NixlConnector.get_required_kvcache_layout
                )
            NixlConnector.get_required_kvcache_layout = _kvcached_layout
        else:
            self.logger.debug(
                "NixlConnector has no get_required_kvcache_layout on this vLLM "
                "version; skipping layout override (NHD already in use)"
            )

        if not hasattr(NixlConnectorWorker, "_kvcached_original_register_kv_caches"):
            NixlConnectorWorker._kvcached_original_register_kv_caches = (
                NixlConnectorWorker.register_kv_caches
            )

        def _patched_register(worker, kv_caches, *args, **kwargs):
            _original_register = NixlConnectorWorker._kvcached_original_register_kv_caches
            if not enable_kvcached():
                return _original_register(worker, kv_caches, *args, **kwargs)
            patch._ensure_supported_kvcached_layout()

            kvcached_num_blocks = patch._infer_registered_num_blocks(worker, kv_caches)
            if (
                kvcached_num_blocks is not None
                and kvcached_num_blocks != worker.num_blocks
            ):
                patch.logger.info(
                    "kvcached: NixlConnector num_blocks %d -> %d",
                    worker.num_blocks, kvcached_num_blocks,
                )
                worker.num_blocks = kvcached_num_blocks

            return _original_register(worker, kv_caches, *args, **kwargs)

        NixlConnectorWorker.register_kv_caches = _patched_register
        self.logger.info("Patched NixlConnector for kvcached PD disagg compatibility")
        return True

    def _import_nixl_connector_classes(self) -> Optional[Tuple[Any, Any]]:
        for connector_module_name, worker_module_name in self._CONNECTOR_MODULES:
            try:
                connector_module = importlib.import_module(connector_module_name)
                worker_module = importlib.import_module(worker_module_name)
            except ImportError:
                continue

            NixlConnector = getattr(connector_module, "NixlConnector", None)
            NixlConnectorWorker = getattr(worker_module, "NixlConnectorWorker", None)
            if NixlConnector is not None and NixlConnectorWorker is not None:
                return NixlConnector, NixlConnectorWorker

        return None

    def _ensure_supported_kvcached_layout(self) -> None:
        if not CONTIGUOUS_LAYOUT:
            return
        raise RuntimeError(
            "kvcached NixlConnector requires KVCACHED_CONTIGUOUS_LAYOUT=false. "
            "The default contiguous layout interleaves physical pages across "
            "layers and K/V buffers, but vLLM's NixlConnector currently "
            "registers each layer's K/V regions as block-contiguous memory."
        )

    def _iter_kv_cache_tensors(self, value: Any) -> Iterator[Any]:
        if hasattr(value, "shape"):
            yield value
            return

        if isinstance(value, (list, tuple)):
            for item in value:
                yield from self._iter_kv_cache_tensors(item)

    def _infer_cache_num_blocks(self, worker: Any, cache: Any) -> Optional[int]:
        shape = tuple(getattr(cache, "shape", ()))
        if not shape:
            return None

        # MLA and FlashInfer expose the block count in dim 0. NixlConnectorWorker
        # reports the attention backend via ``backend_name`` (set from
        # ``attn_backends[0].get_name()``); there is no ``_use_flashinfer`` attr.
        use_mla = bool(getattr(worker, "use_mla", False))
        backend_name = (getattr(worker, "backend_name", "") or "").upper()
        use_flashinfer = "FLASHINFER" in backend_name
        if use_mla or use_flashinfer:
            return int(shape[0])

        # FlashAttn family stacks K/V in dim 0 and keeps blocks in dim 1, e.g.
        # (2, num_blocks, block_size, num_kv_heads, head_size).
        if len(shape) >= 5 and shape[0] == 2:
            return int(shape[1])

        return int(shape[0])

    def _infer_registered_num_blocks(self, worker: Any, kv_caches: Any) -> Optional[int]:
        counts = []
        for value in kv_caches.values():
            for cache in self._iter_kv_cache_tensors(value):
                count = self._infer_cache_num_blocks(worker, cache)
                if count is not None and count > 0:
                    counts.append(count)
        if not counts:
            return None
        unique_counts = set(counts)
        if len(unique_counts) != 1:
            raise RuntimeError(
                "kvcached: NixlConnector saw inconsistent KV block counts: "
                f"{sorted(unique_counts)}"
            )
        return counts[0]


def patch_nixl_connector() -> bool:
    return NixlConnectorPatch().patch_connector()

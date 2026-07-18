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

    # Class-level signal from the register wrapper (_reconcile_blocks_first_layout)
    # to the TransferTopology.__init__ hook: True iff the wrapper just transposed a
    # split-half tensor and the topology should force split_k_and_v. Kept at class
    # scope (not per-instance) because patch_nixl_connector() may be called multiple
    # times, so the wrapper and the topology hook can close over different
    # NixlConnectorPatch instances; a shared class attribute keeps them in lockstep.
    _pending_force_split: bool = False

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

        # NixlConnectorWorker resolves to NixlPullConnectorWorker, which inherits
        # register_kv_caches from the base worker. NixlPushConnectorWorker overrides
        # it but delegates to the base via super().register_kv_caches(). Patch the
        # class that actually *defines* register_kv_caches so BOTH pull and push
        # modes are covered -- patching the pull subclass alone would leave push
        # silently unpatched (and thus silently corrupting).
        register_owner = NixlConnectorWorker
        for _cls in NixlConnectorWorker.__mro__:
            if "register_kv_caches" in _cls.__dict__:
                register_owner = _cls
                break

        if not hasattr(register_owner, "_kvcached_original_register_kv_caches"):
            register_owner._kvcached_original_register_kv_caches = (
                register_owner.register_kv_caches
            )

        def _patched_register(worker, kv_caches, *args, **kwargs):
            _original_register = register_owner._kvcached_original_register_kv_caches
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

            # Reconcile kvcached's split-half KV memory with vLLM >=0.23's
            # blocks-first NIXL descriptor assumptions (see method docstring).
            kv_caches = patch._reconcile_blocks_first_layout(worker, kv_caches)

            return _original_register(worker, kv_caches, *args, **kwargs)

        # Install the register wrapper only once. Re-wrapping on every
        # patch_nixl_connector() call is redundant (the wrapper always calls the
        # once-saved original) and would churn the closure for no benefit.
        if not getattr(register_owner, "_kvcached_register_patched", False):
            register_owner.register_kv_caches = _patched_register
            register_owner._kvcached_register_patched = True

        self._patch_transfer_topology()
        self.logger.info("Patched NixlConnector for kvcached PD disagg compatibility")
        return True

    def _patch_transfer_topology(self) -> None:
        """Force NIXL's split_k_and_v path for kvcached's split-half KV layout.

        vLLM >=0.23 blocks-first backends make TransferTopology report
        is_kv_layout_blocks_first=True (K/V interleaved within a block), but
        kvcached lays K and V out as two separate block-contiguous regions. Paired
        with the tensor transpose in _reconcile_blocks_first_layout, we clear that
        flag (-> split_k_and_v=True, virtually_split=False) so NIXL registers K and
        V as separate regions matching kvcached's real memory. See
        _reconcile_blocks_first_layout for the full rationale.

        The topology is rebuilt inside register_kv_caches, so this patches the
        class __init__ rather than a per-worker instance. The flip is layout-driven,
        NOT unconditional: it only fires when _reconcile_blocks_first_layout actually
        transposed a split-half tensor for this registration (signalled via
        ``_pending_force_split``). If kvcached's layout ever matches NIXL natively
        (interleaved / K-V-first), the wrapper leaves the tensors alone and does not
        set the flag, so NIXL's native path is kept. MLA / mamba topologies and
        non-kvcached runs are untouched.
        """
        try:
            from vllm.distributed.kv_transfer.kv_connector import utils as _kv_utils
        except ImportError:
            return
        TransferTopology = getattr(_kv_utils, "TransferTopology", None)
        if TransferTopology is None:
            return
        if getattr(TransferTopology, "_kvcached_topology_patched", False):
            return

        _orig_init = TransferTopology.__init__

        def _patched_init(topo_self, *args, **kwargs):
            _orig_init(topo_self, *args, **kwargs)
            if not enable_kvcached():
                return
            # Only flip when the paired register wrapper transposed a split-half
            # tensor for this registration (it runs first and sets the signal on
            # `patch`). Consume it so it can't leak into an unrelated topology.
            if not NixlConnectorPatch._pending_force_split:
                return
            NixlConnectorPatch._pending_force_split = False
            # MLA / mamba topologies use different split logic; never flip them.
            if getattr(topo_self, "is_mla", False) or getattr(topo_self, "is_mamba", False):
                return
            if getattr(topo_self, "_is_kv_layout_blocks_first", False):
                topo_self._is_kv_layout_blocks_first = False

        TransferTopology.__init__ = _patched_init
        TransferTopology._kvcached_topology_patched = True

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

    def _reconcile_blocks_first_layout(self, worker: Any, kv_caches: Any) -> Any:
        """Match kvcached's split-half KV memory to vLLM >=0.23 blocks-first NIXL.

        vLLM >=0.23 FlashAttention/FlashInfer/Triton/Flex use a *blocks-first* KV
        shape ``(num_blocks, 2, block_size, H, D)`` (vLLM PR #42095, RFC #42082).
        NIXL's TransferTopology then sets ``is_kv_layout_blocks_first=True`` and
        assumes K and V are *interleaved within each block*: per-block stride =
        full page (2*hidden), V at intra-block offset +hidden.

        kvcached (non-contiguous MHA/GQA) instead lays each layer out *split-half*:
        all K blocks contiguous in one region, all V blocks in a separate region
        far away; per-block stride = hidden. NIXL's per-block stride is therefore
        exactly 2x kvcached's real stride and its within-block V descriptor lands
        inside kvcached's K region -> every transferred block is misaligned and V
        is never transferred -> garbled decode. (Single-instance attention is fine
        because the kernel honors the tensor's real strides; only NIXL's raw-offset
        RDMA ignores them.)

        Fix: present each split-half attention tensor to NIXL as a K/V-first
        ``(2, num_blocks, ...)`` view (transpose dims 0 and 1) and force the
        ``split_k_and_v`` registration path (``_is_kv_layout_blocks_first=False``),
        so NIXL registers K and V as two separate block-contiguous regions whose
        base addresses / block_len exactly match kvcached's real memory. Returns a
        (possibly transposed) kv_caches dict.

        This is layout-driven, not assumption-driven:
          * split-half tensor  -> transpose + signal the paired topology patch to
            force split_k_and_v (the current kvcached non-contiguous MHA case);
          * already interleaved / K-V-first tensor (matches NIXL natively) -> left
            untouched, no signal, NIXL keeps its native path (future-proofs a
            kvcached layout change);
          * anything else       -> fail loud rather than silently corrupting.

        The actual topology flip lives in the paired TransferTopology.__init__
        patch, because the topology is (re)constructed *inside* register_kv_caches,
        after this wrapper runs; we hand it the decision via ``_pending_force_split``.
        """
        fixed: dict = {}
        saw_split_half = False
        for name, value in kv_caches.items():
            t = value
            # Blocks-first attention tensors are (num_blocks, 2, block, H, D)
            # (shape[1]==2). Older K/V-first tensors are (2, num_blocks, ...) and
            # won't match here; mamba/MLA tensors have different shapes. All left
            # untouched unless they are blocks-first attention.
            if not (hasattr(t, "dim") and t.dim() == 5 and int(t.shape[1]) == 2):
                fixed[name] = value
                continue
            hidden = 1
            for d in t.shape[2:]:
                hidden *= int(d)
            block_stride = int(t.stride()[0])
            if block_stride == 2 * hidden:
                # Interleaved/contiguous already: matches NIXL, nothing to fix.
                fixed[name] = value
                continue
            if block_stride != hidden:
                raise RuntimeError(
                    "kvcached NixlConnector: unexpected attention KV block stride "
                    f"(layer={name}, shape={tuple(t.shape)}, stride={tuple(t.stride())}); "
                    "expected split-half (block stride == K-or-V hidden size = "
                    f"{hidden}). Refusing to register a layout NIXL would corrupt."
                )
            # Split-half confirmed: (num_blocks, 2, ...) -> (2, num_blocks, ...) view.
            # view[0] = all K blocks (base = K region), view[1] = all V blocks
            # (base = V region); each is block-contiguous with stride = hidden,
            # exactly what the split_k_and_v register path expects.
            fixed[name] = t.transpose(0, 1)
            saw_split_half = True

        # Signal the paired TransferTopology.__init__ patch whether to force the
        # split_k_and_v path for the topology it is about to build. Only True when
        # we actually transposed a split-half tensor, so the flip stays in lockstep
        # with the transpose (interleaved/native layouts keep NIXL's default path).
        NixlConnectorPatch._pending_force_split = saw_split_half

        if not saw_split_half:
            return kv_caches

        if getattr(worker, "_has_mamba", False):
            NixlConnectorPatch._pending_force_split = False
            raise RuntimeError(
                "kvcached NixlConnector: PD disaggregation is not supported for "
                "hybrid/mamba models on blocks-first vLLM (>=0.23). Refusing to "
                "register a layout that would silently corrupt KV transfers."
            )

        self.logger.info(
            "kvcached: reconciled blocks-first KV layout -> split_k_and_v "
            "(K/V registered as separate regions to match kvcached split-half memory)"
        )
        return fixed

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

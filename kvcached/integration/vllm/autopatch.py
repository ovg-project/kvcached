# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import os
import time
import types
from typing import Any

from wrapt.importer import when_imported

from kvcached.integration.patch_base import PatchManager, log_patch_results
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


def _truthy_env(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in ("true", "1", "yes", "on")


def _p2p_trace_tensors_enabled() -> bool:
    return _truthy_env("KVCACHED_P2P_TRACE_TENSORS", "true")


def _p2p_wait_log_interval_s() -> float:
    raw = os.getenv("KVCACHED_P2P_WAIT_LOG_INTERVAL_S", "5")
    try:
        interval = float(raw)
    except ValueError:
        logger.warning(
            "kvcached p2p debug: invalid KVCACHED_P2P_WAIT_LOG_INTERVAL_S=%r; using 5s",
            raw,
        )
        return 5.0
    return interval if interval > 0 else 5.0


def _safe_len(value: Any) -> Any:
    try:
        return len(value)
    except Exception:
        return "unknown"


def _safe_getattr(value: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(value, name, default)
    except Exception:
        return default


def _safe_call(value: Any, name: str) -> Any:
    method = _safe_getattr(value, name)
    if method is None:
        return "unavailable"
    try:
        return method()
    except Exception as exc:
        return f"error:{type(exc).__name__}"


def _p2p_tensor_summary(tensor: Any) -> str:
    if tensor is None:
        return "tensor=None"
    if not _p2p_trace_tensors_enabled():
        return "tensor=<trace-disabled>"

    shape = _safe_getattr(tensor, "shape", "unavailable")
    dtype = _safe_getattr(tensor, "dtype", "unavailable")
    device = _safe_getattr(tensor, "device", "unavailable")
    numel = _safe_call(tensor, "numel")
    contiguous = _safe_call(tensor, "is_contiguous")
    stride = _safe_call(tensor, "stride")
    return (
        f"shape={shape} dtype={dtype} device={device} "
        f"numel={numel} contiguous={contiguous} stride={stride}"
    )


def _p2p_block_count(block_ids: Any) -> Any:
    if block_ids is None:
        return 0
    return _safe_len(block_ids)


def _p2p_request_summary(request: Any) -> str:
    request_id = _safe_getattr(request, "request_id", _safe_getattr(request, "req_id", "?"))
    block_ids = _safe_getattr(request, "block_ids", None)
    num_tokens = _safe_getattr(request, "num_tokens", "?")
    if isinstance(block_ids, (list, tuple)) and block_ids and isinstance(block_ids[0], list):
        block_count = [len(group) for group in block_ids]
    else:
        block_count = _p2p_block_count(block_ids)
    return f"request_id={request_id} tokens={num_tokens} blocks={block_count}"


def _p2p_meta_summary(metadata: Any) -> str:
    requests = _safe_getattr(metadata, "requests", [])
    return "[" + "; ".join(_p2p_request_summary(req) for req in requests) + "]"


def _p2p_role(self: Any) -> str:
    if _safe_getattr(self, "is_producer", False):
        return "producer"
    return "consumer"


def _p2p_parse_remote_address(connector: Any, request_id: str, is_prefill: bool) -> str:
    try:
        ip, port = connector.parse_request_id(request_id, is_prefill)
        rank = int(_safe_getattr(connector, "_rank", 0))
        return f"{ip}:{port + rank}"
    except Exception as exc:
        return f"parse-error:{type(exc).__name__}:{exc}"


def _p2p_expected_shape(layer: Any, block_ids: Any, attn_metadata: Any) -> Any:
    shape = _safe_getattr(layer, "shape")
    if shape is None:
        return "unknown"

    block_count = _p2p_block_count(block_ids)
    try:
        from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata

        is_mla = isinstance(attn_metadata, MLACommonMetadata)
    except Exception:
        is_mla = False

    try:
        if is_mla or shape[1] == 2:
            return (block_count, *tuple(shape[1:]))
        if shape[0] == 2:
            return (shape[0], block_count, *tuple(shape[2:]))
    except Exception:
        return "unknown"
    return "unsupported-layout"


class _DebugSocketProxy:
    """Log P2P ZMQ traffic while delegating all behavior to the real socket."""

    def __init__(self, socket: Any, label: str, socket_kind: str) -> None:
        self._socket = socket
        self._label = label
        self._socket_kind = socket_kind

    def __getattr__(self, name: str) -> Any:
        return getattr(self._socket, name)

    def send(self, *args: Any, **kwargs: Any) -> Any:
        logger.info(
            "kvcached p2p debug: socket send begin kind=%s label=%s bytes=%s",
            self._socket_kind,
            self._label,
            _safe_len(args[0]) if args else "unknown",
        )
        result = self._socket.send(*args, **kwargs)
        logger.info(
            "kvcached p2p debug: socket send end kind=%s label=%s",
            self._socket_kind,
            self._label,
        )
        return result

    def recv(self, *args: Any, **kwargs: Any) -> Any:
        logger.info(
            "kvcached p2p debug: socket recv begin kind=%s label=%s",
            self._socket_kind,
            self._label,
        )
        result = self._socket.recv(*args, **kwargs)
        logger.info(
            "kvcached p2p debug: socket recv end kind=%s label=%s bytes=%s",
            self._socket_kind,
            self._label,
            _safe_len(result),
        )
        return result

    def send_multipart(self, *args: Any, **kwargs: Any) -> Any:
        parts = args[0] if args else []
        logger.info(
            "kvcached p2p debug: socket send_multipart begin kind=%s label=%s parts=%s",
            self._socket_kind,
            self._label,
            _safe_len(parts),
        )
        result = self._socket.send_multipart(*args, **kwargs)
        logger.info(
            "kvcached p2p debug: socket send_multipart end kind=%s label=%s",
            self._socket_kind,
            self._label,
        )
        return result

    def recv_multipart(self, *args: Any, **kwargs: Any) -> Any:
        logger.info(
            "kvcached p2p debug: socket recv_multipart begin kind=%s label=%s",
            self._socket_kind,
            self._label,
        )
        result = self._socket.recv_multipart(*args, **kwargs)
        _log_p2p_multipart_payload(self._socket_kind, self._label, result)
        return result


def _wrap_debug_socket(socket: Any, label: str, socket_kind: str) -> Any:
    if isinstance(socket, _DebugSocketProxy):
        return socket
    return _DebugSocketProxy(socket, label, socket_kind)


def _log_p2p_multipart_payload(socket_kind: str, label: str, result: Any) -> None:
    remote = "unknown"
    cmd = "unknown"
    tensor_id = None
    shape = None
    dtype = None
    try:
        remote, message = result
        if isinstance(remote, bytes):
            remote = remote.decode()
        import msgpack

        data = msgpack.loads(message)
        cmd = data.get("cmd", "unknown")
        tensor_id = data.get("tensor_id")
        shape = data.get("shape")
        dtype = data.get("dtype")
    except Exception as exc:
        cmd = f"decode-error:{type(exc).__name__}"

    logger.info(
        "kvcached p2p debug: listener received kind=%s label=%s remote=%s "
        "cmd=%s tensor_id=%s shape=%s dtype=%s",
        socket_kind,
        label,
        remote,
        cmd,
        tensor_id,
        shape,
        dtype,
    )


def _wrap_router_socket_recv_multipart(engine: Any) -> None:
    if getattr(engine, "__kvcached_p2p_router_recv_wrapped__", False):
        return

    socket = _safe_getattr(engine, "router_socket")
    original_recv_multipart = _safe_getattr(socket, "recv_multipart")
    if original_recv_multipart is None:
        return

    label = str(_safe_getattr(engine, "zmq_address", "unknown"))

    def _debug_recv_multipart(*args: Any, **kwargs: Any) -> Any:
        logger.info(
            "kvcached p2p debug: socket recv_multipart begin kind=router label=%s",
            label,
        )
        result = original_recv_multipart(*args, **kwargs)
        _log_p2p_multipart_payload("router", label, result)
        return result

    try:
        setattr(socket, "recv_multipart", _debug_recv_multipart)
    except Exception as exc:
        logger.warning(
            "kvcached p2p debug: could not wrap router recv_multipart "
            "for listener command logs: %s",
            exc,
        )
    setattr(engine, "__kvcached_p2p_router_recv_wrapped__", True)


@when_imported("vllm")
def _patch_vllm(_vllm: types.ModuleType) -> None:
    # Diagnostic-only P2P NCCL instrumentation is intentionally unconditional
    # in this debug branch so hangs always leave breadcrumbs.
    _patch_p2p_nccl_debug()

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

    # Log results
    log_patch_results("vllm", results)


def _patch_p2p_nccl_debug() -> None:
    """Patch P2pNcclConnector/P2pNcclEngine with debug-only logging wrappers."""
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import (
            P2pNcclConnector,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (
            P2pNcclEngine,
        )
    except ImportError as exc:
        logger.info("kvcached p2p debug: P2pNcclConnector unavailable: %s", exc)
        return

    _patch_p2p_connector_debug(P2pNcclConnector)
    _patch_p2p_engine_debug(P2pNcclEngine)
    logger.info("kvcached p2p debug: patched P2pNcclConnector/P2pNcclEngine")


def _patch_p2p_connector_debug(P2pNcclConnector: Any) -> None:
    if getattr(P2pNcclConnector, "__kvcached_p2p_debug_patched__", False):
        return

    original_build_connector_meta = P2pNcclConnector.build_connector_meta
    original_save_kv_layer = P2pNcclConnector.save_kv_layer
    original_start_load_kv = P2pNcclConnector.start_load_kv

    def _debug_build_connector_meta(self: Any, scheduler_output: Any, *args: Any,
                                    **kwargs: Any) -> Any:
        new_reqs = _safe_getattr(scheduler_output, "scheduled_new_reqs", [])
        cached_reqs = _safe_getattr(scheduler_output, "scheduled_cached_reqs", None)
        cached_req_ids = _safe_getattr(cached_reqs, "req_ids", [])
        logger.info(
            "kvcached p2p debug: build_connector_meta start role=%s "
            "scheduled_new=%s scheduled_cached=%s requests_need_load=%s chunked=%s",
            _p2p_role(self),
            [_safe_getattr(req, "req_id", "?") for req in new_reqs],
            list(cached_req_ids) if cached_req_ids is not None else [],
            list(_safe_getattr(self, "_requests_need_load", {}).keys()),
            list(_safe_getattr(self, "chunked_prefill", {}).keys()),
        )
        meta = original_build_connector_meta(self, scheduler_output, *args, **kwargs)
        requests = _safe_getattr(meta, "requests", [])
        logger.info(
            "kvcached p2p debug: build_connector_meta end role=%s meta_count=%s meta=%s",
            _p2p_role(self),
            _safe_len(requests),
            _p2p_meta_summary(meta),
        )
        return meta

    def _debug_save_kv_layer(self: Any, layer_name: str, kv_layer: Any, attn_metadata: Any,
                             *args: Any, **kwargs: Any) -> Any:
        metadata = None
        try:
            metadata = self._get_connector_metadata()
        except Exception as exc:
            logger.info(
                "kvcached p2p debug: save_kv_layer metadata unavailable "
                "layer=%s error=%s",
                layer_name,
                exc,
            )

        for request in _safe_getattr(metadata, "requests", []):
            request_id = _safe_getattr(request, "request_id", "?")
            block_ids = _safe_getattr(request, "block_ids", None)
            logger.info(
                "kvcached p2p debug: save_kv_layer before-send role=%s "
                "tensor_id=%s#%s layer=%s remote_address=%s block_count=%s "
                "expected_send_shape=%s kv_layer=%s",
                _p2p_role(self),
                request_id,
                layer_name,
                layer_name,
                _p2p_parse_remote_address(self, request_id, True),
                _p2p_block_count(block_ids),
                _p2p_expected_shape(kv_layer, block_ids, attn_metadata),
                _p2p_tensor_summary(kv_layer),
            )

        result = original_save_kv_layer(
            self, layer_name, kv_layer, attn_metadata, *args, **kwargs
        )
        logger.info(
            "kvcached p2p debug: save_kv_layer end role=%s layer=%s",
            _p2p_role(self),
            layer_name,
        )
        return result

    def _debug_start_load_kv(self: Any, forward_context: Any, *args: Any,
                             **kwargs: Any) -> Any:
        try:
            metadata = self._get_connector_metadata()
        except Exception as exc:
            metadata = None
            logger.info("kvcached p2p debug: start_load_kv metadata unavailable: %s", exc)

        no_compile_layers = _safe_getattr(forward_context, "no_compile_layers", {}) or {}
        for request in _safe_getattr(metadata, "requests", []):
            request_id = _safe_getattr(request, "request_id", "?")
            block_ids = _safe_getattr(request, "block_ids", None)
            remote_address = _p2p_parse_remote_address(self, request_id, False)
            for layer_name, layer in no_compile_layers.items():
                kv_cache = _safe_getattr(layer, "kv_cache")
                if kv_cache is None:
                    continue
                logger.info(
                    "kvcached p2p debug: start_load_kv waiting role=%s "
                    "tensor_id=%s#%s layer=%s remote_address=%s block_count=%s "
                    "target_kv_cache=%s",
                    _p2p_role(self),
                    request_id,
                    layer_name,
                    layer_name,
                    remote_address,
                    _p2p_block_count(block_ids),
                    _p2p_tensor_summary(kv_cache),
                )

        result = original_start_load_kv(self, forward_context, *args, **kwargs)
        logger.info("kvcached p2p debug: start_load_kv end role=%s", _p2p_role(self))
        return result

    P2pNcclConnector.build_connector_meta = _debug_build_connector_meta
    P2pNcclConnector.save_kv_layer = _debug_save_kv_layer
    P2pNcclConnector.start_load_kv = _debug_start_load_kv
    P2pNcclConnector.__kvcached_p2p_debug_patched__ = True


def _patch_p2p_engine_debug(P2pNcclEngine: Any) -> None:
    if getattr(P2pNcclEngine, "__kvcached_p2p_debug_patched__", False):
        return

    original_init = P2pNcclEngine.__init__
    original_create_connect = P2pNcclEngine.create_connect
    original_send_tensor = P2pNcclEngine.send_tensor
    original_recv_tensor = P2pNcclEngine.recv_tensor
    original_listen_for_requests = P2pNcclEngine.listen_for_requests
    original_send_sync = P2pNcclEngine.send_sync
    original_send = P2pNcclEngine.send
    original_recv = P2pNcclEngine.recv

    def _debug_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        logger.info(
            "kvcached p2p debug: engine init rank=%s local_rank=%s "
            "zmq_address=%s send_type=%s buffer_threshold=%s",
            _safe_getattr(self, "rank", "?"),
            _safe_getattr(self, "local_rank", "?"),
            _safe_getattr(self, "zmq_address", "?"),
            _safe_getattr(self, "send_type", "?"),
            _safe_getattr(self, "buffer_size_threshold", "?"),
        )

    def _debug_create_connect(self: Any, remote_address: str | None = None) -> Any:
        logger.info(
            "kvcached p2p debug: create_connect begin local=%s remote=%s known=%s",
            _safe_getattr(self, "zmq_address", "?"),
            remote_address,
            remote_address in _safe_getattr(self, "socks", {}),
        )
        result = original_create_connect(self, remote_address)
        if remote_address is not None and remote_address in self.socks:
            self.socks[remote_address] = _wrap_debug_socket(
                self.socks[remote_address], str(remote_address), "dealer"
            )
            result = (self.socks[remote_address], self.comms[remote_address])
        logger.info(
            "kvcached p2p debug: create_connect end local=%s remote=%s comm_known=%s",
            _safe_getattr(self, "zmq_address", "?"),
            remote_address,
            remote_address in _safe_getattr(self, "comms", {}),
        )
        return result

    def _debug_send_tensor(self: Any, tensor_id: str, tensor: Any,
                           remote_address: str | None = None) -> Any:
        logger.info(
            "kvcached p2p debug: send_tensor enter local=%s send_type=%s "
            "tensor_id=%s remote=%s %s",
            _safe_getattr(self, "zmq_address", "?"),
            _safe_getattr(self, "send_type", "?"),
            tensor_id,
            remote_address,
            _p2p_tensor_summary(tensor),
        )
        result = original_send_tensor(self, tensor_id, tensor, remote_address)
        logger.info(
            "kvcached p2p debug: send_tensor exit local=%s tensor_id=%s result=%s",
            _safe_getattr(self, "zmq_address", "?"),
            tensor_id,
            result,
        )
        return result

    def _debug_recv_tensor(self: Any, tensor_id: str,
                           remote_address: str | None = None) -> Any:
        send_type = _safe_getattr(self, "send_type", "?")
        logger.info(
            "kvcached p2p debug: recv_tensor enter local=%s send_type=%s "
            "tensor_id=%s remote=%s",
            _safe_getattr(self, "zmq_address", "?"),
            send_type,
            tensor_id,
            remote_address,
        )
        if send_type in ("PUT", "PUT_ASYNC"):
            start = time.monotonic()
            interval = _p2p_wait_log_interval_s()
            with self.recv_store_cv:
                while tensor_id not in self.recv_store:
                    logger.info(
                        "kvcached p2p debug: recv_tensor waiting local=%s "
                        "tensor_id=%s remote=%s recv_store_size=%s interval_s=%s",
                        _safe_getattr(self, "zmq_address", "?"),
                        tensor_id,
                        remote_address,
                        _safe_len(self.recv_store),
                        interval,
                    )
                    self.recv_store_cv.wait(timeout=interval)
                    if tensor_id not in self.recv_store:
                        logger.warning(
                            "kvcached p2p debug: recv_tensor still waiting local=%s "
                            "tensor_id=%s remote=%s elapsed_s=%.3f recv_store_size=%s",
                            _safe_getattr(self, "zmq_address", "?"),
                            tensor_id,
                            remote_address,
                            time.monotonic() - start,
                            _safe_len(self.recv_store),
                        )

        result = original_recv_tensor(self, tensor_id, remote_address)
        logger.info(
            "kvcached p2p debug: recv_tensor exit local=%s tensor_id=%s %s",
            _safe_getattr(self, "zmq_address", "?"),
            tensor_id,
            _p2p_tensor_summary(result),
        )
        return result

    def _debug_listen_for_requests(self: Any) -> Any:
        logger.info(
            "kvcached p2p debug: listen_for_requests start local=%s",
            _safe_getattr(self, "zmq_address", "?"),
        )
        _wrap_router_socket_recv_multipart(self)
        return original_listen_for_requests(self)

    def _debug_send_sync(self: Any, item: Any) -> Any:
        tensor_id = _safe_getattr(item, "tensor_id", "?")
        remote_address = _safe_getattr(item, "remote_address", "?")
        logger.info(
            "kvcached p2p debug: send_sync begin local=%s tensor_id=%s "
            "remote=%s %s",
            _safe_getattr(self, "zmq_address", "?"),
            tensor_id,
            remote_address,
            _p2p_tensor_summary(_safe_getattr(item, "tensor")),
        )
        logger.info(
            "kvcached p2p debug: send_sync path local=%s tensor_id=%s "
            "if create_connect ends and nccl send does not begin, likely waiting for PUT ack",
            _safe_getattr(self, "zmq_address", "?"),
            tensor_id,
        )
        result = original_send_sync(self, item)
        logger.info(
            "kvcached p2p debug: send_sync end local=%s tensor_id=%s result=%s",
            _safe_getattr(self, "zmq_address", "?"),
            tensor_id,
            result,
        )
        return result

    def _debug_send(self: Any, comm: Any, tensor: Any, dst: int,
                    stream: Any = None) -> Any:
        logger.info(
            "kvcached p2p debug: nccl send begin local=%s dst=%s %s",
            _safe_getattr(self, "zmq_address", "?"),
            dst,
            _p2p_tensor_summary(tensor),
        )
        result = original_send(self, comm, tensor, dst, stream)
        logger.info(
            "kvcached p2p debug: nccl send end local=%s dst=%s",
            _safe_getattr(self, "zmq_address", "?"),
            dst,
        )
        return result

    def _debug_recv(self: Any, comm: Any, tensor: Any, src: int,
                    stream: Any = None) -> Any:
        logger.info(
            "kvcached p2p debug: nccl recv begin local=%s src=%s %s",
            _safe_getattr(self, "zmq_address", "?"),
            src,
            _p2p_tensor_summary(tensor),
        )
        result = original_recv(self, comm, tensor, src, stream)
        logger.info(
            "kvcached p2p debug: nccl recv end local=%s src=%s",
            _safe_getattr(self, "zmq_address", "?"),
            src,
        )
        return result

    P2pNcclEngine.__init__ = _debug_init
    P2pNcclEngine.create_connect = _debug_create_connect
    P2pNcclEngine.send_tensor = _debug_send_tensor
    P2pNcclEngine.recv_tensor = _debug_recv_tensor
    P2pNcclEngine.listen_for_requests = _debug_listen_for_requests
    P2pNcclEngine.send_sync = _debug_send_sync
    P2pNcclEngine.send = _debug_send
    P2pNcclEngine.recv = _debug_recv
    P2pNcclEngine.__kvcached_p2p_debug_patched__ = True

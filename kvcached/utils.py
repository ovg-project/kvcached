# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import logging
import os
import signal
import socket
import threading
import uuid
import weakref
from typing import BinaryIO, Callable, Optional, Tuple


class KVCachedConfigError(RuntimeError):
    """Raised for kvcached misconfiguration the user must fix (e.g. a KV block
    larger than the page size). Integration patches re-raise this loudly to
    abort startup instead of silently falling back to non-kvcached behavior."""


class KVCachePoolExhausted(ValueError):
    """Raised when the shared physical KV pool cannot back an allocation.

    This is a transient condition, not a defect: colocated engines share one
    physical pool, so a peer can take the last pages between the moment
    availability is observed and the moment the pages are claimed. Serving
    engines already know how to respond -- free something and try again -- so
    integrations translate this into whatever "cannot allocate right now"
    signal their engine understands, rather than letting it terminate the
    process.

    It subclasses ValueError so callers written against the pre-existing
    behavior keep working.
    """


def _sanitize_segment(segment: str) -> str:
    """Sanitize a segment to safe characters for SHM names."""
    allowed = []
    for ch in segment:
        if ch.isalnum() or ch in ("_", "-"):
            allowed.append(ch)
        else:
            allowed.append("-")
    # Collapse to at most 64 chars to keep names short
    return "".join(allowed)[:64]


def _detect_engine_tag() -> str:
    """Best-effort detection of the hosting engine for naming.
    """
    if importlib.util.find_spec("vllm") is not None:
        return "vLLM"
    if importlib.util.find_spec("sglang") is not None:
        return "SGLang"
    return "proc"


def _ipc_segment_exists(name: str) -> bool:
    """Return True if a shared-memory segment/file with this name exists.
    """
    try:
        return os.path.exists(os.path.join(SHM_DIR, name))
    except Exception:
        return False


def _obtain_default_ipc_name() -> str:
    """Return a default IPC name like kvcached_<Engine>_<PGID>.

    Precedence:
    1) KVCACHED_IPC_NAME: explicit full name
    2) Otherwise, derive from engine tag and process group id (PGID).
    Using PGID ensures all processes within the same instance share a
    single segment, while separate launches get distinct names.
    """

    explicit = os.getenv("KVCACHED_IPC_NAME")
    if explicit:
        # An explicit IPC name is a contract between all processes in one
        # serving instance.  Treat it as exact after SHM-safe sanitization: do
        # not probe or auto-suffix it, because late-importing TP workers must
        # still join the same namespace.
        return _sanitize_segment(explicit)

    engine_tag = _detect_engine_tag()
    try:
        group_id = os.getpgid(0)
    except Exception:
        try:
            group_id = os.getsid(0)
        except Exception:
            group_id = os.getpid()

    # No explicit override: start from conventional base and ensure uniqueness
    base = "kvcached"
    name = f"{base}_{engine_tag}_{group_id}"
    if not _ipc_segment_exists(name):
        return name
    for i in range(1, 100):
        candidate = f"{name}_{i}"
        if not _ipc_segment_exists(candidate):
            return candidate
    return f"{name}_{os.getpid()}"


def _get_page_size() -> int:
    """Get PAGE_SIZE from environment variable with validation.

    Returns:
        PAGE_SIZE in bytes, must be a multiple of 2MB (2097152 bytes)

    Raises:
        ValueError: If PAGE_SIZE is not a multiple of 2MB
    """
    default_page_size = 2 * 1024 * 1024  # 2MB
    page_size_mb_str = os.getenv("KVCACHED_PAGE_SIZE_MB")

    if page_size_mb_str is None:
        return default_page_size

    try:
        page_size = int(page_size_mb_str) * 1024 * 1024
    except ValueError:
        raise ValueError(
            f"Invalid KVCACHED_PAGE_SIZE_MB: {page_size_mb_str}. Must be an integer."
        )

    # Validate that PAGE_SIZE is a multiple of 2MB
    base_size = 2 * 1024 * 1024  # 2MB
    if page_size <= 0 or page_size % base_size != 0:
        raise ValueError(
            f"PAGE_SIZE must be a positive multiple of 2MB (2097152 bytes), "
            f"got: {page_size}")

    return page_size


PAGE_SIZE = _get_page_size()


def get_page_size_for_block(block_mem_size: int, configured_page_size: int) -> int:
    """Resolve a KV pool's page size without changing process-wide defaults.

    Explicit page settings keep their existing validation behavior. Resolve
    from the final block geometry in both the scheduler and worker, before
    constructing the manager or its backing tensors.
    """
    if (block_mem_size <= configured_page_size
            or os.getenv("KVCACHED_PAGE_SIZE_MB") is not None):
        return configured_page_size
    from kvcached.kv_geometry import select_page_size

    page_size = select_page_size(block_mem_size)
    get_kvcached_logger().info(
        "Default kvcached page (%d bytes) cannot hold the KV block (%d bytes); "
        "using %d-byte pages for this pool", configured_page_size,
        block_mem_size, page_size)
    return page_size

# Configuration constants for KVCacheManager
GPU_UTILIZATION = float(os.getenv("KVCACHED_GPU_UTILIZATION", "0.95"))
PAGE_PREALLOC_ENABLED = os.getenv("KVCACHED_PAGE_PREALLOC_ENABLED",
                                  "true").lower() == "true"
MIN_RESERVED_PAGES = int(os.getenv("KVCACHED_MIN_RESERVED_PAGES", "5"))
MAX_RESERVED_PAGES = int(os.getenv("KVCACHED_MAX_RESERVED_PAGES", "10"))
MAX_CACHED_BLOCKS = int(os.getenv("KVCACHED_MAX_CACHED_BLOCKS", "1000"))
SANITY_CHECK = os.getenv("KVCACHED_SANITY_CHECK", "false").lower() == "true"
# Maximum number of tokens the cache may hold as evictable (cached) entries.
# Semantics:
#   < 0  → unlimited (closest to vanilla vLLM/SGLang prefix-cache behavior;
#          gives up memory elasticity)
#   == 0 → disabled (cached prefixes are evicted as soon as they become
#          evictable, effectively turning off prefix-cache reuse)
#   > 0  → proactively evict after each finished request to stay within
#          this many cached tokens
# Used by both SGLang (RadixCacheLimitPatch) and vLLM (ElasticBlockPool),
# which converts to blocks internally via MAX_CACHED_TOKENS // block_size.
MAX_CACHED_TOKENS = int(os.getenv("KVCACHED_MAX_CACHED_TOKENS", "16000"))


def _default_contiguous_layout() -> bool:
    """Default KV-cache layout: contiguous on CUDA, non-contiguous on HIP/ROCm.

    An explicit ``KVCACHED_CONTIGUOUS_LAYOUT`` always wins. Otherwise we pick
    non-contiguous on ROCm: the contiguous (compound-page) layout hands the
    attention backend strided/interleaved per-layer KV tensors, which vLLM's
    ROCm attention path (``split_kv_cache`` + paged kernels) reads incorrectly,
    whereas CUDA's FlashAttention/FlashInfer tolerate it.
    """
    explicit = os.getenv("KVCACHED_CONTIGUOUS_LAYOUT")
    if explicit is not None:
        return explicit.lower() == "true"
    try:
        import torch
        if getattr(torch.version, "hip", None):
            return False  # ROCm/HIP: non-contiguous is required for correctness
    except Exception:
        pass
    return True


CONTIGUOUS_LAYOUT = _default_contiguous_layout()

DEFAULT_IPC_NAME = _obtain_default_ipc_name()
SHM_DIR = "/dev/shm"

# Root of the per-instance TP worker socket directories (kvcached.tp_ipc_util).
# The naming rule lives here, next to the IPC name, so tools that never load
# the compiled extension (kvctl) can derive the directory from an IPC name.
TP_SOCKET_DIR_ROOT = "/tmp"


def get_tp_socket_dir(ipc_name: Optional[str] = None) -> str:
    """Return the TP worker socket directory for *ipc_name* (default: this
    instance's DEFAULT_IPC_NAME).

    The directory keeps the IPC name readable and appends a short
    deterministic hash, so every worker of one engine instance agrees on it.
    Unix domain socket paths are limited to 108 characters on Linux; the
    caller validates the final socket path length.
    """
    name = DEFAULT_IPC_NAME if ipc_name is None else ipc_name
    suffix = uuid.uuid5(uuid.NAMESPACE_DNS, name).hex[:8]
    return os.path.join(TP_SOCKET_DIR_ROOT, f"kvcached-tp-{name}-{suffix}")


def get_tp_worker_socket_path(socket_dir: str, rank: int, pp_rank: int = 0) -> str:
    """Return the socket path of TP worker *rank* in PP stage *pp_rank*
    under *socket_dir*: w<rank>.sock, inside a pp<pp_rank> subdirectory for
    every stage after the first so that stages starting concurrently never
    bind the same path."""
    if pp_rank > 0:
        return os.path.join(socket_dir, f"pp{pp_rank}", f"w{rank}.sock")
    return os.path.join(socket_dir, f"w{rank}.sock")


def path_identity(path: str) -> Optional[Tuple[int, int]]:
    """(st_dev, st_ino) of the node at *path*, or None when there is none.

    The pathname alone does not identify a socket: a same-name restart
    binds a new node at the same path."""
    try:
        st = os.lstat(path)
    except OSError:
        return None
    return st.st_dev, st.st_ino


def remove_dir_if_empty(path: str) -> None:
    try:
        os.rmdir(path)
    except OSError:
        # Still holds another worker's socket, or already gone.
        pass


def _accepts_connections(path: str) -> Optional[bool]:
    """Whether some process still accepts connections on the socket node at
    *path*. A Unix stream socket node answers connect() only while the
    process that bound it is alive (a stopped process still queues the
    connection); once that process has exited the connection is refused.
    None when the probe cannot tell."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
        probe.settimeout(1.0)
        try:
            probe.connect(path)
        except (ConnectionRefusedError, FileNotFoundError):
            return False
        except OSError:
            return None
    return True


class WorkerSocketCleanup:
    """Remember one deployment's worker sockets while its workers are live
    and remove, later, only the ones no worker serves any more (issue #510).

    A worker unlinks its own socket from Worker.shutdown(). On a process
    group SIGTERM vLLM's process manager kills the engine tree before that
    runs, so the parent that removes the segment the engines left removes
    the sockets too, behind the same confirmed engine exit. A socket node
    cannot be pinned by an open descriptor, so each one is remembered by
    (st_dev, st_ino): a node that no longer matches belongs to a same-name
    replacement and is kept. A remembered node is removed only once a
    connect() to it is refused; a stopped or delayed worker still accepts
    and keeps its socket for a later retry. The check-then-unlink window
    of the worker's own listener stop applies here as well.

    A directory is removed only by the call that removed one of the
    deployment's nodes from it, and a completed cleanup is a no-op: a
    same-name replacement creates the same directory again between its
    makedirs() and bind(), and vLLM calls the owner's shutdown more than
    once per teardown.
    """

    def __init__(self, socket_dir: str, tp_size: int, pp_size: int) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, Tuple[int, int]] = {}
        for pp_rank in range(max(int(pp_size), 1)):
            for rank in range(max(int(tp_size), 1)):
                path = get_tp_worker_socket_path(socket_dir, rank, pp_rank)
                identity = path_identity(path)
                if identity is not None:
                    self._pending[path] = identity
        # Deepest first: a pp<k> directory before the deployment root, which
        # is removed once empty like the worker's own stop removes it.
        directories = {os.path.dirname(path) for path in self._pending}
        if directories:
            directories.add(socket_dir)
        self._dirs = sorted(directories, key=len, reverse=True)

    def unlink(self) -> bool:
        """Remove the remembered sockets nobody serves and the directories
        that leaves empty. Return True when nothing is left to retry."""
        with self._lock:
            if not self._pending:
                return True
            logger = get_kvcached_logger()
            emptied: set[str] = set()
            for path, identity in list(self._pending.items()):
                current = path_identity(path)
                if current is None or current != identity:
                    del self._pending[path]  # gone, or a replacement's
                    continue
                served = _accepts_connections(path)
                if served is None:
                    logger.warning("Keeping worker socket %s: cannot tell whether "
                                   "a worker still serves it", path)
                    continue
                if served:
                    logger.warning("Keeping worker socket %s: a worker still "
                                   "accepts connections on it", path)
                    continue
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
                except OSError as e:
                    logger.warning("Failed to remove worker socket %s: %s", path, e)
                    continue
                logger.info("Removed worker socket %s left by a killed worker", path)
                del self._pending[path]
                emptied.add(os.path.dirname(path))
            for directory in self._dirs:
                if any(target == directory or target.startswith(directory + os.sep)
                       for target in emptied):
                    remove_dir_if_empty(directory)
            return not self._pending


class IPCSegmentCleanup:
    """Remember one segment across teardown and retry only its failed unlink.

    Capture before stopping the engine: it may remove its own segment during
    shutdown. The open file pins the inode, so a later file at the same path
    cannot inherit its identity. No contents are read or modified. This is
    a replacement check, not a lock against concurrent instance startup.
    """

    def __init__(self, segment: str) -> None:
        self.segment = segment
        self._lock = threading.Lock()
        self._file: Optional[BinaryIO]
        try:
            self._file = open(segment, "rb")
        except FileNotFoundError:
            self._file = None

    def unlink(self) -> bool:
        """Return True when done; keep the original file open on failure."""
        with self._lock:
            return self._unlink()

    def _unlink(self) -> bool:
        if self._file is None:
            return True
        try:
            current = os.stat(self.segment)
            original = os.fstat(self._file.fileno())
            if os.path.samestat(current, original):
                os.unlink(self.segment)
                get_kvcached_logger().info(
                    "Unlinked KV cache limit segment %s", self.segment)
        except FileNotFoundError:
            pass
        except OSError as e:
            get_kvcached_logger().warning(
                "Failed to unlink %s on shutdown: %s", self.segment, e)
            return False
        self._file.close()
        self._file = None
        return True

    def close(self) -> None:
        """Discard an unconfirmed identity without deleting anything."""
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

LOG_USE_COLOR = os.getenv("KVCACHED_LOG_COLOR", "true").lower() == "true"
_UNIFORM_COLOR = os.getenv("KVCACHED_LOG_COLOR_CODE", "\033[36m")

_LEVEL_TO_COLOR = {
    logging.DEBUG: "\033[36m",  # Cyan
    logging.INFO: "\033[32m",  # Green
    logging.WARNING: "\033[33m",  # Yellow
    logging.ERROR: "\033[31m",  # Red
    logging.CRITICAL: "\033[35m",  # Magenta
}
_COLOR_RESET = "\033[0m"


def normalize_gpu_device(device: str) -> str:
    """Map a ``hip[:N]`` device string to ``cuda[:N]``.

    PyTorch-ROCm and the C++ extension (``c10::Device``) address AMD GPUs as
    ``cuda``; kvcached's integration accepts ``hip`` strings, so normalize them
    before handing the device to any ``torch.cuda`` API or ``create_kv_tensors``.
    """
    dev = str(device)
    if dev.lower().startswith("hip"):
        return "cuda" + dev[3:]
    return dev


def align_to(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def align_up_to_page(n_cells: int, cell_size: int) -> int:
    n_cells_per_page = PAGE_SIZE // cell_size
    aligned_n_cells = align_to(n_cells, n_cells_per_page)
    return aligned_n_cells


class ColorFormatter(logging.Formatter):
    """A logging formatter that injects ANSI colors based on the log level."""

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)

        color = _LEVEL_TO_COLOR.get(record.levelno, _UNIFORM_COLOR)

        prefix, sep, rest = formatted.partition("] ")

        if sep:
            prompt = f"{prefix}{sep}"
            return f"{color}{prompt}{_COLOR_RESET}{rest}"
        else:
            return f"{color}{formatted}{_COLOR_RESET}"


def get_log_level():
    level = os.getenv("KVCACHED_LOG_LEVEL", "INFO").upper()
    return getattr(logging, level, logging.INFO)


def get_kvcached_logger(name: str = "kvcached") -> logging.Logger:
    logger = logging.getLogger(name)

    # Only add handler if none exists (prevents duplicate handlers)
    if not logger.handlers:
        handler = logging.StreamHandler()

        fmt_str = (f"[{name}]"
                   "[%(levelname)s]"
                   "[%(asctime)s]"
                   "[%(filename)s:%(lineno)d] %(message)s")

        if LOG_USE_COLOR and handler.stream.isatty():
            formatter: logging.Formatter = ColorFormatter(
                fmt_str, datefmt="%Y-%m-%d %H:%M:%S")
        else:
            formatter = logging.Formatter(fmt_str, datefmt="%Y-%m-%d %H:%M:%S")

        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(get_log_level())
        # Prevent propagation to inference engines; avoid duplicate messages
        logger.propagate = False

    return logger


# Default SIGTERM bypasses Python teardown. Only known process owners register;
# existing application/event-loop handlers remain authoritative.
_sigterm_owner_callbacks: list[tuple[int, weakref.WeakMethod]] = []
_pending_owner_sigterm: Optional[Tuple[int, int, Tuple[Callable[[], None], ...]]] = None
_sigterm_finalizer_pid: Optional[int] = None
_owner_sigterm_loop_handler: Optional[Callable[..., None]] = None


def _unwind_for_owner_sigterm(signum, frame) -> None:
    global _pending_owner_sigterm
    # Do not acquire a lock, log, join, or call shutdown in a signal handler:
    # the interrupted frame may hold the very lock shutdown needs. Unwind the
    # Python stack first, as SIGINT already does. A second TERM remains fatal.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    pid = os.getpid()
    # Keep owners alive only during TERM unwinding: newer Python versions can
    # release a function-local owner before process finalizers run.
    callbacks = tuple(callback for owner_pid, ref in _sigterm_owner_callbacks
                      if owner_pid == pid and (callback := ref()) is not None)
    _pending_owner_sigterm = (pid, signum, callbacks)
    raise SystemExit(128 + signum)


def _finish_owner_sigterm() -> None:
    global _pending_owner_sigterm
    pending, _pending_owner_sigterm = _pending_owner_sigterm, None
    if pending is None or pending[0] != os.getpid():
        return
    try:
        for callback in pending[2]:
            try:
                callback()
            except BaseException as error:
                get_kvcached_logger().warning("Owner shutdown during SIGTERM failed: %s", error)
    finally:
        # Preserve WIFSIGNALED/SIGTERM, not a successful exit or exit code 143.
        # This runs after stack unwinding, never recursively inside shutdown.
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), pending[1])


def register_owner_sigterm_cleanup(callback) -> bool:
    """Arrange owner teardown for an otherwise-default SIGTERM.

    Only the main thread may register a bound shutdown method, after capturing
    process ownership. Never replace a custom/ignored/event-loop handler. Weak
    callbacks do not retain owners during normal execution; a received TERM
    holds them through stack unwinding. PID checks exclude inherited owners.
    Applications that suppress SystemExit or bypass Python exit with os._exit
    still own their shutdown policy. No thread, daemon, or polling is added.
    """
    global _sigterm_finalizer_pid
    if threading.current_thread() is not threading.main_thread():
        return False
    previous = signal.getsignal(signal.SIGTERM)
    if (previous not in (signal.SIG_DFL, _unwind_for_owner_sigterm)
            and (previous is not _owner_sigterm_loop_handler
                 or _owner_sigterm_loop_handler is None)):
        return False
    pid = os.getpid()
    ref = weakref.WeakMethod(callback)
    _sigterm_owner_callbacks[:] = [(owner_pid, item)
                                  for owner_pid, item in _sigterm_owner_callbacks
                                  if owner_pid == pid and item() is not None]
    if (pid, ref) not in _sigterm_owner_callbacks:
        _sigterm_owner_callbacks.append((pid, ref))
    if _sigterm_finalizer_pid != pid:
        # multiprocessing's fork bootstrap uses os._exit after its own
        # finalizers, bypassing atexit. Cover that normal bootstrap too, before
        # its automatic child joins. The pending marker makes dispatch one-shot.
        from multiprocessing.util import Finalize

        Finalize(None, _finish_owner_sigterm, exitpriority=1)
        # CPython joins non-daemon threads before ordinary atexit callbacks.
        # A serving thread may need owner.shutdown() to release it, so use the
        # pre-join exit hook available on our supported Python >= 3.10 instead.
        # Like Finalize, this runs after the main stack has unwound, not from
        # the signal handler. Cover this private runtime contract in tests.
        # SGLang 0.5.11/0.5.12 disable _register_atexit at import time,
        # but CPython still drains this list before joining threads.
        getattr(threading, "_threading_atexits").append(_finish_owner_sigterm)
        _sigterm_finalizer_pid = pid
    if previous is not _owner_sigterm_loop_handler:
        signal.signal(signal.SIGTERM, _unwind_for_owner_sigterm)
    return True


def register_owner_sigterm_loop(loop, previous) -> bool:
    """Preserve owner exit when SGLang's registered loop stops between requests.

    Call only immediately after the framework installs its own loop handler.
    A running loop keeps its original signal delivery and wakeup fd. A stopped
    synchronous Engine loop cannot consume that delivery: unwind instead. Do
    not adopt an application's custom/ignored prior disposition or a process
    without registered ownership. No shutdown work runs in this handler.
    """
    global _owner_sigterm_loop_handler
    if threading.current_thread() is not threading.main_thread():
        return False
    if (previous not in (signal.SIG_DFL, _unwind_for_owner_sigterm)
            and (previous is not _owner_sigterm_loop_handler
                 or _owner_sigterm_loop_handler is None)):
        return False
    if not any(pid == os.getpid() and ref() is not None
               for pid, ref in _sigterm_owner_callbacks):
        return False
    original = signal.getsignal(signal.SIGTERM)
    if not callable(original):
        return False
    loop_ref = weakref.ref(loop)

    def handle(signum, frame):
        active_loop = loop_ref()
        if active_loop is not None and active_loop.is_running():
            original(signum, frame)
        else:
            _unwind_for_owner_sigterm(signum, frame)

    _owner_sigterm_loop_handler = handle
    signal.signal(signal.SIGTERM, handle)
    return True

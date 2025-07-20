import logging
import os

PAGE_SIZE = 2 * 1024 * 1024  # 2MB

# Configuration constants for KVCacheManager
GPU_UTILIZATION = float(os.getenv("KVCACHED_GPU_UTILIZATION", "0.95"))
PAGE_PREALLOC_ENABLED = os.getenv("KVCACHED_PAGE_PREALLOC_ENABLED",
                                  "true").lower() == "true"
MIN_RESERVED_PAGES = int(os.getenv("KVCACHED_MIN_RESERVED_PAGES", "5"))
MAX_RESERVED_PAGES = int(os.getenv("KVCACHED_MAX_RESERVED_PAGES", "10"))
SANITY_CHECK = os.getenv("KVCACHED_SANITY_CHECK", "false").lower() == "true"

# Allow overriding the shared-memory segment name via env var so multiple
# kvcached deployments on one machine can coexist without collision.
DEFAULT_IPC_NAME = os.getenv("KVCACHED_IPC_NAME", "kvcached_mem_info")
SHM_DIR = "/dev/shm"


def align_to(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def align_up_to_page(n_cells: int, cell_size: int) -> int:
    n_cells_per_page = PAGE_SIZE // cell_size
    aligned_n_cells = align_to(n_cells, n_cells_per_page)
    return aligned_n_cells


def get_log_level():
    import os  # noqa: E501

    level = os.getenv("KVCACHED_LOG_LEVEL", "INFO").upper()
    return getattr(logging, level, logging.INFO)


def get_kvcached_logger(name: str = "kvcached") -> logging.Logger:
    logger = logging.getLogger(name)

    # Only add handler if none exists (prevents duplicate handlers)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f"[{name}][%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(get_log_level())
        # Prevent propagation to inference engines; avoid duplicate messages
        logger.propagate = False

    return logger

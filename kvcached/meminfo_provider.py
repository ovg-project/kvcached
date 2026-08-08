# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import time
from typing import Optional, Tuple

from kvcached.tp_ipc_util import query_worker_cuda_mem_get_info
from kvcached.utils import (
    MEMINFO_QUERY_TIMEOUT,
    get_kvcached_logger,
)

logger = get_kvcached_logger()


def query_mem_info(world_size: int, pp_rank: int = 0) -> Tuple[int, int]:
    """Read CUDA memory information from the PP group's TP0 worker.

    The caller may intentionally have no CUDA context. Retry until the worker
    listener is ready, but fail closed if the representative worker remains
    unavailable after a bounded timeout.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")

    effective_pp_rank = max(pp_rank, 0)
    started_at = time.monotonic()
    deadline = started_at + MEMINFO_QUERY_TIMEOUT
    last_error: Optional[Exception] = None
    attempts = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.error(
                "Timed out querying CUDA memory from TP0: "
                "pp=%d tp=%d attempts=%d elapsed_ms=%.3f last_error=%r; "
                "refusing new physical mappings",
                effective_pp_rank,
                world_size,
                attempts,
                (time.monotonic() - started_at) * 1000,
                last_error,
            )
            raise TimeoutError(
                "timed out waiting for kvcached worker CUDA memory info"
            ) from last_error
        try:
            attempts += 1
            attempt_started_at = time.monotonic()
            free_bytes, total_bytes = query_worker_cuda_mem_get_info(
                world_size,
                pp_rank=effective_pp_rank,
                timeout=min(remaining, 1.0),
            )
            logger.debug(
                "Queried CUDA memory from TP0: "
                "pp=%d tp=%d attempts=%d elapsed_ms=%.3f "
                "free_bytes=%d total_bytes=%d",
                effective_pp_rank,
                world_size,
                attempts,
                (time.monotonic() - attempt_started_at) * 1000,
                free_bytes,
                total_bytes,
            )
            return free_bytes, total_bytes
        except (ConnectionError, OSError, RuntimeError, TimeoutError) as exc:
            last_error = exc
            logger.debug(
                "CUDA meminfo TP-group query attempt failed: "
                "pp=%d tp=%d attempt=%d error=%r",
                effective_pp_rank,
                world_size,
                attempts,
                exc,
            )
            time.sleep(min(0.01, remaining))

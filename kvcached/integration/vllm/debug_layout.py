# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""KV tensor layout dump + per-block SHA helpers.

No-op unless KVCACHED_DUMP_LAYOUT=1. Emits tagged JSON log lines so the
validation harness can grep both prefill and decode logs and compare.
"""

import hashlib
import json
import os
from typing import Iterable, Optional, Sequence, Tuple

import torch

from kvcached.utils import get_kvcached_logger

logger = get_kvcached_logger()

_LAYOUT_TAG = "KVCACHED_LAYOUT_DUMP"
_BLOCK_SHA_TAG = "KVCACHED_BLOCK_SHA"


def _enabled() -> bool:
    return os.getenv("KVCACHED_DUMP_LAYOUT", "0").lower() in ("1", "true", "yes")


def _sha_block(t: torch.Tensor, blocks_dim_idx: int, block_idx: int) -> str:
    idx: list = [slice(None)] * t.ndim
    idx[blocks_dim_idx] = block_idx
    block = t[tuple(idx)].contiguous()
    u8 = block.view(torch.uint8)
    return hashlib.sha256(u8.cpu().numpy().tobytes()).hexdigest()


def dump_kv_layout(
    *,
    tag: str,
    kv_tensors: Sequence[torch.Tensor],
    blocks_dim_idx: Optional[int] = None,
    extra: Optional[dict] = None,
    sha_blocks: Optional[Tuple[int, int]] = None,
) -> None:
    if not _enabled():
        return

    tensors = list(kv_tensors)
    if not tensors:
        return

    layers = []
    for i, t in enumerate(tensors):
        layers.append({
            "layer": i,
            "shape": list(t.shape),
            "stride": list(t.stride()),
            "dtype": str(t.dtype).replace("torch.", ""),
            "device": str(t.device),
            "data_ptr": hex(t.data_ptr()),
        })

    payload = {
        "tag": tag,
        "pid": os.getpid(),
        "num_layers": len(tensors),
        "blocks_dim_idx": blocks_dim_idx,
        "layers": layers,
    }
    if extra is not None:
        payload["extra"] = extra

    logger.info("%s=%s", _LAYOUT_TAG, json.dumps(payload, sort_keys=True))

    if sha_blocks is not None and blocks_dim_idx is not None:
        start, end = sha_blocks
        for b in range(start, end):
            try:
                sha = _sha_block(tensors[0], blocks_dim_idx, b)
            except Exception as e:
                logger.warning("kvcached dump: block %d SHA failed: %s", b, e)
                continue
            logger.info(
                "%s=%s",
                _BLOCK_SHA_TAG,
                json.dumps({
                    "tag": tag,
                    "pid": os.getpid(),
                    "layer": 0,
                    "block": b,
                    "sha256": sha,
                }),
            )


def dump_active_blocks_sha(
    *,
    tag: str,
    kv_tensors: Sequence[torch.Tensor],
    blocks_dim_idx: int,
    block_ids: Iterable[int],
    layer: int = 0,
) -> None:
    if not _enabled():
        return
    tensors = list(kv_tensors)
    if not tensors or layer >= len(tensors):
        return
    t = tensors[layer]
    for b in block_ids:
        try:
            sha = _sha_block(t, blocks_dim_idx, b)
        except Exception as e:
            logger.warning("kvcached dump: block %d SHA failed: %s", b, e)
            continue
        logger.info(
            "%s=%s",
            _BLOCK_SHA_TAG,
            json.dumps({
                "tag": tag,
                "pid": os.getpid(),
                "layer": layer,
                "block": b,
                "sha256": sha,
            }),
        )

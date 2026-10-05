# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Pin the init_kvcached contiguous_layout default to true.

Counts the tensors create_kv_tensors returns: the contiguous default fuses all
layers into one, per-layer returns one each.

Exercises the real extension and VMM, so it needs a GPU.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("kvcached._C")

from kvcached.vmm_ops import (  # noqa: E402
    create_kv_tensors,
    init_kvcached,
    shutdown_kvcached,
)

DEVICE = "cuda:0"
NUM_LAYERS = 2
DTYPE_SIZE = 2
# Per-layer size must be a multiple of kPageSize (2MB) * num_kv_buffers (2).
PER_LAYER_SIZE = 4 * 1024 * 1024


def _num_kv_tensors(**init_kwargs) -> int:
    init_kvcached(DEVICE, **init_kwargs)
    try:
        tensors = create_kv_tensors(PER_LAYER_SIZE, DTYPE_SIZE, DEVICE, NUM_LAYERS)
        return len(tensors)
    finally:
        shutdown_kvcached()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_init_kvcached_defaults_to_contiguous_layout():
    # Contrast the explicit per-layer layout so the tensor count is a meaningful
    # signal, then assert the default matches the contiguous (fused) layout.
    assert _num_kv_tensors(contiguous_layout=False) == NUM_LAYERS
    assert _num_kv_tensors() == 1

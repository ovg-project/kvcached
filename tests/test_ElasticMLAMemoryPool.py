# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""ElasticMLATokenToKVPool: SGLang's MLA pool, backed by kvcached.

Shape parameters mirror DeepSeek-V2-Lite at small scale.

The pool class is installed by ``ElasticMLAMemoryPoolPatch``, which normally
runs from the ``kvcached_autopatch.pth`` dropped into site-packages at install
time. This module imports ``kvcached.autopatch`` explicitly instead, before
touching SGLang, so it also works from a source tree where no ``.pth`` exists.
``KVCACHED_AUTOPATCH`` gates the patch and must be set before that import.

Device-agnostic: the device string comes from ``get_current_device_str()``, so
this runs on whichever accelerator kvcached was built for.
"""
import os

import pytest

torch = pytest.importorskip("torch")

os.environ.setdefault("ENABLE_KVCACHED", "1")
os.environ["KVCACHED_AUTOPATCH"] = "true"

pytest.importorskip("wrapt", reason="autopatch registers its hooks through wrapt")
pytest.importorskip("sglang")

from kvcached.utils import get_current_device_str, get_device_module  # noqa: E402

if not get_device_module().is_available():
    pytest.skip("needs an accelerator", allow_module_level=True)

# Registers the when_imported hook, so it has to precede the sglang import
# below; sorted, the two swap and the patch never runs.
import kvcached.autopatch  # noqa: E402,F401  # isort:skip

from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool  # noqa: E402

SIZE = 8192  # tokens
PAGE_SIZE = 1
DTYPE = torch.bfloat16
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
LAYER_NUM = 4


@pytest.fixture(scope="module")
def pool():
    if "Elastic" not in MLATokenToKVPool.__name__:
        pytest.skip(
            f"ElasticMLAMemoryPoolPatch did not alias MLATokenToKVPool "
            f"(still {MLATokenToKVPool.__name__}); this SGLang version may have "
            f"moved the class"
        )

    from kvcached.integration.sglang.interfaces import shutdown_kvcached

    p = MLATokenToKVPool(
        size=SIZE,
        page_size=PAGE_SIZE,
        dtype=DTYPE,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        layer_num=LAYER_NUM,
        device=get_current_device_str(),
        enable_memory_saver=False,
    )
    try:
        yield p
    finally:
        shutdown_kvcached()


def test_pool_geometry(pool):
    """One buffer per layer, each shaped for the combined MLA latent + rope
    dimensions."""
    assert len(pool.kv_buffer) == LAYER_NUM
    assert pool.kv_cache_dim == KV_LORA_RANK + QK_ROPE_HEAD_DIM
    for buf in pool.kv_buffer:
        assert buf.shape[-1] == pool.kv_cache_dim
        assert buf.dtype == DTYPE


def test_pool_uses_the_kvcached_allocator(pool):
    """The point of the patch: the pool must be backed by kvcached's elastic
    allocator rather than SGLang's static one."""
    assert hasattr(pool, "kvcached_allocator"), (
        "pool was constructed without a kvcached allocator, so it is not elastic"
    )
    assert pool.kvcached_allocator.available_size() > 0


def test_alloc_then_free_restores_availability(pool):
    """Allocation reduces availability by exactly the requested amount and
    freeing gives it all back."""
    allocator = pool.kvcached_allocator
    before = allocator.available_size()

    indices = allocator.alloc(64)
    assert indices is not None, f"alloc(64) failed with {before} available"
    assert len(indices) == 64
    assert allocator.available_size() == before - 64

    allocator.free(indices)
    assert allocator.available_size() == before

# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""ElasticMLATokenToKVPool: SGLang's MLA pool, backed by kvcached.

The pool class is installed by ``ElasticMLAMemoryPoolPatch``, which normally
runs from the ``kvcached_autopatch.pth`` dropped into site-packages at install
time. This module imports ``kvcached.autopatch`` explicitly instead, before
touching SGLang, so it also works from a source tree where no ``.pth`` exists.
``KVCACHED_AUTOPATCH`` gates the patch and must be set before that import.

Device-agnostic: the device string comes from ``get_current_device_str()``, so
this runs on whichever accelerator kvcached was built for.

One capacity case runs per process, chosen by ``KVCACHED_TEST_MLA_CAPACITY``
(default ``small``): kvcached fixes its page size at import, and the
``oversized_explicit`` case needs ``KVCACHED_PAGE_SIZE_MB=6`` set beforehand.
"""
import contextlib
import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

os.environ.setdefault("ENABLE_KVCACHED", "1")
os.environ["KVCACHED_AUTOPATCH"] = "true"

pytest.importorskip("wrapt", reason="autopatch registers its hooks through wrapt")
pytest.importorskip("sglang")

from kvcached.utils import get_current_device_str, get_device_module  # noqa: E402

if not get_device_module().is_available():
    pytest.skip("needs an accelerator", allow_module_level=True)

# Registers the when_imported hook, so it has to precede the sglang imports
# below; sorted, they swap and the patch never runs.
import kvcached.autopatch  # noqa: E402,F401  # isort:skip

from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator  # noqa: E402
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool  # noqa: E402

# Include pools whose logical byte count fits but whole-block packing does not.
SIZE, PAGE_SIZE, BACKING_BLOCKS, PHYSICAL_PAGES, KV_LORA_RANK, PHYSICAL_PAGE_MB = {
    "small": (1024, 64, 29, 1, 512, 2),
    "page64": (8192, 64, 143, 5, 512, 2),
    "packing": (16320, 64, 285, 10, 512, 2),
    "page32": (8192, 32, 285, 5, 512, 2),
    "oversized": (256, 16, 17, 17, 131008, 4),
    "oversized_packing": (256, 16, 27, 11, 81856, 6),
    # This case is run with KVCACHED_PAGE_SIZE_MB=6.
    "oversized_explicit": (256, 16, 26, 17, 131008, 6),
}[os.environ.get("KVCACHED_TEST_MLA_CAPACITY", "small")]
DTYPE = torch.bfloat16
QK_ROPE_HEAD_DIM = 64
LAYER_NUM = 4


def _single_rank_parallel():
    """A pool built outside an engine has no published parallel config, and
    SGLang's MLA write path reads ``dcp_enabled`` from it. State the
    single-rank value for the duration of the writes."""
    try:
        from sglang.srt.runtime_context import get_parallel
    except ImportError:  # SGLang without a runtime parallel context
        return contextlib.nullcontext()
    return get_parallel().override(dcp_enabled=False)


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


def test_logical_capacity_is_served_by_the_backing_pages(pool):
    """The real page allocator backs exactly the configured token budget:
    every logical token is allocatable and writable, nothing beyond it is, and
    physical quota growth does not enlarge the budget."""
    device = get_current_device_str()
    allocator = pool.kvcached_allocator
    # Wait explicitly for null-block setup.
    assert allocator._post_init_done.wait(timeout=10.0)
    assert allocator.null_block == [0]
    assert allocator.page_size == PHYSICAL_PAGE_MB * 1024 * 1024
    assert allocator.num_blocks == BACKING_BLOCKS
    assert allocator.page_allocator.get_num_total_pages() == PHYSICAL_PAGES

    logical_allocator = PagedTokenToKVPoolAllocator(
        SIZE, PAGE_SIZE, DTYPE, device, pool, False
    )
    assert logical_allocator.available_size() == SIZE

    if os.environ.get("KVCACHED_TEST_MLA_RESERVE") == "1":
        assert allocator.try_to_reserve(SIZE // PAGE_SIZE)
        assert len(allocator.reserved_blocks) == SIZE // PAGE_SIZE
        assert logical_allocator.available_size() == SIZE

    # Fill the logical pool, reject the extra backing-only capacity, and verify
    # that the mapped device rows are writable through the inherited MLA write
    # path.
    indices = logical_allocator.alloc(SIZE)
    assert indices is not None and len(indices) == SIZE
    assert indices.unique().numel() == SIZE
    assert logical_allocator.available_size() == 0
    assert logical_allocator.alloc(PAGE_SIZE) is None
    # Physical quota growth must not enlarge the configured token budget,
    # including geometries whose initial logical/backing counts are equal.
    assert allocator.resize((PHYSICAL_PAGES + 1) * allocator.page_size)
    assert logical_allocator.available_size() == 0
    assert logical_allocator.alloc(PAGE_SIZE) is None
    assert allocator.resize(PHYSICAL_PAGES * allocator.page_size)

    expected = torch.arange(
        pool.kv_cache_dim, dtype=torch.float32, device=device
    ).to(DTYPE).reshape(1, 1, pool.kv_cache_dim).expand(SIZE, 1, -1).contiguous()
    with _single_rank_parallel():
        for layer in range(LAYER_NUM):
            pool.set_kv_buffer(
                SimpleNamespace(layer_id=pool.start_layer + layer), indices, expected,
                expected
            )
            torch.testing.assert_close(pool.kv_buffer[layer][indices].cpu(),
                                       expected.cpu())

    logical_allocator.free(indices)
    assert logical_allocator.available_size() == SIZE
    allocator.clear()
    assert allocator.null_block == [0]
    assert logical_allocator.available_size() == SIZE
    refilled = logical_allocator.alloc(SIZE)
    assert refilled is not None and len(refilled) == SIZE
    assert logical_allocator.available_size() == 0
    logical_allocator.free(refilled)
    assert logical_allocator.available_size() == SIZE

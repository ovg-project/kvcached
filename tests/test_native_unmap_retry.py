# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.mark.parametrize(
    ("contiguous_layout", "unified_pool"),
    [(True, False), (False, True), (False, False)],
)
@pytest.mark.parametrize("transactional", [False, True])
def test_unmap_retry_skips_fully_unmapped_offsets(
    contiguous_layout, unified_pool, transactional
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native unmap retry validation")
    from kvcached import vmm_ops

    if not hasattr(vmm_ops, "prepare_map_to_kv_tensors"):
        pytest.skip("compiled VMM reservation API is required")
    page_size = 2 * 1024 * 1024
    stride = 4 * page_size if contiguous_layout else page_size
    first, second = 0, stride
    vmm_ops.init_kvcached("cuda:0", page_size, contiguous_layout)
    vmm_ops.create_kv_tensors(8 * page_size, 2, "cuda:0", 2, 2, 0, unified_pool)
    try:
        prepared = vmm_ops.prepare_map_to_kv_tensors("map", [first, second], 0)
        assert prepared["success"]
        assert vmm_ops.commit_prepared_map("map", 0)["success"]
        assert vmm_ops.unmap_from_kv_tensors([first], 0)

        # One earlier worker response may be lost after a successful release.
        # Also cover repeated offsets within the same release request.
        offsets = [first, second, second]
        if transactional:
            assert vmm_ops.prepare_unmap_from_kv_tensors(offsets, "unmap", 0)
            assert vmm_ops.commit_unmap_from_kv_tensors("unmap", 0)
        else:
            assert vmm_ops.unmap_from_kv_tensors(offsets, 0)
        assert vmm_ops.unmap_from_kv_tensors([first, second], 0)

        prepared = vmm_ops.prepare_map_to_kv_tensors("remap", [first, second], 0)
        assert prepared["success"]
        assert prepared["required_bytes"] > 0
        assert vmm_ops.commit_prepared_map("remap", 0)["success"]
        assert vmm_ops.unmap_from_kv_tensors([first, second], 0)
    finally:
        vmm_ops.shutdown_kvcached()


@pytest.mark.parametrize(
    ("contiguous_layout", "unified_pool"),
    [(True, False), (False, True), (False, False)],
)
def test_legacy_map_accepts_existing_and_new_offsets(contiguous_layout, unified_pool):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native map retry validation")
    from kvcached import vmm_ops

    page_size = 2 * 1024 * 1024
    first, second = 0, 4 * page_size if contiguous_layout else page_size
    vmm_ops.init_kvcached("cuda:0", page_size, contiguous_layout)
    vmm_ops.create_kv_tensors(8 * page_size, 2, "cuda:0", 2, 2, 0, unified_pool)
    try:
        assert vmm_ops.map_to_kv_tensors([first, first], 0)
        assert vmm_ops.map_to_kv_tensors([first], 0)
        assert vmm_ops.map_to_kv_tensors_with_result(
            [first, second, second], 0
        ) == (True, [second])
        assert vmm_ops.map_to_kv_tensors([first, second, second], 0)
        assert vmm_ops.map_to_kv_tensors_with_result([first, second], 0) == (True, [])
        assert vmm_ops.unmap_from_kv_tensors([first, second], 0)
    finally:
        vmm_ops.shutdown_kvcached()

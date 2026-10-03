# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Real VMM readback with pages selected after native initialization."""
import pytest
from test_vmm_transaction import _compiled_vmm_ops


@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("unified", [False, True])
def test_pool_page_sizes_do_not_change_existing_or_default_pools(contiguous, unified):
    import torch

    vmm = _compiled_vmm_ops()
    mib = 1024**2
    buffers = 1 if unified else 2
    layers = 2
    vmm.init_kvcached("cuda:0", 2 * mib, contiguous)
    pools = []
    try:
        for group, page in enumerate((4 * mib, 6 * mib, 10 * mib, 2 * mib)):
            kwargs = {"page_size": page} if group < 3 else {}
            raw = vmm.create_kv_tensors(
                4 * page, 1, "cuda:0", layers, num_kv_buffers=buffers,
                group_id=group, unified_pool=unified, **kwargs)
            stride = page * layers * buffers if contiguous else page
            views = ([raw[0][:stride], raw[0][stride:2 * stride]] if contiguous else
                     [tensor[start:start + page]
                      for tensor in raw
                      for start in ([0, page] if unified else [0, page, 2 * page, 3 * page])])
            pools.append((group, stride, views))

        for cycle in range(3):
            for group, stride, views in pools:
                assert vmm.map_to_kv_tensors([0, stride], group)
                for index, view in enumerate(views):
                    view.fill_(10 * group + index + cycle + 1)
            torch.cuda.synchronize()
            for group, stride, views in pools:
                for index, view in enumerate(views):
                    assert torch.all(view == 10 * group + index + cycle + 1).item()
            torch.cuda.synchronize()
            for group, stride, _ in pools:
                assert vmm.unmap_from_kv_tensors([0, stride], group)
    finally:
        vmm.shutdown_kvcached()


@pytest.mark.parametrize("page_size", [-2097152, 1, 3145728])
def test_invalid_tensor_page_size_is_rejected(page_size):
    vmm = _compiled_vmm_ops()
    vmm.init_kvcached("cuda:0", 2097152, True)
    try:
        with pytest.raises((ValueError, RuntimeError), match="page size"):
            vmm.create_kv_tensors(16 * 1024**2, 1, "cuda:0", 1, page_size=page_size)
    finally:
        vmm.shutdown_kvcached()


def test_tensor_page_size_cannot_change_after_creation():
    vmm = _compiled_vmm_ops()
    vmm.init_kvcached("cuda:0", 2097152, True)
    try:
        vmm.create_kv_tensors(16 * 1024**2, 1, "cuda:0", 1, page_size=4 * 1024**2)
        with pytest.raises(RuntimeError, match="Cannot change page size"):
            vmm.create_kv_tensors(16 * 1024**2, 1, "cuda:0", 1, page_size=2 * 1024**2)
    finally:
        vmm.shutdown_kvcached()

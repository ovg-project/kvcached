# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Python wrapper for kvcached VMM operations."""

import torch

# Importing _C defines the PageAllocator / InternalPage classes and registers
# the KV tensor ops as torch.ops.kvcached.*.
from kvcached import _C  # type: ignore[attr-defined]

init_kvcached = torch.ops.kvcached.init_kvcached.default
shutdown_kvcached = torch.ops.kvcached.shutdown_kvcached.default
create_kv_tensors = torch.ops.kvcached.create_kv_tensors.default
kv_tensors_created = torch.ops.kvcached.kv_tensors_created.default
map_to_kv_tensors = torch.ops.kvcached.map_to_kv_tensors.default
unmap_from_kv_tensors = torch.ops.kvcached.unmap_from_kv_tensors.default

# Transactional map/unmap ops and the page-management classes stay on the
# pybind11 module (they take no tensor arguments).
map_to_kv_tensors_with_result = _C.map_to_kv_tensors_with_result
prepare_map_to_kv_tensors = _C.prepare_map_to_kv_tensors
commit_prepared_map = _C.commit_prepared_map
abort_prepared_map = _C.abort_prepared_map
has_prepared_map = _C.has_prepared_map
current_device_pci_bus_id = _C.current_device_pci_bus_id
prepare_unmap_from_kv_tensors = _C.prepare_unmap_from_kv_tensors
commit_unmap_from_kv_tensors = _C.commit_unmap_from_kv_tensors
abort_unmap_from_kv_tensors = _C.abort_unmap_from_kv_tensors
PageAllocator = _C.PageAllocator
InternalPage = _C.InternalPage

__all__ = [
    "InternalPage",
    "PageAllocator",
    "abort_prepared_map",
    "abort_unmap_from_kv_tensors",
    "commit_prepared_map",
    "commit_unmap_from_kv_tensors",
    "create_kv_tensors",
    "current_device_pci_bus_id",
    "has_prepared_map",
    "init_kvcached",
    "kv_tensors_created",
    "map_to_kv_tensors",
    "map_to_kv_tensors_with_result",
    "prepare_map_to_kv_tensors",
    "prepare_unmap_from_kv_tensors",
    "shutdown_kvcached",
    "unmap_from_kv_tensors",
]

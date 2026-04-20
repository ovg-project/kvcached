# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Python wrapper for kvcached VMM operations.

Loads the native stable-ABI library and re-exports operations registered
in PyTorch's dispatcher as module-level functions.
"""

import torch

__import__("kvcached._vmm_ops_lib")  # triggers op registration

init_kvcached = torch.ops.kvcached.init_kvcached.default
shutdown_kvcached = torch.ops.kvcached.shutdown_kvcached.default
create_kv_tensors = torch.ops.kvcached.create_kv_tensors.default
kv_tensors_created = torch.ops.kvcached.kv_tensors_created.default
map_to_kv_tensors = torch.ops.kvcached.map_to_kv_tensors.default
unmap_from_kv_tensors = torch.ops.kvcached.unmap_from_kv_tensors.default

# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Runtime reports must not own CUDA tensors or publish failed allocations."""

import os
import subprocess
import sys

import pytest

SCENARIO = r"""
import gc
import importlib
import json
import sys
import weakref

import torch

runtime = importlib.import_module('kvcached.integration.' + sys.argv[1] + '.interfaces')
torch.cuda.set_device(0)

def allocate(size):
    tensor = torch.empty(size, dtype=torch.uint8, device='cuda:0')
    runtime.register_runtime_owned_reservation(
        'cuda:0', 'workspace', tensor.nbytes, owner=tensor)
    return tensor

target = allocate(4096)
draft = allocate(2048)
target.fill_(7)
draft.fill_(13)
saved = runtime.runtime_reservation_snapshot_dicts()
assert saved[0]['num_bytes'] == 6144 and saved[0]['owner_count'] == 2
json.dumps(saved)
injections = 0
for _ in range(2):
    try:
        allocate(torch.cuda.get_device_properties(0).total_memory * 2)
    except torch.OutOfMemoryError:
        injections += 1
    else:
        raise AssertionError('oversized CUDA allocation unexpectedly succeeded')
    assert runtime.runtime_reservation_snapshot_dicts() == saved
    assert torch.all(target == 7).item() and torch.all(draft == 13).item()
assert injections == 2

target_ref = weakref.ref(target)
del target
gc.collect()
assert target_ref() is None
assert runtime.get_runtime_owned_reservation_bytes('cuda:0') == 2048
assert saved[0]['num_bytes'] == 6144
runtime.register_runtime_owned_reservation('cuda:0', 'workspace', 0, owner=draft)
assert runtime.runtime_reservation_snapshot_dicts() == []
assert torch.all(draft == 13).item()
draft_ref = weakref.ref(draft)
del draft
gc.collect()
assert draft_ref() is None
recovered = allocate(1024)
recovered.fill_(19)
assert torch.all(recovered == 19).item()
assert runtime.get_runtime_owned_reservation_bytes('cuda:0') == 1024
del recovered
gc.collect()
assert runtime.runtime_reservation_snapshot_dicts() == []
print('PASS: injections=2, reports preserved, CUDA readback and recovery, owners released')
"""


@pytest.mark.parametrize("engine", ["sglang", "vllm"])
def test_cuda_owner_lifetime_and_failed_allocation_reports(engine):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    env = dict(os.environ, ENABLE_KVCACHED="false", KVCACHED_AUTOPATCH="0")
    result = subprocess.run([sys.executable, "-c", SCENARIO, engine],
                            env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: injections=2" in result.stdout

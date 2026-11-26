# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""The CPU groups SGLang ranks agree over, and one reduce across them.

Every caller must issue the same reduces in the same order on each rank of
the groups; only the values are rank-local.
"""

import importlib
from typing import Any, List, Optional, Sequence


def _parallel_state() -> Optional[Any]:
    try:
        return importlib.import_module("sglang.srt.distributed.parallel_state")
    except ImportError:
        return None


def queue_cpu_groups() -> List[Any]:
    """CPU groups of the ranks that share this scheduler's waiting queue.

    SGLang hands one request stream to every rank of an attention TP group
    and, under context parallelism, along the attention CP group as well
    (``attn_cp_tp_broadcast_pyobj``), while each of those ranks drives its
    own local kvcached pool. ``get_attn_tp_group`` is the TP group itself
    when DP attention is off. Empty when no peer shares the queue, which
    includes a process without a parallel state.
    """
    parallel_state = _parallel_state()
    if parallel_state is None:
        return []
    groups: List[Any] = []
    for name in ("get_attn_tp_group", "get_attn_cp_group"):
        getter = getattr(parallel_state, name, None)
        if getter is None:
            continue
        try:
            group = getter()
        except AssertionError:  # the group is not initialized
            continue
        if int(group.world_size) <= 1:
            continue
        cpu_group = group.cpu_group
        # The CP group aliases the TP group when their sizes coincide.
        if all(cpu_group is not seen for seen in groups):
            groups.append(cpu_group)
    return groups


def pipeline_stages() -> int:
    """Pipeline stages of this process's model; 1 without a parallel state."""
    parallel_state = _parallel_state()
    if parallel_state is None:
        return 1
    try:
        return int(parallel_state.get_pipeline_model_parallel_world_size())
    except (AttributeError, AssertionError):
        return 1


def pp_cpu_group() -> Optional[Any]:
    """CPU group of this rank's pipeline stages; None without a pipeline."""
    parallel_state = _parallel_state()
    if parallel_state is None:
        return None
    try:
        group = parallel_state.get_pp_group()
    except (AttributeError, AssertionError):
        return None
    return group.cpu_group if int(group.world_size) > 1 else None


def stage_cpu_groups() -> List[Any]:
    """The queue groups and, under a pipeline, the PP group.

    Reducing over these connects every TP, CP and PP rank of one model
    replica.
    """
    groups = queue_cpu_groups()
    pipeline = pp_cpu_group()
    return groups if pipeline is None else [*groups, pipeline]


def reduce_tensor(tensor: Any, groups: Sequence[Any], op: Any) -> None:
    """All-reduce a CPU tensor in place with ``op`` over each group in turn.

    ``op`` is a ``ReduceOp`` or its name. Reducing group after group gives
    the extreme over every rank the groups connect.
    """
    if not groups:
        return
    import torch

    if isinstance(op, str):
        op = getattr(torch.distributed.ReduceOp, op)
    for cpu_group in groups:
        torch.distributed.all_reduce(tensor, op=op, group=cpu_group)


def reduce_counts(counts: Sequence[int], groups: Sequence[Any], op: Any) -> List[int]:
    """``reduce_tensor`` over a list of ints. Without a group nothing is reduced."""
    if not groups:
        return [int(count) for count in counts]
    import torch

    values = torch.tensor(list(counts), dtype=torch.int64)
    reduce_tensor(values, groups, op)
    return [int(value) for value in values.tolist()]

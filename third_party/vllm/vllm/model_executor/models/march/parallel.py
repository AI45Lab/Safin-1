# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scope MoE DP communication sizes to the executor's memory-anchor rows."""

from contextlib import contextmanager

import torch
import torch.distributed as dist

from vllm.config import CUDAGraphMode
from vllm.distributed import get_dp_group
from vllm.forward_context import DPMetadata, get_forward_context


@contextmanager
def hsd_moe_dp_metadata(num_rows: int):
    """Keep scheduler text counts intact while MoE sees expanded local rows.

    All eager DP ranks participate, including ranks without anchors or active
    requests. A local-only update cannot describe an unequal remote prefill.
    Graph decode has no inserted rows and keeps its captured communication;
    it must not introduce a host collective during capture or replay.
    """
    context = get_forward_context()
    original = context.dp_metadata
    if original is None:
        yield
        return

    if context.cudagraph_runtime_mode != CUDAGraphMode.NONE:
        # Dynamo must keep the batch dimension symbolic across Graph sizes.
        # Comparing it with this step's Python descriptor would specialize the
        # compiled model to that one batch. The Graph caller supplies text rows;
        # retain the defensive shape check only for uncompiled execution.
        if not torch.compiler.is_compiling():
            descriptor = context.batch_descriptor
            if descriptor is None or num_rows != descriptor.num_tokens:
                raise RuntimeError("HSD Graph decode cannot insert memory-anchor rows")
        yield
        return

    # The HSD runner disables DBO/microbatching. Do not silently reinterpret a
    # scheduler microbatch's text offsets as expanded memory-anchor offsets.
    if context.ubatch_slices is not None:
        raise NotImplementedError(
            "HSD MoE DP memory anchors do not support microbatching"
        )
    group = get_dp_group()
    counts = torch.zeros_like(original.num_tokens_across_dp_cpu)
    counts[group.rank_in_group] = num_rows
    dist.all_reduce(counts, group=group.cpu_group)
    context.dp_metadata = DPMetadata(counts)
    try:
        yield
    finally:
        context.dp_metadata = original

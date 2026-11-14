# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warmup must cover the pointer offsets the chunked sparse prefill passes."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda_alike():
    pytest.skip("Triton kernels require a GPU", allow_module_level=True)

triton = pytest.importorskip("triton")

from vllm.models.deepseek_v4.common.ops.cache_utils import (  # noqa: E402
    _COMBINE_TOPK_SWA_INDICES_KERNEL,
    combine_topk_swa_indices,
)

INDEX_TOPK = 512
C128A_TOPK = 128
WINDOW = 128
NUM_REQS = 3
QUERY_LEN = 4


def _vllm_config():
    hf_config = SimpleNamespace(
        compress_ratios=[0, 0, 4, 128, 4, 128, 0],
        index_topk=INDEX_TOPK,
        sliding_window=WINDOW,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config, max_model_len=20480),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16384),
    )


@contextmanager
def _record_compiles():
    from triton import knobs

    previous = knobs.runtime.jit_post_compile_hook
    compiled: list[str] = []

    def hook(**kwargs):
        compiled.append(getattr(kwargs.get("fn"), "name", "<unknown>"))
        if previous is not None:
            return previous(**kwargs)
        return None

    knobs.runtime.jit_post_compile_hook = hook
    try:
        yield compiled
    finally:
        knobs.runtime.jit_post_compile_hook = previous


def _int32(values: int, fill: int, offset: int) -> torch.Tensor:
    buffer = torch.full((offset + values,), fill, dtype=torch.int32, device="cuda")
    return buffer[offset:]


# query_start_loc/seq_lens sit at num_decodes + chunk_start, gather_lens at
# chunk_start: both offset (0, 1), or only gather_lens misaligned (3, 1).
@pytest.mark.parametrize("num_decodes, chunk_start", [(0, 1), (3, 1)])
def test_warmup_covers_chunked_prefill_pointer_alignments(num_decodes, chunk_start):
    _COMBINE_TOPK_SWA_INDICES_KERNEL.warmup(_vllm_config())

    batch_offset = num_decodes + chunk_start
    query_start_loc = torch.arange(
        0,
        (batch_offset + NUM_REQS + 1) * QUERY_LEN,
        QUERY_LEN,
        dtype=torch.int32,
        device="cuda",
    )[batch_offset : batch_offset + NUM_REQS + 1]
    seq_lens = _int32(NUM_REQS, 4096, batch_offset)
    gather_lens = _int32(NUM_REQS, 1024, chunk_start)
    with _record_compiles() as compiled:
        for ratio, width, topk in (
            (1, INDEX_TOPK, 0),
            (4, INDEX_TOPK, INDEX_TOPK),
            (128, C128A_TOPK, C128A_TOPK),
        ):
            topk_indices = torch.zeros(
                (NUM_REQS * QUERY_LEN, width), dtype=torch.int32, device="cuda"
            )
            combine_topk_swa_indices(
                topk_indices,
                query_start_loc,
                seq_lens,
                gather_lens,
                WINDOW,
                ratio,
                topk,
                4096,
                4096,
            )
        torch.accelerator.synchronize()

    assert not compiled, f"JIT compiled after warmup: {compiled}"

"""CUDA sparse top-k forward/backward parity, adapted from the original kernel regression."""

import pytest
import torch

from march.kernels.march_global_read_topk import hsd_sparse_topk_global_read, sparse_topk_global_read_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _run_and_grad(
    *,
    q: torch.Tensor,
    states: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    grad_out: torch.Tensor,
    reference: bool,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    q_arg = q.detach().requires_grad_(True)
    state_arg = states.detach().requires_grad_(True)
    weight_arg = weights.detach().requires_grad_(True)
    op = sparse_topk_global_read_reference if reference else hsd_sparse_topk_global_read
    out = op(
        read_q=q_arg,
        states=state_arg,
        topk_indices=indices,
        topk_weights=weight_arg,
        topk_valid=valid,
        read_scale=q.shape[-1] ** -0.5,
    )
    grads = torch.autograd.grad(out, (q_arg, state_arg, weight_arg), grad_out)
    return out, grads


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("topk", (1, 2, 4, 8, 16))
@pytest.mark.parametrize("execution_mode", ("grouped", "token"))
def test_sparse_topk_cuda_modes_match_reference(monkeypatch, dtype, topk, execution_mode,
                                                heads=2, key_dim=32, value_dim=48):
    monkeypatch.setenv("HSD_TOPK_GLOBAL_READ_STRICT", "1")
    monkeypatch.setenv("HSD_TOPK_EXECUTION_MODE", execution_mode)
    monkeypatch.setenv("HSD_TOPK_GROUP_SPLITS", "1")
    torch.manual_seed(7100 + topk)
    batch, text_len, anchors = 1, 19, 11
    q = torch.randn(batch, text_len, heads, key_dim, device="cuda", dtype=dtype)
    states = torch.randn(
        batch, anchors, heads, key_dim, value_dim, device="cuda", dtype=torch.float32
    )
    indices = torch.randint(0, anchors, (batch, text_len, heads, topk), device="cuda")
    valid = torch.rand(batch, text_len, heads, topk, device="cuda") > 0.2
    weights = torch.rand(batch, text_len, heads, topk, device="cuda", dtype=torch.float32) * valid
    grad_out = torch.randn(batch, text_len, heads, value_dim, device="cuda", dtype=dtype)

    actual = _run_and_grad(
        q=q,
        states=states,
        indices=indices,
        weights=weights,
        valid=valid,
        grad_out=grad_out,
        reference=False,
    )
    expected = _run_and_grad(
        q=q,
        states=states,
        indices=indices,
        weights=weights,
        valid=valid,
        grad_out=grad_out,
        reference=True,
    )
    torch.testing.assert_close(actual[0], expected[0], rtol=2e-2, atol=2e-2)
    for actual_grad, expected_grad in zip(actual[1], expected[1]):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-2, atol=2e-2)
        assert bool(torch.isfinite(actual_grad).all().item())


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("execution_mode", ("grouped", "token"))
def test_sparse_topk_production_geometry(monkeypatch, dtype, execution_mode):
    # Both supplied production checkpoints use 32 value heads and 128x128 states.
    test_sparse_topk_cuda_modes_match_reference(
        monkeypatch, dtype, 4, execution_mode, heads=32, key_dim=128, value_dim=128)

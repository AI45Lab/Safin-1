"""GDN forward and all six input gradients against an FP32 recurrence oracle."""
import importlib.util
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


def recurrence(inputs, boundaries, packed):
    q, k, v, beta, g, anchor_q = inputs
    batch, length, heads, dim = q.shape
    outputs, states, anchors = [], [], []
    state = torch.zeros(batch, heads, dim, v.shape[-1], device=q.device)
    for index in range(length):
        if packed and index == length // 2:
            state = torch.zeros_like(state)
        key = k[:, index].float()
        state = state * g[:, index].float().exp()[..., None, None]
        delta = (v[:, index].float() - torch.einsum('bhk,bhkv->bhv', key, state)) * beta[:, index].float()[..., None]
        state = state + key[..., None] * delta[..., None, :]
        outputs.append(torch.einsum('bhk,bhkv->bhv', q[:, index].float(), state) * dim**-0.5)
        if index + 1 in boundaries:
            anchor = len(states)
            states.append(state)
            anchors.append(torch.einsum('bhk,bhkv->bhv', anchor_q[:, anchor].float(), state) * dim**-0.5)
    return torch.stack(outputs, 1).to(q.dtype), torch.stack(anchors, 1).to(q.dtype), torch.stack(states, 1)


@pytest.mark.parametrize('dim', [96, 128])
@pytest.mark.parametrize('packed', [False, True])
def test_chunk_forward_backward(dim, packed, monkeypatch):
    from march.kernels import march_chunk_gdn as chunk
    assert chunk.triton is not None, 'installed Triton/FLA backend unavailable'
    torch.manual_seed(761)
    batch, length, heads = (1 if packed else 2), 128, 16
    shape = (batch, length, heads, dim)
    q = torch.nn.functional.normalize(torch.randn(shape, device='cuda'), dim=-1).to(torch.bfloat16)
    k = torch.nn.functional.normalize(torch.randn(shape, device='cuda'), dim=-1).to(torch.bfloat16)
    v = (torch.randn(shape, device='cuda') * 0.1).to(torch.bfloat16)
    beta = torch.rand(batch, length, heads, device='cuda') * 0.5
    g = -torch.rand(batch, length, heads, device='cuda') * 0.08
    anchor_q = torch.nn.functional.normalize(torch.randn(batch, 2, heads, dim, device='cuda'), dim=-1).to(torch.bfloat16)
    source = [q, k, v, beta, g, anchor_q]
    boundaries = [64, 128]
    kwargs = dict(boundary_counts=torch.tensor(boundaries, device='cuda'))
    if packed:
        kwargs.update(text_cu_seqlens=torch.tensor([0, 64, 128], device='cuda'),
                      anchor_cu_seqlens=torch.tensor([0, 1, 2], device='cuda'))
    inputs = [x.detach().clone().requires_grad_() for x in source]
    reference_inputs = [x.detach().clone().requires_grad_() for x in source]
    actual = chunk.hsd_chunk_gated_delta_rule_with_anchor_states(
        query=inputs[0], key=inputs[1], value=inputs[2], beta=inputs[3], g=inputs[4], anchor_q=inputs[5], **kwargs)
    assert actual is not None, 'requested training kernel did not run'
    expected = recurrence(reference_inputs, boundaries, packed)
    for value, oracle in zip(actual, expected, strict=True):
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, oracle, atol=2e-3, rtol=2e-2)
    torch.manual_seed(11)
    weights = [torch.randn_like(value) * (0.01 if i == 2 else 1) for i, value in enumerate(expected)]
    sum((value.float() * weight.float()).sum() for value, weight in zip(actual, weights)).backward()
    sum((value.float() * weight.float()).sum() for value, weight in zip(expected, weights)).backward()
    for value, oracle in zip(inputs, reference_inputs, strict=True):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        torch.testing.assert_close(value.grad, oracle.grad, atol=6e-3, rtol=4e-2)

    baseline = os.environ.get('MARCH_CHUNK_BASELINE')
    if baseline:
        spec = importlib.util.spec_from_file_location('march_chunk_before', Path(baseline))
        previous = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(previous)
        before_inputs = [x.detach().clone().requires_grad_() for x in source]
        before = previous.hsd_chunk_gated_delta_rule_with_anchor_states(
            query=before_inputs[0], key=before_inputs[1], value=before_inputs[2], beta=before_inputs[3],
            g=before_inputs[4], anchor_q=before_inputs[5], **kwargs)
        assert before is not None
        for value, old in zip(actual, before, strict=True):
            torch.testing.assert_close(value, old, atol=2e-3, rtol=2e-2)
        sum((value.float() * weight.float()).sum() for value, weight in zip(before, weights)).backward()
        for value, old in zip(inputs, before_inputs, strict=True):
            torch.testing.assert_close(value.grad, old.grad, atol=6e-3, rtol=4e-2)

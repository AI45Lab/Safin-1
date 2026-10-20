"""JIT warmup after checkpoint load must not advance the next dropout mask."""
import ast
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def warmup_wrapper(body):
    source = ROOT / 'swift/megatron/utils/megatron_lm_utils.py'
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'warmup_jit_function')
    namespace = {'torch': torch, '_warmup_jit_function': body}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['warmup_jit_function']


@pytest.mark.parametrize('fail', [False, True])
def test_warmup_restores_rng_and_uses_only_current_rank_device(monkeypatch, fail):
    original_context = torch.random.fork_rng
    devices = []

    @contextmanager
    def cpu_context(*, devices):
        assert devices == [3]
        with original_context(devices=[]):
            yield

    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 3)
    monkeypatch.setattr(torch.random, 'fork_rng', cpu_context)

    def random_body(config, args):
        devices.append(torch.cuda.current_device())
        torch.rand(128)
        if fail:
            raise RuntimeError('warmup failed')

    before = torch.get_rng_state().clone()
    try:
        warmup_wrapper(random_body)(None, None)
    except RuntimeError:
        assert fail
    else:
        assert not fail
    assert devices == [3]
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA dropout required')
@pytest.mark.parametrize('fail', [False, True])
def test_cuda_warmup_preserves_next_dropout_mask(fail):
    state = torch.cuda.get_rng_state().clone()
    cpu_state = torch.get_rng_state().clone()
    value = torch.ones(64, 128, device='cuda')
    expected = torch.nn.functional.dropout(value, p=0.05, training=True)
    torch.cuda.set_rng_state(state)

    def random_body(config, args):
        torch.rand(128)
        torch.rand(128, device='cuda')
        torch.nn.functional.dropout(value, p=0.1, training=True)
        if fail:
            raise RuntimeError('warmup failed')

    try:
        warmup_wrapper(random_body)(None, None)
    except RuntimeError:
        assert fail
    else:
        assert not fail
    assert torch.equal(torch.get_rng_state(), cpu_state)
    assert torch.equal(torch.cuda.get_rng_state(), state)
    actual = torch.nn.functional.dropout(value, p=0.05, training=True)
    assert torch.equal(expected, actual)

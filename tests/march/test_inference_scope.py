"""CPU contracts for vendored numeric paths without importing native vLLM."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / 'third_party/vllm/vllm/model_executor/models'


def method(path, class_name, method_name, namespace):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name))
    fn.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace[method_name]


@pytest.mark.parametrize('tp_size,rank,input_is_parallel,reduce_results,skip_bias,return_bias', [
    (1, 0, True, True, False, False),
    (1, 0, False, True, True, True),
    (2, 0, False, True, False, True),
    (2, 1, False, True, False, False),
    (2, 1, True, False, True, True),
])
def test_row_parallel_uses_standard_gemm_and_reduction(tp_size, rank, input_is_parallel, reduce_results, skip_bias,
                                                       return_bias):
    torch.manual_seed(12)
    x = torch.randn(3, 4 if input_is_parallel else 4 * tp_size)
    weight, bias = torch.randn(5, 4), torch.randn(5)
    calls = []

    def reduce(value):
        calls.append('reduce')
        return value * tp_size

    fn = method(
        ROOT / 'third_party/vllm/vllm/model_executor/layers/linear.py', 'RowParallelLinear', 'forward', {
            'tensor_model_parallel_all_reduce': reduce,
            'split_tensor_along_last_dim': lambda value, num_partitions: value.chunk(num_partitions, -1),
        })
    layer = SimpleNamespace(
        input_is_parallel=input_is_parallel,
        tp_rank=rank,
        tp_size=tp_size,
        skip_bias_add=skip_bias,
        bias=bias,
        reduce_results=reduce_results,
        return_bias=return_bias,
        quant_method=SimpleNamespace(apply=lambda _, value, b: torch.nn.functional.linear(value, weight, b)))
    result = fn(layer, x)
    local = x if input_is_parallel else x.chunk(tp_size, -1)[rank]
    expected = torch.nn.functional.linear(local, weight, None if rank > 0 or skip_bias else bias)
    if reduce_results and tp_size > 1:
        expected = expected * tp_size
    actual = result[0] if return_bias else result
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert calls == (['reduce'] if reduce_results and tp_size > 1 else [])
    if return_bias:
        assert result[1] is (bias if skip_bias else None)


@pytest.mark.parametrize('gated,sidecar', [(False, False), (False, True), (True, False), (True, True)])
def test_full_attention_default_projection_norm_gate(gated, sidecar):
    torch.manual_seed(24)
    x = torch.randn(3, 8)
    q_size = 4
    qkv_weight = torch.randn(q_size * (2 if gated else 1) + 8, 8)
    out_weight = torch.randn(8, 4)
    q_norm = lambda value: value / (value.square().mean(-1, keepdim=True) + 1e-5).sqrt()
    layer = SimpleNamespace(
        q_size=4,
        kv_size=4,
        num_heads=2,
        num_kv_heads=2,
        head_dim=2,
        attn_output_gate=gated,
        qkv_proj=lambda value: (torch.nn.functional.linear(value, qkv_weight), None),
        o_proj=lambda value: (torch.nn.functional.linear(value, out_weight), None),
        q_norm=q_norm,
        k_norm=q_norm,
        rotary_emb=lambda positions, q, k: (q, k),
        attn=lambda q, k, v: q + k + v,
        _capture_hsd_prefill_passive_sidecar=lambda *args: None,
        _capture_hsd_graph_v3_tensor_sidecar=lambda *args: None,
    )
    fn = method(MODELS / 'qwen3_next.py', 'Qwen3NextAttention', 'forward', {'torch': torch})
    actual = torch.empty_like(x)
    fn(layer, torch.arange(3), actual, x, capture_hsd_graph_v3_sidecar=sidecar)
    projected = torch.nn.functional.linear(x, qkv_weight)
    if gated:
        q_gate, k, v = projected.split((8, 4, 4), -1)
        q, gate = q_gate.view(3, 2, 4).chunk(2, -1)
        q, gate = q.reshape(3, 4), gate.reshape(3, 4)
    else:
        q, k, v = projected.split((4, 4, 4), -1)
    attention = q_norm(q.view(3, 2, 2)).view(3, 4) + q_norm(k.view(3, 2, 2)).view(3, 4) + v
    if gated:
        attention = attention * gate.sigmoid()
    expected = torch.nn.functional.linear(attention, out_weight)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_public_cli_routes_exclude_experimental_training():
    from swift.cli.main import ROUTE_MAPPING
    from swift.cli._megatron.main import ROUTE_MAPPING as megatron_routes

    assert {'pt', 'sft', 'infer', 'export', 'eval'} <= ROUTE_MAPPING.keys()
    assert not {'rlhf', 'rollout', 'app', 'web-ui', 'sample'} & ROUTE_MAPPING.keys()
    assert set(megatron_routes) == {'pt', 'sft', 'export'}

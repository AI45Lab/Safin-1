"""CPU contracts for real MoE DP metadata/dispatch with executor anchor rows."""

import ast
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import IntEnum
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
VLLM = ROOT / "third_party/vllm/vllm"


def definitions(path, names, namespace):
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert len(nodes) == len(names)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


@pytest.fixture
def parallel(monkeypatch):
    class Mode(IntEnum):
        NONE = 0
        PIECEWISE = 1
        FULL = 2

    context = NS(dp_metadata=None, cudagraph_runtime_mode=Mode.NONE, ubatch_slices=None,
                 batch_descriptor=NS(num_tokens=32))
    group = NS(rank_in_group=0, cpu_group=object())
    calls = []
    expected = [514, 8015]

    def dependency(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    dependency("vllm.config", CUDAGraphMode=Mode)
    dependency("vllm.distributed", get_dp_group=lambda: group)
    forward = dependency("vllm.forward_context", get_forward_context=lambda: context,
                         torch=torch, dataclass=dataclass, contextmanager=contextmanager)
    definitions(VLLM / "forward_context.py", {"DPMetadata", "_compute_sp_num_tokens"}, forward.__dict__)

    def reduce(counts, group):
        assert group is parallel_group.cpu_group
        assert counts.tolist() == [expected[i] if i == parallel_group.rank_in_group else 0 for i in range(2)]
        counts.copy_(torch.tensor(expected, dtype=counts.dtype))
        calls.append("all_reduce")

    parallel_group = group
    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    path = VLLM / "model_executor/models/march/parallel.py"
    spec = importlib.util.spec_from_file_location("_march_parallel_contract", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    spec.loader.exec_module(module)

    def gather(tensors, dim, sizes):
        assert sizes == expected and dim == 0
        calls.append("dispatch")
        return [torch.cat([tensor, tensor.new_zeros(sum(sizes) - len(tensor), *tensor.shape[1:])]) for tensor in tensors]

    group.all_gatherv = gather
    native = {"__name__": __name__, "torch": torch, "All2AllManagerBase": object,
              "get_forward_context": lambda: context, "get_dp_group": lambda: group,
              "get_ep_group": lambda: group}
    definitions(VLLM / "distributed/device_communicators/all2all.py", {"AgRsAll2AllManager"}, native)
    manager = object.__new__(native["AgRsAll2AllManager"])
    return NS(scope=module.hsd_moe_dp_metadata, context=context, group=group,
              Metadata=forward.DPMetadata, Mode=Mode, calls=calls, expected=expected, manager=manager)


@pytest.mark.parametrize("counts", [(514, 8015), (511, 514), (0, 514), (513, 0)])
@pytest.mark.parametrize("rank", [0, 1])
def test_unequal_anchor_rows_drive_native_moe_dispatch(parallel, counts, rank):
    p = parallel
    p.expected[:] = counts
    p.group.rank_in_group = rank
    original = p.Metadata(torch.tensor([513, 8000], dtype=torch.int32))
    p.context.dp_metadata = original
    rows = counts[rank]
    with p.scope(rows):
        assert p.context.dp_metadata is not original
        with p.context.dp_metadata.sp_local_sizes(1):
            hidden, weights, ids = p.manager.dispatch(torch.zeros(rows, 2), torch.zeros(rows, 1), torch.zeros(rows, 1, dtype=torch.long))
        assert len(hidden) == len(weights) == len(ids) == sum(counts)
    assert p.context.dp_metadata is original
    assert original.num_tokens_across_dp_cpu.tolist() == [513, 8000]
    assert original.local_sizes is None
    assert p.calls == ["all_reduce", "dispatch"]


def test_metadata_is_restored_after_decoder_failure(parallel):
    p = parallel
    original = p.Metadata(torch.tensor([513, 8000]))
    p.context.dp_metadata = original
    with pytest.raises(RuntimeError, match="decoder failure"):
        with p.scope(514):
            raise RuntimeError("decoder failure")
    assert p.context.dp_metadata is original


def test_dense_or_dp1_has_no_collective(parallel):
    with parallel.scope(514):
        assert parallel.context.dp_metadata is None
    assert not parallel.calls


@pytest.mark.parametrize("mode", ["PIECEWISE", "FULL"])
def test_graph_keeps_captured_counts_and_has_no_host_collective(parallel, mode):
    p = parallel
    p.context.cudagraph_runtime_mode = p.Mode[mode]
    original = p.Metadata(torch.tensor([32, 32]))
    p.context.dp_metadata = original
    with p.scope(32):
        assert p.context.dp_metadata is original
    assert not p.calls
    with pytest.raises(RuntimeError, match="cannot insert memory-anchor"):
        with p.scope(33):
            pytest.fail("Invalid Graph rows entered the decoder")
    assert p.context.dp_metadata is original


def test_microbatch_rejected_before_collective(parallel):
    p = parallel
    p.context.dp_metadata = p.Metadata(torch.tensor([513, 8000]))
    p.context.ubatch_slices = [object()]
    with pytest.raises(NotImplementedError, match="microbatching"):
        with p.scope(514):
            pytest.fail("Unsupported offsets entered the decoder")
    assert not p.calls


@pytest.mark.parametrize("entrypoint", ["execute_model", "_dummy_run"])
@pytest.mark.parametrize("variant", ["moe_dp_eager", "moe_dp_graph", "moe_single_gpu", "dense_dp", "unanchored"])
def test_real_compile_wrapper_reenters_dp_scope_on_every_eager_step(parallel, entrypoint, variant):
    """Run native compile bypass against both actual runner call-site flags.

    The cached branch deliberately has no Python metadata scope, as in the
    observed AOT Graph path. Eager DP must reach the real scope on every call,
    with changing local/remote anchor counts, including an idle local rank.
    """
    p = parallel
    runner_path = VLLM / "v1/worker/gpu_model_runner.py"
    runner_tree = ast.parse(runner_path.read_text())
    cls = next(node for node in runner_tree.body if isinstance(node, ast.ClassDef) and node.name == "GPUModelRunner")
    policy = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_historical_hsd_moe_dp_needs_python")
    native_method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == entrypoint)
    flags = [keyword.value for call in ast.walk(native_method) if isinstance(call, ast.Call)
             and isinstance(call.func, ast.Name) and call.func.id == "set_forward_context"
             for keyword in call.keywords if keyword.arg == "skip_compiled"]
    assert len(flags) == 1
    namespace = {"CUDAGraphMode": p.Mode}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[policy], type_ignores=[])), str(runner_path), "exec"), namespace)
    mode = p.Mode.FULL if variant == "moe_dp_graph" else p.Mode.NONE
    runner = NS(parallel_config=NS(data_parallel_size=1 if variant == "moe_single_gpu" else 2,
                                  is_moe_model=variant != "dense_dp"),
                _uses_historical_hsd=lambda: variant != "unanchored",
                _enable_historical_hsd_graph_v3=lambda: True)
    policy_function = namespace[policy.name]
    runner._historical_hsd_moe_dp_needs_python = lambda value: policy_function(runner, value)
    p.context.skip_compiled = eval(compile(ast.Expression(flags[0]), str(runner_path), "eval"),
                                  {"self": runner, "has_encoder_input": False,
                                   "uses_historical_hsd": runner._uses_historical_hsd(),
                                   "hsd_padded_static_decode": True, "hsd_static_dummy_decode": True,
                                   "cudagraph_mode": mode, "cudagraph_runtime_mode": mode})
    tree = ast.parse((VLLM / "compilation/decorators.py").read_text())
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "__call__")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), call], type_ignores=[])
    namespace = {"torch": torch, "is_forward_context_available": lambda: True,
                 "get_forward_context": lambda: p.context,
                 "maybe_use_cudagraph_partition_wrapper": lambda _: nullcontext()}
    exec(compile(ast.fix_missing_locations(module), str(VLLM / "compilation/decorators.py"), "exec"), namespace)
    p.context.cudagraph_runtime_mode = mode
    original = p.Metadata(torch.tensor([513, 8000], dtype=torch.int32))
    p.context.dp_metadata = original

    def forward(rows):
        with p.scope(rows):
            with p.context.dp_metadata.sp_local_sizes(1):
                p.manager.dispatch(torch.zeros(rows, 2), torch.zeros(rows, 1), torch.zeros(rows, 1, dtype=torch.long))
        return "python"

    model = NS(do_not_compile=False, forward=forward,
               aot_compiled_fn=lambda *args: "captured", vllm_config=None)
    for counts in [(514, 8015), (1, 2), (0, 513)]:
        p.expected[:] = counts
        result = namespace["__call__"](model, counts[0])
        assert result == ("python" if variant == "moe_dp_eager" else "captured")
        assert p.context.dp_metadata is original
    assert p.calls == (["all_reduce", "dispatch"] * 3 if variant == "moe_dp_eager" else [])


@pytest.mark.parametrize("mode", ["FULL", "PIECEWISE"])
def test_actual_torch_compile_keeps_graph_batch_dimension_dynamic(parallel, mode):
    """A real Dynamo trace must accept multiple batches in the [1,32] range.

    Comparing a symbolic row count with the current scheduler descriptor pins
    that dimension to the tracing batch, even though other Graph batches are
    valid. The descriptor changes between calls as it does in the runner.
    """
    p = parallel
    p.context.cudagraph_runtime_mode = p.Mode[mode]
    p.context.dp_metadata = p.Metadata(torch.tensor([32, 32], dtype=torch.int32))

    def forward(value):
        with p.scope(int(value.shape[0])):
            return value.sin()

    traces = []

    def backend(graph, inputs):
        traces.append(graph)
        return graph.forward

    with torch._dynamo.config.patch(recompile_limit=8):
        compiled = torch.compile(forward, backend=backend, fullgraph=True, dynamic=True)
        for rows in (32, 8, 4):
            value = torch.randn(rows, 3)
            torch._dynamo.mark_dynamic(value, 0, min=1, max=32)
            p.context.batch_descriptor.num_tokens = rows
            p.context.dp_metadata.num_tokens_across_dp_cpu.fill_(rows)
            torch.testing.assert_close(compiled(value), value.sin())
    assert len(traces) == 1
    assert not p.calls

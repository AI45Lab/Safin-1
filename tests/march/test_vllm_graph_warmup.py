"""Execute native runner/wrapper control flow with mocked CUDA boundaries.

The kernel spy models cold Inductor autotuning (a host synchronization), not
GPU numerics. A real Graph mode must compile/warm outside capture even when
ordinary eager MoE DP warmups bypass the cached model.
"""

import ast
from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import ContextVar, copy_context
import dataclasses
from dataclasses import dataclass
from enum import IntEnum
import os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import weakref

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
VLLM = ROOT / "third_party/vllm/vllm"


class Mode(IntEnum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


@dataclass(frozen=True)
class Descriptor:
    num_tokens: int
    uniform: bool
    num_active_loras: int


def load_definitions(path, names, namespace, owner=None):
    nodes = ast.parse(path.read_text()).body
    if owner:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner
        ).body
    definitions = [
        n
        for n in nodes
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    assert len(definitions) == len(names)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.Module(body=[future, *definitions], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


@pytest.fixture
def native(monkeypatch):
    monkeypatch.delenv("VLLM_HSD_VARLEN_COUNTER_FILE", raising=False)
    events = []
    context = NS(
        cudagraph_runtime_mode=Mode.NONE, batch_descriptor=None, skip_compiled=False
    )
    state = NS(capturing=False, tuned=set())

    def synchronize():
        assert not state.capturing, (
            "cold autotune synchronized inside CUDA Graph capture"
        )
        events.append("sync")

    @contextmanager
    def graph(*args, **kwargs):
        assert not state.capturing
        events.append("capture")
        state.capturing = True
        try:
            yield
        finally:
            state.capturing = False

    monkeypatch.setattr(torch.accelerator, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "graph", graph)
    monkeypatch.setattr(
        torch.cuda, "CUDAGraph", lambda: NS(replay=lambda: events.append("replay"))
    )
    namespace = {
        "torch": torch,
        "os": os,
        "weakref": weakref,
        "ContextVar": ContextVar,
        "contextmanager": contextmanager,
        "dataclasses": dataclasses,
        "CUDAGraphMode": Mode,
        "is_forward_context_available": lambda: True,
        "get_forward_context": lambda: context,
        "ExitStack": ExitStack,
        "patch": patch,
        "logger": NS(debug=lambda *a: None),
        "validate_cudagraph_capturing_enabled": lambda: events.append("validate"),
        "set_graph_pool_id": lambda value: None,
        "current_platform": NS(graph_pool_handle=lambda: None),
        "current_stream": lambda: None,
        "weak_ref_tensors": lambda value: value,
        "compilation_counter": NS(num_cudagraph_captured=0),
        "get_offloader": lambda: NS(
            sync_prev_onload=lambda: None, join_after_forward=lambda: None
        ),
    }
    load_definitions(
        VLLM / "compilation/cuda_graph.py",
        {"CUDAGraphEntry", "CUDAGraphWrapper"},
        namespace,
    )
    wrapper_cls = namespace["CUDAGraphWrapper"]

    def wrapper(runnable, mode):
        instance = object.__new__(wrapper_cls)
        instance.runnable = runnable
        instance.runtime_mode = mode
        instance.graph_pool = None
        instance.is_debugging_mode = False
        instance.cudagraph_options = NS(
            debug_log_enable=False, gc_disable=False, weak_ref_output=False
        )
        instance.concrete_cudagraph_entries = {}
        return instance

    compile_namespace = {
        "torch": torch,
        "is_forward_context_available": lambda: True,
        "get_forward_context": lambda: context,
        "maybe_use_cudagraph_partition_wrapper": lambda _: nullcontext(),
    }
    # Use the real decorated-model __call__, including the cached-AOT bypass.
    tree = ast.parse((VLLM / "compilation/decorators.py").read_text())
    call = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "__call__"
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[
                        ast.ImportFrom(
                            module="__future__",
                            names=[ast.alias(name="annotations")],
                            level=0,
                        ),
                        call,
                    ],
                    type_ignores=[],
                )
            ),
            str(VLLM / "compilation/decorators.py"),
            "exec",
        ),
        compile_namespace,
    )
    runner_namespace = {
        "CUDAGraphMode": Mode,
        "CUDAGraphWrapper": wrapper_cls,
        "torch": torch,
    }
    load_definitions(
        VLLM / "v1/worker/gpu_model_runner.py",
        {"_warmup_and_capture", "_historical_hsd_moe_dp_needs_python"},
        runner_namespace,
        owner="GPUModelRunner",
    )
    return NS(
        events=events,
        context=context,
        state=state,
        wrapper=wrapper,
        Wrapper=wrapper_cls,
        namespace=namespace,
        runner_namespace=runner_namespace,
        compiled_call=compile_namespace["__call__"],
    )


@pytest.mark.parametrize("mode", [Mode.FULL, Mode.PIECEWISE])
@pytest.mark.parametrize("warmups", [0, 2])
@pytest.mark.parametrize("tokens", [8, 64])
def test_cold_compiled_kernel_is_warmed_before_first_capture(
    native, mode, warmups, tokens
):
    n = native
    desc = Descriptor(tokens, True, 2)
    runner = NS(
        compilation_config=NS(cudagraph_num_of_warmups=warmups),
        max_num_reqs=32,
        parallel_config=NS(data_parallel_size=2, is_moe_model=True),
        _uses_historical_hsd=lambda: True,
        _enable_historical_hsd_graph_v3=lambda: True,
    )
    policy = n.runner_namespace["_historical_hsd_moe_dp_needs_python"]
    runner._historical_hsd_moe_dp_needs_python = lambda value: policy(runner, value)

    def kernel(*args):
        key = (n.context.cudagraph_runtime_mode, n.context.batch_descriptor.num_tokens)
        if key not in n.state.tuned:
            torch.accelerator.synchronize()
            n.state.tuned.add(key)
        n.events.append("compiled kernel")
        return "output"

    # A FULL wrapper outside PIECEWISE partitions models the nested native path.
    inner = n.wrapper(kernel, Mode.PIECEWISE)
    model = NS(
        do_not_compile=False,
        forward=lambda *a: n.events.append("python eager"),
        aot_compiled_fn=inner,
        vllm_config=None,
    )
    outer = n.wrapper(lambda: n.compiled_call(model), Mode.FULL)
    calls = []

    def dummy_run(num_tokens, **kwargs):
        calls.append((num_tokens, kwargs))
        n.context.cudagraph_runtime_mode = kwargs["cudagraph_runtime_mode"]
        n.context.batch_descriptor = desc
        n.context.skip_compiled = policy(runner, n.context.cudagraph_runtime_mode)
        return outer()

    runner._dummy_run = dummy_run
    n.runner_namespace["_warmup_and_capture"](
        runner, desc, mode, profile_seq_lens=16, allow_microbatching=False
    )
    assert n.events.count("python eager") == warmups
    assert n.events.count("compiled kernel") == 2  # uncaptured warmup, then capture
    assert n.events.index("compiled kernel") < n.events.index("capture")
    assert n.events.count("capture") == n.events.count("validate") == 1
    assert n.namespace["compilation_counter"].num_cudagraph_captured == 1
    assert len(calls) == warmups + 2
    warmup_call, capture_call = calls[-2:]
    assert warmup_call == capture_call  # same mode, padding, LoRA and metadata recipe
    assert warmup_call[0] == min(tokens, runner.max_num_reqs)
    assert warmup_call[1]["uniform_decode"] is True
    assert warmup_call[1]["num_active_loras"] == 2
    assert warmup_call[1]["profile_seq_lens"] == 16
    assert warmup_call[1]["is_graph_capturing"] is True
    # The scope must end before real capture; an ordinary call now replays.
    outer()
    assert n.events[-1] == "replay"
    assert n.events.count("compiled kernel") == 2
    # Real eager DP must still execute its Python metadata scope every time.
    n.context.cudagraph_runtime_mode = Mode.NONE
    n.context.skip_compiled = policy(runner, Mode.NONE)
    outer()
    assert n.events[-1] == "python eager"


@pytest.mark.parametrize(
    "variant", ["dense_dp", "moe_dp1", "unanchored", "graph_v3_disabled"]
)
def test_other_models_keep_existing_warmup_policy(native, variant):
    n = native
    runner = NS(
        compilation_config=NS(cudagraph_num_of_warmups=2),
        max_num_reqs=32,
        parallel_config=NS(
            data_parallel_size=1 if variant == "moe_dp1" else 2,
            is_moe_model=variant != "dense_dp",
        ),
        _uses_historical_hsd=lambda: variant != "unanchored",
        _enable_historical_hsd_graph_v3=lambda: variant != "graph_v3_disabled",
    )
    policy = n.runner_namespace["_historical_hsd_moe_dp_needs_python"]
    runner._historical_hsd_moe_dp_needs_python = lambda value: policy(runner, value)
    calls = []
    runner._dummy_run = lambda *args, **kwargs: calls.append(kwargs)
    desc = NS(num_tokens=8, uniform=True, num_active_loras=0)
    n.runner_namespace["_warmup_and_capture"](runner, desc, Mode.FULL)
    assert [c["cudagraph_runtime_mode"] for c in calls] == [
        Mode.NONE,
        Mode.NONE,
        Mode.FULL,
    ]


def test_warmup_scope_bypasses_nested_capture_and_replay_without_caching(native):
    n = native
    n.context.cudagraph_runtime_mode = Mode.FULL
    n.context.batch_descriptor = "batch"
    inner = n.wrapper(lambda: n.events.append("inner kernel"), Mode.FULL)
    outer = n.wrapper(inner, Mode.FULL)
    with n.Wrapper.warmup():
        outer()
        assert not outer.concrete_cudagraph_entries
        assert not inner.concrete_cudagraph_entries
        with n.Wrapper.warmup():
            outer()
        # Exiting a nested scope must leave the outer warmup active.
        outer()
        lazy = n.wrapper(lambda: n.events.append("lazy kernel"), Mode.FULL)
        lazy()
        assert not lazy.concrete_cudagraph_entries
    assert n.events == ["inner kernel"] * 3 + ["lazy kernel"]
    # Existing cached entries must also be bypassed, not replayed during warmup.
    inner.concrete_cudagraph_entries["batch"] = NS(
        cudagraph=NS(replay=lambda: n.events.append("replay")), output="cached"
    )
    with n.Wrapper.warmup():
        inner()
    assert n.events[-1] == "inner kernel"
    inner()
    assert n.events[-1] == "replay"


def test_warmup_exception_restores_scope_and_other_context_is_unchanged(native):
    n = native
    independent = copy_context()
    with pytest.raises(ValueError, match="warmup failed"):
        with n.Wrapper.warmup():
            assert n.Wrapper._warmup_active.get()
            assert not independent.run(n.Wrapper._warmup_active.get)
            raise ValueError("warmup failed")
    assert not n.Wrapper._warmup_active.get()


def test_runner_does_not_capture_after_failed_compiled_warmup(native):
    n = native
    runner = NS(
        compilation_config=NS(cudagraph_num_of_warmups=0),
        max_num_reqs=32,
        _enable_historical_hsd_graph_v3=lambda: True,
        _historical_hsd_moe_dp_needs_python=lambda mode: True,
    )
    calls = []

    def fail_warmup(*args, **kwargs):
        calls.append(kwargs)
        assert n.Wrapper._warmup_active.get()
        raise RuntimeError("compiled warmup failed")

    runner._dummy_run = fail_warmup
    with pytest.raises(RuntimeError, match="compiled warmup failed"):
        n.runner_namespace["_warmup_and_capture"](
            runner, Descriptor(8, True, 0), Mode.FULL
        )
    assert len(calls) == 1
    assert not n.Wrapper._warmup_active.get()
    assert not n.events  # neither synchronization nor capture after the failure

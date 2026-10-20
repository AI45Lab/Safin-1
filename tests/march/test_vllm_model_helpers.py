"""Exercise the real Safin-1 layout/telemetry helpers without a CUDA engine."""

import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "third_party/vllm/vllm/model_executor/models/march"


@pytest.fixture(scope="module")
def helpers():
    # Load the helper package directly so parent model imports cannot require
    # native vLLM extensions on CPU-only test hosts.
    name = "_safin1_helper_contract"
    spec = importlib.util.spec_from_file_location(
        name, PACKAGE / "__init__.py", submodule_search_locations=[str(PACKAGE)]
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / "third_party/vllm"))
        spec.loader.exec_module(package)
        runtime = importlib.import_module(name + ".runtime")
        layout = importlib.import_module(name + ".layout")
        diagnostics = importlib.import_module(name + ".diagnostics")
        yield SimpleNamespace(runtime=runtime, layout=layout, diagnostics=diagnostics)
    for module in list(sys.modules):
        if module == name or module.startswith(name + "."):
            del sys.modules[module]


def config(model_type, persistent):
    return SimpleNamespace(
        model_type=model_type,
        anchor_chunk_size=512,
        anchor_token_num=1,
        anchor_position_ids_type="zeros",
        hsd_persistent_memory_enabled=persistent,
    )


@pytest.mark.parametrize("model_type", ["safin1_4b_text", "safin1_35b_a3b_text"])
@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("length", [511, 512, 513, 1025])
def test_anchor_layout_and_output_filtering(helpers, model_type, persistent, length):
    layout = helpers.layout._build_hsd_layout(
        config=config(model_type, persistent),
        positions=torch.arange(length),
        query_start_loc=torch.tensor([0, length]),
        num_computed_tokens=torch.tensor([0]),
        state_indices=torch.tensor([2]),
        history_state_indices=torch.tensor([9]),
    )
    expected_boundaries = ([0] if persistent else []) + list(
        range(512, length + 1, 512)
    )
    assert layout.anchor_boundaries.tolist() == expected_boundaries
    assert layout.anchor_positions.tolist() == ([0] if persistent else []) + [
        end + index + int(persistent)
        for index, end in enumerate(range(512, length + 1, 512))
    ]
    assert layout.text_to_expanded.tolist() == [
        t + t // 512 + int(persistent) for t in range(length)
    ]
    assert layout.persistent_memory_mask.sum().item() == int(persistent)
    assert layout.state_indices_list == [2] and layout.history_state_indices_list == [9]
    assert layout.num_text_tokens == length
    assert layout.num_expanded_tokens == length + len(expected_boundaries)
    hidden = torch.arange(layout.num_expanded_tokens * 2).view(-1, 2).float()
    text = layout.text_hidden(hidden)
    assert text.shape == (length, 2)
    restored = layout.scatter_text_hidden(hidden, text + 10)
    torch.testing.assert_close(
        restored[layout.anchor_mask], hidden[layout.anchor_mask], atol=0, rtol=0
    )
    torch.testing.assert_close(layout.text_hidden(restored), text + 10, atol=0, rtol=0)
    torch.testing.assert_close(
        layout.text_positions(), torch.arange(length), atol=0, rtol=0
    )


@pytest.mark.parametrize("persistent", [False, True])
def test_mixed_boundary_decode_keeps_slot_and_history_identity(helpers, persistent):
    layout = helpers.layout._build_hsd_layout(
        config=config("safin1_4b_text", persistent),
        positions=torch.tensor([511, 512]),
        query_start_loc=torch.tensor([0, 1, 2]),
        num_computed_tokens=torch.tensor([511, 512]),
        state_indices=torch.tensor([2, 0]),
        history_state_indices=torch.tensor([7, 8]),
        hsd_prefix_snapshot_requests=[[("key", 512, 1)], []],
        hsd_prefix_restored_flags=[False, True],
    )
    assert layout.anchor_boundaries.tolist() == [512]
    assert layout.req_anchor_lens == [1, 0]
    assert layout.decode_valid_flags == [True, False]
    assert layout.decode_history_valid_flags == [persistent, False]
    assert layout.history_state_indices_key == (7, 8)
    assert layout.hsd_prefix_restored_flags == [False, True]
    assert layout.hsd_prefix_snapshot_requests == [[("key", 512, 1)], []]
    assert layout.persistent_memory_mask.sum().item() == 0


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("past", [511, 512])
def test_static_decode_history_mask_and_fixed_input_buffers(helpers, persistent, past):
    valid = torch.tensor([True, False])
    slots, history = (
        torch.tensor([2, 0], dtype=torch.int32),
        torch.tensor([7, 8], dtype=torch.int32),
    )
    layout = helpers.layout._build_hsd_padded_static_decode_layout(
        config=config("safin1_35b_a3b_text", persistent),
        positions=torch.tensor([past, past]),
        hsd_num_computed_tokens=torch.tensor([past, past]),
        hsd_state_indices=slots,
        hsd_history_state_indices=history,
        hsd_decode_valid_mask=valid,
    )
    assert layout.padded_static_decode and not layout.has_anchor_tokens
    assert layout.state_indices.data_ptr() == slots.data_ptr()
    assert layout.history_state_indices.data_ptr() == history.data_ptr()
    assert layout.decode_valid_mask.data_ptr() == valid.data_ptr()
    assert layout.decode_history_valid_flags == [past >= 512 or persistent, False]
    assert layout.decode_history_candidate_indices.tolist() == (
        [0] if past >= 512 or persistent else []
    )


def test_bad_layout_metadata_is_rejected(helpers):
    args = dict(
        config=config("safin1_4b_text", False),
        positions=torch.tensor([0]),
        query_start_loc=torch.tensor([0, 1]),
        num_computed_tokens=torch.tensor([0]),
    )
    with pytest.raises(ValueError, match="state_indices"):
        helpers.layout._build_hsd_layout(
            **args, state_indices=torch.tensor([], dtype=torch.long)
        )
    with pytest.raises(ValueError, match="restore flags"):
        helpers.layout._build_hsd_layout(
            **args, hsd_prefix_restored_flags=[False, False]
        )


def test_collector_and_environment_share_one_trace_state(helpers, monkeypatch):
    diagnostic = helpers.diagnostics
    monkeypatch.delenv("VLLM_HSD_TRACE", raising=False)
    previous = diagnostic.set_vllm_hsd_trace_collector(None)
    try:
        assert not diagnostic._hsd_trace_requested()
        collector = {}
        assert diagnostic.set_vllm_hsd_trace_collector(collector) is None
        assert diagnostic._hsd_trace_requested()
        diagnostic.trace_vllm_hsd_internal(0, "fixture", torch.tensor([1.0, 2.0]))
        assert collector and any(
            torch.equal(value, torch.tensor([1.0, 2.0])) for value in collector.values()
        )
        assert diagnostic.set_vllm_hsd_trace_collector(None) is collector
        monkeypatch.setenv("VLLM_HSD_TRACE", "1")
        assert diagnostic._hsd_trace_requested()
    finally:
        diagnostic.set_vllm_hsd_trace_collector(previous)


@pytest.mark.parametrize("aggregate", [False, True])
def test_event_counter_flush_and_failure_telemetry(
    helpers, monkeypatch, tmp_path, aggregate
):
    diagnostic = helpers.diagnostics
    path = tmp_path / "events.jsonl"
    monkeypatch.setenv("VLLM_HSD_VARLEN_COUNTER_FILE", str(path))
    monkeypatch.setenv("VLLM_HSD_VARLEN_COUNTER_AGGREGATE", str(int(aggregate)))
    monkeypatch.setenv("VLLM_HSD_VARLEN_COUNTER_AGGREGATE_FLUSH_EVENTS", "0")
    layer = diagnostic.AttentionDiagnosticsMixin()
    layer.layer_idx = 2
    layer.hsd_prefill_varlen_counters, layer.hsd_prefill_varlen_fallback_counts = {}, {}
    layer._record_hsd_varlen_counter("calls", 2)
    layer._record_hsd_varlen_event("test", tokens=2, backend="fixture")
    layer._record_hsd_varlen_event("test", tokens=3, backend="fixture")
    layer._record_hsd_varlen_fallback("fixture-rejection")
    diagnostic._flush_hsd_varlen_event_aggregates()
    assert layer.hsd_prefill_varlen_counters == {"calls": 2}
    assert layer.hsd_prefill_varlen_fallback_counts == {"fixture-rejection": 1}
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    values = [row for row in rows if row["event"] == "test"]
    assert all(row["layer"] == 3 and row["backend"] == "fixture" for row in values)
    assert sum(row["tokens"] for row in values) == 5
    assert len(values) == (1 if aggregate else 2)
    assert any(
        row["event"] == "fallback" and row["reason"] == "fixture-rejection"
        for row in rows
    )


def test_graph_sync_is_opt_in_and_stage_scoped(helpers, monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("sync"))
    monkeypatch.delenv("VLLM_HSD_GRAPH_V3_DEBUG_SYNC", raising=False)
    helpers.diagnostics._hsd_graph_v3_debug_sync("model")
    monkeypatch.setenv("VLLM_HSD_GRAPH_V3_DEBUG_SYNC", "attention")
    helpers.diagnostics._hsd_graph_v3_debug_sync("model")
    assert not calls
    helpers.diagnostics._hsd_graph_v3_debug_sync("attention")
    assert calls == ["sync"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("past", [511, 512])
def test_cuda_layout_filter_and_scatter_graph_replay(helpers, persistent, past):
    slots = torch.tensor([2, 0], device="cuda", dtype=torch.int32)
    layout = helpers.layout._build_hsd_padded_static_decode_layout(
        config=config("safin1_4b_text", persistent),
        positions=torch.tensor([past, past], device="cuda"),
        hsd_num_computed_tokens=torch.tensor([past, past], device="cuda"),
        hsd_state_indices=slots,
        hsd_history_state_indices=slots.clone(),
        hsd_decode_valid_mask=torch.tensor([True, False], device="cuda"),
    )
    value = torch.randn(2, 8, device="cuda")
    result = torch.empty_like(value)

    def forward():
        result.copy_(layout.scatter_text_hidden(value, layout.text_hidden(value) + 1))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    for _ in range(3):
        value.add_(1)
        graph.replay()
        torch.testing.assert_close(result, value + 1, atol=0, rtol=0)
    assert layout.state_indices.data_ptr() == slots.data_ptr()
    graph.reset()

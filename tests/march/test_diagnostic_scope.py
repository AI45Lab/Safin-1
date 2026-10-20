"""Nested and concurrent diagnostics retain their own collectors and model metadata."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest
import torch

from march.safin1.march_gdn import hsd_trace_collector, set_hsd_trace_collector, trace_hsd_internal
from march.diagnostics.precision_trace import (
    active_precision_tracer, get_active_precision_tracer, set_active_precision_tracer,
    _collect_hsd_runtime_metadata, is_hsd_extra_param,
)


def test_nested_hf_collector_restores_after_exception():
    outer, inner = {}, {}
    previous = set_hsd_trace_collector(None)
    try:
        with hsd_trace_collector(outer):
            trace_hsd_internal(0, 'outer', torch.tensor([1.0]))
            with pytest.raises(RuntimeError, match='fixture'):
                with hsd_trace_collector(inner):
                    trace_hsd_internal(0, 'inner', torch.tensor([2.0]))
                    raise RuntimeError('fixture')
            trace_hsd_internal(0, 'restored', torch.tensor([3.0]))
        trace_hsd_internal(0, 'outside', torch.tensor([4.0]))
        assert set(outer) == {'layer_1/outer', 'layer_1/restored'}
        assert set(inner) == {'layer_1/inner'}
    finally:
        set_hsd_trace_collector(previous)


def test_concurrent_hf_and_training_diagnostics_are_isolated():
    barrier = Barrier(4)
    previous = set_hsd_trace_collector(None)
    previous_tracer = get_active_precision_tracer()
    set_active_precision_tracer(None)

    def record(index):
        collector, tracer = {}, SimpleNamespace(index=index)
        assert get_active_precision_tracer() is None
        with hsd_trace_collector(collector), active_precision_tracer(tracer):
            barrier.wait(timeout=10)
            trace_hsd_internal(0, 'value', torch.tensor([index]))
            assert get_active_precision_tracer() is tracer
            barrier.wait(timeout=10)
        assert get_active_precision_tracer() is None
        return collector['layer_1/value'].item()

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            assert list(pool.map(record, range(4))) == list(range(4))
        assert get_active_precision_tracer() is None
    finally:
        set_hsd_trace_collector(previous)
        set_active_precision_tracer(previous_tracer)


def test_nested_training_tracer_and_legacy_setter_restore():
    previous = get_active_precision_tracer()
    outer, inner = object(), object()
    try:
        set_active_precision_tracer(outer)
        with pytest.raises(ValueError, match='fixture'):
            with active_precision_tracer(inner):
                assert get_active_precision_tracer() is inner
                raise ValueError('fixture')
        assert get_active_precision_tracer() is outer
    finally:
        set_active_precision_tracer(previous)


def test_metadata_covers_persistent_state_without_retired_flags():
    args = SimpleNamespace(hsd_runtime_config={
        'hsd_persistent_memory_enabled': True, 'hsd_persistent_memory_route_key_enabled': True,
        'hsd_route_head_dim': 128, 'hsd_history_read_q_lora_rank': 32, 'hsd_anchor_mlp': False,
        'hsd_history_out_grad_scale': 0.5,
    })
    metadata = _collect_hsd_runtime_metadata(args)
    assert metadata['hsd_persistent_memory_enabled'] is True
    assert metadata['hsd_persistent_memory_route_key_enabled'] is True
    assert metadata['hsd_route_head_dim'] == 128
    assert metadata['hsd_history_out_grad_scale'] == 0.5
    assert 'hsd_history_read_q_lora_rank' not in metadata
    assert 'hsd_anchor_mlp' not in metadata
    assert is_hsd_extra_param('decoder.layers.0.persistent_memory_state')
    assert is_hsd_extra_param('decoder.layers.0.persistent_memory_route_key')

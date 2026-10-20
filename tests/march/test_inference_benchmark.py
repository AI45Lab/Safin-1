"""The batch acceptance must detect state collisions and invalid measurements."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/march'))
    spec = importlib.util.spec_from_file_location('march_inference_benchmark', ROOT/'scripts/march/benchmark_inference.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_run_models_capacity_separately_from_concurrent_load(tmp_path):
    model = tmp_path/'model'
    model.mkdir()
    (model/'config.json').write_text(json.dumps({'model_type': 'safin1_35b_a3b'}))
    output = tmp_path/'result.json'
    result = subprocess.run([sys.executable, str(ROOT/'scripts/march/benchmark_inference.py'),
                             '--model', str(model), '--output', str(output), '--dry-run'],
                            check=True, text=True, capture_output=True)
    plan = json.loads(result.stdout)
    assert plan['capacity'] == 32 and plan['batches'] == [1, 4, 8, 32]
    assert plan['prefix_cache'] is False and plan['tp'] == plan['dp'] == 1
    assert not output.exists()
    assert not output.with_suffix('.events.jsonl').exists()


def allocation(ids, slots):
    return {'event': 'hsd_compact_state_slot_allocate', 'req_ids': ids, 'slot_indices': slots}


def test_slot_reuse_requires_release_and_preserves_live_owners(benchmark):
    events = [allocation(['A', 'B'], [0, 1]), allocation(['B'], [1]),
              {'event': 'hsd_compact_state_slot_release', 'req_ids': ['A'], 'released': 1},
              allocation(['C', 'B'], [0, 1])]
    result = benchmark.slot_evidence(events)
    assert result['reused_slots'] == 1 and result['allocations'] == 3
    assert result['max_live_slots'] == 2 and result['remaining_deferred_slots'] == 2
    # Reassigning a finished slot is valid; reassigning a live slot is not.
    with pytest.raises(ValueError, match='share'):
        benchmark.slot_evidence(events[:2] + [allocation(['C'], [0])])
    with pytest.raises(ValueError, match='changed'):
        benchmark.slot_evidence(events[:1] + [allocation(['A'], [2])])
    with pytest.raises(ValueError, match='requires ordered'):
        benchmark.slot_evidence([{'event': 'hsd_compact_state_slot_allocate', 'allocated': 2}])


def output_fixture():
    metrics = SimpleNamespace(arrival_time=1800000000, first_token_latency=0.2,
                              first_token_ts=10., last_token_ts=11., is_corrupted=False)
    value = SimpleNamespace(token_ids=[5, 6], logprobs=[{5: SimpleNamespace(logprob=-0.1)},
                                                       {6: SimpleNamespace(logprob=-0.2)}])
    return SimpleNamespace(prompt_token_ids=[1, 2, 3], finished=True, request_id='A',
                           outputs=[value], metrics=metrics)


def test_request_metrics_do_not_mix_monotonic_and_wall_clock_domains(benchmark):
    result = benchmark.summarize_outputs([output_fixture()], [{'key': 0, 'prompt': [1, 2, 3], 'max_tokens': 2}])[0]
    assert result['ttft_seconds'] == 0.2
    assert result['decode_seconds'] == result['decode_tokens_per_second'] == 1.


@pytest.mark.parametrize('corruption', ['prompt', 'ids', 'logprobs', 'nan', 'metrics', 'corrupted'])
def test_incomplete_or_invalid_generation_cannot_pass(benchmark, corruption):
    output = output_fixture()
    if corruption == 'prompt': output.prompt_token_ids = [9]
    elif corruption == 'ids': output.outputs[0].token_ids = [5]
    elif corruption == 'logprobs': output.outputs[0].logprobs = []
    elif corruption == 'nan': output.outputs[0].logprobs[0][5].logprob = float('nan')
    elif corruption == 'metrics': output.metrics = None
    else: output.metrics.is_corrupted = True
    with pytest.raises(ValueError):
        benchmark.summarize_outputs([output], [{'key': 0, 'prompt': [1, 2, 3], 'max_tokens': 2}])


def test_numeric_drift_is_recorded_without_erasing_changed_ids(benchmark):
    reference = [{'key': 0, 'ids': [1, 2], 'logprobs': [-0.1, -0.2]}]
    changed = [{'key': 0, 'ids': [1, 2], 'logprobs': [-0.1, -0.205]}]
    result = benchmark.compare_runs(reference, changed)
    assert result['ids_exact'] and not result['selected_logprob_passed']
    changed[0]['ids'] = [1, 3]
    result = benchmark.compare_runs(reference, changed)
    assert not result['ids_exact'] and result['requests'][0]['max_abs_error'] is None
    with pytest.raises(ValueError, match='identity'):
        benchmark.compare_runs(reference, [])


def test_persistent_parameter_checks_include_route_keys_but_exclude_counters(benchmark):
    a = [{'mode': 'learned', 'tensor_count': 1, 'finite': True, 'local_checksum': 'state',
          'route_key': {'tensor_count': 1, 'finite': True, 'local_checksum': 'key'}, 'metrics': {'reads': 1}}]
    b = json.loads(json.dumps(a))
    b[0]['metrics']['reads'] = 10
    assert benchmark.persistent_identity(a) == benchmark.persistent_identity(b)
    b[0]['route_key']['local_checksum'] = 'mutated'
    assert benchmark.persistent_identity(a) != benchmark.persistent_identity(b)


def test_missing_checkpoint_shards_fail_before_gpu_setup(benchmark, tmp_path):
    (tmp_path/'config.json').write_text('{}')
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'weight': 'part.safetensors'}}))
    with pytest.raises(ValueError, match='shard'):
        benchmark.inspect_model(tmp_path)
    (tmp_path/'part.safetensors').write_bytes(b'0'*32)
    before = benchmark.inspect_model(tmp_path)
    (tmp_path/'part.safetensors').write_bytes(b'1'*64)
    assert benchmark.inspect_model(tmp_path) != before


def test_nonfinite_persistent_parameters_cannot_pass_audit(benchmark):
    with pytest.raises(ValueError, match='Non-finite'):
        benchmark.persistent_identity([{'tensor_count': 1, 'finite': False}])

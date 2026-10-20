#!/usr/bin/env python3
"""Offline TP1/DP1 batch, cache-reuse and latency probe for Safin-1.

Run each profile in a fresh process. Existing native extensions and telemetry
costs must be reported separately from a clean build or peak-throughput claim.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import sys
import time
from types import SimpleNamespace

from serve_vllm import build_launch, read_config

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LENGTHS = (511, 512, 513, 1025, 2048, 2049, 4096, 8000)


def positive_csv(value):
    values = tuple(int(v) for v in value.split(','))
    if not values or any(v < 1 for v in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError('provide distinct positive integers')
    return values


def requests(batch, lengths, max_tokens, poison=False):
    rows = []
    for index in range(batch):
        length = lengths[index % len(lengths)]
        # Every request has distinct synthetic content; B uses a separate range.
        start = 100 + 17 * index + (701 if poison else 0)
        rows.append({'key': index, 'prompt': [start + i % 13 for i in range(length)],
                     'max_tokens': max(2, max_tokens // (1, 2, 4, 8)[index % 4])})
    return rows


def summarize_outputs(outputs, specs):
    if len(outputs) != len(specs):
        raise ValueError('Incomplete request batch')
    rows = []
    for output, spec in zip(outputs, specs):
        if list(output.prompt_token_ids) != spec['prompt'] or not output.finished:
            raise ValueError('Output request order/prompt or completion status changed')
        if len(output.outputs) != 1:
            raise ValueError('Expected exactly one greedy completion')
        value = output.outputs[0]
        ids = list(value.token_ids)
        if len(ids) != spec['max_tokens'] or any(type(v) is not int or v < 0 for v in ids):
            raise ValueError('Incomplete generated token IDs')
        if value.logprobs is None or len(value.logprobs) != len(ids):
            raise ValueError('Incomplete selected-token logprobs')
        logprobs = [float(row[token].logprob) for token, row in zip(ids, value.logprobs)]
        if not all(math.isfinite(v) for v in logprobs):
            raise ValueError('Non-finite generated logprobs')
        metrics = output.metrics
        if metrics is None:
            raise ValueError('Request timing requires disable_log_stats=False')
        ttft = metrics.first_token_latency
        decode = metrics.last_token_ts - metrics.first_token_ts
        if not all(math.isfinite(v) and v >= 0 for v in (ttft, decode)) or ttft == 0:
            raise ValueError('Invalid TTFT or monotonic decode interval')
        if getattr(metrics, 'is_corrupted', False):
            raise ValueError('Engine detected corrupted logits')
        rows.append({'key': spec['key'], 'request_id': output.request_id,
                     'input_tokens': len(spec['prompt']), 'ids': ids, 'logprobs': logprobs,
                     'ttft_seconds': ttft, 'decode_seconds': decode,
                     'decode_tokens_per_second': (len(ids) - 1) / decode if decode > 0 else None})
    return rows


def compare_runs(reference, candidate, atol=1e-3, rtol=1e-3):
    if [r['key'] for r in reference] != [r['key'] for r in candidate]:
        raise ValueError('Request identity/order changed between waves')
    rows = []
    for before, after in zip(reference, candidate):
        ids_equal = before['ids'] == after['ids']
        a, b = before['logprobs'], after['logprobs']
        if len(a) != len(before['ids']) or len(b) != len(after['ids']):
            raise ValueError('Missing logprobs in comparison')
        valid = ids_equal and all(math.isfinite(x) and math.isfinite(y)
                                 and abs(x - y) <= atol + rtol * abs(x) for x, y in zip(a, b))
        rows.append({'key': before['key'], 'ids_exact': ids_equal,
                     'selected_logprob_passed': valid,
                     'max_abs_error': max((abs(x - y) for x, y in zip(a, b)), default=0) if ids_equal else None})
    return {'ids_exact': all(r['ids_exact'] for r in rows),
            'selected_logprob_passed': all(r['selected_logprob_passed'] for r in rows), 'requests': rows}


def slot_evidence(events):
    """Check allocation/release order; a final deferred release is allowed."""
    live, owners, used = {}, {}, set()
    reuse, peak, allocations, releases = 0, 0, 0, 0
    for row in events:
        if row.get('event') == 'hsd_compact_state_slot_release':
            releases += int(row.get('released', 0))
            for request in row.get('req_ids', []):
                slot = live.pop(request, None)
                if slot is not None:
                    owners.pop(slot, None)
        elif row.get('event') == 'hsd_compact_state_slot_allocate':
            req_ids, slots = row.get('req_ids'), row.get('slot_indices')
            if req_ids is None or slots is None or len(req_ids) != len(slots):
                raise ValueError('Slot audit requires ordered request IDs and slot indices')
            for request, slot in zip(req_ids, slots):
                if request is None or (request == 'None' and isinstance(slot, int) and slot < 0):
                    continue
                if type(slot) is not int or slot < 0:
                    raise ValueError('Invalid live compact slot')
                if slot in owners and owners[slot] != request:
                    raise ValueError('Two live requests share a compact state slot')
                if request in live and live[request] != slot:
                    raise ValueError('Live request changed compact slot')
                if request not in live:
                    allocations += 1
                    reuse += int(slot in used)
                live[request], owners[slot] = slot, request
                used.add(slot)
            peak = max(peak, len(live))
    return {'allocations': allocations, 'released': releases, 'reused_slots': reuse,
            'max_live_slots': peak, 'remaining_deferred_slots': len(live), 'unique_slots': len(used)}


def persistent_identity(audits):
    for row in audits:
        for owner in (row, row.get('route_key', {})):
            if owner.get('tensor_count', 0) and not owner.get('finite', False):
                raise ValueError('Non-finite persistent model parameter')
    return [{key: row.get(key) for key in ('mode', 'tensor_count', 'finite', 'local_checksum')}
            | {'route_key': {key: row.get('route_key', {}).get(key)
                             for key in ('tensor_count', 'finite', 'local_checksum')}} for row in audits]


def inspect_model(model):
    model = Path(model).resolve()
    index = model/'model.safetensors.index.json'
    shards = set(json.loads(index.read_text())['weight_map'].values()) if index.is_file() else {'model.safetensors'}
    files = {}
    for name in sorted(shards):
        path = (model/name).resolve()
        if not path.is_relative_to(model) or not path.is_file() or path.stat().st_size < 16:
            raise ValueError('Missing/invalid local weight shard: '+name)
        files[name] = {'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
    return {'config_sha256': hashlib.sha256((model/'config.json').read_bytes()).hexdigest(),
            'index_sha256': hashlib.sha256(index.read_bytes()).hexdigest() if index.is_file() else None,
            'shards': files, 'weight_check': 'local existence/size; full weight hashes are not recomputed'}


def fingerprints():
    paths = [*ROOT.joinpath('march').rglob('*.py'),
             *ROOT.joinpath('third_party/vllm/vllm/model_executor/models/march').glob('*.py'),
             ROOT/'third_party/vllm/vllm/model_executor/models/safin1.py',
             ROOT/'third_party/vllm/vllm/v1/worker/gpu_model_runner.py',
             ROOT/'scripts/march/serve_vllm.py', Path(__file__)]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--model', required=True)
    parser.add_argument('--profile', choices=('graph', 'eager'), default='graph')
    parser.add_argument('--batches', type=positive_csv, default=(1, 4, 8, 32))
    parser.add_argument('--lengths', type=positive_csv, default=DEFAULT_LENGTHS)
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--max-num-seqs', type=int, default=32)
    parser.add_argument('--max-model-len', type=int, default=8192)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    parser.add_argument('--native-build', choices=('unknown', 'reused', 'source'), default='unknown')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if (args.max_tokens < 8 or not 1 <= args.max_num_seqs <= 32
            or max(args.batches) > args.max_num_seqs
            or max(args.lengths) + args.max_tokens + 32 > args.max_model_len):
        parser.error('require tokens >=8, batch <= capacity <=32, and context headroom for anchors')
    if not Path(args.model).is_dir():
        parser.error('provide a local checkpoint; this tool never downloads models')
    model_type, config = read_config(args.model, with_config=True)
    events_path = args.output.with_suffix('.events.jsonl')
    launch = SimpleNamespace(**vars(args), tensor_parallel_size=1, data_parallel_size=1,
                             served_model_name='Safin-1', revision=None,
                             experimental_prefix_cache=False, prefix_cache_max_bytes=0, backend_events=False)
    command, env, controls = build_launch(launch, [], model_type, dict(os.environ, MARCH_EVENT_LOG=str(events_path)), config)
    plan = {'model_type': model_type, 'profile': args.profile, 'batches': args.batches,
            'capacity': args.max_num_seqs, 'lengths': args.lengths, 'max_tokens': args.max_tokens,
            'tp': 1, 'dp': 1, 'prefix_cache': False, 'native_build': args.native_build}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if args.output.exists() or events_path.exists():
        parser.error('choose new output and event paths for each attempt')
    for name in ('OFFLINE', 'HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_EVALUATE_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        env[name] = '1'
    env['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    env['VLLM_HSD_COMPACT_SLOT_DEBUG'] = '1'
    model_inventory = inspect_model(args.model)
    for name in list(os.environ):
        if name.startswith(('HSD_', 'VLLM_HSD_')):
            del os.environ[name]
    os.environ.update(env)
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models import safin1
    if not Path(vllm.__file__).resolve().is_relative_to(ROOT/'third_party/vllm'):
        raise RuntimeError('Use the vLLM Python source from this fork')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('This TP1/DP1 in-process benchmark requires one leased GPU')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    original_sources = fingerprints()
    report = dict(plan, status='running', source_sha256=original_sources, cases=[],
                  gpu=torch.cuda.get_device_name(0), torch=torch.__version__, controls=controls,
                  atol=1e-3, rtol=1e-3, seed=123, model_inventory=model_inventory,
                  memory_scope='same-process torch allocator including resident model/cache; incremental peaks also recorded',
                  timing_scope='engine request metrics and synchronized batch wall time; diagnostics enabled')
    report['controls'] = dict(controls, VLLM_HSD_COMPACT_SLOT_DEBUG='1',
                              VLLM_HSD_VARLEN_COUNTER_FILE=events_path.name)
    def save():
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        extra = {}
        if args.profile == 'graph':
            extra['compilation_config'] = json.loads(command[command.index('--compilation-config')+1])
        if '35b' in model_type:
            extra['moe_backend'] = 'triton'
        engine = LLM(model=args.model, trust_remote_code=True, dtype='bfloat16',
                     enforce_eager=args.profile == 'eager', tensor_parallel_size=1,
                     max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len,
                     gpu_memory_utilization=args.gpu_memory_utilization, enable_prefix_caching=False,
                     async_scheduling=False, mamba_cache_mode='none', language_model_only=True,
                     disable_log_stats=False,
                     additional_config=json.loads(command[command.index('--additional-config')+1]), **extra)
        persistent_before = persistent_identity(engine.get_persistent_memory_audit(reset_metrics=False))
        report['persistent_state'] = {'enabled': any(r['tensor_count'] for r in persistent_before),
                                      'before': persistent_before}
        replay_count = 0
        original_replay = torch.cuda.CUDAGraph.replay
        def replay(graph):
            nonlocal replay_count
            value = original_replay(graph)
            replay_count += 1
            return value
        torch.cuda.CUDAGraph.replay = replay
        try:
            def generate(specs):
                samples = [SamplingParams(temperature=0, seed=123, max_tokens=s['max_tokens'],
                                          ignore_eos=True, logprobs=1) for s in specs]
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                before = replay_count
                started = time.perf_counter()
                outputs = engine.generate([{'prompt_token_ids': s['prompt']} for s in specs], samples, use_tqdm=False)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                peak, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
                rows = summarize_outputs(outputs, specs)
                return {'requests': rows, 'seconds': seconds,
                        'output_tokens_per_second': sum(len(r['ids']) for r in rows)/seconds,
                        'ttft_median_seconds': median(r['ttft_seconds'] for r in rows),
                        'peak_allocated_bytes': peak, 'peak_reserved_bytes': reserved,
                        'incremental_peak_allocated_bytes': peak-baseline,
                        'completed_graph_replays': replay_count-before}
            for batch in args.batches:
                # Uniform warmup gives the scheduler an opportunity to reach the requested batch.
                generate(requests(batch, (513,), args.max_tokens))
                specs = requests(batch, args.lengths, args.max_tokens)
                first = generate(specs)
                poison = generate(requests(batch, tuple(reversed(args.lengths)), args.max_tokens, poison=True))
                repeated = generate(specs)
                comparison = compare_runs(first['requests'], repeated['requests'])
                report['cases'].append({'batch': batch, 'first': first, 'poison': poison,
                                        'after_other_requests': repeated, 'repeat_comparison': comparison})
                save()
                print(json.dumps({'batch': batch, 'profile': args.profile,
                                  'output_tokens_per_second': repeated['output_tokens_per_second'],
                                  'ids_exact': comparison['ids_exact'],
                                  'selected_logprob_passed': comparison['selected_logprob_passed']}), flush=True)
        finally:
            torch.cuda.CUDAGraph.replay = original_replay
        safin1._flush_hsd_varlen_event_aggregates()
        events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
        report['event_counts'] = dict(Counter(r.get('event') for r in events))
        report['slots'] = slot_evidence(events)
        report['max_graph_requests'] = max((int(r.get('requests', 0)) for r in events
                                            if r.get('event') == 'hsd_graphable_decode_hit'), default=0)
        report['requested_graph_batch_observed'] = args.profile != 'graph' or report['max_graph_requests'] >= max(args.batches)
        if args.profile == 'graph':
            if not report['slots']['reused_slots'] or not any(c['after_other_requests']['completed_graph_replays'] for c in report['cases']):
                raise RuntimeError('No actual compact-slot reuse or completed CUDA Graph replay')
        persistent_after = persistent_identity(engine.get_persistent_memory_audit(reset_metrics=False))
        if persistent_before != persistent_after:
            raise RuntimeError('Persistent model parameters changed during inference')
        report['persistent_state'].update(after=persistent_after, unchanged=True)
        if model_inventory != inspect_model(args.model):
            raise RuntimeError('Checkpoint metadata/size/mtime changed during the benchmark')
        if original_sources != fingerprints():
            raise RuntimeError('Source changed during the benchmark')
        report['status'] = 'passed' if report['requested_graph_batch_observed'] and all(c['repeat_comparison']['selected_logprob_passed'] for c in report['cases']) else 'conditional'
        save()
        return 0
    except Exception as error:
        report.update(status='failed', error=type(error).__name__+': '+str(error))
        save()
        raise


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""Evaluate local Safin-1 checkpoints with HF or the shared vLLM profile."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from march.inference.vllm_profile import (  # noqa: E402
    build_engine_arguments, clean_environment, positive, read_config,
)

OFFLINE_FLAGS = (
    'OFFLINE', 'HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE',
    'HF_EVALUATE_OFFLINE', 'TRANSFORMERS_OFFLINE',
)


def boolean(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value not in ('0', '1', 'false', 'true'):
        raise argparse.ArgumentTypeError('use 0/1 or false/true')
    return value in ('1', 'true')


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--backend', choices=('hf', 'vllm'), required=True)
    p.add_argument('--model', default=os.environ.get('MODEL'))
    p.add_argument('--revision', default=os.environ.get('MODEL_REVISION'))
    p.add_argument('--tasks', default=os.environ.get('TASKS'))
    p.add_argument('--output-dir', default=os.environ.get('OUTPUT_DIR'))
    p.add_argument('--profile', choices=('graph', 'eager'),
                   default=os.environ.get('MARCH_INFERENCE_PROFILE', 'graph'))
    p.add_argument('--batch-size', default=os.environ.get('BATCH_SIZE', 'auto'))
    p.add_argument('--dtype', default=os.environ.get('DTYPE', 'bfloat16'))
    p.add_argument('--seed', type=int, default=os.environ.get('SEED', '123'))
    p.add_argument('--limit', type=float, default=os.environ.get('LIMIT') or None)
    p.add_argument('--apply-chat-template', type=boolean,
                   default=os.environ.get('APPLY_CHAT_TEMPLATE', '1'))
    p.add_argument('--enable-thinking', type=boolean,
                   default=os.environ.get('ENABLE_THINKING', 'false'))
    p.add_argument('--think-end-token', default=os.environ.get('THINK_END_TOKEN'))
    p.add_argument('--offline', type=boolean, default=os.environ.get('OFFLINE', '1'))
    p.add_argument('--dry-run', action='store_true', default=os.environ.get('DRY_RUN') == '1')
    for flag, variables, default in (
        ('tensor-parallel-size', ('TENSOR_PARALLEL_SIZE', 'TP_SIZE'), '1'),
        ('data-parallel-size', ('DATA_PARALLEL_SIZE', 'DP_SIZE'), '1'),
        ('max-num-seqs', ('MAX_NUM_SEQS',), '32'),
        ('max-model-len', ('MAX_MODEL_LEN',), '8192'),
    ):
        p.add_argument('--' + flag, type=positive,
                       default=next((os.environ[k] for k in variables if k in os.environ), default))
    p.add_argument('--gpu-memory-utilization', type=float,
                   default=os.environ.get('GPU_MEMORY_UTILIZATION', '0.85'))
    return p


def build_plan(args, extra, parent):
    if not args.model or not args.tasks or not args.output_dir:
        raise ValueError('Set MODEL, TASKS and OUTPUT_DIR (or their CLI options)')
    tasks = args.tasks.split(',')
    if any(not t.strip() for t in tasks):
        raise ValueError('TASKS must contain comma-separated task names')
    tasks = [t.strip() for t in tasks]
    if args.limit is not None and args.limit <= 0:
        raise ValueError('LIMIT must be positive')
    if args.enable_thinking and not args.think_end_token:
        raise ValueError('ENABLE_THINKING=true requires THINK_END_TOKEN')
    managed = {'--model', '-M', '--model_args', '-a', '--config', '-C', '--config_file',
               '--tasks', '-t', '--output_path', '-o', '--seed', '--apply_chat_template',
               '--batch_size', '-b', '--limit', '-L', '--hf_overrides'}
    managed_names = {key.replace('-', '_').lstrip('_') for key in managed}
    short_options = {key for key in managed if key.startswith('-') and not key.startswith('--')}
    for value in extra:
        option = value.split('=', 1)[0].replace('-', '_').lstrip('_')
        # The harness accepts long-option abbreviations and attached short values.
        # They must obey the same profile ownership as the full option spellings.
        abbreviation = value.startswith('--') and any(key.startswith(option) for key in managed_names)
        attached_short = value.startswith('-') and not value.startswith('--') and value[:2] in short_options
        if option in managed_names or abbreviation or attached_short:
            raise ValueError(f'{value} is controlled by the MARCH evaluation entrypoint')
    model_type, config = read_config(args.model, args.revision, with_config=True)
    if args.backend == 'vllm':
        if args.dtype != 'bfloat16':
            raise ValueError('The validated vLLM profiles require DTYPE=bfloat16')
        # Evaluation intentionally uses the default prefix-disabled contract.
        args.experimental_prefix_cache = False
        args.prefix_cache_max_bytes = 0
        args.served_model_name = 'Safin-1'
        args.backend_events = False
        model_args, env, controls = build_engine_arguments(args, model_type, parent, config)
    else:
        model_args = dict(pretrained=args.model, dtype=args.dtype, max_length=args.max_model_len)
        if args.revision:
            model_args['revision'] = args.revision
        env, controls = clean_environment(parent), {}
    model_args.update(trust_remote_code=True, enable_thinking=args.enable_thinking)
    if args.backend == 'vllm':
        model_args['seed'] = args.seed
    if args.think_end_token:
        model_args['think_end_token'] = args.think_end_token
    offline = {name: str(int(args.offline)) for name in OFFLINE_FLAGS}
    env.update(offline)
    env['PYTHONPATH'] = os.pathsep.join([
        str(ROOT / 'third_party/lm-evaluation-harness'),
        str(ROOT / 'third_party/vllm'), str(ROOT),
        *([parent['PYTHONPATH']] if parent.get('PYTHONPATH') else []),
    ])
    command = [sys.executable, '-m', 'lm_eval', 'run', '--model', args.backend,
               '--model_args', json.dumps(model_args), '--tasks', *tasks,
               '--batch_size', args.batch_size, '--output_path', args.output_dir,
               '--seed', str(args.seed)]
    if args.apply_chat_template:
        command += ['--apply_chat_template']
    if args.limit is not None:
        command += ['--limit', str(args.limit)]
    return dict(backend=args.backend, model_type=model_type,
                profile=args.profile if args.backend == 'vllm' else None,
                model_args=model_args, tasks=tasks, environment=controls | offline,
                command=command + extra), env


def check_resources(tasks):
    required = ['lm_eval']
    if any(task.startswith('ifeval') for task in tasks):
        required += ['langdetect', 'immutabledict', 'nltk']
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise ValueError('Missing evaluation dependencies: ' + ', '.join(missing)
                         + '; run bash scripts/march/install.sh eval')
    if 'nltk' in required:
        from lm_eval.tasks._nltk_resources import require_punkt_tab
        require_punkt_tab()


def main():
    p = parser()
    args, extra = p.parse_known_args()
    # Hub configuration lookup must honor offline mode too, before building a plan.
    os.environ.update({name: str(int(args.offline)) for name in OFFLINE_FLAGS})
    try:
        plan, env = build_plan(args, extra, os.environ)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return
        os.environ.clear()
        os.environ.update(env)
        sys.path[:0] = env['PYTHONPATH'].split(os.pathsep)
        check_resources(plan['tasks'])
    except (OSError, ValueError, LookupError) as error:
        p.error(str(error))
    os.execvpe(plan['command'][0], plan['command'], env)


if __name__ == '__main__':
    main()

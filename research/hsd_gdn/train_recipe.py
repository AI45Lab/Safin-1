#!/usr/bin/env python3
"""Launch a local research training recipe once per node, or print its plan."""
import argparse
import glob
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_RECIPE = HERE / 'configs/hsd_gdn_0p8b_readq32_full32.json'
RESERVED_OPTIONS = {'model', 'dataset', 'cached_dataset', 'output_dir', 'external_plugins',
                    'resume_from_checkpoint', 'ignore_data_skip'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive_int(value, label):
    require(type(value) is int and value > 0, f'{label} must be a positive integer')
    return value


def load_recipe(path):
    path = path.resolve()
    recipe = json.loads(path.read_text())
    require(isinstance(recipe, dict), 'Recipe must be a JSON object')
    for key in ('distributed', 'initialization', 'training', 'environment'):
        require(isinstance(recipe.get(key), dict), f'Recipe requires an object: {key}')
    dist, train = recipe['distributed'], recipe['training']
    require(dist.get('launcher') == 'torchrun', 'Only the torchrun launcher is supported')
    for key in ('nodes', 'gpus_per_node', 'world_size', 'global_batch_size'):
        positive_int(dist.get(key), f'distributed.{key}')
    require(dist['nodes'] * dist['gpus_per_node'] == dist['world_size'],
            'world_size must equal nodes * gpus_per_node')
    for key in ('per_device_train_batch_size', 'gradient_accumulation_steps', 'max_length', 'max_steps'):
        positive_int(train.get(key), f'training.{key}')
    batch = dist['world_size'] * train['per_device_train_batch_size'] * train['gradient_accumulation_steps']
    require(batch == dist['global_batch_size'], 'Training batch layout does not match global_batch_size')
    warmup = train.get('warmup_steps', 0)
    require(type(warmup) is int and 0 <= warmup < train['max_steps'], 'Invalid warmup_steps')
    require(train.get('add_version') is False, 'Recipe must set add_version=false for shared output paths')
    seed = recipe['initialization'].get('seed')
    require(type(seed) is int and 0 <= seed < 2**32 and train.get('seed') == seed,
            'Initialization and training seeds must match and be in [0, 2**32)')
    require(not RESERVED_OPTIONS.intersection(train), 'Model/data/output/resume options belong to the launcher')
    config_name = recipe.get('model_config')
    require(isinstance(config_name, str) and config_name, 'Recipe requires model_config')
    config = json.loads((path.parent / config_name).read_text())
    if config.get('model_type') == 'hsd_gated_deltanet':
        sys.path.insert(0, str(HERE / 'src'))
        from march_research.runtime_policy import sanitize_research_config
        config = sanitize_research_config(config)
    require(config.get('model_type') == train.get('model_type'), 'Recipe/model_config model_type mismatch')
    require(config.get('vocab_size') == recipe['initialization'].get('tokenizer_vocab_size'),
            'Initialization vocabulary does not match model_config')
    budget = recipe.get('budget', {}).get('nominal_packed_token_slots')
    require(budget is None or budget == batch * train['max_length'] * train['max_steps'],
            'Nominal token budget does not match the training recipe')
    for key, value in recipe['environment'].items():
        require(isinstance(key, str) and re.fullmatch(r'[A-Z][A-Z0-9_]*', key)
                and isinstance(value, str), 'Environment entries must be string variables')
    return recipe, config


def option_values(value):
    if isinstance(value, bool):
        return [str(value).lower()]
    if isinstance(value, dict):
        return [json.dumps(value, separators=(',', ':'), allow_nan=False)]
    if isinstance(value, list):
        require(value and all(isinstance(v, str) for v in value), 'List options must contain strings')
        return value
    require(isinstance(value, (str, int, float)) and not (isinstance(value, float) and not math.isfinite(value)),
            'Training options must be finite scalar values, objects or string lists')
    return [str(value)]


def build_plan(args):
    recipe, config = load_recipe(args.recipe)
    dist = recipe['distributed']
    rank = args.node_rank if args.node_rank is not None else (0 if dist['nodes'] == 1 else None)
    require(rank is not None and 0 <= rank < dist['nodes'], 'Set NODE_RANK / --node-rank within the recipe node count')
    address = args.master_addr or ('127.0.0.1' if dist['nodes'] == 1 else None)
    require(bool(address), 'Set MASTER_ADDR / --master-addr to the rank-0 node address')
    require(1 <= args.master_port <= 65535, 'MASTER_PORT must be in [1, 65535]')
    require(args.model is not None and args.output is not None, 'Set MODEL and OUTPUT_DIR, or pass --model and --output')
    require(bool(args.dataset) != bool(args.cached_dataset), 'Supply exactly one of DATASET or CACHED_DATASET')
    model, output = args.model.resolve(), args.output.resolve()
    command = [sys.executable, '-m', 'torch.distributed.run',
               '--nnodes', str(dist['nodes']), '--nproc-per-node', str(dist['gpus_per_node']),
               '--node-rank', str(rank), '--master-addr', address, '--master-port', str(args.master_port),
               str(ROOT / 'swift/cli/pt.py'),
               '--external_plugins', str(HERE / 'src/march_research/swift_plugin.py'),
               '--model', str(model), '--output_dir', str(output / 'training')]
    dataset = args.dataset
    arrow = dataset and (glob.has_magic(dataset) or Path(dataset).suffix == '.arrow' or Path(dataset).is_dir())
    if arrow:
        dataset = 'march_research_pretrain_arrow'
    command += ['--dataset', dataset] if dataset else ['--cached_dataset', args.cached_dataset]
    for key, value in recipe['training'].items():
        require(re.fullmatch(r'[a-z][a-z0-9_]*', key) is not None, 'Invalid training option name')
        command += ['--' + key, *option_values(value)]
    if args.resume:
        command += ['--resume_from_checkpoint', str(args.resume.resolve()), '--ignore_data_skip', 'false']
    env = dict(os.environ)
    env.update(recipe['environment'])
    env.pop('MARCH_RESEARCH_ARROW_DATASET', None)
    if arrow:
        env['MARCH_RESEARCH_ARROW_DATASET'] = args.dataset
    env.update(PYTHONPATH=os.pathsep.join([str(ROOT / 'third_party/flash-linear-attention'),
                                         str(HERE / 'src'), str(ROOT)]),
               NPROC_PER_NODE=str(dist['gpus_per_node']), NODE_RANK=str(rank),
               PYTHONDONTWRITEBYTECODE='1',
               TOKENIZERS_PARALLELISM='false')
    env.setdefault('MARCH_TRITON_CACHE_BASE', str(output / 'kernel-cache'))
    recorded_keys = [*recipe['environment'], 'PYTHONPATH', 'NPROC_PER_NODE', 'NODE_RANK', 'MARCH_TRITON_CACHE_BASE']
    if arrow:
        recorded_keys.append('MARCH_RESEARCH_ARROW_DATASET')
    plan = {'recipe': str(args.recipe.resolve()), 'node_rank': rank, 'command': command,
            'environment': {key: env[key] for key in recorded_keys},
            'global_batch_size': dist['global_batch_size'],
            'nominal_packed_token_slots': dist['global_batch_size'] * recipe['training']['max_length']
                                          * recipe['training']['max_steps']}
    return plan, env, config


def preflight(args, plan, config):
    saved = json.loads((args.model / 'config.json').read_text())
    if saved.get('model_type') == 'hsd_gated_deltanet':
        from march_research.runtime_policy import sanitize_research_config
        saved = sanitize_research_config(saved)
    differences = [key for key, value in config.items() if saved.get(key) != value]
    require(not differences, 'MODEL configuration differs from recipe model_config: ' + ', '.join(differences))
    require(any(args.model.glob('*.safetensors')) or any(args.model.glob('pytorch_model*.bin')),
            'MODEL must contain local model weights; initialize it first with run.py init')
    source = args.dataset or args.cached_dataset
    require(bool(glob.glob(source)), 'Dataset input must exist locally')
    if args.resume:
        require(args.resume.is_dir() and (args.resume / 'trainer_state.json').is_file(),
                'Resume requires a complete local training checkpoint with trainer_state.json')
    if plan['node_rank'] == 0 and not args.resume:
        require(not (args.output / 'training').exists(), 'Choose a new OUTPUT_DIR or pass --resume')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, default=DEFAULT_RECIPE)
    parser.add_argument('--model', type=Path, default=os.getenv('MODEL'))
    parser.add_argument('--dataset', default=os.getenv('DATASET'))
    parser.add_argument('--cached-dataset', default=os.getenv('CACHED_DATASET'))
    parser.add_argument('--output', type=Path, default=os.getenv('OUTPUT_DIR'))
    parser.add_argument('--node-rank', type=int, default=os.getenv('NODE_RANK'))
    parser.add_argument('--master-addr', default=os.getenv('MASTER_ADDR'))
    parser.add_argument('--master-port', type=int, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--dry-run', action='store_true', default=os.getenv('DRY_RUN') == '1')
    args = parser.parse_args()
    try:
        plan, env, config = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return
        preflight(args, plan, config)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    receipt = args.output / 'launch'
    receipt.mkdir(parents=True, exist_ok=True)
    (receipt / f'node-{plan["node_rank"]}.json').write_text(json.dumps(plan, indent=2) + '\n')
    subprocess.run(plan['command'], cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()

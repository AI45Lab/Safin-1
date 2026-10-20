#!/usr/bin/env python3
"""Opt-in historical HSD-GDN/FLA training and inference from this checkout."""
import argparse
import glob
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VENDOR = ROOT / 'third_party/flash-linear-attention'
PROFILES = {
    'hsd_gdn': ('hsd_gdn_0p8b.json', 'hsd_gated_deltanet'),
    'hsd_gdn_topk4': ('hsd_gdn_0p8b_topk4.json', 'hsd_gated_deltanet'),
    'hsd_gdn_readq32': ('hsd_gdn_0p8b_readq32.json', 'hsd_gated_deltanet'),
    'gdn': ('gdn_0p8b.json', 'gated_deltanet'),
    'delta_net': ('delta_net_0p8b.json', 'delta_net'),
    'kda': ('kda_0p8b.json', 'kda'),
    'transformer': ('transformer_0p8b.json', 'transformer'),
}


def configure_imports():
    for path in reversed([str(VENDOR), str(HERE / 'src'), str(ROOT)]):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def environment():
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join([str(VENDOR), str(HERE / 'src'), str(ROOT)]),
               OFFLINE='1', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               HF_EVALUATE_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
    return env


def model_classes(model_type):
    configure_imports()
    from march_research.runtime import model_classes as load
    return load(model_type)


def profile(args):
    filename, model_type = PROFILES[args.profile]
    config = args.config or HERE / 'configs' / filename
    data = json.loads(config.read_text())
    if data.get('model_type') != model_type:
        raise ValueError(f'Profile {args.profile} conflicts with model_type in {config}')
    if model_type == 'hsd_gated_deltanet':
        from march_research.runtime_policy import sanitize_research_config
        data = sanitize_research_config(data)
    return config.resolve(), model_type, data


def validate_tokenizer_vocabulary(tokenizer, config):
    expected = int(config['vocab_size'])
    if len(tokenizer) != expected:
        raise ValueError(f'Tokenizer has {len(tokenizer)} tokens; model config requires {expected}')


def generation_inputs(tokenizer, prompt, device):
    encoded = tokenizer(prompt, return_tensors='pt', return_attention_mask=True)
    return {name: encoded[name].to(device) for name in ['input_ids', 'attention_mask']}


def initialize(args):
    config_file, model_type, data = profile(args)
    if args.dry_run:
        print(json.dumps({'action': 'init', 'profile': args.profile, 'config': str(config_file),
                          'model_type': model_type, 'seed': args.seed,
                          'tokenizer': str(args.tokenizer), 'output': str(args.output)}, indent=2))
        return
    if args.output.exists():
        raise ValueError(f'Use a new seed directory: {args.output}')
    if args.tokenizer is None or not args.tokenizer.is_dir():
        raise ValueError('TOKENIZER / --tokenizer must be an existing local tokenizer directory')
    from march_research.runtime import assert_runtime
    assert_runtime()
    import random
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    config_cls, model_cls = model_classes(model_type)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    validate_tokenizer_vocabulary(tokenizer, data)
    config = config_cls(**data)
    config.model_init_seed = args.seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = model_cls(config)
    model.config.parameter_count = sum(p.numel() for p in model.parameters())
    args.output.mkdir(parents=True)
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    print(json.dumps({'output': str(args.output), 'parameters': model.config.parameter_count,
                      'seed': args.seed}))


def train_plan(args):
    _, model_type, config = profile(args)
    model = args.model or args.output / 'seed42'
    if model_type == 'hsd_gated_deltanet' and (model / 'config.json').is_file():
        from march_research.runtime_policy import sanitize_research_config
        sanitize_research_config(json.loads((model / 'config.json').read_text()))
    env = environment()
    dataset = args.dataset
    if '*' in dataset or Path(dataset).suffix == '.arrow' or Path(dataset).is_dir():
        env['MARCH_RESEARCH_ARROW_DATASET'] = dataset
        dataset = 'march_research_pretrain_arrow'
    env.setdefault('NPROC_PER_NODE', '1')
    env.setdefault('FUSE_CROSS_ENTROPY', 'true')
    if model_type == 'hsd_gated_deltanet':
        # The recommended dense profile uses the online reader. Other historical
        # profiles retain their reader; top-k uses its serialized sparse path.
        env.setdefault('HSD_TRAIN_SCAN_BACKEND', 'flashqla_tile')
        env.setdefault('HSD_FLASHQLA_TILE_BWD_BACKEND', 'optimized')
        env.setdefault('HSD_FUSED_GLOBAL_READ', '1')
        reader = 'tile_online' if args.profile == 'hsd_gdn_readq32' else 'tile_speed'
        env.setdefault('HSD_FUSED_GLOBAL_READ_BACKEND', reader)
        if args.profile == 'hsd_gdn_readq32':
            env.setdefault('HSD_FUSED_GLOBAL_READ_STRICT', '1')
        env.setdefault('HSD_DEBUG_STATS', '0')
    command = [sys.executable, '-m', 'swift.cli.main', 'pt',
               '--external_plugins', str(HERE / 'src/march_research/swift_plugin.py'),
               '--model', str(model), '--model_type', model_type, '--dataset', dataset,
               '--output_dir', str(args.output / 'training'), '--add_version', 'false',
               '--tuner_type', 'full', '--torch_dtype', 'bfloat16', '--streaming', 'false',
               '--dataset_num_proc', str(args.dataset_num_proc), '--dataloader_num_workers', '0',
               '--max_length', str(args.max_length), '--max_steps', str(args.steps),
               '--per_device_train_batch_size', str(args.batch_size),
               '--gradient_accumulation_steps', str(args.grad_acc),
               '--learning_rate', str(args.lr), '--lr_scheduler_type', 'warmup_stable_decay',
               '--lr_scheduler_kwargs', json.dumps(args.lr_scheduler_kwargs or {'num_decay_steps': max(1, args.steps // 10)}),
               '--warmup_steps', str(args.warmup_steps), '--weight_decay', '0.1',
               '--adam_beta1', '0.9', '--adam_beta2', '0.95', '--max_grad_norm', '1.0',
               '--seed', str(args.seed), '--data_seed', str(args.seed),
               '--gradient_checkpointing', 'false', '--ddp_find_unused_parameters', 'false',
               '--packing', str(args.packing).lower(), '--packing_num_proc', str(args.dataset_num_proc),
               '--truncation_strategy', 'split', '--attn_impl', 'flash_attn',
               '--save_steps', str(args.save_steps), '--save_total_limit', '2',
               '--logging_steps', '1', '--report_to', 'none', '--split_dataset_ratio', '0',
               '--eval_strategy', 'no', '--use_logits_to_keep', 'true']
    if args.deepspeed.lower() != 'none':
        command += ['--deepspeed', args.deepspeed]
    if args.resume:
        command += ['--resume_from_checkpoint', str(args.resume), '--ignore_data_skip', 'false']
    command += args.extra
    return command, env, model, model_type


def train(args):
    command, env, model, model_type = train_plan(args)
    if args.dry_run:
        keys = ['PYTHONPATH', 'NPROC_PER_NODE', 'OFFLINE', 'HSD_TRAIN_SCAN_BACKEND',
                'HSD_FLASHQLA_TILE_BWD_BACKEND', 'HSD_FUSED_GLOBAL_READ_BACKEND',
                'HSD_FUSED_GLOBAL_READ_STRICT',
                'MARCH_RESEARCH_ARROW_DATASET']
        print(json.dumps({'action': 'train', 'profile': args.profile, 'command': command,
                          'initialize_seed': args.model is None,
                          'environment': {k: env[k] for k in keys if k in env}}, indent=2))
        return
    if args.output.exists():
        raise ValueError(f'Choose a new output directory: {args.output}')
    if not glob.glob(args.dataset):
        raise ValueError(f'Local dataset does not exist: {args.dataset}')
    configure_imports()
    from march_research.runtime import assert_runtime
    assert_runtime(require_cuda=True)
    if args.model:
        saved = json.loads((model / 'config.json').read_text())
        if saved.get('model_type') != model_type:
            raise ValueError('Checkpoint model_type conflicts with selected profile')
        if not list(model.glob('*.safetensors')) and not list(model.glob('pytorch_model*.bin')):
            raise ValueError(f'No local checkpoint weights under {model}')
    else:
        os.environ.update(env)
        initialize(argparse.Namespace(profile=args.profile, config=args.config, seed=args.seed,
                                      tokenizer=args.tokenizer, output=model, dry_run=False))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def infer(args):
    if args.dry_run:
        print(json.dumps({'action': 'infer', 'model': str(args.model), 'prompt': args.prompt,
                          'max_new_tokens': args.max_new_tokens, 'offline': True}, indent=2))
        return
    from march_research.runtime import assert_runtime
    assert_runtime(require_cuda=True)
    import torch
    from transformers import AutoTokenizer
    config = json.loads((args.model / 'config.json').read_text())
    _, model_cls = model_classes(config['model_type'])
    model, info = model_cls.from_pretrained(args.model, dtype=torch.bfloat16,
                                           local_files_only=True, output_loading_info=True)
    if any(info.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']):
        raise RuntimeError(f'Strict checkpoint load failed: {info}')
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    inputs = generation_inputs(tokenizer, args.prompt, 'cuda')
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                                use_cache=True, pad_token_id=tokenizer.eos_token_id)
    prompt_tokens = inputs['input_ids'].shape[1]
    suffix = output[0, prompt_tokens:].tolist()
    print(json.dumps({'model_type': config['model_type'], 'prompt_tokens': prompt_tokens,
                      'generated_ids': suffix, 'text': tokenizer.decode(suffix)}, ensure_ascii=False))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    for name in ['init', 'train']:
        q = sub.add_parser(name)
        q.add_argument('--profile', choices=PROFILES, default=os.getenv('MODEL_PROFILE', 'hsd_gdn'))
        q.add_argument('--config', type=Path, default=os.getenv('MODEL_CONFIG'))
        q.add_argument('--tokenizer', type=Path, default=os.getenv('TOKENIZER'))
        q.add_argument('--output', type=Path, default=os.getenv('OUTPUT_DIR'), required=not os.getenv('OUTPUT_DIR'))
        q.add_argument('--seed', type=int, default=42)
        q.add_argument('--dry-run', action='store_true', default=os.getenv('DRY_RUN') == '1')
        if name == 'train':
            q.add_argument('--model', type=Path, default=os.getenv('MODEL'))
            q.add_argument('--dataset', default=os.getenv('DATASET'), required=not os.getenv('DATASET'))
            q.add_argument('--steps', type=int, default=int(os.getenv('MAX_STEPS', '20')))
            q.add_argument('--max-length', type=int, default=int(os.getenv('MAX_LENGTH', '16384')))
            q.add_argument('--batch-size', type=int, default=int(os.getenv('BATCH_SIZE', '1')))
            q.add_argument('--grad-acc', type=int, default=int(os.getenv('GRAD_ACC', '1')))
            q.add_argument('--warmup-steps', type=int, default=int(os.getenv('WARMUP_STEPS', '0')))
            q.add_argument('--save-steps', type=int, default=int(os.getenv('SAVE_STEPS', '10')))
            q.add_argument('--lr', type=float, default=float(os.getenv('LEARNING_RATE', '4e-4')))
            q.add_argument('--lr-scheduler-kwargs', type=json.loads,
                           default=os.getenv('LR_SCHEDULER_KWARGS'),
                           help='JSON object passed to the WSD scheduler')
            q.add_argument('--dataset-num-proc', type=int, default=1)
            q.add_argument('--deepspeed', default=os.getenv('DEEPSPEED', 'zero2'))
            q.add_argument('--packing', action=argparse.BooleanOptionalAction, default=True)
            q.add_argument('--resume', type=Path)
            q.add_argument('extra', nargs=argparse.REMAINDER)
    q = sub.add_parser('infer')
    q.add_argument('--model', type=Path, default=os.getenv('MODEL'), required=not os.getenv('MODEL'))
    q.add_argument('--prompt', default='Memory anchors preserve historical information. ' * 100)
    q.add_argument('--max-new-tokens', type=int, default=16)
    q.add_argument('--dry-run', action='store_true', default=os.getenv('DRY_RUN') == '1')
    return p


def main():
    args = parser().parse_args()
    if getattr(args, 'extra', [])[:1] == ['--']:
        args.extra = args.extra[1:]
    if getattr(args, 'seed', 0) not in range(2**32):
        raise ValueError('Seed must be in [0, 2**32)')
    if args.action == 'train' and min(args.steps, args.max_length, args.batch_size,
                                     args.grad_acc, args.save_steps, args.dataset_num_proc) < 1:
        raise ValueError('Training sizes must be positive')
    if args.action == 'train' and args.lr_scheduler_kwargs is not None and not isinstance(args.lr_scheduler_kwargs, dict):
        raise ValueError('lr-scheduler-kwargs must be a JSON object')
    if args.action == 'train' and not 0 <= args.warmup_steps < args.steps:
        raise ValueError('Require 0 <= warmup-steps < steps')
    configure_imports()
    os.environ.update(environment())
    globals()[args.action if args.action != 'init' else 'initialize'](args)


if __name__ == '__main__':
    main()

from __future__ import annotations

# Resolve the optional research runtime from this source checkout.
import sys
from pathlib import Path
_RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RESEARCH))
from run import configure_imports
configure_imports()
from march_research.runtime import assert_runtime
assert_runtime()

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors import safe_open

from fla.models.kda.configuration_kda import KDAConfig
from fla.models.kda.modeling_kda import KDAForCausalLM


DEFAULT_KDA_CONFIG = str(Path(__file__).resolve().parents[1] / 'configs/kda_0p8b.json')
TOKENIZER_FILES = (
    'tokenizer.json',
    'tokenizer_config.json',
    'special_tokens_map.json',
    'tokenizer.model',
    'merges.txt',
    'vocab.json',
    'added_tokens.json',
    'generation_config.json',
    'chat_template.jinja',
)
TRAINING_STATE_NAMES = {
    'trainer_state.json',
    'optimizer.pt',
    'optimizer.bin',
    'scheduler.pt',
    'scaler.pt',
    'zero_to_fp32.py',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Create a from-scratch KDA seed by copying every shape-compatible '
            'GDN tensor bitwise and initializing only the key-wise decay projection.'
        ))
    parser.add_argument('--gdn-seed-dir', required=True)
    parser.add_argument('--kda-config-json', default=DEFAULT_KDA_CONFIG)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--new-params-seed', type=int, default=42)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--safe-serialization', action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def reject_training_checkpoint(model_dir: Path) -> None:
    if model_dir.name.startswith('checkpoint-'):
        raise ValueError(f'Expected a random seed directory, not a training checkpoint: {model_dir}')
    found = []
    for path in model_dir.rglob('*'):
        if path.name in TRAINING_STATE_NAMES or path.name.startswith('rng_state'):
            found.append(str(path.relative_to(model_dir)))
    if found:
        raise ValueError(
            f'Refusing to initialize from a directory containing training state: {found[:10]}')


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f'Output directory already exists: {output_dir}')
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def build_config(config_json: Path, source_dir: Path, new_params_seed: int) -> KDAConfig:
    with config_json.open('r', encoding='utf-8') as handle:
        values = json.load(handle)
    values['model_type'] = KDAConfig.model_type
    values['architectures'] = ['KDAForCausalLM']
    values['initialization_type'] = 'from_scratch_shared_random_gdn_seed'
    values['source_random_gdn_seed'] = str(source_dir)
    values['kda_new_params_seed'] = int(new_params_seed)
    values['model_init_seed'] = int(new_params_seed)
    return KDAConfig(**values)


def find_weight_files(model_dir: Path) -> list[Path]:
    files = sorted(model_dir.glob('*.safetensors'))
    if not files:
        raise FileNotFoundError(f'No safetensors weights found in random GDN seed: {model_dir}')
    return files


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _is_gdn_decay_projection(key: str) -> bool:
    return key.endswith('.attn.a_proj.weight')


def _is_decay_bias(key: str) -> bool:
    return key.endswith('.attn.dt_bias')


def _is_kda_decay_projection(key: str) -> bool:
    return key.endswith('.attn.f_proj.0.weight') or key.endswith('.attn.f_proj.1.weight')


def copy_shared_weights(model: KDAForCausalLM, files: list[Path]) -> dict:
    """Copy all common tensors exactly, rejecting any unplanned schema difference."""
    target_state = model.state_dict()
    source_keys = set()
    copied = []
    expected_source_only = []
    unexpected = []
    shape_mismatches = []

    with torch.no_grad():
        for file_path in files:
            with safe_open(str(file_path), framework='pt', device='cpu') as handle:
                for key in handle.keys():
                    if key in source_keys:
                        raise ValueError(f'Duplicate tensor key across safetensors files: {key}')
                    source_keys.add(key)
                    if key not in target_state:
                        if _is_gdn_decay_projection(key):
                            expected_source_only.append(key)
                        else:
                            unexpected.append(key)
                        continue
                    source_tensor = handle.get_tensor(key)
                    target_tensor = target_state[key]
                    if source_tensor.shape != target_tensor.shape:
                        mismatch = {
                            'key': key,
                            'source': list(source_tensor.shape),
                            'target': list(target_tensor.shape),
                        }
                        if not _is_decay_bias(key):
                            shape_mismatches.append(mismatch)
                        else:
                            mismatch['expected'] = True
                            shape_mismatches.append(mismatch)
                        continue
                    target_tensor.copy_(source_tensor)
                    if not torch.equal(target_tensor, source_tensor):
                        raise AssertionError(f'Bitwise copy verification failed for {key}')
                    copied.append(key)

    target_only = sorted(set(target_state) - source_keys)
    expected_target_only = sorted(key for key in target_state if _is_kda_decay_projection(key))
    unexpected_mismatches = [item for item in shape_mismatches if not item.get('expected')]
    expected_mismatches = [item for item in shape_mismatches if item.get('expected')]
    if (
        unexpected
        or unexpected_mismatches
        or target_only != expected_target_only
        or len(expected_mismatches) != model.config.num_hidden_layers
        or len(expected_source_only) != model.config.num_hidden_layers
    ):
        raise ValueError(json.dumps({
            'unexpected': sorted(unexpected),
            'expected_source_only': sorted(expected_source_only),
            'shape_mismatches': shape_mismatches,
            'target_only': target_only,
            'expected_target_only': expected_target_only,
        }, indent=2))
    return {
        'copied_keys': sorted(copied),
        'copied_key_count': len(copied),
        'kda_only_keys': sorted(expected_target_only + [item['key'] for item in expected_mismatches]),
        'gdn_only_keys': sorted(expected_source_only),
        'expected_shape_mismatches': expected_mismatches,
        'unexpected_keys': sorted(unexpected),
        'all_shared_tensors_bitwise_equal': True,
    }


def initialize_kda_only_parameters(model: KDAForCausalLM, seed: int) -> dict:
    """Initialize f_proj and expanded dt_bias without perturbing copied tensors."""
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    new_keys = []
    new_parameter_count = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if module.__class__.__name__ != 'KimiDeltaAttention':
                continue
            for index in (0, 1):
                weight = module.f_proj[index].weight
                torch.nn.init.normal_(
                    weight,
                    mean=0.0,
                    std=model.config.initializer_range,
                    generator=generator,
                )
                new_keys.append(f'{name}.f_proj.{index}.weight')
                new_parameter_count += weight.numel()

            dt = torch.exp(
                torch.rand(module.dt_bias.shape, generator=generator, dtype=torch.float32)
                * (math.log(0.1) - math.log(0.001))
                + math.log(0.001),
            ).clamp(min=1e-4)
            module.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            module.dt_bias._no_weight_decay = True
            new_keys.append(f'{name}.dt_bias')
            new_parameter_count += module.dt_bias.numel()

    expected_key_count = 3 * model.config.num_hidden_layers
    if len(new_keys) != expected_key_count:
        raise AssertionError(f'Expected {expected_key_count} KDA-only parameter keys, got {len(new_keys)}')
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError('Every KDA parameter must remain trainable for from-scratch training')
    return {
        'new_parameter_keys': sorted(new_keys),
        'new_parameter_key_count': len(new_keys),
        'new_parameter_count': new_parameter_count,
        'new_params_seed': seed,
        'all_parameters_trainable': True,
    }


def copy_tokenizer_files(source_dir: Path, output_dir: Path) -> None:
    for filename in TOKENIZER_FILES:
        source = source_dir / filename
        if source.is_file():
            shutil.copy2(source, output_dir / filename)


def main() -> None:
    args = parse_args()
    source_dir = Path(args.gdn_seed_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    config_json = Path(args.kda_config_json).resolve()
    reject_training_checkpoint(source_dir)
    prepare_output_dir(output_dir, args.overwrite)

    config = build_config(config_json, source_dir, args.new_params_seed)
    model = KDAForCausalLM(config)
    weight_files = find_weight_files(source_dir)
    shared_report = copy_shared_weights(model, weight_files)
    new_report = initialize_kda_only_parameters(model, args.new_params_seed)

    if config.hidden_size == 1536 and config.num_heads == 6 and config.num_hidden_layers == 21:
        if new_report['new_parameter_count'] != 21_700_224:
            raise AssertionError(
                f"Expected 21,700,224 KDA-only parameters, got {new_report['new_parameter_count']}")

    config.save_pretrained(output_dir)
    copy_tokenizer_files(source_dir, output_dir)
    model.save_pretrained(output_dir, safe_serialization=args.safe_serialization)

    report = {
        'initialization_type': 'from_scratch_shared_random_gdn_seed',
        'global_step': 0,
        'source_random_gdn_seed': str(source_dir),
        'source_weight_files': [
            {
                'name': path.name,
                'size_bytes': path.stat().st_size,
                'sha256': sha256_file(path),
            }
            for path in weight_files
        ],
        **shared_report,
        **new_report,
        'optimizer_state_copied': False,
        'scheduler_state_copied': False,
        'trainer_state_copied': False,
        'rng_state_copied': False,
    }
    report_path = output_dir / 'init_kda_from_gdn_seed_report.json'
    with report_path.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write('\n')
    (output_dir / '.init_kda_seed.done').write_text('ok\n', encoding='utf-8')
    print(f'Created from-scratch KDA seed: {output_dir}')
    print(f"Copied shared tensors: {shared_report['copied_key_count']}")
    print(f"New trainable parameters: {new_report['new_parameter_count']}")
    print(f'Report: {report_path}')


if __name__ == '__main__':
    main()

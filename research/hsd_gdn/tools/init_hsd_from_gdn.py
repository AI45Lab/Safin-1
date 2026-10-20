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
import json
import os
import shutil
from pathlib import Path
from typing import Dict

import torch
from safetensors import safe_open

from march_research.hsd.configuration_hsd_gated_deltanet import HSDGatedDeltaNetConfig
from march_research.hsd.modeling_hsd_gated_deltanet import HSDGatedDeltaNet, HSDGatedDeltaNetForCausalLM


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Initialize an HSD Gated DeltaNet HF directory from a trained Gated DeltaNet checkpoint.')
    parser.add_argument('--gdn-model-dir', required=True, help='Source GatedDeltaNet HF checkpoint directory.')
    parser.add_argument('--hsd-config-json', required=True, help='HSD config JSON to use for the target model.')
    parser.add_argument('--output-dir', required=True, help='Target HSD HF checkpoint directory.')
    parser.add_argument('--overwrite', action='store_true', help='Remove output_dir first if it already exists.')
    parser.add_argument(
        '--safe-serialization',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Whether to save target weights in safetensors format. Enabled by default.')
    parser.add_argument(
        '--init-hsd-projections-from-gdn',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Initialize HSD summary/router projections from GDN q/k projections when shapes match.')
    return parser.parse_args()


def prepare_output_dir(output_dir: str, overwrite: bool) -> None:
    output_path = Path(output_dir)
    if output_path.exists() and overwrite:
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)


def copy_tokenizer_files(src_dir: str, output_dir: str) -> None:
    for filename in TOKENIZER_FILES:
        src = os.path.join(src_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, filename))


def build_hsd_config(config_json: str) -> HSDGatedDeltaNetConfig:
    with open(config_json, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    config_dict['model_type'] = 'hsd_gated_deltanet'
    config_dict['architectures'] = ['HSDGatedDeltaNetForCausalLM']
    return HSDGatedDeltaNetConfig(**config_dict)


def find_safetensors_files(model_dir: str) -> list[str]:
    files = sorted(str(path) for path in Path(model_dir).glob('*.safetensors'))
    if not files:
        raise FileNotFoundError(f'No *.safetensors files found in {model_dir}')
    return files


def index_safetensors(files: list[str]) -> Dict[str, str]:
    key_to_file: Dict[str, str] = {}
    for file_path in files:
        with safe_open(file_path, framework='pt', device='cpu') as handle:
            for key in handle.keys():
                key_to_file[key] = file_path
    return key_to_file


def load_tensor(key_to_file: Dict[str, str], key: str) -> torch.Tensor | None:
    file_path = key_to_file.get(key)
    if file_path is None:
        return None
    with safe_open(file_path, framework='pt', device='cpu') as handle:
        return handle.get_tensor(key)


def load_matching_gdn_weights(model: HSDGatedDeltaNetForCausalLM, gdn_model_dir: str,
                              init_hsd_projections_from_gdn: bool) -> dict:
    target_state = model.state_dict()
    source_files = find_safetensors_files(gdn_model_dir)
    key_to_file = index_safetensors(source_files)
    loaded_state = {}
    skipped_shape = []

    for key in sorted(set(target_state) & set(key_to_file)):
        tensor = load_tensor(key_to_file, key)
        if tensor is None:
            continue
        if tuple(tensor.shape) == tuple(target_state[key].shape):
            loaded_state[key] = tensor
        else:
            skipped_shape.append((key, tuple(tensor.shape), tuple(target_state[key].shape)))

    projection_copies = []
    if init_hsd_projections_from_gdn:
        for layer_idx in range(model.config.num_hidden_layers):
            prefix = f'model.layers.{layer_idx}.attn.'
            candidates = {
                prefix + 'summary_q_proj.weight': prefix + 'q_proj.weight',
                prefix + 'summary_k_proj.weight': prefix + 'k_proj.weight',
                prefix + 'router_q_proj.weight': prefix + 'q_proj.weight',
            }
            for target_key, source_key in candidates.items():
                if target_key in loaded_state or target_key not in target_state:
                    continue
                source_tensor = load_tensor(key_to_file, source_key)
                if source_tensor is None:
                    continue
                if tuple(source_tensor.shape) == tuple(target_state[target_key].shape):
                    loaded_state[target_key] = source_tensor.clone()
                    projection_copies.append((target_key, source_key))

    embedding_key = 'model.summary_embedding.weight'
    token_embedding_key = 'model.embeddings.weight'
    if embedding_key in target_state and embedding_key not in loaded_state:
        token_embedding = load_tensor(key_to_file, token_embedding_key)
        placeholder_id = int(model.config.summary_placeholder_token_id)
        if token_embedding is not None and token_embedding.ndim == 2 and token_embedding.shape[1] == target_state[embedding_key].shape[1]:
            row = token_embedding[placeholder_id].detach().clone()
            loaded_state[embedding_key] = row.view(1, -1).repeat(target_state[embedding_key].shape[0], 1)

    incompatible = model.load_state_dict(loaded_state, strict=False)
    kda_gate_copies = []
    for module_name, module in model.named_modules():
        if isinstance(module, HSDGatedDeltaNet) and module.sync_kda_gate_from_gdn():
            kda_gate_copies.append(module_name)
    return {
        'source_files': source_files,
        'loaded': len(loaded_state),
        'target': len(target_state),
        'missing': list(incompatible.missing_keys),
        'unexpected': list(incompatible.unexpected_keys),
        'skipped_shape': skipped_shape,
        'projection_copies': projection_copies,
        'kda_gate_copies': kda_gate_copies,
    }


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir, args.overwrite)

    config = build_hsd_config(args.hsd_config_json)
    model = HSDGatedDeltaNetForCausalLM(config)
    report = load_matching_gdn_weights(
        model=model,
        gdn_model_dir=args.gdn_model_dir,
        init_hsd_projections_from_gdn=args.init_hsd_projections_from_gdn,
    )

    config.save_pretrained(args.output_dir)
    copy_tokenizer_files(args.gdn_model_dir, args.output_dir)
    model.save_pretrained(args.output_dir, safe_serialization=args.safe_serialization)

    report_path = os.path.join(args.output_dir, 'init_hsd_from_gdn_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write('\n')

    print(f'Created HSD checkpoint initialized from GDN: {args.output_dir}')
    print(f'Loaded {report["loaded"]}/{report["target"]} target tensors.')
    print(f'Missing target tensors after load: {len(report["missing"])}')
    print(f'Projection copies from GDN q/k: {len(report["projection_copies"])}')
    print(f'KDA vector gates initialized from GDN scalar gates: {len(report["kda_gate_copies"])}')
    print(f'Report: {report_path}')


if __name__ == '__main__':
    main()

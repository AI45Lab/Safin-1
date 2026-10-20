"""Contracts for the restored historical HSD/FLA source workspace."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / 'research/hsd_gdn'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('profile,model_type', [
    ('hsd_gdn', 'hsd_gated_deltanet'), ('hsd_gdn_topk4', 'hsd_gated_deltanet'),
    ('hsd_gdn_readq32', 'hsd_gated_deltanet'),
    ('gdn', 'gated_deltanet'), ('delta_net', 'delta_net'),
    ('kda', 'kda'), ('transformer', 'transformer'),
])
def test_real_launcher_resolves_profile_without_loading_cuda(tmp_path, profile, model_type):
    env = dict(os.environ)
    for key in ['MODEL', 'MODEL_CONFIG', 'TOKENIZER', 'DATASET', 'OUTPUT_DIR', 'MODEL_PROFILE']:
        env.pop(key, None)
    output = tmp_path / 'unused'
    result = subprocess.run([sys.executable, str(RESEARCH / 'run.py'), 'train',
                             '--profile', profile, '--dataset', str(tmp_path / 'train.jsonl'),
                             '--tokenizer', str(tmp_path / 'tokenizer'), '--output', str(output),
                             '--dry-run'], env=env, check=True, capture_output=True, text=True)
    plan = json.loads(result.stdout)
    command = plan['command']
    assert command[command.index('--model_type') + 1] == model_type
    assert command[command.index('--max_length') + 1] == '16384'
    assert command[command.index('--max_steps') + 1] == '20'
    assert plan['environment']['OFFLINE'] == '1'
    assert str(ROOT / 'third_party/flash-linear-attention') in plan['environment']['PYTHONPATH']
    assert not output.exists()


def test_wrong_config_family_fails_before_training(tmp_path):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'model_type': 'safin1_4b'}))
    result = subprocess.run([sys.executable, str(RESEARCH / 'run.py'), 'train',
                             '--profile', 'gdn', '--config', str(config),
                             '--dataset', '/path/to/data.jsonl', '--output', str(tmp_path / 'out'),
                             '--dry-run'], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'conflicts with model_type' in result.stderr
    assert not (tmp_path / 'out').exists()


def test_historical_sources_match_restoration_and_explicit_cleanup():
    manifest = json.loads((RESEARCH / 'SOURCE_MANIFEST.json').read_text())
    for original, entry in manifest['hsd_sources'].items():
        restored = RESEARCH / 'src/march_research/hsd' / original.removeprefix('loglinear/hsd/')
        cleanup = manifest.get('public_cleanup', {})
        if original in cleanup.get('removed_sources', []):
            assert not restored.exists()
            continue
        raw = restored.read_bytes()
        changed = cleanup.get('changed_sources', {}).get(original)
        if changed:
            assert changed['restored_sha256'] == entry['restored_sha256']
            assert hashlib.sha256(raw).hexdigest() == changed['current_sha256']
        else:
            assert hashlib.sha256(raw).hexdigest() == entry['restored_sha256']
            assert hashlib.sha256(raw.replace(b'march_research.hsd', b'loglinear.hsd')).hexdigest() == entry['original_sha256']
    for name, entry in manifest['configs'].items():
        assert hashlib.sha256((RESEARCH / 'configs' / name).read_bytes()).hexdigest() == entry['sha256']


def test_historical_fla_modifications_survive_restoration():
    vendor = ROOT / 'third_party/flash-linear-attention'
    provenance = json.loads((vendor / 'PROVENANCE.json').read_text())
    for original, checksum in provenance['historical_overlays'].items():
        path = vendor / 'fla' / original.removeprefix('loglinear/fla/')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum
    assert len(provenance['wheels']) == 2
    for name in ['LICENSE', 'NOTICE', 'UPSTREAM.md']:
        assert (vendor / name).is_file()


def test_research_cannot_silently_use_an_installed_fla(monkeypatch):
    monkeypatch.syspath_prepend(str(RESEARCH / 'src'))
    runtime = load('march_research.runtime', RESEARCH / 'src/march_research/runtime.py')
    monkeypatch.setitem(sys.modules, 'fla', SimpleNamespace(__file__='/other-environment/fla/__init__.py'))
    with pytest.raises(RuntimeError, match='vendored FLA fork'):
        runtime.assert_runtime()




def test_research_code_is_private_path_free():
    scanner = load('research_release_scan', ROOT / 'scripts/march/check_public_release.py')
    assert not scanner.scan(RESEARCH)
    assert not scanner.scan(ROOT / 'third_party/flash-linear-attention')


def test_generation_preserves_attention_mask_with_shared_pad_eos():
    import torch
    runner = load('research_generation_contract', RESEARCH / 'run.py')

    class Tokenizer:
        def __call__(self, prompt, **kwargs):
            assert kwargs['return_attention_mask']
            # EOS occurs both inside the prompt and in a padded position.
            return {'input_ids': torch.tensor([[1, 2, 7, 2]]),
                    'attention_mask': torch.tensor([[1, 1, 1, 0]])}

    inputs = runner.generation_inputs(Tokenizer(), 'example', 'cpu')
    assert inputs['attention_mask'].tolist() == [[1, 1, 1, 0]]
    assert inputs['input_ids'].tolist() == [[1, 2, 7, 2]]


def test_seed_rejects_the_wrong_tokenizer_vocabulary():
    runner = load('research_vocabulary_contract', RESEARCH / 'run.py')

    class Tokenizer:
        def __len__(self):
            return 248320

    with pytest.raises(ValueError, match='model config requires 32000'):
        runner.validate_tokenizer_vocabulary(Tokenizer(), {'vocab_size': 32000})


@pytest.mark.parametrize('smoke', [False, True])
@pytest.mark.parametrize('entrypoint,profile', [('train_topk4_reference.sh', 'hsd_gdn_topk4'),
                                               ('train_reference.sh', 'hsd_gdn_readq32')])
def test_reference_preserves_historical_scheduler_and_batch(tmp_path, smoke, entrypoint, profile):
    env = dict(os.environ)
    for key in ['MODEL', 'MODEL_CONFIG', 'MODEL_PROFILE', 'NPROC_PER_NODE', 'MAX_STEPS',
                'MAX_LENGTH', 'BATCH_SIZE', 'GRAD_ACC', 'WARMUP_STEPS', 'SAVE_STEPS',
                'LEARNING_RATE', 'LR_SCHEDULER_KWARGS', 'DEEPSPEED']:
        env.pop(key, None)
    env.update(PYTHON_BIN=sys.executable, TOKENIZER=str(tmp_path / 'tokenizer'),
               DATASET=str(tmp_path / 'train.jsonl'), OUTPUT_DIR=str(tmp_path / 'new'),
               DRY_RUN='1')
    if smoke:
        env.update(NPROC_PER_NODE='1', BATCH_SIZE='1', GRAD_ACC='1',
                   MAX_STEPS='20', WARMUP_STEPS='2', SAVE_STEPS='10')
    result = subprocess.run(['bash', str(RESEARCH / entrypoint)],
                            env=env, check=True, capture_output=True, text=True)
    plan = json.loads(result.stdout)
    command = plan['command']
    recipe = json.loads((RESEARCH / 'configs/hsd_gdn_0p8b_readq32_full32.json').read_text())
    arguments = recipe['training']
    assert plan['profile'] == profile
    assert command[command.index('--max_length') + 1] == str(arguments['max_length'])
    scheduler = json.loads(command[command.index('--lr_scheduler_kwargs') + 1])
    if smoke:
        assert scheduler == {'num_decay_steps': 2, 'decay_type': 'cosine', 'min_lr_ratio': 0.1}
        assert command[command.index('--max_steps') + 1] == '20'
        assert command[command.index('--warmup_steps') + 1] == '2'
    else:
        assert scheduler == arguments['lr_scheduler_kwargs']
        assert command[command.index('--max_steps') + 1] == str(arguments['max_steps'])
        assert command[command.index('--warmup_steps') + 1] == '256'
        micro = int(command[command.index('--per_device_train_batch_size') + 1])
        accumulation = int(command[command.index('--gradient_accumulation_steps') + 1])
        assert int(plan['environment']['NPROC_PER_NODE']) * micro * accumulation == 256
    assert not (tmp_path / 'new').exists()

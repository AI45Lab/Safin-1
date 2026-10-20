"""CPU contracts for the configuration-driven, per-node research launcher."""
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / 'research/hsd_gdn'
RECIPE = RESEARCH / 'configs/hsd_gdn_0p8b_readq32_full32.json'


def load_runner():
    spec = importlib.util.spec_from_file_location('march_research_recipe', RESEARCH / 'train_recipe.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def clean_environment(tmp_path):
    env = dict(os.environ)
    for name in ('MODEL', 'TOKENIZER', 'MODEL_CONFIG', 'MODEL_PROFILE', 'DATASET', 'CACHED_DATASET',
                 'OUTPUT_DIR', 'NODE_RANK', 'MASTER_ADDR', 'MASTER_PORT', 'DRY_RUN',
                 'HSD_FUSED_GLOBAL_READ_BACKEND', 'HSD_FUSED_GLOBAL_READ_STRICT'):
        env.pop(name, None)
    env.update(MODEL=str(tmp_path / 'model'), DATASET=str(tmp_path / 'data.jsonl'),
               OUTPUT_DIR=str(tmp_path / 'output'), NODE_RANK='0', MASTER_ADDR='192.0.2.10',
               PYTHON_BIN=sys.executable)
    return env


@pytest.mark.parametrize('rank', range(4))
def test_dry_run_builds_configured_torchrun_on_each_node_without_cuda(tmp_path, rank):
    env = clean_environment(tmp_path)
    env.update(NODE_RANK=str(rank), DRY_RUN='1', HSD_FUSED_GLOBAL_READ_BACKEND='tile_speed')
    result = subprocess.run(['bash', str(RESEARCH / 'train_full32.sh')],
                            env=env, capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    command = plan['command']
    for key, value in (('--nnodes', '4'), ('--nproc-per-node', '8'), ('--node-rank', str(rank)),
                       ('--master-addr', '192.0.2.10'), ('--optim', 'adamw_torch_fused'),
                       ('--gradient_accumulation_steps', '4'), ('--max_steps', '11921'),
                       ('--padding_free', 'true')):
        assert command[command.index(key) + 1] == value
    assert json.loads(command[command.index('--lr_scheduler_kwargs') + 1]) == {
        'num_decay_steps': 1192, 'decay_type': 'cosine', 'min_lr_ratio': 0.1}
    assert plan['environment']['HSD_FUSED_GLOBAL_READ_BACKEND'] == 'tile_online'
    assert plan['environment']['HSD_FUSED_GLOBAL_READ_STRICT'] == '1'
    assert plan['environment']['NODE_RANK'] == str(rank)
    assert plan['global_batch_size'] == 256
    assert plan['nominal_packed_token_slots'] == 50000297984
    assert not (tmp_path / 'output').exists()
    # With site packages disabled, planning still requires no PyTorch/Swift/FLA.
    subprocess.run([sys.executable, '-S', str(RESEARCH / 'train_recipe.py'), '--dry-run'],
                   env=env, capture_output=True, text=True, check=True)


def test_cached_dataset_is_exclusive_and_passed_as_encoded_input(tmp_path):
    env = clean_environment(tmp_path)
    env.pop('DATASET')
    env['CACHED_DATASET'] = str(tmp_path / 'encoded')
    result = subprocess.run([sys.executable, str(RESEARCH / 'train_recipe.py'), '--dry-run'],
                            env=env, capture_output=True, text=True, check=True)
    command = json.loads(result.stdout)['command']
    assert '--dataset' not in command
    assert command[command.index('--cached_dataset') + 1] == env['CACHED_DATASET']
    env['DATASET'] = str(tmp_path / 'text.jsonl')
    rejected = subprocess.run([sys.executable, str(RESEARCH / 'train_recipe.py'), '--dry-run'],
                              env=env, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert 'exactly one' in rejected.stderr


def test_cli_node_rank_updates_the_worker_cache_namespace(tmp_path, monkeypatch):
    env = clean_environment(tmp_path)
    result = subprocess.run([sys.executable, str(RESEARCH / 'train_recipe.py'),
                             '--node-rank', '2', '--dry-run'],
                            env=env, capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan['node_rank'] == 2
    assert plan['environment']['NODE_RANK'] == '2'
    monkeypatch.setenv('NODE_RANK', plan['environment']['NODE_RANK'])
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('MARCH_TRITON_CACHE_BASE', str(tmp_path / 'cache'))
    monkeypatch.setenv('TRITON_CACHE_DIR', str(tmp_path / 'initial-cache'))
    monkeypatch.delenv('SWIFT_DISABLE_RANK_PRIVATE_TRITON_CACHE', raising=False)
    runpy.run_path(str(ROOT / 'swift/cli/pt.py'))['configure_rank_private_triton_cache']()
    assert Path(os.environ['TRITON_CACHE_DIR']) == tmp_path / 'cache/node_2_local_0'


def test_arrow_glob_uses_the_research_plugin_instead_of_a_hub_lookup(tmp_path):
    env = clean_environment(tmp_path)
    env['DATASET'] = str(tmp_path / 'data-*.arrow')
    result = subprocess.run([sys.executable, str(RESEARCH / 'train_recipe.py'), '--dry-run'],
                            env=env, capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    command = plan['command']
    assert command[command.index('--dataset') + 1] == 'march_research_pretrain_arrow'
    assert plan['environment']['MARCH_RESEARCH_ARROW_DATASET'] == env['DATASET']


def test_generated_training_arguments_are_accepted_by_the_real_swift_parser(tmp_path, monkeypatch):
    from transformers import HfArgumentParser
    from swift.arguments import PretrainArguments

    runner = load_runner()
    env = clean_environment(tmp_path)
    args = type('Args', (), dict(recipe=RECIPE, node_rank=0, master_addr='192.0.2.10',
                                master_port=29500, model=Path(env['MODEL']), output=Path(env['OUTPUT_DIR']),
                                dataset=env['DATASET'], cached_dataset=None, resume=None))()
    plan, _, _ = runner.build_plan(args)
    command = plan['command']
    # Exercise real CLI types/options without post-init model I/O or GPU setup.
    monkeypatch.setattr(PretrainArguments, '__post_init__', lambda self: None)
    (parsed,) = HfArgumentParser(PretrainArguments).parse_args_into_dataclasses(
        command[command.index('--external_plugins'):])
    assert parsed.max_steps == 11921
    assert parsed.gradient_accumulation_steps == 4
    assert parsed.optim == 'adamw_torch_fused'
    assert parsed.train_dataloader_shuffle is True


@pytest.mark.parametrize('section,key,value,error', [
    ('distributed', 'world_size', 16, 'world_size'),
    ('distributed', 'global_batch_size', 128, 'global_batch_size'),
    ('training', 'warmup_steps', 11921, 'warmup_steps'),
    ('training', 'max_steps', 11922, 'Nominal token budget'),
    ('training', 'output_dir', '/path/to/conflicting-output', 'belong to the launcher'),
])
def test_invalid_recipe_fails_before_writing_output(tmp_path, section, key, value, error):
    recipe = json.loads(RECIPE.read_text())
    recipe['model_config'] = str(RESEARCH / 'configs' / recipe['model_config'])
    recipe[section][key] = value
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(recipe))
    result = subprocess.run([sys.executable, str(RESEARCH / 'train_recipe.py'),
                             '--recipe', str(path), '--dry-run'], env=clean_environment(tmp_path),
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert error in result.stderr
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('variable,value,error', [
    ('NODE_RANK', '4', 'node count'), ('MASTER_ADDR', '', 'rank-0'),
    ('MASTER_PORT', '65536', 'MASTER_PORT'),
])
def test_invalid_rendezvous_fails_before_launch(tmp_path, variable, value, error):
    env = clean_environment(tmp_path)
    env[variable] = value
    result = subprocess.run([sys.executable, str(RESEARCH / 'train_recipe.py'), '--dry-run'],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert error in result.stderr


def test_actual_launch_checks_inputs_then_records_the_exact_plan(tmp_path, monkeypatch):
    runner = load_runner()
    env = clean_environment(tmp_path)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    model = Path(env['MODEL'])
    model.mkdir()
    config = json.loads((RESEARCH / 'configs/hsd_gdn_0p8b_readq32.json').read_text())
    (model / 'config.json').write_text(json.dumps(config))
    (model / 'model.safetensors').touch()
    Path(env['DATASET']).write_text('{"text":"training fixture"}\n')
    calls = []

    def launch(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(runner.subprocess, 'run', launch)
    monkeypatch.setattr(sys, 'argv', ['train_recipe.py'])
    runner.main()
    assert len(calls) == 1
    receipt = json.loads((Path(env['OUTPUT_DIR']) / 'launch/node-0.json').read_text())
    assert calls[0][0] == receipt['command']
    assert calls[0][1]['check'] is True
    assert calls[0][1]['env']['HSD_FUSED_GLOBAL_READ_BACKEND'] == 'tile_online'
    # Reject a different architecture before starting any distributed child.
    config['hsd_read_topk'] = 4
    (model / 'config.json').write_text(json.dumps(config))
    with pytest.raises(SystemExit):
        runner.main()
    assert len(calls) == 1


@pytest.mark.parametrize('profile,override,expected', [
    ('hsd_gdn_readq32', None, 'tile_online'),
    ('hsd_gdn_readq32', 'tile_speed', 'tile_speed'),
    ('hsd_gdn', None, 'tile_speed'),
    ('hsd_gdn_topk4', None, 'tile_speed'),
])
def test_profile_reader_default_and_explicit_historical_comparison(tmp_path, profile, override, expected):
    env = clean_environment(tmp_path)
    if override:
        env['HSD_FUSED_GLOBAL_READ_BACKEND'] = override
    result = subprocess.run([sys.executable, str(RESEARCH / 'run.py'), 'train',
                             '--profile', profile, '--dry-run'],
                            env=env, capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan['environment']['HSD_FUSED_GLOBAL_READ_BACKEND'] == expected
    if profile == 'hsd_gdn_readq32':
        assert plan['environment']['HSD_FUSED_GLOBAL_READ_STRICT'] == '1'

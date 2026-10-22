"""Publication must not accept a checkbox without current installation evidence."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/march/check_installation_evidence.py'
EVIDENCE = ROOT / 'docs/march/installation-acceptance-20261017.json'


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def evidence(tmp_path):
    checker = load(SCRIPT)
    # Synthetic source files exercise source invalidation without copying the
    # large training/native trees. The immutable real report is checked below.
    names = list(checker.RECIPE_FILES) + list(checker.NATIVE_FILES)
    names += [name + '/runtime.py' for name in checker.TRAINING_DIRECTORIES]
    names += [name + '/kernel.cu' for name in checker.NATIVE_DIRECTORIES]
    names += ['requirements/dependencies.txt', 'configs/march/model.yaml']
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic acceptance fixture\n')
    data = json.loads(EVIDENCE.read_text())
    data['source_fingerprints'] = checker.fingerprints(tmp_path)
    return checker, tmp_path, data


def test_actual_frozen_installation_receipt_matches_current_source():
    checker = load(SCRIPT)
    result = checker.review(ROOT, json.loads(EVIDENCE.read_text()))
    assert result['ready'], result


def test_complete_separately_scoped_receipts_are_accepted(evidence):
    checker, root, data = evidence
    assert checker.review(root, data)['ready']


@pytest.mark.parametrize('change', [
    'training_edit', 'training_add', 'training_remove', 'native_edit',
    'native_add', 'installer', 'requirements', 'checkpoint_loader',
    'profile', 'package_recipe', 'bridge', 'file_symlink', 'directory_symlink',
])
def test_changes_to_consumed_sources_invalidate_the_evidence(evidence, change):
    checker, root, data = evidence
    path = root / 'march/runtime.py'
    if change == 'training_remove':
        path.unlink()
    elif change == 'file_symlink':
        path.unlink()
        target = root / 'elsewhere.py'
        target.write_text('# redirected\n')
        path.symlink_to(target)
    elif change == 'directory_symlink':
        target = root / 'elsewhere'
        target.mkdir()
        (target / 'runtime.py').write_text('# redirected\n')
        (root / 'swift/runtime.py').unlink()
        (root / 'swift').rmdir()
        (root / 'swift').symlink_to(target, target_is_directory=True)
    else:
        paths = {
            'training_edit': 'march/runtime.py',
            'training_add': 'swift/new_training_path.py',
            'native_edit': 'third_party/vllm/csrc/kernel.cu',
            'native_add': 'third_party/vllm/cmake/new_build.cmake',
            'installer': 'scripts/march/install.sh',
            'requirements': 'requirements/dependencies.txt',
            'checkpoint_loader': 'scripts/march/verify_checkpoint.py',
            'profile': 'configs/march/model.yaml',
            'package_recipe': 'setup.py',
            'bridge': 'third_party/mcore-bridge/src/runtime.py',
        }
        (root / paths[change]).write_text('# changed after acceptance\n')
    assert not checker.review(root, data)['ready']


def test_documentation_and_release_checks_do_not_expire_runtime_evidence(evidence):
    checker, root, data = evidence
    for name in ['docs/march/note.md', 'scripts/march/check_publication.py']:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# publication-only change\n')
    assert checker.review(root, data)['ready']


@pytest.mark.parametrize('change', [
    'schema', 'partial_build', 'partial_cpu', 'missing_model', 'duplicate_model',
    'no_step', 'nan_loss', 'infinite_gradient', 'no_gradient', 'wrong_tuner',
    'wrong_model_type', 'unmerged_lora', 'no_weights', 'mismatched_weights',
    'nonfinite_logits', 'no_generation', 'legacy_keys', 'no_runtime',
    'missing_log_hash', 'cpu_gpu_source', 'cpu_gpu_manifest', 'no_cpu_install',
    'missing_cpu_wheel', 'prebuilt_native', 'missing_native_module',
    'bad_native_hash', 'missing_native_receipt', 'wrong_native_source',
    'misstated_repair', 'no_serving', 'missing_serving_model', 'no_graph',
    'wrong_sequence_length', 'online', 'changed_input', 'missing_pip_check',
    'bad_fingerprints', 'bad_cpu_object', 'bad_graph_object',
    'no_te_build', 'no_te_hash', 'missing_native_import', 'changed_saved_runtime',
])
def test_incomplete_or_inconsistent_receipts_are_rejected(evidence, change):
    checker, root, data = evidence
    gpu = data['gpu_clean_build']
    training = gpu['training_full_size']
    case = training['cases'][0]
    cpu = data['fresh_repaired_training_cpu']
    native = gpu['native_source_build']
    serving = gpu['vllm_serving']
    if change == 'no_te_build':
        cpu['te_bindings_source_build']['status'] = 'pending'
    elif change == 'no_te_hash':
        cpu['te_bindings_source_build'].pop('built_wheel_sha256')
    elif change == 'missing_native_import':
        cpu['native_imports'].pop()
    elif change == 'changed_saved_runtime':
        case['runtime']['hsd_read_block_size'] = 16384
    elif change == 'schema':
        data.pop('installation_evidence_schema_version')
    elif change == 'partial_build':
        gpu['status'] = 'partial_pass'
    elif change == 'partial_cpu':
        cpu['status'] = 'pending'
    elif change == 'missing_model':
        training['cases'].pop()
    elif change == 'duplicate_model':
        training['cases'][1] = deepcopy(case)
    elif change == 'no_step':
        case['optimizer_steps'] = 0
    elif change == 'nan_loss':
        case['loss'] = float('nan')
    elif change == 'infinite_gradient':
        case['grad_norm'] = float('inf')
    elif change == 'no_gradient':
        case['grad_norm'] = 0
    elif change == 'wrong_tuner':
        training['cases'][1]['tuner'] = 'full'
    elif change == 'wrong_model_type':
        case['model_type'] = 'historical_qwen3_5'
    elif change == 'unmerged_lora':
        training['cases'][1]['merged_full_weight_export'] = False
    elif change == 'no_weights':
        case['weight_keys'] = 0
    elif change == 'mismatched_weights':
        case['missing_unexpected_mismatched_weights'] = 1
    elif change == 'nonfinite_logits':
        case['finite_reload_logits'] = False
    elif change == 'no_generation':
        case['greedy_generated_tokens'] = 0
    elif change == 'legacy_keys':
        case['legacy_summary_config_keys'] = 1
    elif change == 'no_runtime':
        case.pop('runtime')
    elif change == 'missing_log_hash':
        case['training_log_sha256'] = None
    elif change == 'cpu_gpu_source':
        cpu['source_revision'] = '0' * 40
    elif change == 'cpu_gpu_manifest':
        cpu['source_manifest_sha256'] = '1' * 64
    elif change == 'no_cpu_install':
        cpu['installer'] = 'pending'
    elif change == 'missing_cpu_wheel':
        cpu['package_and_fixture_gates'].pop('cpu-framework-wheel')
    elif change == 'prebuilt_native':
        native['inherited_native_binaries'] = 8
    elif change == 'missing_native_module':
        native['extensions'].pop('vllm._C')
    elif change == 'bad_native_hash':
        native['extensions']['vllm._C']['sha256'] = 'not-a-hash'
    elif change == 'missing_native_receipt':
        native.pop('native_receipt_sha256')
    elif change == 'wrong_native_source':
        native['source_archive_sha256'] = '2' * 64
    elif change == 'misstated_repair':
        serving['dependency_repair'] = 'rebuilt'
    elif change == 'no_serving':
        serving['status'] = 'failed'
    elif change == 'missing_serving_model':
        serving['cases'].pop()
    elif change == 'no_graph':
        serving['cases'][0]['graph_dispatches'] = {}
    elif change == 'wrong_sequence_length':
        training['sequence_length'] = 8192
    elif change == 'online':
        training['offline'] = False
    elif change == 'changed_input':
        training['input_weights_and_saved_runtime_unchanged'] = False
    elif change == 'missing_pip_check':
        training.pop('pip_check')
    elif change == 'bad_fingerprints':
        data['source_fingerprints'] = []
    elif change == 'bad_cpu_object':
        data['fresh_repaired_training_cpu'] = []
    elif change == 'bad_graph_object':
        serving['cases'][0]['graph_dispatches'] = []
    assert not checker.review(root, data)['ready']


def publication_fixture(root, data):
    scripts = root / 'scripts/march'
    shutil.copyfile(SCRIPT, scripts / SCRIPT.name)
    shutil.copyfile(ROOT / 'scripts/march/check_publication.py', scripts / 'check_publication.py')
    # Numerical and TP/DP gates have independent negative tests. Isolate them
    # to exercise the installation checker through the real publication CLI.
    for name in ['check_operator_correctness.py', 'check_serving_evidence.py']:
        (scripts / name).write_text('def review(root, data): return {"ready": True}\n')
    docs = root / 'docs/march'
    docs.mkdir(parents=True)
    for name in ['operator-correctness-20261015.json', 'serving.json']:
        (docs / name).write_text('{}')
    (docs / 'installation.json').write_text(json.dumps(data))
    (docs / 'card.md').write_text('synthetic model card\n')
    metadata = json.loads((ROOT / 'configs/release/publication.json').read_text())
    metadata['status'] = 'ready'
    metadata['models'] = [{'name': 'fixture', 'card': 'docs/march/card.md'}]
    metadata['evidence'] = {
        'fresh_installation': 'docs/march/installation.json',
        'batch_dp_long_context': 'docs/march/serving.json',
    }
    config = root / 'configs/release/publication.json'
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps(metadata))
    return config, metadata


@pytest.mark.parametrize('change', [
    'valid', 'checkbox_only', 'outside', 'invalid_json', 'changed_runtime',
    'bad_evidence_map', 'bad_evidence_path', 'missing_evidence_path', 'allow_draft',
])
def test_real_publication_cli_checks_the_installation_receipt(evidence, change):
    _, root, data = evidence
    config, metadata = publication_fixture(root, data)
    if change in ('checkbox_only', 'allow_draft'):
        (root / 'docs/march/installation.json').unlink()
    elif change == 'outside':
        outside = root.parent / (root.name + '-outside.json')
        outside.write_text(json.dumps(data))
        metadata['evidence']['fresh_installation'] = '../' + outside.name
    elif change == 'invalid_json':
        (root / 'docs/march/installation.json').write_text('{')
    elif change == 'changed_runtime':
        (root / 'swift/runtime.py').write_text('# changed after GPU acceptance\n')
    elif change == 'bad_evidence_map':
        metadata['evidence'] = None
    elif change == 'bad_evidence_path':
        metadata['evidence']['fresh_installation'] = []
    elif change == 'missing_evidence_path':
        metadata['evidence'].pop('fresh_installation')
    config.write_text(json.dumps(metadata))
    args = [sys.executable, str(root / 'scripts/march/check_publication.py'),
            '--root', str(root), '--scope', 'code']
    if change == 'allow_draft':
        args.append('--allow-draft')
    result = subprocess.run(args, text=True, capture_output=True)
    report = json.loads(result.stdout)
    assert (result.returncode == 0) == (change in ('valid', 'allow_draft'))
    assert report['ready'] == (change == 'valid')
    if change != 'valid':
        assert 'acceptance.clean_cuda_build.evidence' in report['pending']


def test_standalone_checker_does_not_import_torch(evidence):
    _, root, data = evidence
    path = root / 'receipt.json'
    path.write_text(json.dumps(data))
    program = 'import runpy, sys; sys.modules["torch"] = None; path = sys.argv.pop(1); runpy.run_path(path, run_name="__main__")'
    result = subprocess.run(
        [sys.executable, '-c', program, str(SCRIPT), '--root', str(root), '--evidence', str(path)],
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['ready']


@pytest.mark.parametrize("count", [None, 0, 2, True, "1"])
def test_installation_requires_recorded_single_gpu_training(evidence, count):
    checker, root, data = evidence
    data["gpu_clean_build"]["training_full_size"]["gpu_count"] = count
    assert not checker.review(root, data)["ready"]

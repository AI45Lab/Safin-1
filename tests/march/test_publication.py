"""Publication controls must distinguish drafts, readiness and private overlays."""
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts/march' / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_model_revision_is_not_publishable(monkeypatch):
    path = ROOT / 'configs/release/publication.json'
    metadata = json.loads(path.read_text())
    metadata['models'][0]['revision'] = None
    original_read = Path.read_text

    def read_text(candidate, *args, **kwargs):
        if candidate.resolve() == path.resolve():
            return json.dumps(metadata)
        return original_read(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read_text)
    report = load_script('check_publication.py').review(ROOT)
    assert not report['schema_errors']
    assert not report['ready']
    assert f"models.{metadata['models'][0]['name']}.revision" in report['pending']


def test_model_reviews_remain_required_without_an_explicit_policy(tmp_path):
    required, errors = load_script('check_publication.py').model_review_requirements(tmp_path, {})
    assert not errors
    assert set(required) == {'training_data_review', 'evaluation_report'}


def test_model_review_policy_requires_matching_record_and_preserves_unknowns(tmp_path):
    checker = load_script('check_publication.py')
    policy = {'training_data_review': 'disclose_only', 'evaluation_report': 'optional'}
    metadata = {'models': [{'name': 'fixture'}], 'training_data_review': None, 'evaluation_report': None,
                'model_release_policy': {**policy, 'decision_record': 'decision.json'}}
    required, errors = checker.model_review_requirements(tmp_path, metadata)
    assert errors and len(required) == 2
    (tmp_path / 'decision.json').write_text(json.dumps({
        'status': 'maintainer_decision', 'policy': policy, 'models': ['fixture']}))
    required, errors = checker.model_review_requirements(tmp_path, metadata)
    assert not required and not errors
    assert metadata['training_data_review'] is None and metadata['evaluation_report'] is None
    (tmp_path / 'decision.json').write_text(json.dumps({
        'status': 'maintainer_decision', 'policy': policy, 'models': ['other-model']}))
    required, errors = checker.model_review_requirements(tmp_path, metadata)
    assert errors and len(required) == 2


def test_model_review_policy_rejects_unknown_state_and_outside_record(tmp_path):
    checker = load_script('check_publication.py')
    metadata = {'model_release_policy': {'training_data_review': 'approved_without_review'}}
    assert checker.model_review_requirements(tmp_path, metadata)[1]
    metadata = {'models': [{'name': 'fixture'}], 'model_release_policy': {
        'training_data_review': 'disclose_only', 'decision_record': '../decision.json'}}
    assert checker.model_review_requirements(tmp_path, metadata)[1]


def test_strict_vendor_provenance_reports_missing_revision(tmp_path):
    checker = load_script('check_vendor_manifest.py')
    directory = tmp_path / 'vendor'
    directory.mkdir()
    for name in ['LICENSE', 'NOTICE', 'UPSTREAM.md']:
        (directory / name).write_text('fixture\n')
    (tmp_path / 'third_party').mkdir()
    manifest = {'components': [{'name': 'fixture', 'path': 'vendor',
                              'upstream_commit': None, 'files': checker.inventory(directory)}]}
    (tmp_path / 'third_party/dependency-manifest.json').write_text(json.dumps(manifest))
    result = subprocess.run(['python', str(ROOT / 'scripts/march/check_vendor_manifest.py'),
                             '--root', str(tmp_path), '--strict-provenance'], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'fixture' in result.stdout


def test_tracked_private_overlay_is_not_hidden_by_tree_exclusion(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / '.local').mkdir()
    (tmp_path / '.local/checkpoints.env').write_text('MODEL=/path/to/private-model\n')
    subprocess.run(['git', '-C', str(tmp_path), 'add', '-f', '.local'], check=True)
    result = subprocess.run(['python', str(ROOT / 'scripts/march/check_public_release.py'), str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'tracked_private_overlay' in result.stdout


def test_history_reports_removed_secret_without_printing_value(tmp_path):
    scanner = load_script('check_public_release.py')
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    def commit(message):
        subprocess.run(['git', '-C', str(tmp_path), 'add', '-A'], check=True)
        subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Fixture',
                        '-c', 'user.email=fixture@localhost', 'commit', '-qm', message], check=True)
    (tmp_path / 'scanner.txt').write_text('pattern=r"rlaunch.*--"')
    commit('regex literal')
    assert scanner.history_findings(tmp_path) == []
    secret = 'hf_' + 'A' * 32
    (tmp_path / 'bad.txt').write_text(secret)
    commit('fixture secret')
    (tmp_path / 'bad.txt').unlink()
    commit('remove fixture')
    report = scanner.history_findings(tmp_path)
    assert any(item['file'] == 'bad.txt' for item in report)
    assert secret not in json.dumps(report)


def test_vllm_install_dry_run_uses_source_fork_version():
    import os
    environment = os.environ.copy()
    environment['DRY_RUN'] = '1'
    environment.pop('VLLM_VERSION_OVERRIDE', None)
    result = subprocess.run(['bash', str(ROOT / 'scripts/march/install.sh'), 'vllm'],
                            env=environment, capture_output=True, text=True, check=True)
    assert 'VLLM_VERSION_OVERRIDE=0.20.2rc1.dev92+g20cac26b1.march0.1.0' in result.stdout
    assert 'VLLM_USE_PRECOMPILED=0' in result.stdout

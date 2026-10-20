import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_scan_distinguishes_parameter_names_from_values(tmp_path):
    spec = importlib.util.spec_from_file_location('public_scan', ROOT/'scripts/march/check_public_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path/'api.py').write_text('def connect(api_key=None, password=None): pass\n')
    assert module.scan(tmp_path) == []
    # Build synthetic strings at runtime to avoid putting a credential-shaped
    # literal or a private filesystem prefix in the public source tree.
    (tmp_path/'bad.txt').write_text('hf_' + 'A'*32 + '\n' + '/mnt/shared' + '-storage/private/model\n')
    report = module.scan(tmp_path)
    assert {item['kind'] for item in report} == {'credential_pattern', 'private_path_or_launcher'}
    assert all(set(item) == {'kind', 'file', 'line'} for item in report)
    assert all(item['file'] == 'bad.txt' for item in report)


def test_scan_prunes_local_records_and_catches_artifacts(tmp_path):
    spec = importlib.util.spec_from_file_location('public_scan_local', ROOT/'scripts/march/check_public_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    local = tmp_path/'.local'
    local.mkdir()
    (local/'locations.txt').write_text('/mnt/shared' + '-storage/private/model')
    assert module.scan(tmp_path) == []
    (tmp_path/'weights.safetensors').write_bytes(b'fixture')
    assert module.scan(tmp_path) == [{'kind': 'artifact', 'file': 'weights.safetensors'}]


def test_scan_prunes_local_agent_links(tmp_path):
    spec = importlib.util.spec_from_file_location('public_scan_agents', ROOT/'scripts/march/check_public_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agents = tmp_path/'.agents'/'skills'
    agents.mkdir(parents=True)
    (agents/'internal').symlink_to('/tmp/internal-skill')
    assert module.scan(tmp_path) == []


def test_model_scan_never_reads_weight_payloads_but_scans_metadata(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('public_model_scan', ROOT/'scripts/march/check_public_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    weights = tmp_path/'model.safetensors'
    weights.write_bytes(b'weight payload is validated separately')
    (tmp_path/'config.json').write_text('{"_name_or_path":"' + '/mnt/shared' + '-storage/private/model"}')
    original = Path.read_bytes

    def guarded_read(path):
        if path.suffix == '.safetensors':
            raise AssertionError('The model scanner must not read weight payloads')
        return original(path)

    monkeypatch.setattr(Path, 'read_bytes', guarded_read)
    findings = module.scan(tmp_path, model_files=True)
    assert findings == [{'kind': 'private_path_or_launcher', 'file': 'config.json', 'line': 1}]
    assert {'kind': 'artifact', 'file': 'model.safetensors'} in module.scan(tmp_path)
    weights.unlink()
    weights.symlink_to(tmp_path/'config.json')
    assert {'kind': 'symlink', 'file': 'model.safetensors'} in module.scan(tmp_path, model_files=True)


def test_scan_private_infrastructure_without_private_organization_literals(tmp_path):
    spec = importlib.util.spec_from_file_location('infrastructure_scan', ROOT/'scripts/march/check_public_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    private_registry = '.'.join(['registry', 'node', 'example', 'cn']) + '/team/image:v1'
    private_host = '.'.join(['worker', 'project', 'svc', 'example', 'local'])
    path = tmp_path/'endpoint.txt'
    for value in [private_registry, private_host]:
        path.write_text(value)
        assert module.scan(tmp_path) == [{'kind': 'private_path_or_launcher', 'file': 'endpoint.txt', 'line': 1}]
    path.write_text('registry.npmjs.org/package\nservice.default.svc.cluster.local\n')
    assert module.scan(tmp_path) == []


def test_history_scan_checks_identity_without_printing_private_value(tmp_path):
    import json
    import os
    import subprocess
    import sys

    private_email = 'contributor@' + '.'.join(['worker', 'project', 'svc', 'example', 'local'])
    env = dict(os.environ, GIT_AUTHOR_NAME='Fixture', GIT_AUTHOR_EMAIL=private_email,
               GIT_COMMITTER_NAME='Fixture', GIT_COMMITTER_EMAIL=private_email)
    subprocess.run(['git', 'init', '-q', str(tmp_path)], env=env, check=True)
    (tmp_path/'source.py').write_text('value = 1\n')
    subprocess.run(['git', '-C', str(tmp_path), 'add', 'source.py'], env=env, check=True)
    subprocess.run(['git', '-C', str(tmp_path), 'commit', '-qm', 'fixture'], env=env, check=True)
    result = subprocess.run([sys.executable, str(ROOT/'scripts/march/check_public_release.py'),
                             str(tmp_path), '--history'], env=env, capture_output=True, text=True)
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report['tree_findings'] == report['tracked_findings'] == report['history_findings'] == []
    assert {row['field'] for row in report['git_identity_findings']} == {'author_email', 'committer_email'}
    assert private_email not in result.stdout

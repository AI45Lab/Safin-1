"""FLA artifact provenance fails on altered artifacts or unrecorded sources."""
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def verifier():
    spec = importlib.util.spec_from_file_location('fla_provenance', ROOT / 'scripts/march/verify_fla_provenance.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def snapshot(tmp_path, verifier):
    source = tmp_path / 'third_party/flash-linear-attention'
    (source / 'fla/ops').mkdir(parents=True)
    artifacts = tmp_path / 'artifacts'
    artifacts.mkdir()
    namespace = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)\n__version__ = '0.5.0'\n"
    generator = b'''content = f"""__path__ = __import__('pkgutil').extend_path(__path__, __name__)\n__version__ = '{version}'\n"""'''
    base = {'fla/__init__.py': namespace, 'fla/ops/kernel.py': b'upstream\n'}
    archive = artifacts / 'upstream.tar.gz'
    with tarfile.open(archive, 'w:gz') as handle:
        for name, data in {'fla/__init__.py': b'original init\n', 'fla/ops/kernel.py': base['fla/ops/kernel.py'],
                           'scripts/build_packages.py': generator}.items():
            member = tarfile.TarInfo('upstream/' + name)
            member.size = len(data)
            handle.addfile(member, io.BytesIO(data))
    wheel = artifacts / 'fixture.whl'
    with zipfile.ZipFile(wheel, 'w') as handle:
        for name, data in base.items():
            handle.writestr(name, data)
    for name, data in base.items():
        (source / name).write_bytes(data)
    (source / 'fla/ops/kernel.py').write_bytes(b'historical local change\n')
    provenance = {'version': '0.5.0', 'upstream_commit': 'fixture',
                  'upstream_verification': {'archive': {'filename': archive.name,
                                                       'sha256': verifier.sha256(archive.read_bytes())}},
                  'wheels': [{'filename': wheel.name, 'sha256': verifier.sha256(wheel.read_bytes())}],
                  'historical_overlays': {'loglinear/fla/ops/kernel.py':
                                          verifier.sha256(b'historical local change\n')}}
    (source / 'PROVENANCE.json').write_text(json.dumps(provenance))
    return tmp_path, artifacts, source


def test_recorded_overlay_and_generated_namespace_are_reproducible(verifier, snapshot):
    root, artifacts, _ = snapshot
    result = verifier.verify(root, artifacts)
    assert result['status'] == 'passed'
    assert result['changed_or_added_files_from_wheels'] == 1


@pytest.mark.parametrize('altered', ['archive', 'wheel', 'overlay', 'unrecorded'])
def test_tampered_or_unrecorded_input_is_rejected(verifier, snapshot, altered):
    root, artifacts, source = snapshot
    if altered in {'archive', 'wheel'}:
        path = artifacts / ('upstream.tar.gz' if altered == 'archive' else 'fixture.whl')
        path.write_bytes(path.read_bytes() + b'tampered')
    elif altered == 'overlay':
        (source / 'fla/ops/kernel.py').write_bytes(b'not recorded\n')
    else:
        (source / 'fla/ops/unknown.py').write_bytes(b'extra source\n')
    with pytest.raises(ValueError):
        verifier.verify(root, artifacts)


def test_namespace_template_never_evaluates_arbitrary_expressions(verifier):
    source = '''value = f"extend_path {__import__('os').getcwd()}"'''
    with pytest.raises(ValueError, match='Unsupported expression'):
        verifier.namespace_init(source, '0.5.0')

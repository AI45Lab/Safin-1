#!/usr/bin/env python3
"""Verify the optional FLA snapshot from locally prepared upstream artifacts.

This command is CPU-only and never downloads, extracts, imports or executes
upstream code. See third_party/flash-linear-attention/UPSTREAM.md for inputs.
"""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def namespace_init(source, version):
    """Read the upstream split-package literal without evaluating Python code."""
    templates = [node for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.JoinedStr) and any(
                     isinstance(value, ast.Constant) and isinstance(value.value, str)
                     and 'extend_path' in value.value for value in node.values)]
    if len(templates) != 1:
        raise ValueError('Expected exactly one upstream namespace template')
    parts = []
    for value in templates[0].values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        elif (isinstance(value, ast.FormattedValue) and isinstance(value.value, ast.Name)
              and value.value.id == 'version' and value.conversion == -1
              and value.format_spec is None):
            parts.append(version)
        else:
            raise ValueError('Unsupported expression in upstream namespace template')
    return ''.join(parts).encode()


def checked_bytes(path, expected):
    data = path.read_bytes()
    if sha256(data) != expected:
        raise ValueError(f'Artifact SHA-256 mismatch: {path.name}')
    return data


def verify(root, artifacts):
    directory = root / 'third_party/flash-linear-attention'
    provenance = json.loads((directory / 'PROVENANCE.json').read_text())
    verification = provenance['upstream_verification']
    archive = artifacts / verification['archive']['filename']
    archive_bytes = checked_bytes(archive, verification['archive']['sha256'])
    with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as handle:
        sources = {'/'.join(member.name.split('/')[1:]): handle.extractfile(member).read()
                   for member in handle.getmembers() if member.isfile()}
    base = {}
    for wheel in provenance['wheels']:
        path = artifacts / wheel['filename']
        wheel_bytes = checked_bytes(path, wheel['sha256'])
        with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as handle:
            files = {name: handle.read(name) for name in handle.namelist()
                     if name.startswith('fla/') and not name.endswith('/')}
        if base.keys() & files.keys():
            raise ValueError('Unexpected overlapping FLA wheel sources')
        base.update(files)
    generated = namespace_init(sources['scripts/build_packages.py'], provenance['version'])
    for name, data in base.items():
        expected = generated if name == 'fla/__init__.py' else sources.get(name)
        if data != expected:
            raise ValueError(f'Wheel source differs from verified upstream: {name}')
    overlays = {name.removeprefix('loglinear/'): digest
                for name, digest in provenance['historical_overlays'].items()}
    local = {path.relative_to(directory).as_posix(): path.read_bytes()
             for path in (directory / 'fla').rglob('*') if path.is_file()
             and '__pycache__' not in path.parts and path.suffix not in {'.pyc', '.pyo'}}
    if local.keys() != base.keys() | overlays.keys():
        raise ValueError('Local FLA source inventory differs from wheels plus recorded overlays')
    for name, data in local.items():
        if name in overlays:
            if sha256(data) != overlays[name]:
                raise ValueError(f'Historical overlay SHA-256 mismatch: {name}')
        elif data != base[name]:
            raise ValueError(f'Unrecorded local FLA change: {name}')
    return {'status': 'passed', 'upstream_commit': provenance['upstream_commit'],
            'wheel_package_files': len(base), 'exact_upstream_source_files': len(base) - 1,
            'generated_namespace_files': 1, 'historical_overlay_files': len(overlays),
            'changed_or_added_files_from_wheels': sum(data != base.get(name) for name, data in local.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--artifacts', type=Path, required=True,
                        help='Directory containing the pinned upstream.tar.gz and two original wheels')
    args = parser.parse_args()
    try:
        result = verify(args.root, args.artifacts)
    except (ValueError, OSError, KeyError, tarfile.TarError, zipfile.BadZipFile) as error:
        parser.exit(1, f'FLA provenance verification failed: {error}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

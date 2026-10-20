#!/usr/bin/env python3
"""Check vendored file hashes and retained attribution files."""
import argparse
import hashlib
import json
from pathlib import Path

EXCLUDED = {'__pycache__', '.git', '.pytest_cache', '.ruff_cache', '.mypy_cache', 'build', 'dist', '.venv'}


def inventory(root):
    files = {}
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if path.is_file() and not set(rel.parts) & EXCLUDED and path.suffix not in {'.pyc', '.pyo'}:
            if path.is_symlink():
                raise ValueError(f'vendored source must not be a symlink: {rel}')
            files[rel.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--strict-provenance', action='store_true', help='Fail when upstream revisions are unresolved')
    args = parser.parse_args()
    manifest = json.loads((args.root/'third_party/dependency-manifest.json').read_text())
    unresolved = []
    for item in manifest['components']:
        directory = args.root/item['path']
        actual = inventory(directory)
        if actual != item['files']:
            changes = sorted(key for key in actual.keys() | item['files'].keys()
                             if actual.get(key) != item['files'].get(key))
            raise SystemExit(f'{item["name"]}: snapshot differs: {changes[:20]}')
        for filename in (item.get('license_file', 'LICENSE'), 'NOTICE', 'UPSTREAM.md'):
            if filename not in actual:
                raise SystemExit(f'{item["name"]}: attribution file missing: {filename}')
        print(f'{item["name"]}: {len(actual)} files match; LICENSE/NOTICE/source record present')
        if not item['upstream_commit']:
            unresolved.append(item['name'])
    if unresolved:
        print('Upstream revisions still require provenance review: ' + ', '.join(unresolved))
        if args.strict_provenance:
            raise SystemExit(1)


if __name__ == '__main__':
    main()

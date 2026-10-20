#!/usr/bin/env python3
"""Build a scanned source archive of the working tree, without Git history."""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


EXCLUDED_PARTS = {'.git', '.agents', '.local', '__pycache__', '.pytest_cache', '.ruff_cache', '.mypy_cache', '.venv'}
EXCLUDED_ROOTS = {'build', 'dist'}


def exclude_from_export(relative):
    return relative.parts[0] in EXCLUDED_ROOTS or bool(set(relative.parts) & EXCLUDED_PARTS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(root):
        raise SystemExit('Choose a new archive path outside the repository')
    tracked = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z', '--cached', '--others', '--exclude-standard'])
    with tempfile.TemporaryDirectory(prefix='march-release-') as temp:
        dest = Path(temp, 'march-safin1')
        for name in sorted(set(tracked.decode().split('\0')) - {''}):
            rel = Path(name)
            if exclude_from_export(rel):
                continue
            src = root / rel
            if not src.exists():
                continue  # deleted files still present in the index
            if src.is_symlink():
                raise SystemExit(f'Refusing source symlink: {rel}')
            if src.is_file():
                dst = dest / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        spec = importlib.util.spec_from_file_location('release_scan', root/'scripts/march/check_public_release.py')
        scanner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scanner)
        findings = scanner.scan(dest)
        if findings:
            raise SystemExit(f'Source scan failed: {findings[:20]}')
        output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output, 'w:gz') as archive:
            archive.add(dest, arcname=dest.name)
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix+'.sha256').write_text(f'{checksum}  {output.name}\n')
    print(f'{output}\nsha256: {checksum}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Scan a source snapshot without printing secrets; optionally audit Git history.

This is a targeted privacy/artifact check, not a full credential or license
audit. It does not flag generic API parameter names such as password/api_key.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

PRIVATE_PREFIX = "/mnt/shared" + "-storage"
# Match infrastructure shapes without embedding a private organization name.
PRIVATE_REGISTRY = r"\bregistry(?:\.[a-z0-9-]+){2,}\.(?:cn|local|internal)\b[^\s'\"]*"
PRIVATE_HOST = r"\b(?:[a-z0-9-]+\.)+svc\.(?!cluster\.local\b)(?:[a-z0-9-]+\.)*local\b"
PRIVATE = re.compile(
    re.escape(PRIVATE_PREFIX) + r"[^\s'\"]*|" + PRIVATE_REGISTRY + "|" + PRIVATE_HOST
    + r"|\b(?:rlaunch|rjob)\s+(?:\\\s*)?--", re.I)
SECRET = re.compile(r"(?:AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")
EXCLUDED = {".git", ".agents", ".local", "__pycache__", ".pytest_cache", ".venv", "build", "dist", ".mypy_cache", ".ruff_cache"}
ARTIFACT_DIRS = {"artifacts", "eval_cache", "eval_run_logs", "share_exports", "external_delivery"}
ARTIFACT_SUFFIXES = {".so", ".whl", ".ptx", ".mlir", ".safetensors", ".ckpt", ".pth", ".pt", ".pyc", ".log"}


def scan(root, model_files=False):
    root = Path(root).resolve()
    findings = []
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED)
        for name in files:
            path = Path(base, name)
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                findings.append({"kind": "symlink", "file": rel})
                continue
            if (path.suffix.lower() in ARTIFACT_SUFFIXES and not (model_files and path.suffix.lower() == '.safetensors')) or ARTIFACT_DIRS & set(path.relative_to(root).parts):
                findings.append({"kind": "artifact", "file": rel})
                continue
            # Model shards are allowed binary payloads. Do not read multi-GB
            # weights into RAM; checkpoint validation checks their headers/index.
            if model_files and path.suffix.lower() == '.safetensors':
                continue
            try:
                content = path.read_bytes()
                if b'\x00' in content:
                    continue
                content = content.decode('utf-8')
            except (OSError, UnicodeError):
                continue
            for line, text in enumerate(content.splitlines(), 1):
                for kind, pattern in (("private_path_or_launcher", PRIVATE), ("credential_pattern", SECRET)):
                    if pattern.search(text):
                        findings.append({"kind": kind, "file": rel, "line": line})
    return findings


def tracked_findings(root):
    result = subprocess.run(['git', '-C', str(root), 'ls-files', '-z'],
                            text=True, capture_output=True)
    if result.returncode:
        return []  # standalone source/model directory
    findings = []
    for name in result.stdout.split('\0'):
        if name and {'.local', '.agents'} & set(Path(name).parts):
            findings.append({'kind': 'tracked_private_overlay', 'file': name})
    return findings


def history_findings(root):
    # Select candidate changes with Git, then check actual blobs with the same
    # patterns as the tree scanner. Regex literals are not leaked values.
    pattern = PRIVATE_PREFIX + r'|registry\.|\.svc\.|rlaunch.*--|rjob.*--|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{30,}|BEGIN .*PRIVATE KEY'
    exists = subprocess.run(['git', '-C', str(root), 'rev-parse', '--verify', 'HEAD'],
                            capture_output=True)
    if exists.returncode:
        return []
    result = subprocess.run(['git', '-C', str(root), 'log', '--all', '--format=COMMIT:%H',
                             '--name-only', '-G', pattern], text=True, capture_output=True, check=True)
    findings = []
    seen = set()
    commit = None
    for name in result.stdout.splitlines():
        if name.startswith('COMMIT:'):
            commit = name.split(':', 1)[1]
            continue
        if not name or commit is None:
            continue
        parents = subprocess.check_output(['git', '-C', str(root), 'show', '-s', '--format=%P', commit], text=True).split()
        for revision in [commit, *parents]:
            blob = subprocess.run(['git', '-C', str(root), 'show', f'{revision}:{name}'],
                                  text=True, errors='replace', capture_output=True)
            if blob.returncode:
                continue
            for line, value in enumerate(blob.stdout.splitlines(), 1):
                for kind, regex in (('private_path_or_launcher', PRIVATE), ('credential_pattern', SECRET)):
                    if regex.search(value):
                        key = (revision, name, line, kind)
                        if key not in seen:
                            seen.add(key)
                            findings.append({'commit': revision, 'file': name, 'line': line, 'kind': kind})
    return findings



def identity_findings(root):
    """Check commit identities without returning their private values."""
    result = subprocess.run(
        ['git', '-C', str(root), 'log', '--all', '--format=%H%x00%an%x00%ae%x00%cn%x00%ce'],
        text=True, capture_output=True)
    if result.returncode:
        return []  # standalone source directory or repository without commits
    findings = []
    names = ('author_name', 'author_email', 'committer_name', 'committer_email')
    for line in result.stdout.splitlines():
        fields = line.split('\0')
        if len(fields) != 5:
            continue
        for name, value in zip(names, fields[1:]):
            if PRIVATE.search(value):
                findings.append({'kind': 'private_git_identity', 'commit': fields[0], 'field': name})
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--history', action='store_true')
    parser.add_argument('--model-files', action='store_true', help='Permit safetensors in a reviewed model directory')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = {"tree_findings": scan(args.root, model_files=args.model_files),
              "tracked_findings": tracked_findings(args.root)}
    if args.history:
        report['history_findings'] = history_findings(args.root)
        report['git_identity_findings'] = identity_findings(args.root)
    output = json.dumps(report, indent=2) + '\n'
    if args.report:
        args.report.write_text(output)
    print(output)
    return int(any(report.values()))


if __name__ == '__main__':
    raise SystemExit(main())

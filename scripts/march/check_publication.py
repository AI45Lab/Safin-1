#!/usr/bin/env python3
"""Validate publication metadata; draft mode checks structure without claiming readiness."""
import argparse
import json
from pathlib import Path
import re
import runpy


def model_review_requirements(root, metadata):
    """Keep missing reviews visible while allowing an explicit release policy."""
    defaults = {'training_data_review': 'required', 'evaluation_report': 'required'}
    policy = metadata.get('model_release_policy', {})
    if not isinstance(policy, dict):
        return list(defaults), ['model_release_policy must be an object']
    errors = []
    if set(policy) - {*defaults, 'decision_record'}:
        errors.append('model_release_policy contains unsupported fields')
    allowed = {'training_data_review': {'required', 'disclose_only'},
               'evaluation_report': {'required', 'optional'}}
    choices = {field: policy.get(field, default) for field, default in defaults.items()}
    for field, value in choices.items():
        if not isinstance(value, str) or value not in allowed[field]:
            errors.append(f'model_release_policy.{field}: invalid state')
    relaxed = any(choices[field] != default for field, default in defaults.items())
    if relaxed and not errors:
        try:
            name = policy.get('decision_record')
            if not isinstance(name, str):
                raise ValueError('missing decision record')
            path = (Path(root) / name).resolve()
            if not path.is_relative_to(Path(root).resolve()):
                raise ValueError('decision outside checkout')
            decision = json.loads(path.read_text())
            names = sorted(model['name'] for model in metadata['models'])
            if (decision.get('status') != 'maintainer_decision'
                    or decision.get('policy') != choices
                    or sorted(decision.get('models', [])) != names):
                raise ValueError('decision does not match this release')
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            errors.append('model_release_policy requires a matching in-checkout maintainer decision')
    if errors:
        return list(defaults), errors
    return [field for field, value in choices.items() if value == 'required'], []


def review(root, scope='all'):
    root = Path(root).resolve()
    metadata = json.loads((root / 'configs/release/publication.json').read_text())
    errors = []
    pending = []
    if metadata.get('status') not in {'draft', 'ready'}:
        errors.append('status must be draft or ready')
    required = ['source_repository', 'maintainer_contact', 'release_tag', 'code_license']
    if scope in {'all', 'models'}:
        model_required, policy_errors = model_review_requirements(root, metadata)
        required += ['weight_license', *model_required]
        errors.extend(policy_errors)
    for key in required:
        value = metadata.get(key)
        if not isinstance(value, str) or not value.strip():
            pending.append(key)
    repository = metadata.get('source_repository')
    if repository and not re.fullmatch(r'https://[^\s]+/[^/\s]+/[^/\s]+', repository):
        errors.append('source_repository must be a public repository URL')
    acceptance = metadata.get('acceptance', {})
    for key in ('cpu_regression', 'clean_cuda_build', 'license_review', 'credential_review',
                'batch_dp_long_context', 'numerical_limits_documented', 'operator_correctness'):
        value = acceptance.get(key, 'pending')
        if value not in {'pending', 'complete'}:
            errors.append(f'acceptance.{key}: invalid state')
        elif value != 'complete':
            pending.append('acceptance.' + key)
    # A manually set checkbox cannot override missing/stale CUDA evidence.
    operator_checker = root / 'scripts/march/check_operator_correctness.py'
    operator_evidence = root / 'docs/march/operator-correctness-20261015.json'
    try:
        operator_review = runpy.run_path(str(operator_checker))['review']
        if not operator_review(root, json.loads(operator_evidence.read_text()))['ready']:
            pending.append('acceptance.operator_correctness.evidence')
    except (OSError, ValueError, KeyError):
        pending.append('acceptance.operator_correctness.evidence')
    # A complete TP/DP checkbox requires a full, current-source API receipt.
    if acceptance.get('batch_dp_long_context') == 'complete':
        try:
            evidence_map = metadata.get('evidence')
            if not isinstance(evidence_map, dict):
                raise ValueError('missing serving evidence map')
            evidence_name = evidence_map.get('batch_dp_long_context')
            if not isinstance(evidence_name, str):
                raise ValueError('missing serving evidence path')
            evidence_path = (root / evidence_name).resolve()
            if not evidence_path.is_relative_to(root):
                raise ValueError('serving evidence outside checkout')
            serving_review = runpy.run_path(str(root / 'scripts/march/check_serving_evidence.py'))['review']
            if not serving_review(root, json.loads(evidence_path.read_text()))['ready']:
                pending.append('acceptance.batch_dp_long_context.evidence')
        except (OSError, ValueError, TypeError, KeyError):
            pending.append('acceptance.batch_dp_long_context.evidence')
    # Fresh CUDA readiness requires matching build and full-size training receipts.
    if acceptance.get('clean_cuda_build') == 'complete':
        try:
            evidence_map = metadata.get('evidence')
            if not isinstance(evidence_map, dict):
                raise ValueError('missing installation evidence map')
            evidence_name = evidence_map.get('fresh_installation')
            if not isinstance(evidence_name, str):
                raise ValueError('missing installation evidence path')
            evidence_path = (root / evidence_name).resolve()
            if not evidence_path.is_relative_to(root):
                raise ValueError('installation evidence outside checkout')
            install_review = runpy.run_path(str(root / 'scripts/march/check_installation_evidence.py'))['review']
            if not install_review(root, json.loads(evidence_path.read_text()))['ready']:
                pending.append('acceptance.clean_cuda_build.evidence')
        except (OSError, ValueError, TypeError, KeyError):
            pending.append('acceptance.clean_cuda_build.evidence')
    models = metadata.get('models', [])
    if not isinstance(models, list) or not models:
        errors.append('models must contain model-card entries')
        models = []
    for model in models:
        name = model.get('name', '<unnamed>')
        card = model.get('card')
        card_path = (root / card).resolve() if isinstance(card, str) else None
        if card_path is None or not card_path.is_relative_to(root) or not card_path.is_file():
            errors.append(f'{name}: model card missing or outside checkout')
        if scope in {'all', 'models'}:
            if not isinstance(model.get('model_id'), str) or not re.fullmatch(r'[^/\s]+/[^/\s]+', model['model_id']):
                pending.append(f'models.{name}.model_id')
            if not model.get('revision'):
                pending.append(f'models.{name}.revision')
    if metadata.get('status') != 'ready':
        pending.append('status')
    return {'schema_errors': errors, 'pending': pending, 'ready': not errors and not pending, 'scope': scope}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--scope', choices=('code', 'models', 'all'), default='all')
    parser.add_argument('--allow-draft', action='store_true')
    args = parser.parse_args()
    report = review(args.root.resolve(), args.scope)
    print(json.dumps(report, indent=2))
    return int(bool(report['schema_errors']) or (not args.allow_draft and not report['ready']))


if __name__ == '__main__':
    raise SystemExit(main())

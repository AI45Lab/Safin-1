#!/usr/bin/env python3
"""Check an installed MARCH package against a reviewed source checkout."""
import argparse
from pathlib import Path
import tempfile


def verify(source_root):
    source_root = Path(source_root).resolve()
    import march
    import swift
    installed = Path(march.__file__).resolve().parent.parent
    if installed == source_root:
        raise ValueError('Run outside the source checkout with the wheel installed; source imports cannot verify a wheel')
    if Path(swift.__file__).resolve().parent.parent != installed:
        raise ValueError('march and swift resolve to different installations')
    count = 0
    for package in ('march', 'swift'):
        for source in (source_root/package).rglob('*.py'):
            target = installed/source.relative_to(source_root)
            if not target.is_file() or target.read_bytes() != source.read_bytes():
                raise ValueError(f'Installed code differs from source: {source.relative_to(source_root)}')
            count += 1
    if (installed/'swift/ui').exists():
        raise ValueError('Retired UI files are present in the installed package')
    if not (installed/'march/kernels/_flashqla/NOTICE').is_file():
        raise ValueError('FlashQLA attribution is absent')
    from march.safin1.config_utils import copy_safin1_remote_code, safin1_remote_code_files
    from march.safin1 import register_safin1_models
    from transformers import AutoConfig, AutoModelForCausalLM
    register_safin1_models()
    exports = {}
    for family, name in [('safin1_4b', 'Safin1ForConditionalGeneration'),
                         ('safin1_35b_a3b', 'Safin1MoeForConditionalGeneration')]:
        config = AutoConfig.for_model(family)
        if AutoModelForCausalLM._model_mapping[type(config)].__name__ != name:
            raise ValueError(f'Model registration differs: {family}')
        with tempfile.TemporaryDirectory() as temp:
            copy_safin1_remote_code(temp, family)
            files = safin1_remote_code_files(family)
            for filename in files:
                if Path(temp, filename).read_bytes() != (installed/'march/safin1'/filename).read_bytes():
                    raise ValueError(f'Remote-code export differs: {filename}')
            exports[family] = len(files)
    return {'code_files_byte_equal': count, 'remote_code_exports': exports,
            'model_registry': 'passed', 'flashqla_notice': 'present', 'ui': 'absent'}


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.source_root), indent=2))


if __name__ == '__main__':
    main()

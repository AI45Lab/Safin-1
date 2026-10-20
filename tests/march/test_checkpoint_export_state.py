"""Execute the actual metadata copy method without optional Megatron imports."""
import ast
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def copy_metadata():
    source = ROOT / 'swift/megatron/trainers/base.py'
    tree = ast.parse(source.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == '_copy_hf_metadata_to_checkpoint')
    namespace = {'Path': Path, 'json': json, 'os': os, 'is_master': lambda: True}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[method.name]


@pytest.mark.parametrize('family', ['safin1_4b', 'safin1_35b_a3b', 'llama'])
def test_export_keeps_current_training_state_and_code(tmp_path, family):
    source, target = tmp_path / 'source', tmp_path / 'checkpoint-2'
    source.mkdir()
    target.mkdir()
    originals = {'latest_checkpointed_iteration.txt': '5961', 'args.json': 'old arguments',
                 'trainer_state.json': 'old progress', 'config.json': 'old config',
                 'modeling_safin1.py': 'old implementation', 'tokenizer.json': 'tokenizer',
                 'chat_template.jinja': 'template'}
    for name, content in originals.items():
        (source / name).write_text(content)
    saved = {'latest_checkpointed_iteration.txt': '2', 'args.json': 'new arguments',
             'trainer_state.json': 'new progress', 'config.json': json.dumps({'model_type': family}),
             'modeling_safin1.py': 'current implementation'}
    for name, content in saved.items():
        (target / name).write_text(content)
    trainer = SimpleNamespace(args=SimpleNamespace(model=str(source)), copy_path=shutil.copy2)
    copy_metadata()(trainer, str(target))
    for name in ('latest_checkpointed_iteration.txt', 'args.json', 'trainer_state.json', 'config.json'):
        assert (target / name).read_text() == saved[name]
    expected = 'old implementation' if family == 'llama' else 'current implementation'
    assert (target / 'modeling_safin1.py').read_text() == expected
    assert (target / 'tokenizer.json').read_text() == 'tokenizer'
    assert (target / 'chat_template.jinja').read_text() == 'template'
    assert all((source / name).read_text() == value for name, value in originals.items())

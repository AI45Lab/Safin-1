"""Evaluation profile, real adapter initialization and offline resource contracts."""
import argparse
import ast
import __future__
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / 'third_party/lm-evaluation-harness/lm_eval'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def evaluator(monkeypatch):
    for key in list(os.environ):
        if key.startswith(('HSD_', 'VLLM_HSD_', 'MARCH_')) or key in (
            'MODEL', 'MODEL_REVISION', 'TASKS', 'OUTPUT_DIR', 'DTYPE', 'LIMIT', 'SEED',
            'ENABLE_THINKING', 'THINK_END_TOKEN', 'APPLY_CHAT_TEMPLATE', 'OFFLINE',
            'TP_SIZE', 'DP_SIZE', 'TENSOR_PARALLEL_SIZE', 'DATA_PARALLEL_SIZE',
            'MAX_NUM_SEQS', 'MAX_MODEL_LEN', 'GPU_MEMORY_UTILIZATION',
        ):
            monkeypatch.delenv(key)
    return load('evaluation_entrypoint', ROOT / 'scripts/march/evaluate.py')


@pytest.mark.parametrize('model_type', ['safin1_4b', 'safin1_35b_a3b'])
@pytest.mark.parametrize('profile', ['graph', 'eager'])
def test_engine_parameters_survive_real_harness_cli_parser(evaluator, tmp_path, model_type, profile):
    model = tmp_path / 'model, with spaces'
    model.mkdir()
    (model / 'config.json').write_text(json.dumps({'model_type': model_type}))
    args = evaluator.parser().parse_args([
        '--backend', 'vllm', '--model', str(model), '--tasks', 'ifeval,mmlu_pro',
        '--output-dir', str(tmp_path / 'results'), '--profile', profile,
    ])
    plan, env = evaluator.build_plan(args, [], {'VLLM_HSD_BAD_CONTROL': '1', 'CUDA_VISIBLE_DEVICES': '3'})
    cli = load('harness_cli_actions', VENDOR / '_cli/utils.py')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_args', nargs='+', action=cli.MergeDictAction)
    i = plan['command'].index('--model_args')
    parsed = parser.parse_args(plan['command'][i:i+2]).model_args
    assert parsed == plan['model_args']
    assert parsed['pretrained'] == str(model)
    assert parsed['enable_prefix_caching'] is False and parsed['async_scheduling'] is False
    assert parsed['additional_config']['gdn_prefill_backend'] == 'triton'
    assert parsed['additional_config']['hsd_prefill_backend'] == ('varlen' if profile == 'graph' else 'segments')
    assert parsed['enforce_eager'] == (profile == 'eager')
    assert parsed['enable_thinking'] is False and parsed['seed'] == 123
    assert 'VLLM_HSD_BAD_CONTROL' not in env
    assert env['CUDA_VISIBLE_DEVICES'] == '3'
    assert all(env[key] == '1' for key in evaluator.OFFLINE_FLAGS)
    if profile == 'graph':
        assert env['VLLM_HSD_ENABLE_GRAPH_V3'] == '1'
        assert env['VLLM_HSD_DETERMINISTIC_PREFILL_AUTOTUNE'] == '1'
        if model_type == 'safin1_4b':
            assert env['VLLM_HSD_DETERMINISTIC_VARLEN_KKT'] == '1'
            assert env['VLLM_HSD_DETERMINISTIC_ANCHOR_READ'] == '1'
            assert env['VLLM_HSD_MOE_DECODE_BATCH_INVARIANT'] == '1'
            assert env['VLLM_HSD_MOE_BATCH_INVARIANT_COMPONENTS'] == 'norm'
            assert env['VLLM_HSD_BATCH_INVARIANT_PROJECTIONS'] == '1'
            assert env['VLLM_HSD_BATCH_INVARIANT_GATE_NORM'] == '1'
        assert parsed['compilation_config']['cudagraph_mode'] == 'FULL'
    else:
        assert 'VLLM_HSD_ENABLE_GRAPH_V3' not in env
    if model_type == 'safin1_35b_a3b':
        assert parsed['moe_backend'] == 'triton'


def test_hf_structured_arguments_and_thinking(evaluator, tmp_path):
    (tmp_path / 'config.json').write_text('{"model_type":"safin1_4b"}')
    args = evaluator.parser().parse_args([
        '--backend', 'hf', '--model', str(tmp_path), '--tasks', 'ifeval',
        '--output-dir', str(tmp_path / 'out'), '--enable-thinking', 'true',
        '--think-end-token', '</think>', '--dtype', 'float32',
    ])
    plan, _ = evaluator.build_plan(args, ['--gen_kwargs', '{"max_gen_toks":8}'], {})
    assert plan['model_args']['enable_thinking'] is True
    assert plan['model_args']['think_end_token'] == '</think>'
    assert plan['model_args']['dtype'] == 'float32'
    assert '--apply_chat_template' in plan['command']
    assert plan['command'][-1] == '{"max_gen_toks":8}'
    assert plan['profile'] is None


@pytest.mark.parametrize('extra', [
    ['--model_args', '{}'], ['--model-args={}'], ['--config', 'other.yaml'],
    ['--seed=7'], ['--hf_overrides', '{}'], ['-a', 'enable_prefix_caching=True'],
    ['-C', 'other.yaml'], ['-Cother.yaml'], ['--conf', 'other.yaml'],
    ['-Mhf'], ['-aenforce_eager=True'], ['-L5'], ['-b4'], ['-oother'], ['-tother'],
    ['--model_ar', '{}'], ['--see', '7'], ['--outp', 'other'],
])
def test_profile_cannot_be_overridden_through_passthrough(evaluator, tmp_path, extra):
    (tmp_path / 'config.json').write_text('{"model_type":"safin1_4b"}')
    args = evaluator.parser().parse_args(['--backend', 'vllm', '--model', str(tmp_path),
        '--tasks', 'ifeval', '--output-dir', str(tmp_path / 'out')])
    with pytest.raises(ValueError, match='controlled'):
        evaluator.build_plan(args, extra, {})


def adapter_fixture(monkeypatch, calls):
    """Execute the complete real constructor and chat method with local engine/tokenizer doubles."""
    source = VENDOR / 'models/vllm_causallms.py'
    original = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == 'VLLM')
    cls = ast.ClassDef(name='Adapter', bases=[ast.Name(id='object', ctx=ast.Load())],
        keywords=[], decorator_list=[], body=[n for n in original.body
        if isinstance(n, ast.FunctionDef) and n.name in ('__init__', 'apply_chat_template')])
    helper = load('thinking_resolution', VENDOR / 'models/utils_thinking.py')
    tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: kw)
    transformers = ModuleType('transformers')
    transformers.AutoConfig = SimpleNamespace(from_pretrained=lambda *a, **kw: SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    def engine(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace()
    namespace = dict(__name__='adapter_contract', object=object, find_spec=lambda name: True,
        LLM=engine, get_tokenizer=lambda *a, **kw: tokenizer,
        configure_pad_token=lambda tok, **kw: tok,
        parse_version=Version, version=lambda name: '0.8.2',
        eval_logger=logging.getLogger(__name__), resolve_thinking=helper.resolve_thinking, cast=cast)
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(module, str(source), 'exec', flags=__future__.annotations.compiler_flag), namespace)
    return namespace['Adapter']


@pytest.mark.parametrize('direct,nested,expected', [
    (None, False, False), (None, True, True),
    (True, False, True), (False, True, False), (None, None, None),
])
def test_real_adapter_preserves_thinking_and_input_mapping(monkeypatch, direct, nested, expected):
    calls = []
    cls = adapter_fixture(monkeypatch, calls)
    mapping = {'other_option': 7, 'enable_thinking': nested}
    original = dict(mapping)
    adapter = cls(pretrained='local-fixture', enable_thinking=direct,
        chat_template_args=mapping, think_end_token='</think>')
    assert adapter.enable_thinking is expected and mapping == original
    result = adapter.apply_chat_template([{'role': 'user', 'content': 'fixture'}])
    assert result['enable_thinking'] is expected and result['other_option'] == 7
    assert len(calls) == 1


def test_thinking_error_happens_before_engine_allocation(monkeypatch):
    calls = []
    cls = adapter_fixture(monkeypatch, calls)
    with pytest.raises(ValueError, match='think_end_token'):
        cls(pretrained='local-fixture', chat_template_args={'enable_thinking': True})
    assert not calls


@pytest.mark.parametrize('offline', ['0', '1'])
@pytest.mark.parametrize('missing', [None, 'ortho_context.tab'])
def test_nltk_resources_never_download(monkeypatch, offline, missing):
    helper = load('nltk_local_resources', VENDOR / 'tasks/_nltk_resources.py')
    requested = []
    def find(path):
        requested.append(path)
        if missing and path.endswith(missing):
            raise LookupError(path)
        return path
    def download(*args, **kwargs):
        pytest.fail('Resource validation must not download')
    monkeypatch.setenv('OFFLINE', offline)
    monkeypatch.setitem(sys.modules, 'nltk', SimpleNamespace(data=SimpleNamespace(find=find), download=download))
    monkeypatch.setattr(helper, 'version', lambda name: '3.9.2')
    if missing:
        with pytest.raises(LookupError, match='outside the GPU/offline task'):
            helper.require_punkt_tab()
    else:
        helper.require_punkt_tab()
    assert len(requested) == 4


def test_eval_install_includes_reference_task_dependencies():
    process = subprocess.run(['bash', str(ROOT / 'scripts/march/install.sh'), 'eval'],
        env=dict(os.environ, DRY_RUN='1', PYTHON_BIN=sys.executable), capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert 'lm-evaluation-harness' in process.stdout and 'ifeval' in process.stdout

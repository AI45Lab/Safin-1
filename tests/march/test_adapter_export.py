"""Real MCore projection PEFT conversion, safetensors and fresh HF/PEFT load.

Only the model walker is replaced by a tiny layer; generic bridge parameter
conversion is real. Host runs emulate TP1 device metadata/communication. CUDA
runs use the actual generic CUDA metadata and TP1 transfer paths.
"""
import copy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(os.getenv('MARCH_TEST_MEGATRON') != '1', reason='training environment required')
PROJECTIONS = ('anchor_q_proj', 'anchor_k_proj', 'router_q_proj', 'null_gate_proj')
RAW_PARAMETERS = ('hsd_null_read_gate_bias', 'hsd_history_gate_logit', 'hsd_history_out_rmsnorm_weight',
                  'persistent_memory_state', 'persistent_memory_route_key')


@pytest.fixture
def bridge(monkeypatch):
    from mcore_bridge.model.mm_gpts.qwen3_5_gdn import Qwen3_5Bridge
    from mcore_bridge.bridge import gpt_bridge
    value = object.__new__(Qwen3_5Bridge)
    value.config = SimpleNamespace(task_type='causal_lm', num_moe_experts=4)
    value.mcore_014 = True
    value.is_multimodal = False
    value._target_device = 'cpu'
    value._only_master_rank = False
    value._peft_format = True
    value._adapter_name = 'review'
    value._peft_target_modules = set()
    value._peft_modules_to_save = set()
    for field in ('tp_size', 'pp_size', 'etp_size', 'ep_size', 'ep_pp_size'):
        setattr(value, field, 1)
    for field in ('tp_rank', 'pp_rank', 'etp_rank', 'ep_pp_rank'):
        setattr(value, field, 0)
    for field in ('tp_group', 'pp_group', 'etp_group', 'ep_pp_group'):
        setattr(value, field, None)
    if not torch.cuda.is_available():
        # The generic exporter allocates a CUDA boolean communication flag.
        # Redirect only that host-test allocation; do not alter conversion math.
        tensor = torch.tensor
        def host_tensor(*args, **kwargs):
            if kwargs.get('device') == 'cuda':
                kwargs['device'] = 'cpu'
            return tensor(*args, **kwargs)
        monkeypatch.setattr(gpt_bridge.torch, 'tensor', host_tensor)
        value._all_gather_tp = lambda tensor, dim, expert: tensor
        value._broadcast_ep_pp = lambda tensor, expert: tensor
    return value


def adapter_projection(base, name='review'):
    from mcore_bridge.tuners import LoraParallelLinear
    projection = object.__new__(LoraParallelLinear)
    torch.nn.Module.__init__(projection)
    projection.base_layer = copy.deepcopy(base).requires_grad_(False)
    projection.lora_A = torch.nn.ModuleDict()
    projection.lora_B = torch.nn.ModuleDict()
    for index, adapter in enumerate((name, 'unused')):
        projection.lora_A[adapter] = torch.nn.Linear(base.in_features, 2, bias=False,
            device=base.weight.device, dtype=base.weight.dtype)
        projection.lora_B[adapter] = torch.nn.Linear(2, base.out_features, bias=False,
            device=base.weight.device, dtype=base.weight.dtype)
        with torch.no_grad():
            projection.lora_A[adapter].weight.fill_(0.03 + index)
            projection.lora_B[adapter].weight.fill_(0.07 + index)
    return projection


def layer_fixture(mode='lora', dtype=torch.float32):
    from peft.utils import ModulesToSaveWrapper
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    layer = torch.nn.Module()
    for name in PROJECTIONS:
        base = torch.nn.Linear(3, 2, bias=False, dtype=dtype, device=device).requires_grad_(False)
        projection = adapter_projection(base) if mode == 'lora' else base
        if mode == 'modules_to_save':
            projection = ModulesToSaveWrapper(base, 'review')
            with torch.no_grad():
                projection.modules_to_save['review'].weight.fill_(0.4)
        setattr(layer, name, projection)
    for name in RAW_PARAMETERS:
        shape = (1, 2, 2) if name == 'persistent_memory_state' else (2,)
        layer.register_parameter(name, torch.nn.Parameter(torch.full(shape, 0.7, dtype=dtype, device=device),
                                                        requires_grad=False))
    return layer


def attach_walker(bridge):
    # Keep the real export_weights/load_weights prefix and safetensors API;
    # only avoid constructing a distributed full-size Megatron decoder.
    def walk(models, state, prefix, loading, *args, **kwargs):
        for model in models:
            for layer_prefix, layer in model.routes:
                local = bridge._remove_prefix(state, prefix) if loading else {}
                result = bridge._set_historical_route_state(layer, local, layer_prefix, loading)
                if not loading:
                    yield from bridge._add_prefix(result, prefix).items()
    bridge._convert = walk


def export_file(bridge, routes, output, mode='lora'):
    from mcore_bridge.utils import StreamingSafetensorSaver
    from peft import LoraConfig
    model = torch.nn.Module()
    model.config = bridge.config
    model.routes = routes
    attach_walker(bridge)
    rows = dict(bridge.export_weights([model], target_device='cpu', peft_format=True, adapter_name='review'))
    saver = StreamingSafetensorSaver(str(output), peft_format=True)
    for key, tensor in rows.items():
        saver.add_tensor(key, tensor)
    saver.finalize()
    if rows:
        config = LoraConfig(r=2, lora_alpha=2, task_type='CAUSAL_LM',
            target_modules=sorted(bridge._peft_target_modules) or ['unused'],
            modules_to_save=sorted(bridge._peft_modules_to_save) or None)
        config.save_pretrained(output)
    return rows


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('mode', ['lora', 'modules_to_save', 'frozen'])
def test_route_adapter_safetensors_roundtrip(bridge, tmp_path, mode, dtype):
    source = layer_fixture(mode, dtype)
    base = {name: getattr(source, name).weight.detach().clone() for name in PROJECTIONS}
    prefix = bridge.hf_layers_prefix + '.0.linear_attn.'
    rows = export_file(bridge, [(prefix, source)], tmp_path, mode)
    if mode == 'frozen':
        assert rows == {}
        assert not bridge._peft_target_modules and not bridge._peft_modules_to_save
        return
    expected = {f'base_model.model.{prefix}{name}' + suffix for name in PROJECTIONS
                for suffix in (('.lora_A.weight', '.lora_B.weight') if mode == 'lora' else ('.weight',))}
    assert rows.keys() == expected
    assert all(not any(raw in key for raw in RAW_PARAMETERS) for key in rows)
    assert all('review' not in key and 'unused' not in key for key in rows)
    assert bridge._peft_target_modules == (set(PROJECTIONS) if mode == 'lora' else set())
    assert bridge._peft_modules_to_save == (set(PROJECTIONS) if mode == 'modules_to_save' else set())
    target = copy.deepcopy(source)
    with torch.no_grad():
        for name in PROJECTIONS:
            proj = getattr(target, name)
            if mode == 'lora':
                proj.lora_A['review'].weight.zero_()
                proj.lora_B['review'].weight.zero_()
            else:
                proj.modules_to_save['review'].weight.zero_()
    model = torch.nn.Module()
    model.routes = [(prefix, target)]
    bridge.load_weights([model], str(tmp_path), peft_format=True, adapter_name='review')
    for name in PROJECTIONS:
        original, loaded = getattr(source, name), getattr(target, name)
        if mode == 'lora':
            for field in ('lora_A', 'lora_B'):
                torch.testing.assert_close(getattr(loaded, field)['review'].weight,
                                           getattr(original, field)['review'].weight, rtol=0, atol=0)
                torch.testing.assert_close(getattr(loaded, field)['unused'].weight,
                                           getattr(original, field)['unused'].weight, rtol=0, atol=0)
        torch.testing.assert_close(loaded.weight, base[name], rtol=0, atol=0)
    for name in RAW_PARAMETERS:
        torch.testing.assert_close(getattr(target, name), getattr(source, name), rtol=0, atol=0)


@pytest.mark.parametrize('name', RAW_PARAMETERS)
def test_unmerged_adapter_rejects_trainable_raw_state(bridge, name):
    layer = layer_fixture()
    getattr(layer, name).requires_grad_(True)
    with pytest.raises(ValueError, match='--merge_lora true'):
        bridge._set_historical_route_state(layer, {}, '', False)


@pytest.mark.parametrize('key', ['history_read_q_down_proj.weight', 'history_read_q_up_proj.lora_A.weight'])
def test_retired_adapter_import_rejected_before_copy(bridge, key):
    with pytest.raises(ValueError, match='Retired independent history-query'):
        bridge._set_historical_route_state(layer_fixture(), {key: object()}, '', True)


def test_full_export_retains_march_base_and_state(bridge):
    layer = layer_fixture('frozen')
    bridge._peft_format = False
    state = bridge._set_historical_route_state(layer, {}, 'linear_attn.', False)
    expected = {'linear_attn.' + name + '.weight' for name in PROJECTIONS}
    expected.update('linear_attn.' + name for name in RAW_PARAMETERS)
    assert state.keys() == expected
    for name in PROJECTIONS:
        torch.testing.assert_close(state['linear_attn.' + name + '.weight'], getattr(layer, name).weight.cpu())


@pytest.mark.parametrize('variant', ['safin1_4b', 'safin1_35b_a3b'])
def test_exported_march_adapter_loads_in_hf_peft(bridge, tmp_path, variant):
    from peft import PeftModel, get_peft_model_state_dict
    from march.safin1 import Safin1ForConditionalGeneration, Safin1MoeForConditionalGeneration
    path = Path(__file__).resolve().parents[2] / 'scripts/march/smoke_hf.py'
    spec = importlib.util.spec_from_file_location('adapter_hf_fixture', path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    cls = Safin1ForConditionalGeneration if variant == 'safin1_4b' else Safin1MoeForConditionalGeneration
    with fixture.reference_kernels():
        hf_model = cls(fixture.tiny_config(variant, chunk_size=4)).eval()
        base = copy.deepcopy(hf_model)
        routes = []
        for name, module in hf_model.named_modules():
            if name.endswith('.linear_attn') and hasattr(module, 'anchor_q_proj'):
                layer = torch.nn.Module()
                for field in PROJECTIONS:
                    projection = getattr(module, field, None)
                    if projection is not None:
                        setattr(layer, field, adapter_projection(projection))
                routes.append((name + '.', layer))
        assert routes
        rows = export_file(bridge, routes, tmp_path)
        loaded = PeftModel.from_pretrained(base, tmp_path, is_trainable=False, local_files_only=True).eval()
        actual = get_peft_model_state_dict(loaded)
        assert actual.keys() == rows.keys()
        for key in rows:
            torch.testing.assert_close(actual[key].cpu(), rows[key], rtol=0, atol=0)
        # Check real HF projection output against the exported MCore adapter
        # tensors, then model logits against PEFT's merge-and-unload path.
        for prefix, layer in routes:
            for field in PROJECTIONS:
                if not hasattr(layer, field):
                    continue
                projection = getattr(layer, field)
                x = torch.ones(2, projection.base_layer.in_features)
                expected = F.linear(x, projection.weight) + F.linear(
                    F.linear(x, projection.lora_A['review'].weight), projection.lora_B['review'].weight)
                hf_proj = loaded.get_submodule('base_model.model.' + prefix.rstrip('.') + '.' + field)
                torch.testing.assert_close(hf_proj(x), expected, atol=1e-6, rtol=1e-5)
        ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10, 11]])
        with torch.no_grad():
            logits = loaded(input_ids=ids, use_cache=False).logits
            assert torch.isfinite(logits).all()
            merged = loaded.merge_and_unload().eval()
            after = merged(input_ids=ids, use_cache=False).logits
            torch.testing.assert_close(after, logits, atol=2e-5, rtol=2e-4)

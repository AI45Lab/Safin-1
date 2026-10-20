"""Strict full-checkpoint preflight, base initialization and live tensor import."""
import json
import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(os.getenv("MARCH_TEST_MEGATRON") != "1", reason="training environment required")


@pytest.fixture
def bridge(monkeypatch):
    from mcore_bridge.model.mm_gpts.qwen3_5_gdn import Qwen3_5Bridge
    for name in ("USE_HIST_LINEAR_ATTN", "USE_MCORE_GDN", "SWIFT_USE_MCORE_GDN"):
        monkeypatch.delenv(name, raising=False)
    value = object.__new__(Qwen3_5Bridge)
    value.config = SimpleNamespace(
        hf_model_type="safin1_4b", anchor_chunk_size=512, anchor_token_num=1,
        anchor_embedding_type="learned", linear_attention_freq=[1, 0, 1], num_layers=3,
        hsd_global_read_layers="1,3", hsd_null_read_gate=True, hsd_learnable_history_scale=True,
        hsd_history_out_rmsnorm=True, hsd_persistent_memory_enabled=False,
        hsd_persistent_memory_route_key_enabled=False, linear_num_key_heads=1,
        linear_key_head_dim=2, linear_value_head_dim=2, linear_num_value_heads=1, hidden_size=3)
    return value


def required_keys(bridge):
    return set().union(*bridge._expected_hsd_export_keys(hsd_enabled=True).values())


def write_checkpoint(path, keys, source_type="safin1_4b", memory=False, adapter=False):
    from safetensors.torch import save_file
    if not adapter:
        (path/"config.json").write_text(json.dumps({"model_type": source_type, "text_config": {
            "hsd_persistent_memory_enabled": memory}}))
    save_file({key: torch.ones(1) for key in keys}, path/("adapter_model.safetensors" if adapter else "model.safetensors"))


def invoke_import(bridge, path, calls, adapter=False):
    # The real full-import API must preflight before its conversion/copy loop.
    def convert(*args):
        calls.append("convert")
        return iter(())
    bridge._convert = convert
    bridge.load_weights([torch.nn.Linear(1, 1)], str(path), peft_format=adapter)


@pytest.mark.parametrize("suffix", ["anchor_embedding.weight", "anchor_q_proj.weight", "anchor_k_proj.weight",
    "router_q_proj.weight", "null_gate_proj.weight", "hsd_null_read_gate_bias",
    "hsd_history_gate_logit", "hsd_history_out_rmsnorm_weight"])
def test_missing_active_checkpoint_weight_fails_before_any_conversion(bridge, tmp_path, suffix):
    keys = required_keys(bridge)
    missing = next(k for k in keys if k.endswith(suffix))
    keys.remove(missing)
    write_checkpoint(tmp_path, keys)
    calls = []
    with pytest.raises(KeyError, match="Incomplete Safin-1"):
        invoke_import(bridge, tmp_path, calls)
    assert not calls


def test_complete_checkpoint_reaches_conversion(bridge, tmp_path):
    write_checkpoint(tmp_path, required_keys(bridge))
    calls = []
    invoke_import(bridge, tmp_path, calls)
    assert calls == ["convert"]
    assert not bridge._hsd_import_is_base_qwen


def test_base_qwen_checkpoint_retains_explicit_initialization_path(bridge, tmp_path):
    write_checkpoint(tmp_path, {"model.language_model.layers.0.linear_attn.in_proj_qkv.weight"}, "qwen3_5")
    calls = []
    invoke_import(bridge, tmp_path, calls)
    assert calls == ["convert"] and bridge._hsd_import_is_base_qwen


def test_partially_march_weights_never_count_as_base_qwen(bridge, tmp_path):
    keys = required_keys(bridge)
    keys.remove(next(k for k in keys if k.endswith("router_q_proj.weight")))
    write_checkpoint(tmp_path, keys, "qwen3_5")
    calls = []
    with pytest.raises(KeyError, match="Incomplete Safin-1"):
        invoke_import(bridge, tmp_path, calls)
    assert not calls


@pytest.mark.parametrize("source_memory", [False, True])
def test_only_new_persistent_state_may_use_documented_zero_initialization(bridge, tmp_path, source_memory):
    bridge.config.hsd_persistent_memory_enabled = True
    keys = {k for k in required_keys(bridge) if not k.endswith(".persistent_memory_state")}
    write_checkpoint(tmp_path, keys, memory=source_memory)
    calls = []
    if source_memory:
        with pytest.raises(KeyError, match="persistent_memory_state"):
            invoke_import(bridge, tmp_path, calls)
        assert not calls
    else:
        invoke_import(bridge, tmp_path, calls)
        assert calls == ["convert"] and bridge._hsd_allow_new_persistent_state


def test_dedicated_memory_route_key_still_requires_artifact(bridge, tmp_path):
    bridge.config.hsd_persistent_memory_enabled = True
    bridge.config.hsd_persistent_memory_route_key_enabled = True
    keys = {k for k in required_keys(bridge) if not k.endswith((".persistent_memory_state", ".persistent_memory_route_key"))}
    write_checkpoint(tmp_path, keys)
    calls = []
    with pytest.raises(KeyError, match="persistent_memory_route_key"):
        invoke_import(bridge, tmp_path, calls)
    assert not calls


def test_lora_adapter_remains_partial_and_needs_no_full_config(bridge, tmp_path):
    write_checkpoint(tmp_path, {"base_model.model.layers.0.q_proj.lora_A.weight"}, adapter=True)
    calls = []
    invoke_import(bridge, tmp_path, calls, adapter=True)
    assert calls == ["convert"]


class Weight:
    def __init__(self, tensor):
        self.tensor = tensor

    def load(self):
        return self.tensor


def layer_fixture(bridge):
    projection = lambda: SimpleNamespace(weight=torch.full((2, 3), -1.0))
    layer = SimpleNamespace(anchor_q_proj=projection(), anchor_k_proj=projection(), router_q_proj=projection(),
        null_gate_proj=projection(), hsd_null_read_gate_bias=torch.tensor(-1.),
        hsd_history_gate_logit=torch.tensor(-1.), hsd_history_out_rmsnorm_weight=torch.full((2,), -1.))
    keys = {"anchor_q_proj.weight": torch.full((2, 3), 99.), "anchor_k_proj.weight": torch.full((2, 3), 88.),
        "router_q_proj.weight": torch.full((2, 3), 77.), "null_gate_proj.weight": torch.full((2, 3), 66.),
        "hsd_null_read_gate_bias": torch.tensor(55.), "hsd_history_gate_logit": torch.tensor(44.),
        "hsd_history_out_rmsnorm_weight": torch.full((2,), 33.)}
    calls = []
    def assign(target, name, state, source_key, loading):
        calls.append(name)
        obj = target
        for part in name.split("."):
            obj = getattr(obj, part)
        obj.copy_(state[source_key].load())
    bridge._set_state_dict = assign
    bridge._set_weight = lambda target, value, name: (calls.append(name), target.copy_(value))
    return layer, {key: Weight(value) for key, value in keys.items()}, calls


@pytest.mark.parametrize("missing", ["router_q_proj.weight", "null_gate_proj.weight", "hsd_null_read_gate_bias"])
def test_layer_import_cannot_reinitialize_or_skip_missing_parameters(bridge, missing):
    layer, state, calls = layer_fixture(bridge)
    state.pop(missing)
    state["in_proj_qkv.weight"] = Weight(torch.arange(18.).reshape(6, 3))
    with pytest.raises(KeyError, match="Incomplete Safin-1"):
        bridge._set_historical_route_state(layer, state, "", True)
    assert not calls
    assert torch.all(layer.anchor_q_proj.weight == -1)


def test_layer_import_restores_live_weights(bridge):
    layer, state, calls = layer_fixture(bridge)
    with torch.no_grad():
        bridge._set_historical_route_state(layer, state, "", True)
    assert len(calls) == len(state)
    assert torch.all(layer.anchor_q_proj.weight == 99)
    assert torch.all(layer.null_gate_proj.weight == 66)
    assert layer.hsd_null_read_gate_bias.item() == 55


def test_base_qwen_layer_still_initializes_from_qkv(bridge):
    layer, _, calls = layer_fixture(bridge)
    bridge._hsd_import_is_base_qwen = True
    state = {"in_proj_qkv.weight": Weight(torch.arange(18.).reshape(6, 3))}
    with torch.no_grad():
        bridge._set_historical_route_state(layer, state, "", True)
    assert calls == ["anchor_q_proj.weight", "anchor_k_proj.weight", "router_q_proj.weight"]
    torch.testing.assert_close(layer.anchor_q_proj.weight, state["in_proj_qkv.weight"].load()[:2])


@pytest.mark.parametrize("source", ["safin1_4b", "safin1_35b_a3b", "qwen3_5"])
def test_real_loader_selects_march_from_model_contract_without_legacy_flag(bridge, monkeypatch, source):
    from mcore_bridge.model.mm_gpts.qwen3_5_gdn import Qwen3_5Loader, MarchGatedDeltaNet, GatedDeltaNet
    from megatron.core.models.gpt import experimental_attention_variant_module_specs as specs
    fake = type("LinearAttention", (), {})
    attention = SimpleNamespace(module=fake)
    result = SimpleNamespace(layer_specs=[SimpleNamespace(submodules=SimpleNamespace(self_attention=attention))])
    monkeypatch.setattr(specs, "get_transformer_block_with_experimental_attention_variant_spec", lambda *a: result)
    loader = object.__new__(Qwen3_5Loader)
    loader.config = SimpleNamespace(hf_model_type=source, linear_decoupled_in_proj=False)
    loader.get_transformer_layer_spec()
    assert attention.module is (GatedDeltaNet if source == "qwen3_5" else MarchGatedDeltaNet)


@pytest.mark.parametrize("flag", ["USE_HIST_LINEAR_ATTN", "USE_MCORE_GDN", "SWIFT_USE_MCORE_GDN"])
def test_real_loader_refuses_architecture_downgrade_before_build(bridge, monkeypatch, flag):
    from mcore_bridge.model.mm_gpts.qwen3_5_gdn import Qwen3_5Loader
    from megatron.core.models.gpt import experimental_attention_variant_module_specs as specs
    allocations = []
    monkeypatch.setattr(specs, "get_transformer_block_with_experimental_attention_variant_spec", lambda *a: allocations.append(1))
    monkeypatch.setenv(flag, "0")
    loader = object.__new__(Qwen3_5Loader)
    loader.config = bridge.config
    with pytest.raises(ValueError, match=flag):
        loader.get_transformer_layer_spec()
    assert not allocations


def test_model_config_and_saved_runtime_cannot_drift(bridge, monkeypatch):
    from mcore_bridge import ModelConfig
    from swift.megatron.init import _collect_hsd_runtime_config
    monkeypatch.setenv("HSD_READ_TOPK", "8")
    with pytest.raises(ValueError, match="hsd_read_topk"):
        ModelConfig(hsd_read_topk=4)
    with pytest.raises(ValueError, match="hsd_read_topk"):
        _collect_hsd_runtime_config(SimpleNamespace(hsd_read_topk=4))
    assert _collect_hsd_runtime_config(SimpleNamespace(hsd_read_topk=8))["hsd_read_topk"] == 8


@pytest.mark.parametrize("suffix", ["null_gate_proj.weight", "hsd_null_read_gate_bias",
    "hsd_history_gate_logit", "hsd_history_out_rmsnorm_weight"])
def test_gate_only_partial_checkpoint_cannot_initialize_as_base(bridge, tmp_path, suffix):
    write_checkpoint(tmp_path, {"model.language_model.layers.0.linear_attn." + suffix}, "qwen3_5")
    calls = []
    with pytest.raises(KeyError, match="Incomplete Safin-1"):
        invoke_import(bridge, tmp_path, calls)
    assert not calls


def test_base_dedicated_memory_requires_artifact_before_conversion(bridge, tmp_path):
    bridge.config.hsd_persistent_memory_enabled = True
    bridge.config.hsd_persistent_memory_route_key_enabled = True
    write_checkpoint(tmp_path, {"model.language_model.layers.0.linear_attn.in_proj_qkv.weight"}, "qwen3_5")
    calls = []
    with pytest.raises(KeyError, match="route-key artifact"):
        invoke_import(bridge, tmp_path, calls)
    assert not calls


@pytest.mark.parametrize("route_adapter", [False, True])
def test_real_layer_adapter_import_preserves_base_and_persistent_state(bridge, route_adapter):
    from mcore_bridge.tuners import LoraParallelLinear
    layer, _, calls = layer_fixture(bridge)
    del bridge._set_state_dict  # Use the real generic adapter conversion.
    layer.persistent_memory_state = torch.full((1, 2, 2), 17.)
    bridge._peft_format = True
    bridge._adapter_name = "default"
    state = {}
    if route_adapter:
        projection = object.__new__(LoraParallelLinear)
        torch.nn.Module.__init__(projection)
        projection.base_layer = torch.nn.Linear(3, 2, bias=False)
        projection.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(3, 1, bias=False)})
        projection.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(1, 2, bias=False)})
        layer.anchor_q_proj = projection
        state = {"anchor_q_proj.lora_A.weight": Weight(torch.full((1, 3), 23.)),
                 "anchor_q_proj.lora_B.weight": Weight(torch.full((2, 1), 29.))}
    base = layer.anchor_q_proj.weight.detach().clone()
    bridge._set_weight = lambda target, value, name, *a, **kw: (calls.append(name), target.copy_(value))
    with torch.no_grad():
        bridge._set_historical_route_state(layer, state, "", True)
    torch.testing.assert_close(layer.anchor_q_proj.weight, base, rtol=0, atol=0)
    assert torch.all(layer.persistent_memory_state == 17)
    assert layer.hsd_null_read_gate_bias.item() == -1
    if route_adapter:
        assert len(calls) == 2
        assert torch.all(layer.anchor_q_proj.lora_A["default"].weight == 23)
        assert torch.all(layer.anchor_q_proj.lora_B["default"].weight == 29)
    else:
        assert not calls

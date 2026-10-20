"""Regression checks for public profiles and the real Megatron export hook."""
import importlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(os.getenv("MARCH_TEST_MEGATRON") != "1",
                               reason="set MARCH_TEST_MEGATRON=1 in the training environment")


@pytest.mark.parametrize("variant,base", [("safin1_4b", "qwen3_5"), ("safin1_35b_a3b", "qwen3_5_moe")])
def test_real_profile_merge_and_export(variant, base):
    from mcore_bridge.config.parser import merge_hsd_runtime_config
    from swift.megatron.init import _finalize_hsd_hf_config_file
    from transformers import AutoConfig
    profile_path = ROOT / f"configs/march/{variant}.yaml"
    profile = json.loads(profile_path.read_text())
    for source_type in (base, variant):
        merged = merge_hsd_runtime_config({"hf_model_type": source_type,
                                          "llm_model_type": source_type + "_text"}, str(profile_path))
        assert "model_type" not in merged
        for key, value in profile.items():
            if key != "model_type":
                assert merged[key] == value
    with tempfile.TemporaryDirectory() as temp:
        config = Path(temp, "config.json")
        config.write_text(json.dumps({"model_type": base, "text_config": {"model_type": base + "_text"}}))
        _finalize_hsd_hf_config_file(temp, SimpleNamespace(hsd_runtime_config=profile), SimpleNamespace(**merged))
        loaded = AutoConfig.from_pretrained(temp, trust_remote_code=True, local_files_only=True)
        assert loaded.model_type == variant
        for key, value in profile.items():
            if key != "model_type":
                assert getattr(loaded.text_config, key) == value


def test_bad_profiles_fail_before_model_construction():
    from mcore_bridge.config.parser import merge_hsd_runtime_config
    dense = {"hf_model_type": "safin1_4b", "llm_model_type": "safin1_4b_text"}
    with pytest.raises(ValueError, match="model_type"):
        merge_hsd_runtime_config(dense.copy(), {"model_type": "safin1_35b_a3b"})
    with pytest.raises(ValueError, match="Unknown"):
        merge_hsd_runtime_config(dense.copy(), {"hsd_typo": 123})
    # Historical runtime-only JSON remains supported.
    assert merge_hsd_runtime_config(dense.copy(), {"hsd_read_topk": 4})["hsd_read_topk"] == 4


def test_mandatory_training_modules_import_from_this_fork():
    for name in ("mcore_bridge.model", "swift.megatron.init", "swift.megatron.utils.megatron_lm_utils",
                 "swift.megatron.utils.hsd_checkpoint_utils", "swift.megatron.utils.persistent_memory_artifact"):
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(ROOT), module.__file__


@pytest.mark.parametrize("to_mcore", [True, False])
def test_safin_moe_shared_expert_checkpoint_layout(to_mcore):
    """Renaming the architecture must retain the singular shared_expert key."""
    from mcore_bridge.model.mm_gpts.qwen3_5_gdn import Qwen3_5Bridge
    bridge = object.__new__(Qwen3_5Bridge)
    bridge.config = SimpleNamespace(add_bias_linear=False, moe_router_enable_expert_bias=False,
                                    moe_shared_expert_intermediate_size=8, moe_shared_expert_gate=True)
    bridge.model_type = "safin1_35b_a3b"
    bridge.llm_model_type = "safin1_35b_a3b_text"
    bridge.ep_size = 0  # Only exercise shared-expert dispatch; no distributed groups needed.
    bridge._set_state_dict = lambda *args: None
    key = "shared_expert.gate_proj.weight"
    marker = object()

    def convert_shared_expert(module, state, prefix, layer, loading):
        if loading:
            assert state[prefix + "gate_proj.weight"] is marker
            return {}
        return {prefix + "gate_proj.weight": marker}

    bridge._set_mlp_state = convert_shared_expert
    result = bridge._set_moe_state(None, {"mlp." + key: marker}, "mlp.", 0, to_mcore)
    assert result == ({} if to_mcore else {"mlp." + key: marker})


@pytest.mark.parametrize("runtime", [
    {"hsd_local_state_mode": "chunk"}, {"hsd_history_read_q_mode": "hidden_residual_lora"},
    {"hsd_anchor_mlp": True},
])
def test_retired_runtime_rejected_during_bridge_merge(runtime):
    from mcore_bridge.config.parser import merge_hsd_runtime_config
    config = {"hf_model_type": "safin1_4b"}
    with pytest.raises(ValueError, match="[Rr]etired"):
        merge_hsd_runtime_config(config, runtime)
    assert config == {"hf_model_type": "safin1_4b"}, "failed merge mutated input"


def test_bridge_config_rejects_retired_mode_before_initialization():
    from mcore_bridge import ModelConfig
    with pytest.raises(ValueError, match="retired variant"):
        ModelConfig(hsd_local_state_mode="chunk")


def test_retired_query_rank_does_not_block_bridge_config(monkeypatch):
    from mcore_bridge import ModelConfig
    import importlib.util
    path = ROOT / "scripts/march/smoke_hf.py"
    spec = importlib.util.spec_from_file_location("retired_rank_bridge_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    from mcore_bridge.model.mm_gpts.qwen3_5_gdn import Qwen3_5Bridge
    # Exercise real config validation without requiring distributed GPU groups.
    monkeypatch.setattr(Qwen3_5Bridge, "__init__", lambda self, config: None)
    config = ModelConfig(num_layers=2, hidden_size=32, num_attention_heads=4,
                         ffn_hidden_size=64, hf_model_type="safin1_4b",
                         hf_config=fixture.tiny_config("safin1_4b"), hsd_history_read_q_lora_rank=-7)
    assert config.hsd_history_read_q_lora_rank == -7
    assert config.hsd_history_read_q_mode == "shared_local"


@pytest.mark.parametrize("runtime", [
    {"hsd_history_mix_mode": "convex"}, {"hsd_history_mix_mode": "joint_softmax_logm"},
    {"hsd_history_mix_mode": "gated_additive"}, {"hsd_route_weight_momentum_gamma": 0.25},
])
def test_unimplemented_training_config_rejected_before_setup_and_merge(runtime):
    from mcore_bridge import ModelConfig
    from mcore_bridge.config.parser import merge_hsd_runtime_config

    field = next(iter(runtime))
    with pytest.raises(ValueError, match=field + "=.*not implemented"):
        ModelConfig(**runtime)
    config = {"hf_model_type": "safin1_4b", "hsd_null_read_gate": True, "hsd_read_topk": 0}
    original = config.copy()
    with pytest.raises(ValueError, match=field):
        merge_hsd_runtime_config(config, runtime)
    assert config == original
    config.update(runtime)
    original = config.copy()
    with pytest.raises(ValueError, match=field):
        merge_hsd_runtime_config(config)
    assert config == original


@pytest.mark.parametrize("memory", [False, True])
def test_hsd_export_requires_all_active_parameters_and_no_retired_adapters(memory):
    from mcore_bridge.model.mm_gpts.qwen3_5_gdn import Qwen3_5Bridge

    bridge = object.__new__(Qwen3_5Bridge)
    bridge.config = SimpleNamespace(
        anchor_chunk_size=512, anchor_token_num=1, linear_attention_freq=[1, 0, 1], num_layers=3,
        hsd_global_read_layers="1,3", hsd_null_read_gate=True, hsd_learnable_history_scale=True,
        hsd_history_out_rmsnorm=True, hsd_persistent_memory_enabled=memory,
        hsd_persistent_memory_route_key_enabled=memory,
    )
    keys = set().union(*bridge._expected_hsd_export_keys(hsd_enabled=True).values())
    expected = {bridge._hf_anchor_embedding_key()}
    suffixes = ["anchor_q_proj.weight", "anchor_k_proj.weight", "router_q_proj.weight",
                "null_gate_proj.weight", "hsd_null_read_gate_bias", "hsd_history_gate_logit",
                "hsd_history_out_rmsnorm_weight"]
    if memory:
        suffixes += ["persistent_memory_state", "persistent_memory_route_key"]
    for index in (0, 2):
        expected.update(f"{bridge.hf_layers_prefix}.{index}.linear_attn.{suffix}" for suffix in suffixes)
    assert keys == expected
    assert bridge._expected_hsd_export_keys(peft_format=True, hsd_enabled=True) == {}


@pytest.mark.parametrize("adapter_keys", [
    ["history_read_q_down_proj.weight"], ["history_read_q_up_proj.weight"],
    ["history_read_q_down_proj.weight", "history_read_q_up_proj.weight"],
])
def test_retired_query_weights_rejected_before_any_weight_copy(adapter_keys):
    from mcore_bridge.model.mm_gpts.qwen3_5_gdn import Qwen3_5Bridge

    bridge = object.__new__(Qwen3_5Bridge)
    copies = []
    bridge._set_state_dict = lambda *args: copies.append(args)
    prefix = "model.language_model.layers.0.linear_attn."
    weights = {prefix + name: object() for name in ["anchor_q_proj.weight", *adapter_keys]}
    with pytest.raises(ValueError, match="Retired independent history-query adapter weights"):
        bridge._set_historical_route_state(SimpleNamespace(anchor_q_proj=object()), weights, prefix, True)
    assert not copies


@pytest.mark.parametrize("masked", [False, True])
def test_shared_hf_gdn_adapter_resolves_attention_mask(monkeypatch, masked):
    import torch
    from mcore_bridge.model.gpts import qwen3_next

    layer = object.__new__(qwen3_next.Qwen3NextGatedDeltaNet)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(sequence_parallel=False, tensor_model_parallel_size=1)
    monkeypatch.setattr(qwen3_next, "is_torch_npu_available", lambda: False)
    observed = []

    def capture_forward(self, hidden_states, attention_mask):
        observed.append(attention_mask)
        return hidden_states.clone()

    monkeypatch.setattr(qwen3_next._Qwen3NextGatedDeltaNet, "forward", capture_forward)
    hidden = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
    kwargs = {}
    if masked:
        mask = torch.zeros(2, 1, 4, 4, dtype=torch.bool)
        mask[0, :, :, 3] = True
        kwargs["attention_mask"] = mask
    output, bias = layer.forward(hidden, **kwargs)
    torch.testing.assert_close(output, hidden, rtol=0, atol=0)
    assert bias is None
    assert len(observed) == 1
    if masked:
        assert observed[0].tolist() == [[True, True, True, False], [True, True, True, True]]
    else:
        assert observed[0] is None

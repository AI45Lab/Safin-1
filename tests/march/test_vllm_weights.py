"""Execute Safin-1 loading/audit mixins with CPU tensors and native-loader spies.

CUDA/engine imports are isolated at the dependency boundary. The real mapper,
loading, TP slicing and audit methods execute; native Qwen loading is a spy.
"""

import ast
from collections.abc import Mapping
from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "third_party/vllm/vllm/model_executor/models"


@pytest.fixture
def helpers(monkeypatch):
    name = "_march_loading_contract"
    parent = ModuleType(name)
    parent.__path__ = [str(MODELS)]
    monkeypatch.setitem(sys.modules, name, parent)
    package = ModuleType(name + ".march")
    package.__path__ = [str(MODELS / "march")]
    monkeypatch.setitem(sys.modules, package.__name__, package)

    def dependency(module_name, **attributes):
        module = ModuleType(module_name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, module_name, module)
        return module

    def copy_weight(param, value):
        with torch.no_grad():
            param.copy_(value)

    class NativeLoader:
        def __init__(self, model, skip_prefixes):
            self.model, self.skip_prefixes = model, skip_prefixes

        def load_weights(self, rows):
            loaded = set()
            for key, value in rows:
                if any(key.startswith(prefix) for prefix in self.skip_prefixes):
                    continue
                self.model.native_loads.append(key)
                copy_weight(dict(self.model.named_parameters())[key], value)
                loaded.add(key)
            return loaded

    utils = dependency(name + ".utils", AutoWeightsLoader=NativeLoader,
                       is_pp_missing_parameter=lambda key, model: False,
                       dataclass=dataclass, field=field, Mapping=Mapping,
                       torch=torch, re=re)
    mapper = next(node for node in ast.parse((MODELS / "utils.py").read_text()).body
                  if isinstance(node, ast.ClassDef) and node.name == "WeightsMapper")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), mapper], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(MODELS / "utils.py"), "exec"), utils.__dict__)
    dependency("vllm.logger", init_logger=lambda _: NS(warning_once=lambda *a: None))
    dependency("vllm.model_executor.model_loader.weight_utils", default_weight_loader=copy_weight)
    dependency("vllm.transformers_utils.configs.safin1", Safin1TextConfig=NS)
    dependency("vllm.transformers_utils.configs.safin1_moe", Safin1MoeTextConfig=NS)
    distributed = dependency("vllm.distributed", get_tensor_model_parallel_rank=lambda: 0,
                             get_tensor_model_parallel_world_size=lambda: 1)

    def load(filename):
        fullname = name + ".march." + filename
        spec = importlib.util.spec_from_file_location(fullname, MODELS / "march" / (filename + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, fullname, module)
        spec.loader.exec_module(module)
        return module

    audit, weights = load("persistent_audit"), load("weights")

    class Attention(nn.Module):
        def __init__(self, persistent):
            super().__init__()
            self.anchor_q_proj = nn.Linear(2, 2, bias=False)
            if persistent:
                self.persistent_memory_state = nn.Parameter(torch.zeros(2, 2, 3))
                self.persistent_memory_route_key = nn.Parameter(torch.zeros(2, 2))

        def get_persistent_memory_metrics(self, **kwargs):
            return {"memory_probability": 0.25}

    dependency(name + ".safin1", HistoricalGatedDeltaNetAttention=Attention)

    class Model(weights.CausalLMWeightLoadingMixin, audit.PersistentMemoryAuditMixin, nn.Module):
        def __init__(self, mode="enabled", persistent=True, tp=1, family="safin1_4b_text"):
            super().__init__()
            self.config = NS(model_type=family, hsd_persistent_memory_enabled=persistent,
                             hsd_persistent_memory_route_key_enabled=persistent)
            self.vllm_config = NS(parallel_config=NS(tensor_parallel_size=tp), additional_config={
                "hsd_persistent_memory_expected_tensor_count": 1,
                "hsd_persistent_memory_expected_parameter_count": 12 * tp,
                "hsd_persistent_memory_route_key_expected_tensor_count": 1,
                "hsd_persistent_memory_route_key_expected_parameter_count": 4 * tp,
            })
            self.hsd_persistent_memory_mode = mode
            self.hf_to_vllm_mapper = weights._HISTORICAL_HF_TO_VLLM_MAPPER
            self.model = nn.Module()
            self.model.anchor_embedding = nn.Embedding(1, 2)
            layer = nn.Module()
            layer.linear_attn = Attention(persistent)
            self.model.layers = nn.ModuleList([layer])
            self.lm_head = nn.Linear(2, 2, bias=False)
            self.native_loads, self.invalidations = [], []

        def invalidate_hsd_prefix_cache(self):
            self.invalidations.append("reset")

    def checkpoint(model, tp=1):
        rows = []
        for index, (key, parameter) in enumerate(model.named_parameters()):
            shape = list(parameter.shape)
            if key.endswith((".persistent_memory_state", ".persistent_memory_route_key")):
                shape[0] *= tp
            value = torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape) + index + 1
            rows.append((key.replace("model.", "model.language_model.", 1), value))
        return rows

    return NS(weights=weights, audit=audit, Model=Model, checkpoint=checkpoint,
              distributed=distributed, copy_weight=copy_weight)


@pytest.mark.parametrize("family", ["safin1_4b_text", "safin1_35b_a3b_text"])
@pytest.mark.parametrize("mode", ["enabled", "zero"])
@pytest.mark.parametrize("rank", [0, 1])
def test_checkpoint_tp_slices_audit_and_parameter_ownership(helpers, monkeypatch, family, mode, rank):
    monkeypatch.setattr(helpers.distributed, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(helpers.distributed, "get_tensor_model_parallel_rank", lambda: rank)
    model = helpers.Model(mode, tp=2, family=family)
    before = {key: parameter.data_ptr() for key, parameter in model.named_parameters()}
    rows = helpers.checkpoint(model, tp=2)
    loaded = model.load_weights(iter(rows))
    assert loaded == set(before)
    assert before == {key: parameter.data_ptr() for key, parameter in model.named_parameters()}
    assert model.native_loads == ["lm_head.weight"]
    for key, value in model.hf_to_vllm_mapper.apply(rows):
        if key.endswith((".persistent_memory_state", ".persistent_memory_route_key")):
            expected = value.chunk(2, dim=0)[rank]
        else:
            expected = value
        if mode == "zero" and key.endswith(".persistent_memory_state"):
            expected = torch.zeros_like(expected)
        torch.testing.assert_close(dict(model.named_parameters())[key], expected, atol=0, rtol=0)
    audit = model.get_persistent_memory_audit()
    assert audit["finite"] and audit["tensor_count"] == 1
    assert audit["local_numel"] == 12 and audit["global_numel"] == 24
    assert audit["route_key"]["local_numel"] == 4 and audit["route_key"]["global_numel"] == 8
    assert audit["metrics"]["memory_probability"] == 0.25
    assert audit["local_checksum"] == model._hsd_persistent_memory_runtime_local_checksum
    assert (audit["parameter_rms"] == 0) == (mode == "zero")


@pytest.mark.parametrize("suffix", [".persistent_memory_state", ".persistent_memory_route_key"])
@pytest.mark.parametrize("failure", ["duplicate", "nonfinite", "dtype", "shape", "missing"])
def test_bad_persistent_checkpoints_fail_closed(helpers, suffix, failure):
    model = helpers.Model()
    rows = helpers.checkpoint(model)
    index = next(i for i, (key, _) in enumerate(rows) if key.endswith(suffix))
    key, value = rows[index]
    if failure == "duplicate":
        rows.append((key, value))
        message = "Duplicate persistent"
    elif failure == "nonfinite":
        value.flatten()[0] = float("nan")
        message = "non-finite"
    elif failure == "dtype":
        rows[index] = key, value.to(torch.bfloat16)
        message = "FP32"
    elif failure == "shape":
        rows[index] = key, value[:1]
        message = "shape mismatch"
    else:
        rows.pop(index)
        message = "missing required"
    with pytest.raises(RuntimeError, match=message):
        model.load_weights(rows)
    assert not getattr(model, "_historical_hsd_weights_validated", False)


@pytest.mark.parametrize("suffix", [".persistent_memory_state", ".persistent_memory_route_key"])
def test_worker_audit_detects_post_load_mutation(helpers, suffix):
    model = helpers.Model()
    model.load_weights(helpers.checkpoint(model))
    parameter = next(p for key, p in model.named_parameters() if key.endswith(suffix))
    with torch.no_grad():
        parameter.flatten()[0].add_(1)
    with pytest.raises(RuntimeError, match="mutated after"):
        model.get_persistent_memory_audit()


def test_first_load_strict_then_partial_update_invalidates_before_iterator(helpers):
    model = helpers.Model(persistent=False)
    rows = helpers.checkpoint(model)
    with pytest.raises(RuntimeError, match="missing required"):
        model.load_weights([row for row in rows if "anchor_embedding" not in row[0]])
    model.load_weights(rows)
    assert model.get_persistent_memory_audit()["checksum"] is None

    def update():
        assert model.invalidations == ["reset"]
        yield "lm_head.weight", torch.full_like(model.lm_head.weight, 7)

    assert model.load_weights(update()) == {"lm_head.weight"}
    assert torch.all(model.lm_head.weight == 7)


def test_inner_model_super_delegation_keeps_native_and_hsd_ownership(helpers):
    class Native(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor_embedding = nn.Embedding(1, 2)
            self.norm = nn.LayerNorm(2)

        def load_weights(self, rows):
            rows = list(rows)
            assert [key for key, _ in rows] == ["norm.weight"]
            helpers.copy_weight(self.norm.weight, rows[0][1])
            return {"norm.weight"}

    class Inner(helpers.weights.ModelWeightLoadingMixin, Native):
        pass

    model = Inner()
    loaded = model.load_weights([("anchor_embedding.weight", torch.full((1, 2), 3.0)),
                                 ("norm.weight", torch.full((2,), 5.0))])
    assert loaded == {"anchor_embedding.weight", "norm.weight"}
    assert torch.all(model.anchor_embedding.weight == 3)
    assert torch.all(model.norm.weight == 5)


def test_conv_and_null_gate_tp_slices_preserve_qkv_order(helpers, monkeypatch):
    monkeypatch.setattr(helpers.distributed, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(helpers.distributed, "get_tensor_model_parallel_rank", lambda: 1)
    config = NS(linear_num_key_heads=2, linear_key_head_dim=2,
                linear_num_value_heads=2, linear_value_head_dim=3)
    full = torch.arange(28).reshape(14, 1, 2).float()
    actual = helpers.weights._maybe_shard_hsd_conv1d_weight(
        config, "model.layers.0.linear_attn.conv1d.weight", torch.empty(7, 1, 2), full)
    torch.testing.assert_close(actual, full[[2, 3, 6, 7, 11, 12, 13]], atol=0, rtol=0)
    bias = helpers.weights._maybe_shard_hsd_null_gate_bias(
        "model.layers.0.linear_attn.hsd_null_read_gate_bias", torch.empty(2), torch.arange(4))
    assert bias.tolist() == [2, 3]

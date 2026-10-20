"""Compact persistent-memory artifacts validate before modifying a live model."""
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from safetensors.torch import save_file

from march.safin1.persistent_memory import load_persistent_memory_artifact, save_persistent_memory_artifact


def memory_model(route_keys=False, value=0.0):
    model = nn.Module()
    model.config = SimpleNamespace(model_type="safin1_4b", text_config=SimpleNamespace(hsd_persistent_memory_enabled=True))
    for name in ("a", "b"):
        layer = nn.Module()
        layer.persistent_memory_state = nn.Parameter(torch.full((2, 2), value))
        if route_keys:
            layer.persistent_memory_route_key = nn.Parameter(torch.full((2,), value + 1))
        model.add_module(name, layer)
    return model


def assert_unchanged(model, before):
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def snapshot(model):
    return {name: value.clone() for name, value in model.state_dict().items()}


@pytest.mark.parametrize("route_keys", [False, True])
def test_artifact_v1_v2_roundtrip(route_keys, tmp_path):
    source = memory_model(route_keys, value=2.0)
    tensors, metadata_path = save_persistent_memory_artifact(source, tmp_path)
    metadata = json.loads(metadata_path.read_text())
    assert metadata["format"] == ("hsd_persistent_memory_v2" if route_keys else "hsd_persistent_memory_v1")
    assert len(metadata["tensor_sha256"]) == 64
    target = memory_model(route_keys)
    info = load_persistent_memory_artifact(target, tmp_path)
    assert info["missing"] == info["unexpected"] == []
    assert_unchanged(target, source.state_dict())
    # Old sidecars lack checksum/model-family fields; they remain readable.
    metadata.pop("tensor_sha256")
    metadata.pop("model_type")
    metadata_path.write_text(json.dumps(metadata))
    assert load_persistent_memory_artifact(target, tensors)["loaded"]


@pytest.mark.parametrize("strict", [True, False])
def test_later_shape_error_does_not_apply_earlier_tensor(strict, tmp_path):
    model = memory_model()
    before = snapshot(model)
    artifact = tmp_path / "state.safetensors"
    save_file({"a.persistent_memory_state": torch.ones(2, 2),
               "b.persistent_memory_state": torch.ones(3, 3)}, str(artifact))
    with pytest.raises(RuntimeError, match="shape mismatch"):
        load_persistent_memory_artifact(model, artifact, strict=strict)
    assert_unchanged(model, before)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_artifact_never_updates_model(bad, tmp_path):
    model = memory_model()
    before = snapshot(model)
    artifact = tmp_path / "state.safetensors"
    save_file({"a.persistent_memory_state": torch.ones(2, 2),
               "b.persistent_memory_state": torch.full((2, 2), bad)}, str(artifact))
    with pytest.raises(RuntimeError, match="finite"):
        load_persistent_memory_artifact(model, artifact)
    assert_unchanged(model, before)


def test_conversion_error_never_updates_model(tmp_path, monkeypatch):
    model = memory_model()
    before = snapshot(model)
    artifact = tmp_path / "state.safetensors"
    save_file({name: torch.ones_like(value) for name, value in before.items()}, str(artifact))
    original = torch.Tensor.to
    count = 0

    def convert(value, *args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("conversion failed")
        return original(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", convert)
    with pytest.raises(RuntimeError, match="conversion failed"):
        load_persistent_memory_artifact(model, artifact)
    assert_unchanged(model, before)


@pytest.mark.parametrize("field,value", [("format", "unknown"), ("tensor_sha256", "stale"),
                                         ("tensors", {}), ("tensor_count", 0),
                                         ("model_type", "safin1_35b_a3b")])
def test_incompatible_metadata_does_not_update_model(field, value, tmp_path):
    _, path = save_persistent_memory_artifact(memory_model(value=2.0), tmp_path)
    metadata = json.loads(path.read_text())
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    target = memory_model()
    before = snapshot(target)
    with pytest.raises(RuntimeError):
        load_persistent_memory_artifact(target, tmp_path)
    assert_unchanged(target, before)


def test_strict_keys_and_partial_loading(tmp_path):
    model = memory_model()
    before = snapshot(model)
    artifact = tmp_path / "state.safetensors"
    save_file({"a.persistent_memory_state": torch.ones(2, 2)}, str(artifact))
    with pytest.raises(RuntimeError, match="key mismatch"):
        load_persistent_memory_artifact(model, artifact)
    assert_unchanged(model, before)
    info = load_persistent_memory_artifact(model, artifact, strict=False)
    assert info["loaded"] == ["a.persistent_memory_state"]
    assert info["missing"] == ["b.persistent_memory_state"]
    torch.testing.assert_close(model.a.persistent_memory_state, torch.ones(2, 2))
    torch.testing.assert_close(model.b.persistent_memory_state, before["b.persistent_memory_state"])


def test_save_replaces_hardlinked_tensor_file(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    tensor, _ = save_persistent_memory_artifact(memory_model(value=2.0), source)
    original = tensor.read_bytes()
    target.mkdir()
    (target / tensor.name).hardlink_to(tensor)
    save_persistent_memory_artifact(memory_model(value=3.0), target)
    assert tensor.read_bytes() == original
    assert (target / tensor.name).read_bytes() != original


def test_incomplete_pair_from_different_saves_is_rejected(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    original, _ = save_persistent_memory_artifact(memory_model(value=2.0), first)
    replacement, _ = save_persistent_memory_artifact(memory_model(value=3.0), second)
    # Simulate a writer replacing the tensor file before publishing its sidecar.
    original.write_bytes(replacement.read_bytes())
    target = memory_model()
    before = snapshot(target)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        load_persistent_memory_artifact(target, first)
    assert_unchanged(target, before)

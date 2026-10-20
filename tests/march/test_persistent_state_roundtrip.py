"""Persistent memory remains trainable and survives a fresh remote-code reload."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("variant", ["safin1_4b", "safin1_35b_a3b"])
def test_persistent_state_training_save_reload(variant, tmp_path):
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM
    from march.safin1 import Safin1ForConditionalGeneration, Safin1MoeForConditionalGeneration
    from march.safin1.config_utils import normalize_safin1_config, copy_safin1_remote_code

    script = Path(__file__).resolve().parents[2] / "scripts/march/smoke_hf.py"
    spec = importlib.util.spec_from_file_location("march_memory_smoke", script)
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    torch.manual_seed(123)
    with smoke.reference_kernels(), smoke.remote_registry():
        config = smoke.tiny_config(variant, chunk_size=8)
        config.text_config.hsd_persistent_memory_enabled = True
        config.text_config.hsd_persistent_memory_route_key_enabled = True
        config.text_config.hsd_persistent_memory_route_key_mode = "dedicated"
        cls = Safin1ForConditionalGeneration if variant == "safin1_4b" else Safin1MoeForConditionalGeneration
        model = cls(config)
        memory = {name: parameter for name, parameter in model.named_parameters()
                  if name.endswith("persistent_memory_state")}
        assert memory
        assert any(name.endswith("persistent_memory_route_key") for name, _ in model.named_parameters())
        before = {name: parameter.detach().clone() for name, parameter in memory.items()}
        ids = torch.randint(3, 59, (1, 18))
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        assert torch.isfinite(loss)
        loss.backward()
        for parameter in memory.values():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
        optimizer.step()
        assert all(not torch.equal(before[name], parameter) for name, parameter in memory.items())
        model.eval()
        with torch.no_grad():
            expected = model(input_ids=ids, use_cache=False).logits
        model.save_pretrained(tmp_path, safe_serialization=True)
        config_path = tmp_path / "config.json"
        saved = normalize_safin1_config(json.loads(config_path.read_text()))
        config_path.write_text(json.dumps(saved, indent=2) + "\n")
        copy_safin1_remote_code(tmp_path, variant)
        loaded, info = AutoModelForCausalLM.from_pretrained(
            tmp_path, trust_remote_code=True, local_files_only=True,
            attn_implementation="eager", output_loading_info=True,
        )
        assert not any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys")), info
        assert type(loaded).__module__.startswith("transformers_modules.")
        assert loaded.config.text_config.hsd_persistent_memory_enabled
        assert loaded.config.text_config.hsd_persistent_memory_route_key_enabled
        for name, parameter in memory.items():
            torch.testing.assert_close(loaded.state_dict()[name], parameter, atol=0, rtol=0)
        loaded.eval()
        with torch.no_grad():
            actual = loaded(input_ids=ids, use_cache=False).logits
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            for prefix in (7, 8, 9, 16, 17):
                cached = loaded(input_ids=ids[:, :prefix], use_cache=True)
                prepared = loaded.prepare_inputs_for_generation(
                    ids[:, :prefix+1], past_key_values=cached.past_key_values, use_cache=True)
                step = loaded(**prepared)
                full = loaded(input_ids=ids[:, :prefix+1], use_cache=False)
                torch.testing.assert_close(step.logits[:, -1], full.logits[:, -1], atol=2e-5, rtol=2e-4)

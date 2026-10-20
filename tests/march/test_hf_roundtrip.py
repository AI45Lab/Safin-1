"""Actual optimizer step, safetensors export, remote reload and cached decoding."""
import importlib.util
from pathlib import Path
import pytest


@pytest.mark.parametrize("variant", ["safin1_4b", "safin1_35b_a3b"])
def test_hf_training_save_reload_and_anchor_boundary(variant, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    path = Path(__file__).resolve().parents[2] / "scripts/march/smoke_hf.py"
    spec = importlib.util.spec_from_file_location("march_smoke_hf", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.run_variant(variant, tmp_path / variant)
    assert report["reload_max_abs_error"] == 0
    assert report["anchor_embedding_updated"]

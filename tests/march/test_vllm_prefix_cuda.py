"""Opt-in production-model GPU acceptance for the experimental prefix profile."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(os.environ.get("MARCH_TEST_VLLM_PREFIX_CACHE") != "1",
                    reason="set MARCH_TEST_VLLM_PREFIX_CACHE=1 with built fork and production checkpoints")
@pytest.mark.parametrize("profile", ["eager", "graph"])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("model_env,tp_env", [
    ("MARCH_PREFIX_TEST_4B_MODEL", "MARCH_PREFIX_TEST_4B_TP"),
    ("MARCH_PREFIX_TEST_35B_MODEL", "MARCH_PREFIX_TEST_35B_TP"),
], ids=["safin1-4b", "safin1-35b-a3b"])
def test_production_prefix_restore_and_policy_update(tmp_path, model_env, tp_env, profile, batch_size):
    # Explicit opt-in must fail if its required model/GPU is missing.
    import torch
    assert torch.cuda.is_available(), "CUDA is required by the requested acceptance run"
    assert os.environ.get(model_env), f"Set {model_env} to the production checkpoint"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "third_party/vllm"), str(ROOT),
                                        env.get("PYTHONPATH", "")])
    report = tmp_path / "prefix.json"
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/march/check_prefix_cache.py"),
        "--model", env[model_env], "--profile", profile, "--batch-size", str(batch_size),
        "--tensor-parallel-size", env.get(tp_env, "1"),
        "--output", str(report)], env=env, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    data = json.loads(report.read_text())
    assert data["accepted"]
    if profile == "graph":
        assert data["graph_replays"] > 0 and data["graph_storage_preserved"]
    assert data["scheduler_hits"] > 0 and data["restore_events"] > 0
    assert data["weight_update_invalidated"]
    assert all(not worker["entries"] and not worker["bytes"] for worker in data["final_cache"])

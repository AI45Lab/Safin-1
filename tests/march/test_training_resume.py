"""Acceptance checks must reject reset state and inspect unsampled weight values."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def tool():
    spec = importlib.util.spec_from_file_location("march_resume_acceptance",
                                                ROOT / "scripts/march/smoke_training_resume.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("tuner,key", [("full", "--mcore_model"), ("lora", "--mcore_adapter")])
def test_resume_dry_run_does_not_create_outputs_or_download(tmp_path, tuner, key):
    out = tmp_path / "new"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/march/smoke_training_resume.py"),
                             "--model", str(tmp_path / "missing-model"), "--output", str(out),
                             "--task", "sft", "--tuner", tuner, "--dry-run"],
                            text=True, capture_output=True, check=True)
    report = json.loads(result.stdout)
    resumed = report["cases"][1]["command"]
    assert resumed[resumed.index("--finetune") + 1] == "false"
    assert resumed[resumed.index(key) + 1].endswith("continuous/checkpoint-2")
    for option in ("--no_save_optim", "--no_save_rng", "--no_load_optim", "--no_load_rng"):
        assert resumed[resumed.index(option) + 1] == "false"
    assert not out.exists()


@pytest.mark.parametrize("field", ["iteration", "consumed_train_samples", "optimizer_sample_sha256",
                                  "rng_sha256", "scheduler"])
def test_resume_rejects_each_reset_component(tmp_path, field):
    a, b = tmp_path / "continuous", tmp_path / "resumed"
    for root in (a, b):
        (root / "probe").mkdir(parents=True)
    saved = {"event": "before_save", "rank": 0, "iteration": 2, "consumed_train_samples": 2,
             "optimizer_sample_sha256": "moments", "rng_sha256": "rng", "scheduler": {"num_steps": 2}}
    loaded = dict(saved, event="after_load")
    (a / "probe/rank-0.jsonl").write_text(json.dumps(saved) + "\n")
    (b / "probe/rank-0.jsonl").write_text(json.dumps(loaded) + "\n")
    assert tool().validate_restoration(a, b, 2, 1)[0]["loaded"]["iteration"] == 2
    loaded[field] = "reset"
    (b / "probe/rank-0.jsonl").write_text(json.dumps(loaded) + "\n")
    with pytest.raises(ValueError, match="restored state mismatch"):
        tool().validate_restoration(a, b, 2, 1)


def test_final_weight_comparison_checks_every_value(tmp_path):
    import torch
    from safetensors.torch import save_file
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    x = torch.zeros(512)
    y = x.clone()
    y[113] = 1  # Outside the probe's prefix/middle/suffix samples.
    save_file({"anchor.weight": x}, a / "model.safetensors")
    save_file({"anchor.weight": y}, b / "model.safetensors")
    result = tool().compare_weights(a, b, atol=0, rtol=0)
    assert result["outside_tolerance"] == ["anchor.weight"]
    assert result["max_abs_error"] == 1


def test_training_environment_uses_short_ipc_path_and_preserves_parent(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import os
    long_path = str(tmp_path / ("long-shared-storage-path" * 8))
    monkeypatch.setenv("TMPDIR", long_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("SWIFT_RELEASE_OPTIM_BEFORE_FINAL_SAVE", "1")
    monkeypatch.setenv("MEGATRON_SKIP_FINAL_SAVE", "1")
    args = SimpleNamespace(model=tmp_path / "model", output=tmp_path / "out",
                           nproc=1, tp=1, ep=1, max_length=1024)
    env = tool().training_environment(args)
    assert env["TMPDIR"] == "/tmp"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert "SWIFT_RELEASE_OPTIM_BEFORE_FINAL_SAVE" not in env
    assert "MEGATRON_SKIP_FINAL_SAVE" not in env
    assert os.environ["TMPDIR"] == long_path
    assert os.environ["HF_HUB_OFFLINE"] == "0"


@pytest.mark.parametrize("cursor", ["5961", "2"])
def test_resume_preflight_rejects_source_model_cursor(tmp_path, cursor):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text(cursor)
    state = tmp_path / "iter_0000002"
    state.mkdir()
    (state / "common.pt").write_bytes(b"common metadata")
    (state / ".metadata").write_bytes(b"shard metadata")
    if cursor == "2":
        tool().validate_resume_checkpoint(tmp_path, 2)
    else:
        with pytest.raises(ValueError, match="tracker"):
            tool().validate_resume_checkpoint(tmp_path, 2)


@pytest.mark.parametrize("steps,resume_step,count", [(3, 2, 3), (6, 2, 5), (7, 2, 7)])
def test_storage_preflight_counts_periodic_and_final_checkpoints(tmp_path, monkeypatch, steps, resume_step, count):
    from types import SimpleNamespace
    module = tool()
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.safetensors").write_bytes(b"x" * 128)
    (model / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"a": "weights.safetensors", "b": "weights.safetensors"}}))
    args = SimpleNamespace(model=model, output=tmp_path / "new" / "run", steps=steps,
                           resume_step=resume_step, tuner="full", storage_budget_gib=None)
    seen = []
    def disk_usage(path):
        seen.append(path)
        return SimpleNamespace(free=10 ** 10)
    monkeypatch.setattr(module.shutil, "disk_usage", disk_usage)
    full = module.storage_preflight(args)
    args.tuner = "lora"
    lora = module.storage_preflight(args)
    assert full["retained_checkpoints"] == count
    assert full["weight_bytes"] == 128  # A shard is counted once despite multiple index keys.
    assert full["required_bytes"] > lora["required_bytes"] > 128 * count
    assert seen == [tmp_path, tmp_path]
    assert not args.output.parent.exists()


def test_storage_preflight_rejects_low_space_without_creating_output(tmp_path, monkeypatch):
    from types import SimpleNamespace
    module = tool()
    (tmp_path / "model.safetensors").write_bytes(b"x" * 128)
    args = SimpleNamespace(model=tmp_path, output=tmp_path / "new" / "run", steps=3,
                           resume_step=2, tuner="full", storage_budget_gib=2)
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=1024 ** 3))
    with pytest.raises(ValueError, match="Insufficient output storage"):
        module.storage_preflight(args)
    assert not args.output.parent.exists()
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=2 * 1024 ** 3))
    report = module.storage_preflight(args)
    assert report["kind"] == "explicit"
    assert report["required_bytes"] == 2 * 1024 ** 3


@pytest.mark.parametrize("budget", ["0", "-1", "nan", "inf"])
def test_dry_run_rejects_invalid_storage_budgets(tmp_path, budget):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/march/smoke_training_resume.py"),
                             "--model", "/missing", "--output", str(tmp_path / "new"),
                             "--task", "cpt", "--storage-budget-gib", budget, "--dry-run"],
                            text=True, capture_output=True)
    assert result.returncode != 0
    assert "Storage budget must be finite and positive" in result.stderr
    assert not (tmp_path / "new").exists()



def test_low_capacity_run_fails_before_gpu_query_and_output_creation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    module = tool()
    (tmp_path / "model.safetensors").write_bytes(b"x" * 128)
    args = SimpleNamespace(model=tmp_path, output=tmp_path / "new", steps=3, resume_step=2,
                           tuner="full", task="cpt", max_length=1024, atol=0, rtol=0,
                           storage_budget_gib=None, nproc=8, tp=1, ep=8, dry_run=False)
    monkeypatch.setitem(sys.modules, "verify_checkpoint", SimpleNamespace(
        inspect_checkpoint=lambda _: None, resolve_training_checkpoint=lambda _: None))
    def unexpected_gpu_query():
        raise AssertionError("Capacity must be checked before querying CUDA")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        device_count=unexpected_gpu_query)))
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    with pytest.raises(ValueError, match="Insufficient output storage"):
        module.run(args)
    assert not args.output.exists()

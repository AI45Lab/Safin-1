"""CPU checks for public training selection and inherited environment guards."""
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def launch(task, model, **overrides):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("HSD_", "MARCH_")) and k != "MODEL_PROFILE"}
    import sys
    env.update(MODEL=str(model), DATASET="examples/march/sft.jsonl", OUTPUT_DIR="/tmp/unused-march-dry-run",
               DRY_RUN="1", PYTHON_BIN=sys.executable, **overrides)
    return subprocess.run(["bash", str(ROOT / f"scripts/march/train_{task}.sh")], env=env,
                          text=True, capture_output=True)


@pytest.mark.parametrize("task", ["cpt", "sft"])
def test_unknown_model_requires_explicit_family(task):
    result = launch(task, "Qwen/Qwen3.5-35B-A3B")
    assert result.returncode != 0
    assert "Set MODEL_PROFILE" in result.stderr
    explicit = launch(task, "Qwen/Qwen3.5-35B-A3B", MODEL_PROFILE="35ba3b")
    assert explicit.returncode == 0, explicit.stderr
    assert "safin1_35b_a3b.yaml" in explicit.stdout
    assert "--max_length 16384" in explicit.stdout


@pytest.mark.parametrize("variant,profile", [("safin1_4b", "4b"), ("safin1_35b_a3b", "35ba3b")])
def test_saved_runtime_is_preserved_and_family_conflicts_fail(tmp_path, variant, profile):
    runtime = json.loads((ROOT / f"configs/march/{variant}.yaml").read_text())
    runtime.pop("model_type")
    runtime["hsd_read_block_size"] = 4096
    (tmp_path / "config.json").write_text(json.dumps({"model_type": variant, "text_config": runtime}))
    result = launch("sft", tmp_path)
    assert result.returncode == 0, result.stderr
    assert "--hsd-runtime-config" not in result.stdout
    wrong = "4b" if profile == "35ba3b" else "35ba3b"
    result = launch("sft", tmp_path, MODEL_PROFILE=wrong)
    assert result.returncode != 0 and "conflicts" in result.stderr


@pytest.mark.parametrize("name", ["HSD_DEBUG_DETACH_TEXT_O", "HSD_DEBUG_ZERO_TEXT_O", "HSD_ANCHOR_MLP"])
def test_intrusive_environment_fails_before_training(name):
    from march.safin1.runtime_validation import validate_training_environment
    with pytest.raises(ValueError, match=name):
        validate_training_environment({name: "TRUE"})
    validate_training_environment({name: "0"})
    result = launch("cpt", "remote/model", MODEL_PROFILE="4b", **{name: "1"})
    assert result.returncode != 0 and name in result.stderr


def test_bad_local_config_is_not_silently_defaulted(tmp_path):
    (tmp_path / "config.json").write_text("{invalid}")
    result = launch("sft", tmp_path, MODEL_PROFILE="4b")
    assert result.returncode != 0 and "JSONDecodeError" in result.stderr



@pytest.mark.parametrize("task", ["cpt", "sft"])
def test_moe_tp_enables_sequence_parallel_and_rejects_disabling_it(task):
    result = launch(task, "remote/35b", MODEL_PROFILE="35ba3b", TENSOR_MODEL_PARALLEL_SIZE="2")
    assert result.returncode == 0, result.stderr
    assert "--sequence_parallel true" in result.stdout
    result = launch(task, "remote/35b", MODEL_PROFILE="35ba3b", TENSOR_MODEL_PARALLEL_SIZE="2",
                    SEQUENCE_PARALLEL="false")
    assert result.returncode != 0 and "requires SEQUENCE_PARALLEL=true" in result.stderr
    single = launch(task, "remote/35b", MODEL_PROFILE="35ba3b", TENSOR_MODEL_PARALLEL_SIZE="1")
    assert single.returncode == 0 and "--sequence_parallel false" in single.stdout


def test_dense_tp_default_remains_without_sequence_parallel():
    result = launch("sft", "remote/4b", MODEL_PROFILE="4b", TENSOR_MODEL_PARALLEL_SIZE="2")
    assert result.returncode == 0 and "--sequence_parallel false" in result.stdout


@pytest.mark.parametrize("runtime", [
    {"hsd_history_mix_mode": "convex"}, {"hsd_route_weight_momentum_gamma": 0.25},
])
def test_saved_unimplemented_training_runtime_rejected_before_cli(tmp_path, runtime):
    config = json.loads((ROOT / "configs/march/safin1_4b.yaml").read_text())
    model_type = config.pop("model_type")
    config.update(runtime)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type, "text_config": config}))
    result = launch("sft", tmp_path)
    assert result.returncode != 0
    assert next(iter(runtime)) in result.stderr
    assert not result.stdout


@pytest.mark.parametrize("name,value", [
    ("HSD_LOCAL_STATE_MODE", "chunk"), ("HSD_HISTORY_READ_Q_MODE", "hidden_residual_lora"),
])
def test_retired_mode_environment_rejected_before_cli(name, value):
    result = launch("cpt", "remote/model", MODEL_PROFILE="4b", **{name: value})
    assert result.returncode != 0 and name in result.stderr


@pytest.mark.parametrize("task", ["cpt", "sft"])
@pytest.mark.parametrize("name", ["USE_HIST_LINEAR_ATTN", "USE_MCORE_GDN", "SWIFT_USE_MCORE_GDN"])
def test_public_entrypoint_cannot_disable_march(task, name):
    result = launch(task, "remote/model", MODEL_PROFILE="4b", **{name: "0"})
    assert result.returncode != 0 and name in result.stderr
    assert not result.stdout


def test_entrypoint_rejects_runtime_environment_drift_before_cli(tmp_path):
    config = json.loads((ROOT / "configs/march/safin1_4b.yaml").read_text())
    model_type = config.pop("model_type")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type, "text_config": config}))
    result = launch("sft", tmp_path, HSD_READ_TOPK="8")
    assert result.returncode != 0 and "hsd_read_topk" in result.stderr
    assert not result.stdout
    matching = launch("sft", tmp_path, HSD_READ_TOPK=str(config["hsd_read_topk"]))
    assert matching.returncode == 0, matching.stderr

"""CPU checks for offline preflight and the bounded production serving matrix."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def test_serving_dry_run_needs_no_gpu_or_model_download(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/march/smoke_serving_matrix.py"),
                             "--model-4b", "/models/4b", "--model-35b-a3b", "/models/35b",
                             "--output", str(tmp_path), "--dry-run"], check=True, text=True, capture_output=True)
    report = json.loads(result.stdout)
    assert [(row["tp"], row["dp"]) for row in report["cases"]] == [(1, 2), (2, 1), (1, 2)]
    assert report["concurrency"] == 8
    assert report["prompt_lengths"] == [511, 512, 513, 4096, 8000]
    assert list(tmp_path.iterdir()) == []


def test_offline_preflight_requires_real_source_markers(tmp_path):
    path = ROOT / "scripts/march/check_offline_build.py"
    spec = importlib.util.spec_from_file_location("march_offline_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = {"FETCHCONTENT_BASE_DIR": str(tmp_path)}
    assert not any(row["present"] for row in module.inspect(env).values())
    for name, (variable, marker) in module.DEPENDENCIES.items():
        source = tmp_path / (name + "-src")
        file = source / marker
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("fixture")
    assert all(row["present"] for row in module.inspect(env).values())



def load_verifier():
    path = ROOT / "scripts/march/verify_checkpoint.py"
    spec = importlib.util.spec_from_file_location("march_verify_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_training_export_path(tmp_path):
    (tmp_path / "args.json").write_text(json.dumps({"tuner_type": "full", "merge_lora": True}))
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    assert load_verifier().resolve_training_checkpoint(tmp_path) == checkpoint


def test_lora_reload_requires_the_merged_export(tmp_path):
    import pytest
    module = load_verifier()
    (tmp_path / "args.json").write_text(json.dumps({"tuner_type": "lora", "merge_lora": True}))
    # An unmerged checkpoint directory is not a valid substitute.
    (tmp_path / "checkpoint-1").mkdir()
    with pytest.raises(ValueError, match="expected HF export"):
        module.resolve_training_checkpoint(tmp_path)
    checkpoint = tmp_path / "checkpoint-1-merged"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    assert module.resolve_training_checkpoint(tmp_path) == checkpoint
    (tmp_path / "args.json").write_text(json.dumps({"tuner_type": "lora", "merge_lora": False}))
    with pytest.raises(ValueError, match="merge_lora true"):
        module.resolve_training_checkpoint(tmp_path)



def test_single_gpu_matrix_records_single_replica_scope(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/march/smoke_serving_matrix.py"),
                             "--model-4b", "/models/4b", "--model-35b-a3b", "/models/35b",
                             "--output", str(tmp_path), "--single-gpu", "--dry-run"],
                            check=True, text=True, capture_output=True)
    report = json.loads(result.stdout)
    assert [(row["tp"], row["dp"]) for row in report["cases"]] == [(1, 1), (1, 1)]


def test_serving_response_requires_generated_ids_and_finite_logprobs():
    import pytest
    path = ROOT / "scripts/march/smoke_serving_matrix.py"
    spec = importlib.util.spec_from_file_location("march_serving_acceptance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Special tokens may decode to empty text; recorded IDs still prove generation.
    data = {"usage": {"prompt_tokens": 512, "completion_tokens": 2},
            "choices": [{"text": "", "token_ids": [10, 11], "finish_reason": "length",
                         "logprobs": {"token_logprobs": [-0.1, -0.2]}}]}
    assert module.validate_completion(data, 512)["generated_ids"] == [10, 11]
    for values in ([], [None, None], [float("nan"), -0.2], [float("inf"), -0.2]):
        data["choices"][0]["logprobs"]["token_logprobs"] = values
        with pytest.raises(ValueError, match="finite"):
            module.validate_completion(data, 512)
    data["choices"][0]["logprobs"]["token_logprobs"] = [-0.1, -0.2]
    data["choices"][0]["token_ids"] = []
    with pytest.raises(ValueError, match="token IDs"):
        module.validate_completion(data, 512)
    data["choices"][0]["token_ids"] = [10, 11]
    with pytest.raises(ValueError, match="counts"):
        module.validate_completion(data, 513)


def test_serving_graph_acceptance_requires_graph_dispatch(tmp_path):
    import pytest
    path = ROOT / "scripts/march/smoke_serving_matrix.py"
    spec = importlib.util.spec_from_file_location("march_serving_graph_acceptance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    events = tmp_path / "events.jsonl"
    events.write_text(json.dumps({"event": "hsd_decode_dispatch_policy", "cudagraph_mode": "NONE"}) + "\n")
    with pytest.raises(ValueError, match="actual Graph"):
        module.read_graph_evidence(events)
    with events.open("a") as stream:
        stream.write(json.dumps({"event": "hsd_graphable_decode_hit", "cudagraph_mode": "FULL", "requests": 8}) + "\n")
    evidence = module.read_graph_evidence(events)
    assert evidence["graph_dispatches"] == {"FULL": 1}
    assert evidence["max_graph_requests"] == 8

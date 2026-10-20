"""Exercise the public launcher without CUDA or model weights."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("march_serve", ROOT / "scripts/march/serve_vllm.py")
serve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve)


def launch(tmp_path, model_type="safin1_4b", flags=(), overrides=None):
    model = tmp_path / "model with spaces"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text(json.dumps({"model_type": model_type}))
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("MARCH_", "HSD_", "VLLM_HSD_"))
           and key not in {"TP_SIZE", "DP_SIZE", "TENSOR_PARALLEL_SIZE", "DATA_PARALLEL_SIZE",
                           "MAX_NUM_SEQS", "MAX_MODEL_LEN", "GPU_MEMORY_UTILIZATION", "MODEL_REVISION"}}
    env.update(MODEL=str(model), PYTHON_BIN=sys.executable, PYTHONDONTWRITEBYTECODE="1")
    env.update(overrides or {})
    return subprocess.run(["bash", str(ROOT / "scripts/march/infer_vllm.sh"), "--dry-run", *flags],
                          env=env, capture_output=True, text=True)


@pytest.mark.parametrize("model_type", list(serve.MODEL_TYPES))
def test_graph_default_by_saved_model_type(tmp_path, model_type):
    process = launch(tmp_path, model_type, overrides={
        "VLLM_HSD_DETERMINISTIC_PREFILL_AUTOTUNE": "0",
        "VLLM_HSD_DETERMINISTIC_VARLEN_KKT": "0",
        "VLLM_HSD_DETERMINISTIC_ANCHOR_READ": "0",
    })
    assert process.returncode == 0, process.stderr
    plan = json.loads(process.stdout)
    cmd, env = plan["command"], plan["environment"]
    assert plan["profile"] == "graph"
    assert "--enforce-eager" not in cmd
    assert cmd[cmd.index("--served-model-name") + 1] == "Safin-1"
    assert "--no-enable-prefix-caching" in cmd
    assert env["VLLM_HSD_ALLOW_CUDAGRAPH"] == "1"
    assert env["VLLM_HSD_GRAPH_V3_FULL_CUDAGRAPH"] == "1"
    assert env["VLLM_HSD_DETERMINISTIC_PREFILL_AUTOTUNE"] == "1"
    assert json.loads(cmd[cmd.index("--additional-config") + 1])["hsd_prefill_backend"] == "varlen"
    assert json.loads(cmd[cmd.index("--compilation-config") + 1])["cudagraph_mode"] == "FULL"
    if serve.MODEL_TYPES[model_type] == "moe":
        assert env["VLLM_HSD_MOE_BATCH_INVARIANT_COMPONENTS"] == "norm"
        assert env["VLLM_HSD_GRAPH_V3_NUMERIC_SAFE_CAPTURE_SIZES"] == "1,16,32"
        assert cmd[cmd.index("--moe-backend") + 1] == "triton"
    else:
        assert "VLLM_HSD_GRAPH_V3_NUMERIC_SAFE_CAPTURE_SIZES" not in env
        assert env["VLLM_HSD_DETERMINISTIC_VARLEN_KKT"] == "1"
        assert env["VLLM_HSD_DETERMINISTIC_ANCHOR_READ"] == "1"
        assert env["VLLM_HSD_MOE_DECODE_BATCH_INVARIANT"] == "1"
        assert env["VLLM_HSD_MOE_BATCH_INVARIANT_COMPONENTS"] == "norm"
        assert env["VLLM_HSD_BATCH_INVARIANT_PROJECTIONS"] == "1"
        assert env["VLLM_HSD_BATCH_INVARIANT_GATE_NORM"] == "1"


@pytest.mark.parametrize("flags,overrides", [
    (("--profile", "eager"), {}),
    (("--enforce-eager",), {}),
    ((), {"MARCH_INFERENCE_PROFILE": "eager"}),
])
def test_eager_rollback_clears_graph_and_diagnostics(tmp_path, flags, overrides):
    overrides = dict(overrides, VLLM_HSD_ENABLE_GRAPH_V3="1",
                     VLLM_HSD_FLASH_ATTN_DECODE_MATCH_SINGLE_REQUEST_SCHEDULE="1",
                     VLLM_HSD_ENABLE_PREFIX_CACHE="1")
    process = launch(tmp_path, "safin1_35b_a3b", flags, overrides)
    assert process.returncode == 0, process.stderr
    plan = json.loads(process.stdout)
    assert "--enforce-eager" in plan["command"]
    assert plan["environment"]["VLLM_HSD_HISTORY_READ_BACKEND"] == "loop"
    assert "VLLM_HSD_ENABLE_GRAPH_V3" not in plan["environment"]
    assert "VLLM_HSD_ENABLE_PREFIX_CACHE" not in plan["environment"]
    assert "VLLM_HSD_FLASH_ATTN_DECODE_MATCH_SINGLE_REQUEST_SCHEDULE" not in plan["environment"]


def test_poisoned_parent_environment_is_not_forwarded():
    parent = {"VLLM_HSD_GRAPH_V3_OPAQUE_MLP": "1", "HSD_TOPK_BACKEND": "bad",
              "VLLM_FORCE_BATCH_INVARIANT_GEMMA_RMSNORM": "1",
              "CUDA_VISIBLE_DEVICES": "0", "HF_HUB_OFFLINE": "1"}
    assert serve.clean_environment(parent) == {"CUDA_VISIBLE_DEVICES": "0", "HF_HUB_OFFLINE": "1"}


def test_tp_dp_and_api_options_are_forwarded(tmp_path):
    process = launch(tmp_path, flags=("--port", "8123", "--served-model-name", "Safin-1-preview"),
                     overrides={"TP_SIZE": "1", "DP_SIZE": "8"})
    assert process.returncode == 0, process.stderr
    cmd = json.loads(process.stdout)["command"]
    assert cmd[cmd.index("--data-parallel-size") + 1] == "8"
    assert cmd[cmd.index("--served-model-name") + 1] == "Safin-1-preview"
    assert cmd[-2:] == ["--port", "8123"]


@pytest.mark.parametrize("model_type,flags,error", [
    ("safin1_35b_a3b", ("--max-num-seqs", "64"), "requires MAX_NUM_SEQS"),
    ("historical_qwen3_5_moe", (), "migrate to Safin-1"),
    ("safin1_4b", ("--enable-prefix-caching",), "controlled by the MARCH profile"),
    ("safin1_4b", ("--compilation-config.cudagraph_mode=NONE",), "controlled by the MARCH profile"),
    ("safin1_4b", ("-cc", '{"cudagraph_mode":"NONE"}'), "controlled by the MARCH profile"),
    ("safin1_35b_a3b", ("--max_num_seqs=64",), "controlled by the MARCH profile"),
    ("safin1_4b", ("--max-model-len", "0"), "positive integer"),
])
def test_invalid_profiles_fail_before_loading_weights(tmp_path, model_type, flags, error):
    process = launch(tmp_path, model_type, flags)
    assert process.returncode != 0
    assert error in process.stderr


def test_hub_config_revision_is_used_without_remote_code(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text('{"model_type":"safin1_35b_a3b"}')
    calls = []
    def download(*args, **kwargs):
        calls.append((args, kwargs))
        return str(config)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    assert serve.read_config("example/Safin-1", "release-1") == "safin1_35b_a3b"
    assert calls == [(("example/Safin-1", "config.json"), {"revision": "release-1"})]


def test_conflicting_config_families_rejected(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "safin1_4b",
        "text_config": {"model_type": "safin1_35b_a3b_text"}}))
    with pytest.raises(ValueError, match="same Safin-1 family"):
        serve.read_config(str(tmp_path))


@pytest.mark.parametrize("model_type", list(serve.MODEL_TYPES))
def test_explicit_eager_prefix_opt_in(tmp_path, model_type):
    result = launch(tmp_path, model_type, flags=(
        "--profile", "eager", "--experimental-prefix-cache", "--prefix-cache-max-bytes", "1048576"),
        overrides={"VLLM_HSD_PREFIX_CACHE_ALLOW_UNSAFE_UNALIGNED": "1"})
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    cmd, env = plan["command"], plan["environment"]
    assert plan["prefix_cache"] == "experimental"
    assert "--enable-prefix-caching" in cmd and "--no-enable-prefix-caching" not in cmd
    assert "--enforce-eager" in cmd and "--no-async-scheduling" in cmd
    assert cmd[cmd.index("--mamba-cache-mode") + 1] == "none"
    assert env["VLLM_HSD_ENABLE_PREFIX_CACHE"] == "1"
    assert env["VLLM_HSD_PREFIX_CACHE_MAX_BYTES"] == "1048576"
    assert "VLLM_HSD_ENABLE_GRAPH_V3" not in env
    assert "VLLM_HSD_PREFIX_CACHE_ALLOW_UNSAFE_UNALIGNED" not in env
    if serve.MODEL_TYPES[model_type] == "moe":
        assert env["VLLM_HSD_MOE_PREFIX_CANONICAL_CHUNK_SIZE"] == "1024"


@pytest.mark.parametrize("flags,overrides,error", [
    (("--profile", "eager", "--experimental-prefix-cache", "--prefix-cache-max-bytes", "0"), {}, "positive integer"),
    (("--profile", "eager"), {"MARCH_EXPERIMENTAL_PREFIX_CACHE": "yes"}, "must be 0 or 1"),
])
def test_bad_prefix_opt_in_fails_early(tmp_path, flags, overrides, error):
    result = launch(tmp_path, flags=flags, overrides=overrides)
    assert result.returncode != 0 and error in result.stderr


@pytest.mark.parametrize("nested", [False, True])
def test_persistent_prefix_opt_in_rejected(tmp_path, nested):
    cfg = {"model_type": "safin1_4b", "hsd_persistent_memory_enabled": True}
    if nested:
        cfg = {"model_type": "safin1_4b", "text_config": {
            "model_type": "safin1_4b_text", "hsd_persistent_memory_enabled": True}}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    result = subprocess.run([sys.executable, str(ROOT / "scripts/march/serve_vllm.py"),
        "--model", str(tmp_path), "--profile", "eager", "--experimental-prefix-cache", "--dry-run"],
        env={k: v for k, v in os.environ.items() if not k.startswith(("MARCH_", "HSD_", "VLLM_HSD_"))},
        capture_output=True, text=True)
    assert result.returncode != 0 and "Safety State" in result.stderr


def test_gpu_comparison_rejects_ids_and_logprob_changes():
    sys.path.insert(0, str(ROOT / "scripts/march"))
    try:
        spec = importlib.util.spec_from_file_location("march_prefix_check", ROOT / "scripts/march/check_prefix_cache.py")
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
    finally:
        sys.path.pop(0)
    ref = {"ids": [1, 2], "logprobs": [-0.2, -0.1]}
    assert checker.compare(ref, ref, 1e-3, 1e-3) == 0
    with pytest.raises(AssertionError, match="IDs"):
        checker.compare(ref, {"ids": [1, 3], "logprobs": [-0.2, -0.1]}, 1e-3, 1e-3)
    with pytest.raises(AssertionError, match="tolerance"):
        checker.compare(ref, {"ids": [1, 2], "logprobs": [-0.2, -0.3]}, 1e-3, 1e-3)


@pytest.mark.parametrize("model_type", list(serve.MODEL_TYPES))
def test_explicit_graph_prefix_opt_in(tmp_path, model_type):
    result = launch(tmp_path, model_type, flags=("--experimental-prefix-cache",))
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    cmd, env = plan["command"], plan["environment"]
    assert plan["profile"] == "graph" and plan["prefix_cache"] == "experimental"
    assert "--enable-prefix-caching" in cmd and "--enforce-eager" not in cmd
    assert env["VLLM_HSD_ENABLE_GRAPH_PREFIX_CACHE"] == "1"
    assert env["VLLM_HSD_ENABLE_GRAPH_V3"] == "1"
    assert env["VLLM_HSD_USE_PERSISTENT_STATE_SIDECAR"] == "1"
    assert env["VLLM_HSD_ENABLE_COMPACT_STATE_SLOTS"] == "1"
    assert env["VLLM_HSD_SERIALIZE_PREFIX_PREFILL"] == "1"
    assert env["VLLM_HSD_HISTORY_CACHE_BACKEND"] == "scheduler_paged"
    compilation = json.loads(cmd[cmd.index("--compilation-config") + 1])
    assert compilation["cudagraph_mode"] == "FULL"


def test_full_vocabulary_comparison_handles_masks_and_rejects_drift():
    sys.path.insert(0, str(ROOT / "scripts/march"))
    try:
        spec = importlib.util.spec_from_file_location("march_prefix_full_check", ROOT / "scripts/march/check_prefix_cache.py")
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
    finally:
        sys.path.pop(0)
    ref = [([0, 1, 2], [-0.1, -0.2, float("-inf")])]
    assert checker.compare_vocabulary(ref, ref, 1e-3, 1e-3) == 0
    with pytest.raises(AssertionError, match="tolerance"):
        checker.compare_vocabulary(ref, [([0, 1, 2], [-0.1, -0.4, float("-inf")])], 1e-3, 1e-3)
    with pytest.raises(AssertionError, match="layout"):
        checker.compare_vocabulary(ref, [([0, 2, 3], [-0.1, -0.2, float("-inf")])], 1e-3, 1e-3)


def test_backend_events_are_explicit_and_keep_graph_profile(tmp_path):
    process = launch(tmp_path, flags=('--backend-events',), overrides={'MARCH_EVENT_LOG': '/tmp/march-events.jsonl'})
    assert process.returncode == 0, process.stderr
    plan = json.loads(process.stdout)
    assert plan['profile'] == 'graph'
    assert plan['environment']['HSD_PAGED_HISTORY_BACKEND_EVENTS'] == '1'
    assert plan['environment']['VLLM_HSD_VARLEN_COUNTER_FILE'] == '/tmp/march-events.jsonl'
    ordinary = json.loads(launch(tmp_path).stdout)
    assert 'HSD_PAGED_HISTORY_BACKEND_EVENTS' not in ordinary['environment']

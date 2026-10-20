#!/usr/bin/env python3
"""Shared Safin-1 serving/evaluation profiles, built before CUDA imports."""
import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
MODEL_TYPES = {
    "safin1_4b": "dense", "safin1_4b_text": "dense",
    "safin1_35b_a3b": "moe", "safin1_35b_a3b_text": "moe",
}
GRAPH_SWITCHES = """
ALLOW_CUDAGRAPH ENABLE_GRAPHABLE_DECODE ENABLE_PADDED_STATIC_DECODE
ENABLE_UNSAFE_PADDED_STATIC_DECODE ENABLE_GRAPH_V3 GRAPH_V3_FULL_CUDAGRAPH
ENABLE_MULTI_REQUEST_VARLEN_PREFILL ENABLE_ARBITRARY_ANCHOR_KERNEL
USE_PERSISTENT_STATE_SIDECAR DISABLE_TEXT_STATE_CACHE ENABLE_COMPACT_STATE_SLOTS
ENABLE_COMPACT_STATE_KV_SPEC ENABLE_RAW_Q_PAGED_READ
ENABLE_FUSED_PAGED_HISTORY_SCALE_ADD USE_NATIVE_CONV_STATE_UPDATE
ENABLE_NO_ANCHOR_FAST_PATH ENABLE_UNSAFE_NO_ANCHOR_MULTI
ENABLE_NO_ANCHOR_COMMON_FAST_PATH ENABLE_UNSAFE_NO_ANCHOR_COMMON_MULTI
ENABLE_NO_ANCHOR_READ_ONLY SKIP_REDUNDANT_RECURRENT_STATE_STORE
COMMON_FAST_HISTORY_SETUP GRAPH_V3_FUSED_RMSNORM ENABLE_DECODE_PROJECTION_FUSION
GRAPH_V3_STRIDED_BA GRAPH_V3_STRIDED_QKV GRAPH_V3_FUSED_HISTORY_PREP
GRAPH_V3_FUSED_OUTPUT_MASK GRAPH_V3_STRIDED_HISTORY_INPUTS
GRAPH_V3_SPLIT_HISTORY_READ ENABLE_PAGED_PREFILL_HISTORY_READ
DISABLE_COMPILE_ALLREDUCE_RMS
""".split()


def read_config(model, revision=None, with_config=False):
    path = Path(model).expanduser()
    if path.is_dir():
        config_path = path / "config.json"
    elif path.is_absolute() or model.startswith((".", "~")):
        raise ValueError(f"Model directory does not exist: {model}")
    else:
        # Read JSON only; never execute checkpoint remote code to choose a profile.
        from huggingface_hub import hf_hub_download
        config_path = Path(hf_hub_download(model, "config.json", revision=revision))
    config = json.loads(config_path.read_text())
    model_type = config.get("model_type")
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model_type={model_type!r}; migrate to Safin-1 first")
    nested_type = config.get("text_config", {}).get("model_type", model_type)
    if MODEL_TYPES.get(nested_type) != MODEL_TYPES[model_type]:
        raise ValueError("Top-level and text_config model_type must identify the same Safin-1 family")
    return (model_type, config) if with_config else model_type


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def clean_environment(parent):
    # Profiles own kernel controls so an inherited diagnostic switch cannot
    # change the selected execution path. Preserve unrelated settings/tokens.
    return {key: value for key, value in parent.items()
            if not key.startswith(("VLLM_HSD_", "HSD_"))
            and key != "VLLM_FORCE_BATCH_INVARIANT_GEMMA_RMSNORM"}


def build_launch(args, extra, model_type, parent, config=None):
    graph = args.profile == "graph"
    moe = MODEL_TYPES[model_type] == "moe"
    prefix = args.experimental_prefix_cache
    if prefix:
        text = (config or {}).get("text_config", config or {})
        if any(bool(c.get("hsd_persistent_memory_enabled", False)) for c in (config or {}, text)):
            raise ValueError("Persistent state / Safety State requires prefix caching to remain disabled")
        if int(text.get("anchor_chunk_size", 512)) <= 0 or int(text.get("anchor_token_num", 1)) <= 0:
            raise ValueError("Experimental prefix reuse requires enabled memory anchors")
    if graph and moe and args.max_num_seqs not in (1, 16, 32):
        raise ValueError("MoE graph profile requires MAX_NUM_SEQS / --max-num-seqs of 1, 16 or 32")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("GPU_MEMORY_UTILIZATION must be between 0 and 1")
    managed = ("--additional-config", "--compilation-config", "--enable-prefix-caching",
               "--no-enable-prefix-caching",
               "--dtype", "--moe-backend", "--mamba-cache-mode", "--async-scheduling",
               "--no-enforce-eager", "--hf-config-path", "--hf-overrides", "--config", "-cc",
               "--model", "--revision", "--max-num-seqs", "--max-model-len",
               "--tensor-parallel-size", "--data-parallel-size", "--gpu-memory-utilization")
    for value in extra:
        option = value.split("=", 1)[0].replace("_", "-")
        if any(option == key or option.startswith(key + ".") for key in managed):
            raise ValueError(f"{option} is controlled by the MARCH profile; use --profile eager for rollback")
    env = clean_environment(parent)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "third_party/vllm"), str(ROOT)] +
        ([parent["PYTHONPATH"]] if parent.get("PYTHONPATH") else []))
    controls = {
        "VLLM_HSD_HISTORY_CACHE_BACKEND": "scheduler_paged" if graph else "legacy_side_paged",
        "VLLM_HSD_HISTORY_READ_BACKEND": "paged_batched" if graph else "loop",
    }
    additional = {
        "gdn_prefill_backend": "triton",
        "hsd_prefill_backend": "varlen" if graph else "segments",
        "hsd_decode_backend": "packed",
        "hsd_decode_prepare_backend": "batched",
        "hsd_history_cache_backend": controls["VLLM_HSD_HISTORY_CACHE_BACKEND"],
        "hsd_history_read_backend": controls["VLLM_HSD_HISTORY_READ_BACKEND"],
    }
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", args.model, "--dtype", "bfloat16",
               "--served-model-name", args.served_model_name,
               "--tensor-parallel-size", str(args.tensor_parallel_size),
               "--data-parallel-size", str(args.data_parallel_size),
               "--max-model-len", str(args.max_model_len),
               "--max-num-seqs", str(args.max_num_seqs),
               "--gpu-memory-utilization", str(args.gpu_memory_utilization),
               "--enable-prefix-caching" if prefix else "--no-enable-prefix-caching", "--no-async-scheduling",
               "--language-model-only", "--mamba-cache-mode", "none",
               "--additional-config", json.dumps(additional)]
    if args.revision:
        command += ["--revision", args.revision]
    if moe:
        command += ["--moe-backend", "triton"]
    if graph:
        controls.update({"VLLM_HSD_" + switch: "1" for switch in GRAPH_SWITCHES})
        controls.update({
            "VLLM_HSD_VARLEN_MIN_FRESH_PREFILL_TOKENS": "2048",
            "VLLM_HSD_VARLEN_REQUIRE_HOMOGENEOUS_PREFILL": "1",
            "VLLM_HSD_HISTORY_CACHE_MEMORY_FRACTION": "0.10",
            "VLLM_HSD_HISTORY_CACHE_MIN_PAGES": str(args.max_num_seqs),
            "VLLM_HSD_PADDED_STATIC_CAPTURE_MAX_BATCH": str(args.max_num_seqs),
            "HSD_PAGED_HISTORY_MAX_BATCH": str(args.max_num_seqs),
            "HSD_PAGED_HISTORY_FUSED_MAX_PAGES": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
        })
        # PyTorch accepts the newer alias too; do not leave a conflicting parent value.
        env.pop("PYTORCH_ALLOC_CONF", None)
        if moe:
            controls.update({
                "VLLM_HSD_GRAPH_V3_NUMERIC_SAFE_CAPTURE_SIZES": "1,16,32",
                "VLLM_HSD_MOE_DECODE_BATCH_INVARIANT": "1",
                "VLLM_HSD_MOE_BATCH_INVARIANT_COMPONENTS": "norm",
                "VLLM_HSD_BATCH_INVARIANT_PROJECTIONS": "1",
                "VLLM_HSD_BATCH_INVARIANT_GATE_NORM": "1",
                "VLLM_HSD_DETERMINISTIC_PREFILL_AUTOTUNE": "1",
            })
        else:
            # Dense needs the complete fixed prefill policy: autotune alone
            # left fresh-process boundary logprobs outside the strict limit.
            controls.update({
                "VLLM_HSD_DETERMINISTIC_PREFILL_AUTOTUNE": "1",
                "VLLM_HSD_DETERMINISTIC_VARLEN_KKT": "1",
                "VLLM_HSD_DETERMINISTIC_ANCHOR_READ": "1",
                "VLLM_HSD_MOE_DECODE_BATCH_INVARIANT": "1",
                "VLLM_HSD_MOE_BATCH_INVARIANT_COMPONENTS": "norm",
                "VLLM_HSD_BATCH_INVARIANT_PROJECTIONS": "1",
                "VLLM_HSD_BATCH_INVARIANT_GATE_NORM": "1",
            })
        command += ["--compilation-config", json.dumps({
            "cudagraph_mode": "FULL", "pass_config": {"fuse_allreduce_rms": False}})]
    else:
        controls.update({"VLLM_HSD_DISABLE_" + switch: "1" for switch in (
            "NO_ANCHOR_FAST_PATH", "NO_ANCHOR_COMMON_FAST_PATH", "NATIVE_CONV_STATE_UPDATE",
            "RAW_Q_PAGED_READ", "FUSED_PAGED_HISTORY_SCALE_ADD", "SKIP_REDUNDANT_RECURRENT_STATE_STORE")})
        command += ["--enforce-eager"]
    if prefix:
        if graph:
            controls["VLLM_HSD_ENABLE_GRAPH_PREFIX_CACHE"] = "1"
        controls.update({
            "VLLM_HSD_ENABLE_PREFIX_CACHE": "1",
            "VLLM_HSD_USE_PERSISTENT_STATE_SIDECAR": "1",
            "VLLM_HSD_ENABLE_COMPACT_STATE_SLOTS": "1",
            "VLLM_HSD_ENABLE_COMPACT_STATE_KV_SPEC": "1",
            "VLLM_HSD_DISABLE_TEXT_STATE_CACHE": "1",
            "VLLM_HSD_PREFIX_CACHE_MAX_BYTES": str(args.prefix_cache_max_bytes),
            "VLLM_HSD_PREFIX_CACHE_ALIGNED_PREFILL": "1",
            "VLLM_HSD_PREFIX_CACHE_ALIGN_ALL_PREFILLS": "1",
            "VLLM_HSD_PREFIX_CACHE_MIN_OBSERVATIONS": "1",
            "VLLM_HSD_SERIALIZE_PREFIX_PREFILL": "1",
            "VLLM_HSD_AVOID_MIXED_PREFILL_DECODE": "1",
        })
        if moe:
            controls["VLLM_HSD_MOE_PREFIX_CANONICAL_CHUNK_SIZE"] = "1024"
            controls["VLLM_HSD_SERIALIZE_MOE_PREFIX_HIT_PREFILL"] = "1"
            controls["VLLM_HSD_DETERMINISTIC_PREFILL_AUTOTUNE"] = "1"
    if getattr(args, "backend_events", False):
        controls["HSD_PAGED_HISTORY_BACKEND_EVENTS"] = "1"
    if parent.get("MARCH_EVENT_LOG"):
        controls["VLLM_HSD_VARLEN_COUNTER_FILE"] = parent["MARCH_EVENT_LOG"]
        controls["VLLM_HSD_VARLEN_COUNTER_AGGREGATE"] = "1"
    env.update(controls)
    return command + extra, env, controls



def build_engine_arguments(args, model_type, parent, config=None):
    """Return lm-eval/vLLM kwargs from the exact serving command/profile."""
    command, env, controls = build_launch(args, [], model_type, parent, config)
    values = {}
    scalar = {
        '--model': ('pretrained', str), '--revision': ('revision', str),
        '--dtype': ('dtype', str), '--tensor-parallel-size': ('tensor_parallel_size', int),
        '--data-parallel-size': ('data_parallel_size', int),
        '--max-model-len': ('max_model_len', int), '--max-num-seqs': ('max_num_seqs', int),
        '--gpu-memory-utilization': ('gpu_memory_utilization', float),
        '--moe-backend': ('moe_backend', str), '--mamba-cache-mode': ('mamba_cache_mode', str),
        '--additional-config': ('additional_config', json.loads),
        '--compilation-config': ('compilation_config', json.loads),
    }
    for index, option in enumerate(command):
        if option in scalar:
            name, convert = scalar[option]
            values[name] = convert(command[index + 1])
    values.update(enforce_eager='--enforce-eager' in command,
                  enable_prefix_caching='--enable-prefix-caching' in command,
                  async_scheduling=False, language_model_only=True)
    return values, env, controls

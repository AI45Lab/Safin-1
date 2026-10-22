#!/usr/bin/env python3
"""Check source-bound fresh-install, native-build and one-step training receipts.

This is an offline metadata check, not a replacement for the recorded GPU runs.
It does not certify convergence, full 35B optimizer resume or numerical parity.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVIDENCE = "docs/march/installation-acceptance-20261017.json"
TRAINING_DIRECTORIES = ("march", "swift", "third_party/mcore-bridge/src")
RECIPE_FILES = (
    "setup.py", "setup.cfg", "pyproject.toml", "MANIFEST.in",
    "third_party/mcore-bridge/setup.py", "third_party/mcore-bridge/setup.cfg",
    "scripts/march/install.sh", "scripts/march/check_environment.py",
    "scripts/march/_training_common.sh", "scripts/march/train_cpt.sh",
    "scripts/march/train_sft.sh", "scripts/march/smoke_megatron.sh",
    "scripts/march/smoke_hf.py", "scripts/march/verify_checkpoint.py",
    "scripts/march/verify_package.py",
)
NATIVE_DIRECTORIES = (
    "third_party/vllm/csrc", "third_party/vllm/cmake",
    "third_party/vllm/requirements/build",
)
NATIVE_FILES = (
    "third_party/vllm/setup.py", "third_party/vllm/pyproject.toml",
    "third_party/vllm/CMakeLists.txt",
)
NATIVE_MODULES = {
    "vllm._C", "vllm._C_stable_libtorch", "vllm._moe_C",
    "vllm.cumem_allocator", "vllm._flashmla_C", "vllm._flashmla_extension_C",
    "vllm.vllm_flash_attn._vllm_fa2_C", "vllm.vllm_flash_attn._vllm_fa3_C",
}
PACKAGE_GATES = {
    "cpu-framework-wheel", "cpu-framework-wheel-install", "cpu-verify-package",
    "cpu-native-training-imports", "cpu-tiny-hf-fixtures",
    "cpu-meta-startup-4B", "cpu-meta-startup-35B-A3B",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value, length=64):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%d}" % length, value) is not None


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def source_groups(root):
    root = Path(root).resolve()

    def paths(directory, pattern):
        base = root / directory
        require(base.is_dir() and not base.is_symlink(), f"missing source directory: {directory}")
        entries = sorted(base.rglob("*"))
        require(not any(p.is_symlink() for p in entries), f"symlink in source scope: {directory}")
        selected = [p for p in entries if p.is_file() and p.match(pattern)]
        require(selected, f"empty source scope: {directory}")
        return selected

    groups = {name: paths(name, "*.py") for name in TRAINING_DIRECTORIES}
    groups["installation_recipe"] = (
        [root / name for name in RECIPE_FILES]
        + paths("requirements", "*.txt")
        + paths("configs/march", "*.yaml")
    )
    groups["vllm_native_build"] = (
        [root / name for name in NATIVE_FILES]
        + [p for name in NATIVE_DIRECTORIES for p in paths(name, "*")]
    )
    for selected in groups.values():
        for path in selected:
            require(
                path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),
                "installation evidence sources must be regular files inside the checkout",
            )
    return groups


def fingerprints(root):
    root = Path(root).resolve()
    result = {}
    for name, paths in source_groups(root).items():
        files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        result[name] = {"files": len(files), "sha256": hashlib.sha256(encoded).hexdigest()}
    return result


def review(root, data):
    errors = []
    try:
        require(isinstance(data, dict), "installation evidence must be an object")
        require(type(data.get("installation_evidence_schema_version")) is int
                and data["installation_evidence_schema_version"] == 1, "missing installation evidence schema")
        require(data.get("source_fingerprints") == fingerprints(root),
                "training, dependency or native-build sources changed; renew the affected acceptance")
        cpu, gpu = data["fresh_repaired_training_cpu"], data["gpu_clean_build"]
        training, serving = gpu["training_full_size"], gpu["vllm_serving"]
        native = gpu["native_source_build"]
        require(gpu["status"] == cpu["status"] == training["status"] == serving["status"]
                == native["status"] == "passed", "incomplete fresh-install/build/training acceptance")
        require(digest(training["source_revision"], 40)
                and cpu["source_revision"] == training["source_revision"], "CPU/GPU training source revisions disagree")
        require(digest(training["source_archive_sha256"])
                and training["source_archive_sha256"] == cpu["source_archive_sha256"], "CPU/GPU training archives disagree")
        require(cpu["gpu_used"] is False and type(cpu["gpu_count"]) is int and cpu["gpu_count"] == 0,
                "CPU preparation must retain its zero-GPU boundary")
        require(cpu["installer"] == "passed" and training["pip_check"] == "passed",
                "fresh training installation or dependency consistency check did not pass")
        require(cpu["package_and_fixture_gates"] == {key: "passed" for key in PACKAGE_GATES},
                "fresh installed-wheel/import/fixture gates are incomplete")
        te = cpu["te_bindings_source_build"]
        require(te["status"] == "passed" and te["version"] == "2.14.1"
                and type(te["built_wheel_bytes"]) is int and te["built_wheel_bytes"] > 0,
                "fresh TE PyTorch binding source build did not pass")
        require(set(cpu["native_imports"]) == {
            "transformer_engine.pytorch", "megatron.core", "mcore_bridge.model.modules.march",
        }, "training native/import checks are incomplete")
        for record, keys in (
            (cpu, ("preparation_receipt_sha256", "startup_receipt_sha256", "source_manifest_sha256")),
            (training, ("completion_receipt_sha256", "source_manifest_sha256", "environment_receipt_sha256")),
            (native, ("source_archive_sha256", "source_manifest_sha256", "build_result_sha256", "native_receipt_sha256")),
            (serving, ("native_receipt_sha256", "api_results_sha256")),
            (te, ("installer_log_sha256", "source_archive_sha256", "built_wheel_sha256")),
        ):
            for key in keys:
                require(digest(record.get(key)), f"missing raw receipt fingerprint: {key}")
        require(training["source_manifest_sha256"] == cpu["source_manifest_sha256"],
                "CPU/GPU training manifests disagree")
        require(gpu["hardware"] == "one CUDA GPU" and training["offline"] is True
                and type(training["gpu_count"]) is int and training["gpu_count"] == 1,
                "missing offline single-GPU training boundary")
        for key, expected in (("sequence_length", 1024), ("tensor_parallel_size", 1),
                              ("expert_parallel_size", 1), ("optimizer_steps_completed", 2)):
            require(type(training[key]) is int and training[key] == expected, f"wrong training recipe: {key}")
        require(training["precision"] == "bfloat16" and training["task"] == "sft"
                and training["input_weights_and_saved_runtime_unchanged"] is True,
                "missing training recipe or unchanged-input verification")
        cases = training["cases"]
        require(isinstance(cases, list) and [c["model"] for c in cases] == ["4B", "35B-A3B"],
                "missing or duplicate full-size training cases")
        for case, tuner, model_type in zip(cases, ("full", "lora"), ("safin1_4b", "safin1_35b_a3b")):
            require(case["status"] == "passed" and case["tuner"] == tuner
                    and case["model_type"] == model_type and case["text_model_type"] == model_type + "_text",
                    "wrong tuner or exported model type")
            require(type(case["optimizer_steps"]) is int and case["optimizer_steps"] == 1,
                    "missing optimizer execution")
            require(finite(case["loss"]) and finite(case["grad_norm"]) and case["grad_norm"] > 0,
                    "nonfinite loss or missing finite nonzero gradients")
            require(case["loaded_all_weights"] is True and case["finite_reload_logits"] is True
                    and type(case["missing_unexpected_mismatched_weights"]) is int
                    and case["missing_unexpected_mismatched_weights"] == 0, "strict HF reload did not pass")
            for key in ("weight_shards", "weight_keys", "exported_saved_runtime_keys_preserved"):
                require(type(case[key]) is int and case[key] > 0, f"incomplete checkpoint export: {key}")
            require(type(case["legacy_summary_config_keys"]) is int and case["legacy_summary_config_keys"] == 0,
                    "legacy summary configuration keys remain")
            require(type(case["greedy_generated_tokens"]) is int and case["greedy_generated_tokens"] == 8
                    and type(case["seed"]) is int and case["seed"] == 123, "missing seeded greedy reload generation")
            for key in ("training_log_sha256", "training_args_sha256", "reload_validation_sha256"):
                require(digest(case.get(key)), f"missing training case receipt: {key}")
            require(case["runtime"]["anchor_chunk_size"] == 512 and case["runtime"]["anchor_token_num"] == 1
                    and case["runtime"]["hsd_read_topk"] == 4
                    and case["runtime"]["hsd_read_block_size"] == 32768
                    and case["runtime"]["hsd_global_read_layers"] == "10-30", "saved anchor/runtime recipe differs")
            if tuner == "lora":
                require(case["merged_full_weight_export"] is True and case["lora_rank"] == 8
                        and case["target_modules"] == ["linear_qkv", "linear_proj"], "35B LoRA/merged export differs")
        require(native["native_build"] == "source" and native["inherited_native_binaries"] == 0,
                "native builds must not inherit prebuilt vLLM extensions")
        modules = native["extensions"]
        require(isinstance(modules, dict) and set(modules) == NATIVE_MODULES, "eight native extension receipts required")
        for record in modules.values():
            require(type(record["bytes"]) is int and record["bytes"] > 0 and digest(record["sha256"]),
                    "invalid native extension fingerprint")
        require(native["source_archive_sha256"] == serving["source_archive_sha256"]
                and serving["native_extensions"] == 8, "native-build and serving provenance disagree")
        require(serving["dependency_repair"] == "incremental_fla_only",
                "retain the incremental FLA repair boundary; it was not a second native rebuild")
        require([(c["model"], c["tp"], c["dp"]) for c in serving["cases"]]
                == [("4B", 1, 1), ("35B-A3B", 1, 1)], "missing fresh-native TP1/DP1 serving cases")
        for case in serving["cases"]:
            require(case["passed"] is True and case["api_requests_checked"] == 13
                    and case["serial_prompt_lengths"] == [511, 512, 513, 4096, 8000]
                    and case["concurrency"] == 8 and isinstance(case["graph_dispatches"], dict)
                    and type(case["graph_dispatches"].get("FULL")) is int
                    and case["graph_dispatches"]["FULL"] > 0,
                    "fresh-native serving smoke is incomplete")
    except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
        errors.append(str(error))
    return {"ready": not errors, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--evidence", default=DEFAULT_EVIDENCE)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        path = (root / args.evidence).resolve()
        require(path.is_relative_to(root), "evidence must stay inside the checkout")
        result = review(root, json.loads(path.read_text()))
    except (OSError, ValueError) as error:
        result = {"ready": False, "errors": [str(error)]}
    print(json.dumps(result, indent=2))
    return int(not result["ready"])


if __name__ == "__main__":
    raise SystemExit(main())

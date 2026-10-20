#!/usr/bin/env python3
"""Run or verify source-bound MARCH CUDA operator correctness acceptance.

Static verification needs no Torch/GPU. GPU execution is offline, disallows
skipped/failed cases and records actual custom Triton launches per test.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVIDENCE = "docs/march/operator-correctness-20261015.json"
TESTS = (
    "tests/march/test_chunk_gdn_cuda.py",
    "tests/march/test_sparse_topk_cuda.py",
    "tests/march/test_backends.py",
    "tests/march/test_backend_policy.py",
    "tests/march/test_paged_history_cuda.py",
    "tests/march/test_operator_acceptance_cuda.py",
    "tests/march/test_training_contract.py",
    "tests/march/test_training_import.py",
    "tests/march/test_adapter_export.py",
    "tests/march/test_training_runtime_policy.py",
    "tests/march/test_training_entrypoints.py",
)
CONTRACT_SOURCES = (
    "swift/megatron/init.py", "swift/megatron/trainers/utils.py", "swift/megatron/utils/utils.py",
    "scripts/march/_training_common.sh", "scripts/march/train_cpt.sh", "scripts/march/train_sft.sh",
)
# Every custom JIT source must have an explicit family. Vendored FlashQLA is
# checked through its admitted Hopper entrypoints; unused upstream CP helpers
# are not exposed as MARCH-supported operators.
FAMILIES = {
    "chunk": ["march/kernels/chunk_anchor_triton.py", "march/kernels/chunk_scan_triton.py"],
    "dense_global_read": ["march/kernels/march_global_read.py"],
    "sparse_topk": ["march/kernels/march_global_read_topk.py"],
    "hf_segment_states": ["march/safin1/fused_march_segments.py"],
    "hf_top4": ["march/safin1/fused_history_read.py"],
    "varlen": ["march/inference/varlen_segments.py"],
    "paged_decode": ["march/inference/paged_topk_triton.py", "march/inference/paged_fused_triton.py",
                     "march/inference/paged_memory_triton.py"],
    "paged_prefill": ["march/inference/paged_prefill_triton.py"],
    "epilogues": ["march/inference/paged_epilogue_triton.py"],
}
REQUIRED_TESTS = (
    "test_chunk_forward_backward", "test_sparse_topk_cuda_modes_match_reference",
    "test_sparse_topk_production_geometry", "test_paged_cuda_matches_torch_reference",
    "test_explicit_cuda_backend_matches_reference", "test_dense_history_cuda_graph_replay",
    "test_persistent_state_cuda_train_save_reload", "test_chunk_state_cuda_all_batches_and_heads",
    "test_dense_packed_read_all_input_gradients", "test_dense_decode_oracle", "test_hf_top4_read_oracle",
    "test_fused_epilogues_oracle", "test_paged_prefill_oracle", "test_varlen_recurrence_oracle",
    "test_flashqla_forward_all_gradients", "test_flashqla_inference_states",
    "test_sparse_alternate_kernels", "test_hf_selective_anchor_states",
    "test_missing_active_checkpoint_weight_fails_before_any_conversion",
    "test_layer_import_restores_live_weights", "test_base_qwen_layer_still_initializes_from_qkv",
    "test_real_loader_selects_march_from_model_contract_without_legacy_flag",
    "test_environment_cannot_change_serialized_runtime", "test_public_entrypoint_cannot_disable_march",
    "test_real_layer_adapter_import_preserves_base_and_persistent_state",
    "test_route_adapter_safetensors_roundtrip", "test_exported_march_adapter_loads_in_hf_peft",
    "test_unmerged_adapter_rejects_trainable_raw_state",
)


def fingerprints(root):
    paths = set((root / "march").rglob("*.py"))
    paths.update((root / "third_party/mcore-bridge/src/mcore_bridge").rglob("*.py"))
    paths.update((root / "third_party/vllm/vllm/model_executor/layers/fla").rglob("*.py"))
    paths.update(root / name for name in TESTS)
    paths.update(root / name for name in CONTRACT_SOURCES)
    paths.add(root / "scripts/march/check_operator_correctness.py")
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def inventory(root):
    mapped = {name for names in FAMILIES.values() for name in names}
    definitions = {}
    for path in sorted((root / "march").rglob("*.py")):
        name = str(path.relative_to(root))
        if "/_flashqla/" in name:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and any("jit" in ast.unparse(d) for d in node.decorator_list):
                if name not in mapped:
                    raise ValueError(f"unreviewed custom operator source: {name}")
                definitions[name + "::" + node.name] = next(f for f, names in FAMILIES.items() if name in names)
    return definitions


def review(root, data):
    errors = []
    if data.get("schema_version") != 1:
        errors.append("unsupported evidence schema")
    if data.get("source_sha256") != fingerprints(root):
        errors.append("source/test fingerprints changed or are incomplete; rerun CUDA acceptance")
    expected = inventory(root)
    if data.get("operator_inventory") != expected:
        errors.append("operator inventory changed; review and rerun acceptance")
    if data.get("test_files") != list(TESTS):
        errors.append("required test selection does not match")
    rows = data.get("tests", [])
    nodes = [row.get("nodeid", "") for row in rows]
    if not rows or len(set(nodes)) != len(nodes) or len(nodes) != data.get("collected"):
        errors.append("missing, duplicate or incomplete test reports")
    if any(row.get("outcome") != "passed" or row.get("phases") !=
           {"setup": "passed", "call": "passed", "teardown": "passed"} for row in rows) or data.get("pytest_exit_code") != 0:
        errors.append("operator acceptance contains failed/skipped/incomplete tests")
    for name in REQUIRED_TESTS:
        if not any(node.split("::")[-1].split("[")[0] == name for node in nodes):
            errors.append("required numerical case missing: " + name)
    hits = data.get("kernel_tests", {})
    for name in expected:
        associated = hits.get(name, [])
        if not associated or any(node not in nodes for node in associated):
            errors.append("custom Triton kernel has no passing launch evidence: " + name)
    if data.get("source_changed_during_run"):
        errors.append("source changed during GPU execution")
    if not data.get("environment", {}).get("cuda_available"):
        errors.append("no actual CUDA environment recorded")
    return {"ready": not errors, "errors": errors, "tests": len(rows),
            "custom_triton_kernels": len(expected), "status_counts": dict(Counter(r.get("outcome") for r in rows))}


def execute(root, output):
    import torch
    import pytest
    from triton.runtime.jit import JITFunction
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("Hopper CUDA GPU required, including admitted FlashQLA geometry")
    for key in ("OFFLINE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_EVALUATE_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[key] = "1"
    os.environ["PYTEST_ADDOPTS"] = ""
    os.environ["MARCH_TEST_MEGATRON"] = "1"
    sys.path[:0] = [str(root / "third_party/vllm"), str(root / "third_party/mcore-bridge/src"), str(root)]
    before = fingerprints(root)
    definitions = inventory(root)
    data = dict(schema_version=1, started_at=datetime.now(timezone.utc).isoformat(),
                source_sha256=before, operator_inventory=definitions, test_files=list(TESTS),
                tests=[], kernel_tests={}, collected=0)
    packages = {}
    for name in ("torch", "triton", "flash-linear-attention", "tilelang", "megatron-core", "transformer-engine"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    data["environment"] = dict(cuda_available=True, gpu=torch.cuda.get_device_name(),
                               compute_capability=list(torch.cuda.get_device_capability()),
                               cuda_runtime=torch.version.cuda, packages=packages)
    data["revision"] = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()

    class Reports:
        current = None
        phases = {}

        def pytest_collection_finish(self, session):
            data["collected"] = len(session.items)

        def pytest_runtest_setup(self, item):
            self.current = item.nodeid

        def pytest_runtest_logreport(self, report):
            row = self.phases.setdefault(report.nodeid, {})
            row[report.when] = "xfail" if hasattr(report, "wasxfail") else report.outcome

    plugin = Reports()
    original = JITFunction.run

    def record_launch(self, *args, **kwargs):
        value = original(self, *args, **kwargs)
        path = Path(self.fn.__code__.co_filename).resolve()
        if path.is_relative_to(root) and plugin.current and not kwargs.get("warmup", False):
            name = str(path.relative_to(root)) + "::" + self.fn.__name__
            if name in definitions:
                tests = data["kernel_tests"].setdefault(name, [])
                if plugin.current not in tests:
                    tests.append(plugin.current)
        return value

    JITFunction.run = record_launch
    try:
        code = pytest.main(["-q", "-p", "no:cacheprovider", *[str(root / name) for name in TESTS]], plugins=[plugin])
        torch.cuda.synchronize()
    finally:
        JITFunction.run = original
    data["pytest_exit_code"] = int(code)
    for nodeid, phases in sorted(plugin.phases.items()):
        passed = phases == {"setup": "passed", "call": "passed", "teardown": "passed"}
        data["tests"].append(dict(nodeid=nodeid, outcome="passed" if passed else "not_passed", phases=phases))
    data["source_changed_during_run"] = before != fingerprints(root)
    data["completed_at"] = datetime.now(timezone.utc).isoformat()
    data["acceptance"] = review(root, data)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    return data["acceptance"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--run", action="store_true", help="execute all required tests on Hopper")
    parser.add_argument("--output", type=Path, help="new GPU evidence; keep raw outputs outside checkout")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.run:
        if args.output is None:
            parser.error("--run requires --output")
        result = execute(root, args.output)
    else:
        path = args.evidence or root / DEFAULT_EVIDENCE
        try:
            result = review(root, json.loads(path.read_text()))
        except (OSError, ValueError) as exc:
            result = {"ready": False, "errors": [str(exc)]}
    print(json.dumps(result, indent=2))
    return int(not result["ready"])


if __name__ == "__main__":
    raise SystemExit(main())

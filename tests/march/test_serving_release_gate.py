"""A publication checkbox cannot substitute for complete, current API evidence."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/march/check_serving_evidence.py"


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def evidence(tmp_path):
    checker = load(SCRIPT)
    for name in (
        "march/runtime.py",
        "third_party/vllm/vllm/runtime.py",
        "scripts/march/serve_vllm.py",
        "scripts/march/smoke_serving_matrix.py",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic runtime fixture\n")

    def responses(lengths):
        return [
            {
                "input_tokens": length,
                "generated_ids": [42, 43],
                "token_logprobs": [-1.0, -2.0],
                "usage": {"prompt_tokens": length, "completion_tokens": 2},
            }
            for length in lengths
        ]

    results, samples = [], []
    for model, tp, dp in (("4B", 1, 2), ("35B-A3B", 2, 1), ("35B-A3B", 1, 2)):
        results.append(
            {
                "model": model,
                "tp": tp,
                "dp": dp,
                "passed": True,
                "api_requests_checked": 13,
                "completion_tokens": 26,
                "graph_dispatches": {"FULL": 3},
                "max_graph_requests": 4,
                "server_log_sha256": "a" * 64,
                "event_log_sha256": "b" * 64,
            }
        )
        samples.append(
            {
                "model": model,
                "tp": tp,
                "dp": dp,
                "serial": responses([511, 512, 513, 4096, 8000]),
                "concurrent": responses([511, 512, 513, 1024, 2048, 4096, 8000, 512]),
            }
        )
    attempt = {
        "status": "passed",
        "source_revision": "c" * 40,
        "results": deepcopy(results),
        "case_count_passed": 3,
        "api_requests_checked": 39,
        "completion_tokens": 78,
        "post_run_source_drivers_and_eight_native_hashes_verified": True,
        **{
            name: "d" * 64
            for name in (
                "raw_report_sha256",
                "acceptance_sha256",
                "matrix_log_sha256",
                "source_manifest_sha256",
            )
        },
    }
    data = {
        "serving_evidence_schema_version": 1,
        "status": "passed",
        "source_revision": "c" * 40,
        "runtime_source_sha256": checker.fingerprints(tmp_path),
        "profile": "graph",
        "prefix_cache": False,
        "offline": True,
        "hardware": "two NVIDIA H200 GPUs",
        "native_build": "reused",
        "max_model_len": 8192,
        "max_num_seqs": 32,
        "submitted_http_concurrency": 8,
        "max_new_tokens": 8,
        "seed": 123,
        "temperature": 0,
        "serial_prompt_lengths": [511, 512, 513, 4096, 8000],
        "concurrent_prompt_lengths": [511, 512, 513, 1024, 2048, 4096, 8000, 512],
        "post_run_source_drivers_and_eight_native_hashes_verified": True,
        "preflight": {"native_extensions_verified": 8},
        "results": results,
        "api_samples": samples,
        "attempts": [attempt],
        "case_count_passed": 3,
        "api_requests_checked": 39,
        "completion_tokens": 78,
    }
    return checker, tmp_path, data


def test_complete_source_bound_receipt_accepts_shorter_eos_responses(evidence):
    checker, root, data = evidence
    assert checker.review(root, data)[
        "ready"
    ]  # counts need not equal the eight-token maximum


@pytest.mark.parametrize("change", ["edit", "add", "remove", "entrypoint", "symlink"])
def test_changed_runtime_invalidates_gpu_acceptance(evidence, change):
    checker, root, data = evidence
    path = root / "march/runtime.py"
    if change == "edit":
        path.write_text("# changed computation\n")
    elif change == "add":
        (root / "march/new_kernel.py").write_text("# new computation\n")
    elif change == "remove":
        path.unlink()
    elif change == "symlink":
        path.unlink()
        target = root / "elsewhere.py"
        target.write_text("# redirected source\n")
        path.symlink_to(target)
    else:
        (root / "scripts/march/serve_vllm.py").write_text("# changed runtime flags\n")
    assert not checker.review(root, data)["ready"]


@pytest.mark.parametrize("name", ["docs/note.md", "scripts/march/check_environment.py"])
def test_unrelated_documentation_and_installer_checks_do_not_invalidate_runtime(
    evidence, name
):
    checker, root, data = evidence
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# unrelated change\n")
    assert checker.review(root, data)["ready"]


@pytest.mark.parametrize(
    "change",
    [
        "partial",
        "missing_case",
        "duplicate_case",
        "missing_samples",
        "wrong_boundary",
        "wrong_usage",
        "missing_ids",
        "boolean_id",
        "missing_logprob",
        "nan",
        "inf",
        "no_graph",
        "eager",
        "prefix_cache",
        "offline",
        "no_native",
        "no_latest_native",
        "no_log_hash",
        "wrong_totals",
        "wrong_latest",
        "missing_source",
        "forged_source",
    ],
)
def test_incomplete_or_inconsistent_acceptance_is_rejected(evidence, change):
    checker, root, data = evidence
    response = data["api_samples"][0]["serial"][0]
    if change == "partial":
        data["status"] = "partial_failure"
    elif change == "missing_case":
        data["results"].pop()
    elif change == "duplicate_case":
        data["api_samples"][2] = deepcopy(data["api_samples"][1])
    elif change == "missing_samples":
        data.pop("api_samples")
    elif change == "wrong_boundary":
        response["input_tokens"] = 510
    elif change == "wrong_usage":
        response["usage"]["completion_tokens"] = 3
    elif change == "missing_ids":
        response["generated_ids"] = []
    elif change == "boolean_id":
        response["generated_ids"][0] = True
    elif change == "missing_logprob":
        response["token_logprobs"].pop()
    elif change == "nan":
        response["token_logprobs"][0] = float("nan")
    elif change == "inf":
        response["token_logprobs"][0] = float("inf")
    elif change == "no_graph":
        data["results"][0]["graph_dispatches"] = {}
    elif change == "eager":
        data["profile"] = "eager"
    elif change == "prefix_cache":
        data["prefix_cache"] = True
    elif change == "offline":
        data["offline"] = False
    elif change == "no_native":
        data["post_run_source_drivers_and_eight_native_hashes_verified"] = False
    elif change == "no_latest_native":
        data["attempts"][-1][
            "post_run_source_drivers_and_eight_native_hashes_verified"
        ] = False
    elif change == "no_log_hash":
        data["results"][0].pop("server_log_sha256")
    elif change == "wrong_totals":
        data["completion_tokens"] = 79
    elif change == "wrong_latest":
        data["attempts"][-1]["status"] = "failed"
    elif change == "missing_source":
        data.pop("runtime_source_sha256")
    elif change == "forged_source":
        data["runtime_source_sha256"]["march"]["sha256"] = "e" * 64
    if change in {"no_graph", "no_log_hash"}:
        # Keep summaries consistent so rejection must come from the missing
        # Graph/log proof rather than a separate latest-attempt mismatch.
        data["attempts"][-1]["results"] = deepcopy(data["results"])
    result = checker.review(root, data)
    assert not result["ready"], change
    if change == "no_graph":
        assert "Graph dispatch" in result["errors"][0]
    elif change == "no_log_hash":
        assert "case fingerprint" in result["errors"][0]


@pytest.mark.parametrize("value", [None, [], {}, {"status": "passed"}, 1])
def test_malformed_evidence_fails_closed(evidence, value):
    checker, root, _ = evidence
    assert not checker.review(root, value)["ready"]


def publication_fixture(root, data):
    # Operator evidence has independent numerical tests. Isolate that gate here
    # to demonstrate the new serving check through the actual publication CLI.
    shutil.copyfile(SCRIPT, root / "scripts/march/check_serving_evidence.py")
    shutil.copyfile(
        ROOT / "scripts/march/check_publication.py",
        root / "scripts/march/check_publication.py",
    )
    (root / "scripts/march/check_operator_correctness.py").write_text(
        'def review(root, data): return {"ready": True}\n'
    )
    docs = root / "docs/march"
    docs.mkdir(parents=True)
    (docs / "operator-correctness-20261015.json").write_text("{}")
    # Installation acceptance is independently exercised by its release-gate tests.
    (root / "scripts/march/check_installation_evidence.py").write_text(
        'def review(root, data): return {"ready": True}\n'
    )
    (docs / "installation.json").write_text("{}")
    (docs / "serving.json").write_text(json.dumps(data))
    (docs / "card.md").write_text("synthetic model card\n")
    config = root / "configs/release/publication.json"
    config.parent.mkdir(parents=True)
    metadata = {
        "status": "ready",
        "source_repository": "https://example.com/team/code",
        "maintainer_contact": "https://example.com/team/code/issues",
        "release_tag": "v0.1.0",
        "code_license": "Apache-2.0",
        "models": [{"name": "fixture", "card": "docs/march/card.md"}],
        "acceptance": {
            key: "complete"
            for key in (
                "cpu_regression",
                "clean_cuda_build",
                "license_review",
                "credential_review",
                "batch_dp_long_context",
                "numerical_limits_documented",
                "operator_correctness",
            )
        },
        "evidence": {
            "batch_dp_long_context": "docs/march/serving.json",
            "fresh_installation": "docs/march/installation.json",
        },
    }
    config.write_text(json.dumps(metadata))
    return config, metadata


@pytest.mark.parametrize(
    "change",
    [
        "valid",
        "checkbox_only",
        "outside",
        "invalid_json",
        "changed_runtime",
        "bad_evidence_map",
        "bad_evidence_path",
    ],
)
def test_real_publication_cli_cannot_override_missing_or_stale_evidence(
    evidence, change
):
    _, root, data = evidence
    config, metadata = publication_fixture(root, data)
    if change == "checkbox_only":
        (root / "docs/march/serving.json").unlink()
    elif change == "outside":
        outside = root.parent / (root.name + "-outside.json")
        outside.write_text(json.dumps(data))
        metadata["evidence"]["batch_dp_long_context"] = "../" + outside.name
        config.write_text(json.dumps(metadata))
    elif change == "bad_evidence_map":
        metadata["evidence"] = None
        config.write_text(json.dumps(metadata))
    elif change == "bad_evidence_path":
        metadata["evidence"]["batch_dp_long_context"] = []
        config.write_text(json.dumps(metadata))
    elif change == "invalid_json":
        (root / "docs/march/serving.json").write_text("{")
    elif change == "changed_runtime":
        (root / "march/runtime.py").write_text("# changed after GPU run\n")
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/march/check_publication.py"),
            "--root",
            str(root),
            "--scope",
            "code",
        ],
        text=True,
        capture_output=True,
    )
    report = json.loads(result.stdout)
    assert (result.returncode == 0) == (change == "valid")
    assert report["ready"] == (change == "valid")
    if change != "valid":
        assert "acceptance.batch_dp_long_context.evidence" in report["pending"]

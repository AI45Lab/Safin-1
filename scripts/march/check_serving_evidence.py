#!/usr/bin/env python3
"""Verify source-bound evidence for the first-release two-GPU API smoke.

This gate covers the recorded Python runtime and API response checks. It does
not certify native source builds, numerical parity or maximum throughput.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVIDENCE = "docs/march/serving-acceptance-20261017.json"
RUNTIME_DIRECTORIES = ("march", "third_party/vllm/vllm")
RUNTIME_FILES = ("scripts/march/serve_vllm.py", "scripts/march/smoke_serving_matrix.py")
CASES = (("4B", 1, 2), ("35B-A3B", 2, 1), ("35B-A3B", 1, 2))
SERIAL_LENGTHS = [511, 512, 513, 4096, 8000]
CONCURRENT_LENGTHS = [511, 512, 513, 1024, 2048, 4096, 8000, 512]


def runtime_paths(root):
    root = Path(root).resolve()
    groups = {}
    for name in RUNTIME_DIRECTORIES:
        paths = sorted((root / name).rglob("*.py"))
        if not paths:
            raise ValueError(f"missing serving runtime: {name}")
        groups[name] = paths
    groups["serving_entrypoints"] = [root / name for name in RUNTIME_FILES]
    for paths in groups.values():
        for path in paths:
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError(
                    "serving runtime sources must stay inside the checkout"
                )
    return groups


def fingerprints(root):
    root = Path(root).resolve()
    result = {}
    for name, paths in runtime_paths(root).items():
        files = {
            path.relative_to(root).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in paths
        }
        encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        result[name] = {
            "files": len(files),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value, length=64):
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{%d}" % length, value) is not None
    )


def case(row):
    require(isinstance(row, dict), "case must be an object")
    require(
        type(row.get("tp")) is int and type(row.get("dp")) is int, "invalid TP/DP sizes"
    )
    return row["model"], row["tp"], row["dp"]


def review(root, data):
    errors = []
    try:
        require(isinstance(data, dict), "serving evidence must be an object")
        require(
            type(data.get("serving_evidence_schema_version")) is int
            and data["serving_evidence_schema_version"] == 1,
            "missing serving evidence schema",
        )
        require(data.get("status") == "passed", "serving matrix did not pass")
        require(
            digest(data.get("source_revision"), 40), "missing frozen source revision"
        )
        require(
            data.get("runtime_source_sha256") == fingerprints(root),
            "serving runtime fingerprints changed or are incomplete; rerun GPU acceptance",
        )
        require(
            data.get("profile") == "graph" and data.get("prefix_cache") is False,
            "evidence must use the default Graph profile without prefix caching",
        )
        require(
            data.get("offline") is True
            and data.get("hardware") == "two CUDA GPUs"
            and type(data.get("gpu_count")) is int and data["gpu_count"] == 2,
            "missing offline two-GPU execution evidence",
        )
        for key, expected in (
            ("max_model_len", 8192),
            ("max_num_seqs", 32),
            ("submitted_http_concurrency", 8),
            ("max_new_tokens", 8),
            ("seed", 123),
        ):
            require(
                type(data.get(key)) is int and data[key] == expected,
                f"wrong recipe: {key}",
            )
        require(
            type(data.get("temperature")) in (int, float) and data["temperature"] == 0,
            "evidence must use greedy decoding",
        )
        require(
            data.get("serial_prompt_lengths") == SERIAL_LENGTHS
            and data.get("concurrent_prompt_lengths") == CONCURRENT_LENGTHS,
            "missing anchor-boundary or long-prompt requests",
        )
        require(
            data.get("native_build") in ("reused", "source"),
            "missing native-build provenance",
        )
        require(
            data.get("post_run_source_drivers_and_eight_native_hashes_verified")
            is True,
            "post-run source/native verification is missing",
        )
        require(
            data["preflight"]["native_extensions_verified"] == 8,
            "native verification is incomplete",
        )
        results, samples, attempts = (
            data["results"],
            data["api_samples"],
            data["attempts"],
        )
        require(
            isinstance(results, list)
            and isinstance(samples, list)
            and isinstance(attempts, list)
            and attempts,
            "missing matrix results/samples/attempts",
        )
        require(
            [case(row) for row in results] == list(CASES)
            and [case(row) for row in samples] == list(CASES),
            "missing or duplicate TP/DP cases",
        )
        latest = attempts[-1]
        require(
            latest["status"] == "passed"
            and latest["source_revision"] == data["source_revision"]
            and latest["results"] == results,
            "latest attempt does not support the accepted matrix",
        )
        require(
            latest.get("post_run_source_drivers_and_eight_native_hashes_verified")
            is True,
            "latest attempt is missing post-run verification",
        )
        for key in (
            "raw_report_sha256",
            "acceptance_sha256",
            "matrix_log_sha256",
            "source_manifest_sha256",
        ):
            require(digest(latest.get(key)), f"missing receipt fingerprint: {key}")
        total_tokens = 0
        for result, sample in zip(results, samples):
            require(result.get("passed") is True, "a serving case did not pass")
            dispatches = result["graph_dispatches"]
            require(
                isinstance(dispatches, dict)
                and dispatches
                and all(
                    mode in ("FULL", "PIECEWISE") and type(count) is int and count > 0
                    for mode, count in dispatches.items()
                ),
                "missing actual Graph dispatch evidence",
            )
            require(
                type(result["max_graph_requests"]) is int
                and 1 <= result["max_graph_requests"] <= 32,
                "invalid observed worker batch; HTTP concurrency is a separate quantity",
            )
            for key in ("server_log_sha256", "event_log_sha256"):
                require(digest(result.get(key)), f"missing case fingerprint: {key}")
            case_tokens = 0
            for group, lengths in (
                ("serial", SERIAL_LENGTHS),
                ("concurrent", CONCURRENT_LENGTHS),
            ):
                responses = sample[group]
                require(
                    isinstance(responses, list) and len(responses) == len(lengths),
                    "missing API response samples",
                )
                for response, length in zip(responses, lengths):
                    require(
                        response["input_tokens"] == length,
                        "wrong prompt length in response sample",
                    )
                    ids, logprobs, usage = (
                        response["generated_ids"],
                        response["token_logprobs"],
                        response["usage"],
                    )
                    require(
                        isinstance(ids, list)
                        and 1 <= len(ids) <= 8
                        and all(type(token) is int and token >= 0 for token in ids),
                        "invalid generated IDs",
                    )
                    require(
                        isinstance(logprobs, list)
                        and len(logprobs) == len(ids)
                        and all(
                            type(value) in (int, float) and math.isfinite(value)
                            for value in logprobs
                        ),
                        "missing or nonfinite selected-token logprobs",
                    )
                    require(
                        type(usage["completion_tokens"]) is int
                        and usage["completion_tokens"] == len(ids)
                        and usage["prompt_tokens"] == length,
                        "token IDs/logprobs/usage counts disagree",
                    )
                    case_tokens += len(ids)
            require(
                result["api_requests_checked"] == 13
                and result["completion_tokens"] == case_tokens,
                "case summary disagrees with API samples",
            )
            total_tokens += case_tokens
        require(
            data["case_count_passed"] == latest["case_count_passed"] == 3
            and data["api_requests_checked"] == latest["api_requests_checked"] == 39
            and data["completion_tokens"]
            == latest["completion_tokens"]
            == total_tokens,
            "matrix summary disagrees with API samples",
        )
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        OverflowError,
    ) as error:
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

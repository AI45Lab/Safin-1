#!/usr/bin/env python3
"""Offline serving smoke for TP/DP, concurrency and memory-anchor boundaries.

Uses two GPUs and synthetic token prompts. This is not an HF/vLLM logits parity
check or a claim of fastest throughput. Logs and measurements go to --output.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
CASES = (("4B", 1, 2), ("35B-A3B", 2, 1), ("35B-A3B", 1, 2))
SINGLE_GPU_CASES = (("4B", 1, 1), ("35B-A3B", 1, 1))
SERIAL_LENGTHS = (511, 512, 513, 4096, 8000)
BATCH_LENGTHS = (511, 512, 513, 1024, 2048, 4096, 8000, 512)


@contextmanager
def running_server(command, log, env=None):
    with log.open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=env)
        try:
            yield process
        finally:
            # Kill only the group created here, including TP/DP subprocesses.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)


def wait_for_server(base_url, process, log):
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server startup failed ({process.returncode}); see {log}")
        try:
            with urlopen(base_url + "/health", timeout=5) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(2)
    raise TimeoutError(f"Server startup timed out; see {log}")


def validate_completion(data, length):
    choice, usage = data["choices"][0], data["usage"]
    count = usage["completion_tokens"]
    if (not isinstance(choice["text"], str) or not isinstance(count, int)
            or not 1 <= count <= 8 or usage["prompt_tokens"] != length):
        raise ValueError("Completion token counts do not match the request")
    ids = choice.get("token_ids")
    if (not isinstance(ids, list) or len(ids) != count
            or any(type(token) is not int or token < 0 for token in ids)):
        raise ValueError("Completion is missing generated token IDs")
    logprobs = choice["logprobs"]["token_logprobs"]
    if (len(logprobs) != count
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in logprobs)):
        raise ValueError("Completion is missing finite generated log probabilities")
    return {"input_tokens": length, "usage": usage, "generated_ids": ids,
            "token_logprobs": logprobs, "finish_reason": choice["finish_reason"], "text": choice["text"]}


def generate(base_url, length):
    body = {"model": "Safin-1", "prompt": [101 + i % 13 for i in range(length)],
            "temperature": 0, "seed": 123, "max_tokens": 8, "logprobs": 1,
            "return_token_ids": True}
    request = Request(base_url + "/v1/completions", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
    started = time.monotonic()
    with urlopen(request, timeout=300) as response:
        data = json.load(response)
    row = validate_completion(data, length)
    row["seconds"] = time.monotonic() - started
    return row


def read_graph_evidence(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    counts = Counter(row.get("event") for row in rows)
    hits = [row for row in rows if row.get("event") == "hsd_graphable_decode_hit"
            and row.get("cudagraph_mode") in ("FULL", "PIECEWISE")]
    if not hits:
        raise ValueError("No actual Graph decode dispatch was recorded")
    return {"event_file": path.name, "event_counts": dict(counts),
            "graph_dispatches": dict(Counter(row["cudagraph_mode"] for row in hits)),
            "max_graph_requests": max(row["requests"] for row in hits)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-4b", required=True)
    parser.add_argument("--model-35b-a3b", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--single-gpu", action="store_true", help="Run TP1/DP1 for both models; does not certify TP/DP parallel serving")
    parser.add_argument("--native-build", choices=("unknown", "source", "reused"), default="unknown")
    args = parser.parse_args()
    models = {"4B": args.model_4b, "35B-A3B": args.model_35b_a3b}
    cases = SINGLE_GPU_CASES if args.single_gpu else CASES
    if args.dry_run:
        print(json.dumps({"cases": [{"model": models[name], "tp": tp, "dp": dp} for name, tp, dp in cases],
                          "profile": "graph", "prompt_lengths": SERIAL_LENGTHS, "concurrency": 8,
                          "max_new_tokens": 8, "native_build": args.native_build}, indent=2))
        return 0
    import torch
    required_gpus = 1 if args.single_gpu else 2
    if torch.cuda.device_count() < required_gpus:
        parser.error(f"Serving matrix requires at least {required_gpus} visible CUDA GPUs")
    for model in models.values():
        if not (Path(model) / "config.json").is_file():
            parser.error(f"Provide a local checkpoint with config.json: {model}")
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("OFFLINE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_EVALUATE_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[name] = "1"
    records = []
    for family, tp, dp in cases:
        name = f"{family}-tp{tp}-dp{dp}"
        log = args.output / f"{name}-server.log"
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        base_url = f"http://127.0.0.1:{port}"
        command = [sys.executable, str(ROOT / "scripts/march/serve_vllm.py"),
                   "--model", models[family], "--profile", "graph",
                   "--tensor-parallel-size", str(tp), "--data-parallel-size", str(dp),
                   "--max-model-len", "8192", "--max-num-seqs", "32",
                   "--gpu-memory-utilization", "0.85", "--host", "127.0.0.1",
                   "--port", str(port), "--no-enable-log-requests"]
        events = args.output / f"{name}-{time.time_ns()}-events.jsonl"
        server_env = dict(os.environ, MARCH_EVENT_LOG=str(events.resolve()))
        with running_server(command, log, server_env) as server:
            wait_for_server(base_url, server, log)
            generate(base_url, 128)
            serial = [generate(base_url, length) for length in SERIAL_LENGTHS]
            started = time.monotonic()
            with ThreadPoolExecutor(max_workers=8) as pool:
                concurrent = list(pool.map(lambda length: generate(base_url, length), BATCH_LENGTHS))
            elapsed = time.monotonic() - started
            record = {"model": family, "tp": tp, "dp": dp, "profile": "graph",
                      "serial": serial, "concurrent": concurrent, "concurrency": 8,
                      "batch_seconds": elapsed,
                      "completion_tokens_per_second": sum(row["usage"]["completion_tokens"] for row in concurrent) / elapsed,
                      "passed": True, "native_build": args.native_build,
                      "graph_evidence": read_graph_evidence(events)}
            records.append(record)
            (args.output / "serving-results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
            print(json.dumps(record, ensure_ascii=False), flush=True)
        time.sleep(5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

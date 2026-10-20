#!/usr/bin/env python3
"""GPU acceptance for MARCH prefix reuse and Graph V3 decode."""

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import time

from serve_vllm import build_launch, read_config


def compare(reference, candidate, atol, rtol):
    if reference["ids"] != candidate["ids"]:
        raise AssertionError("Prefix hit changed greedy generated IDs")
    if len(reference["logprobs"]) != len(reference["ids"]) or len(candidate["logprobs"]) != len(candidate["ids"]):
        raise AssertionError("Incomplete selected-token logprobs")
    errors = [abs(a - b) for a, b in zip(reference["logprobs"], candidate["logprobs"])]
    if any(not math.isfinite(a) or not math.isfinite(b) or abs(a - b) > atol + rtol * abs(a)
           for a, b in zip(reference["logprobs"], candidate["logprobs"])):
        raise AssertionError(f"Prefix-hit selected logprobs exceeded tolerance; max error={max(errors, default=0)}")
    return max(errors, default=0)


def compare_vocabulary(reference, candidate, atol, rtol):
    import numpy as np
    if len(reference) != len(candidate):
        raise AssertionError("Full-vocabulary logprob step count changed")
    error = 0.0
    for (a_ids, a), (b_ids, b) in zip(reference, candidate):
        if a_ids != b_ids:
            raise AssertionError("Full-vocabulary token layout changed")
        a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
        if not np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False):
            raise AssertionError("Full-vocabulary logprobs exceeded tolerance")
        finite = np.isfinite(a) & np.isfinite(b)
        error = max(error, float(np.max(np.abs(a[finite] - b[finite]), initial=0)))
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", choices=("graph", "eager"), default="graph")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--lengths", default="511,512,513,1024,1025,2049")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--full-logprobs", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        lengths = sorted(set(int(x) for x in args.lengths.split(",")))
        if min(lengths) < 2 or args.max_tokens < 2 or args.tensor_parallel_size < 1 or not 1 <= args.batch_size <= 32:
            raise ValueError("positive lengths/TP, max-tokens >= 2, and batch-size 1..32 are required")
        if not all(math.isfinite(x) and x >= 0 for x in (args.atol, args.rtol)):
            raise ValueError("tolerances must be finite and non-negative")
        model_type, config = read_config(args.model, with_config=True)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"accepted": False, "model_type": model_type, "profile": args.profile,
              "batch_size": args.batch_size, "max_tokens": args.max_tokens, "seed": 0, "tp": args.tensor_parallel_size,
              "atol": args.atol, "rtol": args.rtol, "full_logprobs": args.full_logprobs, "cases": []}

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    with tempfile.TemporaryDirectory(prefix="march-prefix-check-") as directory:
        events = Path(directory) / "events.jsonl"
        args.experimental_prefix_cache, args.prefix_cache_max_bytes = True, 2 * 1024**3
        args.max_num_seqs = 1 if args.batch_size == 1 else (16 if args.batch_size <= 16 else 32)
        args.max_model_len = max(lengths) + args.max_tokens + 32
        args.served_model_name, args.data_parallel_size, args.revision = "Safin-1", 1, None
        command, env, controls = build_launch(args, [], model_type, dict(os.environ, MARCH_EVENT_LOG=str(events)), config)
        for key in list(os.environ):
            if key.startswith(("VLLM_HSD_", "HSD_")) or key == "VLLM_FORCE_BATCH_INVARIANT_GEMMA_RMSNORM":
                del os.environ[key]
        os.environ.update(env)
        import torch
        import vllm
        from vllm import LLM, SamplingParams

        root = Path(__file__).resolve().parents[2]
        if not Path(vllm.__file__).resolve().is_relative_to(root / "third_party/vllm/vllm"):
            raise RuntimeError("Use this repository's vLLM fork via PYTHONPATH")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPUs and locally built fork extensions are required")
        report.update(torch=torch.__version__, vllm=vllm.__version__,
                      gpu=torch.cuda.get_device_name(0), controls=controls)
        extra = {}
        if args.profile == "graph":
            extra["compilation_config"] = json.loads(command[command.index("--compilation-config") + 1])
        if "35b" in model_type:
            extra["moe_backend"] = "triton"
        engine = LLM(model=args.model, trust_remote_code=True, dtype="bfloat16",
                     enforce_eager=args.profile == "eager", tensor_parallel_size=args.tensor_parallel_size,
                     max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len,
                     gpu_memory_utilization=args.gpu_memory_utilization, enable_prefix_caching=True,
                     async_scheduling=False, mamba_cache_mode="none", language_model_only=True,
                     additional_config=json.loads(command[command.index("--additional-config") + 1]),
                     max_logprobs=-1 if args.full_logprobs else 20, **extra)
        samples = SamplingParams(temperature=0, seed=0, max_tokens=args.max_tokens, ignore_eos=True,
                                 logprobs=-1 if args.full_logprobs else 1)
        piece = engine.get_tokenizer().encode("MARCH memory anchors read earlier context.", add_special_tokens=False)
        if len(set(piece)) < 2:
            raise RuntimeError("Probe tokenization needs two distinct tokens")
        tail = next(token for token in piece if token != piece[0])
        baseline_storage = None

        def records():
            return [json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []

        def generate(ids):
            start = time.perf_counter()
            outputs = engine.generate([{"prompt_token_ids": ids} for _ in range(args.batch_size)], samples, use_tqdm=False)
            duration = time.perf_counter() - start
            if events.exists():
                args.output.with_suffix(".events.jsonl").write_text(events.read_text())
            results = []
            for request in outputs:
                out = request.outputs[0]
                result = {"ids": list(out.token_ids),
                          "logprobs": [float(lp[token].logprob) for token, lp in zip(out.token_ids, out.logprobs)]}
                if args.full_logprobs:
                    result["vocabulary"] = [(sorted(lp), [float(lp[token].logprob) for token in sorted(lp)])
                                            for lp in out.logprobs]
                results.append(result)
            return results, duration

        def status():
            nonlocal baseline_storage
            values = engine.collective_rpc("get_hsd_prefix_cache_status")
            storage = [value["graph_storage"] for value in values]
            if args.profile == "graph":
                if not all(storage):
                    raise AssertionError("Graph buffers were not preallocated")
                if baseline_storage is None:
                    baseline_storage = storage
                if storage != baseline_storage:
                    raise AssertionError("Reset/restore/weight reload replaced captured Graph storage")
            return values

        def reset():
            if not engine.reset_prefix_cache():
                raise RuntimeError("Native/HSD cache reset failed")
            values = status()
            if any(s["entries"] or s["bytes"] for s in values):
                raise AssertionError("Worker snapshots survived an engine reset")
            return values

        def compare_batch(reference, candidate):
            if len(reference) != args.batch_size or len(candidate) != args.batch_size:
                raise AssertionError("Incomplete rollout batch")
            error, full_error = 0.0, None
            for ref, value in zip(reference, candidate):
                error = max(error, compare(ref, value, args.atol, args.rtol))
                if args.full_logprobs:
                    full_error = max(full_error or 0, compare_vocabulary(ref["vocabulary"], value["vocabulary"], args.atol, args.rtol))
            return error, full_error

        try:
            for length in lengths:
                prefix = [piece[i % len(piece)] for i in range(length - 1)]
                target, primer = prefix + [tail], prefix + [piece[0]]
                reset()
                reference, cold_time = generate(target)
                reset()
                generate(primer)
                offset = len(records())
                candidate, warm_time = generate(target)
                observed = records()[offset:]
                hits = [r for r in observed if r.get("event") == "hsd_prefix_cache_hit"]
                restores = [r for r in observed if r.get("event") == "hsd_prefix_snapshot_restore"]
                replays = [r for r in observed if r.get("event") == "march_cuda_graph_replay"]
                graph_restores = [r for r in observed if r.get("event") == "hsd_prefix_graph_restore_ready"]
                anchor = int(config.get("text_config", config).get("anchor_chunk_size", 512))
                alignment = math.lcm(anchor, 1024) if "35b" in model_type else anchor
                if length - 1 >= alignment and (not hits or not restores):
                    raise AssertionError(f"No actual prefix hit/restore for eligible length {length}")
                if args.profile == "graph" and (not replays or (hits and not graph_restores)):
                    raise AssertionError(f"Prefix restore did not lead to actual Graph V3 replay at length {length}")
                # Preserve selected distributions even if a strict comparison
                # fails; raw vocabulary tensors need not be copied into JSON.
                report["pending_case"] = {
                    "prompt_tokens": length,
                    "reference": [{k: v for k, v in r.items() if k != "vocabulary"} for r in reference],
                    "candidate": [{k: v for k, v in r.items() if k != "vocabulary"} for r in candidate],
                    "scheduler_hits": len(hits), "restore_events": len(restores),
                    "graph_replays": len(replays),
                }
                save()
                error, full_error = compare_batch(reference, candidate)
                report.pop("pending_case")
                status()
                case = {"prompt_tokens": length, "max_selected_logprob_error": error,
                        "max_vocabulary_logprob_error": full_error, "cold_seconds": cold_time, "warm_seconds": warm_time,
                        "scheduler_hits": len(hits), "restore_events": len(restores), "graph_replays": len(replays),
                        "reference": [{k: v for k, v in r.items() if k != "vocabulary"} for r in reference],
                        "generated_ids": [r["ids"] for r in candidate]}
                if args.reference_report:
                    previous = json.loads(args.reference_report.read_text())
                    match = next(c for c in previous["cases"] if c["prompt_tokens"] == length)
                    case["cross_profile_selected_error"] = max(compare(a, b, args.atol, args.rtol)
                        for a, b in zip(match["reference"], reference))
                report["cases"].append(case)
                save()
                print(json.dumps({k: v for k, v in case.items() if k not in {"reference", "generated_ids"}}), flush=True)

            weights = engine.collective_rpc("get_hsd_anchor_probe_weight")
            probe = weights[0]
            name = probe["name"]
            original = torch.tensor(probe["values"], dtype=getattr(torch, probe["dtype"]))
            before = status()
            try:
                engine.collective_rpc("reload_weights", kwargs={"weights_payload": [{"name": name, "values": (original + 0.125).tolist()}],
                                                                 "is_checkpoint_format": False})
                after = status()
                for previous, current in zip(before, after):
                    if current["entries"] or current["bytes"]:
                        raise AssertionError("Snapshots survived a kernel-format weight update")
                    if any(b <= a for a, b in zip(previous["generations"], current["generations"])):
                        raise AssertionError("Weight update did not advance cache generation")
                reference, _ = generate(target)
                reset()
                generate(primer)
                compare_batch(reference, generate(target)[0])
            finally:
                engine.collective_rpc("reload_weights", kwargs={"weights_payload": [{"name": name, "values": original.tolist()}],
                                                                 "is_checkpoint_format": False})
            report["final_cache"] = reset()
            observed = records()
            report.update(scheduler_hits=sum(r.get("event") == "hsd_prefix_cache_hit" for r in observed),
                          restore_events=sum(r.get("event") == "hsd_prefix_snapshot_restore" for r in observed),
                          graph_replays=sum(r.get("event") == "march_cuda_graph_replay" for r in observed),
                          graph_storage_preserved=args.profile == "graph", weight_update_invalidated=True,
                          cache_metadata=[r for r in observed if r.get("event") == "hsd_prefix_snapshot_restore"])
            if not report["scheduler_hits"] or not report["restore_events"]:
                raise AssertionError("No actual scheduler hit and worker restore were observed")
            if args.profile == "graph" and not any(r.get("reason") == "anchor_boundary" for r in observed):
                raise AssertionError("Graph boundary fallback was not exercised; include boundary lengths")
            report["accepted"] = True
            report["limits"] = "Configured batch/profile only; TP/DP, long context, abort/preemption and full RL need separate acceptance."
            save()
            print(json.dumps({k: v for k, v in report.items() if k not in {"cases", "cache_metadata", "controls"}}, indent=2))
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
            save()
            raise
        finally:
            if events.exists():
                args.output.with_suffix(".events.jsonl").write_text(events.read_text())
            engine.llm_engine.engine_core.shutdown(timeout=10)


if __name__ == "__main__":
    main()

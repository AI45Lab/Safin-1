#!/usr/bin/env python3
"""Run bounded real-model CPT/SFT, MCore resume and strict HF reload offline."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def commands(args):
    baseline = args.output / "continuous"
    resumed = args.output / "resumed"
    common = ["--train_iters", str(args.steps), "--save_steps", str(args.resume_step),
              "--logging_steps", "1", "--eval_steps", "100", "--eval_iters", "0",
              "--add_version", "false", "--micro_batch_size", "1",
              "--global_batch_size", str(args.nproc), "--save_safetensors", "true",
              "--no_save_optim", "false", "--no_save_rng", "false",
              "--no_load_optim", "false", "--no_load_rng", "false", "--async_save", "false",
              "--split_dataset_ratio", "0", "--truncation_strategy", "right",
              "--dataloader_num_workers", "0", "--dataset_num_proc", "1",
              "--attention_backend", "unfused", "--tuner_type", args.tuner,
              "--seed", "123", "--lr", "0.00001", "--lr_decay_style", "constant",
              "--lr_decay_iters", str(args.steps), "--external_plugins",
              str(ROOT / "scripts/march/_training_resume_probe.py")]
    if args.tuner == "lora":
        common += ["--target_modules", "linear_qkv", "linear_proj", "--lora_rank", "8",
                   "--merge_lora", "true"]
    launcher = ["bash", str(ROOT / f"scripts/march/train_{args.task}.sh")]
    source_checkpoint = baseline / f"checkpoint-{args.resume_step}"
    resume_key = "--mcore_model" if args.tuner == "full" else "--mcore_adapter"
    return [(baseline, launcher + common + ["--finetune", "true"]),
            (resumed, launcher + common + ["--finetune", "false", "--checkpoint_lifecycle", "resume",
                                           resume_key, str(source_checkpoint)])]


def validate_resume_checkpoint(path, step):
    tracker = path / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or tracker.read_text().strip() != str(step):
        raise ValueError(f"checkpoint tracker does not retain training step {step}: {path}")
    directory = path / f"iter_{step:07d}"
    if not (directory / "common.pt").is_file() or not (directory / ".metadata").is_file():
        raise ValueError(f"missing resumable MCore checkpoint files: {directory}")


def read_probe(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def validate_restoration(continuous, resumed, step, nproc):
    results = []
    for rank in range(nproc):
        saved = [row for row in read_probe(continuous / "probe" / f"rank-{rank}.jsonl")
                 if row["event"] == "before_save" and row["iteration"] == step]
        loaded = [row for row in read_probe(resumed / "probe" / f"rank-{rank}.jsonl")
                  if row["event"] == "after_load"]
        if len(saved) != 1 or len(loaded) != 1:
            raise ValueError(f"rank {rank}: missing or duplicate save/load evidence")
        fields = ("iteration", "consumed_train_samples", "optimizer_sample_sha256", "rng_sha256", "scheduler")
        mismatches = [key for key in fields if saved[0][key] != loaded[0][key]]
        if mismatches:
            raise ValueError(f"rank {rank}: restored state mismatch: {mismatches}")
        if loaded[0]["consumed_train_samples"] != step * nproc:
            raise ValueError("restored data cursor does not match the source step")
        results.append({"rank": rank, "matched_fields": list(fields), "loaded": loaded[0]})
    return results


def read_metrics(path):
    rows = [json.loads(line) for line in (path / "logging.jsonl").read_text().splitlines()]
    metrics = [row for row in rows if "loss" in row]
    for row in metrics:
        for key in ("loss", "grad_norm", "learning_rate"):
            if not isinstance(row.get(key), (float, int)) or not math.isfinite(row[key]):
                raise ValueError(f"missing/non-finite training metric: {key}")
    return metrics


def compare_weights(first, second, atol, rtol):
    import torch
    from safetensors import safe_open

    def weight_map(path):
        index = path / "model.safetensors.index.json"
        if index.is_file():
            return json.loads(index.read_text())["weight_map"]
        with safe_open(path / "model.safetensors", framework="pt", device="cpu") as handle:
            return {key: "model.safetensors" for key in handle.keys()}

    a, b = weight_map(first), weight_map(second)
    if set(a) != set(b):
        raise ValueError("continuous/resumed exports have different weight keys")
    exact = 0
    max_error = 0.0
    outside_tolerance = []
    with ExitStack() as stack:
        handles_a = {file: stack.enter_context(safe_open(first / file, framework="pt", device="cpu"))
                     for file in set(a.values())}
        handles_b = {file: stack.enter_context(safe_open(second / file, framework="pt", device="cpu"))
                     for file in set(b.values())}
        for key in sorted(a):
            x, y = handles_a[a[key]].get_tensor(key), handles_b[b[key]].get_tensor(key)
            if x.shape != y.shape or x.dtype != y.dtype:
                raise ValueError(f"weight shape/dtype changed after resume: {key}")
            x, y = x.reshape(-1), y.reshape(-1)
            identical = True
            valid = True
            for offset in range(0, x.numel(), 1024 * 1024):
                xx, yy = x[offset:offset + 1024 * 1024], y[offset:offset + 1024 * 1024]
                if not torch.isfinite(xx).all() or not torch.isfinite(yy).all():
                    raise ValueError(f"non-finite exported weights: {key}")
                if not torch.equal(xx, yy):
                    identical = False
                    error = (xx.float() - yy.float()).abs().max().item()
                    max_error = max(max_error, error)
                    valid &= torch.allclose(xx, yy, atol=atol, rtol=rtol)
            exact += int(identical)
            if not valid:
                outside_tolerance.append(key)
    return {"weight_tensors": len(a), "exact_tensors": exact, "max_abs_error": max_error,
            "atol": atol, "rtol": rtol, "outside_tolerance": outside_tolerance}


def training_environment(args):
    # MCore's checkpoint writer creates a multiprocessing Manager even for
    # synchronous saves. Its Unix socket must fit the platform path limit.
    env = os.environ.copy()
    env.update({"OFFLINE": "1", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                "HF_EVALUATE_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "TMPDIR": "/tmp", "MODEL": str(args.model), "DATASET": str(args.output / "train.jsonl"),
                "PYTHON_BIN": sys.executable, "NPROC_PER_NODE": str(args.nproc),
                "TENSOR_MODEL_PARALLEL_SIZE": str(args.tp),
                "EXPERT_MODEL_PARALLEL_SIZE": str(args.ep), "MAX_LENGTH": str(args.max_length)})
    env.pop("SWIFT_RELEASE_OPTIM_BEFORE_FINAL_SAVE", None)
    env.pop("MEGATRON_SKIP_FINAL_SAVE", None)
    return env


def storage_preflight(args):
    """Budget all retained exports/state without loading any weight tensors.

    This is an estimate, not a reservation or an upper bound. Full tuning budgets
    BF16 model + FP32 master/moments and HF exports; LoRA mainly retains HF exports.
    A measured configuration can supply an explicit total budget in GiB.
    """
    index = args.model / "model.safetensors.index.json"
    shards = (set(json.loads(index.read_text())["weight_map"].values())
              if index.is_file() else {"model.safetensors"})
    weight_bytes = sum((args.model / shard).stat().st_size for shard in shards)
    # Both processes retain periodic and final saves; resume does not resave its input.
    saves_per_run = (args.steps + args.resume_step - 1) // args.resume_step
    checkpoints = 2 * saves_per_run - 1
    explicit = args.storage_budget_gib
    budget = math.ceil(explicit * 1024 ** 3) if explicit is not None else math.ceil(
        weight_bytes * checkpoints * (8 if args.tuner == "full" else 1.1) * 1.25)
    filesystem = args.output.parent
    while not filesystem.exists():
        filesystem = filesystem.parent
    free = shutil.disk_usage(filesystem).free
    report = {"kind": "explicit" if explicit is not None else "estimated",
              "weight_bytes": weight_bytes, "retained_checkpoints": checkpoints,
              "required_bytes": budget, "free_bytes": free, "filesystem": str(filesystem)}
    if free < budget:
        raise ValueError(f"Insufficient output storage: {free / 1024 ** 3:.1f} GiB free, "
                         f"{budget / 1024 ** 3:.1f} GiB {report['kind']} budget for "
                         f"{checkpoints} retained checkpoints. Select an output filesystem "
                         "with sufficient capacity; an explicit measured budget can use "
                         "--storage-budget-gib.")
    return report


def run(args):
    if args.steps < 3 or not 0 < args.resume_step < args.steps:
        raise ValueError("Require at least 3 steps and 0 < resume-step < steps")
    if min(args.nproc, args.tp, args.ep) < 1 or args.nproc % (args.tp * args.ep):
        raise ValueError("Require positive parallel sizes and nproc divisible by tp * ep")
    if args.max_length < 513:
        raise ValueError("Use max-length >= 513 to exercise an anchor boundary")
    if any(not math.isfinite(value) or value < 0 for value in (args.atol, args.rtol)):
        raise ValueError("Weight tolerances must be finite and nonnegative")
    if args.storage_budget_gib is not None and (
            not math.isfinite(args.storage_budget_gib) or args.storage_budget_gib <= 0):
        raise ValueError("Storage budget must be finite and positive")
    plan = commands(args)
    if args.dry_run:
        print(json.dumps({"task": args.task, "tuner": args.tuner, "steps": args.steps,
                          "resume_step": args.resume_step, "model": str(args.model),
                          "nproc": args.nproc, "tp": args.tp, "ep": args.ep,
                          "storage_budget_gib": args.storage_budget_gib,
                          "disk_preflight_deferred": True,
                          "cases": [{"output": str(out), "command": cmd} for out, cmd in plan]}, indent=2))
        return
    if args.output.exists():
        raise ValueError("Use a new output directory; existing runs are never overwritten")
    import torch
    from verify_checkpoint import inspect_checkpoint, resolve_training_checkpoint
    inspect_checkpoint(args.model)
    storage = storage_preflight(args)
    if torch.cuda.device_count() < args.nproc:
        raise ValueError("Insufficient visible CUDA devices")
    args.output.mkdir(parents=True)
    (args.output / "storage-preflight.json").write_text(json.dumps(storage, indent=2) + "\n")
    config = json.loads((args.model / "config.json").read_text())
    env = training_environment(args)
    with (args.output / "train.jsonl").open("w") as stream:
        for i in range(max(16, args.steps * args.nproc)):
            content = f"Public example {i}. " + "Memory anchors preserve historical information. " * 240
            row = {"text": content} if args.task == "cpt" else {
                "messages": [{"role": "user", "content": f"Explain memory anchors in example {i}."},
                             {"role": "assistant", "content": content}]}
            stream.write(json.dumps(row) + "\n")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for folder in ("march", "swift", "scripts/march") for p in (ROOT / folder).rglob("*.py")}
    (args.output / "source.json").write_text(json.dumps({"commit": source, "files": hashes}, indent=2))
    for out, command in plan:
        if out.name == "resumed":
            validate_resume_checkpoint(plan[0][0] / f"checkpoint-{args.resume_step}", args.resume_step)
        env.update({"OUTPUT_DIR": str(out), "MARCH_RESUME_PROBE_DIR": str(out / "probe")})
        with (args.output / f"{out.name}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    continuous, resumed = plan[0][0], plan[1][0]
    report = {"task": args.task, "tuner": args.tuner, "steps": args.steps,
              "resume_step": args.resume_step, "tp": args.tp, "ep": args.ep, "nproc": args.nproc,
              "max_length": args.max_length, "commit": source, "storage_preflight": storage,
              "restoration": validate_restoration(continuous, resumed, args.resume_step, args.nproc)}
    before, after = read_metrics(continuous), read_metrics(resumed)
    expected = [f"{i}/{args.steps}" for i in range(1, args.steps + 1)]
    if [r["iteration"] for r in before] != expected:
        raise ValueError("continuous run did not complete every requested iteration")
    if [r["iteration"] for r in after] != expected[args.resume_step:]:
        raise ValueError("resume did not start at the next iteration")
    report.update({"continuous_metrics": before, "resumed_metrics": after,
                   "metric_abs_differences": [
                       {"iteration": a["iteration"], **{
                           k: abs(a[k] - b[k]) for k in ("loss", "grad_norm", "learning_rate")}}
                       for a, b in zip(before[args.resume_step:], after)]})
    paths = [resolve_training_checkpoint(out, args.steps) for out, _ in plan]
    runtime = {k: v for k, v in config["text_config"].items() if k.startswith(("anchor_", "hsd_", "hist_"))}
    for path in paths:
        saved = json.loads((path / "config.json").read_text())
        if saved["model_type"] != config["model_type"]:
            raise ValueError("model type changed during training export")
        if any(saved["text_config"].get(k) != v for k, v in runtime.items()):
            raise ValueError("saved MARCH runtime differs from the input checkpoint")
        inspect_checkpoint(path)
    report["runtime_fields_preserved"] = len(runtime)
    report["weights"] = compare_weights(*paths, atol=args.atol, rtol=args.rtol)
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["weights"]["outside_tolerance"]:
        raise ValueError("resumed weights differ from continuous training; see validation.json")
    reload_report = args.output / "hf-reload.json"
    subprocess.run([sys.executable, str(ROOT / "scripts/march/verify_checkpoint.py"), str(paths[1]),
                    "--load-weights", "--report", str(reload_report)], cwd=ROOT, env=env, check=True)
    report["hf_reload"] = json.loads(reload_report.read_text())
    report["status"] = "passed"
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=["cpt", "sft"], required=True)
    parser.add_argument("--tuner", choices=["full", "lora"], default="full")
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--resume-step", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--storage-budget-gib", type=float,
                        help="Measured total output budget in GiB; default estimates all retained state/exports")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.model = args.model.resolve()
    args.output = args.output.resolve()
    run(args)


if __name__ == "__main__":
    main()

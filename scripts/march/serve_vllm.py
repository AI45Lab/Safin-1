#!/usr/bin/env python3
"""Serve Safin-1 with the shared MARCH profile, or print an offline dry-run."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from march.inference.vllm_profile import (  # noqa: E402,F401
    MODEL_TYPES, GRAPH_SWITCHES, build_launch, clean_environment, positive, read_config,
)

def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", default=os.environ.get("MODEL"))
    parser.add_argument("--revision", default=os.environ.get("MODEL_REVISION"))
    parser.add_argument("--served-model-name", default=os.environ.get("SERVED_MODEL_NAME", "Safin-1"))
    parser.add_argument("--profile", choices=("graph", "eager"),
                        default=os.environ.get("MARCH_INFERENCE_PROFILE", "graph"))
    parser.add_argument("--enforce-eager", action="store_true", help="Alias for --profile eager")
    parser.add_argument("--experimental-prefix-cache", action="store_true",
                        default=os.environ.get("MARCH_EXPERIMENTAL_PREFIX_CACHE") == "1",
                        help="Opt into HSD snapshot reuse with eager or Graph V3 decode")
    parser.add_argument("--prefix-cache-max-bytes", type=positive,
                        default=os.environ.get("MARCH_PREFIX_CACHE_MAX_BYTES", str(2 * 1024**3)))
    for flag, names, default in (
        ("tensor-parallel-size", ("TENSOR_PARALLEL_SIZE", "TP_SIZE"), "1"),
        ("data-parallel-size", ("DATA_PARALLEL_SIZE", "DP_SIZE"), "1"),
        ("max-model-len", ("MAX_MODEL_LEN",), "8192"),
        ("max-num-seqs", ("MAX_NUM_SEQS",), "32"),
    ):
        parser.add_argument("--" + flag, type=positive,
                            default=next((os.environ[n] for n in names if n in os.environ), default))
    parser.add_argument("--gpu-memory-utilization", type=float,
                        default=os.environ.get("GPU_MEMORY_UTILIZATION", "0.85"))
    parser.add_argument("--backend-events", action="store_true",
                        default=os.environ.get("MARCH_BACKEND_EVENTS") == "1",
                        help="Log ordinary paged decode selection/fallback reasons to MARCH_EVENT_LOG")
    parser.add_argument("--dry-run", action="store_true", default=os.environ.get("DRY_RUN") == "1")
    args, extra = parser.parse_known_args()
    if not args.model:
        parser.error("Set MODEL or --model to a Safin-1 checkpoint")
    if args.profile not in ("graph", "eager"):
        parser.error("MARCH_INFERENCE_PROFILE must be graph or eager")
    if args.enforce_eager:
        args.profile = "eager"
    try:
        if os.environ.get("MARCH_EXPERIMENTAL_PREFIX_CACHE", "0") not in {"0", "1"}:
            raise ValueError("MARCH_EXPERIMENTAL_PREFIX_CACHE must be 0 or 1")
        model_type, config = read_config(args.model, args.revision, with_config=True)
        command, env, controls = build_launch(args, extra, model_type, os.environ, config)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps({"model_type": model_type, "profile": args.profile,
                          "prefix_cache": "experimental" if args.experimental_prefix_cache else "off",
                          "command": command, "environment": controls}, indent=2))
        return
    print(f"[march-vllm] model_type={model_type} profile={args.profile} "
          f"tp={args.tensor_parallel_size} dp={args.data_parallel_size} "
          f"max_num_seqs={args.max_num_seqs} "
          f"prefix_cache={'experimental' if args.experimental_prefix_cache else 'off'}", flush=True)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()

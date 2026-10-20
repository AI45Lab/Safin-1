#!/usr/bin/env python3
"""Compare full forward and cached decoding at Safin-1 anchor boundaries."""
import argparse
import json
from pathlib import Path


def check_boundaries(model, input_ids, lengths, atol, rtol):
    import torch
    results = {}
    model.eval()
    with torch.inference_mode():
        for length in lengths:
            prefill = model(input_ids=input_ids[:, :length], use_cache=True, logits_to_keep=1)
            prepared = model.prepare_inputs_for_generation(
                input_ids[:, :length + 1], past_key_values=prefill.past_key_values,
                use_cache=True, logits_to_keep=1)
            cached = model(**prepared).logits[:, -1].float()
            full = model(input_ids=input_ids[:, :length + 1], use_cache=False,
                         logits_to_keep=1).logits[:, -1].float()
            difference = (cached - full).abs()
            results[str(length)] = {
                "max_abs_error": difference.max().item(),
                "mean_abs_error": difference.mean().item(),
                "cached_top1": cached.argmax(dim=-1).tolist(),
                "full_top1": full.argmax(dim=-1).tolist(),
                "allclose": torch.allclose(cached, full, atol=atol, rtol=rtol),
                "greedy_equal": torch.equal(cached.argmax(dim=-1), full.argmax(dim=-1)),
            }
            print(length, results[str(length)], flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--lengths", type=int, nargs="+", default=[511, 512, 513, 1024, 1025])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--atol", type=float, default=0.125)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--input-ids", type=Path, help="JSON list of fixed token IDs; otherwise use synthetic public text")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if min(args.lengths) < 1 or args.atol < 0 or args.rtol < 0:
        parser.error("lengths must be positive and tolerances nonnegative")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(123)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, trust_remote_code=True, local_files_only=True,
        device_map=args.device, dtype=getattr(torch, args.dtype)).eval()
    if args.input_ids:
        ids = json.loads(args.input_ids.read_text())
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.checkpoint, trust_remote_code=True, local_files_only=True)
        ids = tokenizer("A short public training example about memory and anchors. "
                        * (max(args.lengths) + 1))["input_ids"]
    if len(ids) < max(args.lengths) + 1:
        parser.error("input IDs must include at least max(lengths) + 1 tokens")
    ids = ids[:max(args.lengths) + 1]
    tensor = torch.tensor([ids], device=model.device, dtype=torch.long)
    results = check_boundaries(model, tensor, args.lengths, args.atol, args.rtol)
    passed = all(case["allclose"] and case["greedy_equal"] for case in results.values())
    report = {"model_type": model.config.model_type, "model_class": type(model).__module__,
              "dtype": args.dtype, "seed": 123, "atol": args.atol, "rtol": args.rtol,
              "input_ids": ids, "cases": results, "passed": passed}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise SystemExit("Cached decode differs: see the recorded per-boundary errors and greedy IDs")


if __name__ == "__main__":
    main()

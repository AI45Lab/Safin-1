#!/usr/bin/env python3
"""Validate a Safin-1 checkpoint manifest; optionally load all weights on GPU."""
import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def inspect_checkpoint(root):
    from march.safin1.config_utils import is_safin1_config, safin1_remote_code_files
    root = Path(root).resolve()
    config = json.loads((root / "config.json").read_text())
    if not is_safin1_config(config):
        raise ValueError("checkpoint must retain Safin-1 model_type at both config levels")
    text = config["text_config"]
    for key in ("anchor_chunk_size", "anchor_token_num", "hsd_read_block_size", "hsd_use_global_read",
                "hsd_global_read_layers", "hsd_read_topk", "hsd_null_read_gate",
                "anchor_embedding_type", "anchor_embedding_init_from"):
        if key not in text:
            raise ValueError(f"missing explicit runtime field: {key}")
    if text.get("anchor_embedding_init_from") == "bos" and text.get("anchor_bos_token_id") is None:
        raise ValueError("BOS-initialized memory anchors require anchor_bos_token_id")
    for name in safin1_remote_code_files(config["model_type"]):
        if not (root / name).is_file() or (root / name).stat().st_size == 0:
            raise ValueError(f"missing remote-code file: {name}")
    index = root / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = ["model.safetensors"]
        weight_map = None
    from safetensors import safe_open
    keys = set()
    for shard in shards:
        path = root / shard
        if path.is_symlink() or path.resolve().parent != root:
            raise ValueError(f"weight shard must be a local file: {shard}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            shard_keys = set(handle.keys())
            if weight_map:
                assigned = {key for key, name in weight_map.items() if name == shard}
                if not assigned <= shard_keys:
                    raise ValueError(f"index references absent tensors in {shard}: {assigned-shard_keys}")
            if keys & shard_keys:
                raise ValueError("duplicate weight keys across shards")
            keys.update(shard_keys)
    if not any("anchor_embedding" in key for key in keys):
        raise ValueError("anchor embedding missing from exported weights")
    if text.get("hsd_null_read_gate") and not any("null_gate_proj" in key for key in keys):
        raise ValueError("null gate weights missing from exported checkpoint")
    return {"model_type": config["model_type"], "weight_shards": len(shards), "weight_keys": len(keys),
            "runtime": {key: value for key, value in text.items() if key.startswith(("hsd_", "anchor_", "hist_"))}}



def resolve_training_checkpoint(output_dir, step=1):
    """Find the HF export from resolved training arguments, including merged LoRA."""
    output = Path(output_dir)
    arguments = json.loads((output / "args.json").read_text())
    tuner = arguments.get("tuner_type", "full")
    merged = tuner != "full" and arguments.get("merge_lora", False)
    if tuner != "full" and not merged:
        raise ValueError("Full-weight smoke reload requires --merge_lora true for adapter training")
    checkpoint = output / (f"checkpoint-{step}-merged" if merged else f"checkpoint-{step}")
    if not (checkpoint / "config.json").is_file():
        raise ValueError(f"Training did not produce the expected HF export: {checkpoint}")
    return checkpoint


def load_check(root, device, dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if device == "auto" and not torch.cuda.is_available():
        raise RuntimeError("GPU reload requested but CUDA is unavailable; use --device cpu explicitly for fixtures")
    torch.manual_seed(123)
    model, info = AutoModelForCausalLM.from_pretrained(
        root, trust_remote_code=True, local_files_only=True, device_map=device,
        dtype=getattr(torch, dtype), output_loading_info=True)
    for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if info.get(field):
            raise RuntimeError(f"incomplete reload: {field}={info[field]}")
    tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=True, local_files_only=True)
    encoded = tokenizer("Describe the purpose of an anchor token.", return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        out = model(**encoded, use_cache=False)
        if not torch.isfinite(out.logits).all():
            raise RuntimeError("non-finite logits after reload")
        generated = model.generate(**encoded, max_new_tokens=8, do_sample=False)
    return {"loaded_all_weights": True, "generated_ids": generated[0].tolist(), "seed": 123}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--load-weights", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = inspect_checkpoint(args.checkpoint)
    if args.load_weights:
        report.update(load_check(args.checkpoint, args.device, args.dtype))
    rendered = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()

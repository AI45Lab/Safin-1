#!/usr/bin/env python3
"""Check a MARCH environment and optional Safin-1 checkpoint."""
import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys


EXPECTED = {
    "torch": "2.11.0",
    "transformers": "5.5.4",
    "safetensors": "0.7.0",
    "peft": "0.19.1",
    "megatron-core": "0.16.1",
    "transformer-engine": "2.14.1",
    "flash-linear-attention": "0.5.0",
    "triton": "3.6.0",
    "tilelang": "0.1.9",
    "qwen-vl-utils": "0.0.14",
    "decord": "0.6.0",
    "torchvision": "0.26.0",
}


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def matches_expected(name, version):
    if name in {"torch", "torchvision"}:
        return version.split("+", 1)[0] == EXPECTED[name]
    return version == EXPECTED[name]


def inspect_checkpoint(path):
    root = Path(path).expanduser().resolve()
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ValueError(f"checkpoint config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text = config.get("text_config") or {}
    model_type = config.get("model_type")
    expected_text = {
        "safin1_4b": "safin1_4b_text",
        "safin1_35b_a3b": "safin1_35b_a3b_text",
    }.get(model_type)
    if expected_text is None or text.get("model_type") != expected_text:
        raise ValueError(
            f"checkpoint is not migrated to Safin-1: model_type={model_type!r}, "
            f"text_model_type={text.get('model_type')!r}"
        )
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from march.safin1.runtime_validation import validate_safin1_runtime
    validate_safin1_runtime(text)
    required = (
        "anchor_chunk_size", "anchor_token_num", "hsd_read_block_size",
        "hsd_use_global_read", "hsd_global_read_layers", "hsd_read_topk",
        "hsd_null_read_gate", "anchor_embedding_type", "anchor_embedding_init_from",
    )
    missing = [name for name in required if name not in text]
    if text.get("anchor_embedding_init_from") == "bos" and text.get("anchor_bos_token_id") is None:
        missing.append("anchor_bos_token_id")
    if missing:
        raise ValueError(f"checkpoint is missing runtime fields: {missing}")
    return {
        "path": str(root),
        "model_type": model_type,
        "anchor_chunk_size": text["anchor_chunk_size"],
        "hsd_read_block_size": text["hsd_read_block_size"],
        "hsd_global_read_layers": text["hsd_global_read_layers"],
    }



def inspect_training_processor(checkpoint):
    """Exercise the actual training loader without loading weights or using CUDA."""
    from swift.model import get_model_processor

    model, processor = get_model_processor(
        checkpoint["path"], model_type=checkpoint["model_type"],
        load_model=False, download_model=False, device_map="cpu", use_hf=True,
    )
    if model is not None:
        raise RuntimeError("processor preflight unexpectedly loaded model weights")
    tokenizer = getattr(processor, "tokenizer", processor)
    tokens = tokenizer("A short memory anchor check.", add_special_tokens=False)["input_ids"]
    if not tokens:
        raise ValueError("training tokenizer returned no tokens")
    return {"processor": type(processor).__name__, "tokenizer": type(tokenizer).__name__,
            "test_tokens": len(tokens), "model_weights_loaded": False, "device": "cpu"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("hf", "training", "vllm", "eval"), default="hf")
    parser.add_argument("--model", help="Optional local Safin-1 checkpoint")
    parser.add_argument("--strict-versions", action="store_true")
    args = parser.parse_args()

    packages = ["torch", "transformers", "safetensors"]
    modules = ["march"]
    if args.backend == "training":
        packages += ["peft", "megatron-core", "transformer-engine", "flash-linear-attention", "triton", "tilelang",
                     "qwen-vl-utils", "decord", "torchvision"]
        modules += ["mcore_bridge", "megatron", "qwen_vl_utils", "decord", "torchvision"]
    elif args.backend == "vllm":
        packages += ["vllm", "flash-linear-attention", "triton"]
        modules += ["vllm", "fla"]

    elif args.backend == "eval":
        packages += ["lm_eval", "langdetect", "immutabledict", "nltk"]
        modules += ["lm_eval", "langdetect", "immutabledict", "nltk"]

    versions = {name: package_version(name) for name in packages}
    missing_packages = [name for name, version in versions.items() if version is None]
    mismatched = {
        name: {"expected": EXPECTED[name], "actual": version}
        for name, version in versions.items()
        if name in EXPECTED and version is not None and not matches_expected(name, version)
    }
    missing_modules = [name for name in modules if importlib.util.find_spec(name) is None]
    report = {
        "python": platform.python_version(),
        "backend": args.backend,
        "packages": versions,
        "missing_packages": missing_packages,
        "missing_modules": missing_modules,
        "version_mismatches": mismatched,
    }
    errors = []
    if sys.version_info[:2] != (3, 11):
        errors.append("the recorded MARCH environment uses Python 3.11")
    if missing_packages:
        errors.append(f"missing packages: {missing_packages}")
    if missing_modules:
        errors.append(f"missing modules: {missing_modules}")
    if args.strict_versions and mismatched:
        errors.append(f"version mismatches: {mismatched}")
    if args.backend == "training":
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from march.safin1.runtime_validation import validate_training_environment
        try:
            validate_training_environment()
        except ValueError as error:
            errors.append(str(error))
    if args.model:
        try:
            report["checkpoint"] = inspect_checkpoint(args.model)
            if args.backend == "training" and not errors:
                report["training_processor"] = inspect_training_processor(report["checkpoint"])
        except (OSError, ValueError, ImportError, RuntimeError, AttributeError) as error:
            errors.append(str(error))
    report["ok"] = not errors
    report["errors"] = errors
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())

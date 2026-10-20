#!/usr/bin/env python3
"""Fail before an offline vLLM build if required CMake source caches are absent."""
import json
import os
from pathlib import Path
import sys

# Directory names and overrides come from the vendored CMake declarations.
DEPENDENCIES = {
    "cutlass": ("VLLM_CUTLASS_SRC_DIR", "include/cutlass/cutlass.h"),
    "vllm-flash-attn": ("VLLM_FLASH_ATTN_SRC_DIR", "CMakeLists.txt"),
    "flashmla": ("FLASH_MLA_SRC_DIR", "setup.py"),
    "deepgemm": ("DEEPGEMM_SRC_DIR", "deep_gemm/__init__.py"),
    "qutlass": ("QUTLASS_SRC_DIR", "qutlass/csrc/bindings.cpp"),
    "triton_kernels": ("TRITON_KERNELS_SRC_DIR", "python/triton_kernels/triton_kernels/__init__.py"),
}


def inspect(environ):
    base = Path(environ.get("FETCHCONTENT_BASE_DIR", Path(__file__).resolve().parents[2] / "third_party/vllm/.deps"))
    report = {}
    for name, (variable, marker) in DEPENDENCIES.items():
        source = Path(environ[variable]) if environ.get(variable) else base / (name + "-src")
        if name == "triton_kernels" and environ.get(variable):
            marker = "__init__.py"
        report[name] = {"path": str(source), "override": variable, "present": (source / marker).is_file()}
    return report


def main():
    report = inspect(os.environ)
    missing = [name for name, row in report.items() if not row["present"]]
    print(json.dumps({"ok": not missing, "dependencies": report, "missing": missing}, indent=2))
    if missing:
        print("Offline native build needs the listed source caches; prepare them before requesting GPUs.", file=sys.stderr)
    return int(bool(missing))


if __name__ == "__main__":
    raise SystemExit(main())

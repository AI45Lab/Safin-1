"""The operator benchmark must validate its plan before importing GPU libraries."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/march/benchmark_operators.py"


def launch_without_torch(*arguments):
    bootstrap = "import runpy,sys;sys.modules['torch']=None;sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')"
    return subprocess.run([sys.executable, "-c", bootstrap, str(SCRIPT), *arguments],
                          capture_output=True, text=True)


def test_benchmark_dry_run_needs_no_cuda_or_output_files(tmp_path):
    output = tmp_path / "not-created" / "results.json"
    result = launch_without_torch("--output", str(output), "--dry-run", "--lengths", "32768",
                                  "--paged-persistent-memory")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["lengths"] == [32768] and plan["paged_persistent_memory"]
    assert plan["graph_read"] == "split"
    assert plan["requires_gpu"] and not plan["requires_models_or_datasets"]
    assert not output.parent.exists()


def test_invalid_chunk_layout_fails_before_cuda_import(tmp_path):
    result = launch_without_torch("--output", str(tmp_path / "results.json"), "--lengths", "513")
    assert result.returncode != 0
    assert "multiples of 512" in result.stderr
    assert "import of torch halted" not in result.stderr


@pytest.mark.parametrize("status,expected", [("ok", 0), ("unsupported", 0),
                                           ("numerical_failed", 1), ("error", 1),
                                           ("group_error", 1)])
def test_benchmark_exit_status_exposes_backend_failures(tmp_path, monkeypatch, status, expected):
    spec = importlib.util.spec_from_file_location("march_operator_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class StubBenchmark:
        def __init__(self, args):
            self.rows = []
            self.metadata = {}

        def chunk(self):
            if status == "group_error":
                raise RuntimeError("backend import failed")
            self.rows.append({"status": status})

        def save(self):
            pass

        def record(self, row):
            self.rows.append(row)

    monkeypatch.setattr(module, "Benchmark", StubBenchmark)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--groups", "chunk", "--output",
                                    str(tmp_path / "results.json")])
    # main prepends bridge paths for real execution; isolate that global mutation.
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert module.main() == expected

"""Release evidence must reject stale source, missing kernels and skipped tests."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/march/check_operator_correctness.py"


def checker():
    spec = importlib.util.spec_from_file_location("operator_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def evidence(tmp_path):
    module = checker()
    for name in (*module.TESTS, *module.CONTRACT_SOURCES, "scripts/march/check_operator_correctness.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n")
    path = tmp_path / "march/inference/paged_epilogue_triton.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("@triton.jit\ndef _history_scale_add_kernel():\n    pass\n")
    inventory = module.inventory(tmp_path)
    rows = [{"nodeid": module.TESTS[-1]+"::"+name, "outcome": "passed",
             "phases": {"setup": "passed", "call": "passed", "teardown": "passed"}}
            for name in module.REQUIRED_TESTS]
    data = dict(schema_version=1, source_sha256=module.fingerprints(tmp_path), operator_inventory=inventory,
                test_files=list(module.TESTS), tests=rows, collected=len(rows), pytest_exit_code=0,
                kernel_tests={name: [rows[0]["nodeid"]] for name in inventory},
                source_changed_during_run=False, environment={"cuda_available": True})
    return module, tmp_path, data


def test_complete_source_bound_evidence(evidence):
    module, root, data = evidence
    assert module.review(root, data)["ready"]


def test_changed_operator_invalidates_previous_acceptance(evidence):
    module, root, data = evidence
    path = root / "march/inference/paged_epilogue_triton.py"
    path.write_text(path.read_text()+"# changed math\n")
    result = module.review(root, data)
    assert not result["ready"]
    assert any("fingerprints" in error for error in result["errors"])


def test_new_kernel_without_launch_evidence_is_blocked(evidence):
    module, root, data = evidence
    path = root / "march/inference/paged_epilogue_triton.py"
    path.write_text(path.read_text()+"\n@triton.jit\ndef _new_kernel():\n    pass\n")
    data["source_sha256"] = module.fingerprints(root)
    data["operator_inventory"] = module.inventory(root)
    assert any("_new_kernel" in error for error in module.review(root, data)["errors"])


@pytest.mark.parametrize("change", ["skip", "exit", "missing_case", "unlaunched", "changed_during_run", "no_cuda", "incomplete_phase"])
def test_incomplete_acceptance_never_passes(evidence, change):
    module, root, original = evidence
    data = deepcopy(original)
    if change == "skip":data["tests"][0]["outcome"] = "not_passed"
    elif change == "exit":data["pytest_exit_code"] = 1
    elif change == "missing_case":data["tests"].pop();data["collected"] -= 1
    elif change == "unlaunched":data["kernel_tests"] = {}
    elif change == "changed_during_run":data["source_changed_during_run"] = True
    elif change == "no_cuda":data["environment"]["cuda_available"] = False
    elif change == "incomplete_phase":data["tests"][0]["phases"].pop("call")
    assert not module.review(root, data)["ready"]


def test_unreviewed_kernel_source_is_blocked(evidence):
    module, root, _ = evidence
    (root/"march/new_kernel.py").write_text("@triton.jit\ndef new():\n    pass\n")
    with pytest.raises(ValueError, match="unreviewed"):
        module.inventory(root)


def test_missing_gpu_evidence_is_failure_without_torch(tmp_path):
    bootstrap = "import runpy,sys;sys.modules['torch']=None;sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')"
    result = subprocess.run([sys.executable, "-c", bootstrap, str(SCRIPT), "--root", str(tmp_path)],
                            text=True, capture_output=True)
    assert result.returncode == 1
    assert not json.loads(result.stdout)["ready"]


@pytest.mark.parametrize("name", ["swift/megatron/init.py", "scripts/march/_training_common.sh"])
def test_changed_training_contract_invalidates_acceptance(evidence, name):
    module, root, data = evidence
    path = root / name
    path.write_text(path.read_text() + "# changed training contract\n")
    assert not module.review(root, data)["ready"]

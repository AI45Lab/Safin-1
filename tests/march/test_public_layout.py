import importlib.util
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).parents[2]


def load_script(name):
    path = ROOT / "scripts/march" / name
    spec = importlib.util.spec_from_file_location(f"march_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_profiles():
    dense = (ROOT / "configs/march/safin1_4b.yaml").read_text()
    moe = (ROOT / "configs/march/safin1_35b_a3b.yaml").read_text()
    assert '"model_type": "safin1_4b"' in dense
    assert "safin1_35b_a3b" in moe
    assert '"hsd_global_read_layers": "10-30"' in dense
    assert '"hsd_global_read_layers": "10-30"' in moe
    assert '"hsd_read_block_size": 32768' in dense
    assert '"hsd_read_block_size": 32768' in moe
    assert '"anchor_embedding_init_from": "bos"' in dense
    assert '"anchor_embedding_init_from": "bos"' in moe
    assert '"anchor_bos_token_id": 248045' in dense
    assert '"anchor_bos_token_id": 248045' in moe

def test_scripts_are_public_safe():
    for path in (ROOT / "scripts/march").glob("*.sh"):
        text = path.read_text()
        assert "/mnt-" + "shared-storage" not in text
        assert "r" + "launch" not in text
        assert "r" + "job" not in text


def test_release_copy_keeps_vllm_build_requirements():
    requirement = Path("third_party/vllm/requirements/build/cuda.txt")
    exporter = load_script("export_source.py")
    assert not exporter.exclude_from_export(requirement)
    assert exporter.exclude_from_export(Path("build/generated.whl"))
    assert exporter.exclude_from_export(Path("dist/package.tar.gz"))

def test_public_scanner_exists():
    assert (ROOT / "scripts/march/check_public_release.py").is_file()

def test_public_assets():
    assert not (ROOT / "docs/march/internal-quickstart.md").exists()
    assert not (ROOT / "scripts/march/create_internal_snapshot.py").exists()
    assert (ROOT / "docs/march/persistent-state.md").is_file()
    assert (ROOT / "examples/march/cpt.jsonl").is_file()
    assert (ROOT / "examples/march/sft.jsonl").is_file()
    assert (ROOT / "scripts/march/eval_hf.sh").is_file()
    assert not (ROOT / "configs/qwen3_5_hsd_runtime.json").exists()


def test_hf_inference_dry_run_does_not_import_runtime():
    environment = os.environ.copy()
    environment.update(MODEL="/tmp/Safin-1-4B", PROMPT="test", DRY_RUN="1")
    result = subprocess.run(
        ["bash", "scripts/march/infer_hf.sh"],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "/tmp/Safin-1-4B" in result.stdout

def test_removed_experiment_and_public_cache_name():
    megatron_init = (ROOT / "swift/megatron/init.py").read_text()
    assert "STAR1" not in megatron_init
    pt_cli = (ROOT / "swift/cli/pt.py").read_text()
    assert "MARCH_TRITON_CACHE_BASE" in pt_cli

def test_flashqla_wheel_attribution_is_declared():
    setup = (ROOT / "setup.py").read_text()
    assert "['LICENSE', 'NOTICE', 'UPSTREAM.md']" in setup


def test_unreleased_training_experiments_are_absent():
    removed_paths = [
        "march/kernels/march_global_read_tile.py",
        "march/kernels/_flashqla/ops/gated_delta_rule/chunk/hopper/direct_a_chunk_bwd.py",
        "march/kernels/_flashqla/ops/gated_delta_rule/chunk/hopper/direct_a_chunk_bwd_inline_dd.py",
        "march/kernels/_flashqla/ops/gated_delta_rule/chunk/hopper/direct_a_chunk_bwd_inline_dd_noatomic.py",
        "march/kernels/_flashqla/ops/gated_delta_rule/chunk/hopper/direct_a_chunk_bwd_reference.py",
        "march/kernels/_flashqla/ops/gated_delta_rule/chunk/hopper/direct_a_chunk_bwd_saved_a_reference.py",
    ]
    assert all(not (ROOT / path).exists() for path in removed_paths)

    checked = [
        "march/safin1/configuration_safin1.py",
        "march/safin1/modeling_safin1.py",
        "march/safin1/march_gdn.py",
        "swift/megatron/init.py",
        "third_party/mcore-bridge/src/mcore_bridge/config/model_config.py",
        "third_party/mcore-bridge/src/mcore_bridge/config/parser.py",
        "third_party/mcore-bridge/src/mcore_bridge/model/mm_gpts/qwen3_5_gdn.py",
        "third_party/mcore-bridge/src/mcore_bridge/model/modules/march.py",
    ]
    forbidden = (
        "hist_replace_self_attn",
        "qwen35_hsd_replace_self_attn",
        "HSD_STATE_READ_OUTPUT",
        "tile_online",
        "tile_speed",
        "direct_a",
    )
    for relative_path in checked:
        text = (ROOT / relative_path).read_text()
        assert all(term not in text for term in forbidden), relative_path

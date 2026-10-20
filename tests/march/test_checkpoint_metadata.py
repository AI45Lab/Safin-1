"""CPU regressions for the Safin-1 rename; no ML dependencies required."""

import ast
from copy import deepcopy
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from march.safin1 import config_utils


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(moe=False):
    return {
        "model_type": "historical_qwen3_5_moe" if moe else "historical_qwen3_5",
        "architectures": ["HistoricalQwen3_5MoeForConditionalGeneration" if moe
                          else "HistoricalQwen3_5ForConditionalGeneration"],
        "auto_map": {"AutoConfig": "configuration_historical_qwen3_5.HistoricalQwen3_5Config"},
        "text_config": {
            "model_type": "historical_qwen3_5_moe_text" if moe else "historical_qwen3_5_text",
            "num_hidden_layers": 40 if moe else 32,
            "anchor_chunk_size": 512, "anchor_token_num": 1,
            "anchor_embedding_type": "learned", "anchor_embedding_init_from": "bos",
            "anchor_bos_token_id": 248045, "anchor_embedding_init_scale": 0.5,
            "hsd_read_block_size": 32768,
            "hsd_global_read_layers": "10-30",
            "hsd_read_topk": 4, "hsd_null_read_gate": True,
        },
        "vision_config": {"model_type": "qwen3_5_vision"},
    }


def safetensors_fixture(key):
    header = json.dumps({key: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
                        separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    return struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0)


class CheckpointMetadataTests(unittest.TestCase):
    def test_normalize_preserves_runtime_and_is_idempotent(self):
        for moe, target in [(False, "safin1_4b"), (True, "safin1_35b_a3b")]:
            with self.subTest(model=target):
                original = fixture(moe)
                before = deepcopy(original)
                result = config_utils.normalize_safin1_config(original)
                self.assertEqual(original, before)
                self.assertTrue(config_utils.is_safin1_config(result))
                self.assertEqual(result["model_type"], target)
                for key, value in original["text_config"].items():
                    if key != "model_type":
                        self.assertEqual(result["text_config"][key], value)
                self.assertEqual(result["vision_config"], original["vision_config"])
                self.assertEqual(config_utils.normalize_safin1_config(result), result)

    def test_training_runtime_overrides_and_variant_conflicts(self):
        result = config_utils.normalize_safin1_config(fixture(), {"hsd_read_topk": 2, "anchor_chunk_size": None})
        self.assertEqual(result["text_config"]["hsd_read_topk"], 2)
        self.assertEqual(result["text_config"]["anchor_chunk_size"], 512)
        for runtime in [{"model_type": "safin1_35b_a3b"}, {"model_type": "unknown"}]:
            with self.assertRaises(ValueError):
                config_utils.normalize_safin1_config(fixture(), runtime)
        mismatched = fixture()
        mismatched["text_config"]["model_type"] = "safin1_35b_a3b_text"
        with self.assertRaises(ValueError):
            config_utils.normalize_safin1_config(mismatched)
        with self.assertRaises(ValueError):
            config_utils.normalize_safin1_config({"model_type": "llama"})

    def test_profile_runtime_is_preserved(self):
        for name, moe in [("safin1_4b", False), ("safin1_35b_a3b", True)]:
            profile = json.loads((ROOT / f"configs/march/{name}.yaml").read_text())
            result = config_utils.normalize_safin1_config(fixture(moe), profile)
            self.assertEqual(result["model_type"], name)
            for key, value in profile.items():
                if key != "model_type":
                    self.assertEqual(result["text_config"][key], value)

    def test_remote_code_export_resolves_auto_map_and_relative_imports(self):
        for target in ["safin1_4b", "safin1_35b_a3b"]:
            with self.subTest(model=target), tempfile.TemporaryDirectory() as temp:
                dest = Path(temp)
                config_utils.copy_safin1_remote_code(dest, target)
                for mapping in config_utils.safin1_auto_map(target).values():
                    module, name = mapping.split(".")
                    tree = ast.parse((dest / f"{module}.py").read_text())
                    self.assertIn(name, {node.name for node in tree.body if isinstance(node, ast.ClassDef)})
                for name in config_utils.safin1_remote_code_files(target):
                    tree = ast.parse((dest / name).read_text())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                            self.assertTrue((dest / (node.module + ".py")).is_file(), (name, node.module))

    def test_export_rejects_incomplete_source_before_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            source.mkdir()
            dest = Path(temp) / "dest"
            with patch.object(config_utils, "__file__", str(source / "config_utils.py")):
                with self.assertRaises(FileNotFoundError):
                    config_utils.copy_safin1_remote_code(dest, "safin1_4b")
            self.assertFalse(dest.exists())

    def test_migration_preserves_hardlinked_originals_and_weights(self):
        migrate = load_file("march_checkpoint_migration", ROOT / "scripts/march/migrate_checkpoint.py")
        for moe in [False, True]:
            with self.subTest(moe=moe), tempfile.TemporaryDirectory() as temp:
                source, dest = Path(temp) / "source", Path(temp) / "copy"
                source.mkdir()
                dest.mkdir()
                legacy_config = fixture(moe)
                legacy_config["text_config"]["summary_chunk_size"] = legacy_config["text_config"].pop(
                    "anchor_chunk_size")
                legacy_config["text_config"]["summary_token_num"] = legacy_config["text_config"].pop(
                    "anchor_token_num")
                legacy_weight = "model.language_model.summary_embedding.weight"
                files = {
                    "config.json": json.dumps(legacy_config).encode(),
                    "args.json": json.dumps({"model_type": fixture(moe)["model_type"],
                        "summary_chunk_size": 512,
                        "model_meta": "HistoricalQwen3_5MoeLoader" if moe else "HistoricalQwen3_5Loader",
                        "export_module": "loglinear.safin1.modeling_safin1",
                        "kernel_module": "loglinear.march.anchor_utils",
                        "legacy_module": "loglinear.qwen35_historical.modeling_historical_qwen3_5"}).encode(),
                    "model.safetensors": safetensors_fixture(legacy_weight),
                    "model.safetensors.index.json": json.dumps(
                        {"weight_map": {legacy_weight: "model.safetensors"}}).encode(),
                    "tokenizer.json": b"{}",
                    "historical_gdn.py": b"# old remote-code file\n",
                    "ops.py": b"# shared remote-code file\n",
                }
                for name, content in files.items():
                    (source / name).write_bytes(content)
                    os.link(source / name, dest / name)
                dry_run = migrate.migrate_checkpoint(dest, dry_run=True)
                self.assertTrue(dry_run["dry_run"])
                self.assertEqual(dry_run["rewritten_weight_shards"], 1)
                self.assertEqual(dry_run["renamed_tensor_keys"], 1)
                self.assertEqual({p.name for p in dest.iterdir()}, set(files))
                self.assertEqual(migrate.migrate_checkpoint(dest)["renamed_tensor_keys"], 1)
                self.assertEqual(migrate.migrate_checkpoint(dest)["renamed_tensor_keys"], 0)
                for name, content in files.items():
                    self.assertEqual((source / name).read_bytes(), content, name)
                for name in ["tokenizer.json"]:
                    self.assertEqual((dest / name).read_bytes(), files[name])
                    self.assertEqual((dest / name).stat().st_ino, (source / name).stat().st_ino)
                for name in ["config.json", "args.json", "model.safetensors",
                             "model.safetensors.index.json", "ops.py"]:
                    self.assertNotEqual((dest / name).stat().st_ino, (source / name).stat().st_ino)
                self.assertFalse((dest / "historical_gdn.py").exists())
                result = json.loads((dest / "config.json").read_text())
                self.assertTrue(config_utils.is_safin1_config(result))
                self.assertNotIn("summary", (dest / "config.json").read_text().lower())
                self.assertNotIn("HistoricalQwen", (dest / "args.json").read_text())
                self.assertNotIn("summary", (dest / "args.json").read_text().lower())
                metadata = json.loads((dest / "args.json").read_text())
                self.assertEqual(metadata["export_module"], "march.safin1.modeling_safin1")
                self.assertEqual(metadata["kernel_module"], "march.kernels.anchor_utils")
                self.assertEqual(metadata["legacy_module"], "march.safin1.modeling_safin1")
                index = json.loads((dest / "model.safetensors.index.json").read_text())
                self.assertEqual(set(index["weight_map"]), {legacy_weight.replace("summary", "anchor")})
                _, header = migrate._read_safetensors_header(dest / "model.safetensors")
                self.assertEqual({key for key in header if key != "__metadata__"},
                                 {legacy_weight.replace("summary", "anchor")})

    def test_megatron_finalizer_saves_complete_safin1_checkpoint(self):
        # Execute the actual metadata-only functions without importing Megatron/CUDA.
        tree = ast.parse((ROOT / "swift/megatron/init.py").read_text())
        names = {"_env_enabled", "_has_hsd_runtime_config", "_collect_hsd_runtime_config", "_finalize_hsd_hf_config_file"}
        nodes = [n for n in tree.body if
                 (isinstance(n, ast.FunctionDef) and n.name in names) or
                 (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "HSD_HF_CONFIG_KEYS" for t in n.targets))]
        namespace = {"os": os, "json": json, "logger": logging.getLogger(__name__)}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "swift/megatron/init.py", "exec"), namespace)
        for moe, target in [(False, "safin1_4b"), (True, "safin1_35b_a3b")]:
            with self.subTest(model=target), tempfile.TemporaryDirectory() as temp:
                cfg = fixture(moe)
                cfg["model_type"] = "qwen3_5_moe" if moe else "qwen3_5"
                cfg["text_config"]["model_type"] = "qwen3_5_moe_text" if moe else "qwen3_5_text"
                path = Path(temp) / "config.json"
                path.write_text(json.dumps(cfg))
                runtime = SimpleNamespace(**cfg["text_config"])
                namespace["_finalize_hsd_hf_config_file"](temp, SimpleNamespace(hsd_runtime_config="profile"), runtime)
                saved = json.loads(path.read_text())
                self.assertEqual(saved["model_type"], target)
                self.assertEqual(saved["text_config"]["hsd_read_block_size"], cfg["text_config"]["hsd_read_block_size"])
                for name in config_utils.safin1_remote_code_files(target):
                    self.assertGreater((Path(temp) / name).stat().st_size, 0)

    def test_vllm_architecture_and_config_registry_targets_exist(self):
        vllm = ROOT / "third_party/vllm/vllm"
        tree = ast.parse((vllm / "model_executor/models/registry.py").read_text())
        found = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value.startswith("Safin1"):
                    module, cls = ast.literal_eval(value)
                    source = vllm / "model_executor/models" / f"{module}.py"
                    classes = {n.name for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef)}
                    self.assertIn(cls, classes)
                    found.add(key.value)
        self.assertEqual(len(found), 4)
        registry = ast.parse((vllm / "transformers_utils/configs/__init__.py").read_text())
        configs = set()
        for node in ast.walk(registry):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value.startswith("Safin1"):
                    module = ast.literal_eval(value)
                    source = vllm.parent / (module.replace(".", "/") + ".py")
                    classes = {n.name for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef)}
                    self.assertIn(key.value, classes)
                    configs.add(key.value)
        self.assertEqual(len(configs), 4)

    def test_launcher_arguments_and_runtime_alias(self):
        with tempfile.TemporaryDirectory() as temp:
            stub_dir = Path(temp)
            custom_runtime = stub_dir / "custom.json"
            custom_runtime.write_text(json.dumps({"hsd_read_topk": 4}))
            for executable in ["megatron"]:
                stub = stub_dir / executable
                stub.write_text(f"#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
                stub.chmod(0o755)
            env = {**os.environ, "PATH": str(stub_dir) + os.pathsep + os.environ.get("PATH", ""),
                   "MODEL": "/model with spaces", "DATASET": "/dataset", "OUTPUT_DIR": "/output", "PYTHON_BIN": sys.executable}
            for name in ["MARCH_RUNTIME_CONFIG", "HSD_RUNTIME_CONFIG", "MODEL_TYPE"]:
                env.pop(name, None)
            for script, task in [("train_cpt.sh", "pt"), ("train_sft.sh", "sft")]:
                for profile, config in [("4b", "safin1_4b"), ("35ba3b", "safin1_35b_a3b")]:
                    with self.subTest(script=script, profile=profile):
                        result = subprocess.run(["bash", str(ROOT / "scripts/march" / script)],
                                                env={**env, "MODEL_PROFILE": profile}, text=True, capture_output=True, check=True)
                        args = json.loads(result.stdout)
                        self.assertEqual(args[0], task)
                        self.assertEqual(args[args.index("--model") + 1], env["MODEL"])
                        self.assertEqual(args[args.index("--hsd-runtime-config") + 1], str(ROOT / f"configs/march/{config}.yaml"))
                for alias in ["MARCH_RUNTIME_CONFIG", "HSD_RUNTIME_CONFIG"]:
                    result = subprocess.run(["bash", str(ROOT / "scripts/march" / script)],
                                            env={**env, "MODEL_PROFILE": "4b", alias: str(custom_runtime)}, text=True, capture_output=True, check=True)
                    args = json.loads(result.stdout)
                    self.assertEqual(args[args.index("--hsd-runtime-config") + 1], str(custom_runtime))
                result = subprocess.run(["bash", str(ROOT / "scripts/march" / script)],
                                        env={**env, "MODEL_PROFILE": "4b", "MARCH_RUNTIME_CONFIG": "/a", "HSD_RUNTIME_CONFIG": "/b"},
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 2)
                checkpoint = stub_dir / "safin-35b"
                checkpoint.mkdir(exist_ok=True)
                (checkpoint / "config.json").write_text(json.dumps({
                    "model_type": "safin1_35b_a3b",
                    "text_config": {
                        "model_type": "safin1_35b_a3b_text",
                        "anchor_chunk_size": 512,
                        "anchor_token_num": 1,
                        "anchor_embedding_type": "learned",
                        "anchor_embedding_init_from": "bos",
                        "anchor_bos_token_id": 248045,
                        "hsd_read_block_size": 32768,
                        "hsd_use_global_read": True,
                        "hsd_global_read_layers": "10-30",
                        "hsd_read_topk": 4,
                        "hsd_null_read_gate": True,
                    },
                }))
                saved_env = {**env, "MODEL": str(checkpoint), "PYTHON_BIN": sys.executable}
                result = subprocess.run(["bash", str(ROOT / "scripts/march" / script)],
                                        env=saved_env, text=True, capture_output=True, check=True)
                args = json.loads(result.stdout)
                self.assertNotIn("--hsd-runtime-config", args)
                self.assertEqual(args[args.index("--max_length") + 1], "16384")
                result = subprocess.run(["bash", str(ROOT / "scripts/march" / script)],
                                        env={**saved_env, "MODEL_PROFILE": "4b"}, text=True, capture_output=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn("conflicts with checkpoint", result.stderr)
            # Graph defaults and eager rollback are exercised through the real
            # shell entry point in test_vllm_launcher.py.


if __name__ == "__main__":
    unittest.main()

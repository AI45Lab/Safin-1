"""Optional CPU checks with the installed torch/Transformers runtime."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(HAS_TORCH, "torch is required for anchor layout checks")
class AnchorLayoutTests(unittest.TestCase):
    def test_text_and_labels_survive_512_token_boundary(self):
        import torch
        from march.training.anchor_batch import maybe_build_hsd_anchor_batch

        config = {"anchor_chunk_size": 512, "anchor_token_num": 1, "anchor_placeholder_token_id": 0}
        for length in [511, 512, 513, 1024, 1025]:
            with self.subTest(length=length):
                ids = torch.arange(1, length + 1).unsqueeze(0)
                batch = {"input_ids": ids, "labels": ids.clone()}
                result, layout = maybe_build_hsd_anchor_batch(batch, config)
                self.assertEqual(result["input_ids"].shape[1], length + length // 512)
                self.assertEqual(int(layout.anchor_mask.sum()), length // 512)
                torch.testing.assert_close(result["input_ids"][layout.text_mask], ids.flatten())
                torch.testing.assert_close(result["labels"], ids)
                second, _ = maybe_build_hsd_anchor_batch(result, config)
                torch.testing.assert_close(second["input_ids"], result["input_ids"])

    def test_packed_documents_have_independent_anchor_boundaries(self):
        import torch
        from march.training.anchor_batch import maybe_build_hsd_anchor_batch

        ids = torch.arange(1, 1030).unsqueeze(0)
        positions = torch.cat([torch.arange(513), torch.arange(516)]).unsqueeze(0)
        result, layout = maybe_build_hsd_anchor_batch(
            {"input_ids": ids, "position_ids": positions, "labels": ids.clone()},
            {"anchor_chunk_size": 512, "anchor_token_num": 1},
        )
        self.assertEqual(result["text_cu_seqlens"].tolist(), [0, 513, 1029])
        self.assertEqual(result["anchor_cu_seqlens"].tolist(), [0, 1, 2])
        self.assertEqual(result["cu_seqlens"].tolist(), [0, 514, 1031])
        self.assertEqual(torch.where(layout.anchor_mask[0])[0].tolist(), [512, 1026])
        torch.testing.assert_close(result["input_ids"][layout.text_mask], ids.flatten())


@unittest.skipUnless(HAS_TORCH and HAS_TRANSFORMERS, "torch and Transformers are required")
class RemoteCodeLoadingTests(unittest.TestCase):
    def test_pre_anchor_config_fields_are_rejected(self):
        from huggingface_hub.errors import StrictDataclassClassValidationError
        from march.safin1.configuration_safin1 import Safin1TextConfig
        from march.safin1.configuration_safin1_moe import Safin1MoeTextConfig

        for config_class in (Safin1TextConfig, Safin1MoeTextConfig):
            with self.subTest(config=config_class.__name__):
                with self.assertRaises(StrictDataclassClassValidationError):
                    config_class(summary_chunk_size=512)
                with self.assertRaises(StrictDataclassClassValidationError):
                    config_class(hsd_summary_mlp=False)

    def test_both_variants_load_config_and_model_class_offline(self):
        from transformers import AutoConfig
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from march.safin1.config_utils import normalize_safin1_config, copy_safin1_remote_code

        for name in ["safin1_4b", "safin1_35b_a3b"]:
            with self.subTest(model=name), tempfile.TemporaryDirectory() as temp:
                profile = json.loads((ROOT / f"configs/march/{name}.yaml").read_text())
                config = normalize_safin1_config({"model_type": name, "text_config": {"model_type": name + "_text"}}, profile)
                copy_safin1_remote_code(temp, name)
                Path(temp, "config.json").write_text(json.dumps(config))
                retained = {key: value for key, value in CONFIG_MAPPING._extra_content.items()
                            if not key.startswith("safin1_")}
                with patch.dict(CONFIG_MAPPING._extra_content, retained, clear=True):
                    loaded = AutoConfig.from_pretrained(temp, trust_remote_code=True, local_files_only=True)
                model_class = get_class_from_dynamic_module(config["auto_map"]["AutoModelForCausalLM"], temp, local_files_only=True)
                self.assertIs(model_class.config_class, type(loaded))
                self.assertEqual(loaded.model_type, name)
                self.assertEqual(loaded.text_config.hsd_read_block_size, profile["hsd_read_block_size"])
                self.assertEqual(loaded.text_config.hsd_global_read_layers, profile["hsd_global_read_layers"])

    def test_local_model_registration(self):
        from transformers import AutoConfig, AutoModelForCausalLM
        from march.safin1 import register_safin1_models

        register_safin1_models()
        for name, cls in [("safin1_4b", "Safin1ForConditionalGeneration"),
                          ("safin1_35b_a3b", "Safin1MoeForConditionalGeneration")]:
            config = AutoConfig.for_model(name)
            self.assertEqual(config.model_type, name)
            self.assertEqual(AutoModelForCausalLM._model_mapping[type(config)].__name__, cls)
            self.assertEqual(AutoConfig.for_model(name + "_text").model_type, name + "_text")


if __name__ == "__main__":
    unittest.main()

"""Public model wrappers share schema and generation while retaining checkpoint names."""
from dataclasses import fields
import importlib.util
from pathlib import Path

import pytest
import torch

from march.safin1.configuration_common import Safin1RuntimeFields
from march.safin1.configuration_safin1 import Safin1TextConfig
from march.safin1.configuration_safin1_moe import Safin1MoeTextConfig


def test_both_configs_share_the_same_runtime_fields_and_defaults():
    dense, moe = Safin1TextConfig(), Safin1MoeTextConfig()
    names = [field.name for field in fields(Safin1RuntimeFields)]
    for name in names:
        assert getattr(dense, name) == getattr(moe, name), name
    assert dense.hsd_read_block_size == moe.hsd_read_block_size == 32768
    assert dense.hist_read_block_size == moe.hist_read_block_size == 32768
    assert {"hsd_route_head_dim", "hsd_route_query_heads", "hsd_route_key_heads"} <= set(names)


@pytest.mark.parametrize("cls", [Safin1TextConfig, Safin1MoeTextConfig])
def test_explicit_legacy_metadata_survives_roundtrip(cls):
    config = cls(hist_read_block_size=1024, hsd_read_block_size=32768,
                 hsd_history_read_q_lora_rank=64, hsd_anchor_mlp=False)
    loaded = cls.from_dict(config.to_dict())
    assert loaded.hist_read_block_size == 1024
    assert loaded.hsd_read_block_size == 32768
    assert loaded.hsd_history_read_q_lora_rank == 64
    assert loaded.hsd_anchor_mlp is False
    assert all(name not in loaded.to_dict() for name in
               ("HSD_COMPATIBILITY_FIELDS", "HSD_RETIRED_FIELDS", "HSD_TRAINING_ONLY_FIELDS"))


@pytest.mark.parametrize("variant", ["safin1_4b", "safin1_35b_a3b"])
def test_retired_query_rank_is_metadata_not_a_model_dimension(variant):
    from march.safin1 import Safin1ForConditionalGeneration, Safin1MoeForConditionalGeneration
    path = Path(__file__).parents[2] / "scripts/march/smoke_hf.py"
    spec = importlib.util.spec_from_file_location("shared_schema_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    cls = Safin1ForConditionalGeneration if variant == "safin1_4b" else Safin1MoeForConditionalGeneration
    tensors = []
    with fixture.reference_kernels():
        for rank in (-7, 32):
            torch.manual_seed(43)
            config = fixture.tiny_config(variant, chunk_size=8)
            config.text_config.hsd_history_read_q_lora_rank = rank
            model = cls(config).eval()
            assert model.config.text_config.hsd_history_read_q_lora_rank == rank
            tensors.append(model.state_dict())
    assert tensors[0].keys() == tensors[1].keys()
    for name in tensors[0]:
        torch.testing.assert_close(tensors[0][name], tensors[1][name], atol=0, rtol=0)

"""Padded HF batches must preserve causal attention and per-row cache lengths."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

from march.safin1.cache_utils import _RowFullAttentionCache
from march.safin1.modeling_common import _run_full_attention_padded_fallback


def smoke_fixture():
    path = Path(__file__).parents[2] / "scripts/march/smoke_hf.py"
    spec = importlib.util.spec_from_file_location("march_batch_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("past_length", [0, 2])
def test_padded_attention_is_causal_with_or_without_cached_prefix(backend, past_length):
    torch.manual_seed(2026)
    torch.set_num_threads(2)
    config = Qwen3_5TextConfig(hidden_size=16, num_attention_heads=2,
                             num_key_value_heads=2, head_dim=8,
                             num_hidden_layers=1, layer_types=["full_attention"],
                             attention_dropout=0.0)
    config._attn_implementation = backend
    attention = Qwen3_5Attention(config, 0).eval()
    layer = SimpleNamespace(self_attn=attention, layer_idx=0)
    hidden = torch.randn(2, 4, 16)
    changed = hidden.clone()
    changed[:, -1] = torch.randn(2, 16) * 10
    prefix = torch.randn(2, past_length, 16)
    valid = torch.tensor([[False, True, True, True], [True, True, True, True]])
    anchors = torch.zeros_like(valid)
    rope = (torch.ones(2, 4, 8), torch.zeros(2, 4, 8))

    def causal_mask(query_length, prefix_length=0):
        query = torch.arange(query_length) + prefix_length
        keys = torch.arange(query_length + prefix_length)
        return torch.zeros(query_length, len(keys)).masked_fill(
            keys[None] > query[:, None], float("-inf"))[None, None]

    def fresh_cache():
        if not past_length:
            return None
        rows = []
        for row in range(2):
            cache = _RowFullAttentionCache(0)
            attention(hidden_states=prefix[row:row + 1],
                      position_embeddings=(torch.ones(1, past_length, 8), torch.zeros(1, past_length, 8)),
                      attention_mask=causal_mask(past_length), past_key_values=cache)
            rows.append(cache)
        return SimpleNamespace(layers=[SimpleNamespace(
            keys=torch.cat([row.keys for row in rows]),
            values=torch.cat([row.values for row in rows]),
            hsd_text_cache_lengths_per_row=torch.full((2,), past_length))])

    def fallback(states):
        return _run_full_attention_padded_fallback(
            layer, hidden_states=states, attention_mask=None,
            position_ids=torch.arange(4)[None].expand(2, -1),
            past_key_values=fresh_cache(), position_embeddings=rope,
            anchor_mask=anchors, text_valid_mask=valid)

    with torch.no_grad():
        before, after = fallback(hidden), fallback(changed)
        for row in range(2):
            first = int(valid[row].nonzero()[0])
            # Future-token changes cannot affect the first valid query.
            torch.testing.assert_close(before[row, first], after[row, first], atol=0, rtol=0)
            assert (before[row, -1] - after[row, -1]).abs().max() > 1e-6
            compact = torch.cat([prefix[row:row + 1], hidden[row:row + 1, valid[row]]], dim=1)
            length = compact.shape[1]
            reference = attention(
                hidden_states=compact, attention_mask=causal_mask(length),
                position_embeddings=(torch.ones(1, length, 8), torch.zeros(1, length, 8)))[0]
            torch.testing.assert_close(before[row, valid[row]], reference[0, past_length:],
                                       atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("variant", ["safin1_4b", "safin1_35b_a3b"])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("persistent", [False, True])
def test_mixed_length_prefill_and_anchor_boundary_decode_match_independent_rows(variant, backend, persistent):
    from march.safin1 import Safin1ForConditionalGeneration, Safin1MoeForConditionalGeneration
    fixture = smoke_fixture()
    cls = Safin1ForConditionalGeneration if variant == "safin1_4b" else Safin1MoeForConditionalGeneration
    torch.manual_seed(42)
    torch.set_num_threads(2)
    config = fixture.tiny_config(variant, chunk_size=8)
    config.text_config.num_hidden_layers = 4
    config.text_config.layer_types = ["linear_attention", "full_attention"] * 2
    config.text_config.hsd_global_read_layers = "1-4"
    config.text_config.hsd_persistent_memory_enabled = persistent
    config._attn_implementation = backend
    with fixture.reference_kernels(), torch.no_grad():
        model = cls(config).eval()
        ids = torch.randint(3, 59, (2, 9))
        mask = torch.ones_like(ids)
        mask[1, :2] = 0
        ids[1, :2] = 0
        output = model(input_ids=ids, attention_mask=mask, use_cache=True)
        for row in range(2):
            compact = ids[row, mask[row].bool()][None]
            reference = model(input_ids=compact, use_cache=False).logits[0]
            torch.testing.assert_close(output.logits[row, mask[row].bool()], reference,
                                       atol=2e-5, rtol=2e-4)
        cache = output.past_key_values
        # The short row crosses an anchor boundary independently of the long row.
        for _ in range(2):
            ids = torch.cat([ids, torch.randint(3, 59, (2, 1))], dim=1)
            mask = torch.cat([mask, torch.ones((2, 1), dtype=mask.dtype)], dim=1)
            prepared = model.prepare_inputs_for_generation(ids, attention_mask=mask,
                                                           past_key_values=cache, use_cache=True)
            output = model(**prepared)
            cache = output.past_key_values
            reference = torch.cat([model(input_ids=ids[row, mask[row].bool()][None],
                                         use_cache=False).logits[:, -1] for row in range(2)])
            torch.testing.assert_close(output.logits[:, -1], reference, atol=2e-5, rtol=2e-4)
            assert cache.hsd_seen_text_tokens_per_row.tolist() == mask.sum(-1).tolist()


@pytest.mark.parametrize("variant", ["safin1_4b", "safin1_35b_a3b"])
def test_fresh_remote_code_preserves_batch_causality_and_rejects_multi_token_suffix(variant, tmp_path):
    from transformers import AutoModelForCausalLM
    from march.safin1 import Safin1ForConditionalGeneration, Safin1MoeForConditionalGeneration
    from march.safin1.config_utils import copy_safin1_remote_code, normalize_safin1_config
    import json
    fixture = smoke_fixture()
    cls = Safin1ForConditionalGeneration if variant == "safin1_4b" else Safin1MoeForConditionalGeneration
    torch.manual_seed(71)
    torch.set_num_threads(2)
    config = fixture.tiny_config(variant, chunk_size=8)
    config.text_config.num_hidden_layers = 4
    config.text_config.layer_types = ["linear_attention", "full_attention"] * 2
    config.text_config.hsd_global_read_layers = "1-4"
    with fixture.reference_kernels(), fixture.remote_registry(), torch.no_grad():
        original = cls(config).eval()
        original.save_pretrained(tmp_path)
        raw = json.loads((tmp_path / "config.json").read_text())
        (tmp_path / "config.json").write_text(json.dumps(normalize_safin1_config(raw), indent=2))
        copy_safin1_remote_code(tmp_path, variant)
        loaded, info = AutoModelForCausalLM.from_pretrained(
            tmp_path, trust_remote_code=True, local_files_only=True,
            attn_implementation="eager", output_loading_info=True)
        loaded.eval()
        assert loaded.__class__.__module__.startswith("transformers_modules.")
        for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            assert not info.get(field)
        ids = torch.randint(3, 59, (2, 9))
        mask = torch.ones_like(ids)
        mask[1, :2] = 0
        ids[1, :2] = 0
        output = loaded(input_ids=ids, attention_mask=mask, use_cache=True)
        for row in range(2):
            compact = ids[row, mask[row].bool()][None]
            reference = loaded(input_ids=compact, use_cache=False).logits[0]
            torch.testing.assert_close(output.logits[row, mask[row].bool()], reference,
                                       atol=2e-5, rtol=2e-4)
        cache = output.past_key_values
        supplied = torch.cat([ids, torch.randint(3, 59, (2, 2))], dim=1)
        full_mask = torch.cat([mask, torch.ones((2, 2), dtype=mask.dtype)], dim=1)
        before = cache.hsd_seen_text_tokens_per_row.clone()
        with pytest.raises(NotImplementedError, match="one new decode token"):
            loaded.prepare_inputs_for_generation(supplied, attention_mask=full_mask,
                                                 past_key_values=cache, use_cache=True)
        torch.testing.assert_close(cache.hsd_seen_text_tokens_per_row, before, atol=0, rtol=0)

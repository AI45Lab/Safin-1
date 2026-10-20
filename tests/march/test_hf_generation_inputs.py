"""Unsupported cached suffixes must fail before advancing or altering cache."""
import importlib.util
from pathlib import Path

import pytest
import torch


def make_model(variant):
    from march.safin1 import Safin1ForConditionalGeneration, Safin1MoeForConditionalGeneration
    path = Path(__file__).parents[2] / "scripts/march/smoke_hf.py"
    spec = importlib.util.spec_from_file_location("march_generation_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    cls = Safin1ForConditionalGeneration if variant == "safin1_4b" else Safin1MoeForConditionalGeneration
    return fixture, cls


@pytest.mark.parametrize("variant", ["safin1_4b", "safin1_35b_a3b"])
@pytest.mark.parametrize("selector", [None, 1, 2, 0])
def test_multi_token_decode_rejected_without_cache_mutation(variant, selector):
    fixture, cls = make_model(variant)
    torch.manual_seed(29)
    torch.set_num_threads(2)
    with fixture.reference_kernels(), torch.no_grad():
        model = cls(fixture.tiny_config(variant, chunk_size=8)).eval()
        ids = torch.randint(3, 59, (1, 8))
        cache = model(input_ids=ids[:, :6], use_cache=True).past_key_values
        old_keys = cache.layers[1].keys.clone()
        old_conv = cache.layers[0].conv_states.clone()
        old_recurrent = cache.layers[0].recurrent_states.clone()
        kwargs = {} if selector is None else {"next_sequence_length": selector}
        with pytest.raises(NotImplementedError, match="one new decode token"):
            model.prepare_inputs_for_generation(ids, past_key_values=cache, use_cache=True, **kwargs)
        assert cache.hsd_seen_text_tokens == 6
        torch.testing.assert_close(cache.layers[1].keys, old_keys, atol=0, rtol=0)
        torch.testing.assert_close(cache.layers[0].conv_states, old_conv, atol=0, rtol=0)
        torch.testing.assert_close(cache.layers[0].recurrent_states, old_recurrent, atol=0, rtol=0)


@pytest.mark.parametrize("variant", ["safin1_4b", "safin1_35b_a3b"])
@pytest.mark.parametrize("mode", ["full_prefix", "one_token", "selected_history"])
def test_supported_cached_decode_matches_full_forward_at_anchor_boundary(variant, mode):
    fixture, cls = make_model(variant)
    torch.manual_seed(91)
    torch.set_num_threads(2)
    with fixture.reference_kernels(), torch.no_grad():
        model = cls(fixture.tiny_config(variant, chunk_size=8)).eval()
        ids = torch.randint(3, 59, (1, 8))
        cache = model(input_ids=ids[:, :7], use_cache=True).past_key_values
        supplied = ids if mode == "full_prefix" else ids[:, -1:] if mode == "one_token" else ids[:, -2:]
        kwargs = {"next_sequence_length": 1} if mode == "selected_history" else {}
        prepared = model.prepare_inputs_for_generation(supplied, past_key_values=cache, use_cache=True, **kwargs)
        assert "next_sequence_length" not in prepared
        output = model(**prepared)
        reference = model(input_ids=ids, use_cache=False)
        assert output.past_key_values.hsd_seen_text_tokens == 8
        torch.testing.assert_close(output.logits[:, -1], reference.logits[:, -1], atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("variant", ["safin1_4b", "safin1_35b_a3b"])
@pytest.mark.parametrize("mode", ["full_prefix", "one_token_full_mask"])
@pytest.mark.parametrize("all_linear", [False, True])
def test_generate_with_existing_anchor_cache_matches_fresh_generation(variant, mode, all_linear):
    fixture, cls = make_model(variant)
    torch.manual_seed(63)
    torch.set_num_threads(2)
    config = fixture.tiny_config(variant, chunk_size=8)
    if all_linear:
        config.text_config.layer_types = ["linear_attention"] * 2
    with fixture.reference_kernels(), torch.no_grad():
        model = cls(config).eval()
        ids = torch.randint(3, 59, (1, 10))
        cache = model(input_ids=ids[:, :9], use_cache=True).past_key_values
        assert cache.hsd_seen_expanded_tokens > cache.hsd_seen_text_tokens  # One inserted anchor.
        if all_linear:
            assert cache.get_seq_length() > cache.hsd_seen_text_tokens
        supplied = ids if mode == "full_prefix" else ids[:, -1:]
        options = dict(max_new_tokens=2, do_sample=False, eos_token_id=None, pad_token_id=0)
        actual = model.generate(input_ids=supplied, attention_mask=torch.ones_like(ids),
                                past_key_values=cache, **options)
        expected = model.generate(input_ids=ids, **options)
        torch.testing.assert_close(actual[:, -2:], expected[:, -2:], atol=0, rtol=0)

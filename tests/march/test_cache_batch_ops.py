"""Cache row transformations must preserve native state and MARCH sidecars."""
from types import SimpleNamespace
import importlib.util
from pathlib import Path

import pytest
import torch
from transformers.cache_utils import Cache, DynamicLayer, LinearAttentionLayer
from march.safin1.cache_utils import _patch_hsd_cache_get_seq_length, _reorder_hsd_sidecar_cache


def fixture_cache():
    linear = LinearAttentionLayer()
    linear.update_conv_state(torch.tensor([[[10.]], [[20.]]]))
    linear.update_recurrent_state(torch.tensor([[[[11.]]], [[[21.]]]]))
    full = DynamicLayer()
    full.update(torch.tensor([[[[30.]]], [[[40.]]]]), torch.tensor([[[[31.]]], [[[41.]]]]))
    linear.hsd_anchor_states = torch.tensor([[[12.]], [[22.]]])
    linear.hsd_anchor_boundaries = torch.tensor([4, 8])  # Shared, happens to have length B.
    linear.hsd_anchor_persistent_memory_mask = torch.tensor([[True], [False]])
    cache = Cache(layers=[linear, full])
    cache.hsd_seen_text_tokens_per_row = torch.tensor([7, 9])
    cache.hsd_seen_expanded_tokens_per_row = torch.tensor([8, 10])
    cache._hsd_layer_states = [linear]  # Alias must be visited only once.
    return _patch_hsd_cache_get_seq_length(cache)


@pytest.mark.parametrize("indices", [[1, 0], [1, 0, 1], [1], []])
def test_native_state_and_sidecars_selected_once(indices):
    cache = fixture_cache()
    ids = torch.tensor(indices, dtype=torch.long)
    cache.reorder_cache(ids)
    assert cache.layers[0].conv_states.flatten().tolist() == [([10., 20.][i]) for i in indices]
    assert cache.layers[0].recurrent_states.flatten().tolist() == [([11., 21.][i]) for i in indices]
    assert cache.layers[1].keys.flatten().tolist() == [([30., 40.][i]) for i in indices]
    assert cache.layers[0].hsd_anchor_states.flatten().tolist() == [([12., 22.][i]) for i in indices]
    assert cache.hsd_seen_text_tokens_per_row.tolist() == [([7, 9][i]) for i in indices]
    assert cache.hsd_seen_text_tokens == (max([7, 9][i] for i in indices) if indices else 0)
    assert cache.layers[0].hsd_anchor_boundaries.tolist() == [4, 8]


def test_repeat_then_prune_handles_linear_layers():
    cache = fixture_cache()
    cache.batch_repeat_interleave(2)
    assert cache.layers[0].conv_states.flatten().tolist() == [10., 10., 20., 20.]
    cache.batch_select_indices(torch.tensor([3, 0]))
    assert cache.layers[0].conv_states.flatten().tolist() == [20., 10.]
    assert cache.hsd_seen_text_tokens_per_row.tolist() == [9, 7]


def test_invalid_indices_do_not_mutate_cache():
    cache = fixture_cache()
    with pytest.raises(IndexError):
        cache.reorder_cache(torch.tensor([2]))
    assert cache.hsd_seen_text_tokens_per_row.tolist() == [7, 9]
    assert cache.layers[0].conv_states.flatten().tolist() == [10., 20.]


@pytest.mark.parametrize("variant", ["safin1_4b", "safin1_35b_a3b"])
def test_resized_cache_decode_matches_independent_rows(variant):
    path = Path(__file__).parents[2] / "scripts/march/smoke_hf.py"
    spec = importlib.util.spec_from_file_location("march_cache_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    from march.safin1 import Safin1ForConditionalGeneration, Safin1MoeForConditionalGeneration
    cls = Safin1ForConditionalGeneration if variant == "safin1_4b" else Safin1MoeForConditionalGeneration
    torch.manual_seed(42)
    torch.set_num_threads(2)
    with fixture.reference_kernels(), torch.no_grad():
        model = cls(fixture.tiny_config(variant, chunk_size=8)).eval()
        ids = torch.randint(3, 59, (2, 10))
        cache = model(input_ids=ids[:, :9], use_cache=True).past_key_values
        selected = torch.tensor([1, 0, 1])
        cache = model._reorder_cache(cache, selected)
        expanded = ids.index_select(0, selected)
        prepared = model.prepare_inputs_for_generation(expanded, past_key_values=cache, use_cache=True)
        actual = model(**prepared).logits[:, -1]
        expected = torch.cat([model(input_ids=row[None], use_cache=False).logits[:, -1] for row in expanded])
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        cache.batch_select_indices(torch.tensor([2]))
        assert cache.hsd_seen_text_tokens_per_row.shape == (1,)
        continuation = torch.cat([expanded[2:3], torch.tensor([[7]])], dim=1)
        prepared = model.prepare_inputs_for_generation(continuation, past_key_values=cache, use_cache=True)
        pruned = model(**prepared).logits[:, -1]
        expected = model(input_ids=continuation, use_cache=False).logits[:, -1]
        torch.testing.assert_close(pruned, expected, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("variant", ["safin1_4b", "safin1_35b_a3b"])
def test_batched_beam_generation_matches_individual_rows(variant):
    path = Path(__file__).parents[2] / "scripts/march/smoke_hf.py"
    spec = importlib.util.spec_from_file_location("march_beam_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    from march.safin1 import Safin1ForConditionalGeneration, Safin1MoeForConditionalGeneration
    cls = Safin1ForConditionalGeneration if variant == "safin1_4b" else Safin1MoeForConditionalGeneration
    torch.manual_seed(11)
    torch.set_num_threads(2)
    with fixture.reference_kernels(), torch.no_grad():
        model = cls(fixture.tiny_config(variant, chunk_size=8)).eval()
        ids = torch.randint(3, 59, (2, 9))
        kwargs = dict(num_beams=2, max_new_tokens=4, do_sample=False, eos_token_id=None, pad_token_id=0)
        actual = model.generate(input_ids=ids, **kwargs)
        expected = torch.cat([model.generate(input_ids=row[None], **kwargs) for row in ids])
        assert actual.shape == (2, 13)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)



def test_optional_cache_layer_lists_can_be_none():
    cache = SimpleNamespace(layers=None, _hsd_layer_states=None,
                            hsd_seen_text_tokens_per_row=torch.tensor([7, 9]))
    _reorder_hsd_sidecar_cache(cache, torch.tensor([1]))
    assert cache.hsd_seen_text_tokens_per_row.tolist() == [9]
    assert cache.hsd_seen_text_tokens == 9

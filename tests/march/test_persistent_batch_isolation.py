"""Learned memory is shared read-only; request caches must remain independent."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture
def smoke():
    path = Path(__file__).resolve().parents[2]/'scripts/march/smoke_hf.py'
    spec = importlib.util.spec_from_file_location('march_persistent_batch_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('variant', ['safin1_4b', 'safin1_35b_a3b'])
@pytest.mark.parametrize('batch', [1, 4, 8, 32])
def test_nonzero_memory_survives_interleaved_requests_and_cache_selection(variant, batch, smoke):
    from march.safin1 import Safin1ForConditionalGeneration, Safin1MoeForConditionalGeneration
    cls = Safin1ForConditionalGeneration if variant == 'safin1_4b' else Safin1MoeForConditionalGeneration
    torch.manual_seed(20261016)
    torch.set_num_threads(2)
    config = smoke.tiny_config(variant, chunk_size=8)
    config.text_config.hsd_persistent_memory_enabled = True
    config.text_config.hsd_persistent_memory_route_key_enabled = True
    config.text_config.hsd_persistent_memory_route_key_mode = 'dedicated'
    with smoke.reference_kernels(), torch.no_grad():
        model = cls(config).eval()
        memory = {name: value for name, value in model.named_parameters()
                  if name.endswith(('persistent_memory_state', 'persistent_memory_route_key'))}
        assert memory and any(name.endswith('persistent_memory_route_key') for name in memory)
        for value in memory.values():
            value.normal_(mean=0, std=0.1)
        before = {name: value.clone() for name, value in memory.items()}
        ids = torch.randint(3, 59, (batch, 9))
        mask = torch.ones_like(ids)
        if batch > 1:
            mask[1::2, :2] = 0
            ids[1::2, :2] = 0
        first = model(input_ids=ids, attention_mask=mask, use_cache=True)
        cache = first.past_key_values
        # Prove that the fixture actually reads the learned memory.
        zero = deepcopy(model)
        for name, value in zero.named_parameters():
            if name.endswith('persistent_memory_state'):
                value.zero_()
        without_memory = zero(input_ids=ids, attention_mask=mask, use_cache=False).logits
        assert (first.logits - without_memory).abs().max().item() > 1e-7
        del zero
        # Keep A's live cache while B crosses several different anchor boundaries.
        poison = torch.randint(3, 59, (max(2, batch // 2), 25))
        unrelated = model(input_ids=poison, use_cache=True)
        assert unrelated.past_key_values is not cache
        for name, value in memory.items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        # Duplicate/reorder requests, then prune them while continuing decoding.
        selected = torch.tensor(list(reversed(range(batch))) + [0])
        cache.reorder_cache(selected)
        ids, mask = ids[selected], mask[selected]
        for turn in range(2):
            ids = torch.cat([ids, torch.randint(3, 59, (ids.shape[0], 1))], dim=1)
            mask = torch.cat([mask, torch.ones((mask.shape[0], 1), dtype=mask.dtype)], dim=1)
            prepared = model.prepare_inputs_for_generation(ids, attention_mask=mask,
                                                           past_key_values=cache, use_cache=True)
            actual = model(**prepared)
            cache = actual.past_key_values
            expected = torch.cat([model(input_ids=ids[i, mask[i].bool()][None],
                                        use_cache=False).logits[:, -1] for i in range(ids.shape[0])])
            torch.testing.assert_close(actual.logits[:, -1], expected, atol=2e-5, rtol=2e-4)
            assert cache.hsd_seen_text_tokens_per_row.tolist() == mask.sum(-1).tolist()
            if turn == 0:
                keep = torch.tensor([ids.shape[0]-1, 0])
                cache.batch_select_indices(keep)
                ids, mask = ids[keep], mask[keep]
                model(input_ids=torch.randint(3, 59, (3, 17)), use_cache=True)
        for name, value in memory.items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)

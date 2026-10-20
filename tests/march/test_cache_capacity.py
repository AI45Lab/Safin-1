"""Rejected cache appends must not reserve pages or alter graph-visible state."""
from copy import deepcopy

import pytest
import torch

from march.inference.history_cache import HSDAnchorPageCache, HSDHistoryCacheSpec


def cache_fixture(limit='columns'):
    cache = HSDAnchorPageCache(page_m=2)
    cache.preallocate(spec=HSDHistoryCacheSpec(
        page_m=2, num_pages=1 if limit == 'pool' else 4, heads=1,
        value_dim=2, key_dim=3, states_dtype=torch.float32, keys_dtype=torch.float32,
    ), device=torch.device('cpu'))
    cache.preallocate_metadata(state_rows=2, max_pages_per_state=1 if limit == 'columns' else 4,
                               device=torch.device('cpu'))
    cache.states_pages.fill_(7)
    cache.keys_pages.fill_(8)
    cache.append(0, torch.ones(1, 2, 1, 2, 3), torch.ones(1, 2, 1, 3), torch.tensor([0, 512]))
    cache.assert_persistent_memory_ready([0])
    cache.export_page_table(torch.tensor([0]), device=torch.device('cpu'), state_indices_key=(0,))
    return cache


def snapshot(cache):
    buffers = {}
    for name in ('states_pages', 'keys_pages', 'boundary_pages', '_lengths_by_state', '_page_table_by_state'):
        value = getattr(cache, name)
        buffers[name] = None if value is None else (value.data_ptr(), value.clone())
    scalars = {name: getattr(cache, name) for name in (
        'heads', 'value_dim', 'key_dim', 'route_heads', 'route_dim', '_max_anchor_len', '_metadata_version',
    )}
    return (buffers, scalars, deepcopy(cache.entries), list(cache.free_pages),
            set(cache._persistent_memory_states), set(cache._invalid_persistent_memory_states),
            set(cache._persistent_memory_validated_batches), tuple(cache._export_page_table_cache))


def assert_unchanged(cache, before):
    after = snapshot(cache)
    assert before[1:] == after[1:]
    for name, value in before[0].items():
        current = after[0][name]
        if value is None:
            assert current is None
        else:
            assert current[0] == value[0], name
            torch.testing.assert_close(current[1], value[1], rtol=0, atol=0)


def append_one(cache, mode, index, keys=None, boundary=1024):
    states = torch.ones(1, 1, 1, 2, 3)
    keys = torch.ones(1, 1, 1, 3) if keys is None else keys
    boundaries = torch.tensor([boundary])
    if mode == 'append':
        return cache.append(index, states, keys, boundaries)
    if mode == 'try':
        return cache.try_append(index, states, keys, boundaries)
    if mode == 'batch_one':
        return cache.try_append_batch_one([index], states[:, 0], keys[:, 0], boundaries)
    return cache.try_append_batch([index], states[:, 0], keys[:, 0], boundaries, [0, 1])


@pytest.mark.parametrize('limit', ['pool', 'rows', 'columns'])
@pytest.mark.parametrize('mode', ['append', 'try', 'batch_one', 'batch'])
def test_capacity_failure_preserves_all_cache_state(limit, mode):
    cache = cache_fixture(limit)
    index = 2 if limit == 'rows' else 0
    before = snapshot(cache)
    assert not cache.can_append(index, 1)
    if mode == 'append':
        with pytest.raises(RuntimeError, match='capacity exceeded|cache is full'):
            append_one(cache, mode, index)
    else:
        assert append_one(cache, mode, index) is False
    assert_unchanged(cache, before)
    cache.assert_persistent_memory_ready([0])


def test_replacement_checks_metadata_capacity_and_reclaims_own_pages():
    cache = cache_fixture('columns')
    before = snapshot(cache)
    assert cache.can_replace(0, 2)
    assert not cache.can_replace(0, 3)
    assert not cache.can_replace(2, 1)
    assert_unchanged(cache, before)
    full = cache_fixture('pool')
    assert not full.can_append(0, 1)
    assert full.can_replace(0, 2)
    assert not full.can_replace(0, 3)


class ConversionFailure(torch.Tensor):
    def to(self, *args, **kwargs):
        raise RuntimeError('fixture conversion failure')


@pytest.mark.parametrize('mode', ['append', 'try', 'batch_one', 'batch'])
def test_conversion_failure_preserves_payloads_and_memory_markers(mode):
    cache = cache_fixture('rows')
    before = snapshot(cache)
    keys = torch.ones(1, 1, 1, 3).as_subclass(ConversionFailure)
    # A second zero boundary would invalidate persistent memory if recorded before conversion.
    with pytest.raises(RuntimeError, match='fixture conversion failure'):
        append_one(cache, mode, 0, keys=keys, boundary=0)
    assert_unchanged(cache, before)


def test_initial_append_conversion_failure_preserves_uninitialized_cache():
    cache = HSDAnchorPageCache(page_m=2)
    before = snapshot(cache)
    keys = torch.ones(1, 1, 1, 3).as_subclass(ConversionFailure)
    with pytest.raises(RuntimeError, match='fixture conversion failure'):
        append_one(cache, 'append', 0, keys=keys, boundary=0)
    assert_unchanged(cache, before)
    append_one(cache, 'append', 0, boundary=0)
    cache.assert_persistent_memory_ready([0])


@pytest.mark.parametrize('mode', ['append', 'try', 'batch_one', 'batch'])
def test_invalid_state_index_does_not_change_ownership(mode):
    cache = cache_fixture('rows')
    before = snapshot(cache)
    if mode in ('append', 'try'):
        with pytest.raises(ValueError, match='non-negative'):
            append_one(cache, mode, -1)
    else:
        assert append_one(cache, mode, -1) is False
    assert_unchanged(cache, before)


@pytest.mark.parametrize('mode', ['batch_one', 'batch'])
def test_mixed_valid_and_overflow_rows_reject_whole_batch(mode):
    cache = cache_fixture('rows')
    before = snapshot(cache)
    states, keys = torch.ones(2, 1, 2, 3), torch.ones(2, 1, 3)
    boundaries = torch.tensor([1024, 0])
    if mode == 'batch_one':
        assert not cache.try_append_batch_one([0, 2], states, keys, boundaries)
    else:
        assert not cache.try_append_batch([0, 2], states, keys, boundaries, [0, 1, 2])
    assert_unchanged(cache, before)


def test_valid_append_at_last_metadata_slot_remains_readable():
    cache = cache_fixture('rows')
    states = torch.arange(6, dtype=torch.float32).view(1, 1, 1, 2, 3)
    keys = torch.arange(3, dtype=torch.float32).view(1, 1, 1, 3)
    assert cache.try_append(1, states, keys, torch.tensor([0]))
    actual = cache.materialize(1)
    torch.testing.assert_close(actual[0], states, atol=0, rtol=0)
    torch.testing.assert_close(actual[1], keys, atol=0, rtol=0)
    cache.assert_persistent_memory_ready([0, 1])

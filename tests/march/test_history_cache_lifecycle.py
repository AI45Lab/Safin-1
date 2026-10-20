"""Dense history storage and persistent-memory ownership across page reuse."""
import pytest

torch = pytest.importorskip("torch")
from march.inference import paged_history_read as phr


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("route_heads", [1, 2])
def test_dense_pages_preserve_memory_across_append_and_reuse(fixed, route_heads):
    torch.manual_seed(41)
    cache = phr.HSDAnchorPageCache(page_m=2)
    if fixed:
        cache.preallocate(spec=phr.HSDHistoryCacheSpec(
            page_m=2, num_pages=12, heads=2, value_dim=3, key_dim=5,
            route_heads=route_heads, states_dtype=torch.float32,
            keys_dtype=torch.float32,
        ), device=torch.device("cpu"))
        cache.preallocate_metadata(state_rows=16, max_pages_per_state=6,
                                   device=torch.device("cpu"))

    expected = {}
    for state_id in (3, 7):
        states = torch.randn(1, 3, 2, 3, 5)
        keys = torch.randn(1, 3, route_heads, 5)
        boundaries = torch.tensor([0, 512, 1024])
        cache.append(state_id, states, keys, boundaries)
        expected[state_id] = [states, keys, boundaries]
    cache.assert_persistent_memory_ready([3, 7])

    # A single-token decode boundary, then an unequal-length packed prefill.
    states = torch.randn(2, 2, 3, 5)
    keys = torch.randn(2, route_heads, 5)
    boundaries = torch.tensor([1536, 1536])
    assert cache.try_append_batch_one([3, 7], states, keys, boundaries)
    for i, state_id in enumerate((3, 7)):
        expected[state_id][0] = torch.cat((expected[state_id][0], states[i:i+1, None]), 1)
        expected[state_id][1] = torch.cat((expected[state_id][1], keys[i:i+1, None]), 1)
        expected[state_id][2] = torch.cat((expected[state_id][2], boundaries[i:i+1]))
    states = torch.randn(3, 2, 3, 5)
    keys = torch.randn(3, route_heads, 5)
    boundaries = torch.tensor([2048, 2048, 2560])
    appended = cache.try_append_batch([3, 7], states, keys, boundaries, [0, 1, 3])
    assert appended is fixed
    if not appended:
        # The dynamic legacy cache falls back when an append needs new pages.
        cache.append(3, states[None, :1], keys[None, :1], boundaries[:1])
        cache.append(7, states[None, 1:], keys[None, 1:], boundaries[1:])
    for state_id, begin, end in ((3, 0, 1), (7, 1, 3)):
        expected[state_id][0] = torch.cat((expected[state_id][0], states[None, begin:end]), 1)
        expected[state_id][1] = torch.cat((expected[state_id][1], keys[None, begin:end]), 1)
        expected[state_id][2] = torch.cat((expected[state_id][2], boundaries[begin:end]))
        for actual, reference in zip(cache.materialize(state_id), expected[state_id]):
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    cache.assert_persistent_memory_ready([3, 7])
    table, lengths, _ = cache.export_page_table(torch.tensor([3, 7]), device=torch.device("cpu"))
    assert lengths.tolist() == [5, 6]
    assert table.shape[1] >= 3
    assert cache.allocated_page_bytes == 2 * (2 * 3 * 5 * 4 + route_heads * 5 * 4 + 8)

    assert cache.clear({3}) == 1
    cache.assert_persistent_memory_ready([7])
    with pytest.raises(RuntimeError):
        cache.assert_persistent_memory_ready([3])
    # Reuse the same slot/page for a new request; the previous memory must be gone.
    fresh_states, fresh_keys = torch.randn(1, 1, 2, 3, 5), torch.randn(1, 1, route_heads, 5)
    cache.append(3, fresh_states, fresh_keys, torch.tensor([0]))
    cache.assert_persistent_memory_ready([3, 7])
    actual_states, actual_keys, actual_boundaries = cache.materialize(3)
    torch.testing.assert_close(actual_states, fresh_states, atol=0, rtol=0)
    torch.testing.assert_close(actual_keys, fresh_keys, atol=0, rtol=0)
    assert actual_boundaries.tolist() == [0]
    for actual, reference in zip(cache.materialize(7), expected[7]):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)


def test_full_pool_append_does_not_corrupt_persistent_memory():
    cache = phr.HSDAnchorPageCache(page_m=2)
    cache.preallocate(spec=phr.HSDHistoryCacheSpec(
        page_m=2, num_pages=1, heads=1, value_dim=3, key_dim=5,
        states_dtype=torch.float32, keys_dtype=torch.float32,
    ), device=torch.device("cpu"))
    states, keys = torch.randn(1, 2, 1, 3, 5), torch.randn(1, 2, 1, 5)
    cache.append(0, states, keys, torch.tensor([0, 512]))
    assert not cache.try_append(0, states[:, :1], keys[:, :1], torch.tensor([1024]))
    cache.assert_persistent_memory_ready([0])
    actual = cache.materialize(0)
    for observed, reference in zip(actual, (states, keys, torch.tensor([0, 512]))):
        torch.testing.assert_close(observed, reference, atol=0, rtol=0)


@pytest.mark.parametrize("state_format", ["low_rank", "unknown"])
def test_unsupported_history_storage_fails_explicitly(state_format):
    with pytest.raises(ValueError, match="only.*dense"):
        phr.HSDAnchorPageCache(state_format=state_format)
    with pytest.raises(ValueError, match="only.*dense"):
        phr.HSDHistoryCacheSpec(page_m=2, num_pages=1, heads=1, value_dim=3,
                               key_dim=5, states_dtype=torch.float32,
                               keys_dtype=torch.float32, state_format=state_format)

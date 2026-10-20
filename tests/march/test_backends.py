"""Paged cache/routing oracle and optional CUDA kernel parity."""
import pytest

torch = pytest.importorskip("torch")
from march.inference import paged_history_read as phr


def test_paged_cache_append_release():
    cache = phr.HSDAnchorPageCache(page_m=4)
    states = torch.arange(1*9*2*3*5, dtype=torch.float32).view(1,9,2,3,5)
    keys = torch.randn(1,9,2,5)
    boundaries = torch.arange(9)
    cache.append(7, states[:,:5], keys[:,:5], boundaries[:5])
    cache.append(7, states[:,5:], keys[:,5:], boundaries[5:])
    for actual, expected in zip(cache.materialize(7), (states, keys, boundaries)):
        torch.testing.assert_close(actual, expected)
    assert cache.clear({7}) == 1
    assert cache.materialize(7) is None


@pytest.mark.parametrize("has_null", [False, True])
@pytest.mark.parametrize("visible", [0, 1, 3, 6])
def test_top4_history_read_against_direct_oracle(has_null, visible):
    torch.manual_seed(123)
    states = torch.randn(1,6,2,3,5)
    keys = torch.randn(1,6,2,5)
    boundaries = torch.arange(1,7)*512
    cache = phr.HSDAnchorPageCache(page_m=4)
    cache.append(1, states, keys, boundaries)
    table, lengths, _ = cache.export_page_table(torch.tensor([1]), device=torch.device('cpu'))
    read_q, route_q = torch.randn(1,2,5), torch.randn(1,2,5)
    null = torch.randn(1,2) if has_null else None
    actual = phr._paged_decode_top4_history_read_torch(
        cache=cache, read_q=read_q, router_q=route_q, page_table=table, anchor_lengths=lengths,
        start_indices=torch.tensor([visible*512]), route_scale=0.5, read_scale=0.25, null_logits=null)
    expected = torch.zeros(1,2,3)
    for head in range(2):
        logits = (keys[0,:visible,head] @ route_q[0,head])*0.5
        selected = torch.argsort(logits, descending=True)[:4]
        selected_logits = logits[selected]
        if has_null:
            weights = torch.softmax(torch.cat([selected_logits, null[0,head:head+1]]), 0)[:-1]
        else:
            weights = torch.softmax(selected_logits, 0)
        for weight, slot in zip(weights, selected):
            expected[0,head] += weight * (states[0,slot,head] @ read_q[0,head])*0.25
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_accelerated_entry_points_present():
    from march.kernels import march_chunk_gdn as chunk
    from march.kernels.march_global_read_topk import hsd_sparse_topk_global_read
    assert callable(chunk.hsd_flashqla_tile_gated_delta_rule_with_anchor_states)
    assert callable(chunk.hsd_flashqla_tile_gated_delta_rule_inference)
    assert callable(hsd_sparse_topk_global_read)


def test_flashqla_release_backends_and_geometries():
    from march.kernels import march_chunk_gdn as chunk

    assert chunk._resolve_flashqla_tile_bwd_backend("optimized") == "optimized"
    assert chunk._resolve_flashqla_tile_bwd_backend("reference") == "reference"
    for removed in ("direct_a", "native", "split"):
        with pytest.raises(RuntimeError, match="reference or optimized"):
            chunk._resolve_flashqla_tile_bwd_backend(removed)

    assert chunk._validate_flashqla_tile_geometry_backend(
        batch=1,
        heads=16,
        key_dim=128,
        value_dim=128,
        backward_backend="optimized",
    )
    assert not chunk._validate_flashqla_tile_geometry_backend(
        batch=1,
        heads=6,
        key_dim=192,
        value_dim=384,
        backward_backend="optimized",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("heads,head_dim", [(2, 32), (32, 128)])
def test_paged_cuda_matches_torch_reference(heads, head_dim):
    torch.manual_seed(123)
    device = torch.device('cuda')
    cache = phr.HSDAnchorPageCache(page_m=4)
    states = torch.randn(1,8,heads,head_dim,head_dim, device=device)
    keys = torch.randn(1,8,heads,head_dim, device=device)
    cache.append(1, states, keys, torch.arange(1,9,device=device)*512)
    read_q = torch.randn(1,heads,head_dim,device=device)
    route_q = torch.randn_like(read_q)
    starts = torch.tensor([3072],device=device)
    slots = torch.tensor([1],device=device)
    null = torch.randn(1,heads,device=device)
    table, lengths, _ = cache.export_page_table(slots, device=device)
    expected = phr._paged_decode_top4_history_read_torch(
        cache=cache, read_q=read_q, router_q=route_q, page_table=table, anchor_lengths=lengths,
        start_indices=starts, route_scale=head_dim**-0.5, read_scale=head_dim**-0.5, null_logits=null)
    actual = phr.paged_batched_decode_history_read(
        cache=cache, read_q=read_q, router_q=route_q, state_indices=slots, start_indices=starts,
        route_scale=head_dim**-0.5, read_scale=head_dim**-0.5, null_logits=null)
    assert actual is not None, "requested GPU backend did not run"
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize('page_m,anchors', [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1)])
@pytest.mark.parametrize('has_null', [False, True])
def test_top4_reference_handles_fewer_than_four_physical_slots(page_m, anchors, has_null):
    torch.manual_seed(41)
    cache = phr.HSDAnchorPageCache(page_m=page_m)
    states = torch.randn(1, anchors, 1, 2, 3)
    keys = torch.randn(1, anchors, 1, 3)
    cache.append(0, states, keys, torch.arange(1, anchors+1)*512)
    table, lengths, _ = cache.export_page_table(torch.tensor([0]), device=torch.device('cpu'))
    q = torch.randn(1, 1, 3)
    route = torch.randn_like(q)
    null = torch.randn(1, 1) if has_null else None
    actual = phr._paged_decode_top4_history_read_torch(
        cache=cache, read_q=q, router_q=route, page_table=table, anchor_lengths=lengths,
        start_indices=torch.tensor([anchors*512]), route_scale=0.5, read_scale=0.25, null_logits=null,
    )
    scores = (keys[0, :, 0] @ route[0, 0])*0.5
    weights = torch.softmax(torch.cat([scores, null[0]]) if has_null else scores, 0)[:anchors]
    expected = torch.einsum('m,mvk,k->v', weights, states[0, :, 0], q[0, 0])*0.25
    torch.testing.assert_close(actual[0, 0], expected, atol=1e-6, rtol=1e-5)

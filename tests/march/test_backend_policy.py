"""Per-call backend isolation and opt-in diagnostics, independent of CUDA availability."""
from concurrent.futures import ThreadPoolExecutor
import os

import pytest
import torch

from march.diagnostics.history_read import compare_fused_history_read
from march.inference.backend_policy import PagedHistoryReadPolicy
from march.inference import paged_history_read as phr


@pytest.mark.parametrize('backend,expected', [
    ('auto', 'auto'), ('', 'auto'), (' TORCH ', 'torch'), ('aten', 'torch'),
    ('paged_torch', 'torch'), ('tensor', 'torch'), ('triton', 'triton'),
    ('split_triton', 'triton'), ('fused', 'fused_triton'), ('fused_triton', 'fused_triton'),
])
def test_explicit_policy_overrides_environment(monkeypatch, backend, expected):
    monkeypatch.setenv('HSD_PAGED_HISTORY_READ_IMPL', 'invalid-global-default')
    monkeypatch.setenv('HSD_PAGED_HISTORY_VALIDATE_FUSED', '1')
    policy = PagedHistoryReadPolicy.resolve(backend=backend, validate_fused=False)
    assert policy.backend == expected
    assert not policy.validate_fused
    assert os.environ['HSD_PAGED_HISTORY_READ_IMPL'] == 'invalid-global-default'


def test_environment_remains_compatible_default(monkeypatch):
    monkeypatch.setenv('HSD_PAGED_HISTORY_READ_IMPL', 'split_triton')
    monkeypatch.setenv('HSD_PAGED_HISTORY_VALIDATE_FUSED', 'yes')
    assert PagedHistoryReadPolicy.resolve() == PagedHistoryReadPolicy('triton', True)
    with pytest.raises(TypeError, match='bool'):
        PagedHistoryReadPolicy.resolve(validate_fused='false')
    with pytest.raises(ValueError, match='backend'):
        PagedHistoryReadPolicy.resolve(backend='unavailable-engine')


def test_invalid_explicit_backend_fails_before_capability_check():
    with pytest.raises(ValueError, match='backend'):
        phr.paged_batched_decode_history_read(
            cache=phr.HSDAnchorPageCache(), read_q=torch.empty(1, 1, 3),
            router_q=torch.empty(1, 1, 3), state_indices=torch.tensor([0]),
            start_indices=torch.tensor([0]), route_scale=1.0, read_scale=1.0,
            backend='typo', validate_fused=False,
        )


class _HostTensorWithCudaCapability(torch.Tensor):
    # Host-only dispatch fixture: exercises splitting/selection, never launches CUDA.
    @property
    def is_cuda(self):
        return True


def test_split_batch_keeps_explicit_backend(monkeypatch):
    monkeypatch.setenv('HSD_PAGED_HISTORY_READ_IMPL', 'invalid-global-default')
    monkeypatch.setenv('HSD_PAGED_HISTORY_VALIDATE_FUSED', '1')
    monkeypatch.setenv('HSD_PAGED_HISTORY_MAX_BATCH', '1')
    monkeypatch.delenv('HSD_DISABLE_PAGED_HISTORY_READ', raising=False)
    monkeypatch.delenv('HSD_PAGED_HISTORY_SYNC_AFTER', raising=False)
    monkeypatch.delenv('VLLM_HSD_PAGED_HISTORY_TIMING', raising=False)
    torch.manual_seed(123)
    cache = phr.HSDAnchorPageCache(page_m=2)
    cache.append(7, torch.randn(1, 3, 2, 4, 3), torch.randn(1, 3, 2, 3), torch.tensor([1, 2, 3]))
    read_q = torch.randn(3, 2, 3).as_subclass(_HostTensorWithCudaCapability)
    route_q = torch.randn(3, 2, 3)
    slots, starts = torch.tensor([7, 7, 7]), torch.tensor([3, 2, 1])
    table, lengths, _ = cache.export_page_table(slots, device=torch.device('cpu'))
    expected = phr._paged_decode_top4_history_read_torch(
        cache=cache, read_q=read_q, router_q=route_q, page_table=table, anchor_lengths=lengths,
        start_indices=starts, route_scale=0.5, read_scale=0.25, null_logits=None,
    )
    actual = phr.paged_batched_decode_history_read(
        cache=cache, read_q=read_q, router_q=route_q, state_indices=slots, start_indices=starts,
        route_scale=0.5, read_scale=0.25, backend='torch', validate_fused=False,
    )
    assert actual is not None
    torch.testing.assert_close(actual, expected)
    assert os.environ['HSD_PAGED_HISTORY_READ_IMPL'] == 'invalid-global-default'


@pytest.mark.parametrize('outcome', ['tensor', 'none', 'error'])
def test_diagnostic_preserves_semantics_and_environment(monkeypatch, outcome):
    monkeypatch.setenv('HSD_PAGED_HISTORY_READ_IMPL', 'torch')
    monkeypatch.setenv('HSD_PAGED_HISTORY_VALIDATE_FUSED', '1')
    before = dict(os.environ)
    metrics = {}
    kwargs = {'read_topk': 0, 'persistent_memory_enabled': False, 'memory_metrics': metrics,
              'state_indices_key': (7,), 'backend': 'triton', 'validate_fused': True}
    events = []

    def read(**values):
        assert values['backend'] == 'fused_triton'
        assert values['validate_fused'] is False
        assert values['read_topk'] == 0
        assert values['memory_metrics'] is metrics
        assert values['state_indices_key'] == (7,)
        assert dict(os.environ) == before
        if outcome == 'error':
            raise RuntimeError('nested read failed')
        return None if outcome == 'none' else torch.tensor([1.0, 2.25])

    def compare():
        compare_fused_history_read(torch.tensor([1.0, 2.0]), read=read, read_kwargs=kwargs,
                                  record_event=lambda event, **payload: events.append((event, payload)),
                                  batch=1, max_pages=2, max_len=3)

    if outcome == 'error':
        with pytest.raises(RuntimeError, match='nested read failed'):
            compare()
    else:
        compare()
        assert events[0][1]['status'] == ('fused_none' if outcome == 'none' else 'compared')
        if outcome == 'tensor':
            assert events[0][1]['max_abs'] == 0.25
            assert events[0][1]['mean_abs'] == 0.125
            assert events[0][1]['finite']
    assert dict(os.environ) == before
    assert kwargs['backend'] == 'triton' and kwargs['validate_fused'] is True


def test_concurrent_diagnostic_does_not_redirect_other_calls(monkeypatch):
    monkeypatch.setenv('HSD_PAGED_HISTORY_READ_IMPL', 'torch')
    monkeypatch.setenv('HSD_PAGED_HISTORY_VALIDATE_FUSED', '0')

    def run(index):
        def read(**kwargs):
            assert PagedHistoryReadPolicy.resolve().backend == 'torch'
            assert kwargs['backend'] == 'fused_triton'
            return torch.tensor([float(index)])
        compare_fused_history_read(torch.tensor([float(index)]), read=read,
                                  read_kwargs={'persistent_memory_enabled': True, 'read_topk': 4},
                                  record_event=lambda *args, **kwargs: None,
                                  batch=1, max_pages=1, max_len=1)
        return PagedHistoryReadPolicy.resolve()

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(policy == PagedHistoryReadPolicy('torch', False) for policy in pool.map(run, range(32)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA GPU required')
@pytest.mark.parametrize('backend', ['torch', 'triton', 'fused_triton'])
@pytest.mark.parametrize('memory,topk', [(False, 4), (False, 0), (True, 4)])
def test_explicit_cuda_backend_matches_reference(monkeypatch, backend, memory, topk):
    monkeypatch.setenv('HSD_PAGED_HISTORY_READ_IMPL', 'invalid-global-default')
    monkeypatch.setenv('HSD_PAGED_HISTORY_VALIDATE_FUSED', '1')
    torch.manual_seed(123)
    device = torch.device('cuda')
    heads, dim = 2, 32
    cache = phr.HSDAnchorPageCache(page_m=4)
    keys = torch.randn(1, 8, heads, dim, device=device)
    if memory:
        keys = torch.nn.functional.normalize(keys, dim=-1)
    cache.append(1, torch.randn(1, 8, heads, dim, dim, device=device), keys,
                 torch.arange(8, device=device) * 512)
    read_q, route_q = torch.randn(1, heads, dim, device=device), torch.randn(1, heads, dim, device=device)
    if memory:
        route_q = torch.nn.functional.normalize(route_q, dim=-1)
    slots, starts = torch.tensor([1], device=device), torch.tensor([4096], device=device)
    table, lengths, _ = cache.export_page_table(slots, device=device)
    if memory:
        # Boundary zero identifies the persistent anchor through the normal append API.
        cache.assert_persistent_memory_ready([1])
    oracle = (phr._paged_decode_top4_memory_history_read_torch if memory else
              phr._paged_decode_read_all_history_read_torch if topk == 0 else
              phr._paged_decode_top4_history_read_torch)
    null = torch.randn(1, heads, device=device)
    expected = oracle(cache=cache, read_q=read_q, router_q=route_q, page_table=table,
                      anchor_lengths=lengths, start_indices=starts, route_scale=dim**-0.5,
                      read_scale=dim**-0.5, null_logits=null)
    actual = phr.paged_batched_decode_history_read(
        cache=cache, read_q=read_q, router_q=route_q, state_indices=slots, start_indices=starts,
        route_scale=dim**-0.5, read_scale=dim**-0.5, null_logits=null,
        backend=backend, validate_fused=False, read_topk=topk, persistent_memory_enabled=memory,
    )
    assert actual is not None, f'{backend} did not run'
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


def test_diagnostic_rejects_graph_capture_before_nested_read(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    value = torch.tensor([1.0]).as_subclass(_HostTensorWithCudaCapability)
    with pytest.raises(RuntimeError, match="outside CUDA graph"):
        compare_fused_history_read(
            value, read=lambda **kwargs: pytest.fail("nested read during capture"), read_kwargs={},
            record_event=lambda *args, **kwargs: pytest.fail("host event during capture"),
            batch=1, max_pages=1, max_len=1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_cuda_split_diagnostic_uses_local_fused_override(monkeypatch):
    from march.inference import paged_decode
    monkeypatch.setenv("HSD_PAGED_HISTORY_READ_IMPL", "torch")
    monkeypatch.setenv("HSD_PAGED_HISTORY_VALIDATE_FUSED", "1")
    events = []
    monkeypatch.setattr(paged_decode, "_record_paged_history_event",
                        lambda event, **payload: events.append((event, payload)))
    cache = phr.HSDAnchorPageCache(page_m=4)
    torch.manual_seed(17)
    cache.append(1, torch.randn(1, 8, 2, 32, 32, device="cuda"),
                 torch.randn(1, 8, 2, 32, device="cuda"), torch.arange(1, 9, device="cuda")*512)
    out = phr.paged_batched_decode_history_read(
        cache=cache, read_q=torch.randn(1, 2, 32, device="cuda"),
        router_q=torch.randn(1, 2, 32, device="cuda"), state_indices=torch.tensor([1], device="cuda"),
        start_indices=torch.tensor([4096], device="cuda"), route_scale=32**-0.5, read_scale=32**-0.5,
        backend="triton", validate_fused=True,
    )
    assert out is not None and torch.isfinite(out).all()
    comparisons = [payload for event, payload in events if event == "paged_history_fused_validate"]
    assert len(comparisons) == 1 and comparisons[0]["status"] == "compared"
    assert comparisons[0]["finite"] and comparisons[0]["max_abs"] < 2e-4
    assert os.environ["HSD_PAGED_HISTORY_READ_IMPL"] == "torch"
    assert os.environ["HSD_PAGED_HISTORY_VALIDATE_FUSED"] == "1"


@pytest.mark.parametrize('enabled', [False, True])
def test_optional_fallback_events_explain_unsupported_cpu(monkeypatch, enabled):
    from march.inference import paged_decode
    monkeypatch.setenv('HSD_PAGED_HISTORY_BACKEND_EVENTS', '1')
    events = []
    monkeypatch.setattr(paged_decode, '_record_paged_history_event',
                        lambda event, **payload: events.append((event, payload)))
    result = phr.paged_batched_decode_history_read(
        cache=phr.HSDAnchorPageCache(), read_q=torch.ones(1, 1, 3), router_q=torch.ones(1, 1, 3),
        state_indices=torch.tensor([0]), start_indices=torch.tensor([512]),
        route_scale=1.0, read_scale=1.0, backend='torch', record_backend=enabled,
    )
    assert result is None
    if enabled:
        assert events == [('paged_history_backend', dict(status='fallback', requested='torch',
                         reason='cuda_required', read_topk=4, persistent_memory=False))]
    else:
        assert not events


def test_optional_selection_events_report_actual_reference_path(monkeypatch):
    from march.inference import paged_decode
    events = []
    monkeypatch.setattr(paged_decode, '_record_paged_history_event',
                        lambda event, **payload: events.append((event, payload)))
    cache = phr.HSDAnchorPageCache(page_m=2)
    cache.append(0, torch.ones(1, 1, 1, 2, 3), torch.ones(1, 1, 1, 3), torch.tensor([512]))
    result = phr.paged_batched_decode_history_read(
        cache=cache, read_q=torch.ones(1, 1, 3).as_subclass(_HostTensorWithCudaCapability),
        router_q=torch.ones(1, 1, 3), state_indices=torch.tensor([0]), start_indices=torch.tensor([512]),
        route_scale=1.0, read_scale=1.0, backend='torch', validate_fused=False, record_backend=True,
    )
    assert result is not None
    assert events == [('paged_history_backend', dict(status='selected', requested='torch',
                     path='torch', read_topk=4, persistent_memory=False))]

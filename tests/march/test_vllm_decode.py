"""Real decode helpers: recurrence/state layout, admission and Graph dispatch."""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "third_party/vllm/vllm/model_executor/models/march"


@pytest.fixture(scope="module")
def helpers():
    name = "_safin1_decode_contract"
    spec = importlib.util.spec_from_file_location(
        name, PACKAGE / "__init__.py", submodule_search_locations=[str(PACKAGE)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / "third_party/vllm"))
        spec.loader.exec_module(module)
        yield NS(
            decode=importlib.import_module(name + ".decode"),
            graph=importlib.import_module(name + ".decode_graph"),
        )
    for key in list(sys.modules):
        if key == name or key.startswith(name + "."):
            del sys.modules[key]


@pytest.mark.parametrize("backend", ["packed", "recurrent"])
@pytest.mark.parametrize("boundary", [False, True])
def test_cached_step_preserves_inplace_state_and_non_square_anchor_layout(
    helpers, monkeypatch, backend, boundary
):
    layer = helpers.decode.CachedDecodeMixin()
    layer.head_k_dim, layer.head_v_dim = 2, 3
    layer.hsd_decode_backend = backend
    layer.A_log, layer.dt_bias = torch.zeros(1), torch.zeros(1)
    native = torch.arange(24, dtype=torch.float32).reshape(4, 1, 3, 2)
    old = native.clone()
    layer._ssm_cache_for_kernel = lambda count: native
    layer._single_state_index_tensor = lambda index, device: torch.tensor(
        [index], dtype=torch.int32, device=device
    )
    calls = []

    def step(**kwargs):
        assert kwargs["initial_state"] is native
        assert kwargs["ssm_state_indices"].tolist() == [2]
        native[2].add_(1)
        out = native[2].sum(-1).view(1, 1, 1, 3)
        calls.append(kwargs)
        return out

    def packed(**kwargs):
        kwargs["out"].copy_(step(**kwargs))

    def recurrent(**kwargs):
        return step(**kwargs), None

    monkeypatch.setattr(
        helpers.decode, "fused_recurrent_gated_delta_rule_packed_decode", packed
    )
    monkeypatch.setattr(helpers.decode, "fused_recurrent_gated_delta_rule", recurrent)
    q = torch.ones(1, 1, 1, 2)
    v = torch.ones(1, 1, 1, 3)
    gates = torch.ones(1, 1, 1)
    aq = torch.ones(1, int(boundary), 1, 2)
    bounds = torch.tensor([512] if boundary else [], dtype=torch.long)
    text, anchor, snapshots, state = layer._run_hsd_decode_vllm(
        torch.ones(1, 1, 7), gates, gates, q, q, v, gates, gates, aq, bounds, 511, 2
    )
    expected = (old[2:3] + 1).transpose(-1, -2).contiguous()
    torch.testing.assert_close(state, expected, atol=0, rtol=0)
    torch.testing.assert_close(native[[0, 1, 3]], old[[0, 1, 3]], atol=0, rtol=0)
    torch.testing.assert_close(
        text.flatten(), (old[2] + 1).sum(-1).flatten(), atol=0, rtol=0
    )
    assert snapshots.shape == (1, int(boundary), 1, 2, 3)
    assert anchor.shape == (1, int(boundary), 1, 3)
    if boundary:
        torch.testing.assert_close(snapshots[:, 0], expected, atol=0, rtol=0)
        torch.testing.assert_close(
            anchor[:, 0], expected.sum(-2) * 2**-0.5, atol=0, rtol=0
        )
    assert len(calls) == 1
    assert calls[0]["use_qk_l2norm_in_kernel"] == (backend == "packed")


@pytest.mark.parametrize(
    "batch,length,state_index,error",
    [
        (2, 1, 2, NotImplementedError),
        (1, 2, 2, NotImplementedError),
        (1, 1, 0, RuntimeError),
    ],
)
def test_bad_cached_step_rejected_before_native_cache_access(
    helpers, batch, length, state_index, error
):
    layer = helpers.decode.CachedDecodeMixin()
    layer.head_k_dim = layer.head_v_dim = 2
    layer._ssm_cache_for_kernel = lambda count: pytest.fail(
        "invalid request reached native cache"
    )
    q = torch.ones(batch, length, 1, 2)
    with pytest.raises(error):
        layer._run_hsd_decode_vllm(
            q, q, q, q, q, q, q, q, q[:, :0], torch.tensor([]), 511, state_index
        )


@pytest.mark.parametrize(
    "text,past,expanded,padded,mask_size,expected",
    [
        ([1], [511], [1], False, None, True),
        ([1], [511], [2], False, None, True),
        ([2], [511], [2], False, None, False),
        ([1], [0], [1], False, None, False),
        ([], [], [], False, None, False),
        ([1], [511], [0], False, None, False),
        ([1, 1], [511, 512], [1, 1], True, 2, True),
        ([1, 1], [511, 512], [1, 1], True, 1, False),
        ([1], [511], [1], True, None, False),
    ],
)
def test_decode_admission_preserves_anchor_and_padding_contract(
    helpers, text, past, expanded, padded, mask_size, expected
):
    layout = NS(
        req_text_lens=text,
        past_text_lens=past,
        req_expanded_lens=expanded,
        padded_static_decode=padded,
        decode_valid_mask=None
        if mask_size is None
        else torch.ones(mask_size, dtype=torch.bool),
    )
    assert (
        helpers.decode.CachedDecodeMixin()._layout_is_cached_decode(layout) == expected
    )


@pytest.mark.parametrize("split", [False, True])
def test_graph_dispatch_preserves_fixed_buffer_identity(helpers, monkeypatch, split):
    monkeypatch.setenv("VLLM_HSD_GRAPH_V3_SPLIT_CORE", str(int(split)))
    monkeypatch.setattr(helpers.graph, "_encode_layer_name", lambda prefix: 123)
    layer = helpers.graph.GraphDecodeMixin()
    layer.prefix = "fixture"
    conv, recurrent = torch.zeros(2, 3, 4), torch.zeros(2, 1, 2, 3)
    layer._hsd_graph_v3_state_tensors = lambda: (conv, recurrent)
    inputs = dict(
        hidden_states=torch.ones(2, 4),
        output=torch.empty(2, 4),
        state_indices=torch.tensor([1, 0]),
        history_state_indices=torch.tensor([2, 0]),
        past_lens=torch.tensor([511, 0]),
        valid_mask=torch.tensor([True, False]),
    )
    calls = []
    layer._forward_hsd_graph_v3_split_core = lambda **kwargs: calls.append(
        ("split", kwargs)
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "hsd_static_decode_attention",
        lambda *args: calls.append(("custom", args)),
        raising=False,
    )
    pointers = [t.data_ptr() for t in [conv, recurrent, *inputs.values()]]
    layer.forward_hsd_graph_v3(**inputs)
    assert len(calls) == 1 and calls[0][0] == ("split" if split else "custom")
    if split:
        assert (
            calls[0][1]["conv_cache"] is conv
            and calls[0][1]["recurrent_cache"] is recurrent
        )
        assert all(calls[0][1][key] is value for key, value in inputs.items())
    else:
        assert all(
            a is b
            for a, b in zip(calls[0][1][:-1], [*inputs.values(), conv, recurrent])
        )
        assert calls[0][1][-1] == 123
    assert pointers == [t.data_ptr() for t in [conv, recurrent, *inputs.values()]]


@pytest.mark.parametrize("exact", [False, True])
def test_graph_history_query_keeps_raw_alias_or_normalized_grouped_heads(
    helpers, exact
):
    layer = helpers.graph.GraphDecodeMixin()
    layer.hsd_graph_v3_exact_read_q = exact
    layer.hsd_post_conv_backend = "torch"
    layer.head_k_dim = 2
    mixed = torch.tensor([[3.0, 4.0, 0.0, 0.0, 9.0, 9.0]])
    read, raw, heads = layer._graph_v3_history_read_query(
        mixed, local_k_heads=2, local_v_heads=4
    )
    assert raw == (not exact)
    if not exact:
        assert read is mixed and heads == 2
    else:
        assert heads is None and read.shape == (1, 4, 2)
        expected = torch.tensor([3.0, 4.0]) / torch.sqrt(torch.tensor(25.0 + 1e-6))
        torch.testing.assert_close(read[0, :2], expected.expand(2, 2))
        assert torch.equal(read[0, 2:], torch.zeros(2, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_fused_graph_read_query_matches_unfused(helpers, dtype):
    layer = helpers.graph.GraphDecodeMixin()
    layer.hsd_graph_v3_exact_read_q = True
    layer.head_k_dim = 128
    mixed = torch.randn(2, (2 * 2 + 4) * 128, device="cuda", dtype=dtype)
    layer.hsd_post_conv_backend = "torch"
    expected, _, _ = layer._graph_v3_history_read_query(
        mixed, local_k_heads=2, local_v_heads=4
    )
    layer.hsd_post_conv_backend = "fused"
    actual, raw, heads = layer._graph_v3_history_read_query(
        mixed, local_k_heads=2, local_v_heads=4
    )
    assert not raw and heads is None
    torch.testing.assert_close(
        actual, expected, atol=2e-3 if dtype == torch.bfloat16 else 1e-5, rtol=2e-3
    )

"""Direct numerical oracles for every additional public MARCH operator family.

No model weights or datasets. Forward-only serving kernels run under no_grad;
training kernels must also match every differentiable input gradient.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def close(actual, expected, *, atol=2e-2, rtol=2e-2):
    assert actual is not None, "requested CUDA operator fell back"
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("has_null,local_prob", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dense_packed_read_all_input_gradients(monkeypatch, dtype, has_null, local_prob):
    from march.kernels.march_global_read import hsd_fused_global_read
    monkeypatch.setenv("HSD_FUSED_GLOBAL_READ", "1")
    torch.manual_seed(102)
    length, heads, dim, anchors = 40, 32, 128, 4
    source = [torch.randn(1, length, heads, dim, device="cuda", dtype=dtype)*.2,
              torch.randn(1, length, heads, dim, device="cuda", dtype=dtype)*.2,
              torch.randn(1, anchors, heads, dim, dim, device="cuda")*.1,
              torch.randn(1, anchors, heads, dim, device="cuda", dtype=dtype)*.2]
    if has_null:
        source.append(torch.randn(1, length, heads, 1, device="cuda")*.1)
    inputs = [x.detach().requires_grad_() for x in source]
    oracle_inputs = [x.detach().clone().requires_grad_() for x in source]
    common = dict(boundary_counts=torch.tensor([4, 12, 24, 32], device="cuda"),
                  text_cu_seqlens=torch.tensor([0, 19, 40], device="cuda"),
                  anchor_cu_seqlens=torch.tensor([0, 2, 4], device="cuda"),
                  route_scale=dim**-.5, read_scale=dim**-.5, return_local_prob=local_prob)
    def run(values, backend):
        return hsd_fused_global_read(read_q=values[0], router_q=values[1], states=values[2],
                                    anchor_k=values[3], null_logits=values[4] if has_null else None,
                                    backend=backend, **common)
    actual, expected = run(inputs, "triton"), run(oracle_inputs, "reference")
    assert actual is not None, "dense Triton path did not execute"
    actual = actual if local_prob else (actual,)
    expected = expected if local_prob else (expected,)
    for got, want in zip(actual, expected, strict=True):
        close(got, want, atol=2e-3)
    cotangents = [torch.randn_like(x) for x in expected]
    ga = torch.autograd.grad(actual, inputs, cotangents)
    ge = torch.autograd.grad(expected, oracle_inputs, cotangents)
    for got, want in zip(ga, ge, strict=True):
        close(got, want, atol=6e-3, rtol=4e-2)


def read_oracle(q, rq, states, keys, boundaries, positions, null=None, topk=None):
    scores = torch.einsum("bthk,bmhk->bthm", rq.float(), keys.float()) * q.shape[-1]**-.5
    visible = boundaries[None, :] <= positions[:, None]
    scores = scores.masked_fill(~visible[None, :, None, :], -torch.inf)
    if topk is not None:
        values, selected = scores.topk(min(topk, scores.shape[-1]), dim=-1)
        scores = torch.full_like(scores, -torch.inf).scatter(-1, selected, values)
    if null is not None:
        scores = torch.cat([scores, null.float()], dim=-1)
    has_any = torch.isfinite(scores).any(-1, keepdim=True)
    weights = torch.softmax(torch.where(has_any, scores, torch.zeros_like(scores)), -1)
    weights = torch.where(has_any, weights, torch.zeros_like(weights))
    local = weights[..., -1:] if null is not None else None
    weights = weights[..., :states.shape[1]]
    reads = torch.einsum("bthk,bmhkv->bthmv", q.float(), states.float()) * q.shape[-1]**-.5
    return torch.einsum("bthm,bthmv->bthv", weights, reads).to(q.dtype), local


@pytest.mark.parametrize("backend", ["streaming", "prob"])
@pytest.mark.parametrize("has_null", [False, True])
@pytest.mark.parametrize("visible", [0, 3, 6])
@torch.no_grad()
def test_dense_decode_oracle(monkeypatch, backend, has_null, visible):
    from march.kernels.march_global_read import hsd_fused_global_read_decode
    monkeypatch.setenv("HSD_FUSED_GLOBAL_READ_DECODE", "1")
    monkeypatch.setenv("HSD_FUSED_GLOBAL_READ_DECODE_BACKEND", backend)
    torch.manual_seed(103)
    q = torch.randn(2, 1, 32, 128, device="cuda", dtype=torch.bfloat16)*.2
    rq = torch.randn_like(q)
    states = torch.randn(2, 6, 32, 128, 128, device="cuda")*.1
    keys = torch.randn(2, 6, 32, 128, device="cuda", dtype=q.dtype)*.2
    boundaries = torch.arange(1, 7, device="cuda")*512
    null = torch.randn(2, 1, 32, 1, device="cuda") if has_null else None
    expected, local = read_oracle(q, rq, states, keys, boundaries,
                                  torch.tensor([visible*512], device="cuda"), null)
    actual = hsd_fused_global_read_decode(read_q=q, router_q=rq, states=states, anchor_k=keys,
        boundary_counts=boundaries, start_index=visible*512, null_logits=null,
        route_scale=128**-.5, read_scale=128**-.5, return_local_prob=has_null)
    if has_null:
        assert actual is not None
        close(actual[0], expected, atol=2e-3)
        close(actual[1], local, atol=2e-4)
    else:
        close(actual, expected, atol=2e-3)


@pytest.mark.parametrize("length", [1, 17])
@pytest.mark.parametrize("visible", [0, 3, 6])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.no_grad()
def test_hf_top4_read_oracle(monkeypatch, length, visible, dtype):
    from march.safin1.fused_history_read import fused_decode_top4_history_read, fused_prefill_top4_history_read
    monkeypatch.setenv("HSD_DISABLE_FUSED_HISTORY_READ", "0")
    monkeypatch.setenv("HSD_DISABLE_FUSED_HISTORY_READ_DECODE", "0")
    monkeypatch.setenv("HSD_DISABLE_FUSED_HISTORY_READ_PREFILL", "0")
    torch.manual_seed(104)
    q = torch.randn(2, length, 32, 128, device="cuda", dtype=dtype)*.2
    rq = torch.randn_like(q)
    states = torch.randn(2, 6, 32, 128, 128, device="cuda")*.1
    keys = torch.randn(2, 6, 32, 128, device="cuda", dtype=dtype)*.2
    boundaries = torch.arange(1, 7, device="cuda")*512
    positions = visible*512 + torch.arange(length, device="cuda")
    expected, _ = read_oracle(q, rq, states, keys, boundaries, positions, topk=4)
    op = fused_decode_top4_history_read if length == 1 else fused_prefill_top4_history_read
    close(op(read_q=q, router_q=rq, states=states, anchor_k=keys, boundary_counts=boundaries,
             start_index=visible*512, route_scale=128**-.5, read_scale=128**-.5), expected, atol=2e-3)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("dim", [96, 128])
@pytest.mark.parametrize("activation", ["silu", "swish", "sigmoid"])
@torch.no_grad()
def test_fused_epilogues_oracle(dtype, dim, activation):
    from march.inference.paged_epilogue import fused_history_scale_add_, fused_valid_mask_copy_, fused_strided_gate_rmsnorm
    torch.manual_seed(105)
    packed = torch.randn(3, 32, 3, dim, device="cuda", dtype=dtype)
    x, gate = packed[:, :, 0], packed[:, :, 1]
    weight = torch.randn(dim, device="cuda", dtype=dtype)
    eps = 1e-6
    normalized = x.float()*torch.rsqrt(x.float().square().mean(-1, keepdim=True)+eps)*weight.float()
    activated = gate.float().sigmoid() if activation == "sigmoid" else torch.nn.functional.silu(gate.float())
    close(fused_strided_gate_rmsnorm(x, gate, weight, eps=eps, activation=activation),
          (normalized*activated).to(dtype), atol=2e-3 if dtype != torch.float32 else 2e-6)
    base = torch.randn(3, 32, dim, device="cuda", dtype=dtype)
    history = torch.randn_like(base)
    expected = (base.float()+history.float()*.3).to(dtype)
    assert fused_history_scale_add_(base, history, torch.tensor(.3, device="cuda"))
    close(base, expected, atol=2e-3 if dtype != torch.float32 else 2e-6)
    source = base.flatten(1).contiguous()
    output = torch.empty_like(source)
    valid = torch.tensor([True, False, True], device="cuda")
    assert fused_valid_mask_copy_(output, source, valid)
    close(output, torch.where(valid[:, None], source, 0), atol=0, rtol=0)


@pytest.mark.parametrize("anchors", [3, 7])
@pytest.mark.parametrize("memory", [False, True])
@pytest.mark.parametrize("selection", ["blocked", "selected", "uniform"])
@torch.no_grad()
def test_paged_prefill_oracle(memory, selection, anchors):
    from march.inference import paged_history_read as p
    from march.inference.paged_prefill import paged_blocked_prefill_history_read
    torch.manual_seed(106)
    cache = p.HSDAnchorPageCache(page_m=4)
    heads, dim = 32, 128
    for row in range(2):
        cache.append(row, torch.randn(1, anchors, heads, dim, dim, device="cuda")*.1,
                     torch.randn(1, anchors, heads, dim, device="cuda", dtype=torch.bfloat16)*.2,
                     (torch.arange(anchors, device="cuda") + (0 if memory else 1))*512)
    q = torch.randn(9, heads, dim, device="cuda", dtype=torch.bfloat16)*.2
    rq = torch.randn_like(q)
    if selection == "uniform":
        rq[:4].copy_(rq[0].clone());rq[4:].copy_(rq[4].clone())
    null = torch.randn(9, heads, device="cuda", dtype=q.dtype)
    states = torch.tensor([0]*4+[1]*5, device="cuda")
    starts = torch.full((9,), 4096, device="cuda", dtype=torch.long)
    starts[0] = 0
    valid = torch.ones(9, device="cuda", dtype=torch.bool);valid[-1] = False
    history = p.paged_batched_decode_history_read(cache=cache, read_q=q, router_q=rq,
        state_indices=states, start_indices=starts, route_scale=dim**-.5, read_scale=dim**-.5,
        null_logits=null, persistent_memory_enabled=memory, backend="torch")
    base = torch.randn_like(q)
    expected = (base.float()+history.float()*.5).to(q.dtype)
    expected = torch.where(valid[:, None, None], expected, base)
    actual = paged_blocked_prefill_history_read(cache=cache, read_q=q, router_q=rq,
        token_state_indices=states, request_state_indices=torch.tensor([0, 1], device="cuda"),
        text_offsets=torch.tensor([0, 4, 9], device="cuda"), start_indices=starts, valid_mask=valid,
        max_text_len=5, route_scale=dim**-.5, read_scale=dim**-.5, base_out=base,
        history_scale=torch.tensor(.5, device="cuda"), null_logits=null, block_t=2,
        arbitrary_top4=selection != "blocked", reuse_uniform_top4=selection == "uniform",
        persistent_memory_enabled=memory)
    close(actual, expected, atol=2e-2)
    # Check the history contribution without a large residual hiding errors.
    isolated = paged_blocked_prefill_history_read(cache=cache, read_q=q, router_q=rq,
        token_state_indices=states, request_state_indices=torch.tensor([0, 1], device="cuda"),
        text_offsets=torch.tensor([0, 4, 9], device="cuda"), start_indices=starts, valid_mask=valid,
        max_text_len=5, route_scale=dim**-.5, read_scale=dim**-.5, base_out=torch.zeros_like(q),
        history_scale=torch.tensor(.5, device="cuda"), null_logits=null, block_t=2,
        arbitrary_top4=selection != "blocked", reuse_uniform_top4=selection == "uniform",
        persistent_memory_enabled=memory)
    isolated_expected = torch.where(valid[:, None, None], (history.float()*.5).to(q.dtype), 0)
    close(isolated, isolated_expected, atol=2e-4)


@pytest.mark.parametrize("mode", ["fla", "local_anchors", "vllm", "deterministic"])
@pytest.mark.parametrize("has_initial", [False, True])
@torch.no_grad()
def test_varlen_recurrence_oracle(mode, has_initial):
    from march.inference.varlen_segments import chunk_gated_delta_rule_with_anchor_states_varlen_forward, deterministic_anchor_state_read
    torch.manual_seed(107)
    lengths, heads, dim = [131, 193], 32, 128
    total = sum(lengths)
    q, k = [torch.nn.functional.normalize(torch.randn(1, total, heads, dim, device="cuda"), dim=-1).bfloat16() for _ in range(2)]
    v = torch.randn_like(q)*.1
    g = -torch.rand(1, total, heads, device="cuda")*.08
    beta = torch.rand(1, total, heads, device="cuda")*.5
    initial = torch.randn(2, heads, dim, dim, device="cuda")*.1 if has_initial else None
    positions = [[64, 128], [64, 128, 192]]
    expected_out, expected_anchors, expected_final = [], [], []
    offset = 0
    for row, length in enumerate(lengths):
        state = torch.zeros(heads, dim, dim, device="cuda") if initial is None else initial[row].clone()
        for t in range(length):
            at = offset+t
            state = state*g[0, at].exp()[:, None, None]
            residual = (v[0, at].float()-(state*k[0, at].float()[..., None]).sum(-2))*beta[0, at, :, None]
            state = state+k[0, at].float()[..., None]*residual[:, None, :]
            expected_out.append((state*q[0, at].float()[..., None]).sum(-2)*dim**-.5)
            if t+1 in positions[row]:expected_anchors.append(state.clone())
        expected_final.append(state);offset += length
    actual = chunk_gated_delta_rule_with_anchor_states_varlen_forward(
        query=q, key=k, value=v, g=g, beta=beta, cu_seqlens=torch.tensor([0, 131, total], device="cuda"),
        anchor_offsets=torch.tensor([0, 2, 5], device="cuda"), anchor_chunk_size=64, initial_state=initial,
        local_anchor_positions=torch.tensor(sum(positions, []), device="cuda") if mode == "local_anchors" else None,
        use_vllm_reference_kernel=mode in {"vllm", "deterministic"}, use_deterministic_kkt=mode == "deterministic")
    assert actual is not None, "varlen path fell back"
    expected = (torch.stack(expected_out)[None], torch.stack(expected_anchors), torch.stack(expected_final))
    for got, want in zip(actual, expected, strict=True):close(got, want, atol=3e-3)
    aq = torch.randn(5, heads, dim, device="cuda", dtype=torch.bfloat16)
    close(deterministic_anchor_state_read(aq, actual[1]),
          torch.einsum("mhk,mhkv->mhv", aq.float(), actual[1])*dim**-.5, atol=3e-3)


def packed_recurrence(values, initial=None):
    q, k, v, beta, g, aq = values
    dim = q.shape[-1]
    states, anchors, outputs, final = [], [], [], []
    sequence_length = q.shape[1] // 2
    for row in range(2):
        state = torch.zeros(1, q.shape[2], dim, dim, device=q.device) if initial is None else initial[row:row+1].clone()
        for offset in range(sequence_length):
            t = row*sequence_length+offset
            state = state*g[:, t].float().exp()[..., None, None]
            residual = (v[:, t].float()-(state*k[:, t].float()[..., None]).sum(-2))*beta[:, t, :, None].float()
            state = state+k[:, t].float()[..., None]*residual[..., None, :]
            outputs.append((state*q[:, t].float()[..., None]).sum(-2)*dim**-.5)
            if (offset+1)%512 == 0:
                anchors.append((state*aq[:, len(states)].float()[..., None]).sum(-2)*dim**-.5)
                states.append(state.clone())
        final.append(state)
    return (torch.stack(outputs, 1).to(q.dtype), torch.stack(anchors, 1).to(q.dtype),
            torch.stack(states, 1), torch.cat(final, 0))


def flash_fixture(dtype, dim):
    torch.manual_seed(108)
    shape = (1, 1024, 16, dim)
    q, k = [torch.nn.functional.normalize(torch.randn(shape, device="cuda"), dim=-1).to(dtype) for _ in range(2)]
    v = torch.randn(shape, device="cuda", dtype=dtype)*.1
    beta = (torch.rand(1, 1024, 16, device="cuda")*.5).to(dtype)
    g = -torch.rand(1, 1024, 16, device="cuda")*.08
    aq = torch.nn.functional.normalize(torch.randn(1, 2, 16, dim, device="cuda"), dim=-1).to(dtype)
    metadata = dict(boundary_counts=torch.tensor([512, 1024], device="cuda"),
                    text_cu_seqlens=torch.tensor([0, 512, 1024], device="cuda"),
                    anchor_cu_seqlens=torch.tensor([0, 1, 2], device="cuda"))
    return (q, k, v, beta, g, aq), metadata


@pytest.mark.parametrize("dim", [96, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backward", ["optimized", "reference"])
def test_flashqla_forward_all_gradients(dim, dtype, backward):
    from march.kernels.march_chunk_gdn import hsd_flashqla_tile_gated_delta_rule_with_anchor_states
    source, metadata = flash_fixture(dtype, dim)
    inputs = [x.detach().requires_grad_() for x in source]
    oracle_inputs = [x.detach().clone().requires_grad_() for x in source]
    actual = hsd_flashqla_tile_gated_delta_rule_with_anchor_states(
        query=inputs[0], key=inputs[1], value=inputs[2], beta=inputs[3], g=inputs[4], anchor_q=inputs[5],
        backward_backend=backward, **metadata)
    assert actual is not None, "Hopper FlashQLA path unavailable; cannot certify release"
    expected = packed_recurrence(oracle_inputs)[:3]
    for got, want in zip(actual, expected, strict=True):close(got, want, atol=2e-3)
    grad = [torch.randn_like(x)*(.01 if i == 2 else 1) for i, x in enumerate(expected)]
    actual_grads = torch.autograd.grad(actual, inputs, grad)
    expected_grads = torch.autograd.grad(expected, oracle_inputs, grad)
    for got, want in zip(actual_grads, expected_grads, strict=True):close(got, want, atol=6e-3, rtol=4e-2)


@pytest.mark.parametrize("dim", [96, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("has_initial", [False, True])
@torch.no_grad()
def test_flashqla_inference_states(dim, dtype, has_initial):
    from march.kernels.march_chunk_gdn import hsd_flashqla_tile_gated_delta_rule_inference
    values, metadata = flash_fixture(dtype, dim)
    initial = torch.randn(2, 16, dim, dim, device="cuda")*.1 if has_initial else None
    actual = hsd_flashqla_tile_gated_delta_rule_inference(
        query=values[0], key=values[1], value=values[2], beta=values[3], g=values[4], initial_state=initial, **metadata)
    expected = packed_recurrence(values, initial)
    for got, want in zip(actual, (expected[0], expected[2], expected[3]), strict=True):
        close(got, want, atol=3e-3)


@pytest.mark.parametrize("mode,fused,staged,prepack,precision,splits", [
    ("grouped", False, False, False, "high_residual", 1),
    ("token", False, False, False, "high_residual", 1),
    ("grouped", True, True, True, "high_residual", 2),
    ("grouped", True, False, False, "bf16_high", 1),
    ("grouped", True, False, False, "tf32", 4),
])
@pytest.mark.parametrize("all_invalid", [False, True])
def test_sparse_alternate_kernels(monkeypatch, mode, fused, staged, prepack, precision, splits, all_invalid):
    from march.kernels.march_global_read_topk import hsd_sparse_topk_global_read, sparse_topk_global_read_reference
    options = dict(HSD_TOPK_GLOBAL_READ_STRICT="1", HSD_TOPK_EXECUTION_MODE=mode,
                   HSD_TOPK_FUSE_DQ_DW=str(int(fused)), HSD_TOPK_STAGED_FORWARD=str(int(staged)),
                   HSD_TOPK_PREPACK_STATE=str(int(prepack)), HSD_TOPK_STATE_PRECISION_MODE=precision,
                   HSD_TOPK_GROUP_SPLITS=str(splits))
    for key, value in options.items():monkeypatch.setenv(key, value)
    torch.manual_seed(109)
    q = torch.randn(1, 19, 4, 128, device="cuda", dtype=torch.bfloat16)
    states = torch.randn(1, 7, 4, 128, 128, device="cuda")*.1
    indices = torch.randint(7, (1, 19, 4, 4), device="cuda")
    valid = torch.zeros_like(indices, dtype=torch.bool) if all_invalid else torch.rand(indices.shape, device="cuda")>.2
    weights = torch.rand(indices.shape, device="cuda")*valid
    inputs = [x.detach().requires_grad_() for x in (q, states, weights)]
    oracle_inputs = [x.detach().clone().requires_grad_() for x in (q, states, weights)]
    def run(op, values):
        return op(read_q=values[0], states=values[1], topk_weights=values[2],
                  topk_indices=indices, topk_valid=valid, read_scale=128**-.5)
    actual = run(hsd_sparse_topk_global_read, inputs)
    expected = run(sparse_topk_global_read_reference, oracle_inputs)
    close(actual, expected)
    cotangent = torch.randn_like(expected)
    for got, want in zip(torch.autograd.grad(actual, inputs, cotangent),
                         torch.autograd.grad(expected, oracle_inputs, cotangent), strict=True):
        close(got, want)


@pytest.mark.parametrize("has_initial", [False, True])
@pytest.mark.parametrize("fused_anchor", [False, True])
@torch.no_grad()
def test_hf_selective_anchor_states(monkeypatch, has_initial, fused_anchor):
    from march.safin1.fused_march_segments import chunk_gated_delta_rule_with_anchor_states_forward
    monkeypatch.setenv("HSD_ENABLE_FUSED_ANCHOR_O", str(int(fused_anchor)))
    monkeypatch.setenv("HSD_DISABLE_FUSED_ANCHOR_O", "0")
    monkeypatch.setenv("HSD_DISABLE_HSD_SEGMENT_FASTPATH", "0")
    torch.manual_seed(110)
    batch, length, heads, dim = 2, 131, 32, 128
    shape = (batch, length, heads, dim)
    q, k = [torch.nn.functional.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16() for _ in range(2)]
    v = torch.randn_like(q)*.1
    beta = torch.rand(shape[:-1], device="cuda")*.5
    g = -torch.rand(shape[:-1], device="cuda")*.08
    aq = torch.randn(batch, 2, heads, dim, device="cuda", dtype=q.dtype)
    initial = torch.randn(batch, heads, dim, dim, device="cuda")*.1 if has_initial else None
    state = torch.zeros(batch, heads, dim, dim, device="cuda") if initial is None else initial.clone()
    outputs, states, anchor_out = [], [], []
    for t in range(length):
        state = state*g[:, t].exp()[..., None, None]
        residual = (v[:, t].float()-(state*k[:, t].float()[..., None]).sum(-2))*beta[:, t, :, None]
        state = state+k[:, t].float()[..., None]*residual[..., None, :]
        outputs.append((state*q[:, t].float()[..., None]).sum(-2)*dim**-.5)
        if (t+1)%64 == 0:
            anchor_out.append((state*aq[:, len(states)].float()[..., None]).sum(-2)*dim**-.5)
            states.append(state.clone())
    actual = chunk_gated_delta_rule_with_anchor_states_forward(
        query=q, key=k, value=v, g=g, beta=beta, anchor_count=2, anchor_chunk_size=64,
        initial_state=initial, anchor_q=aq)
    assert actual is not None, "selective anchor kernel fell back"
    close(actual[0], torch.stack(outputs, 1), atol=3e-3)
    close(actual[2], torch.stack(states, 1), atol=3e-3)
    close(actual[3], state, atol=3e-3)
    if fused_anchor:close(actual[1], torch.stack(anchor_out, 1), atol=3e-3)
    else:assert actual[1] is None

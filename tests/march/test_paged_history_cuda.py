"""CUDA regression for dense history, persistent memory and Graph replay."""
import pytest
import torch

from march.inference import paged_history_read as phr

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("memory", [False, True])
@pytest.mark.parametrize("batch,anchors", [(1, 8), (4, 20)])
@pytest.mark.parametrize("split_read", [False, True])
def test_dense_history_cuda_graph_replay(monkeypatch, memory, batch, anchors, split_read):
    monkeypatch.setenv("VLLM_HSD_GRAPH_V3_SPLIT_HISTORY_READ", str(int(split_read)))
    monkeypatch.setenv("HSD_PAGED_HISTORY_READ_IMPL", "triton")
    torch.manual_seed(941)
    device, dtype = torch.device("cuda"), torch.bfloat16
    heads, dim, page_m = 32, 128, 16
    pages_per_state = (anchors + 1 + page_m - 1) // page_m
    cache = phr.HSDAnchorPageCache(page_m=page_m)
    cache.preallocate(spec=phr.HSDHistoryCacheSpec(
        page_m=page_m, num_pages=batch*pages_per_state, heads=heads,
        value_dim=dim, key_dim=dim, states_dtype=torch.float32, keys_dtype=dtype,
    ), device=device)
    cache.preallocate_metadata(state_rows=batch, max_pages_per_state=pages_per_state, device=device)
    for index in range(batch):
        states = torch.randn(1, anchors, heads, dim, dim, device=device)
        keys = torch.randn(1, anchors, heads, dim, device=device, dtype=dtype)
        boundaries = torch.arange(anchors, device=device)*512 + (0 if memory else 512)
        cache.append(index, states, keys, boundaries)
    indices = torch.arange(batch, device=device)
    starts = torch.full((batch,), (anchors+2)*512, device=device, dtype=torch.long)
    valid = torch.ones(batch, device=device, dtype=torch.bool)
    if batch > 1:
        valid[-1] = False
    read_q = torch.randn(batch, heads, dim, device=device, dtype=dtype)
    router_q = torch.randn_like(read_q)
    null = torch.randn(batch, heads, device=device, dtype=dtype)
    base = torch.randn_like(read_q)
    output = torch.empty_like(base)
    scale = torch.tensor(0.5, device=device)

    def reference():
        table, lengths, _ = cache.export_page_table(indices, device=device)
        oracle = phr._paged_decode_top4_memory_history_read_torch if memory else phr._paged_decode_top4_history_read_torch
        return oracle(cache=cache, read_q=read_q, router_q=router_q, page_table=table,
                      anchor_lengths=lengths, start_indices=starts, route_scale=dim**-0.5,
                      read_scale=dim**-0.5, null_logits=null)

    def run_static():
        output.copy_(base)
        return phr.paged_static_decode_history_read(
            cache=cache, read_q=read_q, router_q=router_q, state_indices=indices,
            start_indices=starts, valid_mask=valid, route_scale=dim**-0.5,
            read_scale=dim**-0.5, base_out=output, history_scale=scale,
            null_logits=null, persistent_memory_enabled=memory)

    expected_history = reference()
    ordinary = phr.paged_batched_decode_history_read(
        cache=cache, read_q=read_q, router_q=router_q, state_indices=indices,
        start_indices=starts, route_scale=dim**-0.5, read_scale=dim**-0.5,
        null_logits=null, persistent_memory_enabled=memory)
    assert ordinary is not None, "requested CUDA path fell back"
    torch.testing.assert_close(ordinary, expected_history, atol=2e-2, rtol=2e-2)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            run_static()
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_static()
    # Replay with new queries, then with updated page contents and metadata.
    for step in range(3):
        read_q.normal_()
        router_q.normal_()
        if step == 1:
            assert cache.try_append_batch_one(
                list(range(batch)),
                torch.randn(batch, heads, dim, dim, device=device),
                torch.randn(batch, heads, dim, device=device, dtype=dtype),
                torch.full((batch,), (anchors+1)*512, device=device, dtype=torch.long))
        if step == 2:
            cache.clear({0})
            cache.append(0, torch.randn(1, 1, heads, dim, dim, device=device),
                         torch.randn(1, 1, heads, dim, device=device, dtype=dtype),
                         torch.tensor([0 if memory else 512], device=device))
        if memory:
            cache.assert_persistent_memory_ready(indices, valid_mask=valid)
        expected = (base.float() + reference().float()*scale).to(dtype)
        expected = torch.where(valid[:, None, None], expected, base)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("variant", ["safin1_4b", "safin1_35b_a3b"])
def test_persistent_state_cuda_train_save_reload(variant, tmp_path):
    import importlib.util
    import json
    from pathlib import Path
    from transformers import AutoModelForCausalLM
    from march.safin1 import Safin1ForConditionalGeneration, Safin1MoeForConditionalGeneration
    from march.safin1.config_utils import normalize_safin1_config, copy_safin1_remote_code

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("gpu_smoke_config", root/"scripts/march/smoke_hf.py")
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    torch.manual_seed(123)
    config = smoke.tiny_config(variant, chunk_size=512)
    config.text_config.hsd_persistent_memory_enabled = True
    config.text_config.hsd_persistent_memory_route_key_enabled = True
    config.text_config.hsd_persistent_memory_route_key_mode = "dedicated"
    cls = Safin1ForConditionalGeneration if variant == "safin1_4b" else Safin1MoeForConditionalGeneration
    # Use the production BF16 loader for both sides of the round trip. A blanket
    # .to(bfloat16) also casts non-persistent RoPE buffers, unlike from_pretrained.
    seed_path = tmp_path / "seed"
    cls(config).save_pretrained(seed_path, safe_serialization=True)
    seed_config = seed_path / "config.json"
    seed_config.write_text(json.dumps(normalize_safin1_config(json.loads(seed_config.read_text())), indent=2)+"\n")
    copy_safin1_remote_code(seed_path, variant)
    with smoke.remote_registry():
        model = AutoModelForCausalLM.from_pretrained(
            seed_path, trust_remote_code=True, local_files_only=True, device_map="cuda",
            dtype=torch.bfloat16, attn_implementation="eager")
    memory = {n: p for n, p in model.named_parameters() if n.endswith("persistent_memory_state")}
    assert memory and all(p.dtype == torch.float32 for p in memory.values())
    before = {n: p.detach().clone() for n, p in memory.items()}
    ids = torch.randint(3, 59, (1, 515), device="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    result = model(input_ids=ids, labels=ids, use_cache=False)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    for p in memory.values():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    optimizer.step()
    assert all(not torch.equal(before[n], p) for n, p in memory.items())
    model.eval()
    with torch.inference_mode():
        expected = model(input_ids=ids, use_cache=False).logits
        assert torch.isfinite(expected).all()
        torch.testing.assert_close(model(input_ids=ids, use_cache=False).logits, expected, atol=0, rtol=0)
    model.save_pretrained(tmp_path, safe_serialization=True)
    config_file = tmp_path/"config.json"
    config_file.write_text(json.dumps(normalize_safin1_config(json.loads(config_file.read_text())), indent=2)+"\n")
    copy_safin1_remote_code(tmp_path, variant)
    with smoke.remote_registry():
        loaded, info = AutoModelForCausalLM.from_pretrained(
            tmp_path, trust_remote_code=True, local_files_only=True, device_map="cuda",
            dtype=torch.bfloat16, attn_implementation="eager", output_loading_info=True)
    assert not any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys")), info
    assert type(loaded).__module__.startswith("transformers_modules.")
    assert loaded.config.text_config.hsd_persistent_memory_enabled
    for n, p in memory.items():
        assert loaded.state_dict()[n].dtype == torch.float32
        torch.testing.assert_close(loaded.state_dict()[n], p, atol=0, rtol=0)
    loaded.eval()
    with torch.inference_mode():
        torch.testing.assert_close(loaded(input_ids=ids, use_cache=False).logits, expected, atol=0, rtol=0)
        for prefix in (511, 512, 513):
            cached = loaded(input_ids=ids[:, :prefix], use_cache=True)
            step = loaded(**loaded.prepare_inputs_for_generation(
                ids[:, :prefix+1], past_key_values=cached.past_key_values, use_cache=True))
            full = loaded(input_ids=ids[:, :prefix+1], use_cache=False)
            torch.testing.assert_close(step.logits[:, -1], full.logits[:, -1], atol=5e-3, rtol=2e-2)


@pytest.mark.parametrize("batch,heads,dim", [(1, 2, 16), (2, 4, 128)])
@pytest.mark.parametrize("has_initial_state", [False, True])
def test_chunk_state_cuda_all_batches_and_heads(batch, heads, dim, has_initial_state):
    from march.safin1.fused_march_segments import chunk_gated_delta_rule_with_chunk_states_forward

    torch.manual_seed(47)
    shape = (batch, 131, heads, dim)
    query = torch.nn.functional.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16()
    key = torch.nn.functional.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16()
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1
    beta = torch.rand(shape[:-1], device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(shape[:-1], device="cuda") * 0.1
    initial = torch.randn(batch, heads, dim, dim, device="cuda") * 0.01 if has_initial_state else None
    # Independent FP32 recurrence records the state at the start of each FLA chunk.
    state = torch.zeros(batch, heads, dim, dim, device="cuda") if initial is None else initial.clone()
    expected_states, expected_outputs = [], []
    for t in range(shape[1]):
        if t % 64 == 0:
            expected_states.append(state.clone())
        state = state * g[:, t].exp()[..., None, None]
        k = key[:, t].float()
        residual = (value[:, t].float() - (state * k[..., None]).sum(-2)) * beta[:, t, :, None].float()
        state = state + k[..., None] * residual[..., None, :]
        expected_outputs.append((state * query[:, t].float()[..., None]).sum(-2) * dim**-0.5)
    expected = (torch.stack(expected_outputs, 1), torch.stack(expected_states, 1), state)
    for _ in range(2):
        actual = chunk_gated_delta_rule_with_chunk_states_forward(
            query=query, key=key, value=value, g=g, beta=beta, initial_state=initial)
        assert actual is not None, "requested FLA chunk-state path fell back"
        for got, want in zip(actual, expected):
            assert torch.isfinite(got).all()
            torch.testing.assert_close(got.float(), want, atol=3e-3, rtol=2e-2)
        torch.testing.assert_close(actual[1][:, 0], expected[1][:, 0], atol=0, rtol=0)

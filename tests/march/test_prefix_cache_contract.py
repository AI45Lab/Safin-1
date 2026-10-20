"""Execute actual prefix-cache methods on CPU without native vLLM imports."""

import ast
import copy
from collections import OrderedDict
from functools import lru_cache
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from march.inference import paged_history_read as phr

ROOT = Path(__file__).resolve().parents[2]
VLLM = ROOT / "third_party/vllm/vllm"
MODEL = VLLM / "model_executor/models/safin1.py"
spec = importlib.util.spec_from_file_location("march_prefix_policy", VLLM / "v1/core/march_prefix_cache.py")
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


@lru_cache()
def source(path):
    return ast.parse(path.read_text())


def cpu_class(path, name, methods, keep_inference_mode=False, **globals_):
    node = next(n for n in source(path).body if isinstance(n, ast.ClassDef) and n.name == name)
    # Resolve the model's actual constructor-free mixins; all existing cache
    # assertions still execute the real production method bodies.
    available = {}
    helpers = {
        "PrefixSnapshotMixin": MODEL.parent / "march/prefix_cache.py",
        "StateCacheMixin": MODEL.parent / "march/state_cache.py",
        "CausalLMWeightLoadingMixin": MODEL.parent / "march/weights.py",
    }
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id in helpers:
            parent = next(n for n in source(helpers[base.id]).body
                          if isinstance(n, ast.ClassDef) and n.name == base.id)
            for fn in parent.body:
                if isinstance(fn, ast.FunctionDef):
                    available.setdefault(fn.name, fn)
    available.update({n.name: n for n in node.body if isinstance(n, ast.FunctionDef)})
    assert set(methods) <= available.keys()
    functions = [copy.deepcopy(available[name]) for name in methods]
    for fn in functions:
        fn.decorator_list = [d for d in fn.decorator_list if
                             (isinstance(d, ast.Name) and d.id == "staticmethod") or
                             (keep_inference_mode and isinstance(d, ast.Call) and
                              isinstance(d.func, ast.Attribute) and d.func.attr == "inference_mode" and
                              isinstance(d.func.value, ast.Name) and d.func.value.id == "torch")]
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            ast.ClassDef(name="CPU", bases=[], keywords=[], body=functions, decorator_list=[]),
        ],
        type_ignores=[],
    )
    ns = {
        "torch": torch,
        "os": os,
        "prefix_cache_namespace": policy.prefix_cache_namespace,
        "_is_valid_hsd_state_index": lambda i: int(i) > 0,
        "_env_truthy": lambda key: os.environ.get(key, "").lower() in {"1", "true", "yes", "on"},
        "logger": NS(warning_once=lambda *a, **kw: None),
    }
    ns.update(globals_)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    return ns["CPU"]


@pytest.fixture(autouse=True)
def clean_controls(monkeypatch):
    for key in list(os.environ):
        if key.startswith(("VLLM_HSD_", "HSD_")):
            monkeypatch.delenv(key)
    for name in ("USE_PERSISTENT_STATE_SIDECAR", "ENABLE_COMPACT_STATE_SLOTS", "ENABLE_COMPACT_STATE_KV_SPEC", "SERIALIZE_PREFIX_PREFILL"):
        monkeypatch.setenv("VLLM_HSD_" + name, "1")


def config():
    return NS(
        model_type="safin1_4b_text",
        anchor_chunk_size=512,
        anchor_token_num=1,
        hsd_read_topk=4,
        hsd_read_block_size=32768,
        hsd_global_read_layers="10-30",
    )


def engine_config():
    return NS(
        cache_config=NS(enable_prefix_caching=True, mamba_cache_mode="none"),
        model_config=NS(enforce_eager=True, quantization=None),
        scheduler_config=NS(async_scheduling=False),
        parallel_config=NS(pipeline_parallel_size=1),
        kv_transfer_config=None,
    )


def test_default_requires_explicit_opt_in(monkeypatch):
    cfg, vcfg = config(), engine_config()
    with pytest.raises(NotImplementedError, match="experimental"):
        policy.validate_march_prefix_cache(cfg, vcfg)
    vcfg.cache_config.enable_prefix_caching = False
    policy.validate_march_prefix_cache(cfg, vcfg)
    vcfg.cache_config.enable_prefix_caching = True
    monkeypatch.setenv("VLLM_HSD_ENABLE_PREFIX_CACHE", "1")
    policy.validate_march_prefix_cache(cfg, vcfg)


@pytest.mark.parametrize(
    "owner,field,value,error",
    [
        ("model_config", "enforce_eager", False, "Graph V3 prefix"),
        ("parallel_config", "pipeline_parallel_size", 2, "pipeline"),
        ("scheduler_config", "async_scheduling", True, "synchronous"),
        ("cache_config", "mamba_cache_mode", "align", "mamba_cache_mode"),
        ("model_config", "quantization", "fp8", "Quantized"),
        (None, "kv_transfer_config", object(), "KV connector"),
    ],
)
def test_unsupported_engines_fail_closed(monkeypatch, owner, field, value, error):
    monkeypatch.setenv("VLLM_HSD_ENABLE_PREFIX_CACHE", "1")
    vcfg = engine_config()
    setattr(getattr(vcfg, owner) if owner else vcfg, field, value)
    with pytest.raises(NotImplementedError, match=error):
        policy.validate_march_prefix_cache(config(), vcfg)


def test_persistent_state_cannot_share_prompt_only_identity(monkeypatch):
    monkeypatch.setenv("VLLM_HSD_ENABLE_PREFIX_CACHE", "1")
    cfg = config()
    cfg.hsd_persistent_memory_enabled = True
    with pytest.raises(NotImplementedError, match="persistent"):
        policy.validate_march_prefix_cache(cfg, engine_config())


@pytest.mark.parametrize(
    "field,value",
    [
        ("hsd_read_topk", 8),
        ("hsd_read_block_size", 16384),
        ("hsd_global_read_layers", "2-39"),
        ("hsd_null_read_gate", True),
    ],
)
def test_identity_covers_runtime_and_reset_generation(field, value):
    cfg = config()
    initial = policy.prefix_cache_namespace(cfg, "bfloat16", 0)
    assert policy.prefix_cache_namespace(NS(**vars(cfg)), "bfloat16", 0) == initial
    assert policy.prefix_cache_namespace(cfg, "float16", 0) != initial
    assert policy.prefix_cache_namespace(cfg, "bfloat16", 1) != initial
    setattr(cfg, field, value)
    assert policy.prefix_cache_namespace(cfg, "bfloat16", 0) != initial


def make_layer(fixed=False, pages=4):
    cls = cpu_class(
        MODEL,
        "HistoricalGatedDeltaNetAttention",
        [
            "_hsd_prefix_snapshot_max_bytes_per_layer",
            "_hsd_prefix_snapshot_tensor_bytes",
            "_hsd_prefix_snapshot_bytes_for",
            "_evict_hsd_prefix_snapshots",
            "_store_hsd_prefix_snapshot_parts",
            "_hsd_prefix_snapshot_restore_preflight",
            "_hsd_graph_prefix_storage",
            "restore_hsd_prefix_snapshot",
            "_anchor_states_to_paged_cache",
        ],
    )
    layer = cls()
    layer.config = NS(anchor_chunk_size=512, num_hidden_layers=1)
    layer.layer_idx = 1
    layer.hsd_prefix_cache_generation = 0
    layer.hsd_prefix_snapshot_cache = OrderedDict()
    layer.hsd_prefix_snapshot_bytes = 0
    layer.hsd_prefix_snapshot_step_ready_keys = set()
    layer.hsd_prefix_snapshot_step_evicted_keys = set()
    layer.hsd_prefix_recurrent_replay = {}
    layer.hsd_text_state_cache, layer.hsd_anchor_cache = {}, {}
    layer.events = []
    layer._record_hsd_varlen_event = lambda event, **kw: layer.events.append((event, kw))
    layer.live_conv, layer.live_recurrent = {}, {}
    layer._store_conv_state = lambda i, t: layer.live_conv.__setitem__(i, t.clone())
    layer._store_recurrent_state = lambda i, t: layer.live_recurrent.__setitem__(i, t.clone())
    layer.hsd_anchor_page_cache = phr.HSDAnchorPageCache(page_m=4)
    if fixed:
        layer.hsd_anchor_page_cache.preallocate(
            spec=phr.HSDHistoryCacheSpec(
                page_m=4,
                num_pages=pages,
                heads=2,
                value_dim=3,
                key_dim=5,
                route_heads=2,
                states_dtype=torch.float32,
                keys_dtype=torch.float32,
            ),
            device=torch.device("cpu"),
        )
        layer.hsd_anchor_page_cache.preallocate_metadata(
            state_rows=16, max_pages_per_state=4, device=torch.device("cpu")
        )
    return layer


def store(layer, key="key"):
    torch.manual_seed(7)
    values = dict(
        conv_state=torch.randn(1, 10, 4),
        recurrent_state=torch.randn(1, 2, 5, 3),
        anchor_states=torch.randn(1, 2, 2, 5, 3),
        anchor_keys=torch.randn(1, 2, 2, 5),
        anchor_boundaries=torch.tensor([512, 1024]),
    )
    assert layer._store_hsd_prefix_snapshot_parts(key=key, state_idx=1, prefix_len=1024, **values)
    return values


@pytest.mark.parametrize("fixed", [False, True])
def test_restore_clones_state_and_preserves_history_read(fixed):
    layer = make_layer(fixed)
    original = store(layer)
    reference = {k: v.clone() for k, v in original.items()}
    # The producer request may continue or have its pages reused after capture.
    for tensor in original.values():
        tensor.zero_()
    for slot in (3, 7):
        assert layer.restore_hsd_prefix_snapshot(key="key", state_idx=slot, history_state_idx=slot, prefix_len=1024)
        torch.testing.assert_close(layer.live_conv[slot], reference["conv_state"], atol=0, rtol=0)
        torch.testing.assert_close(layer.live_recurrent[slot], reference["recurrent_state"], atol=0, rtol=0)
        states, keys, boundaries = layer.hsd_anchor_page_cache.materialize(slot)
        torch.testing.assert_close(states, reference["anchor_states"].transpose(-1, -2), atol=0, rtol=0)
        torch.testing.assert_close(keys, reference["anchor_keys"], atol=0, rtol=0)
        assert boundaries.tolist() == [512, 1024]
    # One restored request must not write into another request or the snapshot.
    layer.live_conv[3].zero_()
    layer.hsd_text_state_cache[3]["recurrent_state"].zero_()
    torch.testing.assert_close(layer.live_conv[7], reference["conv_state"], atol=0, rtol=0)
    torch.testing.assert_close(
        layer.hsd_prefix_snapshot_cache["key"]["recurrent_state"], reference["recurrent_state"], atol=0, rtol=0
    )
    cache = layer.hsd_anchor_page_cache
    cache.append(
        9, reference["anchor_states"].transpose(-1, -2), reference["anchor_keys"], reference["anchor_boundaries"]
    )
    table, lengths, _ = cache.export_page_table(torch.tensor([9, 7]), device=torch.device("cpu"))
    q = torch.randn(1, 2, 5).expand(2, -1, -1).contiguous()
    history = phr._paged_decode_top4_history_read_torch(
        cache=cache,
        read_q=q,
        router_q=q,
        page_table=table,
        anchor_lengths=lengths,
        start_indices=torch.tensor([1025, 1025]),
        route_scale=5**-0.5,
        read_scale=5**-0.5,
        null_logits=torch.zeros(2, 2),
    )
    torch.testing.assert_close(history[0], history[1], atol=0, rtol=0)


@pytest.mark.parametrize(
    "damage,reason",
    [
        ("generation", "snapshot_stale_generation"),
        ("prefix", "config_mismatch"),
        ("partial", "anchor_snapshot_incomplete"),
        ("shape", "anchor_snapshot_shape"),
        ("boundary", "anchor_snapshot_boundaries"),
        ("state", "snapshot_state_shape"),
    ],
)
def test_corrupt_or_stale_snapshot_does_not_mutate_request(damage, reason):
    layer = make_layer()
    store(layer)
    snap = layer.hsd_prefix_snapshot_cache["key"]
    if damage == "generation":
        layer.hsd_prefix_cache_generation += 1
    elif damage == "prefix":
        snap["prefix_len"] = 512
    elif damage == "partial":
        del snap["anchor_keys"]
    elif damage == "shape":
        snap["anchor_keys"] = snap["anchor_keys"][:, :1]
    elif damage == "boundary":
        snap["anchor_boundaries"] = torch.tensor([1024, 512])
    else:
        snap["recurrent_state"] = torch.zeros(2, 5, 3)
    assert not layer.restore_hsd_prefix_snapshot(key="key", state_idx=3, prefix_len=1024)
    assert layer.events[-1][1]["reason"] == reason
    assert not layer.live_conv and not layer.live_recurrent and not layer.hsd_text_state_cache
    assert layer.hsd_anchor_page_cache.materialize(3) is None


def test_all_layers_preflight_before_first_mutation():
    good, full = make_layer(), make_layer(True, pages=1)
    store(good)
    values = store(full)
    full.hsd_anchor_page_cache.append(
        9, values["anchor_states"].transpose(-1, -2), values["anchor_keys"], values["anchor_boundaries"]
    )
    cls = cpu_class(MODEL, "Safin1Model", ["restore_hsd_prefix_snapshot"])
    model = cls()
    model._iter_hsd_linear_attn_modules = lambda: iter([good, full])
    with pytest.raises(RuntimeError, match="anchor_page_capacity"):
        model.restore_hsd_prefix_snapshot(key="key", state_idx=3, prefix_len=1024)
    assert not good.live_conv and not full.live_conv


def make_scheduler(reset=True):
    cls = cpu_class(
        VLLM / "v1/core/sched/scheduler.py",
        "Scheduler",
        [
            "_make_hsd_prefix_config_key",
            "_hsd_prefix_key",
            "reset_prefix_cache",
            "_update_hsd_prefix_snapshot_lifecycle",
        ],
    )
    scheduler = cls()
    scheduler.vllm_config = NS(model_config=NS(hf_text_config=config(), dtype="bfloat16"))
    scheduler.hsd_prefix_cache_enabled = True
    scheduler.hsd_prefix_cache_epoch = 0
    scheduler.hsd_prefix_hash_block_size = 16
    scheduler.kv_cache_manager = NS(reset_prefix_cache=lambda: reset)
    for key in (
        "hsd_prefix_snapshot_ready",
        "hsd_prefix_snapshot_planned_this_step",
        "hsd_prefix_hit_request_ids",
        "hsd_deferred_decode_once",
    ):
        setattr(scheduler, key, {"old"})
    for key in (
        "hsd_prefix_snapshot_observations",
        "hsd_prefix_hit_observations",
        "hsd_prefix_observed_candidates",
        "hsd_prefix_observed_store_boundaries",
    ):
        setattr(scheduler, key, {"old": 1})
    scheduler._record_hsd_prefix_cache_event = lambda *a, **kw: None
    return scheduler


@pytest.mark.parametrize("success", [False, True])
def test_scheduler_reset_and_delayed_ready_messages(success):
    scheduler = make_scheduler(success)
    request = NS(block_hashes=[bytes([i]) for i in range(32)])
    old = scheduler._hsd_prefix_key(request, 512)
    assert scheduler.reset_prefix_cache() is success
    new = scheduler._hsd_prefix_key(request, 512)
    assert (old != new) is success
    scheduler._update_hsd_prefix_snapshot_lifecycle(NS(hsd_prefix_snapshot_ready=[old], hsd_prefix_snapshot_evicted=[]))
    assert (old in scheduler.hsd_prefix_snapshot_ready) is not success
    if success:
        assert not scheduler.hsd_prefix_hit_request_ids
        assert not scheduler.hsd_deferred_decode_once
    scheduler._update_hsd_prefix_snapshot_lifecycle(
        NS(hsd_prefix_snapshot_ready=[new], hsd_prefix_snapshot_evicted=[new])
    )
    assert new not in scheduler.hsd_prefix_snapshot_ready


def make_outer(layer):
    cls = cpu_class(
        MODEL,
        "Safin1MoeForCausalLM",
        [
            "get_hsd_prefix_cache_status",
            "invalidate_hsd_prefix_cache",
            "reset_hsd_runtime_state_after_warmup",
            "_clear_hsd_moe_shared_expert_outputs",
            "load_weights",
        ],
    )
    model = cls()
    model.modules = lambda: iter([layer])
    model.model = NS(
        _iter_hsd_linear_attn_modules=lambda: iter([layer]),
        _hsd_prefix_snapshot_ready=["key"],
        _hsd_prefix_snapshot_evicted=["old"],
    )
    return model


def test_weight_invalidation_preserves_allocations_and_learned_state():
    layer = make_layer(True)
    store(layer)
    assert layer.restore_hsd_prefix_snapshot(key="key", state_idx=3, prefix_len=1024)
    layer.hsd_persistent_state_sidecar = {"state": torch.ones(2, 2)}
    address = layer.hsd_persistent_state_sidecar["state"].data_ptr()
    layer.persistent_memory_state = torch.randn(1, 2, 5, 3)
    learned = layer.persistent_memory_state.clone()
    model = make_outer(layer)
    stats = model.invalidate_hsd_prefix_cache()
    assert stats["prefix_snapshot_entries"] == 1
    assert layer.hsd_prefix_cache_generation == 1
    assert not layer.hsd_prefix_snapshot_cache and layer.hsd_prefix_snapshot_bytes == 0
    assert not layer.hsd_text_state_cache
    assert layer.hsd_anchor_page_cache.materialize(3) is None
    assert layer.hsd_persistent_state_sidecar["state"].data_ptr() == address
    assert not layer.hsd_persistent_state_sidecar["state"].any()
    torch.testing.assert_close(layer.persistent_memory_state, learned, atol=0, rtol=0)
    assert not model.model._hsd_prefix_snapshot_ready


def test_failed_partial_weight_load_invalidates_before_consuming_iterator():
    layer = make_layer()
    store(layer)
    model = make_outer(layer)
    model._historical_hsd_weights_validated = True

    def failed_mapping(weights):
        assert not layer.hsd_prefix_snapshot_cache
        assert layer.hsd_prefix_cache_generation == 1
        raise RuntimeError("partial update failed")

    model.hf_to_vllm_mapper = NS(apply=failed_mapping)
    with pytest.raises(RuntimeError, match="partial update failed"):
        model.load_weights(iter([]))


@pytest.mark.parametrize("active,reset_ok", [(True, True), (False, False), (False, True)])
def test_engine_drains_and_resets_native_and_hsd_before_weight_update(active, reset_ok):
    cls = cpu_class(VLLM / "v1/engine/core.py", "EngineCore", ["reset_prefix_cache", "collective_rpc"])
    core = cls()
    calls = []
    core.scheduler = NS(
        hsd_prefix_cache_enabled=True,
        _uses_historical_hsd_prefix_cache=lambda: True,
        has_unfinished_requests=lambda: active,
        reset_prefix_cache=lambda *a: calls.append("native_reset") or reset_ok,
    )
    core.model_executor = NS(collective_rpc=lambda method, *a, **kw: calls.append(method) or [])
    if active or not reset_ok:
        with pytest.raises(RuntimeError):
            core.collective_rpc("update_weights", kwargs={"update_info": {}})
        assert "update_weights" not in calls
        assert "reset_hsd_prefix_cache" not in calls
    else:
        core.collective_rpc("update_weights", kwargs={"update_info": {}})
        assert calls == ["native_reset", "reset_hsd_prefix_cache", "update_weights"]


def test_kernel_format_weight_transfer_also_invalidates_worker_state(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    layer = make_layer()
    store(layer)
    model = make_outer(layer)
    weight = torch.zeros(1)
    model.get_parameter = lambda name: weight
    runner_cls = cpu_class(VLLM / "v1/worker/gpu_model_runner.py", "GPUModelRunner", ["reset_hsd_prefix_cache"])
    runner = runner_cls()
    runner.model = model
    runner.get_model = lambda: model
    runner._hsd_pending_prefix_restores = {"old": ("key", 1024)}
    worker_cls = cpu_class(VLLM / "v1/worker/gpu_worker.py", "Worker", ["update_weights"])
    worker = worker_cls()
    worker.model_runner = runner

    def receive(info, load_weights):
        assert not layer.hsd_prefix_snapshot_cache
        assert not runner._hsd_pending_prefix_restores
        load_weights([("weight", torch.ones(1))])

    worker.weight_transfer_engine = NS(
        parse_update_info=lambda info: NS(is_checkpoint_format=False), receive_weights=receive
    )
    worker.update_weights({})
    assert weight.item() == 1


@pytest.mark.parametrize("budget", [0, 1])
def test_disabled_or_insufficient_snapshot_budget_publishes_nothing(monkeypatch, budget):
    monkeypatch.setenv("VLLM_HSD_PREFIX_CACHE_MAX_BYTES", str(budget))
    layer = make_layer()
    ok = layer._store_hsd_prefix_snapshot_parts(
        key="key", state_idx=1, prefix_len=512,
        conv_state=torch.ones(1, 10, 4), recurrent_state=torch.ones(1, 2, 5, 3))
    assert not ok and not layer.hsd_prefix_snapshot_cache
    assert not layer.hsd_prefix_snapshot_step_ready_keys
    assert layer.hsd_prefix_snapshot_bytes == 0


def test_snapshot_budget_evicts_old_key_and_accounts_bytes(monkeypatch):
    monkeypatch.setenv("VLLM_HSD_PREFIX_CACHE_MAX_BYTES", "1024")
    layer = make_layer()
    store(layer, "old")
    store(layer, "new")
    assert list(layer.hsd_prefix_snapshot_cache) == ["new"]
    assert layer.hsd_prefix_snapshot_step_evicted_keys == {"old"}
    assert 0 < layer.hsd_prefix_snapshot_bytes <= 1024
    assert not layer.restore_hsd_prefix_snapshot(key="old", state_idx=3, prefix_len=1024)


def test_graph_layer_iteration_does_not_depend_on_prefix_methods():
    from itertools import islice
    cls = cpu_class(MODEL, "Safin1Model",
                    ["_iter_hsd_linear_attn_modules", "ensure_hsd_graph_v3_state_sidecars"], islice=islice)
    attn = NS(ensure_hsd_graph_v3_state_sidecars=lambda: True)
    model = cls()
    model.layers = [NS(_active_hsd_attn=lambda: attn), NS(_active_hsd_attn=lambda: None)]
    model.start_layer, model.end_layer = 0, 2
    assert list(model._iter_hsd_linear_attn_modules()) == [attn]
    assert model.ensure_hsd_graph_v3_state_sidecars()


GRAPH_PREFIX_CONTROLS = (
    "ENABLE_GRAPH_PREFIX_CACHE", "ENABLE_GRAPH_V3", "GRAPH_V3_FULL_CUDAGRAPH",
    "ENABLE_COMPACT_STATE_SLOTS", "USE_PERSISTENT_STATE_SIDECAR",
)


@pytest.mark.parametrize("missing", [None, *GRAPH_PREFIX_CONTROLS])
def test_graph_prefix_requires_complete_fixed_state_profile(monkeypatch, missing):
    monkeypatch.setenv("VLLM_HSD_ENABLE_PREFIX_CACHE", "1")
    for control in GRAPH_PREFIX_CONTROLS:
        if control != missing:
            monkeypatch.setenv("VLLM_HSD_" + control, "1")
        else:
            monkeypatch.delenv("VLLM_HSD_" + control, raising=False)
    cfg = engine_config()
    cfg.model_config.enforce_eager = False
    if missing is None:
        policy.validate_march_prefix_cache(config(), cfg)
    else:
        with pytest.raises(NotImplementedError, match="Graph V3 prefix"):
            policy.validate_march_prefix_cache(config(), cfg)


def graph_layer(monkeypatch):
    monkeypatch.setenv("VLLM_HSD_ENABLE_GRAPH_PREFIX_CACHE", "1")
    layer = make_layer(True)
    layer.hsd_layer_uses_global_read = True
    layer.hsd_persistent_state_sidecar = {
        "conv_state": torch.zeros(16, 10, 4),
        "recurrent_state": torch.zeros(16, 2, 3, 5),
    }
    layer._conv_cache_for_kernel = lambda n: layer.hsd_persistent_state_sidecar["conv_state"]
    layer._ssm_cache_for_kernel = lambda n: layer.hsd_persistent_state_sidecar["recurrent_state"]
    stores = cpu_class(MODEL, "HistoricalGatedDeltaNetAttention", ["_store_conv_state", "_store_recurrent_state"])
    layer._store_conv_state = stores._store_conv_state.__get__(layer)
    layer._store_recurrent_state = stores._store_recurrent_state.__get__(layer)
    return layer


def test_graph_restore_and_weight_reset_keep_captured_storage(monkeypatch):
    layer = graph_layer(monkeypatch)
    values = store(layer)
    addresses = {name: t.data_ptr() for name, t in layer._hsd_graph_prefix_storage().items()}
    for slot in (3, 4):
        assert layer.restore_hsd_prefix_snapshot(key="key", state_idx=slot, prefix_len=1024)
        torch.testing.assert_close(layer.hsd_persistent_state_sidecar["conv_state"][slot], values["conv_state"][0])
        torch.testing.assert_close(layer.hsd_persistent_state_sidecar["recurrent_state"][slot],
                                   values["recurrent_state"][0].transpose(-1, -2))
        assert layer.hsd_anchor_page_cache._lengths_by_state[slot] == 2
    assert any(event == "hsd_prefix_graph_restore_ready" for event, _ in layer.events)
    model = make_outer(layer)
    assert model.get_hsd_prefix_cache_status()["graph_storage"] == [addresses]
    model.invalidate_hsd_prefix_cache()
    assert {name: t.data_ptr() for name, t in layer._hsd_graph_prefix_storage().items()} == addresses
    assert not layer.hsd_anchor_page_cache._lengths_by_state.any()
    assert not layer.hsd_persistent_state_sidecar["recurrent_state"].any()


@pytest.mark.parametrize("damage,reason", [
    ("missing", "graph_storage_unready"),
    ("dynamic", "graph_storage_unready"),
    ("shape", "graph_state_shape"),
    ("state_capacity", "graph_state_capacity"),
    ("history_capacity", "graph_history_capacity"),
])
def test_graph_restore_preflight_rejects_storage_changes(monkeypatch, damage, reason):
    layer = graph_layer(monkeypatch)
    store(layer)
    if damage == "missing":
        layer.hsd_persistent_state_sidecar.pop("conv_state")
    elif damage == "dynamic":
        layer.hsd_anchor_page_cache._metadata_fixed_capacity = False
    elif damage == "shape":
        layer.hsd_persistent_state_sidecar["recurrent_state"] = torch.zeros(16, 2, 3, 6)
    elif damage == "state_capacity":
        layer.hsd_persistent_state_sidecar["conv_state"] = torch.zeros(2, 10, 4)
    else:
        layer.hsd_anchor_page_cache._lengths_by_state = torch.zeros(2, dtype=torch.long)
    assert layer._hsd_prefix_snapshot_restore_preflight(key="key", state_idx=3,
               history_state_idx=3, prefix_len=1024) == reason
    assert not layer.hsd_anchor_page_cache.has_state(3)


def test_graph_restore_detects_unexpected_reallocation(monkeypatch):
    layer = graph_layer(monkeypatch)
    store(layer)
    original_store = layer._store_conv_state
    original_buffer = layer.hsd_persistent_state_sidecar["conv_state"]
    def reallocate(index, value):
        layer.hsd_persistent_state_sidecar["conv_state"] = original_buffer.clone()
        original_store(index, value)
    layer._store_conv_state = reallocate
    with pytest.raises(RuntimeError, match="replaced captured"):
        layer.restore_hsd_prefix_snapshot(key="key", state_idx=3, prefix_len=1024)


def test_prefix_restore_is_eager_before_graph_decode():
    import numpy as np
    cls = cpu_class(VLLM / "v1/worker/gpu_model_runner.py", "GPUModelRunner",
                    ["_classify_historical_hsd_batch"], np=np)
    runner = cls()
    runner.input_batch = NS(req_ids=["warm"], num_computed_tokens_cpu=np.array([100]),
                            num_prompt_tokens=np.array([100]))
    runner.model_config = NS(hf_text_config=NS(anchor_chunk_size=512))
    runner._hsd_pending_prefix_restores = {"warm": ("key", 100)}
    kind, graphable, reason, boundary = runner._classify_historical_hsd_batch(
        num_reqs=1, num_scheduled_tokens_np=np.array([1]), spec_decode_metadata=None,
        num_encoder_reqs=0, allow_hsd_cudagraph=True)
    assert kind == "decode" and not graphable and reason == "prefix_restore" and not boundary


@pytest.mark.parametrize("enabled", [False, True])
def test_replay_evidence_is_recorded_only_after_real_replay(tmp_path, monkeypatch, enabled):
    import dataclasses
    from enum import Enum
    import json
    modes = Enum("Modes", ["NONE", "FULL"])
    batch = dataclasses.make_dataclass("Batch", [("num_tokens", int)], frozen=True)(1)
    context = NS(batch_descriptor=batch, cudagraph_runtime_mode=modes.FULL)
    order = []
    cls = cpu_class(VLLM / "compilation/cuda_graph.py", "CUDAGraphWrapper", ["__call__"],
                    CUDAGraphMode=modes, is_forward_context_available=lambda: True,
                    get_forward_context=lambda: context,
                    get_offloader=lambda: NS(sync_prev_onload=lambda: order.append("sync")), json=json)
    wrapper = cls()
    # This fixture extracts __call__ only; supply its native class-scope flag.
    from contextvars import ContextVar
    wrapper._warmup_active = ContextVar("prefix_test_graph_warmup", default=False)
    wrapper.runtime_mode = modes.FULL
    wrapper.is_debugging_mode = False
    wrapper.concrete_cudagraph_entries = {batch: NS(
        cudagraph=NS(replay=lambda: order.append("actual replay")), output="generated")}
    events = tmp_path / "events.jsonl"
    monkeypatch.setenv("VLLM_HSD_VARLEN_COUNTER_FILE", str(events))
    monkeypatch.setenv("VLLM_HSD_ENABLE_GRAPH_PREFIX_CACHE", "1" if enabled else "0")
    assert wrapper() == "generated"
    assert order == ["sync", "actual replay"]
    assert events.exists() == enabled
    if enabled:
        row = json.loads(events.read_text())
        assert row["event"] == "march_cuda_graph_replay" and row["mode"] == "FULL"


@pytest.mark.parametrize("tp", [1, 2])
def test_compact_prefix_kv_spec_shares_attention_block_geometry(tp):
    import math
    cls = cpu_class(MODEL, "HistoricalGatedDeltaNetAttention", ["get_kv_cache_spec"],
                    MambaSpec=lambda **kwargs: NS(**kwargs), get_dtype_size=lambda dtype: dtype.itemsize)
    layer = cls()
    layer.tp_size = tp
    layer.config = NS(num_key_value_heads=2, hidden_size=512, num_attention_heads=8, head_dim=64)
    layer.get_state_dtype = lambda: (torch.bfloat16, torch.float32)
    layer.mamba_type = "gdn"
    cfg = NS(cache_config=NS(block_size=16, mamba_cache_mode="none"), model_config=NS(dtype=torch.bfloat16))
    spec = layer.get_kv_cache_spec(cfg)
    assert spec.block_size == cfg.cache_config.block_size
    assert math.lcm(512, spec.block_size) == 512
    assert spec.shapes == ((1, 4), (1, 4))
    assert spec.page_size_padded == 2 * 16 * max(1, 2 // tp) * 64 * 2


@pytest.mark.parametrize("graph", [False, True])
def test_runner_emits_request_restore_evidence_independent_of_layer_aggregation(monkeypatch, graph):
    if graph:
        monkeypatch.setenv("VLLM_HSD_ENABLE_GRAPH_PREFIX_CACHE", "1")
    cls = cpu_class(VLLM / "v1/worker/gpu_model_runner.py", "GPUModelRunner",
                    ["_restore_hsd_prefix_snapshots"], NULL_BLOCK_ID=0)
    runner = cls()
    runner._hsd_pending_prefix_restores = {"warm": ("key", 512)}
    runner._uses_historical_hsd = lambda: True
    runner.input_batch = NS(req_ids=["warm"])
    events = []
    runner._record_historical_hsd_runner_event = lambda name, **payload: events.append((name, payload))
    runner.get_model = lambda: NS(restore_hsd_prefix_snapshot=lambda **kwargs: True)
    assert runner._restore_hsd_prefix_snapshots(torch.tensor([1]), torch.tensor([1]), 1) == [True]
    assert not runner._hsd_pending_prefix_restores
    assert sum(name == "hsd_prefix_graph_restore_ready" for name, _ in events) == int(graph)
    assert any(name == "hsd_prefix_snapshot_restore" and payload["request_id"] == "warm"
               for name, payload in events)


@pytest.mark.parametrize("present", [False, True])
def test_acceptance_weight_probe_uses_named_rpc_and_detached_copy(present):
    cls = cpu_class(VLLM / "v1/worker/gpu_worker.py", "Worker", ["get_hsd_anchor_probe_weight"])
    worker = cls()
    original = torch.nn.Parameter(torch.randn(1, 8))
    name = "model.anchor_embedding.weight" if present else "model.weight"
    worker.model_runner = NS(get_model=lambda: NS(named_parameters=lambda: iter([(name, original)])))
    if not present:
        with pytest.raises(RuntimeError, match="learned anchor"):
            worker.get_hsd_anchor_probe_weight()
    else:
        data = worker.get_hsd_anchor_probe_weight()
        assert data["name"] == name and data["dtype"] == "float32"
        value = torch.tensor(data["values"])
        assert value.data_ptr() != original.data_ptr()
        torch.testing.assert_close(value, original)


@pytest.mark.parametrize("shape_ok", [False, True])
def test_json_safe_weight_payload_rehydrates_on_worker(shape_ok):
    cls = cpu_class(VLLM / "v1/worker/gpu_worker.py", "Worker", ["reload_weights"])
    worker = cls()
    parameter = torch.nn.Parameter(torch.zeros(1, 8, dtype=torch.bfloat16))
    calls = []
    worker.model_runner = NS(get_model=lambda: NS(get_parameter=lambda name: parameter),
                            reload_weights=lambda **kwargs: calls.append(kwargs))
    values = [[0.125] * (8 if shape_ok else 7)]
    payload = [{"name": "model.anchor_embedding.weight", "values": values}]
    if not shape_ok:
        with pytest.raises(ValueError, match="shape mismatch"):
            worker.reload_weights(weights_payload=payload, is_checkpoint_format=False)
        assert not calls
    else:
        worker.reload_weights(weights_payload=payload, is_checkpoint_format=False)
        name, tensor = calls[0]["weights_iterator"][0]
        assert name == payload[0]["name"] and tensor.dtype == parameter.dtype
        torch.testing.assert_close(tensor, torch.full_like(parameter, 0.125))
        assert calls[0]["is_checkpoint_format"] is False


def test_actual_runner_reload_updates_learned_anchor_without_autograd_error():
    import time
    cls = cpu_class(VLLM / "v1/worker/gpu_model_runner.py", "GPUModelRunner",
                    ["reload_weights"], keep_inference_mode=True, time=time,
                    logger=NS(info_once=lambda *a, **kw: None, warning_once=lambda *a, **kw: None,
                              warning=lambda *a, **kw: None))
    runner = cls()
    model = torch.nn.Module()
    model.register_parameter("anchor", torch.nn.Parameter(torch.zeros(1, 8)))
    assert model.anchor.requires_grad and torch.is_grad_enabled()
    runner.get_model = lambda: model
    runner.model_config = NS(quantization=None)
    runner.reload_weights(weights_iterator=[("anchor", torch.full((1, 8), 0.125))],
                          is_checkpoint_format=False)
    torch.testing.assert_close(model.anchor, torch.full_like(model.anchor, 0.125))
    assert model.anchor.requires_grad


@pytest.mark.parametrize("past_text_len", [0, 1024])
def test_snapshot_capture_after_grouped_prefill_appended_current_anchors(past_text_len):
    layer = make_layer()
    states = torch.arange(4 * 2 * 5 * 3, dtype=torch.float32).reshape(1, 4, 2, 5, 3)
    keys = torch.randn(1, 4, 2, 5)
    boundaries = torch.tensor([512, 1024, 1536, 2048])
    cls = cpu_class(
        MODEL, "HistoricalGatedDeltaNetAttention", ["capture_hsd_prefix_snapshot_from_tensors"],
        _env_truthy=lambda key, default=False: default,
        _advance_hsd_conv_state=lambda conv, inputs, **kw: conv.clone(),
    )
    capture = cls()
    capture._materialize_hsd_anchor_history_for_snapshot = lambda *a, **kw: (states, keys, boundaries)
    capture._conv_state_for_request = lambda *a: None
    capture._record_hsd_varlen_event = layer._record_hsd_varlen_event
    capture._store_hsd_prefix_snapshot_parts = layer._store_hsd_prefix_snapshot_parts
    new = boundaries > past_text_len
    text_len = 2048 - past_text_len
    capture.capture_hsd_prefix_snapshot_from_tensors(
        state_idx=1, history_state_idx=1, past_text_len=past_text_len,
        snapshot_requests=[("key", 2048, text_len)],
        conv_state_snapshot=torch.zeros(1, 10, 4), pre_conv_inputs=torch.zeros(text_len, 10),
        query=None, key_tensor=None, value=None, g=None, beta=None,
        initial_state=states[:, 0], final_recurrent_state=states[:, -1],
        new_anchor_states=states[:, new], new_anchor_keys=keys[:, new],
        new_anchor_boundaries=boundaries[new],
    )
    snapshot = layer.hsd_prefix_snapshot_cache["key"]
    assert snapshot["anchor_boundaries"].tolist() == [512, 1024, 1536, 2048]
    torch.testing.assert_close(snapshot["anchor_states"], states, atol=0, rtol=0)
    assert layer.restore_hsd_prefix_snapshot(key="key", state_idx=3, prefix_len=2048)


@pytest.mark.parametrize("enabled,expected", [(False, 0), (True, 512)])
def test_dense_prefix_partitions_follow_memory_anchor_boundaries(enabled, expected):
    cls = cpu_class(VLLM / "v1/core/sched/scheduler.py", "Scheduler", [
        "_hsd_prefix_canonical_prefill_chunk_size", "_hsd_prefix_anchor_chunk_size",
        "_align_hsd_prefix_prefill_to_snapshot_boundary",
    ])
    scheduler = cls()
    scheduler.vllm_config = NS(model_config=NS(hf_text_config=config()))
    scheduler.hsd_prefix_cache_enabled = enabled
    scheduler.hsd_prefix_hash_block_size = 16
    scheduler.max_num_scheduled_tokens = 4096
    scheduler._uses_historical_hsd_prefix_cache = lambda: True
    scheduler._hsd_prefix_cache_aligned_prefill_enabled = lambda: True
    scheduler._hsd_prefix_cache_observed_alignment_enabled = lambda: False
    scheduler._record_hsd_prefix_cache_event = lambda *a, **kw: None
    assert scheduler._hsd_prefix_canonical_prefill_chunk_size() == expected
    request = NS(has_encoder_inputs=False, lora_request=None, num_prompt_tokens=2049,
                 num_tokens=2049, request_id="cold")
    partition, start = [], 0
    while start < 2049:
        count = scheduler._align_hsd_prefix_prefill_to_snapshot_boundary(
            request, step_start=start, num_scheduled_tokens=2049-start,
        )
        assert count > 0
        partition.append(count)
        start += count
    assert partition == ([512, 512, 512, 512, 1] if enabled else [2049])
    request.request_id = "restored"
    assert scheduler._align_hsd_prefix_prefill_to_snapshot_boundary(
        request, step_start=2048, num_scheduled_tokens=1,
    ) == 1


@pytest.mark.parametrize("enabled,computed,waiting,capacity,expected", [
    (True, [512, 0], 0, 16, True),
    (True, [512, 512], 0, 16, False),
    (True, [512, 512], 2, 16, True),
    (True, [512, 512], 2, 2, False),
    (False, [512, 0], 2, 16, False),
])
def test_serial_prefill_defers_decode_without_starving_full_batch(enabled, computed, waiting, capacity, expected):
    cls = cpu_class(VLLM / "v1/core/sched/scheduler.py", "Scheduler", ["_hsd_serial_prefix_prefill_pending"])
    scheduler = cls()
    scheduler.hsd_serialize_prefix_prefill = enabled
    scheduler.running = [NS(num_computed_tokens=n, num_prompt_tokens=512) for n in computed]
    scheduler.waiting, scheduler.skipped_waiting = [object()] * waiting, []
    scheduler.max_num_running_reqs = capacity
    assert scheduler._hsd_serial_prefix_prefill_pending() is expected


def test_prefix_profile_rejects_unstable_parallel_prefill(monkeypatch):
    monkeypatch.setenv("VLLM_HSD_ENABLE_PREFIX_CACHE", "1")
    monkeypatch.delenv("VLLM_HSD_SERIALIZE_PREFIX_PREFILL")
    with pytest.raises(NotImplementedError, match="prefix profile"):
        policy.validate_march_prefix_cache(config(), engine_config())


@pytest.mark.parametrize("compact", [False, True])
def test_moved_state_clear_preserves_other_requests_and_native_slot_identity(compact):
    cls = cpu_class(MODEL, "HistoricalGatedDeltaNetAttention", ["_clear_native_state_for_request"])
    layer = cls()
    layer.hsd_use_compact_state_slots = compact
    layer.kv_cache = [torch.arange(24).view(4, 3, 2).float() + 1,
                      torch.arange(48).view(4, 3, 2, 2).float() + 1]
    layer.hsd_persistent_state_sidecar = {"recurrent_state": torch.ones(4, 3, 2, 2)}
    before = [v.clone() for v in layer.kv_cache]
    pointers = [v.data_ptr() for v in layer.kv_cache]
    layer._clear_native_state_for_request(2)
    assert [v.data_ptr() for v in layer.kv_cache] == pointers
    for current, original in zip(layer.kv_cache, before):
        if not compact:
            original[2].zero_()
        torch.testing.assert_close(current, original, atol=0, rtol=0)
    torch.testing.assert_close(layer.hsd_persistent_state_sidecar["recurrent_state"][2], torch.zeros(3, 2, 2))
    assert layer.hsd_persistent_state_sidecar["recurrent_state"][[0, 1, 3]].eq(1).all()

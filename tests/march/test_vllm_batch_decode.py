"""Batched decode phase contracts and visible cache writes on tiny CPU fixtures."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn


@pytest.fixture(scope="module")
def modules():
    import ast
    import copy
    import importlib
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    package = root / "third_party/vllm/vllm/model_executor/models/march"
    name = "_safin1_batch_contract"
    spec = importlib.util.spec_from_file_location(
        name, package / "__init__.py", submodule_search_locations=[str(package)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(root / "third_party/vllm"))
        spec.loader.exec_module(module)
        parts = {
            part: importlib.import_module(name + "." + part)
            for part in (
                "backends",
                "decode",
                "decode_batch",
                "decode_batch_core",
                "decode_batch_no_anchor",
                "decode_batch_plan",
                "decode_batch_reference",
                "decode_batch_state",
                "layout",
                "runtime",
            )
        }
        # Compile the existing small model helpers without importing native
        # Qwen/model parents. These bodies are not substitutes for phase code.
        tree = ast.parse((package.parent / "safin1.py").read_text())
        owner = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef)
            and n.name == "HistoricalGatedDeltaNetAttention"
        )
        names = {
            "_post_conv_prep_hsd",
            "_state_index_for_request",
            "_history_state_index_for_request",
            "_record_hsd_decode_prepare_counter",
            "_record_hsd_decode_prepare_fallback",
        }
        functions = [
            copy.deepcopy(n)
            for n in owner.body
            if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        assert {n.name for n in functions} == names
        cpu = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                ast.ClassDef(
                    name="CPUAttention",
                    bases=[],
                    keywords=[],
                    body=functions,
                    decorator_list=[],
                ),
            ],
            type_ignores=[],
        )
        namespace = dict(
            torch=torch,
            F=torch.nn.functional,
            _env_truthy=parts["runtime"]._env_truthy,
            _l2norm=parts["runtime"]._l2norm,
            logger=NS(info=lambda *args, **kwargs: None),
        )
        exec(
            compile(ast.fix_missing_locations(cpu), "model CPU helpers", "exec"),
            namespace,
        )
        yield NS(
            model=NS(HistoricalGatedDeltaNetAttention=namespace["CPUAttention"]),
            backends=parts["backends"],
            decode=parts["decode"],
            batch=parts["decode_batch"],
            core=parts["decode_batch_core"],
            no_anchor=parts["decode_batch_no_anchor"],
            plan=parts["decode_batch_plan"],
            reference=parts["decode_batch_reference"],
            state=parts["decode_batch_state"],
            layout=parts["layout"],
        )
    for key in list(sys.modules):
        if key == name or key.startswith(name + "."):
            del sys.modules[key]


@pytest.fixture
def tiny(modules, monkeypatch):
    for key in list(__import__("os").environ):
        if key.startswith(("VLLM_HSD_", "HSD_")):
            monkeypatch.delenv(key)
    for name in (
        "ENABLE_NO_ANCHOR_COMMON_FAST_PATH",
        "ENABLE_NO_ANCHOR_COMMON_ALL_LAYERS",
        "ENABLE_UNSAFE_NO_ANCHOR_COMMON_MULTI",
        "USE_NATIVE_CONV_STATE_UPDATE",
        "ENABLE_UNSAFE_PADDED_STATIC_DECODE",
        "ENABLE_PADDED_STATIC_DECODE",
    ):
        monkeypatch.setenv("VLLM_HSD_" + name, "1")
    # The fixture executes real tensor/control-flow code on CPU; only GPU
    # admission and the two native state-update operators are controlled.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda tensor: True))
    context = lambda: NS(attn_metadata=None)
    monkeypatch.setattr(modules.plan, "get_forward_context", context)
    monkeypatch.setattr(modules.decode, "get_forward_context", context)
    holder = {}

    def conv(x, cache, weight, bias, activation, *, conv_state_indices, **kwargs):
        layer = holder["layer"]
        layer.calls.append("conv")
        for row, slot in enumerate(conv_state_indices.tolist()):
            if slot > 0:
                cache[slot].copy_(torch.roll(cache[slot], -1, -1))
                cache[slot, :, -1].copy_(x[row])
                layer.conv_updates[slot] += 1
        return x.clone()

    def recurrent(*, initial_state, out, ssm_state_indices, **kwargs):
        layer = holder["layer"]
        layer.calls.append("gdn")
        for row, slot in enumerate(ssm_state_indices.tolist()):
            if slot > 0:
                initial_state[slot].add_(1)
                layer.ssm_updates[slot] += 1
                out[row, 0].copy_(initial_state[slot].sum(-1))
            else:
                out[row].zero_()

    monkeypatch.setattr(modules.core, "causal_conv1d_update", conv)
    monkeypatch.setattr(
        modules.core, "fused_recurrent_gated_delta_rule_packed_decode", recurrent
    )
    monkeypatch.setattr(
        modules.decode, "fused_recurrent_gated_delta_rule_packed_decode", recurrent
    )

    class Projection(nn.Module):
        def __init__(self, size):
            super().__init__()
            self.weight = nn.Parameter(
                torch.arange(size * 4).reshape(size, 4).float() / 100
            )

        def forward(self, x):
            return x @ self.weight.t(), None

    class Layer(
        modules.batch.BatchedDecodeMixin,
        modules.decode.CachedDecodeMixin,
        modules.model.HistoricalGatedDeltaNetAttention,
        nn.Module,
    ):
        def __init__(self):
            nn.Module.__init__(self)
            self.config = NS(
                anchor_chunk_size=512,
                anchor_token_num=1,
                anchor_position_ids_type="zeros",
                hsd_persistent_memory_enabled=False,
            )
            self.layer_idx = 0
            self.hsd_decode_prepare_backend = "batched"
            self.hsd_decode_backend = "packed"
            self.hsd_post_conv_backend = "torch"
            self.hsd_layer_uses_global_read = False
            self.hsd_use_compact_state_slots = False
            self.hsd_persistent_memory_mode = "removed"
            self.hsd_history_out_rmsnorm_weight = None
            self.tp_size = self.num_k_heads = self.num_v_heads = 1
            self.key_dim = self.value_dim = self.head_k_dim = self.head_v_dim = 4
            self.hsd_local_route_query_heads = self.hsd_local_route_key_heads = 1
            self.hsd_route_head_dim = 4
            self.hsd_route_qk_l2norm = False
            self._hsd_decode_projection_fusion_supported = False
            self.hsd_decode_prepare_counters = {}
            self.hsd_decode_prepare_fallback_counts = {}
            self.in_proj_qkvz, self.in_proj_ba = Projection(16), Projection(2)
            self.anchor_q_proj, self.out_proj = Projection(4), Projection(4)
            self.A_log, self.dt_bias = torch.zeros(1), torch.zeros(1)
            self.conv1d = nn.Conv1d(12, 12, 4, groups=12)
            self.activation = "silu"
            self.norm = lambda x, gate: x * gate.sigmoid()
            self.conv_cache = torch.arange(4.0).view(4, 1, 1).expand(4, 12, 3).clone()
            self.ssm_cache = (
                torch.arange(4.0).view(4, 1, 1, 1).expand(4, 1, 4, 4).clone()
            )
            self.calls, self.events = [], []
            self.conv_updates, self.ssm_updates = [0] * 4, [0] * 4

        def _conv_cache_for_kernel(self, required_rows=None):
            return self.conv_cache

        def _ssm_cache_for_kernel(self, required_rows=None):
            return self.ssm_cache

        def _use_hsd_persistent_state_sidecar(self):
            return False

        def _use_hsd_static_text_state_sidecar(self):
            return False

        def _use_request_loop_conv(self, count):
            return False

        def _capture_hsd_graph_v3_tensor_sidecar(self, *args):
            pass

        def _record_hsd_varlen_event(self, name, **payload):
            self.events.append((name, payload))

        def _store_recurrent_state_batched(self, indices, states):
            valid = indices > 0
            self.ssm_cache.index_copy_(
                0, indices[valid].long(), states[valid].transpose(-1, -2)
            )
            return True

        def _stored_recurrent_state_for_request(self, state_idx, fallback=None):
            return (
                self.ssm_cache[state_idx : state_idx + 1].transpose(-1, -2).contiguous()
            )

        def _legacy_anchor_history(self, state_idx):
            return None, None, None

        def _can_use_paged_history_read(self, **kwargs):
            return False

    layer = Layer()
    holder["layer"] = layer
    return NS(layer=layer, holder=holder, conv=conv, recurrent=recurrent)


def make_layout(modules, layer, past, slots, padded=False):
    if padded:
        return modules.layout._build_hsd_padded_static_decode_layout(
            config=layer.config,
            positions=torch.tensor(past),
            hsd_num_computed_tokens=torch.tensor(past),
            hsd_state_indices=torch.tensor(slots),
            hsd_history_state_indices=torch.tensor(slots),
            hsd_decode_valid_mask=torch.tensor([s > 0 for s in slots]),
        )
    return modules.layout._build_hsd_layout(
        config=layer.config,
        positions=torch.tensor(past),
        query_start_loc=torch.arange(len(past) + 1),
        num_computed_tokens=torch.tensor(past),
        state_indices=torch.tensor(slots),
    )


@pytest.mark.parametrize(
    "past,slots,padded,common",
    [
        ([20], [1], False, True),
        ([20, 30], [2, 1], False, True),
        ([511], [1], False, True),
        ([511, 20], [2, 1], False, True),
        ([511, 511], [2, 1], False, True),
        ([20, 30], [2, 1], False, False),
        ([20, 20], [1, 0], True, True),
    ],
)
def test_decode_updates_each_active_slot_once_and_preserves_null_row(
    modules, tiny, monkeypatch, past, slots, padded, common
):
    if not common:
        monkeypatch.setenv("VLLM_HSD_DISABLE_NO_ANCHOR_COMMON_FAST_PATH", "1")
    layer = tiny.layer
    layout = make_layout(modules, layer, past, slots, padded)
    hidden = torch.arange(layout.num_expanded_tokens * 4).reshape(-1, 4).float() / 10
    output = torch.full_like(hidden, -99)
    conv_zero, ssm_zero = layer.conv_cache[0].clone(), layer.ssm_cache[0].clone()
    pointers = layer.conv_cache.data_ptr(), layer.ssm_cache.data_ptr()
    with torch.inference_mode():
        assert layer._forward_hsd_batched_decode_light(hidden, output, layout) is True
    assert layer.calls == ["conv", "gdn"]
    assert layer.ssm_updates == [int(i in slots and i > 0) for i in range(4)]
    assert layer.conv_updates == layer.ssm_updates
    assert pointers == (layer.conv_cache.data_ptr(), layer.ssm_cache.data_ptr())
    torch.testing.assert_close(layer.conv_cache[0], conv_zero, atol=0, rtol=0)
    torch.testing.assert_close(layer.ssm_cache[0], ssm_zero, atol=0, rtol=0)
    assert torch.isfinite(output).all() and not torch.any(output == -99)
    if padded:
        assert torch.equal(output[1], torch.zeros_like(output[1]))
    for slot in set(slots) - {0}:
        assert torch.all(layer.ssm_cache[slot] == slot + 1)


@pytest.mark.parametrize("result", [None, False, True])
def test_no_anchor_result_distinguishes_continue_and_terminal_failure(modules, result):
    calls = []

    class Flow(modules.batch.BatchedDecodeMixin):
        def _prepare_hsd_batched_decode(self, *args):
            calls.append("plan")
            return object()

        def _prepare_hsd_batched_decode_core(self, *args):
            calls.append("gdn_once")
            return object()

        def _append_hsd_batched_decode_boundaries(self, *args):
            calls.append("boundary")
            return object()

        def _try_hsd_batched_decode_no_anchor(self, *args):
            calls.append("fast")
            return result

        def _run_hsd_batched_decode_reference(self, *args):
            calls.append("reference")
            return object()

        def _finalize_hsd_batched_decode_reference(self, *args):
            calls.append("finalize")
            return True

    value = Flow()._forward_hsd_batched_decode_light(None, None, None)
    assert value is (True if result is None else result)
    assert calls == ["plan", "gdn_once", "boundary", "fast"] + (
        ["reference", "finalize"] if result is None else []
    )


def test_prepared_state_is_frozen_and_aliases_existing_buffers(modules, tiny):
    layer = tiny.layer
    layout = make_layout(modules, layer, [20], [1])
    hidden = torch.ones(1, 4)
    output = torch.empty_like(hidden)
    with torch.inference_mode():
        prepared = layer._prepare_hsd_batched_decode(hidden, output, layout)
    assert prepared.batch.hidden_states is hidden and prepared.batch.output is output
    assert prepared.batch.conv_cache is layer.conv_cache
    assert prepared.batch.ssm_state is layer.ssm_cache
    assert layer.calls == []
    with pytest.raises(FrozenInstanceError):
        prepared.batch.req_count = 7
    with pytest.raises(FrozenInstanceError):
        prepared.policy.no_anchor_common_fast_path = False


@pytest.mark.parametrize("raw_q", [False, True])
def test_history_backend_miss_continues_without_replaying_gdn(
    modules, tiny, monkeypatch, raw_q
):
    monkeypatch.setenv("VLLM_HSD_ENABLE_RAW_Q_PAGED_READ", str(int(raw_q)))
    layer = tiny.layer
    layout = make_layout(modules, layer, [20], [1])
    hidden, output = torch.ones(1, 4), torch.empty(1, 4)
    with torch.inference_mode():
        prepared = layer._prepare_hsd_batched_decode(hidden, output, layout)
        core = layer._prepare_hsd_batched_decode_core(prepared)
        history = replace(
            prepared.history,
            common_history_candidate_count=1,
            common_history_candidate_tensor=torch.tensor([0]),
            page_cache_for_common=object(),
        )
        unavailable = replace(prepared, history=history)
        read_core = replace(core, router_q_all=torch.ones(1, 1, 4))
        monkeypatch.setattr(
            modules.backends, "_load_hsd_paged_history_read", lambda: None
        )
        assert layer._try_hsd_batched_decode_no_anchor(unavailable, read_core) is None
        # Continue with the exact same phase objects, as production does.
        assert read_core.raw_q_paged_read is raw_q
        assert (read_core.query is None) is raw_q
        boundary = layer._append_hsd_batched_decode_boundaries(unavailable, read_core)
        reference = layer._run_hsd_batched_decode_reference(
            unavailable, read_core, boundary
        )
        assert reference.query is not None
        if not raw_q:
            assert reference.query is read_core.query
        assert (
            layer._finalize_hsd_batched_decode_reference(
                unavailable, read_core, reference
            )
            is True
        )
    assert layer.calls == ["conv", "gdn"]
    assert layer.ssm_updates == [0, 1, 0, 0]
    assert any(
        name == "hsd_no_anchor_common_fallback"
        and values["reason"] == "paged_history_loader_missing"
        for name, values in layer.events
    )


def test_static_sidecar_failure_returns_false_after_original_state_update(
    modules, tiny
):
    layer = tiny.layer
    layout = make_layout(modules, layer, [20], [1])
    with torch.inference_mode():
        prepared = layer._prepare_hsd_batched_decode(
            torch.ones(1, 4), torch.empty(1, 4), layout
        )
        core = layer._prepare_hsd_batched_decode_core(prepared)
        prepared = replace(
            prepared,
            policy=replace(
                prepared.policy, persistent_text_cache=True, use_static_sidecar=True
            ),
        )
        layer._update_hsd_static_text_state_sidecar_batched = lambda *args: False
        assert layer._try_hsd_batched_decode_no_anchor(prepared, core) is False
    assert layer.calls == ["conv", "gdn"]
    assert layer.ssm_updates == [0, 1, 0, 0]
    assert layer.hsd_decode_prepare_fallback_counts == {
        "static_text_state_sidecar_update_failed": 1
    }


@pytest.mark.parametrize("failure", ["anchor_count", "null_slot", "missing_conv"])
def test_admission_rejection_does_not_start_native_updates(modules, tiny, failure):
    layer = tiny.layer
    if failure == "anchor_count":
        layer.config.anchor_token_num = 2
    if failure == "missing_conv":
        layer._conv_cache_for_kernel = lambda *args: None
    layout = make_layout(modules, layer, [20], [0 if failure == "null_slot" else 1])
    output = torch.full((layout.num_expanded_tokens, 4), -99.0)
    with torch.inference_mode():
        assert (
            layer._forward_hsd_batched_decode_light(
                torch.ones_like(output), output, layout
            )
            is False
        )
    assert layer.calls == [] and layer.ssm_updates == [0] * 4
    assert torch.all(output == -99)


@pytest.mark.parametrize("where", ["plan", "reference"])
def test_failed_stage_stops_before_later_stages(modules, where):
    calls = []

    class Flow(modules.batch.BatchedDecodeMixin):
        def _prepare_hsd_batched_decode(self, *args):
            return False if where == "plan" else object()

        def _prepare_hsd_batched_decode_core(self, *args):
            calls.append("core")
            return object()

        def _append_hsd_batched_decode_boundaries(self, *args):
            return object()

        def _try_hsd_batched_decode_no_anchor(self, *args):
            return None

        def _run_hsd_batched_decode_reference(self, *args):
            return False

        def _finalize_hsd_batched_decode_reference(self, *args):
            pytest.fail("failed stage reached finalize")

    assert Flow()._forward_hsd_batched_decode_light(None, None, None) is False
    assert calls == ([] if where == "plan" else ["core"])

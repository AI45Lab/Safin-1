"""Prefill backend failures and anchor/state contracts without a CUDA engine."""

import builtins
import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "third_party/vllm/vllm/model_executor/models/march"


@pytest.fixture(scope="module")
def prefill(vllm_cpu_runtime):
    name = "_safin1_prefill_contract"
    spec = importlib.util.spec_from_file_location(
        name, PACKAGE / "__init__.py", submodule_search_locations=[str(PACKAGE)]
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / "third_party/vllm"))
        spec.loader.exec_module(package)
        modules = {
            part: importlib.import_module(name + "." + part)
            for part in (
                "backends",
                "prefill",
                "prefill_segments",
                "prefill_history",
                "prefill_batch",
            )
        }
        yield SimpleNamespace(**modules)
    for key in list(sys.modules):
        if key == name or key.startswith(name + "."):
            del sys.modules[key]


OPTIONAL_BACKENDS = [
    (
        "varlen_prefill",
        "_HSD_VARLEN_PREFILL",
        "march.inference.varlen_segments",
        ("chunk_gated_delta_rule_with_anchor_states_varlen_forward",),
    ),
    (
        "flashqla_prefill",
        "_HSD_FLASHQLA_PREFILL",
        "march.kernels.march_chunk_gdn",
        ("hsd_flashqla_tile_gated_delta_rule_inference",),
    ),
    (
        "deterministic_anchor_read",
        "_HSD_DETERMINISTIC_ANCHOR_READ",
        "march.inference.varlen_segments",
        ("deterministic_anchor_state_read",),
    ),
    (
        "paged_history_read",
        "_HSD_PAGED_HISTORY_READ",
        "march.inference.paged_history_read",
        (
            "HSDHistoryCacheSpec",
            "HSDAnchorPageCache",
            "paged_batched_decode_history_read",
        ),
    ),
    (
        "fused_global_read",
        "_HSD_FUSED_GLOBAL_READ",
        "march.kernels.march_global_read",
        ("hsd_fused_global_read",),
    ),
]


@pytest.mark.parametrize("loader,cache,path,names", OPTIONAL_BACKENDS)
@pytest.mark.parametrize("fails", [False, True])
def test_optional_backend_memoizes_success_or_original_exception(
    prefill, monkeypatch, loader, cache, path, names, fails
):
    backend = prefill.backends
    monkeypatch.setattr(backend, cache, None)
    monkeypatch.setattr(backend, cache + "_IMPORT_ERROR", None)
    error = ImportError("offline fixture backend unavailable")
    symbols = {name: object() for name in names}
    calls = []
    original = builtins.__import__

    def importer(name, *args, **kwargs):
        if name == path:
            calls.append(name)
            if fails:
                raise error
            return SimpleNamespace(**symbols)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importer)
    load = getattr(backend, "_load_hsd_" + loader)
    expected = (
        None
        if fails
        else tuple(symbols[n] for n in names)
        if len(names) > 1
        else symbols[names[0]]
    )
    assert load() == expected
    assert load() == expected
    assert calls == [path]
    assert getattr(backend, cache + "_IMPORT_ERROR") is (error if fails else None)


@pytest.mark.parametrize(
    "loader,cache",
    [
        ("paged_static_history_read", "_HSD_PAGED_STATIC_HISTORY_READ"),
        (
            "paged_blocked_prefill_history_read",
            "_HSD_PAGED_BLOCKED_PREFILL_HISTORY_READ",
        ),
        ("valid_mask_copy", "_HSD_VALID_MASK_COPY"),
        ("strided_gate_rmsnorm", "_HSD_STRIDED_GATE_RMSNORM"),
    ],
)
def test_required_paged_backend_errors_propagate(prefill, monkeypatch, loader, cache):
    monkeypatch.setattr(prefill.backends, cache, None)
    original = builtins.__import__

    def importer(name, *args, **kwargs):
        if name == "march.inference.paged_history_read":
            raise ImportError("required fixture kernel missing")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importer)
    with pytest.raises(ImportError, match="required fixture kernel"):
        getattr(prefill.backends, "_load_hsd_" + loader)()


def test_prefill_modules_share_backend_cache_owner(prefill):
    assert all(
        getattr(prefill, part).backends is prefill.backends
        for part in ("prefill", "prefill_segments", "prefill_history", "prefill_batch")
    )


@pytest.mark.parametrize(
    "hit,force_segments,expected",
    [
        ("flashqla", False, ["flashqla"]),
        ("varlen", False, ["flashqla", "varlen"]),
        ("segments", False, ["flashqla", "varlen", "segments"]),
        ("segments", True, ["flashqla", "segments"]),
    ],
)
def test_single_request_backend_fallback_order(prefill, hit, force_segments, expected):
    calls = []
    result = object()

    def backend(name):
        def run(**kwargs):
            calls.append(name)
            return result if name == hit else None

        return run

    layer = prefill.prefill.PrefillMixin()
    layer._hsd_force_segments_fallback = force_segments
    layer._run_hsd_prefill_flashqla = backend("flashqla")
    layer._run_hsd_prefill_varlen = backend("varlen")
    layer._run_hsd_segments = backend("segments")
    output = layer._run_hsd_prefill_vllm(*([None] * 9))
    assert output is result
    assert calls == expected


def test_strict_producer_failure_never_falls_back(prefill):
    layer = prefill.prefill.PrefillMixin()
    layer.hsd_local_prefill_producer_backend = "flashqla_tile"
    layer.hsd_local_prefill_producer_strict = True
    layer._run_hsd_prefill_varlen = lambda **kw: pytest.fail("strict failure fell back")
    layer._run_hsd_segments = layer._run_hsd_prefill_varlen
    x = torch.ones(1, 1, 1, 1)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="not_cuda"):
        layer._run_hsd_prefill_vllm(
            x,
            x,
            x,
            x[..., 0],
            x[..., 0],
            x[:, :0],
            torch.tensor([], dtype=torch.long),
            0,
            x,
        )


@pytest.mark.parametrize(
    "length,persistent,past",
    [
        (511, False, 0),
        (512, False, 0),
        (513, False, 0),
        (1025, False, 0),
        (513, True, 0),
        (513, False, 512),
    ],
)
def test_segments_preserve_boundary_snapshots_and_do_not_seed_text_with_memory(
    prefill, length, persistent, past
):
    layer = prefill.prefill_segments.SegmentPrefillMixin()
    layer.head_k_dim = 1
    layer._use_deterministic_anchor_read = lambda count: False

    # Analytic scalar recurrence: S[t+1] = S[t] + value[t]. This makes both
    # the anchor timestamp and final-state layout independently observable.
    def scalar_recurrence(*, q, k, v, initial_state, **kwargs):
        text = v.cumsum(1) + initial_state[..., 0].unsqueeze(1)
        return text, text[:, -1].unsqueeze(-1)

    layer.chunk_gated_delta_rule = scalar_recurrence
    x = torch.ones(1, length, 1, 1)
    gates = torch.ones(1, length, 1)
    boundaries = ([0] if persistent else []) + list(
        range(past + 512, past + length + 1, 512)
    )
    anchors = torch.ones(1, len(boundaries), 1, 1)
    initial = torch.full((1, 1, 1, 1), 7.0)
    text, anchor, states, final = layer._run_hsd_segments(
        x, x, x, gates, gates, anchors, torch.tensor(boundaries), past, initial
    )
    expected = 7 + torch.arange(1, length + 1, dtype=torch.float32)
    torch.testing.assert_close(text.flatten(), expected, atol=0, rtol=0)
    expected_states = torch.tensor(
        [0 if b == 0 else 7 + b - past for b in boundaries], dtype=torch.float32
    )
    torch.testing.assert_close(states.flatten(), expected_states, atol=0, rtol=0)
    torch.testing.assert_close(anchor.flatten(), expected_states, atol=0, rtol=0)
    assert final.item() == 7 + length
    assert initial.item() == 7

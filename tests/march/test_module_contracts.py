"""Supported architecture, shared training implementation and cache behavior."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from march.safin1.runtime_validation import RETIRED_ARCHITECTURE_FLAGS, validate_safin1_runtime


@pytest.mark.parametrize('field', RETIRED_ARCHITECTURE_FLAGS)
def test_retired_architecture_rejected_before_construction(field):
    from huggingface_hub.errors import StrictDataclassClassValidationError
    from march.safin1.configuration_safin1 import Safin1TextConfig
    from march.safin1.configuration_safin1_moe import Safin1MoeTextConfig

    for cls in (Safin1TextConfig, Safin1MoeTextConfig):
        with pytest.raises(StrictDataclassClassValidationError, match='Retired'):
            cls(**{field: True})
        # Existing checkpoints may serialize inactive metadata.
        validate_safin1_runtime(cls(**{field: False}))


@pytest.mark.parametrize('field,value', [('hsd_local_state_mode', 'chunk'),
                                       ('hsd_history_read_q_mode', 'hidden_residual_lora')])
def test_retired_modes_rejected(field, value):
    with pytest.raises(ValueError, match='retired variant'):
        validate_safin1_runtime({field: value})


def test_shared_chunk_adapter_and_profile_state():
    from march.kernels import march_chunk_gdn as shared
    path = Path(__file__).parents[2] / "third_party/mcore-bridge/src/mcore_bridge/model/modules/march_chunk_gdn.py"
    spec = importlib.util.spec_from_file_location("march_bridge_chunk_adapter", path)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)

    assert bridge.hsd_chunk_gated_delta_rule_with_anchor_states is shared.hsd_chunk_gated_delta_rule_with_anchor_states
    shared.clear_hsd_chunk_profile_stats()
    shared._PROFILE_STATS['test/value'] = 3.0
    assert bridge.pop_hsd_chunk_profile_stats() == {'test/value': 3.0}
    assert shared.pop_hsd_chunk_profile_stats() == {}


def test_reorder_sidecar_lengths_and_states():
    from march.safin1.cache_utils import _reorder_hsd_sidecar_cache

    cache = SimpleNamespace(hsd_seen_text_tokens_per_row=torch.tensor([7, 9]), hsd_seen_expanded_tokens_per_row=torch.tensor([8, 10]))
    # This helper augments the upstream KV/conv cache reorder, rather than replacing it.
    _reorder_hsd_sidecar_cache(cache, torch.tensor([1, 0]))
    assert list(cache.hsd_seen_text_tokens_per_row) == [9, 7]
    assert list(cache.hsd_seen_expanded_tokens_per_row) == [10, 8]


def test_diagnostic_stats_and_explicit_param_selection(tmp_path, monkeypatch):
    from march.diagnostics.precision_trace import PrecisionTracer
    from march.diagnostics.tensor_stats import _tensor_stats

    value = torch.tensor([1.0, -2.0, 3.0])
    stats = _tensor_stats(value)
    assert stats['finite']
    assert stats['max_abs'] == 3.0
    assert stats['norm'] == pytest.approx(14.0**0.5)
    monkeypatch.setenv('MEGATRON_PRECISION_TRACE_PARAM_CHAIN', '1')
    monkeypatch.delenv('MEGATRON_PRECISION_TRACE_PARAM_CHAIN_REGEX', raising=False)
    monkeypatch.delenv('MEGATRON_PRECISION_TRACE_TARGET_PARAM_REGEX', raising=False)
    with pytest.raises(ValueError, match='requires'):
        PrecisionTracer(tmp_path, {1}, 'dp0')

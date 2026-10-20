"""Training settings must describe the graph that is actually implemented."""
from types import SimpleNamespace

import pytest

from march.safin1.runtime_validation import validate_safin1_training_runtime, validate_training_environment


UNIMPLEMENTED_RUNTIME = [
    {"hsd_history_mix_mode": mode} for mode in ("convex", "joint_softmax_logm", "gated_additive")
] + [{"hsd_route_weight_momentum_gamma": value} for value in (0.25, -0.1, float("nan"), float("inf"))]


@pytest.mark.parametrize("runtime", UNIMPLEMENTED_RUNTIME)
def test_unimplemented_training_settings_fail_for_dicts_and_objects(runtime):
    field = next(iter(runtime))
    for config in (runtime, SimpleNamespace(**runtime)):
        with pytest.raises(ValueError, match=field + "=.*not implemented"):
            validate_safin1_training_runtime(config)


def test_supported_training_metadata_is_preserved():
    runtime = {"hsd_history_mix_mode": "additive", "hsd_route_weight_momentum_gamma": 0.0,
               "hsd_local_state_mode": "full", "hsd_history_read_q_mode": "shared_local",
               "hsd_history_read_q_lora_rank": -7, "hsd_anchor_mlp": False}
    original = runtime.copy()
    validate_safin1_training_runtime(runtime)
    assert runtime == original
    validate_safin1_training_runtime({})


@pytest.mark.parametrize("name,value,supported", [
    ("HSD_LOCAL_STATE_MODE", "chunk", "full"),
    ("HSD_HISTORY_READ_Q_MODE", "hidden_residual_lora", "shared_local"),
])
def test_retired_environment_modes_fail_instead_of_being_ignored(name, value, supported):
    with pytest.raises(ValueError, match=name):
        validate_training_environment({name: value})
    for allowed in ("", supported, "  " + supported.upper() + "  "):
        validate_training_environment({name: allowed})


@pytest.mark.parametrize("field,value,raw", [
    ("hsd_read_topk", 4, "8"), ("hsd_read_block_size", 32768, "1024"),
    ("hsd_use_global_read", True, "false"), ("hsd_topk_backend", "reference", "sparse"),
    ("hsd_history_scale", 1.0, "0.5"), ("hsd_route_head_dim", None, "128"),
    ("hsd_global_read_layers", "10-30", "2-39"),
])
def test_environment_cannot_change_serialized_runtime(field, value, raw):
    runtime = {"model_type": "safin1_4b", field: value}
    for config in (runtime, SimpleNamespace(**runtime)):
        with pytest.raises(ValueError, match=field):
            validate_training_environment({field.upper(): raw}, config=config)
    assert runtime[field] == value


@pytest.mark.parametrize("value,raw", [(4, "4"), (True, "YES"), (False, "off"),
                                       (1.0, "1"), (None, "null"), ("sparse", "SPARSE")])
def test_matching_runtime_environment_remains_compatible(value, raw):
    validate_training_environment({"HSD_EXAMPLE": raw}, config={"hsd_example": value})


def test_legacy_read_block_alias_cannot_override_config():
    with pytest.raises(ValueError, match="HIST_READ_BLOCK_SIZE"):
        validate_training_environment({"HIST_READ_BLOCK_SIZE": "1024"},
                                      config={"hsd_read_block_size": 32768})
    validate_training_environment({"HIST_READ_BLOCK_SIZE": "32768"},
                                  config={"hsd_read_block_size": 32768})


@pytest.mark.parametrize("raw", ["garbage", "4.5", "nan"])
def test_invalid_runtime_environment_does_not_silently_fall_back(raw):
    with pytest.raises(ValueError, match="HSD_READ_TOPK"):
        validate_training_environment({"HSD_READ_TOPK": raw}, config={"hsd_read_topk": 4})


@pytest.mark.parametrize("name", ["USE_HIST_LINEAR_ATTN", "USE_MCORE_GDN", "SWIFT_USE_MCORE_GDN"])
def test_march_architecture_is_selected_by_config_and_conflicting_flags_fail(name):
    from march.safin1.runtime_validation import march_training_enabled
    config = {"model_type": "safin1_35b_a3b"}
    assert march_training_enabled(config, {})
    with pytest.raises(ValueError, match=name):
        march_training_enabled(config, {name: "0"})
    assert march_training_enabled(config, {name: "1"})
    assert not march_training_enabled({"hf_model_type": "qwen3_5"}, {})


def test_active_anchor_profile_enables_march_without_legacy_switch():
    from march.safin1.runtime_validation import march_training_enabled
    assert march_training_enabled({"anchor_chunk_size": 512, "anchor_token_num": 1}, {})
    validate_training_environment({"HSD_TOPK_EXECUTION_MODE": "token"}, config={"hsd_read_topk": 4})


@pytest.mark.parametrize("raw", ["y", "t", "garbage"])
def test_boolean_runtime_guard_uses_same_values_as_training_parser(raw):
    with pytest.raises(ValueError, match="HSD_USE_GLOBAL_READ"):
        validate_training_environment({"HSD_USE_GLOBAL_READ": raw}, config={"hsd_use_global_read": True})

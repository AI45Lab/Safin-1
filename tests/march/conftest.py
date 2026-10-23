"""Explicit CPU substitutes for the native vLLM runtime in contract tests."""

from importlib.machinery import ModuleSpec
import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def vllm_cpu_runtime():
    """Load real MARCH helpers while requiring explicit native-operator doubles.

    Only requesting modules use these substitutes. Every sys.modules entry is
    restored at teardown, so CUDA integration tests keep the actual runtime.
    """
    import torch

    if torch.cuda.is_available():
        yield
        return

    with pytest.MonkeyPatch.context() as patch:
        modules = {}

        def dependency(name, **attributes):
            if name not in modules:
                parent, _, child = name.rpartition(".")
                if parent:
                    dependency(parent)
                module = ModuleType(name)
                module.__spec__ = ModuleSpec(name, loader=None)
                modules[name] = module
                patch.setitem(sys.modules, name, module)
                if parent:
                    owner = modules[parent]
                    owner.__path__ = []
                    owner.__spec__.submodule_search_locations = []
                    setattr(owner, child, module)
            modules[name].__dict__.update(attributes)
            return modules[name]

        def native_operator(*args, **kwargs):
            pytest.fail("CPU contract reached a native vLLM operator without a test double")

        dependency(
            "vllm.forward_context",
            get_forward_context=lambda: SimpleNamespace(attn_metadata=None),
        )
        class ContractLogger(logging.LoggerAdapter):
            info_once = logging.LoggerAdapter.info
            warning_once = logging.LoggerAdapter.warning

        dependency(
            "vllm.logger", init_logger=lambda name: ContractLogger(logging.getLogger(name), {})
        )
        dependency(
            "vllm.model_executor.layers.batch_invariant",
            linear_batch_invariant=native_operator,
        )
        dependency(
            "vllm.model_executor.layers.fla.ops",
            fused_post_conv_read_q=native_operator,
            fused_recurrent_gated_delta_rule=native_operator,
            fused_recurrent_gated_delta_rule_packed_decode=native_operator,
        )
        dependency(
            "vllm.model_executor.layers.mamba.ops.causal_conv1d",
            causal_conv1d_fn=native_operator,
            causal_conv1d_update=native_operator,
        )
        dependency("vllm.utils.torch_utils", _encode_layer_name=native_operator)
        yield

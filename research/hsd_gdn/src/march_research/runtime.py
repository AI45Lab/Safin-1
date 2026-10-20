"""Explicit model registration for the historical HSD/FLA research process."""
from importlib import import_module
from pathlib import Path

MODELS = {
    'hsd_gated_deltanet': ('march_research.hsd', 'HSDGatedDeltaNet'),
    'gated_deltanet': ('fla.models.gated_deltanet', 'GatedDeltaNet'),
    'delta_net': ('fla.models.delta_net', 'DeltaNet'),
    'kda': ('fla.models.kda', 'KDA'),
    'transformer': ('fla.models.transformer', 'Transformer'),
}


def model_classes(model_type):
    if model_type not in MODELS:
        raise ValueError(f'Unsupported research model_type: {model_type}')
    package, name = MODELS[model_type]
    if model_type == 'hsd_gated_deltanet':
        config = import_module(package + '.configuration_hsd_gated_deltanet')
        model = import_module(package + '.modeling_hsd_gated_deltanet')
    else:
        config = import_module(package + '.configuration_' + model_type)
        model = import_module(package + '.modeling_' + model_type)
    cls = getattr(config, name + 'Config')
    causal = getattr(model, name + 'ForCausalLM')
    from transformers import AutoConfig, AutoModelForCausalLM
    AutoConfig.register(model_type, cls, exist_ok=True)
    AutoModelForCausalLM.register(cls, causal, exist_ok=True)
    if model_type == 'gated_deltanet' and isinstance(getattr(causal, '_tied_weights_keys', None), list):
        causal._tied_weights_keys = {'lm_head.weight': 'model.embeddings.weight'}
    return cls, causal


def assert_runtime(require_cuda=False):
    from .runtime_policy import validate_research_environment
    validate_research_environment()
    import fla
    root = Path(__file__).resolve().parents[4]
    expected = root / 'third_party/flash-linear-attention/fla'
    if not Path(fla.__file__).resolve().is_relative_to(expected):
        raise RuntimeError(f'Research must use the vendored FLA fork: {fla.__file__}')
    if require_cuda:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('This historical training/inference path requires CUDA')

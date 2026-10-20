"""Merge adapters for export while retaining exact training weights."""
from contextlib import contextmanager

import torch


@contextmanager
def temporary_merged_lora(models, lora_type, adapter_name='default'):
    """Restore original base weights/merge flags, even when merge/export fails.

    BF16 addition followed by subtraction is not an exact inverse. CPU snapshots
    avoid reserving another copy of all targeted weights in GPU memory. Only
    adapter-targeted base tensors are copied; optimizer/adapter state is untouched.
    """
    records = []
    with torch.no_grad():
        for model in models:
            for module in model.modules():
                if not isinstance(module, lora_type):
                    continue
                base = module.get_base_layer()
                weights = ([getattr(base, f'weight{i}') for i in range(base.num_gemms)]
                           if module.is_grouped else [base.weight])
                records.append((module, [(weight, weight.detach().to(device='cpu', copy=True))
                                         for weight in weights], list(module.merged_adapters)))
    try:
        with torch.no_grad():
            for module, _, _ in records:
                module.merge(adapter_names=[adapter_name])
        yield
    finally:
        with torch.no_grad():
            for module, weights, original_adapters in records:
                for weight, original in weights:
                    weight.copy_(original)
                module.merged_adapters[:] = original_adapters

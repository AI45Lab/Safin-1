"""A merged checkpoint export must leave the live training model unchanged."""
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('march_lora_export',
                                            ROOT / 'swift/megatron/utils/lora_export.py')
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class MutatingAdapter(torch.nn.Module):
    """Exercise the external merge interface, including partial merge failures."""
    def __init__(self, grouped=False, fail=False, device='cpu'):
        super().__init__()
        self.is_grouped = grouped
        self.fail = fail
        self.base = torch.nn.Module()
        self.base.num_gemms = 2 if grouped else 1
        for i in range(self.base.num_gemms):
            key = f'weight{i}' if grouped else 'weight'
            self.base.register_parameter(key, torch.nn.Parameter(
                torch.full((2, 3), 0.00123 + i * 0.0001, dtype=torch.bfloat16, device=device)))
        self.merged_adapters = []

    def get_base_layer(self):
        return self.base

    def merge(self, adapter_names):
        for parameter in self.base.parameters():
            parameter.add_(0.1)
        self.merged_adapters.extend(adapter_names)
        if self.fail:
            raise RuntimeError('partial external merge failure')


@pytest.mark.parametrize('grouped', [False, True])
@pytest.mark.parametrize('failure', ['none', 'merge', 'export'])
def test_export_restores_exact_bf16_weights_and_merge_flags(grouped, failure):
    model = MutatingAdapter(grouped, fail=failure == 'merge')
    original = [p.detach().clone() for p in model.base.parameters()]
    pointers = [p.data_ptr() for p in model.base.parameters()]
    # This fixture is deliberately not reversible with BF16 subtraction.
    assert not torch.equal((original[0] + 0.1) - 0.1, original[0])
    try:
        with export.temporary_merged_lora([model], MutatingAdapter):
            assert model.merged_adapters == ['default']
            assert not torch.equal(next(model.base.parameters()), original[0])
            if failure == 'export':
                raise RuntimeError('export failed')
    except RuntimeError:
        assert failure != 'none'
    else:
        assert failure == 'none'
    assert model.merged_adapters == []
    assert [p.data_ptr() for p in model.base.parameters()] == pointers
    assert all(torch.equal(p, before) for p, before in zip(model.base.parameters(), original))


def test_partial_merge_restores_all_modules_and_existing_flags():
    first = MutatingAdapter()
    second = MutatingAdapter(grouped=True, fail=True)
    first.merged_adapters = ['previous']
    model = torch.nn.ModuleList([first, second])
    weights = [p.detach().clone() for p in model.parameters()]
    with pytest.raises(RuntimeError, match='partial'):
        with export.temporary_merged_lora([model], MutatingAdapter):
            pass
    assert first.merged_adapters == ['previous']
    assert second.merged_adapters == []
    assert all(torch.equal(p, before) for p, before in zip(model.parameters(), weights))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for CPU snapshot transfer')
def test_cuda_export_restores_gpu_storage_and_values():
    model = MutatingAdapter(grouped=True, device='cuda')
    original = [p.detach().clone() for p in model.parameters()]
    pointers = [p.data_ptr() for p in model.parameters()]
    with export.temporary_merged_lora([model], MutatingAdapter):
        assert all(p.is_cuda for p in model.parameters())
    assert [p.data_ptr() for p in model.parameters()] == pointers
    assert all(torch.equal(p, before) for p, before in zip(model.parameters(), original))

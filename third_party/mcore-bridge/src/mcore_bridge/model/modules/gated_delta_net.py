# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import os
import torch
import torch.nn.functional as F
import transformer_engine
from contextlib import nullcontext
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint, sharded_state_dict_default
from typing import List, Optional

from mcore_bridge.config import ModelConfig

try:
    from fla.modules.convolution import causal_conv1d
    from fla.modules.l2norm import l2norm
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
except ImportError:
    causal_conv1d = None
    l2norm = None
    chunk_gated_delta_rule = None

try:
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet as _GatedDeltaNet
    from megatron.core.ssm.gated_delta_net import GatedDeltaNetSubmodules, torch_chunk_gated_delta_rule
except ImportError:
    _GatedDeltaNet = object

try:
    from megatron.core.ssm.gated_delta_net import tensor_a2a_cp2hp, tensor_a2a_hp2cp
except ImportError:
    # Code borrowed from NVIDIA/Megatron-LM main:
    # megatron/core/ssm/gated_delta_net.py
    from megatron.core.ssm.mamba_context_parallel import (
        _all_to_all_cp2hp,
        _all_to_all_hp2cp,
        _redo_attention_load_balancing,
        _undo_attention_load_balancing,
    )

    def tensor_a2a_cp2hp(
        tensor: torch.Tensor,
        seq_dim: int,
        head_dim: int,
        cp_group: torch.distributed.ProcessGroup,
        split_sections: Optional[List[int]] = None,
        undo_attention_load_balancing: bool = True,
    ) -> torch.Tensor:
        """All-to-all communication from context parallel to hidden parallel."""
        cp_size = cp_group.size()
        if cp_size == 1:
            return tensor
        assert seq_dim == 0, f'tensor_a2a_cp2hp only supports seq_dim == 0 for now, but got {seq_dim=}'
        assert head_dim == -1 or head_dim == 2, (
            f'tensor_a2a_cp2hp only supports head_dim == -1 or 2 for now, but got {head_dim=}')
        assert tensor.dim() == 3, f'tensor_a2a_cp2hp only supports 3-d input tensor for now, got {tensor.dim()=}'
        if split_sections is not None:
            outputs = []
            for x in torch.split(tensor, split_sections, dim=head_dim):
                outputs.append(
                    tensor_a2a_cp2hp(
                        x,
                        seq_dim=seq_dim,
                        head_dim=head_dim,
                        cp_group=cp_group,
                        undo_attention_load_balancing=False,
                    ))
            tensor = torch.cat(outputs, dim=head_dim)
        else:
            tensor = _all_to_all_cp2hp(tensor, cp_group)
        if undo_attention_load_balancing:
            tensor = _undo_attention_load_balancing(tensor, cp_size)
        return tensor

    def tensor_a2a_hp2cp(
        tensor: torch.Tensor,
        seq_dim: int,
        head_dim: int,
        cp_group: torch.distributed.ProcessGroup,
        split_sections: Optional[List[int]] = None,
        redo_attention_load_balancing: bool = True,
    ) -> torch.Tensor:
        """All-to-all communication from hidden parallel to context parallel."""
        cp_size = cp_group.size()
        if cp_size == 1:
            return tensor
        assert seq_dim == 0, f'tensor_a2a_hp2cp only supports seq_dim == 0 for now, but got {seq_dim=}'
        assert head_dim == -1 or head_dim == 2, (
            f'tensor_a2a_hp2cp only supports head_dim == -1 or 2 for now, but got {head_dim=}')
        assert tensor.dim() == 3, f'tensor_a2a_hp2cp only supports 3-d input tensor for now, got {tensor.dim()=}'
        if redo_attention_load_balancing:
            tensor = _redo_attention_load_balancing(tensor, cp_size)
        if split_sections is not None:
            outputs = []
            for x in torch.split(tensor, split_sections, dim=head_dim):
                outputs.append(
                    tensor_a2a_hp2cp(
                        x,
                        seq_dim=seq_dim,
                        head_dim=head_dim,
                        cp_group=cp_group,
                        redo_attention_load_balancing=False,
                    ))
            tensor = torch.cat(outputs, dim=head_dim)
        else:
            tensor = _all_to_all_hp2cp(tensor, cp_group)
        return tensor


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, '').strip().lower()
    if not raw:
        return default
    return raw in {'1', 'true', 'yes', 'on'}


def _parse_layer_selector(value: str):
    raw = (value or '').strip().lower()
    if raw in {'all', '*'}:
        return None
    if raw in {'', '0', 'false', 'none', 'off'}:
        return set()
    layers = set()
    for part in raw.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start_s, end_s = part.split('-', 1)
            start, end = int(start_s), int(end_s)
            if start <= 0 or end <= 0 or end < start:
                raise ValueError(f'Invalid MEGATRON_PRECISION_TRACE_INTERNAL_LAYERS range: {part}')
            layers.update(range(start, end + 1))
        else:
            layer = int(part)
            if layer <= 0:
                raise ValueError(f'Invalid MEGATRON_PRECISION_TRACE_INTERNAL_LAYERS value: {part}')
            layers.add(layer)
    return layers


def _precision_trace_internal_layer_allowed(layer_number: int) -> bool:
    layers = _parse_layer_selector(os.environ.get('MEGATRON_PRECISION_TRACE_INTERNAL_LAYERS') or '1')
    return layers is None or int(layer_number) in layers


# Code borrowed from NVIDIA/Megatron-LM
def _unpack_sequence(x, cu_seqlens, dim=1):
    unpacked_x = []
    num_seqs = cu_seqlens.shape[0] - 1
    for i in range(num_seqs):
        idx_start = cu_seqlens[i].item()
        idx_end = cu_seqlens[i + 1].item()
        chunked_index = [slice(None)] * dim + [slice(idx_start, idx_end)]
        unpacked_x.append(x[tuple(chunked_index)])
    return unpacked_x


# Code borrowed from NVIDIA/Megatron-LM
# Avoid the warning caused by `param[slices]`
def get_parameter_local_cp(
    param: torch.Tensor,
    dim: int,
    cp_group: torch.distributed.ProcessGroup,
    split_sections: Optional[List[int]] = None,
) -> torch.Tensor:
    """Get the local parameter for the current context parallel rank.

    Args:
        param (torch.Tensor): The entire parameter to get the local parameter for.
        dim (int): The dimension to split the parameter along. Usually the dimension of head.
        cp_group (torch.distributed.ProcessGroup): The context parallel group.
        split_sections (Optional[List[int]]): If not None,
            first split the parameter along the dimension dim into sections,
            then get the local hidden parallel weights separately,
            finally concatenate the local hidden parallel weights along the dimension dim.

    Returns:
        torch.Tensor: The local parameter for the current context parallel rank.
    """

    cp_size = cp_group.size()
    cp_rank = cp_group.rank()

    # No need to split if CP size is 1.
    if cp_size == 1:
        return param

    # Split first if needed.
    if split_sections is not None:
        inputs = torch.split(param, split_sections, dim=dim)
        outputs = []
        for p in inputs:
            p = get_parameter_local_cp(p, dim, cp_group)
            outputs.append(p)
        return torch.cat(outputs, dim=dim)

    # Slice the parameter.
    slices = [slice(None)] * param.dim()
    dim_size = param.size(dim=dim)
    slices[dim] = slice(cp_rank * dim_size // cp_size, (cp_rank + 1) * dim_size // cp_size)
    param = param[tuple(slices)]
    return param


class GatedDeltaNet(_GatedDeltaNet):

    def __init__(self, config: ModelConfig, submodules: 'GatedDeltaNetSubmodules', *args, **kwargs):
        if config.linear_decoupled_in_proj:
            in_proj = submodules.in_proj
            submodules.in_proj = IdentityOp
        if 'cp_comm_type' not in inspect.signature(_GatedDeltaNet).parameters:
            kwargs.pop('cp_comm_type', None)
        super().__init__(config, submodules, *args, **kwargs)
        if not config.linear_decoupled_in_proj:
            return
        submodules.in_proj = in_proj
        self.in_proj_qkvz_dim = self.qk_dim * 2 + self.v_dim * 2
        self.in_proj_ba_dim = self.num_value_heads * 2
        del self.in_proj
        self.in_proj_qkvz = build_module(
            submodules.in_proj,
            self.hidden_size,
            self.in_proj_qkvz_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='fc1',
            tp_group=self.pg_collection.tp,
        )
        if config.fp8_param:
            fp8_context = transformer_engine.pytorch.fp8_model_init(enabled=False)
        else:
            fp8_context = nullcontext()
        with fp8_context:
            self.in_proj_ba = build_module(
                submodules.in_proj,
                self.hidden_size,
                self.in_proj_ba_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='fc1_ba',
                tp_group=self.pg_collection.tp,
            )

    def _trace_precision_internal_tensor(self, name: str, tensor: Optional[torch.Tensor]) -> None:
        if not isinstance(tensor, torch.Tensor) or not _env_bool('MEGATRON_PRECISION_TRACE_GDN_INTERNALS', False):
            return
        try:
            from march.diagnostics.precision_trace import (
                get_active_precision_tracer,
                precision_trace_runtime_step,
            )
        except Exception:
            return
        tracer = get_active_precision_tracer()
        if tracer is None:
            return
        layer_number = int(getattr(self, 'layer_number', 0) or 0)
        if not _precision_trace_internal_layer_allowed(layer_number):
            return
        tracer.trace_tensor(
            f'layer_{layer_number}/{name}',
            tensor,
            step=precision_trace_runtime_step(),
            category='gdn_internal',
            extra={'layer_number': layer_number},
        )

    def _trace_precision_internal_backward_tensor(self, name: str, tensor: Optional[torch.Tensor]) -> None:
        if (not isinstance(tensor, torch.Tensor)
                or not _env_bool('MEGATRON_PRECISION_TRACE_GDN_INTERNALS', False)
                or not _env_bool('MEGATRON_PRECISION_TRACE_INTERNAL_BACKWARD', False)):
            return
        try:
            from march.diagnostics.precision_trace import (
                get_active_precision_tracer,
                precision_trace_runtime_step,
            )
        except Exception:
            return
        tracer = get_active_precision_tracer()
        if tracer is None:
            return
        layer_number = int(getattr(self, 'layer_number', 0) or 0)
        if not _precision_trace_internal_layer_allowed(layer_number):
            return
        tracer.trace_tensor_backward(
            f'layer_{layer_number}/{name}',
            tensor,
            step=precision_trace_runtime_step(),
            category='gdn_internal_bwd',
            layer_number=layer_number,
        )

    def _trace_precision_hidden_text_before_proj(self, hidden_states: torch.Tensor) -> None:
        if not _env_bool('MEGATRON_PRECISION_TRACE_GDN_INTERNALS', False):
            return
        if self.sp_size == 1:
            full_hidden_states = hidden_states
        else:
            try:
                from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
            except Exception:
                return
            with torch.no_grad():
                full_hidden_states = gather_from_sequence_parallel_region(
                    hidden_states.detach(),
                    tensor_parallel_output_grad=False,
                    group=self.pg_collection.tp,
                )
        self._trace_precision_internal_tensor(
            'hidden_states_text_before_proj',
            full_hidden_states.transpose(0, 1).contiguous(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        **kwargs,
    ):
        """
        Perform a forward pass through the GDN module.

        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            key_value_states (Optional[Tensor]): Key/value states (for cross attention).
            inference_context (Optional[BaseInferenceContext]): Inference context that manages
                KV cache.
            attention_bias (Optional[Tensor]): Attention bias.
            packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.

        Return:
            (Tuple[Tensor, Tensor]) GDN output and bias.

        """
        # TODO: Deal with attention_mask
        from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push

        inference_context = deprecate_inference_params(inference_context, inference_params)

        seq_len, batch, _ = hidden_states.shape
        cp_size = self.config.context_parallel_size
        seq_len = seq_len * self.sp_size * cp_size

        if inference_context is not None:
            assert (
                inference_context.is_static_batching()), 'GDN does not currently support dynamic inference batching.'
            assert not self.config.sequence_parallel
            # TODO: support inference
            raise NotImplementedError('GDN does not support inference for now.')

        cu_seqlens = None if packed_seq_params is None else packed_seq_params.cu_seqlens_q
        self._trace_precision_internal_backward_tensor('hidden_states_local_text_before_proj_grad', hidden_states)
        self._trace_precision_hidden_text_before_proj(hidden_states)
        # Input projection
        num_key_heads_per_device = self.num_key_heads // self.tp_size // cp_size
        nvtx_range_push(suffix='in_proj')
        if self.config.linear_decoupled_in_proj:
            qkvz, _ = self.in_proj_qkvz(hidden_states)
            if self.config.fp8_param:
                fp8_context = transformer_engine.pytorch.fp8_autocast(enabled=False)
            else:
                fp8_context = nullcontext()
            with fp8_context:
                ba, _ = self.in_proj_ba(hidden_states)
            qkvz = qkvz.view(qkvz.shape[:-1] + (num_key_heads_per_device, qkvz.shape[-1] // num_key_heads_per_device))
            ba = ba.view(ba.shape[:-1] + (num_key_heads_per_device, ba.shape[-1] // num_key_heads_per_device))
            qkvzba = torch.concat([qkvz, ba], dim=-1).view(*qkvz.shape[:2], -1)
        else:
            qkvzba, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix='in_proj')

        if cp_size > 1:
            if cu_seqlens is not None:
                unpacked_qkvzba = _unpack_sequence(qkvzba, cu_seqlens // cp_size, dim=0)
                outputs = []
                for qkvzba_i in unpacked_qkvzba:
                    qkvzba_i = tensor_a2a_cp2hp(
                        qkvzba_i,
                        seq_dim=0,
                        head_dim=-1,
                        cp_group=self.pg_collection.cp,
                    )
                    outputs.append(qkvzba_i)
                qkvzba = torch.cat(outputs, dim=0)
            else:
                # CP All to All: CP to HP
                qkvzba = tensor_a2a_cp2hp(
                    qkvzba,
                    seq_dim=0,
                    head_dim=-1,
                    cp_group=self.pg_collection.cp,
                )
        # Transpose: s b x --> b s x
        # From sbhd to bshd format
        qkvzba = qkvzba.view(qkvzba.shape[:-1]
                             + (num_key_heads_per_device, qkvzba.shape[-1] // num_key_heads_per_device))
        qkvzba = qkvzba.transpose(0, 1)
        self._trace_precision_internal_tensor('qkvzba_text_after_proj', qkvzba.contiguous())
        self._trace_precision_internal_backward_tensor('qkvzba_text_after_proj_grad', qkvzba)
        qkv, gate, beta, alpha = torch.split(
            qkvzba,
            [
                (self.qk_dim * 2 + self.v_dim) // self.num_key_heads,
                self.v_dim // self.num_key_heads,
                self.num_value_heads // self.num_key_heads,
                self.num_value_heads // self.num_key_heads,
            ],
            dim=-1,
        )
        gate = gate.reshape(batch, seq_len, -1, self.value_head_dim)
        beta = beta.reshape(batch, seq_len, -1)
        alpha = alpha.reshape(batch, seq_len, -1)
        qkv = qkv.reshape(batch, seq_len, -1)
        self._trace_precision_internal_tensor('gate_text', gate)

        # Convolution on qkv
        nvtx_range_push(suffix='conv1d')
        if cp_size > 1:
            conv1d_weight = get_parameter_local_cp(
                self.conv1d.weight,
                dim=0,
                cp_group=self.pg_collection.cp,
            )
            conv1d_bias = (
                get_parameter_local_cp(
                    self.conv1d.bias,
                    dim=0,
                    cp_group=self.pg_collection.cp,
                ) if self.conv_bias else None)
        else:
            conv1d_weight = self.conv1d.weight
            conv1d_bias = self.conv1d.bias

        if (causal_conv1d is None) or self.config.deterministic_mode:
            assert cu_seqlens is None, 'Packed sequences are not supported when fla is not available.'
            qkv = qkv.transpose(1, 2).contiguous()  # b, s, d -> b, d, s
            conv_out = F.conv1d(
                input=qkv,
                weight=conv1d_weight,
                bias=conv1d_bias,
                stride=self.conv1d.stride,
                padding=self.conv1d.padding,
                dilation=self.conv1d.dilation,
            )
            qkv = self.act_fn(conv_out[..., :seq_len])
            qkv = qkv.transpose(1, 2)  # b, d, s -> b, s, d
        else:
            assert self.activation in ['silu', 'swish']
            qkv = causal_conv1d(
                x=qkv,
                weight=conv1d_weight.squeeze(1),  # d, 1, w -> d, w
                bias=conv1d_bias,
                activation=self.activation,
                cu_seqlens=cu_seqlens,
            )[0]
        nvtx_range_pop(suffix='conv1d')
        self._trace_precision_internal_tensor('qkvzba_text_after_conv', qkv)
        self._trace_precision_internal_backward_tensor('qkvzba_text_after_conv_grad', qkv)
        # Split qkv into query, key, and value
        qkv = qkv.view(qkv.shape[:-1] + (num_key_heads_per_device, qkv.shape[-1] // num_key_heads_per_device))
        query, key, value = torch.split(
            qkv,
            [self.qk_dim // self.num_key_heads, self.qk_dim // self.num_key_heads, self.v_dim // self.num_key_heads],
            dim=-1,
        )
        query = query.reshape(batch, seq_len, -1, self.key_head_dim)
        key = key.reshape(batch, seq_len, -1, self.key_head_dim)
        value = value.reshape(batch, seq_len, -1, self.value_head_dim)
        # Apply L2 norm to query and key
        if self.use_qk_l2norm:
            query = l2norm(query.contiguous())
            key = l2norm(key.contiguous())
        if self.num_value_heads // self.num_key_heads > 1:
            query = query.repeat_interleave(self.num_value_heads // self.num_key_heads, dim=2)
            key = key.repeat_interleave(self.num_value_heads // self.num_key_heads, dim=2)

        # Make contiguous
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        gate = gate.contiguous()
        beta = beta.contiguous()
        alpha = alpha.contiguous() # 进kernel前内存连续

        # Calculate g and beta
        nvtx_range_push(suffix='g_and_beta')
        if cp_size > 1:
            A_log_local_cp = get_parameter_local_cp(self.A_log, dim=0, cp_group=self.pg_collection.cp)
            dt_bias_local_cp = get_parameter_local_cp(self.dt_bias, dim=0, cp_group=self.pg_collection.cp)
        else:
            A_log_local_cp, dt_bias_local_cp = self.A_log, self.dt_bias
        g = -A_log_local_cp.float().exp() * F.softplus(alpha.float() + dt_bias_local_cp)  # In fp32
        beta = beta.sigmoid()
        nvtx_range_pop(suffix='g_and_beta')
        self._trace_precision_internal_tensor('q_text', query)
        self._trace_precision_internal_tensor('k_text', key)
        self._trace_precision_internal_tensor('v_text', value)
        self._trace_precision_internal_tensor('beta_text', beta)
        self._trace_precision_internal_tensor('g_text', g)
        self._trace_precision_internal_backward_tensor('gate_text_grad', gate)
        self._trace_precision_internal_backward_tensor('q_text_grad', query)
        self._trace_precision_internal_backward_tensor('k_text_grad', key)
        self._trace_precision_internal_backward_tensor('v_text_grad', value)
        self._trace_precision_internal_backward_tensor('beta_text_grad', beta)
        self._trace_precision_internal_backward_tensor('g_text_grad', g)

        nvtx_range_push(suffix='gated_delta_rule')
        if self.config.deterministic_mode:
            assert cu_seqlens is None, ('cu_seqlens is not supported for torch_chunk_gated_delta_rule for now.')
            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
            )
        else:
            core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
                cu_seqlens=cu_seqlens,
            )
        nvtx_range_pop(suffix='gated_delta_rule')
        self._trace_precision_internal_tensor('delta_o_text', core_attn_out)
        self._trace_precision_internal_backward_tensor('delta_o_text_grad', core_attn_out)

        # RMSNorm
        nvtx_range_push(suffix='gated_norm')
        norm_out = self._apply_gated_norm(core_attn_out, gate)
        nvtx_range_pop(suffix='gated_norm')

        # Transpose: b s x --> s b x
        # From bshd back to sbhd format
        norm_out = norm_out.reshape(batch, seq_len, -1)
        self._trace_precision_internal_tensor('gated_norm_text', norm_out)
        self._trace_precision_internal_backward_tensor('gated_norm_text_grad', norm_out)
        norm_out = norm_out.transpose(0, 1).contiguous()
        self._trace_precision_internal_backward_tensor('out_proj_input_grad', norm_out)
        if cp_size > 1:
            if cu_seqlens is not None:
                unpacked_norm_out = _unpack_sequence(norm_out, cu_seqlens, dim=0)
                outputs = []
                for norm_out_i in unpacked_norm_out:
                    norm_out_i = tensor_a2a_hp2cp(norm_out_i, seq_dim=0, head_dim=-1, cp_group=self.pg_collection.cp)
                    outputs.append(norm_out_i)
                norm_out = torch.cat(outputs, dim=0)
            else:
                norm_out = tensor_a2a_hp2cp(norm_out, seq_dim=0, head_dim=-1, cp_group=self.pg_collection.cp)

        # Output projection
        nvtx_range_push(suffix='out_proj')
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix='out_proj')
        self._trace_precision_internal_tensor('out_proj_text', out)
        self._trace_precision_internal_tensor('attention_out_text', out)
        self._trace_precision_internal_backward_tensor('attention_out_text_grad', out)

        return out, out_bias

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None, tp_group=None):
        """Provide a sharded state dictionary for distributed checkpointing."""
        from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group

        # Guard for cases metadata is not provided
        metadata = ensure_metadata_has_dp_cp_group(metadata)

        sharded_state_dict = {}
        # Parameters
        self._save_to_state_dict(sharded_state_dict, '', keep_vars=True)
        tensor_parallel_layers_axis_map = {
            'A_log': 0,
            'dt_bias': 0,
        }
        # MarchGatedDeltaNet registers this layer-local value-state
        # parameter directly on the attention module.  It is sharded over the
        # local value-head axis exactly like A_log/dt_bias.
        if 'persistent_memory_state' in sharded_state_dict:
            tensor_parallel_layers_axis_map['persistent_memory_state'] = 0
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict,
            prefix,
            tensor_parallel_layers_axis_map=tensor_parallel_layers_axis_map,
            sharded_offsets=sharded_offsets,
            tp_group=(tp_group if tp_group is not None else self.pg_collection.tp),
            dp_cp_group=metadata['dp_cp_group'],
        )
        # Submodules
        tp_group = tp_group if tp_group is not None else self.pg_collection.tp
        for name, module in self.named_children():
            if name == 'conv1d':
                # Add TP sharding for Conv1d
                module_sd = module.state_dict(prefix='', keep_vars=True)
                tp_sharding_map = {'weight': 0}
                if self.conv_bias:
                    tp_sharding_map['bias'] = 0
                module_sharded_sd = make_sharded_tensors_for_checkpoint(
                    module_sd,
                    f'{prefix}{name}.',
                    tp_sharding_map,
                    sharded_offsets,
                    tp_group=tp_group,
                    dp_cp_group=metadata['dp_cp_group'],
                )
            else:
                module_sharded_sd = sharded_state_dict_default(
                    module, f'{prefix}{name}.', sharded_offsets, metadata, tp_group=tp_group)

            sharded_state_dict.update(module_sharded_sd)

        return sharded_state_dict

# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import math
import megatron.core
import os
import torch
import torch.nn.functional as F
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from megatron.core import parallel_state, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TELinear, TERowParallelLinear
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings import rope_utils
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt import GPTModel as McoreGPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import (gather_from_sequence_parallel_region,
                                                    gather_from_tensor_model_parallel_region)
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler, MTPLossLoggingHelper, roll_tensor
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.utils import WrappedTensor, deprecate_inference_params
from packaging import version
from types import MethodType
from typing import Dict, Optional, Tuple

from mcore_bridge.config import ModelConfig
from mcore_bridge.utils import get_logger, split_cp_inputs

from .modules.gated_delta_net import tensor_a2a_hp2cp
from .rope import dynamic_rope_update, get_rope_inv_freq

logger = get_logger()

mcore_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')


class OutputLayerLinear(TELinear):

    def forward(self, hidden_states, *args, **kwargs):
        return super().forward(hidden_states)

    def sharded_state_dict(
            self,
            prefix: str = '',
            sharded_offsets: Tuple[Tuple[int, int, int]] = (),
            metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        res = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        for k, v in res.items():
            if k.endswith('._extra_state'):
                if v.data is not None and v.data.numel() == 0:
                    v.data = None
        return res


@contextmanager
def _module_sequence_parallel(module, enabled: bool):
    if not hasattr(module, 'sequence_parallel'):
        yield
        return
    old_value = module.sequence_parallel
    module.sequence_parallel = enabled
    try:
        yield
    finally:
        module.sequence_parallel = old_value


@contextmanager
def _child_modules_sequence_parallel(module, enabled: bool):
    changed = []
    for child in module.modules():
        if hasattr(child, 'sequence_parallel'):
            changed.append((child, child.sequence_parallel))
            child.sequence_parallel = enabled
    try:
        yield
    finally:
        for child, old_value in reversed(changed):
            child.sequence_parallel = old_value




class GPTModel(McoreGPTModel):
    config: ModelConfig

    def __init__(
        self,
        config: ModelConfig,
        transformer_layer_spec: ModuleSpec,
        pre_process: bool = True,
        post_process: bool = True,
        mtp_block_spec: Optional[ModuleSpec] = None,
        vp_stage: Optional[int] = None,
    ):
        vocab_size = math.ceil(
            config.padded_vocab_size / config.tensor_model_parallel_size) * config.tensor_model_parallel_size
        hf_rope_scaling = config.rope_scaling
        if config.multi_latent_attention:
            config.rope_type = 'rope'  # use transformers implementation
            # Set default value, the following content will not be used. (dummy)
            config.mscale_all_dim = 0.
            config.cache_mla_latents = False
            config.rotary_scaling_factor = 40
            if hf_rope_scaling and hf_rope_scaling['rope_type'] == 'yarn':
                # softmax_scale
                config.mscale = hf_rope_scaling['mscale']
                config.mscale_all_dim = hf_rope_scaling['mscale_all_dim']
                config.rotary_scaling_factor = hf_rope_scaling['factor']
        self.hf_rope_scaling = hf_rope_scaling
        if mcore_013:
            kwargs = {'vp_stage': vp_stage}
        else:
            self.vp_stage = vp_stage
            assert vp_stage is None, 'megatron-core==0.12 does not support vp_stage'
            kwargs = {}
        super().__init__(
            config,
            transformer_layer_spec,
            vocab_size,
            config.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            share_embeddings_and_output_weights=not config.untie_embeddings_and_output_weights,
            position_embedding_type=config.position_embedding_type,
            rotary_base=config.rotary_base,
            mtp_block_spec=mtp_block_spec,
            **kwargs,
        )
        if config.multi_latent_attention:
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=config.qk_pos_emb_head_dim,
                rotary_percent=1,
                rotary_interleaved=config.rotary_interleaved,
                rotary_base=config.rotary_base,
                use_cpu_initialization=config.use_cpu_initialization,
            )
            # save memory
            for i in range(len(self.decoder.layers)):
                if hasattr(self.decoder.layers[i].self_attention, 'rotary_pos_emb'):
                    del self.decoder.layers[i].self_attention.rotary_pos_emb
        self._configure_rotary_embedding(config)
        if self.config.task_type == 'seq_cls' and self.post_process:
            self.output_layer = OutputLayerLinear(
                config.hidden_size,
                self.config.num_labels,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                parallel_mode=None,
                skip_weight_param_allocation=False,
            )
            self.output_layer.weight.average_gradients_across_tp_domain = True
        elif self.config.task_type == 'embedding' and self.post_process:
            self.output_layer = None

        if self.attention_scaling != 1 and config.apply_rope_fusion:
            config.apply_rope_fusion = False
            logger.warning(f'`apply_rope_fusion` does not support `attention_scaling`. '
                           f'Setting `config.apply_rope_fusion`: {config.apply_rope_fusion}')
        if not config.apply_rope_fusion:
            self._patch_apply_rotary_pos_emb()
        if getattr(self, 'mtp', None) is not None:
            for layer in self.mtp.layers:
                # compat megatron-core main branch
                if not hasattr(layer, 'transformer_layer'):

                    def _value(self):
                        return getattr(self, 'mtp_model_layer')

                    setattr(layer.__class__, 'transformer_layer', property(_value))
                attention = layer.transformer_layer.self_attention
                attention.config = copy.copy(attention.config)
                attention.config.apply_rope_fusion = False
        self._hsd_enabled = self._patch_hsd_layers()
        self._init_hsd_anchor_embedding()

    def _configure_rotary_embedding(self, config: ModelConfig) -> None:
        """Apply HF RoPE overrides only when the MCore base created RoPE.

        With ``position_embedding_type='none'``,
        MCore intentionally has no ``rotary_pos_emb`` attribute.
        """
        self.attention_scaling = 1.
        rotary_pos_emb = getattr(self, 'rotary_pos_emb', None)
        if rotary_pos_emb is None:
            return
        new_inv_freq, self.attention_scaling = get_rope_inv_freq(config)
        rotary_pos_emb.inv_freq = new_inv_freq.to(rotary_pos_emb.inv_freq.device)

    def _lookup_packed_rotary_pos_emb(
        self,
        rotary_pos_emb: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply document-relative THD positions to an ordinary RoPE table."""
        return rotary_pos_emb[position_ids]

    def _get_vocab_embedding_token(self, token_id: int) -> torch.Tensor:
        word_embeddings = getattr(getattr(self, 'embedding', None), 'word_embeddings', None)
        weight = getattr(word_embeddings, 'weight', None)
        if weight is None:
            raise RuntimeError('Cannot initialize HSD anchor embedding from placeholder: word embeddings are missing.')

        token_id = int(token_id)
        if hasattr(word_embeddings, 'vocab_start_index') and hasattr(word_embeddings, 'vocab_end_index'):
            vocab_start = int(word_embeddings.vocab_start_index)
            vocab_end = int(word_embeddings.vocab_end_index)
        elif torch.distributed.is_available() and torch.distributed.is_initialized():
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            vocab_start = int(weight.shape[0]) * int(tp_rank)
            vocab_end = vocab_start + int(weight.shape[0])
        else:
            vocab_start = 0
            vocab_end = int(weight.shape[0])
        local_token = vocab_start <= token_id < vocab_end
        token_vector = torch.zeros(weight.shape[1], dtype=weight.dtype, device=weight.device)
        found = torch.zeros((), dtype=torch.float32, device=weight.device)
        if local_token:
            token_vector.copy_(weight[token_id - vocab_start].detach())
            found.fill_(1.0)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tp_group = parallel_state.get_tensor_model_parallel_group()
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                torch.distributed.all_reduce(token_vector, group=tp_group)
                torch.distributed.all_reduce(found, group=tp_group)
        if float(found.item()) <= 0.0:
            raise ValueError(
                'anchor_placeholder_token_id must be present in the local or tensor-parallel vocab shard, '
                f'got token_id={token_id}, local_range=[{vocab_start}, {vocab_end}).')
        return token_vector

    def _get_vocab_embedding_mean(self) -> torch.Tensor:
        word_embeddings = getattr(getattr(self, 'embedding', None), 'word_embeddings', None)
        weight = getattr(word_embeddings, 'weight', None)
        if weight is None:
            raise RuntimeError('Cannot initialize HSD anchor embedding from vocab mean: word embeddings are missing.')

        local_sum = weight.detach().float().sum(dim=0)
        count = torch.tensor(float(weight.shape[0]), dtype=torch.float32, device=weight.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tp_group = parallel_state.get_tensor_model_parallel_group()
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                torch.distributed.all_reduce(local_sum, group=tp_group)
                torch.distributed.all_reduce(count, group=tp_group)
        if float(count.item()) <= 0.0:
            raise RuntimeError('Cannot initialize HSD anchor embedding from an empty vocab embedding table.')
        return local_sum / count

    def _get_hsd_bos_token_id(self) -> int:
        anchor_bos_token_id = getattr(self.config, 'anchor_bos_token_id', None)
        if anchor_bos_token_id is not None:
            return int(anchor_bos_token_id)

        config_candidates = [self.config]
        hf_config = getattr(self.config, 'hf_config', None)
        if hf_config is not None:
            config_candidates.append(hf_config)
            text_config = getattr(hf_config, 'text_config', None)
            if text_config is not None:
                config_candidates.append(text_config)

        for candidate in config_candidates:
            token_id = getattr(candidate, 'bos_token_id', None)
            if token_id is not None:
                return int(token_id)
        raise ValueError(
            "anchor_embedding_init_from='bos' requires anchor_bos_token_id, bos_token_id, "
            "or hf_config.bos_token_id. If the training config does not carry bos_token_id, set "
            "anchor_bos_token_id to the BOS token id.")

    def _initialize_hsd_anchor_embedding_weight(self) -> None:
        init_from = getattr(self.config, 'anchor_embedding_init_from', 'random')
        if init_from == 'random':
            self.config.init_method(self.anchor_embedding.weight)
        elif init_from in {'placeholder', 'bos'}:
            token_id = (
                self._get_hsd_bos_token_id()
                if init_from == 'bos'
                else getattr(self.config, 'anchor_placeholder_token_id', 0)
            )
            token_vector = self._get_vocab_embedding_token(token_id)
            token_vector = token_vector.to(
                device=self.anchor_embedding.weight.device,
                dtype=self.anchor_embedding.weight.dtype,
            )
            self.anchor_embedding.weight.copy_(token_vector.view(1, -1).expand_as(self.anchor_embedding.weight))
        elif init_from == 'vocab_mean':
            mean_vector = self._get_vocab_embedding_mean().to(
                device=self.anchor_embedding.weight.device,
                dtype=self.anchor_embedding.weight.dtype,
            )
            self.anchor_embedding.weight.copy_(mean_vector.view(1, -1).expand_as(self.anchor_embedding.weight))
        else:
            raise ValueError(
                "anchor_embedding_init_from must be one of 'random', 'placeholder', 'bos', or 'vocab_mean'.")

        init_scale = getattr(self.config, 'anchor_embedding_init_scale', None)
        if init_scale is not None:
            self.anchor_embedding.weight.mul_(float(init_scale))

    def _patch_apply_rotary_pos_emb(self):
        if hasattr(rope_utils, '_origin_apply_rotary_pos_emb_bshd'):
            return
        _origin_apply_rotary_pos_emb_bshd = rope_utils._apply_rotary_pos_emb_bshd

        def _apply_rotary_pos_emb_bshd(
            t: torch.Tensor,
            freqs: torch.Tensor,
            rotary_interleaved: bool = False,
            multi_latent_attention: bool = False,  # not use
            mscale: float = 1.0,
            **kwargs,
        ) -> torch.Tensor:
            """Apply rotary positional embedding to input tensor T.

            check https://kexue.fm/archives/8265 for detailed formulas

            Args:
                t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
                freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]

            Returns:
                Tensor: The input tensor after applying RoPE
            """
            mscale = self.attention_scaling
            rot_dim = freqs.shape[-1]

            # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
            t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
            if multi_latent_attention:
                x1 = t[..., 0::2]
                x2 = t[..., 1::2]
                t = torch.cat((x1, x2), dim=-1)

            # first part is cosine component
            # second part is sine component, need to change signs with _rotate_half method
            cos_ = (torch.cos(freqs) * mscale).to(t.dtype)
            sin_ = (torch.sin(freqs) * mscale).to(t.dtype)

            t = (t * cos_) + (rope_utils._rotate_half(t, rotary_interleaved) * sin_)
            return torch.cat((t, t_pass), dim=-1)

        rope_utils._apply_rotary_pos_emb_bshd = _apply_rotary_pos_emb_bshd
        rope_utils._origin_apply_rotary_pos_emb_bshd = _origin_apply_rotary_pos_emb_bshd

    @staticmethod
    def _is_hsd_attention_module(attention) -> bool:
        if attention is None:
            return False
        return attention.__class__.__name__ == 'MarchGatedDeltaNet' or all(
            hasattr(attention, name) for name in ('anchor_q_proj', 'anchor_k_proj', 'router_q_proj'))

    @staticmethod
    def _precision_trace_layer_io_requested() -> bool:
        value = os.environ.get('MEGATRON_PRECISION_TRACE_LAYER_IO')
        if value in {None, ''}:
            return False
        return value.strip().lower() in {'1', 'true', 'yes', 'on'}

    def _patch_hsd_layers(self) -> bool:
        decoder = getattr(self, 'decoder', None)
        layers = list(getattr(decoder, 'layers', ()) or ())
        has_hsd_layer = any(self._is_hsd_attention_module(getattr(layer, 'self_attention', None)) for layer in layers)
        trace_layer_io = self._precision_trace_layer_io_requested()
        if trace_layer_io:
            for layer in layers:
                self._patch_hsd_router_trace(layer)
                if not self._is_hsd_attention_module(getattr(layer, 'self_attention', None)):
                    self._patch_hsd_attention_trace(layer)
        if not has_hsd_layer:
            return False

        for layer in layers:
            attention = getattr(layer, 'self_attention', None)
            if self._is_hsd_attention_module(attention):
                pass
                if not getattr(attention, '_mcore_bridge_hsd_patched', False):
                    original_forward = attention.forward

                    def hsd_attention_forward(attn_self, *args, _layer=layer, _origin=original_forward, **kwargs):
                        return self._hsd_attention_forward(_layer, _origin, args, kwargs)

                    attention.forward = MethodType(hsd_attention_forward, attention)
                    attention._mcore_bridge_hsd_patched = True
            elif not getattr(layer, '_mcore_bridge_hsd_patched', False):
                original_forward = layer.forward

                def full_attention_forward(layer_self, *args, _origin=original_forward, **kwargs):
                    return self._full_attention_hsd_forward(layer_self, _origin, args, kwargs)

                layer.forward = MethodType(full_attention_forward, layer)
                layer._mcore_bridge_hsd_patched = True
            if trace_layer_io:
                self._patch_hsd_layer_output_trace(layer)
        return True

    def _init_hsd_anchor_embedding(self) -> None:
        if not getattr(self, '_hsd_enabled', False):
            return
        if not getattr(self, 'pre_process', True):
            return
        if getattr(self.config, 'anchor_embedding_type', 'learned') != 'learned':
            return
        anchor_token_num = int(getattr(self.config, 'anchor_token_num', 0) or 0)
        if anchor_token_num <= 0 or hasattr(self, 'anchor_embedding'):
            return
        dtype = getattr(self.config, 'params_dtype', None)
        self.anchor_embedding = torch.nn.Embedding(anchor_token_num, self.config.hidden_size, dtype=dtype)
        self.anchor_embedding.weight.average_gradients_across_tp_domain = True
        self._hsd_anchor_embedding_created_in_init = True
        self._hsd_anchor_embedding_post_load_initialized = False
        with torch.no_grad():
            self._initialize_hsd_anchor_embedding_weight()

    def maybe_initialize_hsd_anchor_embedding_after_load(self, checkpoint_has_anchor_embedding: bool = False) -> bool:
        if not getattr(self, '_hsd_enabled', False):
            return False
        if not getattr(self, 'pre_process', True):
            return False
        if getattr(self.config, 'anchor_embedding_type', 'learned') != 'learned':
            return False
        if not hasattr(self, 'anchor_embedding'):
            return False
        if checkpoint_has_anchor_embedding:
            return False
        if not getattr(self, '_hsd_anchor_embedding_created_in_init', False):
            return False
        if getattr(self, '_hsd_anchor_embedding_post_load_initialized', False):
            return False
        if getattr(self.config, 'anchor_embedding_init_from', 'random') == 'random':
            return False

        with torch.no_grad():
            self._initialize_hsd_anchor_embedding_weight()
        self._hsd_anchor_embedding_post_load_initialized = True
        return True

    @staticmethod
    def _validate_hsd_anchor_mask(
        anchor_mask: torch.Tensor,
        expected_batch: Optional[int] = None,
        expected_seq: Optional[int] = None,
    ) -> torch.Tensor:
        if anchor_mask.ndim != 2:
            raise ValueError(f'anchor_mask must be [batch, expanded_seq], got {tuple(anchor_mask.shape)}.')
        if expected_batch is not None and anchor_mask.shape[0] != expected_batch:
            raise ValueError(
                f'anchor_mask batch {anchor_mask.shape[0]} must match input batch {expected_batch}.')
        if expected_seq is not None and anchor_mask.shape[1] != expected_seq:
            raise ValueError(
                f'anchor_mask length {anchor_mask.shape[1]} must match expanded sequence length {expected_seq}.')
        anchor_mask = anchor_mask.to(dtype=torch.bool)
        if anchor_mask.shape[0] > 0 and not torch.equal(anchor_mask, anchor_mask[:1].expand_as(anchor_mask)):
            raise NotImplementedError('HSD V1 requires the same anchor layout for every batch row.')
        return anchor_mask

    @staticmethod
    def _text_indices_from_anchor_mask(anchor_mask: torch.Tensor) -> torch.Tensor:
        if anchor_mask.ndim != 2:
            raise ValueError(f'anchor_mask must be [batch, seq], got {tuple(anchor_mask.shape)}.')
        if anchor_mask.shape[0] > 0 and not torch.equal(anchor_mask, anchor_mask[:1].expand_as(anchor_mask)):
            raise NotImplementedError('HSD V1 requires the same anchor layout for every batch row.')
        return torch.nonzero(~anchor_mask[0].to(dtype=torch.bool), as_tuple=False).flatten()

    @staticmethod
    def _format_hsd_cu_seqlens_for_error(name: str, cu_seqlens: Optional[torch.Tensor]) -> str:
        if not isinstance(cu_seqlens, torch.Tensor):
            return f'{name}=None'
        cu_list = cu_seqlens.detach().to(device='cpu', dtype=torch.long).tolist()
        return f'{name}={cu_list}'

    def _validate_hsd_sequence_parallel_layout(
        self,
        anchor_mask: torch.Tensor,
        text_cu_seqlens: Optional[torch.Tensor] = None,
        anchor_cu_seqlens: Optional[torch.Tensor] = None,
    ) -> None:
        anchor_mask = self._validate_hsd_anchor_mask(anchor_mask)
        if not getattr(self.config, 'sequence_parallel', False):
            return
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if tp_size <= 1:
            return
        full_seq_len = anchor_mask.shape[1]
        details = (
            f'expanded_len={full_seq_len}, tp_size={tp_size}, '
            f'{self._format_hsd_cu_seqlens_for_error("text_cu_seqlens", text_cu_seqlens)}, '
            f'{self._format_hsd_cu_seqlens_for_error("anchor_cu_seqlens", anchor_cu_seqlens)}'
        )
        if full_seq_len % tp_size != 0:
            raise ValueError(
                f'HSD expanded sequence length {full_seq_len} must be divisible by tensor parallel size {tp_size}; '
                f'{details}.')

    def _hsd_sequence_parallel_text_counts(self, anchor_mask: torch.Tensor) -> list[int]:
        anchor_mask = self._validate_hsd_anchor_mask(anchor_mask)
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        full_seq_len = int(anchor_mask.shape[1])
        if tp_size <= 1:
            return [int((~anchor_mask[0]).sum().item())]
        if full_seq_len % tp_size != 0:
            raise ValueError(
                f'HSD expanded sequence length {full_seq_len} must be divisible by tensor parallel size {tp_size}.')
        shard_len = full_seq_len // tp_size
        return [
            int((~anchor_mask[0, rank * shard_len:(rank + 1) * shard_len]).sum().item())
            for rank in range(tp_size)
        ]

    def _hsd_anchor_layout_config(self) -> Tuple[int, int]:
        return (
            int(getattr(self.config, 'anchor_chunk_size', 0) or 0),
            int(getattr(self.config, 'anchor_token_num', 0) or 0),
        )

    def _hsd_persistent_memory_enabled(self) -> bool:
        return bool(getattr(self.config, 'hsd_persistent_memory_enabled', False))

    def _auto_hsd_anchor_configured(self) -> bool:
        anchor_chunk_size, anchor_token_num = self._hsd_anchor_layout_config()
        if anchor_token_num <= 0:
            return False
        return anchor_chunk_size > 0 or self._hsd_persistent_memory_enabled()

    def _validate_auto_hsd_text_mask(
        self,
        mask: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        name: str,
    ) -> None:
        if not isinstance(mask, torch.Tensor):
            return
        if mask.shape != input_ids.shape:
            return
        if not bool(mask.to(dtype=torch.bool).all().item()):
            raise NotImplementedError(f'Model-side HSD anchor insertion does not support padded {name} yet.')

    @staticmethod
    def _is_thd_packed(packed_seq_params: Optional[PackedSeqParams]) -> bool:
        return packed_seq_params is not None and getattr(packed_seq_params, 'qkv_format', None) == 'thd'

    @staticmethod
    def _packed_cu_seqlens(packed_seq_params: PackedSeqParams, name: str = 'cu_seqlens_q') -> torch.Tensor:
        cu_seqlens = getattr(packed_seq_params, name, None)
        if not isinstance(cu_seqlens, torch.Tensor) or cu_seqlens.ndim != 1:
            raise ValueError(f'HSD packed sequence requires packed_seq_params.{name} to be a 1D tensor.')
        if cu_seqlens.numel() < 2:
            raise ValueError(f'HSD packed sequence requires packed_seq_params.{name} to contain at least 2 entries.')
        return cu_seqlens

    @staticmethod
    def _set_attr_if_present(obj, name: str, value) -> None:
        if hasattr(obj, name):
            setattr(obj, name, value)

    def _copy_packed_seq_params_with_cu_seqlens(
        self,
        packed_seq_params: PackedSeqParams,
        cu_seqlens: torch.Tensor,
    ) -> PackedSeqParams:
        updated = copy.copy(packed_seq_params)
        self._set_attr_if_present(updated, 'cu_seqlens_q', cu_seqlens)
        self._set_attr_if_present(updated, 'cu_seqlens_kv', cu_seqlens)
        self._set_attr_if_present(updated, 'cu_seqlens_q_padded', None)
        self._set_attr_if_present(updated, 'cu_seqlens_kv_padded', None)

        if cu_seqlens.numel() > 1:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        else:
            max_seqlen = cu_seqlens.new_zeros(())

        for name in ('max_seqlen_q', 'max_seqlen_kv'):
            if not hasattr(updated, name):
                continue
            old_value = getattr(updated, name)
            if isinstance(old_value, torch.Tensor):
                value = max_seqlen.to(device=old_value.device, dtype=old_value.dtype)
            else:
                value = int(max_seqlen.item())
            setattr(updated, name, value)
        if hasattr(updated, 'num_samples'):
            try:
                setattr(updated, 'num_samples', int(cu_seqlens.numel() - 1))
            except Exception:
                pass
        return updated

    def _packed_hsd_doc_lengths(self, packed_seq_params: PackedSeqParams, text_seq_len: int) -> list[int]:
        if not self._is_thd_packed(packed_seq_params):
            raise NotImplementedError('Model-side HSD packed anchor insertion only supports qkv_format="thd".')
        cu_seqlens = self._packed_cu_seqlens(packed_seq_params).detach().to(device='cpu', dtype=torch.long)
        if int(cu_seqlens[0].item()) != 0 or int(cu_seqlens[-1].item()) != int(text_seq_len):
            raise ValueError(
                'HSD packed text cu_seqlens must span the flattened text length: '
                f'cu[0]={int(cu_seqlens[0].item())}, cu[-1]={int(cu_seqlens[-1].item())}, text_len={text_seq_len}.')
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        if any(int(length) <= 0 for length in lengths):
            raise ValueError('HSD packed text cu_seqlens must be strictly increasing.')
        return [int(length) for length in lengths]

    def _validate_required_hsd_packed_layout(
        self,
        input_ids: Optional[torch.Tensor],
        packed_seq_params: Optional[PackedSeqParams],
    ) -> None:
        """Fail closed for document-isolated Stage-3 HSD inputs.

        ``cu_seqlens_q`` is the single runtime authority for Local GDN resets,
        per-document anchor insertion and Dense global-read visibility.  A
        missing or degenerate sidecar would silently restore Stage-1/2
        cross-document semantics, so the Stage-3 model contract rejects it
        before embeddings or kernels are entered.
        """
        if not bool(getattr(self.config, 'hsd_require_packed_seq_params', False)):
            return
        if not bool(getattr(self.config, 'hsd_packed_document_isolation', False)):
            raise RuntimeError(
                'HSD packed_seq_params are required but packed document isolation is disabled.')
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                'Stage-3 document-isolated HSD requires flattened input_ids with shape [1, text_length].')
        if not self._is_thd_packed(packed_seq_params):
            raise ValueError(
                'Stage-3 document-isolated HSD requires packed_seq_params with qkv_format="thd".')

        text_len = int(input_ids.shape[1])
        expected_text_len = getattr(self.config, 'max_text_length', None)
        if expected_text_len is not None and text_len != int(expected_text_len):
            raise ValueError(
                'Stage-3 document-isolated HSD text length mismatch: '
                f'expected={int(expected_text_len)}, actual={text_len}.')
        # This validates 0/end coverage and, critically, strict document
        # boundaries.  The same sidecar is later forwarded unchanged to the
        # convolution, recurrent producer and global-read kernels.
        self._packed_hsd_doc_lengths(packed_seq_params, text_len)

    def _build_auto_hsd_packed_anchor_layout(
        self,
        input_ids: torch.Tensor,
        packed_seq_params: PackedSeqParams,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise NotImplementedError(
                'Model-side HSD packed anchor insertion requires flattened THD input_ids with batch size 1.')
        anchor_chunk_size, anchor_token_num = self._hsd_anchor_layout_config()
        doc_lengths = self._packed_hsd_doc_lengths(packed_seq_params, int(input_ids.shape[1]))

        expanded_offsets = [0]
        anchor_offsets = [0]
        total_expanded = 0
        total_anchor = 0
        doc_chunk_counts = []
        memory_count = 1 if self._hsd_persistent_memory_enabled() else 0
        for doc_len in doc_lengths:
            chunk_count = int(doc_len) // anchor_chunk_size if anchor_chunk_size > 0 else 0
            doc_chunk_counts.append(chunk_count)
            doc_anchor_count = memory_count + chunk_count * anchor_token_num
            total_anchor += doc_anchor_count
            total_expanded += int(doc_len) + doc_anchor_count
            anchor_offsets.append(total_anchor)
            expanded_offsets.append(total_expanded)

        max_position_embeddings = getattr(self.config, 'max_position_embeddings', None)
        if max_position_embeddings is not None and total_expanded > int(max_position_embeddings):
            raise ValueError(
                f'HSD expanded packed sequence length {total_expanded} exceeds max_position_embeddings '
                f'{int(max_position_embeddings)}.')

        anchor_mask = torch.zeros(1, total_expanded, dtype=torch.bool, device=device)
        write = 0
        for doc_len, chunk_count in zip(doc_lengths, doc_chunk_counts):
            if memory_count:
                anchor_mask[:, write:write + memory_count] = True
                write += memory_count
            text_written = 0
            for _ in range(chunk_count):
                write += anchor_chunk_size
                text_written += anchor_chunk_size
                anchor_mask[:, write:write + anchor_token_num] = True
                write += anchor_token_num
            tail = int(doc_len) - text_written
            if tail > 0:
                write += tail
        if write != total_expanded:
            raise RuntimeError(f'HSD packed anchor layout write mismatch: wrote {write}, expected {total_expanded}.')

        text_cu_seqlens = self._packed_cu_seqlens(packed_seq_params)
        anchor_cu_seqlens = text_cu_seqlens.new_tensor(anchor_offsets)
        expanded_cu_seqlens = text_cu_seqlens.new_tensor(expanded_offsets)
        return anchor_mask, anchor_cu_seqlens, expanded_cu_seqlens

    def _should_auto_hsd_anchor(
        self,
        anchor_mask: Optional[torch.Tensor],
        input_ids: Optional[torch.Tensor],
        decoder_input: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor],
        inference_context: Optional[BaseInferenceContext],
        packed_seq_params: Optional[PackedSeqParams],
    ) -> bool:
        self._validate_required_hsd_packed_layout(input_ids, packed_seq_params)
        if anchor_mask is not None or not getattr(self, '_hsd_enabled', False):
            return False
        if not self._auto_hsd_anchor_configured():
            return False
        if inference_context is not None:
            raise NotImplementedError('Model-side HSD anchor insertion is supported for training forward only.')
        if getattr(self, 'mtp_process', False) or getattr(self.config, 'mtp_num_layers', None):
            raise NotImplementedError('Model-side HSD anchor insertion does not support MTP yet.')
        if getattr(self.config, 'anchor_embedding_type', 'learned') == 'placeholder':
            raise NotImplementedError(
                'Model-side HSD anchor insertion requires learned or chunk_end anchor embeddings, '
                'not placeholder embeddings.')
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise NotImplementedError('Model-side HSD anchor insertion requires text-only input_ids [batch, seq].')
        anchor_chunk_size, _ = self._hsd_anchor_layout_config()
        if packed_seq_params is not None:
            doc_lengths = self._packed_hsd_doc_lengths(packed_seq_params, int(input_ids.shape[1]))
            if input_ids.shape[0] != 1:
                raise NotImplementedError(
                    'Model-side HSD packed anchor insertion requires flattened THD input_ids with batch size 1.')
            self._validate_auto_hsd_text_mask(attention_mask, input_ids, 'attention_mask')
            self._validate_auto_hsd_text_mask(padding_mask, input_ids, 'padding_mask')
            explicit_packed_kernel_contract = (
                getattr(self.config, 'hsd_train_scan_backend', 'triton') == 'flashqla_tile'
            )
            if explicit_packed_kernel_contract:
                # A real packed batch can contain only documents shorter than
                # one anchor chunk.  Such a batch has a semantic M=0 layout,
                # but it must still enter MarchGatedDeltaNet so the
                # explicitly configured Local/Global kernels execute their
                # supported M=0 path.  Returning False here used to omit the
                # anchor metadata entirely and silently run the parent GDN;
                # the production dispatch gate then (correctly) detected zero
                # FlashQLA hits on the affected DP rank.
                return len(doc_lengths) > 0
            if int(getattr(self.config, 'hsd_fixed_anchor_count', 0) or 0) > 0:
                # A Stage-3 pack can theoretically contain only documents
                # shorter than one anchor chunk. The physical M=128 contract
                # still applies: enter model-side insertion so the invisible
                # fixed suffix is constructed and the HSD packed path remains
                # active even when the semantic prefix is empty.
                return len(doc_lengths) > 0
            if self._hsd_persistent_memory_enabled():
                return len(doc_lengths) > 0
            return anchor_chunk_size > 0 and any(length // anchor_chunk_size > 0 for length in doc_lengths)
        if (not self._hsd_persistent_memory_enabled()
                and (anchor_chunk_size <= 0 or input_ids.shape[1] // anchor_chunk_size == 0)):
            return False
        self._validate_auto_hsd_text_mask(attention_mask, input_ids, 'attention_mask')
        self._validate_auto_hsd_text_mask(padding_mask, input_ids, 'padding_mask')
        return True

    def _build_auto_hsd_anchor_mask(
        self,
        batch_size: int,
        text_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        anchor_chunk_size, anchor_token_num = self._hsd_anchor_layout_config()
        chunk_count = int(text_seq_len) // anchor_chunk_size if anchor_chunk_size > 0 else 0
        memory_count = 1 if self._hsd_persistent_memory_enabled() else 0
        expanded_seq_len = int(text_seq_len) + memory_count + chunk_count * anchor_token_num
        max_position_embeddings = getattr(self.config, 'max_position_embeddings', None)
        if max_position_embeddings is not None and expanded_seq_len > int(max_position_embeddings):
            raise ValueError(
                f'HSD expanded sequence length {expanded_seq_len} exceeds max_position_embeddings '
                f'{int(max_position_embeddings)}.')
        anchor_mask = torch.zeros(batch_size, expanded_seq_len, dtype=torch.bool, device=device)
        write = 0
        if memory_count:
            anchor_mask[:, :memory_count] = True
            write = memory_count
        for _ in range(chunk_count):
            write += anchor_chunk_size
            anchor_mask[:, write:write + anchor_token_num] = True
            write += anchor_token_num
        return anchor_mask

    def _build_hsd_persistent_memory_mask(
        self,
        anchor_mask: torch.Tensor,
        anchor_cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Mark the one persistent virtual anchor at each document boundary.

        The mask is intentionally constructed after sequence-parallel padding:
        padding anchors remain ordinary (invalid-at-end) anchor slots and
        are never mistaken for persistent memory.
        """
        if not self._hsd_persistent_memory_enabled():
            return None
        anchor_mask = self._validate_hsd_anchor_mask(anchor_mask)
        memory_mask = torch.zeros_like(anchor_mask, dtype=torch.bool)
        anchor_positions = torch.nonzero(anchor_mask[0], as_tuple=False).flatten()
        if anchor_cu_seqlens is None:
            if anchor_positions.numel() == 0:
                raise RuntimeError('Persistent HSD memory requires one anchor slot for the document.')
            memory_mask[:, anchor_positions[0]] = True
            return memory_mask

        anchor_cu = anchor_cu_seqlens.detach().to(device='cpu', dtype=torch.long)
        if anchor_cu.numel() < 2:
            raise ValueError('Persistent packed HSD memory requires document anchor cu-seqlens.')
        starts = anchor_cu[:-1]
        ends = anchor_cu[1:]
        if bool((ends <= starts).any().item()):
            raise RuntimeError('Every packed document must contain its persistent memory anchor.')
        if int(ends[-1].item()) > int(anchor_positions.numel()):
            raise ValueError('Packed anchor cu-seqlens exceed the available anchor slots.')
        memory_positions = anchor_positions.index_select(0, starts.to(device=anchor_positions.device))
        memory_mask[:, memory_positions] = True
        return memory_mask

    def _pad_hsd_anchor_layout_for_sequence_parallel(
        self,
        anchor_mask: torch.Tensor,
        anchor_cu_seqlens: Optional[torch.Tensor] = None,
        expanded_cu_seqlens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not getattr(self.config, 'sequence_parallel', False):
            return anchor_mask, anchor_cu_seqlens, expanded_cu_seqlens
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if tp_size <= 1:
            return anchor_mask, anchor_cu_seqlens, expanded_cu_seqlens

        pad_anchor_count = (-int(anchor_mask.shape[1])) % int(tp_size)
        if pad_anchor_count == 0:
            return anchor_mask, anchor_cu_seqlens, expanded_cu_seqlens

        padded_len = int(anchor_mask.shape[1]) + pad_anchor_count
        max_position_embeddings = getattr(self.config, 'max_position_embeddings', None)
        if max_position_embeddings is not None and padded_len > int(max_position_embeddings):
            raise ValueError(
                f'HSD SP padded expanded sequence length {padded_len} exceeds max_position_embeddings '
                f'{int(max_position_embeddings)}.')

        pad_mask = torch.ones(
            anchor_mask.shape[0],
            pad_anchor_count,
            dtype=torch.bool,
            device=anchor_mask.device,
        )
        anchor_mask = torch.cat([anchor_mask.to(dtype=torch.bool), pad_mask], dim=1)
        if anchor_cu_seqlens is not None:
            anchor_cu_seqlens = anchor_cu_seqlens.clone()
            anchor_cu_seqlens[-1] = anchor_cu_seqlens[-1] + pad_anchor_count
        if expanded_cu_seqlens is not None:
            expanded_cu_seqlens = expanded_cu_seqlens.clone()
            expanded_cu_seqlens[-1] = expanded_cu_seqlens[-1] + pad_anchor_count
        return anchor_mask, anchor_cu_seqlens, expanded_cu_seqlens

    def _pad_hsd_packed_anchor_layout_to_fixed_count(
        self,
        anchor_mask: torch.Tensor,
        anchor_cu_seqlens: torch.Tensor,
        expanded_cu_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Append invisible packed-anchor slots up to the configured kernel M.

        Stage-3 resets the 512-token anchor counter at every document.  The
        semantic anchor count therefore varies with the document inventory,
        even though the flattened text length is always 65,536.  TileLang
        specializes on M, so compiling that dynamic count would create one JIT
        variant for nearly every batch.  ``hsd_fixed_anchor_count`` turns the
        unused suffix into anchor-only layout padding.  Its slots sit after
        all text, hence ``_anchor_text_counts`` assigns boundary=text_len and
        no text token can read them.

        The returned cu-seqlens deliberately attach the suffix to the final
        packed document.  The HSD local producer receives the exact padding
        count separately and strips it before FlashQLA; the Dense reader keeps
        the zero states so its physical M remains fixed.
        """
        target_anchor_count = int(
            getattr(self.config, 'hsd_fixed_anchor_count', 0) or 0)
        if target_anchor_count <= 0:
            return anchor_mask, anchor_cu_seqlens, expanded_cu_seqlens
        if not bool(getattr(self.config, 'hsd_packed_document_isolation', False)):
            raise RuntimeError(
                'hsd_fixed_anchor_count is only valid for packed document-isolated HSD layouts.')
        anchor_mask = self._validate_hsd_anchor_mask(anchor_mask)
        if anchor_mask.shape[0] != 1:
            raise NotImplementedError('Fixed packed HSD anchor padding requires flattened batch size 1.')
        semantic_anchor_count = int(anchor_mask[0].sum().item())
        if semantic_anchor_count > target_anchor_count:
            raise ValueError(
                'HSD semantic packed anchor count exceeds the fixed kernel count: '
                f'semantic={semantic_anchor_count}, fixed={target_anchor_count}.')
        if int(anchor_cu_seqlens[-1].detach().cpu()) != semantic_anchor_count:
            raise ValueError(
                'HSD packed anchor cu-seqlens must span semantic anchors before fixed padding.')
        if int(expanded_cu_seqlens[-1].detach().cpu()) != int(anchor_mask.shape[1]):
            raise ValueError(
                'HSD packed expanded cu-seqlens must span the layout before fixed padding.')

        padding_anchor_count = target_anchor_count - semantic_anchor_count
        padded_expanded_len = int(anchor_mask.shape[1]) + padding_anchor_count
        max_position_embeddings = getattr(self.config, 'max_position_embeddings', None)
        if max_position_embeddings is not None and padded_expanded_len != int(max_position_embeddings):
            raise ValueError(
                'Fixed HSD packed layout must exactly fill max_position_embeddings: '
                f'padded={padded_expanded_len}, expected={int(max_position_embeddings)}.')
        if padding_anchor_count == 0:
            return anchor_mask, anchor_cu_seqlens, expanded_cu_seqlens

        padding_mask = torch.ones(
            1,
            padding_anchor_count,
            dtype=torch.bool,
            device=anchor_mask.device,
        )
        anchor_mask = torch.cat((anchor_mask.to(dtype=torch.bool), padding_mask), dim=1)
        anchor_cu_seqlens = anchor_cu_seqlens.clone()
        anchor_cu_seqlens[-1] += padding_anchor_count
        expanded_cu_seqlens = expanded_cu_seqlens.clone()
        expanded_cu_seqlens[-1] += padding_anchor_count
        return anchor_mask, anchor_cu_seqlens, expanded_cu_seqlens

    def _expand_decoder_input_for_hsd_anchor(
        self,
        decoder_input: torch.Tensor,
        anchor_mask: torch.Tensor,
        text_seq_len: int,
    ) -> torch.Tensor:
        anchor_mask = self._validate_hsd_anchor_mask(
            anchor_mask,
            expected_batch=decoder_input.shape[1],
        ).to(device=decoder_input.device, dtype=torch.bool)
        text_idx = self._text_indices_from_anchor_mask(anchor_mask).to(device=decoder_input.device)
        if text_idx.numel() != text_seq_len:
            raise ValueError(
                f'HSD auto anchor layout text count {text_idx.numel()} must match input text length {text_seq_len}.')
        if decoder_input.shape[0] != text_seq_len:
            raise NotImplementedError(
                'Model-side HSD anchor insertion currently requires a full text sequence on this rank; '
                f'got decoder_input length {decoder_input.shape[0]} for text length {text_seq_len}.')
        expanded = decoder_input.new_zeros(anchor_mask.shape[1], decoder_input.shape[1], decoder_input.shape[2])
        expanded.index_copy_(0, text_idx, decoder_input)
        return expanded

    def _expand_padding_mask_for_hsd_anchor(
        self,
        padding_mask: torch.Tensor,
        anchor_mask: torch.Tensor,
        text_seq_len: int,
    ) -> torch.Tensor:
        anchor_mask = self._validate_hsd_anchor_mask(
            anchor_mask,
            expected_batch=padding_mask.shape[0],
        ).to(device=padding_mask.device, dtype=torch.bool)
        text_idx = self._text_indices_from_anchor_mask(anchor_mask).to(device=padding_mask.device)
        if text_idx.numel() != text_seq_len:
            raise ValueError(
                f'HSD auto anchor layout text count {text_idx.numel()} must match padding text length '
                f'{text_seq_len}.')
        if padding_mask.shape[-1] != text_seq_len:
            raise ValueError(
                f'HSD auto anchor padding_mask length {padding_mask.shape[-1]} must match input text length '
                f'{text_seq_len}.')
        expanded = torch.ones(
            padding_mask.shape[0],
            anchor_mask.shape[1],
            dtype=padding_mask.dtype,
            device=padding_mask.device,
        )
        expanded.index_copy_(1, text_idx, padding_mask)
        return expanded

    def _expand_position_ids_for_hsd_anchor(
        self,
        position_ids: Optional[torch.Tensor],
        anchor_mask: torch.Tensor,
        text_seq_len: int,
        persistent_memory_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if not isinstance(position_ids, torch.Tensor):
            return position_ids
        if position_ids.shape[-1] != text_seq_len:
            raise ValueError(
                f'HSD auto anchor position_ids length {position_ids.shape[-1]} must match input text length '
                f'{text_seq_len}.')

        anchor_mask = self._validate_hsd_anchor_mask(anchor_mask).to(device=position_ids.device, dtype=torch.bool)
        text_count = int((~anchor_mask[0]).sum().item())
        if text_count != text_seq_len:
            raise ValueError(
                f'HSD auto anchor layout text count {text_count} must match input text length {text_seq_len}.')

        text_ordinals = ((~anchor_mask[0]).to(dtype=torch.long).cumsum(dim=0) - 1).clamp_min(0)
        expanded = position_ids.index_select(-1, text_ordinals)
        if persistent_memory_mask is not None:
            persistent_memory_mask = self._validate_hsd_anchor_mask(
                persistent_memory_mask,
                expected_batch=anchor_mask.shape[0],
                expected_seq=anchor_mask.shape[1],
            ).to(device=expanded.device, dtype=torch.bool)
            if bool((persistent_memory_mask & ~anchor_mask).any().item()):
                raise ValueError('persistent_memory_mask must be a subset of anchor_mask.')
            expanded = expanded.masked_fill(
                persistent_memory_mask.unsqueeze(0) if expanded.ndim == 3 else persistent_memory_mask,
                0,
            )
        return expanded

    def _local_anchor_mask(
        self,
        anchor_mask: torch.Tensor,
        local_seq_len: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, int]:
        anchor_mask = self._validate_hsd_anchor_mask(anchor_mask)
        full_seq_len = anchor_mask.shape[1]
        if local_seq_len == full_seq_len:
            return anchor_mask.to(device=device, dtype=torch.bool), 0
        cp_size = getattr(self.config, 'context_parallel_size', 1)
        if cp_size > 1:
            mask = anchor_mask.to(device=device, dtype=torch.float32).transpose(0, 1).contiguous().unsqueeze(-1)
            mask = tensor_a2a_hp2cp(
                mask,
                seq_dim=0,
                head_dim=-1,
                cp_group=parallel_state.get_context_parallel_group(),
            )
            cp_anchor_mask = mask.any(dim=-1).transpose(0, 1).contiguous().to(dtype=torch.bool)
            cp_seq_len = cp_anchor_mask.shape[1]
            if local_seq_len == cp_seq_len:
                return cp_anchor_mask, 0
            if not getattr(self.config, 'sequence_parallel', False):
                raise ValueError(
                    f'HSD local sequence length {local_seq_len} does not match CP-local anchor mask length '
                    f'{cp_seq_len}.')
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            if cp_seq_len % tp_size != 0:
                raise ValueError(
                    f'HSD CP-local sequence length {cp_seq_len} must be divisible by tensor parallel size {tp_size}.')
            shard_len = cp_seq_len // tp_size
            if local_seq_len != shard_len:
                raise ValueError(
                    f'HSD local sequence length {local_seq_len} must match CP/SP shard length {shard_len}.')
            start = tp_rank * shard_len
            return cp_anchor_mask[:, start:start + shard_len], start
        if not getattr(self.config, 'sequence_parallel', False):
            raise ValueError(
                f'HSD local sequence length {local_seq_len} does not match anchor_mask length {full_seq_len}.')
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        if full_seq_len % tp_size != 0:
            raise ValueError(
                f'HSD expanded sequence length {full_seq_len} must be divisible by tensor parallel size {tp_size}.')
        shard_len = full_seq_len // tp_size
        if local_seq_len != shard_len:
            raise ValueError(
                f'HSD local sequence length {local_seq_len} must match SP shard length {shard_len}.')
        start = tp_rank * shard_len
        return anchor_mask[:, start:start + shard_len].to(device=device, dtype=torch.bool), start

    @classmethod
    def _filter_attention_tensor_to_text(
        cls,
        tensor: Optional[torch.Tensor],
        anchor_mask: torch.Tensor,
    ):
        if not isinstance(tensor, torch.Tensor):
            return tensor
        anchor_mask = cls._validate_hsd_anchor_mask(anchor_mask).to(device=tensor.device)
        text_idx = cls._text_indices_from_anchor_mask(anchor_mask).to(device=tensor.device)
        batch, seq_len = anchor_mask.shape
        if tensor.ndim >= 2 and tuple(tensor.shape[-2:]) == (seq_len, seq_len):
            return tensor.index_select(-2, text_idx).index_select(-1, text_idx)
        if tensor.ndim == 2 and tuple(tensor.shape) == (batch, seq_len):
            return tensor.index_select(1, text_idx)
        if tensor.ndim >= 1 and tensor.shape[-1] == seq_len:
            return tensor.index_select(-1, text_idx)
        if tensor.ndim >= 1 and tensor.shape[0] == seq_len:
            return tensor.index_select(0, text_idx)
        return tensor

    @classmethod
    def _filter_attention_tensor_to_text_any(
        cls,
        tensor: Optional[torch.Tensor],
        anchor_mask: torch.Tensor,
        local_anchor_mask: Optional[torch.Tensor] = None,
    ):
        filtered = cls._filter_attention_tensor_to_text(tensor, anchor_mask)
        if filtered is tensor and local_anchor_mask is not None and local_anchor_mask.shape != anchor_mask.shape:
            filtered = cls._filter_attention_tensor_to_text(tensor, local_anchor_mask)
        return filtered

    @classmethod
    def _filter_rotary_to_text(
        cls,
        rotary,
        anchor_mask: torch.Tensor,
        expanded_anchor_mask: Optional[torch.Tensor] = None,
    ):
        if rotary is None:
            return None
        if isinstance(rotary, tuple):
            return tuple(cls._filter_rotary_to_text(item, anchor_mask, expanded_anchor_mask) for item in rotary)
        if isinstance(rotary, list):
            return [cls._filter_rotary_to_text(item, anchor_mask, expanded_anchor_mask) for item in rotary]
        if not isinstance(rotary, torch.Tensor):
            return rotary
        for mask in (anchor_mask, expanded_anchor_mask):
            if mask is None:
                continue
            mask = cls._validate_hsd_anchor_mask(mask).to(device=rotary.device)
            if rotary.ndim >= 1 and rotary.shape[0] == mask.shape[1]:
                text_idx = cls._text_indices_from_anchor_mask(mask).to(device=rotary.device)
                return rotary.index_select(0, text_idx)
        return rotary

    @staticmethod
    def _anchor_positions(anchor_mask: torch.Tensor) -> torch.Tensor:
        anchor_mask = GPTModel._validate_hsd_anchor_mask(anchor_mask).to(dtype=torch.bool)
        batch_size, seq_len = anchor_mask.shape
        positions = torch.arange(seq_len, device=anchor_mask.device, dtype=torch.long).expand(batch_size, seq_len)
        text_positions = torch.where(anchor_mask, positions.new_full((), -1), positions)
        anchor_positions = text_positions.cummax(dim=1).values
        if bool((anchor_positions[anchor_mask] < 0).any().item()):
            raise ValueError('anchor tokens must have a preceding text token for chunk_end embeddings.')
        return anchor_positions

    @staticmethod
    def _hsd_env_int(name: str, default: int) -> int:
        value = os.environ.get(name)
        if value in {None, ''}:
            return int(default)
        return int(value)

    @staticmethod
    def _hsd_env_bool(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value in {None, ''}:
            return bool(default)
        return value.strip().lower() in {'1', 'true', 'yes', 'on'}

    @staticmethod
    def _hsd_env_optional_float(name: str, default: Optional[float]) -> Optional[float]:
        value = os.environ.get(name)
        if value in {None, ''}:
            return default
        return float(value)

    def _anchor_embedding_grad_scale_tensor(self, device: torch.device) -> Optional[torch.Tensor]:
        fixed_scale = self._hsd_env_optional_float(
            'HSD_ANCHOR_EMBEDDING_GRAD_SCALE',
            getattr(self.config, 'hsd_anchor_embedding_grad_scale', None),
        )
        start = self._hsd_env_optional_float(
            'HSD_ANCHOR_EMBEDDING_GRAD_SCALE_START',
            getattr(self.config, 'hsd_anchor_embedding_grad_scale_start', None),
        )
        end = self._hsd_env_optional_float(
            'HSD_ANCHOR_EMBEDDING_GRAD_SCALE_END',
            getattr(self.config, 'hsd_anchor_embedding_grad_scale_end', None),
        )
        steps = self._hsd_env_int(
            'HSD_ANCHOR_EMBEDDING_GRAD_SCALE_STEPS',
            int(getattr(self.config, 'hsd_anchor_embedding_grad_scale_steps', 0) or 0),
        )

        if steps > 0:
            if start is None or end is None:
                raise RuntimeError(
                    'HSD anchor embedding grad scale schedule requires start/end when steps > 0.')
            if steps <= 1:
                scale = float(end)
            else:
                step = max(0, self._hsd_env_int('HSD_RUNTIME_STEP', 0))
                progress = min(1.0, float(step) / float(steps - 1))
                scale = float(start) + (float(end) - float(start)) * progress
        elif fixed_scale is not None:
            scale = float(fixed_scale)
        else:
            return None

        if scale < 0:
            raise RuntimeError('HSD anchor embedding grad scale must be non-negative.')
        return torch.tensor(scale, dtype=torch.float32, device=device)

    def _apply_anchor_embedding(
        self,
        decoder_input: Optional[torch.Tensor],
        anchor_mask: Optional[torch.Tensor],
        has_anchor_tokens: bool,
    ) -> Optional[torch.Tensor]:
        if decoder_input is None or anchor_mask is None or not has_anchor_tokens:
            return decoder_input
        local_anchor_mask, local_start = self._local_anchor_mask(
            anchor_mask, decoder_input.shape[0], device=decoder_input.device)
        if not bool(local_anchor_mask.any().item()):
            return decoder_input

        anchor_embedding_type = getattr(self.config, 'anchor_embedding_type', 'learned')
        if anchor_embedding_type == 'placeholder':
            return decoder_input

        output = decoder_input.clone()
        output_bsh = output.transpose(0, 1)
        if anchor_embedding_type == 'learned':
            if not hasattr(self, 'anchor_embedding'):
                raise ValueError('HSD learned anchor embeddings require anchor_token_num > 0.')
            anchor_token_num = int(getattr(self.config, 'anchor_token_num', 0) or 0)
            full_mask = anchor_mask.to(device=decoder_input.device, dtype=torch.bool)
            anchor_ordinals = full_mask.long().cumsum(dim=1).sub(1)
            local_ordinals = anchor_ordinals[:, local_start:local_start + decoder_input.shape[0]]
            anchor_ids = local_ordinals[local_anchor_mask].remainder(anchor_token_num)
            replacements = self.anchor_embedding(anchor_ids.to(device=self.anchor_embedding.weight.device))
            grad_scale = self._anchor_embedding_grad_scale_tensor(replacements.device)
            if grad_scale is not None:
                grad_scale = grad_scale.to(dtype=replacements.dtype)
                replacements = replacements.detach() + grad_scale * (replacements - replacements.detach())
            output_bsh[local_anchor_mask] = replacements.to(device=decoder_input.device, dtype=decoder_input.dtype)
            return output
        if anchor_embedding_type == 'chunk_end':
            full_mask = anchor_mask.to(device=decoder_input.device, dtype=torch.bool)
            anchors = self._anchor_positions(full_mask)[:, local_start:local_start + decoder_input.shape[0]]
            local_anchors = anchors[local_anchor_mask] - local_start
            if bool(((local_anchors < 0) | (local_anchors >= decoder_input.shape[0])).any().item()):
                raise NotImplementedError(
                    'HSD chunk_end anchor embeddings require their source tokens to live in the same SP shard.')
            batch_positions = torch.arange(
                local_anchor_mask.shape[0], device=decoder_input.device, dtype=torch.long).unsqueeze(1)
            batch_positions = batch_positions.expand_as(local_anchor_mask)[local_anchor_mask]
            output_bsh[local_anchor_mask] = output_bsh[batch_positions, local_anchors]
            return output
        raise ValueError(f'Unsupported anchor_embedding_type: {anchor_embedding_type}.')

    def _filter_anchor_hidden_states(
        self,
        hidden_states: torch.Tensor,
        anchor_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if anchor_mask is None or not bool(anchor_mask.any().item()):
            return hidden_states
        anchor_mask = self._validate_hsd_anchor_mask(anchor_mask)
        text_len = int((~anchor_mask[0]).sum().item())
        if labels is not None and labels.ndim >= 2 and labels.shape[-1] != text_len:
            raise ValueError(
                f'HSD labels length {labels.shape[-1]} must match text token length {text_len}.')
        local_anchor_mask, _ = self._local_anchor_mask(anchor_mask, hidden_states.shape[0],
                                                         device=hidden_states.device)
        text_idx = self._text_indices_from_anchor_mask(local_anchor_mask).to(device=hidden_states.device)
        return hidden_states.index_select(0, text_idx)

    def _prepare_hsd_sp_padded_text_loss_inputs(
        self,
        hidden_states: torch.Tensor,
        anchor_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_mask = self._validate_hsd_anchor_mask(anchor_mask)
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        full_seq_len = int(anchor_mask.shape[1])
        if tp_size <= 1 or not getattr(self.config, 'sequence_parallel', False):
            raise RuntimeError('HSD padded SP loss inputs require sequence_parallel with tensor parallel size > 1.')
        if full_seq_len % tp_size != 0:
            raise ValueError(
                f'HSD expanded sequence length {full_seq_len} must be divisible by tensor parallel size {tp_size}.')
        shard_len = full_seq_len // tp_size
        if hidden_states.shape[0] != shard_len:
            raise RuntimeError(
                'HSD SP padded loss expects the local expanded sequence shard before logits: '
                f'got local_seq_len={hidden_states.shape[0]}, shard_len={shard_len}, expanded_len={full_seq_len}.')

        text_counts = self._hsd_sequence_parallel_text_counts(anchor_mask)
        text_len = sum(text_counts)
        if labels.ndim < 2 or labels.shape[-1] != text_len:
            raise ValueError(
                f'HSD labels length {labels.shape[-1] if labels.ndim >= 2 else tuple(labels.shape)} must match '
                f'text token length {text_len}; text_counts={text_counts}.')
        max_text_count = max(text_counts) if text_counts else 0
        if max_text_count <= 0:
            raise ValueError('HSD SP padded loss requires at least one text token across tensor-parallel ranks.')

        local_anchor_mask, _ = self._local_anchor_mask(anchor_mask, shard_len, device=hidden_states.device)
        local_text_idx = self._text_indices_from_anchor_mask(local_anchor_mask).to(device=hidden_states.device)
        local_text_hidden = hidden_states.index_select(0, local_text_idx)
        expected_local_text = text_counts[tp_rank]
        if local_text_hidden.shape[0] != expected_local_text:
            raise RuntimeError(
                'HSD local text count mismatch while preparing SP padded loss inputs: '
                f'got {local_text_hidden.shape[0]}, expected {expected_local_text}, text_counts={text_counts}.')

        padded_hidden = hidden_states.new_zeros(max_text_count, hidden_states.shape[1], hidden_states.shape[2])
        if expected_local_text > 0:
            padded_hidden[:expected_local_text].copy_(local_text_hidden)

        padded_labels = labels.new_full((labels.shape[0], max_text_count * tp_size), -100)
        keep_positions = []
        text_offset = 0
        for rank, count in enumerate(text_counts):
            if count > 0:
                dst_start = rank * max_text_count
                dst_end = dst_start + count
                src_end = text_offset + count
                padded_labels[:, dst_start:dst_end] = labels[:, text_offset:src_end]
                keep_positions.extend(range(dst_start, dst_end))
                text_offset = src_end
        if text_offset != text_len:
            raise RuntimeError(
                f'HSD SP padded label write mismatch: wrote {text_offset} text labels, expected {text_len}.')
        keep_idx = torch.tensor(keep_positions, dtype=torch.long, device=hidden_states.device)
        return padded_hidden, padded_labels, keep_idx

    def _set_hsd_layer_context(
        self,
        anchor_mask: Optional[torch.Tensor],
        has_anchor_tokens: bool,
        text_cu_seqlens: Optional[torch.Tensor] = None,
        anchor_cu_seqlens: Optional[torch.Tensor] = None,
        text_packed_seq_params: Optional[PackedSeqParams] = None,
        persistent_memory_mask: Optional[torch.Tensor] = None,
        sp_padding_anchor_count: int = 0,
    ) -> None:
        for layer in getattr(getattr(self, 'decoder', None), 'layers', ()) or ():
            layer._hsd_anchor_mask = anchor_mask
            layer._hsd_has_anchor_tokens = bool(has_anchor_tokens)
            layer._hsd_text_cu_seqlens = text_cu_seqlens
            layer._hsd_anchor_cu_seqlens = anchor_cu_seqlens
            layer._hsd_text_packed_seq_params = text_packed_seq_params
            layer._hsd_persistent_memory_mask = persistent_memory_mask
            layer._hsd_sp_padding_anchor_count = int(sp_padding_anchor_count)

    def _should_clear_hsd_layer_context_after_forward(self) -> bool:
        return not (self.training and torch.is_grad_enabled())

    def _trace_hsd_layer_tensor(self, layer, name: str, tensor: torch.Tensor, category: str = 'layer_io') -> None:
        if not isinstance(tensor, torch.Tensor) or not self._precision_trace_layer_io_requested():
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
        layer_number = int(getattr(layer, 'layer_number', 0) or 0)
        tracer.trace_layer_tensor(
            f'layer_{layer_number}/{name}',
            tensor,
            step=precision_trace_runtime_step(),
            category=category,
            layer_number=layer_number,
        )

    def _trace_hsd_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        category: str = 'embedding',
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        if not isinstance(tensor, torch.Tensor) or not self._precision_trace_layer_io_requested():
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
        tracer.trace_tensor(
            name,
            tensor,
            step=precision_trace_runtime_step(),
            category=category,
            extra=extra,
        )

    def _trace_hsd_layer_text_tensor(
        self,
        layer,
        name: str,
        tensor: torch.Tensor,
        anchor_mask: Optional[torch.Tensor],
    ) -> None:
        if not isinstance(tensor, torch.Tensor) or anchor_mask is None or not self._precision_trace_layer_io_requested():
            return
        anchor_mask = self._validate_hsd_anchor_mask(anchor_mask)
        tensor_seq_len = int(tensor.shape[0])
        if tensor_seq_len == int((~anchor_mask[0]).sum().item()):
            self._trace_hsd_layer_tensor(layer, name, tensor)
            return
        if getattr(self.config, 'sequence_parallel', False):
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            full_seq_len = int(anchor_mask.shape[1])
            if tp_size > 1 and full_seq_len % tp_size == 0:
                shard_len = full_seq_len // tp_size
                start = tp_rank * shard_len
                local_anchor_mask = anchor_mask[:, start:start + shard_len].to(device=tensor.device, dtype=torch.bool)
                if tensor_seq_len == int((~local_anchor_mask[0]).sum().item()):
                    self._trace_hsd_layer_tensor(layer, name, tensor)
                    return
        local_anchor_mask, _ = self._local_anchor_mask(anchor_mask, tensor_seq_len, device=tensor.device)
        text_idx = self._text_indices_from_anchor_mask(local_anchor_mask).to(device=tensor.device)
        self._trace_hsd_layer_tensor(layer, name, tensor.index_select(0, text_idx))

    def _patch_hsd_attention_trace(self, layer) -> None:
        attention = getattr(layer, 'self_attention', None)
        if attention is None or getattr(attention, '_mcore_bridge_precision_trace_attention_patched', False):
            return
        original_forward = attention.forward

        def attention_forward(attn_self, *args, _layer=layer, _origin=original_forward, **kwargs):
            output = _origin(*args, **kwargs)
            output_tensor = output[0] if isinstance(output, tuple) and output else output
            if isinstance(output_tensor, torch.Tensor):
                self._trace_hsd_layer_tensor(_layer, 'attention_out_text', output_tensor)
            return output

        attention.forward = MethodType(attention_forward, attention)
        attention._mcore_bridge_precision_trace_attention_patched = True

    def _patch_hsd_layer_output_trace(self, layer) -> None:
        if getattr(layer, '_mcore_bridge_precision_trace_layer_output_patched', False):
            return
        original_forward = layer.forward

        def layer_output_forward(layer_self, *args, _layer=layer, _origin=original_forward, **kwargs):
            output = _origin(*args, **kwargs)
            output_tensor = output[0] if isinstance(output, tuple) and output else output
            anchor_mask = getattr(_layer, '_hsd_anchor_mask', None)
            if isinstance(output_tensor, torch.Tensor):
                if anchor_mask is not None:
                    self._trace_hsd_layer_text_tensor(_layer, 'layer_output_text', output_tensor, anchor_mask)
                else:
                    self._trace_hsd_layer_tensor(_layer, 'layer_output', output_tensor)
            return output

        layer.forward = MethodType(layer_output_forward, layer)
        layer._mcore_bridge_precision_trace_layer_output_patched = True

    def _patch_hsd_router_trace(self, layer) -> None:
        mlp = getattr(layer, 'mlp', None)
        router = getattr(mlp, 'router', None)
        if router is None or getattr(router, '_mcore_bridge_precision_trace_patched', False):
            return
        original_forward = router.forward

        def router_forward(router_self, *args, _layer=layer, _origin=original_forward, **kwargs):
            output = _origin(*args, **kwargs)
            routing_scores = None
            if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
                routing_scores = output[0]
            if isinstance(routing_scores, torch.Tensor) and routing_scores.ndim >= 2 and routing_scores.shape[-1] >= 2:
                scores = routing_scores.detach().float().reshape(-1, routing_scores.shape[-1])
                router_topk = max(1, int(getattr(router_self, 'topk', getattr(self.config, 'moe_router_topk', 1)) or 1))
                trace_router_full = self._hsd_env_bool('MEGATRON_PRECISION_TRACE_ROUTER_FULL', False)
                if trace_router_full:
                    anchor_mask = getattr(_layer, '_hsd_anchor_mask', None)
                    if args and isinstance(args[0], torch.Tensor):
                        if anchor_mask is not None:
                            self._trace_hsd_layer_text_tensor(
                                _layer, 'router_input_text', args[0].detach(), anchor_mask)
                        else:
                            self._trace_hsd_layer_tensor(
                                _layer, 'router_input', args[0].detach(), category='router_full')
                    if anchor_mask is not None:
                        self._trace_hsd_layer_text_tensor(_layer, 'router_sparse_probs_text',
                                                          routing_scores.detach(), anchor_mask)
                    else:
                        self._trace_hsd_layer_tensor(
                            _layer, 'router_sparse_probs', routing_scores.detach(), category='router_full')
                    if args and isinstance(args[0], torch.Tensor):
                        try:
                            with torch.no_grad():
                                router_gate_logits = router_self.gating(args[0].detach())
                        except Exception:
                            router_gate_logits = None
                        if isinstance(router_gate_logits, torch.Tensor):
                            if anchor_mask is not None:
                                self._trace_hsd_layer_text_tensor(_layer, 'router_gate_logits_text',
                                                                  router_gate_logits, anchor_mask)
                            else:
                                self._trace_hsd_layer_tensor(
                                    _layer, 'router_gate_logits', router_gate_logits, category='router_full')
                top_values = scores.topk(k=min(max(router_topk + 1, 2), scores.shape[-1]), dim=-1).values
                top1_margin = top_values[:, 0] - top_values[:, 1]
                self._trace_hsd_layer_tensor(_layer, 'top1_margin', top1_margin, category='router_margin')
                thresholds = top1_margin.new_tensor([1e-4, 1e-3, 1e-2, 1e-1])
                top1_near_counts = (top1_margin.unsqueeze(-1) <= thresholds).sum(dim=0).to(dtype=torch.float32)
                self._trace_hsd_layer_tensor(
                    _layer,
                    'top1_margin_le_counts_1e-4_1e-3_1e-2_1e-1',
                    top1_near_counts,
                    category='router_margin',
                )
                if top_values.shape[-1] > router_topk:
                    topk_gap = top_values[:, router_topk - 1] - top_values[:, router_topk]
                    self._trace_hsd_layer_tensor(_layer, 'topk_gap', topk_gap, category='router_margin')
                    topk_near_counts = (topk_gap.unsqueeze(-1) <= thresholds).sum(dim=0).to(dtype=torch.float32)
                    self._trace_hsd_layer_tensor(
                        _layer,
                        'topk_gap_le_counts_1e-4_1e-3_1e-2_1e-1',
                        topk_near_counts,
                        category='router_margin',
                    )
                if self._hsd_env_bool('MEGATRON_PRECISION_TRACE_ROUTER_TOPK', False):
                    router_top = scores.topk(k=min(router_topk, scores.shape[-1]), dim=-1)
                    topk_shape = tuple(routing_scores.shape[:-1]) + (router_top.indices.shape[-1],)
                    router_topk_indices = router_top.indices.to(dtype=torch.int32).reshape(topk_shape)
                    router_topk_scores = router_top.values.reshape(topk_shape)
                    anchor_mask = getattr(_layer, '_hsd_anchor_mask', None)
                    if anchor_mask is not None:
                        self._trace_hsd_layer_text_tensor(
                            _layer, 'router_topk_indices_text', router_topk_indices, anchor_mask)
                        self._trace_hsd_layer_text_tensor(
                            _layer, 'router_topk_scores_text', router_topk_scores, anchor_mask)
                    else:
                        self._trace_hsd_layer_tensor(
                            _layer, 'router_topk_indices', router_topk_indices, category='router_topk')
                        self._trace_hsd_layer_tensor(
                            _layer, 'router_topk_scores', router_topk_scores, category='router_topk')
            routing_map = None
            if isinstance(output, tuple) and len(output) >= 2 and isinstance(output[1], torch.Tensor):
                routing_map = output[1]
            if isinstance(routing_map, torch.Tensor):
                counts = routing_map.detach().to(dtype=torch.float32)
                if counts.ndim >= 2:
                    counts = counts.reshape(-1, counts.shape[-1]).sum(dim=0)
                self._trace_hsd_layer_tensor(_layer, 'expert_counts', counts, category='router_counts')
            return output

        router.forward = MethodType(router_forward, router)
        router._mcore_bridge_precision_trace_patched = True






    def _hsd_attention_forward(self, layer, original_forward, args, kwargs):
        kwargs = dict(kwargs)
        anchor_mask = getattr(layer, '_hsd_anchor_mask', None)
        has_anchor_tokens = bool(getattr(layer, '_hsd_has_anchor_tokens', False))
        if anchor_mask is not None:
            kwargs.setdefault('anchor_mask', anchor_mask)
            kwargs.setdefault('has_anchor_tokens', has_anchor_tokens)
            text_cu_seqlens = getattr(layer, '_hsd_text_cu_seqlens', None)
            anchor_cu_seqlens = getattr(layer, '_hsd_anchor_cu_seqlens', None)
            if text_cu_seqlens is not None:
                kwargs.setdefault('text_cu_seqlens', text_cu_seqlens)
            if anchor_cu_seqlens is not None:
                kwargs.setdefault('anchor_cu_seqlens', anchor_cu_seqlens)
            persistent_memory_mask = getattr(layer, '_hsd_persistent_memory_mask', None)
            if persistent_memory_mask is not None:
                kwargs.setdefault('persistent_memory_mask', persistent_memory_mask)
            kwargs.setdefault(
                'sp_padding_anchor_count',
                int(getattr(layer, '_hsd_sp_padding_anchor_count', 0)),
            )
        output = original_forward(*args, **kwargs)
        if anchor_mask is not None and has_anchor_tokens:
            output_tensor = output[0] if isinstance(output, tuple) and output else output
            self._trace_hsd_layer_text_tensor(layer, 'attention_out_text', output_tensor, anchor_mask)
        return output

    @staticmethod
    def _layer_forward_kwargs(args, kwargs) -> dict:
        kwargs = dict(kwargs)
        names = (
            'hidden_states',
            'attention_mask',
            'context',
            'context_mask',
            'rotary_pos_emb',
            'rotary_pos_cos',
            'rotary_pos_sin',
            'rotary_pos_cos_sin',
            'attention_bias',
            'inference_context',
            'packed_seq_params',
            'sequence_len_offset',
            'padding_mask',
        )
        if len(args) > len(names):
            raise TypeError(f'Too many positional arguments for HSD layer wrapper: {len(args)}.')
        for name, value in zip(names, args):
            if name in kwargs:
                raise TypeError(f'Got multiple values for argument {name}.')
            kwargs[name] = value
        return kwargs

    def _full_attention_hsd_forward(self, layer, original_forward, args, kwargs):
        pass
        return self._full_attention_text_only_forward(layer, original_forward, args, kwargs)

    def _full_attention_text_only_forward(self, layer, original_forward, args, kwargs):
        kwargs = self._layer_forward_kwargs(args, kwargs)
        anchor_mask = kwargs.pop('anchor_mask', getattr(layer, '_hsd_anchor_mask', None))
        has_anchor_tokens = kwargs.pop('has_anchor_tokens', getattr(layer, '_hsd_has_anchor_tokens', False))
        if anchor_mask is None or not has_anchor_tokens:
            return original_forward(**kwargs)

        hidden_states = kwargs.get('hidden_states')
        if not isinstance(hidden_states, torch.Tensor):
            return original_forward(**kwargs)
        anchor_mask = self._validate_hsd_anchor_mask(anchor_mask)
        sp_gathered_hidden = False
        full_hidden_states = hidden_states
        if (
            getattr(self.config, 'sequence_parallel', False)
            and parallel_state.get_tensor_model_parallel_world_size() > 1
            and hidden_states.shape[0] != anchor_mask.shape[1]
        ):
            full_hidden_states = gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=False,
                group=self.pg_collection.tp,
            )
            if full_hidden_states.shape[0] != anchor_mask.shape[1]:
                raise RuntimeError(
                    'HSD full-attention SP gather did not recover the full expanded sequence: '
                    f'got {full_hidden_states.shape[0]} for expanded length {anchor_mask.shape[1]}.')
            sp_gathered_hidden = True

        local_anchor_mask, _ = self._local_anchor_mask(anchor_mask, full_hidden_states.shape[0],
                                                         device=full_hidden_states.device)
        text_idx = self._text_indices_from_anchor_mask(local_anchor_mask).to(device=full_hidden_states.device)
        if text_idx.numel() == hidden_states.shape[0]:
            return original_forward(**kwargs)
        if text_idx.numel() == 0:
            raise ValueError('HSD full-attention bypass requires at least one text token in each local shard.')

        filtered_kwargs = dict(kwargs)
        filtered_kwargs['hidden_states'] = full_hidden_states.index_select(0, text_idx)
        text_packed_seq_params = getattr(layer, '_hsd_text_packed_seq_params', None)
        if text_packed_seq_params is not None:
            filtered_kwargs['packed_seq_params'] = text_packed_seq_params
        for name in ('attention_mask', 'attention_bias', 'padding_mask'):
            filtered_kwargs[name] = self._filter_attention_tensor_to_text_any(
                filtered_kwargs.get(name), anchor_mask, local_anchor_mask)
        for name in ('rotary_pos_emb', 'rotary_pos_cos', 'rotary_pos_sin', 'rotary_pos_cos_sin'):
            filtered_kwargs[name] = self._filter_rotary_to_text(
                filtered_kwargs.get(name), anchor_mask, local_anchor_mask)

        sequence_parallel_context = (
            _child_modules_sequence_parallel(layer, False)
            if sp_gathered_hidden else nullcontext()
        )
        with sequence_parallel_context:
            output = original_forward(**filtered_kwargs)
        if isinstance(output, tuple):
            text_output = output[0]
            rest = output[1:]
        else:
            text_output = output
            rest = None
        expanded_output = full_hidden_states.clone()
        expanded_output.index_copy_(0, text_idx, text_output)
        if sp_gathered_hidden:
            expanded_output = tensor_parallel.scatter_to_sequence_parallel_region(
                expanded_output,
                group=self.pg_collection.tp,
            )
        if rest is None:
            return expanded_output
        return (expanded_output, ) + rest

    def _prepare_hsd_forward(
        self,
        anchor_mask: Optional[torch.Tensor],
        input_ids: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        inference_context: Optional[BaseInferenceContext],
        packed_seq_params: Optional[PackedSeqParams],
    ) -> Tuple[Optional[torch.Tensor], bool]:
        if not getattr(self, '_hsd_enabled', False) or anchor_mask is None:
            return anchor_mask, False
        expected_batch = input_ids.shape[0] if isinstance(input_ids, torch.Tensor) else None
        expected_seq = input_ids.shape[1] if isinstance(input_ids, torch.Tensor) else None
        anchor_mask = self._validate_hsd_anchor_mask(anchor_mask, expected_batch, expected_seq)
        has_anchor_tokens = bool(anchor_mask.any().item())
        if not has_anchor_tokens:
            return anchor_mask, False
        if inference_context is not None:
            raise NotImplementedError('HSD anchor tokens are supported for training forward only.')
        if packed_seq_params is not None:
            raise NotImplementedError(
                'Explicit HSD anchor_mask with packed sequences is not supported yet; '
                'use model-side auto HSD anchor insertion.')
        if getattr(self, 'mtp_process', False) or getattr(self.config, 'mtp_num_layers', None):
            raise NotImplementedError('HSD anchor tokens do not support MTP yet.')
        self._validate_hsd_sequence_parallel_layout(anchor_mask)
        if labels is not None and labels.ndim >= 2:
            text_len = int((~anchor_mask[0]).sum().item())
            if labels.shape[-1] != text_len:
                raise ValueError(
                    f'HSD labels length {labels.shape[-1]} must match text token length {text_len}.')
        return anchor_mask, True

    def _preprocess(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        decoder_input: torch.Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        padding_mask: Optional[torch.Tensor] = None,
        anchor_mask: Optional[torch.Tensor] = None,
        has_anchor_tokens: bool = False,
        auto_hsd_anchor: bool = False,
    ):
        """Preprocesses inputs for the transformer decoder.

        Applies embeddings to input tokens, or uses `decoder_input` from a previous
        pipeline stage. Also sets up rotary positional embeddings.
        """
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.
        in_inference_mode = inference_context is not None and not self.training
        hsd_text_cu_seqlens = None
        hsd_anchor_cu_seqlens = None
        hsd_text_packed_seq_params = None
        hsd_persistent_memory_mask = None
        hsd_sp_padding_anchor_count = 0
        padding_mask_needs_sp_scatter = False
        padding_mask_sp_scattered = False

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            if padding_mask is not None:
                assert padding_mask.shape == input_ids.shape, (
                    f'padding_mask shape {padding_mask.shape} does not match input_ids shape {input_ids.shape}')
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
            if padding_mask is not None and self.config.sequence_parallel:
                if auto_hsd_anchor:
                    padding_mask_needs_sp_scatter = True
                else:
                    padding_mask = (
                        tensor_parallel.scatter_to_sequence_parallel_region(
                            padding_mask.transpose(0, 1).contiguous(),
                            group=self.pg_collection.tp,
                        ).transpose(0, 1).contiguous())
                    padding_mask_sp_scattered = True
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        if auto_hsd_anchor:
            if decoder_input is None:
                raise NotImplementedError('Model-side HSD anchor insertion requires decoder input on this rank.')
            batch_size, text_seq_len = input_ids.shape
            packed_hsd_anchor = packed_seq_params is not None
            hsd_sp_scattered_decoder_input = (
                bool(getattr(self.config, 'sequence_parallel', False))
                and parallel_state.get_tensor_model_parallel_world_size() > 1
                and decoder_input.shape[0] != text_seq_len
            )
            if hsd_sp_scattered_decoder_input:
                decoder_input = gather_from_sequence_parallel_region(
                    decoder_input,
                    tensor_parallel_output_grad=False,
                    group=self.pg_collection.tp,
                )
                if decoder_input.shape[0] != text_seq_len:
                    raise RuntimeError(
                        'HSD auto anchor SP gather did not recover the full text sequence: '
                        f'got {decoder_input.shape[0]} for text length {text_seq_len}.')
            if packed_hsd_anchor:
                hsd_text_cu_seqlens = self._packed_cu_seqlens(packed_seq_params)
                hsd_text_packed_seq_params = packed_seq_params
                anchor_mask, hsd_anchor_cu_seqlens, expanded_cu_seqlens = (
                    self._build_auto_hsd_packed_anchor_layout(
                        input_ids=input_ids,
                        packed_seq_params=packed_seq_params,
                        device=decoder_input.device,
                    ))
                # Keep the semantic count as the strip authority.  Stage-3
                # first pads dynamic per-document anchors to its fixed
                # physical M=128 (also for TP1), then applies any ordinary SP
                # divisibility padding.  Both suffixes are non-semantic and
                # must be excluded from the Local FlashQLA producer.
                unpadded_expanded_len = int(anchor_mask.shape[1])
                anchor_mask, hsd_anchor_cu_seqlens, expanded_cu_seqlens = (
                    self._pad_hsd_packed_anchor_layout_to_fixed_count(
                        anchor_mask,
                        hsd_anchor_cu_seqlens,
                        expanded_cu_seqlens,
                    ))
                anchor_mask, hsd_anchor_cu_seqlens, expanded_cu_seqlens = (
                    self._pad_hsd_anchor_layout_for_sequence_parallel(
                        anchor_mask,
                        anchor_cu_seqlens=hsd_anchor_cu_seqlens,
                        expanded_cu_seqlens=expanded_cu_seqlens,
                    ))
                hsd_sp_padding_anchor_count = int(anchor_mask.shape[1]) - unpadded_expanded_len
                hsd_persistent_memory_mask = self._build_hsd_persistent_memory_mask(
                    anchor_mask,
                    anchor_cu_seqlens=hsd_anchor_cu_seqlens,
                )
                packed_seq_params = self._copy_packed_seq_params_with_cu_seqlens(
                    packed_seq_params,
                    expanded_cu_seqlens,
                )
            else:
                anchor_mask = self._build_auto_hsd_anchor_mask(batch_size, text_seq_len, decoder_input.device)
                unpadded_expanded_len = int(anchor_mask.shape[1])
                anchor_mask, _, _ = self._pad_hsd_anchor_layout_for_sequence_parallel(anchor_mask)
                hsd_sp_padding_anchor_count = int(anchor_mask.shape[1]) - unpadded_expanded_len
                hsd_persistent_memory_mask = self._build_hsd_persistent_memory_mask(anchor_mask)
            has_anchor_tokens = bool(anchor_mask.any().item())
            if has_anchor_tokens:
                decoder_input = self._expand_decoder_input_for_hsd_anchor(
                    decoder_input,
                    anchor_mask,
                    text_seq_len=text_seq_len,
                )
                if self.position_embedding_type == 'mrope' or packed_hsd_anchor:
                    position_ids = self._expand_position_ids_for_hsd_anchor(
                        position_ids,
                        anchor_mask,
                        text_seq_len=text_seq_len,
                        persistent_memory_mask=hsd_persistent_memory_mask,
                    )
                if padding_mask is not None:
                    padding_mask = self._expand_padding_mask_for_hsd_anchor(
                        padding_mask,
                        anchor_mask,
                        text_seq_len=text_seq_len,
                    )
            # Validate and restore the SP layout even for a semantic M=0
            # batch.  The auto-layout path gathers the text sequence before it
            # knows whether any anchor is present; leaving the gathered
            # tensor in place when ``has_anchor_tokens`` is false violates
            # the decoder's sequence-parallel input contract.
            self._validate_hsd_sequence_parallel_layout(
                anchor_mask,
                text_cu_seqlens=hsd_text_cu_seqlens,
                anchor_cu_seqlens=hsd_anchor_cu_seqlens,
            )
            if hsd_sp_scattered_decoder_input:
                decoder_input = tensor_parallel.scatter_to_sequence_parallel_region(
                    decoder_input,
                    group=self.pg_collection.tp,
                )

        if (hsd_persistent_memory_mask is None and anchor_mask is not None
                and has_anchor_tokens and self._hsd_persistent_memory_enabled()):
            # Explicit non-packed layouts are supported for tests/custom
            # callers. Packed layouts are constructed model-side above so the
            # per-document anchor cu-seqlens are available.
            hsd_persistent_memory_mask = self._build_hsd_persistent_memory_mask(anchor_mask)

        if padding_mask_needs_sp_scatter and padding_mask is not None and not padding_mask_sp_scattered:
            padding_mask = (
                tensor_parallel.scatter_to_sequence_parallel_region(
                    padding_mask.transpose(0, 1).contiguous(),
                    group=self.pg_collection.tp,
                ).transpose(0, 1).contiguous())

        if decoder_input is not None and self.training and torch.is_grad_enabled() and not decoder_input.requires_grad:
            # fix LoRA incompatibility with gradient checkpointing
            decoder_input = decoder_input.requires_grad_(True)

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        rotary_pos_cos = None
        rotary_pos_sin = None
        rotary_pos_cos_sin = None
        if self.position_embedding_type in {'rope', 'mrope'}:
            if not self.training and self.config.flash_decode and inference_context:
                assert (inference_context.is_static_batching()
                        ), 'GPTModel currently only supports static inference batching.'
                # Flash decoding uses precomputed cos and sin for RoPE
                rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb_cache.setdefault(
                    inference_context.max_sequence_length,
                    self.rotary_pos_emb.get_cos_sin(inference_context.max_sequence_length),
                )
            else:
                rotary_seq_len = RotaryEmbedding.get_rotary_seq_len(self, inference_context, self.decoder,
                                                                    decoder_input, self.config, packed_seq_params)
                if self.hf_rope_scaling is not None:
                    attention_scaling = dynamic_rope_update(self, self.rotary_pos_emb.inv_freq, rotary_seq_len)
                    if attention_scaling is not None and attention_scaling != self.attention_scaling:
                        raise ValueError('Currently does not support changing attention_scaling during training. '
                                         f'self.attention_scaling: {self.attention_scaling}, '
                                         f'current_attention_scaling: {attention_scaling}.')
                if self.position_embedding_type == 'mrope':
                    rotary_pos_emb = self.rotary_pos_emb(
                        position_ids,
                        mrope_section=self.mrope_section,
                        mrope_interleaved=self.config.mrope_interleaved,
                    )
                else:
                    packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
                    rotary_pos_emb = self.rotary_pos_emb(
                        rotary_seq_len,
                        packed_seq=packed_seq,
                    )

        if (in_inference_mode and ((self.config.enable_cuda_graph and self.config.cuda_graph_scope != 'full_iteration')
                                   or self.config.flash_decode) and rotary_pos_cos is not None
                and inference_context.is_static_batching()):
            current_batch_size = input_ids.shape[0]
            sequence_len_offset = torch.tensor(
                [inference_context.sequence_len_offset] * current_batch_size,
                dtype=torch.int32,
                device=rotary_pos_cos.device,  # Co-locate this with the rotary tensors
            )
        else:
            sequence_len_offset = None

        # Wrap decoder_input to allow the decoder (TransformerBlock) to delete the
        # reference held by this caller function, enabling early garbage collection for
        # inference. Skip wrapping if decoder_input is logged after decoder completion.
        if in_inference_mode and not has_config_logger_enabled(self.config):
            decoder_input = WrappedTensor(decoder_input)

        return (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
            padding_mask,
            rotary_pos_cos_sin,
            anchor_mask,
            has_anchor_tokens,
            position_ids,
            packed_seq_params,
            hsd_text_cu_seqlens,
            hsd_anchor_cu_seqlens,
            hsd_text_packed_seq_params,
            hsd_persistent_memory_mask,
            hsd_sp_padding_anchor_count,
        )

    # Code borrowed from NVIDIA/Megatron-LM
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        decoder_input: torch.Tensor = None,
        labels: torch.Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        anchor_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units

        Args:
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
        """

        if getattr(self.config, 'fine_grained_activation_offloading', False):
            self.preprocess_for_fine_grained_offloading()

        inference_context = deprecate_inference_params(inference_context, inference_params)
        auto_hsd_anchor = self._should_auto_hsd_anchor(
            anchor_mask=anchor_mask,
            input_ids=input_ids,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
        )
        anchor_mask, has_anchor_tokens = self._prepare_hsd_forward(
            anchor_mask=anchor_mask,
            input_ids=input_ids,
            labels=labels if self.config.task_type == 'causal_lm' else None,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
        )

        (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
            padding_mask,
            rotary_pos_cos_sin,
            anchor_mask,
            has_anchor_tokens,
            position_ids,
            packed_seq_params,
            hsd_text_cu_seqlens,
            hsd_anchor_cu_seqlens,
            hsd_text_packed_seq_params,
            hsd_persistent_memory_mask,
            hsd_sp_padding_anchor_count,
        ) = (
            self._preprocess(
                input_ids=input_ids,
                position_ids=position_ids,
                decoder_input=decoder_input,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
                padding_mask=padding_mask,
                anchor_mask=anchor_mask,
                has_anchor_tokens=has_anchor_tokens,
                auto_hsd_anchor=auto_hsd_anchor,
            ))
        decoder_input = self._apply_anchor_embedding(decoder_input, anchor_mask, has_anchor_tokens)
        self._trace_hsd_tensor('expanded_inputs_embeds', decoder_input)
        decoder_rotary_pos_emb = rotary_pos_emb
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        if self.position_embedding_type == 'rope' and packed_seq and not self.config.apply_rope_fusion:
            assert position_ids.shape[0] == 1, f'position_ids.shape: {position_ids.shape}'
            decoder_rotary_pos_emb = self._lookup_packed_rotary_pos_emb(
                rotary_pos_emb, position_ids[0])

        mtp_decoder_input = decoder_input
        if self.config.is_multimodal and self.config.mtp_num_layers and decoder_input is None:
            input_tensor = self.get_input_tensor()
            input_tensor, mtp_decoder_input = input_tensor.chunk(2, dim=0)
            self.set_input_tensor(input_tensor)
        # Run decoder.
        self._set_hsd_layer_context(
            anchor_mask,
            has_anchor_tokens,
            text_cu_seqlens=hsd_text_cu_seqlens,
            anchor_cu_seqlens=hsd_anchor_cu_seqlens,
            text_packed_seq_params=hsd_text_packed_seq_params,
            persistent_memory_mask=hsd_persistent_memory_mask,
            sp_padding_anchor_count=hsd_sp_padding_anchor_count,
        )
        try:
            hidden_states = self.decoder(
                hidden_states=decoder_input,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=decoder_rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                padding_mask=padding_mask,
                **(extra_block_kwargs or {}),
                **kwargs,
            )
        finally:
            # Activation checkpointing recomputes decoder layers during backward,
            # after this forward method has returned. Keep the HSD anchor layout
            # available for that recompute; the next forward overwrites it before
            # entering the decoder, and eval/no-grad paths can be cleared eagerly.
            if self._should_clear_hsd_layer_context_after_forward():
                self._set_hsd_layer_context(None, False)

        # MTP: https://github.com/NVIDIA/Megatron-LM/issues/1661
        return self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            loss_mask=loss_mask,
            decoder_input=mtp_decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            runtime_gather_output=runtime_gather_output,
            extra_block_kwargs=extra_block_kwargs,
            inference_context=inference_context,
            anchor_mask=anchor_mask,
            has_anchor_tokens=has_anchor_tokens,
        )

    def _postprocess(
        self,
        hidden_states,
        input_ids,
        position_ids,
        labels,
        rotary_pos_emb,
        rotary_pos_cos,
        rotary_pos_sin,
        loss_mask=None,
        decoder_input=None,
        attention_mask=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        runtime_gather_output=None,
        extra_block_kwargs=None,
        inference_context=None,
        anchor_mask=None,
        has_anchor_tokens=False,
    ):
        """Postprocesses decoder hidden states to generate logits or compute loss.

        Applies Multi-Token Prediction if enabled, generates output logits through
        the output layer, and computes language model loss when labels are provided.
        """
        if not self.post_process:
            if self.config.is_multimodal and self.config.mtp_num_layers:
                return torch.concat([hidden_states, decoder_input], dim=0)
            else:
                return hidden_states
        labels = labels if self.config.task_type == 'causal_lm' else None
        in_inference_mode = inference_context is not None and not self.training
        if in_inference_mode:
            assert runtime_gather_output, 'Inference must always gather TP logits'

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        if self.config.is_multimodal and self.config.context_parallel_size > 1 and input_ids is not None:
            # input_ids is required by MTP.
            input_ids = split_cp_inputs(input_ids, getattr(packed_seq_params, 'cu_seqlens_q', None), 1)

        if self.mtp_process and labels is not None:
            hidden_states = self.mtp(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                embedding=self.embedding,
                decoder_input=decoder_input if self.config.is_multimodal else None,
                **(extra_block_kwargs or {}),
            )
            mtp_labels = labels.clone()
            hidden_states_list = torch.chunk(hidden_states, 1 + self.config.mtp_unroll_steps, dim=0)
            hidden_states = hidden_states_list[0]
            if loss_mask is None:
                # if loss_mask is not provided, use all ones as loss_mask
                loss_mask = torch.ones_like(mtp_labels)
            for mtp_layer_number in range(self.config.mtp_unroll_steps):
                # output
                mtp_logits, _ = self.output_layer(
                    hidden_states_list[mtp_layer_number + 1],
                    weight=output_weight,
                    runtime_gather_output=runtime_gather_output,
                )
                # Calc loss for the current Multi-Token Prediction (MTP) layers.
                mtp_labels, _ = roll_tensor(
                    mtp_labels,
                    shifts=-1,
                    dims=-1,
                    cp_group=self.cp_group,
                    packed_seq_params=packed_seq_params,
                )
                loss_mask, _ = roll_tensor(
                    loss_mask,
                    shifts=-1,
                    dims=-1,
                    cp_group=self.cp_group,
                    packed_seq_params=packed_seq_params,
                )
                mtp_loss = self.compute_language_model_loss(mtp_labels, mtp_logits)
                loss_mask_ = (loss_mask & (mtp_labels != -100))
                num_tokens = loss_mask_.sum()
                mtp_loss = loss_mask_ * mtp_loss
                if self.training:
                    mtp_loss_for_log = (
                        torch.sum(mtp_loss) / num_tokens if num_tokens > 0 else mtp_loss.new_tensor(0.0))
                    MTPLossLoggingHelper.save_loss_to_tracker(
                        mtp_loss_for_log,
                        mtp_layer_number,
                        self.config.mtp_unroll_steps,
                        avg_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
                    )
                mtp_loss_scale = self.config.mtp_loss_scaling_factor / self.config.mtp_unroll_steps
                if self.config.calculate_per_token_loss:
                    hidden_states = MTPLossAutoScaler.apply(hidden_states, mtp_loss_scale * mtp_loss)
                else:
                    hidden_states = MTPLossAutoScaler.apply(hidden_states, mtp_loss_scale * mtp_loss / num_tokens)
        sequence_parallel_override = False
        output_layer_sequence_parallel_old = None
        if in_inference_mode and inference_context.materialize_only_last_token_logits:
            if inference_context.is_static_batching():
                hidden_states = hidden_states[-1:, :, :]
            else:
                if self.output_layer.sequence_parallel:
                    # Perform the sequence parallel gather here instead of after the output layer
                    # because we need to slice the last token logits from the full view of the
                    # packed logits across all requests.
                    # TODO(ksanthanam): Make the equivalent change in the `MambaModel` code after
                    # merging in !3722.
                    hidden_states = gather_from_sequence_parallel_region(hidden_states, group=self.pg_collection.tp)
                    output_layer_sequence_parallel_old = self.output_layer.sequence_parallel
                    self.output_layer.sequence_parallel = False
                    sequence_parallel_override = True

                # Reshape [B, 1, H] to [1, B, H] → extract each sample’s true last‐token hidden
                # state ([B, H]) → unsqueeze back to [1, B, H]
                # (so that the output layer, which expects S×B×H, receives only the final token)
                hidden_states = inference_context.last_token_logits(hidden_states.squeeze(1).unsqueeze(0)).unsqueeze(1)

        if self.config.task_type in {'seq_cls', 'embedding'
                                     } and self.config.sequence_parallel and self.config.tensor_model_parallel_size > 1:
            hidden_states = gather_from_sequence_parallel_region(hidden_states)

        labels_for_loss = labels
        hsd_loss_keep_idx = None
        hsd_sp_padded_text_loss = False
        if has_anchor_tokens and getattr(self.config, 'sequence_parallel', False):
            anchor_mask = self._validate_hsd_anchor_mask(anchor_mask)
            if (
                labels is not None
                and getattr(self.output_layer, 'sequence_parallel', False)
                and parallel_state.get_tensor_model_parallel_world_size() > 1
            ):
                hidden_states, labels_for_loss, hsd_loss_keep_idx = self._prepare_hsd_sp_padded_text_loss_inputs(
                    hidden_states,
                    anchor_mask,
                    labels,
                )
                hsd_sp_padded_text_loss = True
            else:
                full_hsd_seq_len = anchor_mask.shape[1]
                if hidden_states.shape[0] != full_hsd_seq_len:
                    hidden_states = gather_from_sequence_parallel_region(
                        hidden_states,
                        tensor_parallel_output_grad=False,
                        group=self.pg_collection.tp,
                    )
                    if hidden_states.shape[0] != full_hsd_seq_len:
                        raise RuntimeError(
                            'HSD postprocess SP gather did not recover the full expanded sequence: '
                            f'got {hidden_states.shape[0]} for expanded length {full_hsd_seq_len}.')
                if getattr(self.output_layer, 'sequence_parallel', False):
                    output_layer_sequence_parallel_old = self.output_layer.sequence_parallel
                    self.output_layer.sequence_parallel = False
                    sequence_parallel_override = True

        if has_anchor_tokens and not hsd_sp_padded_text_loss:
            hidden_states = self._filter_anchor_hidden_states(hidden_states, anchor_mask, labels=labels)
        self._trace_hsd_tensor('hidden_last', hidden_states, category='forward')

        if self.config.task_type == 'embedding':
            logits = F.normalize(hidden_states, p=2, dim=-1)
        else:
            logits, _ = self.output_layer(
                hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output)
            self._trace_hsd_tensor(
                'forward/lm_head',
                logits,
                category='forward_logits',
                extra={'layout': 'sequence_batch_vocab'},
            )
            if self.config.task_type == 'generative_reranker':
                logits = gather_from_tensor_model_parallel_region(logits)
                positive_token = os.environ.get('GENERATIVE_RERANKER_POSITIVE_TOKEN', 'yes')
                negative_token = os.environ.get('GENERATIVE_RERANKER_NEGATIVE_TOKEN', 'no')
                positive_token_id = self.tokenizer.convert_tokens_to_ids(positive_token)
                negative_token_id = self.tokenizer.convert_tokens_to_ids(negative_token)
                logits = (logits[..., positive_token_id] - logits[..., negative_token_id])[..., None]
        # Restore sequence parallel execution to the output layer if necessary.
        if sequence_parallel_override:
            self.output_layer.sequence_parallel = output_layer_sequence_parallel_old

        if has_config_logger_enabled(self.config):
            payload = OrderedDict({
                'input_ids': input_ids,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'decoder_input': decoder_input,
                'logits': logits,
            })
            log_config_to_disk(self.config, payload, prefix='input_and_logits')

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        loss = self.compute_language_model_loss(labels_for_loss, logits)
        if hsd_loss_keep_idx is not None:
            loss = loss.index_select(1, hsd_loss_keep_idx.to(device=loss.device))

        return loss

    def get_input_tensor(self):
        return self.decoder.input_tensor

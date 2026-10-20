# Copyright (c) ModelScope Contributors. All rights reserved.
import json
from pathlib import Path
import torch
from march.safin1.runtime_validation import march_training_enabled
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm
from megatron.core.transformer.attention import SelfAttention
from typing import Optional

from mcore_bridge.bridge import MultimodalGPTBridge
from mcore_bridge.utils import get_env_args

from ..constant import ModelType
from ..modules import GatedDeltaNet, GatedSelfAttention, MarchGatedDeltaNet
from ..register import ModelLoader, ModelMeta, register_model
from .qwen3_5 import Qwen3_5Vit, Qwen3NextBridge


def _linear_attention_flags(config):
    linear_attention_freq = getattr(config, 'linear_attention_freq', None)
    num_layers = int(getattr(config, 'num_layers', 0) or 0)
    if linear_attention_freq is None:
        return [False] * num_layers
    if isinstance(linear_attention_freq, str):
        raw = linear_attention_freq.strip()
        if '[' in raw or '(' in raw:
            if any(ch not in '0123456789,[]()+* ' for ch in raw):
                raise ValueError(f'Invalid linear_attention_freq pattern: {linear_attention_freq}')
            linear_attention_freq = eval(raw)
        else:
            linear_attention_freq = int(raw)
    if isinstance(linear_attention_freq, int):
        if linear_attention_freq <= 0:
            raise ValueError('linear_attention_freq must be positive.')
        if num_layers <= 0:
            raise ValueError('num_layers must be positive when linear_attention_freq is an integer.')
        return [((i + 1) % linear_attention_freq) != 0 for i in range(num_layers)]
    return [bool(v) for v in linear_attention_freq]


class Qwen3_5Bridge(MultimodalGPTBridge):
    hf_mtp_prefix = 'mtp.layers'
    # Checkpoint layout is independent of the public model-type name.
    hf_shared_expert_key = 'shared_expert'

    def _requires_hsd_history_gate(self, mg_attn, to_mcore: bool) -> bool:
        if to_mcore or mg_attn is None:
            return False
        if not bool(getattr(mg_attn, 'hsd_layer_uses_global_read', False)):
            return False
        return bool(getattr(mg_attn, 'hsd_learnable_history_scale',
                            getattr(self.config, 'hsd_learnable_history_scale', False)))

    def _requires_hsd_null_read_gate(self, mg_attn, to_mcore: bool) -> bool:
        if to_mcore or mg_attn is None:
            return False
        if not bool(getattr(mg_attn, 'hsd_layer_uses_global_read', False)):
            return False
        return bool(getattr(mg_attn, 'hsd_null_read_gate',
                            getattr(self.config, 'hsd_null_read_gate', False)))

    def _adapt_hsd_route_weight(self, expanded_weight, *, output_heads: int, output_head_dim: int,
                                source_head_dim: int, name: str):
        if output_head_dim > source_head_dim:
            raise ValueError(
                f'Cannot initialize {name}: route head dim {output_head_dim} exceeds source '
                f'GDN key head dim {source_head_dim}.'
            )
        hidden_size = expanded_weight.shape[-1]
        source_heads = expanded_weight.shape[0] // source_head_dim
        per_head = expanded_weight.reshape(source_heads, source_head_dim, hidden_size)[:, :output_head_dim, :]
        if output_heads == source_heads:
            selected = per_head
        elif output_heads == 1:
            selected = per_head.mean(dim=0, keepdim=True)
        elif source_heads % output_heads == 0:
            group = source_heads // output_heads
            selected = per_head.reshape(output_heads, group, output_head_dim, hidden_size).mean(dim=1)
        else:
            raise ValueError(
                f'Cannot initialize {name}: output_heads={output_heads} must be 1 or divide '
                f'source_heads={source_heads}.'
            )
        return selected.reshape(output_heads * output_head_dim, hidden_size)

    def _validate_hf_import(self, state_dict, hf_model_dir, peft_format):
        """Validate a full Safin checkpoint before any parameter is copied."""
        self._hsd_import_is_base_qwen = False
        self._hsd_allow_new_persistent_state = False
        if peft_format or not march_training_enabled(self.config):
            return
        source = json.loads((Path(hf_model_dir) / 'config.json').read_text())
        text = source.get('text_config') or source
        source_type = source.get('model_type') or text.get('model_type')
        base_types = {'qwen3_5', 'qwen3_5_text', 'qwen3_5_moe', 'qwen3_5_moe_text'}
        safin_source = str(source_type).startswith(('safin1_', 'historical_qwen3_5'))
        if source_type not in base_types and not safin_source:
            raise ValueError(f'MARCH import requires a Qwen3.5 or Safin-1 source config, got {source_type!r}.')
        retired = sorted(k for k in state_dict if k.endswith(
            ('history_read_q_down_proj.weight', 'history_read_q_up_proj.weight')))
        if retired:
            raise ValueError(f'Retired independent history-query adapter weights cannot be loaded: {retired}')
        markers = ('anchor_embedding.weight', 'anchor_q_proj.weight', 'anchor_k_proj.weight',
                   'router_q_proj.weight', 'persistent_memory_state', 'persistent_memory_route_key',
                   'null_gate_proj.weight', 'hsd_null_read_gate_bias', 'hsd_history_gate_logit',
                   'hsd_history_out_rmsnorm_weight')
        has_march_weights = any(k.endswith(markers) for k in state_dict)
        if source_type in base_types and not has_march_weights:
            if getattr(self.config, 'hsd_persistent_memory_route_key_enabled', False):
                raise KeyError('Dedicated persistent-memory initialization requires its explicit route-key artifact.')
            self._hsd_import_is_base_qwen = True
            self._hsd_allow_new_persistent_state = True
            return
        required = set().union(*self._expected_hsd_export_keys(hsd_enabled=True).values())
        if not required:
            raise ValueError('MARCH import requires explicit positive anchor_chunk_size and anchor_token_num.')
        # Adding memory to an existing pre-memory model is the documented zero
        # initialization. An already enabled memory checkpoint must be complete.
        self._hsd_allow_new_persistent_state = (
            not bool(text.get('hsd_persistent_memory_enabled', False))
            and bool(getattr(self.config, 'hsd_persistent_memory_enabled', False)))
        if self._hsd_allow_new_persistent_state:
            required = {k for k in required if not k.endswith('.persistent_memory_state')}
        missing = sorted(required - state_dict.keys())
        if missing:
            raise KeyError(
                f'Incomplete Safin-1 checkpoint: missing active MARCH parameters {missing}. '
                'Only a base Qwen3.5 checkpoint may initialize new MARCH projections; '
                'restore complete weights or explicitly migrate a checkpoint copy.')

    def _set_historical_route_state(self, mg_attn, hf_state_dict, hf_prefix: str, to_mcore: bool):
        if mg_attn is None or not hasattr(mg_attn, 'anchor_q_proj'):
            return {}

        if getattr(self, '_peft_format', False):
            # Share the generic LoRA/modules_to_save conversion in both
            # directions, including target-module metadata and TP partitioning.
            # Frozen base projections and persistent state belong to the base
            # checkpoint and must not be copied or zeroed by an adapter import.
            route_state_dict = self._remove_prefix(hf_state_dict, hf_prefix) if to_mcore else {}
            if to_mcore:
                retired_keys = sorted(k for k in route_state_dict if k.split('.', 1)[0] in {
                    'history_read_q_down_proj', 'history_read_q_up_proj'})
                if retired_keys:
                    raise ValueError(f'Retired independent history-query adapter weights cannot be loaded: {retired_keys}')
            else:
                raw_trainable = [name for name in (
                    'hsd_null_read_gate_bias', 'hsd_history_gate_logit', 'hsd_history_out_rmsnorm_weight',
                    'persistent_memory_state', 'persistent_memory_route_key',
                ) if getattr(getattr(mg_attn, name, None), 'requires_grad', False)]
                if raw_trainable:
                    raise ValueError(
                        f'Unmerged PEFT export cannot represent trainable MARCH parameters {raw_trainable}. '
                        'Use --merge_lora true to preserve them in the full checkpoint.')
            for name in ('anchor_q_proj', 'anchor_k_proj', 'router_q_proj', 'null_gate_proj'):
                if getattr(mg_attn, name, None) is not None:
                    self._set_state_dict(mg_attn, f'{name}.weight', route_state_dict, f'{name}.weight', to_mcore)
            return {} if to_mcore else self._add_prefix(route_state_dict, hf_prefix)

        if to_mcore:
            local_hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
            retired_keys = sorted(set(local_hf_state_dict) & {
                'history_read_q_down_proj.weight', 'history_read_q_up_proj.weight',
            })
            if retired_keys:
                raise ValueError(
                    f'Retired independent history-query adapter weights cannot be loaded: {retired_keys}. '
                    'Safin-1 training supports hsd_history_read_q_mode=shared_local only.')
            needs_anchor_k = getattr(mg_attn, 'anchor_k_proj', None) is not None
            needs_router_q = getattr(mg_attn, 'router_q_proj', None) is not None
            needs_history_gate = getattr(mg_attn, 'hsd_history_gate_logit', None) is not None
            needs_history_out_rmsnorm = getattr(mg_attn, 'hsd_history_out_rmsnorm_weight', None) is not None
            needs_null_gate = getattr(mg_attn, 'null_gate_proj', None) is not None
            needs_null_gate_bias = getattr(mg_attn, 'hsd_null_read_gate_bias', None) is not None
            needs_persistent_memory = getattr(mg_attn, 'persistent_memory_state', None) is not None
            needs_persistent_route_key = getattr(mg_attn, 'persistent_memory_route_key', None) is not None
            required = {'anchor_q_proj.weight'}
            for needed, key in (
                (needs_anchor_k, 'anchor_k_proj.weight'), (needs_router_q, 'router_q_proj.weight'),
                (needs_history_gate, 'hsd_history_gate_logit'),
                (needs_history_out_rmsnorm, 'hsd_history_out_rmsnorm_weight'),
                (needs_null_gate, 'null_gate_proj.weight'), (needs_null_gate_bias, 'hsd_null_read_gate_bias'),
                (needs_persistent_route_key, 'persistent_memory_route_key'),
                (needs_persistent_memory and not getattr(self, '_hsd_allow_new_persistent_state', False),
                 'persistent_memory_state'),
            ):
                if needed:
                    required.add(key)
            initialize_base = bool(getattr(self, '_hsd_import_is_base_qwen', False))
            if not initialize_base or required.intersection(local_hf_state_dict):
                missing = sorted(required - local_hf_state_dict.keys())
                if missing:
                    raise KeyError(f'Incomplete Safin-1 layer: missing active MARCH parameters {missing}')
            elif needs_persistent_route_key and 'persistent_memory_route_key' not in local_hf_state_dict:
                raise KeyError('Dedicated persistent-memory initialization requires its explicit route-key artifact.')
            has_new_hsd = (
                'anchor_q_proj.weight' in local_hf_state_dict
                and (not needs_anchor_k or 'anchor_k_proj.weight' in local_hf_state_dict)
                and (not needs_router_q or 'router_q_proj.weight' in local_hf_state_dict)
            )
            if has_new_hsd:
                self._set_state_dict(mg_attn, 'anchor_q_proj.weight', local_hf_state_dict, 'anchor_q_proj.weight',
                                     to_mcore)
                if needs_anchor_k:
                    self._set_state_dict(mg_attn, 'anchor_k_proj.weight', local_hf_state_dict,
                                         'anchor_k_proj.weight', to_mcore)
                if needs_router_q:
                    self._set_state_dict(mg_attn, 'router_q_proj.weight', local_hf_state_dict,
                                         'router_q_proj.weight', to_mcore)
            else:
                config = self.config
                num_key_heads = config.linear_num_key_heads
                key_dim = config.linear_key_head_dim
                value_dim = config.linear_value_head_dim * config.linear_num_value_heads // num_key_heads
                num_value_groups = config.linear_num_value_heads // num_key_heads

                qkv = local_hf_state_dict['in_proj_qkv.weight'].load()
                q, k, _ = torch.split(
                    qkv, [key_dim * num_key_heads, key_dim * num_key_heads, value_dim * num_key_heads], dim=0)
                expanded_q = q.reshape(num_key_heads, key_dim, config.hidden_size)
                expanded_q = expanded_q[:, None].expand(-1, num_value_groups, -1, -1).reshape(
                    config.linear_num_value_heads * key_dim, config.hidden_size)
                expanded_k = k.reshape(num_key_heads, key_dim, config.hidden_size)
                expanded_k = expanded_k[:, None].expand(-1, num_value_groups, -1, -1).reshape(
                    config.linear_num_value_heads * key_dim, config.hidden_size)
                self._set_weight(mg_attn.anchor_q_proj.weight, expanded_q, 'anchor_q_proj.weight')
                if needs_anchor_k:
                    route_head_dim = int(getattr(mg_attn, 'hsd_route_head_dim', key_dim) or key_dim)
                    route_key_heads = int(getattr(mg_attn, 'hsd_route_key_heads',
                                                   config.linear_num_value_heads) or config.linear_num_value_heads)
                    route_k = self._adapt_hsd_route_weight(
                        expanded_k,
                        output_heads=route_key_heads,
                        output_head_dim=route_head_dim,
                        source_head_dim=key_dim,
                        name='anchor_k_proj.weight',
                    )
                    self._set_weight(mg_attn.anchor_k_proj.weight, route_k, 'anchor_k_proj.weight')
                if needs_router_q:
                    route_head_dim = int(getattr(mg_attn, 'hsd_route_head_dim', key_dim) or key_dim)
                    route_query_heads = int(getattr(mg_attn, 'hsd_route_query_heads',
                                                     config.linear_num_value_heads) or config.linear_num_value_heads)
                    route_q = self._adapt_hsd_route_weight(
                        expanded_q,
                        output_heads=route_query_heads,
                        output_head_dim=route_head_dim,
                        source_head_dim=key_dim,
                        name='router_q_proj.weight',
                    )
                    self._set_weight(mg_attn.router_q_proj.weight, route_q, 'router_q_proj.weight')
            if needs_null_gate and 'null_gate_proj.weight' in local_hf_state_dict:
                self._set_state_dict(mg_attn, 'null_gate_proj.weight', local_hf_state_dict,
                                     'null_gate_proj.weight', to_mcore)
            if needs_null_gate_bias and 'hsd_null_read_gate_bias' in local_hf_state_dict:
                self._set_state_dict(mg_attn, 'hsd_null_read_gate_bias', local_hf_state_dict,
                                     'hsd_null_read_gate_bias', to_mcore)
            if needs_history_gate and 'hsd_history_gate_logit' in local_hf_state_dict:
                self._set_state_dict(mg_attn, 'hsd_history_gate_logit', local_hf_state_dict,
                                     'hsd_history_gate_logit', to_mcore)
            if needs_history_out_rmsnorm and 'hsd_history_out_rmsnorm_weight' in local_hf_state_dict:
                self._set_state_dict(mg_attn, 'hsd_history_out_rmsnorm_weight', local_hf_state_dict,
                                     'hsd_history_out_rmsnorm_weight', to_mcore)
            if needs_persistent_memory:
                if 'persistent_memory_state' in local_hf_state_dict:
                    self._set_state_dict(mg_attn, 'persistent_memory_state', local_hf_state_dict,
                                         'persistent_memory_state', to_mcore)
                else:
                    # Loading a pre-memory HSD checkpoint is an explicit,
                    # deterministic zero initialization, not a fresh-model RNG
                    # initialization.
                    with torch.no_grad():
                        mg_attn.persistent_memory_state.zero_()
            if needs_persistent_route_key:
                if 'persistent_memory_route_key' not in local_hf_state_dict:
                    raise KeyError(
                        'Dedicated persistent-memory checkpoint is missing persistent_memory_route_key. '
                        'Create the immutable legacy-key initialization artifact before MCore import.')
                self._set_state_dict(
                    mg_attn,
                    'persistent_memory_route_key',
                    local_hf_state_dict,
                    'persistent_memory_route_key',
                    to_mcore,
                )
            return {}

        if self._requires_hsd_history_gate(mg_attn, to_mcore) and getattr(mg_attn, 'hsd_history_gate_logit', None) is None:
            raise RuntimeError(
                'HSD export requires hsd_history_gate_logit for learnable-history global-read layers, but it is '
                'missing. Ensure hsd_learnable_history_scale is applied before exporting.'
            )
        if self._requires_hsd_null_read_gate(mg_attn, to_mcore) and (
                getattr(mg_attn, 'null_gate_proj', None) is None
                or getattr(mg_attn, 'hsd_null_read_gate_bias', None) is None):
            raise RuntimeError(
                'HSD export requires null_gate_proj and hsd_null_read_gate_bias for null-read global-read layers, '
                'but one is missing. Ensure hsd_null_read_gate is applied before exporting.'
            )
        route_state_dict = {}
        anchor_q, _ = self._get_weight(mg_attn.anchor_q_proj.weight.data, 'anchor_q_proj.weight')
        route_state_dict['anchor_q_proj.weight'] = anchor_q
        if getattr(mg_attn, 'anchor_k_proj', None) is not None:
            anchor_k, _ = self._get_weight(mg_attn.anchor_k_proj.weight.data, 'anchor_k_proj.weight')
            route_state_dict['anchor_k_proj.weight'] = anchor_k
        if getattr(mg_attn, 'router_q_proj', None) is not None:
            router_q, _ = self._get_weight(mg_attn.router_q_proj.weight.data, 'router_q_proj.weight')
            route_state_dict['router_q_proj.weight'] = router_q
        if getattr(mg_attn, 'null_gate_proj', None) is not None:
            null_gate, _ = self._get_weight(mg_attn.null_gate_proj.weight.data, 'null_gate_proj.weight')
            route_state_dict['null_gate_proj.weight'] = null_gate
        if getattr(mg_attn, 'hsd_null_read_gate_bias', None) is not None:
            null_gate_bias, _ = self._get_weight(
                mg_attn.hsd_null_read_gate_bias.data, 'hsd_null_read_gate_bias')
            route_state_dict['hsd_null_read_gate_bias'] = null_gate_bias
        if getattr(mg_attn, 'hsd_history_gate_logit', None) is not None:
            history_gate, _ = self._get_weight(mg_attn.hsd_history_gate_logit.data, 'hsd_history_gate_logit')
            route_state_dict['hsd_history_gate_logit'] = history_gate
        if getattr(mg_attn, 'hsd_history_out_rmsnorm_weight', None) is not None:
            history_out_rmsnorm, _ = self._get_weight(
                mg_attn.hsd_history_out_rmsnorm_weight.data, 'hsd_history_out_rmsnorm_weight')
            route_state_dict['hsd_history_out_rmsnorm_weight'] = history_out_rmsnorm
        if getattr(mg_attn, 'persistent_memory_state', None) is not None:
            persistent_memory, _ = self._get_weight(
                mg_attn.persistent_memory_state.data, 'persistent_memory_state')
            route_state_dict['persistent_memory_state'] = persistent_memory
        if getattr(mg_attn, 'persistent_memory_route_key', None) is not None:
            persistent_route_key, _ = self._get_weight(
                mg_attn.persistent_memory_route_key.data, 'persistent_memory_route_key')
            route_state_dict['persistent_memory_route_key'] = persistent_route_key
        return self._add_prefix(route_state_dict, hf_prefix)

    def _set_historical_route_layer_norm_state(self, mg_attn, hf_state_dict, hf_key: str, to_mcore: bool):
        if not to_mcore or mg_attn is None:
            return
        for proj_name in (
            'anchor_q_proj',
            'anchor_k_proj',
            'router_q_proj',
            'null_gate_proj',
        ):
            proj = getattr(mg_attn, proj_name, None)
            if getattr(proj, 'layer_norm_weight', None) is None:
                continue
            self._set_state_dict(mg_attn, f'{proj_name}.layer_norm_weight', hf_state_dict, hf_key, to_mcore)

    def _set_layer_attn(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool):
        mg_attn = None if mg_layer is None else mg_layer.self_attention
        is_linear_attention = self.config.linear_attention_freq[layer_idx]
        if is_linear_attention:
            hf_state_dict.update(
                self._set_linear_attn_state(mg_attn, hf_state_dict, 'linear_attn.', layer_idx, to_mcore))
            hf_state_dict.update(self._set_historical_route_state(mg_attn, hf_state_dict, 'linear_attn.', to_mcore))

            if self.config.linear_decoupled_in_proj:
                self._set_state_dict(mg_layer, 'input_layernorm.weight', hf_state_dict, 'input_layernorm.weight',
                                     to_mcore)
            else:
                self._set_state_dict(mg_layer, 'self_attention.in_proj.layer_norm_weight', hf_state_dict,
                                     'input_layernorm.weight', to_mcore)
                self._set_historical_route_layer_norm_state(
                    mg_attn, hf_state_dict, 'input_layernorm.weight', to_mcore)
        else:
            hf_state_dict.update(self._set_attn_state(mg_attn, hf_state_dict, 'self_attn.', layer_idx, to_mcore))
            self._set_state_dict(mg_layer, 'self_attention.linear_qkv.layer_norm_weight', hf_state_dict,
                                 'input_layernorm.weight', to_mcore)
        return hf_state_dict

    def _expected_hsd_export_keys(self, peft_format: bool = False, hsd_enabled: bool = False):
        expected = super()._expected_hsd_export_keys(peft_format=peft_format, hsd_enabled=hsd_enabled)
        if peft_format or not hsd_enabled:
            return expected
        anchor_token_num = int(getattr(self.config, 'anchor_token_num', 0) or 0)
        anchor_chunk_size = int(getattr(self.config, 'anchor_chunk_size',
                                         getattr(self.config, 'hist_chunk_size', 0)) or 0)
        if anchor_token_num <= 0 or anchor_chunk_size <= 0:
            return expected

        use_global_read = bool(getattr(self.config, 'hsd_use_global_read', True)) and bool(
            getattr(self.config, 'hist_use_history_read', True))
        global_layers = self._parse_hsd_layer_selector(getattr(self.config, 'hsd_global_read_layers', None))
        persistent_route_key = bool(
            getattr(self.config, 'hsd_persistent_memory_route_key_enabled', False))
        expected.setdefault('persistent_memory_route_key', set())
        last_global_read_layer = None
        if use_global_read:
            last_global_read_layer = (
                len(_linear_attention_flags(self.config)) if global_layers is None
                else (max(global_layers) if global_layers else None)
            )
        if persistent_route_key and last_global_read_layer is not None:
            for layer_idx, is_linear in enumerate(_linear_attention_flags(self.config)):
                global_read_layer = (
                    is_linear
                    and use_global_read
                    and (global_layers is None or (layer_idx + 1) in global_layers)
                )
                if global_read_layer:
                    expected['persistent_memory_route_key'].add(
                        f'{self.hf_layers_prefix}.{layer_idx}.linear_attn.'
                        'persistent_memory_route_key')
        return expected

    def _convert_mtp_extra(self, mtp_layer, hf_state_dict, to_mcore, origin_hf_state_dict):
        return Qwen3NextBridge._convert_mtp_extra(self, mtp_layer, hf_state_dict, to_mcore, origin_hf_state_dict)


class Qwen3_5Loader(ModelLoader):

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import \
            get_transformer_block_with_experimental_attention_variant_spec
        use_march = march_training_enabled(self.config)
        layer_specs = get_transformer_block_with_experimental_attention_variant_spec(self.config, vp_stage)
        for layer_spec in layer_specs.layer_specs:
            attn_module = layer_spec.submodules.self_attention.module
            if issubclass(attn_module, SelfAttention):
                layer_spec.submodules.self_attention.module = GatedSelfAttention
            else:
                layer_spec.submodules.self_attention.module = (
                    MarchGatedDeltaNet if use_march else GatedDeltaNet)
                if self.config.linear_decoupled_in_proj:
                    layer_spec.submodules.input_layernorm = TENorm
                    layer_spec.submodules.self_attention.submodules.in_proj = TEColumnParallelLinear
        return layer_specs

    def build_model(
        self,
        pre_process=True,
        post_process=True,
        vp_stage: Optional[int] = None,
    ):
        model = super().build_model(pre_process, post_process, vp_stage)
        lm_model = model.language_model if hasattr(model, 'language_model') else model
        for layer in lm_model.decoder.layers:
            if hasattr(layer.self_attention, 'out_norm'):
                assert hasattr(layer.self_attention.out_norm, 'zero_centered_gamma')
                layer.self_attention.out_norm.zero_centered_gamma = False
        return model


use_mcore_gdn = get_env_args('USE_MCORE_GDN', bool, True)

if use_mcore_gdn:
    register_model(
        ModelMeta(
            ModelType.qwen3_5,
            ['qwen3_5', 'safin1_4b', 'qwen3_5_moe', 'safin1_35b_a3b'],
            bridge_cls=Qwen3_5Bridge,
            visual_cls=Qwen3_5Vit,
            loader=Qwen3_5Loader,
        ))

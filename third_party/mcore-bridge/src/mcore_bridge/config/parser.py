# Copyright (c) ModelScope Contributors. All rights reserved.
from transformers import PretrainedConfig
from march.safin1.runtime_validation import validate_safin1_training_runtime
from typing import Any, Dict

from mcore_bridge.utils import get_env_args, json_parse_to_dict

config_mapping = {
    'num_layers': ['num_hidden_layers'],
    'hidden_size': ['hidden_size'],
    'mlp_ffn_hidden_size': ['intermediate_size_mlp'],
    'ffn_hidden_size': ['intermediate_size'],
    'num_attention_heads': ['num_attention_heads'],
    'num_query_groups': ['num_key_value_heads'],
    'max_position_embeddings': ['max_position_embeddings'],
    'max_text_length': ['max_text_length'],
    'layernorm_epsilon': ['rms_norm_eps'],
    'rotary_base': ['rope_theta'],
    'padded_vocab_size': ['padded_vocab_size', 'vocab_size'],
    'attention_dropout': ['attention_dropout'],
    'hidden_dropout': ['hidden_dropout'],
    'untie_embeddings_and_output_weights': ['tie_word_embeddings'],
    'swiglu': ['hidden_act'],
    'add_qkv_bias': ['attention_bias', 'qkv_bias', 'use_bias'],
    'add_bias_linear': ['mlp_bias'],
    'kv_channels': ['head_dim'],
    'hf_model_type': ['model_type'],
    # moe
    'moe_ffn_hidden_size': ['moe_intermediate_size'],
    'moe_shared_expert_intermediate_size': ['shared_expert_intermediate_size'],
    'moe_router_topk': ['num_experts_per_tok', 'moe_topk', 'moe_k'],
    'moe_router_num_groups': ['n_group'],
    'moe_router_group_topk': ['topk_group'],
    'num_moe_experts': ['num_experts', 'n_routed_experts', 'moe_num_experts', 'num_local_experts'],
    'moe_router_pre_softmax': ['norm_topk_prob'],
    # deepseek
    'q_lora_rank': ['q_lora_rank'],
    'kv_lora_rank': ['kv_lora_rank'],
    'moe_router_score_function': ['scoring_func', 'moe_router_use_sigmoid'],
    'moe_router_bias_update_rate': ['aux_loss_alpha'],
    'qk_head_dim': ['qk_nope_head_dim'],
    'qk_pos_emb_head_dim': ['qk_rope_head_dim'],
    'v_head_dim': ['v_head_dim'],
    'moe_router_topk_scaling_factor': ['routed_scaling_factor', 'router_scaling_factor'],
    'qk_layernorm': ['use_qk_norm', 'qk_norm'],
    # qwen3_next/qwen3_5
    'linear_attention_freq': ['full_attention_interval'],
    'linear_num_key_heads': ['linear_num_key_heads'],
    'linear_num_value_heads': ['linear_num_value_heads'],
    'linear_key_head_dim': ['linear_key_head_dim'],
    'linear_value_head_dim': ['linear_value_head_dim'],
    'linear_conv_kernel_dim': ['linear_conv_kernel_dim'],
    'linear_expand_v': ['expand_v'],
    'hsd_sequence_profile': ['hsd_sequence_profile'],
    'hsd_packed_document_isolation': ['hsd_packed_document_isolation'],
    'hsd_require_packed_seq_params': ['hsd_require_packed_seq_params'],
    'hsd_fixed_anchor_count': ['hsd_fixed_anchor_count'],
    'hsd_read_block_size': ['hsd_read_block_size'],
    'hsd_use_global_read': ['hsd_use_global_read'],
    'hsd_use_route_scale': ['hsd_use_route_scale'],
    'hsd_route_logit_scale': ['hsd_route_logit_scale'],
    'hsd_route_head_dim': ['hsd_route_head_dim'],
    'hsd_route_query_heads': ['hsd_route_query_heads'],
    'hsd_route_key_heads': ['hsd_route_key_heads'],
    'hsd_route_qk_l2norm': ['hsd_route_qk_l2norm'],
    'hsd_route_position_encoding': ['hsd_route_position_encoding'],
    'hsd_route_position_mode': ['hsd_route_position_mode'],
    'hsd_read_topk': ['hsd_read_topk'],
    'hsd_topk_backend': ['hsd_topk_backend'],
    'hsd_route_weight_momentum_gamma': ['hsd_route_weight_momentum_gamma'],
    'hsd_null_read_gate': ['hsd_null_read_gate'],
    'hsd_null_read_gate_init_bias': ['hsd_null_read_gate_init_bias'],
    'hsd_persistent_memory_enabled': ['hsd_persistent_memory_enabled'],
    'hsd_persistent_memory_route_key_enabled': ['hsd_persistent_memory_route_key_enabled'],
    'hsd_persistent_memory_route_key_mode': ['hsd_persistent_memory_route_key_mode'],
    'hsd_history_mix_mode': ['hsd_history_mix_mode'],
    'hsd_train_selective_scan': ['hsd_train_selective_scan'],
    'hsd_train_scan_backend': ['hsd_train_scan_backend'],
    'hsd_train_scan_strict': ['hsd_train_scan_strict'],
    'hsd_train_selective_scan_packed': ['hsd_train_selective_scan_packed'],
    'hsd_train_selective_scan_packed_strict': ['hsd_train_selective_scan_packed_strict'],
    'hsd_flashqla_tile_bwd_backend': ['hsd_flashqla_tile_bwd_backend'],
    'hsd_fused_global_read_backend': ['hsd_fused_global_read_backend'],
    'hsd_fused_global_read_strict': ['hsd_fused_global_read_strict'],
    'hsd_history_scale': ['hsd_history_scale'],
    'hsd_history_out_rmsnorm': ['hsd_history_out_rmsnorm'],
    'hsd_history_out_rmsnorm_eps': ['hsd_history_out_rmsnorm_eps'],
    'hsd_learnable_history_scale': ['hsd_learnable_history_scale'],
    'hsd_learnable_history_scale_init': ['hsd_learnable_history_scale_init'],
    'hsd_learnable_history_scale_max': ['hsd_learnable_history_scale_max'],
    'hsd_history_gate_schedule_start': ['hsd_history_gate_schedule_start'],
    'hsd_history_gate_schedule_end': ['hsd_history_gate_schedule_end'],
    'hsd_history_gate_schedule_steps': ['hsd_history_gate_schedule_steps'],
    'hsd_history_out_grad_scale': ['hsd_history_out_grad_scale'],
    'hsd_history_out_grad_scale_start': ['hsd_history_out_grad_scale_start'],
    'hsd_history_out_grad_scale_end': ['hsd_history_out_grad_scale_end'],
    'hsd_history_out_grad_scale_steps': ['hsd_history_out_grad_scale_steps'],
    'hsd_anchor_content_grad_scale': ['hsd_anchor_content_grad_scale'],
    'hsd_anchor_content_grad_scale_start': ['hsd_anchor_content_grad_scale_start'],
    'hsd_anchor_content_grad_scale_end': ['hsd_anchor_content_grad_scale_end'],
    'hsd_anchor_content_grad_scale_steps': ['hsd_anchor_content_grad_scale_steps'],
    'hsd_local_state_mode': ['hsd_local_state_mode'],
    'hsd_global_read_layers': ['hsd_global_read_layers'],
    'hsd_history_read_q_mode': ['hsd_history_read_q_mode'],
    'hsd_history_read_q_lora_rank': ['hsd_history_read_q_lora_rank'],
    'hsd_full_attention_anchor_tokens': ['hsd_full_attention_anchor_tokens'],
    'hsd_text_only_mlp': ['hsd_text_only_mlp'],
    'hsd_text_only_in_proj': ['hsd_text_only_in_proj'],
    'hsd_anchor_isolated_in_proj': ['hsd_anchor_isolated_in_proj'],
    'hsd_text_single_pass_delta': ['hsd_text_single_pass_delta'],
    'hsd_anchor_mlp': ['hsd_anchor_mlp'],
    'hsd_anchor_mlp_hidden_size': ['hsd_anchor_mlp_hidden_size'],
    'anchor_chunk_size': ['anchor_chunk_size'],
    'anchor_token_num': ['anchor_token_num'],
    'anchor_placeholder_token_id': ['anchor_placeholder_token_id'],
    'anchor_bos_token_id': ['anchor_bos_token_id'],
    'anchor_position_ids_type': ['anchor_position_ids_type'],
    'anchor_embedding_type': ['anchor_embedding_type'],
    'anchor_embedding_init_from': ['anchor_embedding_init_from'],
    'anchor_embedding_init_scale': ['anchor_embedding_init_scale'],
    'hsd_anchor_embedding_grad_scale': ['hsd_anchor_embedding_grad_scale'],
    'hsd_anchor_embedding_grad_scale_start': ['hsd_anchor_embedding_grad_scale_start'],
    'hsd_anchor_embedding_grad_scale_end': ['hsd_anchor_embedding_grad_scale_end'],
    'hsd_anchor_embedding_grad_scale_steps': ['hsd_anchor_embedding_grad_scale_steps'],
    'hsd_anchor_embedding_lr_mult': ['hsd_anchor_embedding_lr_mult'],
    'qwen35_hsd_replace_linear_attn': ['hist_replace_linear_attn', 'qwen35_hsd_replace_linear_attn'],
    'init_method_std': ['initializer_range'],
    # dsa
    'dsa_indexer_n_heads': ['index_n_heads'],
    'dsa_indexer_head_dim': ['index_head_dim'],
    'dsa_indexer_topk': ['index_topk'],
    'dsa_indexer_rotary_interleaved': ['indexer_rope_interleave'],
    # other
    'original_max_position_embeddings': ['original_max_position_embeddings'],
    'partial_rotary_factor': ['partial_rotary_factor'],
    'first_k_dense_replace': ['first_k_dense_replace', 'moe_layer_start_index'],
    'n_shared_experts': ['n_shared_experts', 'num_shared_expert', 'moe_num_shared_experts', 'num_shared_experts'],
    'window_size': ['sliding_window'],
    'layer_types': ['layer_types'],
    'interleave_moe_layer_step': ['interleave_moe_layer_step'],
}

HSD_CONFIG_KEYS = {
    'hsd_sequence_profile',
    'hsd_packed_document_isolation',
    'hsd_require_packed_seq_params',
    'hsd_fixed_anchor_count',
    'hsd_read_block_size',
    'hsd_use_global_read',
    'hsd_use_route_scale',
    'hsd_route_logit_scale',
    'hsd_route_head_dim',
    'hsd_route_query_heads',
    'hsd_route_key_heads',
    'hsd_route_qk_l2norm',
    'hsd_route_position_encoding',
    'hsd_route_position_mode',
    'hsd_read_topk',
    'hsd_topk_backend',
    'hsd_route_weight_momentum_gamma',
    'hsd_null_read_gate',
    'hsd_null_read_gate_init_bias',
    'hsd_persistent_memory_enabled',
    'hsd_persistent_memory_route_key_enabled',
    'hsd_persistent_memory_route_key_mode',
    'hsd_history_mix_mode',
    'hsd_train_selective_scan',
    'hsd_train_scan_backend',
    'hsd_train_scan_strict',
    'hsd_train_selective_scan_packed',
    'hsd_train_selective_scan_packed_strict',
    'hsd_flashqla_tile_bwd_backend',
    'hsd_fused_global_read_backend',
    'hsd_fused_global_read_strict',
    'hsd_history_scale',
    'hsd_history_out_rmsnorm',
    'hsd_history_out_rmsnorm_eps',
    'hsd_learnable_history_scale',
    'hsd_learnable_history_scale_init',
    'hsd_learnable_history_scale_max',
    'hsd_history_gate_schedule_start',
    'hsd_history_gate_schedule_end',
    'hsd_history_gate_schedule_steps',
    'hsd_history_out_grad_scale',
    'hsd_history_out_grad_scale_start',
    'hsd_history_out_grad_scale_end',
    'hsd_history_out_grad_scale_steps',
    'hsd_anchor_content_grad_scale',
    'hsd_anchor_content_grad_scale_start',
    'hsd_anchor_content_grad_scale_end',
    'hsd_anchor_content_grad_scale_steps',
    'hsd_local_state_mode',
    'hsd_global_read_layers',
    'hsd_history_read_q_mode',
    'hsd_history_read_q_lora_rank',
    'hsd_full_attention_anchor_tokens',
    'hsd_text_only_mlp',
    'hsd_text_only_in_proj',
    'hsd_anchor_isolated_in_proj',
    'hsd_text_single_pass_delta',
    'hsd_anchor_mlp',
    'hsd_anchor_mlp_hidden_size',
    'anchor_chunk_size',
    'anchor_token_num',
    'anchor_placeholder_token_id',
    'anchor_bos_token_id',
    'anchor_position_ids_type',
    'anchor_embedding_type',
    'anchor_embedding_init_from',
    'anchor_embedding_init_scale',
    'hsd_anchor_embedding_grad_scale',
    'hsd_anchor_embedding_grad_scale_start',
    'hsd_anchor_embedding_grad_scale_end',
    'hsd_anchor_embedding_grad_scale_steps',
    'hsd_anchor_embedding_lr_mult',
    'hist_replace_linear_attn',
    'qwen35_hsd_replace_linear_attn',
}

HSD_RUNTIME_CONFIG_ALIASES = {
    'hist_replace_linear_attn': 'qwen35_hsd_replace_linear_attn',
}


# Model identifiers are profile metadata, not TransformerConfig keyword args.
_MARCH_MODEL_VARIANTS = {
    'safin1_4b': 'safin1_4b',
    'safin1_4b_text': 'safin1_4b',
    'qwen3_5': 'safin1_4b',
    'qwen3_5_text': 'safin1_4b',
    'historical_qwen3_5': 'safin1_4b',
    'historical_qwen3_5_text': 'safin1_4b',
    'safin1_35b_a3b': 'safin1_35b_a3b',
    'safin1_35b_a3b_text': 'safin1_35b_a3b',
    'qwen3_5_moe': 'safin1_35b_a3b',
    'qwen3_5_moe_text': 'safin1_35b_a3b',
    'historical_qwen3_5_moe': 'safin1_35b_a3b',
    'historical_qwen3_5_moe_text': 'safin1_35b_a3b',
}


def _validate_march_profile_model_type(megatron_config: Dict[str, Any], profile: Dict[str, Any]) -> None:
    profile_type = profile.get('model_type')
    if profile_type is None:
        return  # Older runtime-only JSON files remain supported.
    if not isinstance(profile_type, str) or profile_type not in {'safin1_4b', 'safin1_35b_a3b'}:
        raise ValueError(f'Unsupported MARCH profile model_type: {profile_type!r}.')
    for key in ('hf_model_type', 'llm_model_type'):
        actual_type = megatron_config.get(key)
        if actual_type is not None and _MARCH_MODEL_VARIANTS.get(actual_type) != profile_type:
            raise ValueError(
                f'MARCH profile {profile_type!r} does not match {key}={actual_type!r}. '
                'Select the matching MODEL_PROFILE or MARCH_RUNTIME_CONFIG.')


def merge_hsd_runtime_config(megatron_config: Dict[str, Any], hsd_runtime_config=None) -> Dict[str, Any]:
    """Merge a standalone HSD JSON config into mcore config kwargs."""
    if hsd_runtime_config is None or hsd_runtime_config == '':
        validate_safin1_training_runtime(megatron_config)
        return megatron_config
    hsd_config = json_parse_to_dict(hsd_runtime_config)
    if not isinstance(hsd_config, dict):
        raise ValueError('hsd_runtime_config must be a JSON object or a path to one.')
    unknown_keys = sorted(set(hsd_config) - HSD_CONFIG_KEYS - {'model_type'})
    if unknown_keys:
        raise ValueError(f'Unknown HSD runtime config keys: {unknown_keys}. Supported keys: {sorted(HSD_CONFIG_KEYS)}.')
    _validate_march_profile_model_type(megatron_config, hsd_config)
    validate_safin1_training_runtime(hsd_config)
    merged = dict(megatron_config)
    for key in HSD_CONFIG_KEYS:
        if key in hsd_config and hsd_config[key] is not None:
            merged[HSD_RUNTIME_CONFIG_ALIASES.get(key, key)] = hsd_config[key]
    validate_safin1_training_runtime(merged)
    for key in HSD_CONFIG_KEYS:
        if key in hsd_config and hsd_config[key] is not None:
            megatron_config[HSD_RUNTIME_CONFIG_ALIASES.get(key, key)] = hsd_config[key]
    megatron_config['hsd_runtime_config'] = hsd_runtime_config
    return megatron_config


def _linear_attention_freq_from_layer_types(layer_types):
    if not layer_types:
        return None

    flags = []
    for layer_type in layer_types:
        if layer_type == 'linear_attention':
            flags.append(1)
        elif layer_type == 'full_attention':
            flags.append(0)
        else:
            return None

    full_layers = [i + 1 for i, flag in enumerate(flags) if flag == 0]
    if full_layers:
        freq = full_layers[0]
        full_layers_match = full_layers == list(range(freq, len(flags) + 1, freq))
        if full_layers_match:
            return freq

    return f"[{','.join(str(flag) for flag in flags)}]"


def _convert_config(config, _internal_call=False) -> Dict[str, Any]:
    megatron_config = {}
    for k, hf_keys in config_mapping.items():
        for hf_k in hf_keys:
            if hasattr(config, hf_k):
                hf_v = getattr(config, hf_k)
                if hf_v is None:
                    continue
                if k == 'rotary_base':
                    megatron_config[k] = int(hf_v)
                elif k in {'untie_embeddings_and_output_weights', 'moe_router_pre_softmax'}:
                    megatron_config[k] = not hf_v
                elif k == 'swiglu':
                    if hf_v == 'silu':
                        megatron_config[k] = True
                elif hf_k == 'moe_router_use_sigmoid':
                    if hf_v:
                        megatron_config[k] = 'sigmoid'
                    else:
                        continue
                else:
                    if k == 'kv_lora_rank':
                        megatron_config['multi_latent_attention'] = True
                    elif k == 'hf_model_type':
                        if _internal_call:
                            k = 'llm_model_type'
                    megatron_config[k] = hf_v
                break
    for key in ['text_config', 'llm_config', 'thinker_config']:
        if hasattr(config, key):
            megatron_config.update(_convert_config(getattr(config, key), _internal_call=True))
    # compat llama3
    if getattr(config, 'rope_scaling', None) is not None:
        if isinstance(config.rope_scaling, int):
            megatron_config['rope_scaling'] = {'factor': config.rope_scaling, 'type': 'linear'},
        elif isinstance(config.rope_scaling, dict):
            megatron_config['rope_scaling'] = config.rope_scaling
    return megatron_config


def hf_to_mcore_config(hf_config: PretrainedConfig) -> Dict[str, Any]:
    res = _convert_config(hf_config)
    res['hf_config'] = hf_config
    hf_model_type = res.get('hf_model_type')
    llm_model_type = res.get('llm_model_type') or hf_model_type
    res['llm_model_type'] = llm_model_type
    hsd_model_types = {
        'safin1_4b',
        'safin1_35b_a3b',
        'safin1_4b_text',
        'safin1_35b_a3b_text',
    }
    should_merge_hsd_runtime = (
        get_env_args('USE_HIST_LINEAR_ATTN', bool, False)
        or hf_model_type in hsd_model_types
        or llm_model_type in hsd_model_types)
    if should_merge_hsd_runtime:
        res = merge_hsd_runtime_config(res, get_env_args('HSD_RUNTIME_CONFIG', str, None))
    qwen35_dense_model_types = {'qwen3_5', 'safin1_4b', 'safin1_4b_text'}
    qwen35_moe_model_types = {'qwen3_5_moe', 'safin1_35b_a3b'}
    qwen35_gdn_model_types = qwen35_dense_model_types | qwen35_moe_model_types

    first_k_dense_replace = res.pop('first_k_dense_replace', None)
    n_shared_experts = res.pop('n_shared_experts', None)
    layer_types = res.pop('layer_types', None)
    mlp_ffn_hidden_size = res.pop('mlp_ffn_hidden_size', None)
    interleave_moe_layer_step = res.pop('interleave_moe_layer_step', None)
    window_size = res.pop('window_size', None)
    rope_scaling = res.get('rope_scaling') or {}
    qwen_qk_layernorm_types = {
        'qwen3_omni_moe', 'qwen3_omni', 'qwen3_vl',
        'qwen3_vl_moe', 'llavaonevision1_5'
    } | qwen35_dense_model_types
    if (
            llm_model_type in {'qwen3', 'qwen3_moe', 'qwen3_next'}
            or hf_model_type in qwen_qk_layernorm_types
            or llm_model_type in qwen_qk_layernorm_types):
        res['qk_layernorm'] = True
    if hf_model_type in qwen35_moe_model_types:
        res['qk_layernorm'] = True
    qwen_moe_types = {'qwen3_omni_moe', 'qwen3_vl_moe'} | qwen35_moe_model_types
    if llm_model_type in {'qwen2_moe', 'qwen3_moe', 'qwen3_next'} or hf_model_type in qwen_moe_types:
        res.pop('ffn_hidden_size', None)
        if llm_model_type in {'qwen2_moe', 'qwen3_next'} or hf_model_type in qwen35_moe_model_types:
            res['moe_shared_expert_gate'] = True
    if llm_model_type in {'deepseek', 'deepseek_v2', 'deepseek_v3', 'kimi_k2', 'deepseek_v32', 'dots1'
                            } or hf_model_type == 'kimi_vl':
        if llm_model_type != 'deepseek':
            res['qk_layernorm'] = True
        res['moe_router_load_balancing_type'] = 'seq_aux_loss'
        res.pop('num_query_groups', None)  # https://github.com/NVIDIA/Megatron-LM/issues/1475
        if llm_model_type == 'dots1':
            res['moe_router_score_function'] = 'sigmoid'
        elif llm_model_type == 'deepseek_v32':
            res['experimental_attention_variant'] = 'dsa'
    elif llm_model_type == 'hunyuan':
        # Since HunYuan’s attention applies RoPE before using q/k_layernorm,
        # which is incompatible with megatron-core, support is not provided here.
        res['n_shared_experts'] = n_shared_experts
        for key in ['moe_ffn_hidden_size', 'n_shared_experts', 'moe_router_topk']:
            val = res.get(key)
            if isinstance(val, list) and val and min(val) == max(val):
                res[key] = val[0]
        n_shared_experts = res.pop('n_shared_experts')
    elif llm_model_type in {'ernie4_5', 'ernie4_5_moe', 'glm4'}:
        res['rotary_interleaved'] = True
    elif llm_model_type == 'gpt_oss':
        res['add_bias_linear'] = True
        res['bias_dropout_fusion'] = False
        res['softmax_type'] = 'learnable'
        res['swiglu'] = False
        res['quick_geglu'] = True
        res['activation_func_clamp_value'] = 7
        res['glu_linear_offset'] = 1
        res['window_size'] = f'{window_size},0'
        if layer_types is None:
            res['window_attn_skip_freq'] = '2'
        else:
            window_attn_skip_freq = ','.join(['1' if lt == 'sliding_attention' else '0' for lt in layer_types])
            res['window_attn_skip_freq'] = f'[{window_attn_skip_freq}]'
    elif llm_model_type in {'glm4_moe', 'glm4_moe_lite', 'glm_moe_dsa'} or hf_model_type == 'glm4v_moe':
        res['moe_router_score_function'] = 'sigmoid'
        if llm_model_type in {'glm4_moe_lite', 'glm_moe_dsa'}:
            res['qk_layernorm'] = True
            res.pop('num_query_groups', None)
        if llm_model_type == 'glm_moe_dsa':
            res['experimental_attention_variant'] = 'dsa'
    elif (
            llm_model_type == 'qwen3_next' or hf_model_type in qwen35_gdn_model_types
            or llm_model_type in qwen35_gdn_model_types):
        use_mcore_gdn = get_env_args('USE_MCORE_GDN', bool, True)
        res['layernorm_zero_centered_gamma'] = True
        res['attention_output_gate'] = True
        if use_mcore_gdn and llm_model_type != 'qwen3_next':
            res['experimental_attention_variant'] = 'gated_delta_net'
        res.setdefault('linear_attention_freq', _linear_attention_freq_from_layer_types(layer_types) or 4)
    elif llm_model_type == 'minimax_m2':
        res['add_qkv_bias'] = False
    elif hf_model_type == 'llama4':
        qk_layernorm = res.pop('qk_layernorm', False)
        if qk_layernorm:
            res['qk_l2_norm'] = True
        res['no_rope_freq'] = 4
        res['moe_apply_probs_on_input'] = True
        res['rotary_interleaved'] = True
        res['moe_router_score_function'] = 'sigmoid'
        res['moe_ffn_hidden_size'] = res['ffn_hidden_size']
        res['ffn_hidden_size'] = mlp_ffn_hidden_size
        res['moe_router_enable_expert_bias'] = False
        res['moe_shared_expert_intermediate_size'] = res['moe_ffn_hidden_size']
        if interleave_moe_layer_step > 1:
            moe_layer_freq = [
                '1' if i % interleave_moe_layer_step == (interleave_moe_layer_step - 1) else '0'
                for i in range(res['num_layers'])
            ]
            res['moe_layer_freq'] = f"[{','.join(moe_layer_freq)}]"
    elif hf_model_type == 'glm4v':
        res['rotary_interleaved'] = True

    if 'partial_rotary_factor' not in res and 'partial_rotary_factor' in rope_scaling:
        res['partial_rotary_factor'] = rope_scaling['partial_rotary_factor']
    if 'rotary_base' not in res and 'rope_theta' in rope_scaling:
        res['rotary_base'] = rope_scaling['rope_theta']
    if rope_scaling.get('mrope_section') is not None:
        res['position_embedding_type'] = 'mrope'
        res['mrope_section'] = rope_scaling['mrope_section']
        mrope_interleaved = rope_scaling.get('mrope_interleaved', False) or rope_scaling.get('interleaved', False)
        res['mrope_interleaved'] = mrope_interleaved

    if first_k_dense_replace is not None:
        res['moe_layer_freq'] = f'[0]*{first_k_dense_replace}+[1]*{res["num_layers"] - first_k_dense_replace}'
    if res.get('moe_router_score_function', 'softmax') == 'sigmoid' and 'moe_router_enable_expert_bias' not in res:
        res['moe_router_enable_expert_bias'] = True
    if n_shared_experts is not None and 'moe_shared_expert_intermediate_size' not in res:
        res['moe_shared_expert_intermediate_size'] = n_shared_experts * res['moe_ffn_hidden_size']
    return res

# Copyright (c) ModelScope Contributors. All rights reserved.
import concurrent.futures
import importlib.metadata
import inspect
import json
import logging
import os
import torch
import torch.distributed as dist
from contextlib import contextmanager
from copy import copy
from packaging import version
from tqdm import tqdm
from transformers.modeling_utils import custom_object_save
from transformers.utils import is_torch_npu_available
from transformers.utils.versions import require_version

from swift.model import get_model_processor, save_checkpoint
from swift.utils import get_logger, get_modules_to_not_convert, get_multimodal_target_regex, is_master, split_list
from .utils.persistent_memory_artifact import export_persistent_memory_artifact

logger = get_logger()

HSD_HF_CONFIG_KEYS = (
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
)


def _env_enabled(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.strip().lower() in {'1', 'true', 'yes', 'on'}


def _has_hsd_runtime_config(args, config) -> bool:
    if _env_enabled('USE_HIST_LINEAR_ATTN'):
        return True
    runtime = getattr(args, 'hsd_runtime_config', None)
    if runtime is not None and runtime != '':
        return True
    return int(getattr(config, 'anchor_chunk_size', 0) or 0) > 0 and int(
        getattr(config, 'anchor_token_num', 0) or 0) > 0


def _get_hsd_hf_config_target(hf_config):
    return getattr(hf_config, 'text_config', None) or getattr(hf_config, 'llm_config', None) or hf_config


def _collect_hsd_runtime_config(config) -> dict:
    from march.safin1.runtime_validation import validate_training_environment
    validate_training_environment(config=config)
    runtime_config = {}
    for key in HSD_HF_CONFIG_KEYS:
        if not hasattr(config, key):
            continue
        value = getattr(config, key)
        if value is None:
            continue
        runtime_config[key] = value
    if hasattr(config, 'qwen35_hsd_replace_linear_attn'):
        runtime_config['hist_replace_linear_attn'] = bool(getattr(config, 'qwen35_hsd_replace_linear_attn'))
    return runtime_config


def _save_hsd_runtime_config_to_hf_config(hf_config, config, args) -> None:
    if not _has_hsd_runtime_config(args, config):
        return

    target_config = _get_hsd_hf_config_target(hf_config)
    runtime_config = _collect_hsd_runtime_config(config)
    for key, value in runtime_config.items():
        setattr(target_config, key, value)

    if runtime_config:
        logger.info(f'Saved HSD runtime config fields into HF config: {sorted(runtime_config)}')


def _finalize_hsd_hf_config_file(output_dir, args, config) -> None:
    if not _has_hsd_runtime_config(args, config):
        return
    config_path = os.path.join(output_dir, 'config.json')
    if not os.path.exists(config_path):
        raise RuntimeError(
            f'Safin-1 HSD config finalizer cannot find `{config_path}`; '
            'refusing to leave an incomplete checkpoint.')

    try:
        from march.safin1.config_utils import (
            copy_safin1_remote_code,
            safin1_remote_code_files,
            is_safin1_config,
            normalize_safin1_config,
        )
    except Exception as e:
        raise RuntimeError(
            'Failed to import the Safin-1 config finalizer required '
            f'for an HSD checkpoint: {e}') from e

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    text_config = config_dict.get('text_config') or {}
    runtime_config = {key: text_config.get(key) for key in HSD_HF_CONFIG_KEYS if text_config.get(key) is not None}
    runtime_config.update(_collect_hsd_runtime_config(config))
    if not runtime_config:
        raise RuntimeError(
            f'HSD HF config finalizer was requested for `{output_dir}`, but no HSD runtime fields were collected.')
    normalized = normalize_safin1_config(config_dict, runtime_config)
    if not is_safin1_config(normalized):
        raise RuntimeError(f'Safin-1 finalizer produced a non-HSD config for `{output_dir}`.')
    model_type = normalized.get('model_type')
    text_model_type = (normalized.get('text_config') or {}).get('model_type')
    tmp_config_path = f'{config_path}.tmp.{os.getpid()}'
    with open(tmp_config_path, 'w', encoding='utf-8') as f:
        json.dump(normalized, f, indent=2, sort_keys=True)
        f.write('\n')
    os.replace(tmp_config_path, config_path)
    copied = copy_safin1_remote_code(output_dir, model_type=model_type)
    expected_remote_code = set(safin1_remote_code_files(model_type))
    missing_remote_code = sorted(
        filename for filename in expected_remote_code
        if not os.path.isfile(os.path.join(output_dir, filename))
        or os.path.getsize(os.path.join(output_dir, filename)) <= 0
    )
    if missing_remote_code:
        raise RuntimeError(
            f'Safin-1 remote-code copy is incomplete for `{output_dir}`: '
            f'missing_or_empty={missing_remote_code}, copied={sorted(copied)}.')

    with open(config_path, 'r', encoding='utf-8') as f:
        saved_config = json.load(f)
    saved_text_config = saved_config.get('text_config') or {}
    missing_required = [
        key for key in ('anchor_chunk_size', 'anchor_token_num')
        if saved_text_config.get(key) is None
    ]
    if (saved_config.get('model_type') != model_type
            or saved_text_config.get('model_type') != text_model_type
            or not is_safin1_config(saved_config)
            or missing_required):
        raise RuntimeError(
            f'Safin-1 config finalization did not persist for `{output_dir}`: '
            f'model_type={saved_config.get("model_type")!r}, '
            f'text_model_type={saved_text_config.get("model_type")!r}, '
            f'missing_required={missing_required}.')
    hsd_field_count = sum(1 for key in saved_text_config if key.startswith('hsd_') or key.startswith('anchor_'))
    logger.info(
        f'Finalized Safin-1 HF config at {config_path}; '
        f'model_type={model_type}; text_model_type={text_model_type}; '
        f'hsd_field_count={hsd_field_count}; copied_remote_code={sorted(copied)}')


def _save_hsd_hf_metadata_without_dummy_model(output_dir, hf_config, config, args, processor) -> None:
    if config.mtp_num_layers:
        hf_config.num_nextn_predict_layers = config.mtp_num_layers
    if config.fp8 is not None and config.fp8_recipe == 'blockwise' and config.fp8_param:
        raise RuntimeError('HSD lightweight HF metadata save does not support fp8 blockwise param export.')
    if hasattr(hf_config, 'quantization_config'):
        del hf_config.quantization_config

    _save_hsd_runtime_config_to_hf_config(hf_config, config, args)
    hf_config.save_pretrained(output_dir)

    model_meta = getattr(processor, 'model_meta', None)
    additional_saved_files = getattr(model_meta, 'additional_saved_files', None)
    save_checkpoint(
        None,
        processor,
        output_dir,
        model_dirs=[args.model_dir],
        additional_saved_files=additional_saved_files)
    _finalize_hsd_hf_config_file(output_dir, args, config)
    logger.info(
        f'Successfully saved HSD HF metadata in `{output_dir}` without constructing a dummy HF model.')


def _patch__batched_p2p_ops():
    from megatron.core.pipeline_parallel import p2p_communication

    _batched_p2p_ops_origin = p2p_communication._batched_p2p_ops

    def _batched_p2p_ops(**kwargs):
        kwargs['group'] = None
        return _batched_p2p_ops_origin(**kwargs)

    p2p_communication._batched_p2p_ops = _batched_p2p_ops


def _patch_torch_FileSystemReader():
    from torch.distributed.checkpoint.filesystem import FileSystemReader
    from torch.futures import Future
    _origin_read_data = FileSystemReader.read_data
    _origin__slice_file = FileSystemReader._slice_file
    READER_MAX_WORKERS = int(os.environ.get('MCORE_READER_MAX_WORKERS', '16'))

    @contextmanager
    def _patch__slice_file(prog_bar):

        def _slice_file(self, *args, **kwargs):
            prog_bar.update()
            return _origin__slice_file(self, *args, **kwargs)

        FileSystemReader._slice_file = _slice_file
        try:
            yield
        finally:
            FileSystemReader._slice_file = _origin__slice_file

    def read_data(self, plan, planner):

        def _worker(plan_shard):
            _origin_read_data(self, plan_shard, planner)

        prog_bar = tqdm(total=len(plan.items), dynamic_ncols=True, desc='Loading: ')
        plan_shards = split_list(plan.items, READER_MAX_WORKERS, contiguous=False)
        with _patch__slice_file(prog_bar):
            with concurrent.futures.ThreadPoolExecutor(max_workers=READER_MAX_WORKERS) as pool:
                futures = []
                for i in range(READER_MAX_WORKERS):
                    plan_shard = copy(plan)
                    plan_shard.items = plan_shards[i]
                    futures.append(pool.submit(_worker, plan_shard))
                concurrent.futures.wait(futures)
        prog_bar.close()
        fut: Future = Future()
        fut.set_result(None)
        return fut

    FileSystemReader.read_data = read_data


def _patch_validate_non_overlapping_shards_metadata():
    # too slow
    from torch.distributed._shard.sharded_tensor import api
    from torch.distributed._shard.sharding_spec import api as api2
    from torch.distributed.checkpoint import default_planner

    def validate_non_overlapping_shards_metadata(*args, **kwargs):
        pass

    api.validate_non_overlapping_shards_metadata = validate_non_overlapping_shards_metadata
    api2.validate_non_overlapping_shards_metadata = validate_non_overlapping_shards_metadata

    def _validate_global_plan(*args, **kwargs):
        return True

    default_planner._validate_global_plan = _validate_global_plan


def _patch__write_item():
    import megatron.core
    if version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0'):
        return
    # mcore 0.12
    from megatron.core.dist_checkpointing.strategies import filesystem_async

    _origin__write_item = filesystem_async._write_item
    if 'serialization_format' in inspect.signature(_origin__write_item).parameters:
        from torch.distributed.checkpoint.filesystem import SerializationFormat

        def _write_item(self, *args, **kwargs):
            if 'serialization_format' not in kwargs:
                kwargs['serialization_format'] = SerializationFormat.TORCH_SAVE
            return _origin__write_item(self, *args, **kwargs)

        filesystem_async._write_item = _write_item


def _patch_unified_memory():
    if is_torch_npu_available():
        return

    mcore_015 = version.parse(importlib.metadata.version('megatron-core')) >= version.parse('0.15.0rc0')
    if not mcore_015:
        return
    from torch.utils import cpp_extension
    load_inline = cpp_extension.load_inline

    def _new_load_inline(*args, **kwargs):
        name = kwargs.get('name')
        if name == 'managed_alloc_runtime':
            raise RuntimeError
        return load_inline(*args, **kwargs)

    # not create unified memory mempool
    cpp_extension.load_inline = _new_load_inline
    try:
        from megatron.core.inference import unified_memory  # noqa: F401
    except Exception:
        pass
    finally:
        cpp_extension.load_inline = load_inline


def _patch_mcore_bridge():
    require_version('mcore-bridge>=1.0.2', 'please install mcore-bridge via `pip install mcore-bridge -U`')
    import mcore_bridge
    from mcore_bridge import GPTBridge
    logger.info(f'mcore_bridge.__version__: {mcore_bridge.__version__}')
    origin_save_weights = GPTBridge.save_weights

    def save_weights(
        self,
        mg_models,
        output_dir: str,
        peft_format: bool = False,
        max_shard_size: str = '5GB',
        args=None,
        processor=None,
    ) -> None:
        origin_save_weights(self, mg_models, output_dir, peft_format=peft_format, max_shard_size=max_shard_size)
        if (not peft_format and bool(getattr(self.config, 'hsd_persistent_memory_enabled', False))):
            artifact_error = [None]
            artifact_metadata = None
            if is_master():
                try:
                    artifact_metadata = export_persistent_memory_artifact(output_dir, self.config)
                except Exception as error:
                    artifact_error[0] = f'{type(error).__name__}: {error}'
            if dist.is_available() and dist.is_initialized():
                dist.broadcast_object_list(artifact_error, src=0)
            if artifact_error[0] is not None:
                raise RuntimeError(
                    f'Failed to export persistent HSD memory artifact: {artifact_error[0]}')
            if is_master():
                logger.info(
                    'Saved persistent HSD memory artifact: '
                    f'tensors={artifact_metadata["tensor_count"]}, '
                    f'parameters={artifact_metadata["parameter_count"]}, '
                    f'output_dir={output_dir}')
        if processor is None or args is None:
            return
        hf_config = self.config.hf_config
        hf_config = copy(hf_config)
        if _has_hsd_runtime_config(args, self.config) and not peft_format:
            if is_master():
                logger.info('Using lightweight HSD HF metadata save; skip dummy HF model construction.')
                _save_hsd_hf_metadata_without_dummy_model(output_dir, hf_config, self.config, args, processor)
            dist.barrier()  # Ensure all metadata files are saved completely
            return

        if is_master() and not hasattr(self, 'hf_model'):
            if hasattr(self, 'get_hf_meta_model'):
                self.hf_model = self.get_hf_meta_model()
                self.hf_model.model_meta = processor.model_meta
                self.hf_model.model_info = processor.model_info
            else:
                with torch.device('meta'):
                    self.hf_model = get_model_processor(
                        args.model_dir, model_type=args.model_type, return_dummy_model=True)[0]

        if is_master():
            if peft_format:
                peft_config = copy(mg_models[0].peft_config[self._adapter_name])
                if self.config.task_type == 'seq_cls':
                    peft_config.task_type = 'SEQ_CLS'
                if self.is_multimodal and 'all-linear' in args.target_modules:
                    peft_config.target_modules = get_multimodal_target_regex(
                        self.hf_model,
                        freeze_llm=args.freeze_llm,
                        freeze_vit=args.freeze_vit,
                        freeze_aligner=args.freeze_aligner,
                        include_embedding='all-embedding' in args.target_modules,
                        exclude_router='all-router' not in args.target_modules)
                else:
                    assert not isinstance(peft_config.target_modules, str), (
                        'target_regex is not currently supported for LoRA conversion. Please set `--merge_lora true`.')
                    peft_config.target_modules = self._peft_target_modules
                peft_config.modules_to_save = self._peft_modules_to_save
                peft_config.save_pretrained(output_dir)
            else:
                config = self.config
                if config.mtp_num_layers:
                    hf_config.num_nextn_predict_layers = config.mtp_num_layers
                if config.fp8 is not None and config.fp8_recipe == 'blockwise' and config.fp8_param:
                    if getattr(hf_config, 'quantization_config', None) is None:
                        from transformers.utils.quantization_config import FineGrainedFP8Config
                        modules_to_not_convert = get_modules_to_not_convert(self.hf_model)
                        if hasattr(self, '_fp8_skip_modules'):
                            modules_to_not_convert = (modules_to_not_convert or []) + list(self._fp8_skip_modules)
                        hf_config.quantization_config = FineGrainedFP8Config(
                            modules_to_not_convert=modules_to_not_convert)
                elif hasattr(hf_config, 'quantization_config'):
                    del hf_config.quantization_config
                _save_hsd_runtime_config_to_hf_config(hf_config, config, args)
                hf_config.save_pretrained(output_dir)
                if getattr(self.hf_model, '_auto_class') is not None:
                    try:
                        custom_object_save(self.hf_model, output_dir, config=hf_config)
                    except FileNotFoundError as e:
                        logger.error(f'custom_object_save Error: {e}')
                save_checkpoint(
                    None,
                    processor,
                    output_dir,
                    model_dirs=[args.model_dir],
                    additional_saved_files=self.hf_model.model_meta.additional_saved_files)
                _finalize_hsd_hf_config_file(output_dir, args, config)
            logger.info(f'Successfully saved `safetensors` model weights in `{output_dir}`.')
        dist.barrier()  # Ensure all weights are saved completely

    GPTBridge.save_weights = save_weights


def init_megatron_env():
    os.environ.pop('VLLM_USE_MODELSCOPE', None)
    logging_level = logging.root.level
    _patch_unified_memory()
    _patch_mcore_bridge()
    _patch__batched_p2p_ops()
    _patch__write_item()
    logging.root.setLevel(logging_level)  # revert logger level
    try:
        _patch_torch_FileSystemReader()
    except Exception:
        logger.warning('Failed to patch FileSystemReader.')
    try:
        _patch_validate_non_overlapping_shards_metadata()
    except Exception:
        logger.warning('Patch validate_non_overlapping_shards_metadata failed.')
        pass
    import megatron.core
    logger.info(f'megatron.core.__version__: {megatron.core.__version__}')

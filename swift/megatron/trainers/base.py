# Copyright (c) ModelScope Contributors. All rights reserved.
import dataclasses
import gc
import importlib
import json
import logging
import math
import megatron.core
import operator
import os
import re
import shutil
import time
import torch
import torch.nn
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from functools import partial
from mcore_bridge import LoraParallelLinear
from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import track_moe_metrics
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from modelscope import check_local_model_is_latest
from packaging import version
from pathlib import Path
from typing import Callable, Dict, List, Optional

from swift.dataset import RowPreprocessor
from swift.megatron.callbacks import megatron_callbacks_map
from swift.megatron.model import get_mcore_model
from swift.megatron.utils import (apply_router_replay_patch, disable_forward_pre_hook, enable_forward_pre_hook,
                                  get_optimizer_param_scheduler, get_padding_to, init_persistent_async_worker,
                                  initialize_tp_communicators, load_mcore_checkpoint,
                                  logical_and_across_model_parallel_group, maybe_finalize_async_save,
                                  prepare_mcore_model, reduce_max_stat_across_model_parallel_group,
                                  save_mcore_checkpoint, should_disable_forward_pre_hook, warmup_jit_function,
                                  wrap_model)
from swift.megatron.utils.lora_export import temporary_merged_lora
from swift.template import Template
from swift.trainers import dynamic_gradient_checkpointing
from swift.trainers.utils import patch_modelscope_hub_timeout
from swift.utils import deep_getattr, get_last_valid_indices, get_logger, is_last_rank, is_master, ms_logger_context
from .batch_sampler import MegatronPretrainingRandomSampler, MegatronPretrainingSampler
from .utils import (TrainerState, build_streaming_dataloader, get_batch_on_this_cp_rank, get_batch_on_this_pp_rank,
                    get_packed_seq_params)

try:
    from megatron.core.optimizer import param_group_identifier_keys
    from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
except ImportError:
    param_group_identifier_keys = None
    RouterReplay = None
    RouterReplayAction = None

mcore_016 = version.parse(megatron.core.__version__) >= version.parse('0.16.0rc0')

logger = get_logger()


class BaseMegatronTrainer(ABC):

    def __init__(self, args, template: Template):
        # validate mcore version and patch routing_replay
        self.enable_routing_replay = args.router_replay_mode != 'disabled'
        if self.enable_routing_replay:
            apply_router_replay_patch()

        self.args = args
        self.template = template
        self._torch_profiler = None
        self._torch_profiler_rank = None
        self.prepare_model()
        # Sync template.padding_free after prepare_model(), because _check_padding_free
        # may override args.padding_free for certain models (e.g. DSA attention).
        if template.padding_free != args.padding_free:
            logger.warning(f'template.padding_free({template.padding_free}) != args.padding_free({args.padding_free}), '
                           f'syncing template.padding_free to {args.padding_free}.')
            template.padding_free = args.padding_free
        self.optimizer, self.opt_param_scheduler = self.get_optimizer_and_scheduler()
        self.data_collator = self._get_data_collator()
        self._clear_dataloader_timing_metrics()

        self.state = TrainerState(max_steps=args.train_iters)
        initialize_embedding = args.new_special_tokens or args.task_type == 'seq_cls'
        if initialize_embedding:
            for m in self.unwrapped_models:
                self._initialize_embedding(m)
        self._load_checkpoint()

        self.eval_metrics = None
        if args.check_model and hasattr(args, 'model_dir'):
            with ms_logger_context(logging.CRITICAL), patch_modelscope_hub_timeout():
                config_info = self._collect_config_info()
                config_info.update({
                    'invoked_by': 'local_trainer',
                    'third_party': 'swift',
                    'trainer_class': self.__class__.__name__,
                    'trainer_backend': 'megatron',
                })
                check_local_model_is_latest(args.model_info.model_dir, user_agent=config_info)

        self.mcore_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')
        self.mcore_014 = version.parse(megatron.core.__version__) >= version.parse('0.14.0rc0')
        self.callbacks = []
        for callback in args.callbacks:
            self.callbacks.append(megatron_callbacks_map[callback](self))

        if args.tp_comm_overlap:
            initialize_tp_communicators(args, self.config)

        warmup_jit_function(self.config, args)

        if args.async_save and args.use_persistent_ckpt_worker:
            init_persistent_async_worker()

    def _load_checkpoint(self):
        args = self.args
        # A stage transition deliberately resets the source iteration/data
        # cursor.  Do not pre-load counters from ``args.model`` before the
        # MCore transition loader installs the source model/optimizer state.
        checkpoint_lifecycle = getattr(args, 'checkpoint_lifecycle', None) or 'resume'
        if not args.finetune and checkpoint_lifecycle not in {'transition', 'hf_bootstrap'}:
            self.state.iteration = self._load_iteration()
        if checkpoint_lifecycle == 'hf_bootstrap':
            if args.mcore_model is not None:
                raise ValueError('checkpoint_lifecycle=hf_bootstrap forbids mcore_model')
            if args.hf_model_weights is None:
                raise ValueError('checkpoint_lifecycle=hf_bootstrap requires hf_model_weights')
            if args.tuner_type != 'full':
                raise ValueError('checkpoint_lifecycle=hf_bootstrap only supports tuner_type=full')
            if args.finetune:
                raise ValueError('checkpoint_lifecycle=hf_bootstrap requires finetune=false')
            if args.no_save_optim:
                raise ValueError('checkpoint_lifecycle=hf_bootstrap requires no_save_optim=false')
            args.consumed_train_samples = 0
            args.consumed_valid_samples = 0
            self.state.iteration = 0
        loaded_checkpoint = False
        checkpoint_has_anchor_embedding = getattr(self, '_base_checkpoint_has_hsd_anchor_embedding', False)
        if args.mcore_model is not None:
            loaded_checkpoint = True
            self.state.iteration, model_missing_keys = load_mcore_checkpoint(
                args,
                self.wrapped_models,
                self.optimizer,
                self.opt_param_scheduler,
                load_arg='mcore_model',
                return_missing_keys=True)
            checkpoint_has_anchor_embedding = not self._keys_include_hsd_anchor_embedding(model_missing_keys)
        if args.mcore_adapter is not None:
            loaded_checkpoint = True
            self.state.iteration, _ = load_mcore_checkpoint(
                args,
                self.wrapped_models,
                self.optimizer,
                self.opt_param_scheduler,
                load_arg='mcore_adapter',
                return_missing_keys=True)
        if loaded_checkpoint:
            self._maybe_initialize_hsd_anchor_embeddings_after_load(
                checkpoint_has_anchor_embedding=checkpoint_has_anchor_embedding,
                source='mcore_checkpoint')
        self.state.consumed_train_samples = getattr(args, 'consumed_train_samples', 0)
        if checkpoint_lifecycle == 'hf_bootstrap':
            logger.info(
                'HF bootstrap loaded model weights before optimizer construction and started fresh '
                'optimizer/master parameters, scheduler, RNG, iteration, and sample counters.')

    @staticmethod
    def _is_hsd_anchor_embedding_key(key):
        return str(key).endswith('anchor_embedding.weight')

    @classmethod
    def _keys_include_hsd_anchor_embedding(cls, keys):
        return any(cls._is_hsd_anchor_embedding_key(key) for key in keys)

    def _hf_checkpoint_has_hsd_anchor_embedding(self, model_dir) -> bool:
        model_dir = Path(model_dir)
        for index_name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
            index_path = model_dir / index_name
            if not index_path.exists():
                continue
            try:
                with open(index_path, 'r') as f:
                    index = json.load(f)
            except Exception:
                continue
            return self._keys_include_hsd_anchor_embedding((index.get('weight_map') or {}).keys())

        try:
            from safetensors import safe_open
        except ImportError:
            return False

        for tensor_path in sorted(model_dir.glob('*.safetensors')):
            try:
                with safe_open(str(tensor_path), framework='pt', device='cpu') as f:
                    if self._keys_include_hsd_anchor_embedding(f.keys()):
                        return True
            except Exception:
                continue
        return False

    def _maybe_initialize_hsd_anchor_embeddings_after_load(self, checkpoint_has_anchor_embedding: bool, source: str):
        initialized = 0
        for model in self.unwrapped_models:
            seen = set()
            for module in model.modules():
                if id(module) in seen:
                    continue
                seen.add(id(module))
                init_fn = getattr(module, 'maybe_initialize_hsd_anchor_embedding_after_load', None)
                if init_fn is None:
                    continue
                if init_fn(checkpoint_has_anchor_embedding=checkpoint_has_anchor_embedding):
                    initialized += 1
        if initialized:
            logger.info(
                f'Post-load initialized HSD anchor_embedding from {source}; '
                f'checkpoint_has_anchor_embedding={checkpoint_has_anchor_embedding}, model_chunks={initialized}.')

    def call_event(self, event, **kwargs):
        for callback in self.callbacks:
            getattr(callback, event)(**kwargs)

    def on_log(self, logs, prefix=''):
        n_steps = logs.pop('n_steps')
        self._log_callback(logs, n_steps)
        if prefix:
            logs = {f'{prefix}{k}': v for k, v in logs.items()}
        self.call_event('on_log', logs=logs)

    def _log_callback(self, logs, n_steps):
        args = self.args
        config = self.config
        if config.num_moe_experts is not None:
            moe_loss_scale = 1 / args.num_microbatches / n_steps
            track_names = []
            if config.moe_router_load_balancing_type == 'aux_loss':
                track_names.append('load_balancing_loss')
            elif config.moe_router_load_balancing_type == 'seq_aux_loss':
                if self.mcore_014:
                    track_names.append('seq_load_balancing_loss')
                else:
                    track_names.append('load_balancing_loss')
            elif config.moe_router_load_balancing_type == 'global_aux_loss':
                track_names.append('global_load_balancing_loss')
            if config.moe_z_loss_coeff is not None:
                track_names.append('z_loss')
            track_moe_kwargs = {'mtp_num_layers': args.mtp_num_layers} if self.mcore_013 else {}
            track_moe_metrics(
                loss_scale=moe_loss_scale,
                iteration=self.state.iteration,
                writer=None,
                total_loss_dict=logs,
                force_initialize=True,
                track_names=track_names,
                num_layers=config.num_layers,
                moe_layer_freq=config.moe_layer_freq,
                **track_moe_kwargs)
        if args.mtp_num_layers is not None:
            mtp_loss_scale = 1 / args.num_microbatches / n_steps
            mtp_logs = {}
            MTPLossLoggingHelper.track_mtp_metrics(mtp_loss_scale, self.state.iteration, None, None, mtp_logs)
            logs.update({k.replace(' ', '_'): v for k, v in mtp_logs.items()})
        # Track sparse attention indexer loss.
        if args.dsa_indexer_loss_coeff is not None and args.dsa_indexer_loss_coeff > 0:
            from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossLoggingHelper
            indexer_loss_scale = 1 / args.num_microbatches / n_steps
            idx_logs = {}
            DSAIndexerLossLoggingHelper.track_indexer_metrics(
                loss_scale=indexer_loss_scale,
                iteration=self.state.iteration,
                writer=None,
                wandb_writer=None,
                total_loss_dict=idx_logs,
            )
            logs.update({k.replace(' ', '_'): v for k, v in idx_logs.items()})

        for k, v in logs.items():
            if isinstance(v, torch.Tensor):
                if v.numel() == 2:
                    v = v[0] / v[1]
                v = v.item()
            logs[k] = v

    def prepare_model(self):
        args = self.args
        self.unwrapped_models = get_mcore_model(args, self.template.config)
        self.config = self.unwrapped_models[0].config
        self.bridge = self.config.bridge
        self.peft_models = self._prepare_peft_model(self.unwrapped_models)
        self.wrapped_models = wrap_model(args, self.unwrapped_models)

    def _prepare_peft_model(self, models):
        args = self.args
        if args.mcore_model is None:
            weights_dir = args.hf_model_weights or args.model_dir
            checkpoint_has_anchor_embedding = self._hf_checkpoint_has_hsd_anchor_embedding(weights_dir)
            self._base_checkpoint_has_hsd_anchor_embedding = checkpoint_has_anchor_embedding
            self.bridge.load_weights(models, weights_dir)
            self._maybe_initialize_hsd_anchor_embeddings_after_load(
                checkpoint_has_anchor_embedding=checkpoint_has_anchor_embedding,
                source='hf_model_weights' if args.hf_model_weights else 'hf_model')
        peft_models = [prepare_mcore_model(args, model) for model in models]
        if args.tuner_type == 'lora' and args.adapters and args.mcore_adapter is None:
            assert len(args.adapters) == 1, 'Currently only support one adapter.'
            self.bridge.load_weights(models, args.adapters[0], peft_format=True, adapter_name='default')
        return peft_models

    def get_optimizer_and_scheduler(self):
        args = self.args
        if mcore_016:
            from megatron.core.optimizer import AdamOptimizerConfig, SGDOptimizerConfig
            if args.optimizer == 'adam' or 'muon' in args.optimizer:
                # TODO(deyuf): Muon needs both adam + muon but get() only receive one config
                # So for now we keep using adam config that's back compat with old way
                config_cls = AdamOptimizerConfig
            elif args.optimizer == 'sgd':
                config_cls = SGDOptimizerConfig
            else:
                raise ValueError(f'Invalid optimizer type: {args.optimizer}')
        else:
            config_cls = OptimizerConfig

        kwargs = {
            f.name: getattr(args, f.name)
            for f in dataclasses.fields(config_cls) if hasattr(args, f.name) and f.name != 'loss_scale'
        }
        config = config_cls(**kwargs)

        model_config_overrides = self._get_model_optimizer_config_overrides()
        if (model_config_overrides is None
                and (args.apply_wd_to_qk_layernorm or self.args.vit_lr is not None
                     or self.args.aligner_lr is not None)):
            param_groups_context = self._patch_get_param_groups()
        else:
            param_groups_context = nullcontext()
        with param_groups_context:

            if 'muon' not in config.optimizer:
                # If the user is asking for a non-zero embedding init std, skip weight decay for embeddings
                # to avoid embeddings from shrinking to zero as recommended in https://arxiv.org/abs/2312.16903
                # default_skip_embedding_weight_decay=args.embedding_init_method_std is not None,
                optimizer_kwargs = {}
                if model_config_overrides is not None:
                    if not mcore_016:
                        raise RuntimeError(
                            'model-owned optimizer config overrides require Megatron-Core >= 0.16')
                    optimizer_kwargs['config_overrides'] = model_config_overrides
                optimizer = get_megatron_optimizer(config, self.wrapped_models, **optimizer_kwargs)
            else:
                from megatron.core.optimizer.muon import get_megatron_muon_optimizer
                optimizer = get_megatron_muon_optimizer(
                    config,
                    self.wrapped_models,
                    layer_wise_distributed_optimizer='dist' in config.optimizer,
                )
        opt_param_scheduler = get_optimizer_param_scheduler(args, optimizer)
        self._log_hsd_optimizer_debug(optimizer)
        return optimizer, opt_param_scheduler

    def _get_model_optimizer_config_overrides(self):
        """Return an explicitly opted-in model's MCore optimizer overrides.

        Merely exposing ``_optimizer_config_overrides`` is not sufficient: an
        older in-tree model may carry dormant metadata whose activation would
        change an existing checkpoint's parameter-group layout.  New models
        must also set ``_use_optimizer_config_overrides=True`` so resume
        behavior stays backward compatible.
        """
        opted_in = [
            model for model in self.unwrapped_models
            if bool(getattr(model, '_use_optimizer_config_overrides', False))
        ]
        if not opted_in:
            return None
        if len(opted_in) != len(self.unwrapped_models):
            raise RuntimeError(
                'all model chunks must agree on model-owned optimizer config overrides')
        overrides = getattr(opted_in[0], '_optimizer_config_overrides', None)
        if overrides is None:
            raise RuntimeError(
                '_use_optimizer_config_overrides=True requires _optimizer_config_overrides')
        baseline_repr = repr(overrides)
        for model in opted_in[1:]:
            other = getattr(model, '_optimizer_config_overrides', None)
            if repr(other) != baseline_repr:
                raise RuntimeError('model chunks disagree on optimizer config overrides')
        return overrides

    def _get_data_collator(self):
        data_collator = self.template.data_collator
        padding_to = get_padding_to(self.args)
        logger.info(f'padding_to: {padding_to}')
        data_collator = partial(data_collator, padding_to=padding_to)
        return data_collator

    def cyclic_iter(self, iterable, use_origin_cyclic: bool = False):
        training = self.unwrapped_models[0].training
        if not training or use_origin_cyclic:
            while True:
                for x in iterable:
                    yield x
            return

        args = self.args
        epoch = 0
        is_finished = False
        while True:
            if not is_finished:
                logger.info(f'The training of Epoch {epoch} starts...')
            for x in iterable:
                yield x
            # streaming
            if training and args.num_train_epochs and epoch >= args.num_train_epochs - 1:
                is_finished = True
            epoch += 1
            if is_finished:
                # Note that this approach will train for one additional step.
                logger.info(f'Training of {epoch} epochs has been completed, the training has finished.')
                args.train_iters = self.state.iteration + 1

    def _get_param_groups_mcore_016(
        self,
        model_chunks: List[MegatronModule],
        config: OptimizerConfig,
        config_overrides,
    ) -> List[Dict]:
        return self._get_param_groups(
            model_chunks,
            no_weight_decay_cond=None,
            scale_lr_cond=None,
            lr_mult=1.,
            lr=config.lr,
            min_lr=config.min_lr,
            decoupled_lr=config.decoupled_lr,
            decoupled_min_lr=config.decoupled_min_lr,
        )

    # Code borrowed from Megatron-LM
    def _get_param_groups(
        self,
        model_chunks: List[MegatronModule],
        no_weight_decay_cond: Optional[Callable],
        scale_lr_cond: Optional[Callable],
        lr_mult: float,
        lr: float,
        min_lr: float,
        decoupled_lr: Optional[float],
        decoupled_min_lr: Optional[float],
        default_skip_embedding_weight_decay: bool = False,
    ) -> List[Dict]:
        """Create parameter groups for optimizer.

        Creates parameter groups based on weight decay condition (regularized vs
        non regularized), learning rate scale condition (lr vs lr_mult * lr),
        and whether it is expert parameters. scale_lr_cond is used during finetuning
        where head of the network requires a scaled version of the base learning rate.

        Args:
            model_chunks (List[MegatronModule]): model chunks to create parameter
                groups for.
            no_weight_decay_cond (func, optional): function to determine whether a
                parameter should not perform weight decay.
            scale_lr_cond (func, optional): function to determine whether a parameter
                should have a scaled learning rate.
            lr_mult (float): learning rate multiplier for parameters that
                satisfy scale_lr_cond.
            lr (float): learning rate.
            min_lr (float): minimum learning rate.
            decoupled_lr (Optional[float]): optional decoupled learning rate.
            decoupled_min_lr (Optional[float]): optional decoupled minimum learning rate.
            default_skip_embedding_weight_decay (bool): whether to skip weight decay for embedding
                parameters by default, if no_weight_decay_cond is not provided.

        Returns:
            List of parameter groups.
        """
        args = self.args
        if decoupled_min_lr is None:
            decoupled_min_lr = min_lr
        is_multimodal = args.megatron_model_meta.is_multimodal
        if args.vit_lr is not None or args.aligner_lr is not None:
            assert is_multimodal, 'vit_lr and aligner_lr are only supported for multimodal models.'
            vit_lr = args.vit_lr if args.vit_lr is not None else args.lr
            aligner_lr = args.aligner_lr if args.aligner_lr is not None else args.lr
            logger.info_once(f'vit_lr: {vit_lr}, aligner_lr: {aligner_lr}, llm_lr: {args.lr}')
        anchor_embedding_lr_mult = getattr(args, 'hsd_anchor_embedding_lr_mult', None)
        if anchor_embedding_lr_mult is None:
            raw_anchor_embedding_lr_mult = os.environ.get('HSD_ANCHOR_EMBEDDING_LR_MULT', '').strip()
            if raw_anchor_embedding_lr_mult:
                anchor_embedding_lr_mult = float(raw_anchor_embedding_lr_mult)
        if anchor_embedding_lr_mult is not None:
            if anchor_embedding_lr_mult <= 0:
                raise ValueError('hsd_anchor_embedding_lr_mult must be positive.')
            logger.info_once(
                f'hsd_anchor_embedding_lr_mult: {anchor_embedding_lr_mult}, '
                f'lr: {args.lr}, anchor_embedding_lr: {args.lr * anchor_embedding_lr_mult}')
        use_decoupled_learning_rate = decoupled_lr is not None

        # Map (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr) to params.
        params_map = {}
        for model_chunk in model_chunks:
            visual = model_chunk.module.module.visual if is_multimodal else None
            for name, param in model_chunk.named_parameters():
                if not param.requires_grad:
                    continue

                is_expert_parallel = not getattr(param, 'allreduce', True)

                if no_weight_decay_cond is not None:
                    no_wd: bool = no_weight_decay_cond(name, param)
                elif args.apply_wd_to_qk_layernorm and any(
                        name.endswith(k) for k in ['q_layernorm.weight', 'k_layernorm.weight']):
                    no_wd = False
                else:
                    # Do not regularize biases and norm parameters.
                    #  optionally, also skip weight decay for embedding parameters if requested
                    #  (useful if you do not want embeddings to shrink to zero in training
                    #  https://arxiv.org/abs/2312.16903)
                    no_wd = (
                        name.endswith('.bias') or len(param.shape) == 1
                        or (default_skip_embedding_weight_decay and 'embedding' in name))
                _lr_mult = lr_mult
                if scale_lr_cond is not None:
                    scale_lr = scale_lr_cond(name, param)
                else:
                    scale_lr = False
                    # Handling multimodal models: vit_lr, aligner_lr
                    unwrapped_name = name.removeprefix('module.').removeprefix('module.')
                    if visual is not None:
                        is_aligner = any(unwrapped_name.startswith(f'visual.{k}') for k in visual._aligner or [])
                        is_vit = any(unwrapped_name.startswith(f'visual.{k}')
                                     for k in visual._vision_tower) and not is_aligner
                    else:
                        is_aligner, is_vit = False, False
                    if anchor_embedding_lr_mult is not None and unwrapped_name.endswith('anchor_embedding.weight'):
                        scale_lr = True
                        _lr_mult = anchor_embedding_lr_mult
                    elif is_vit and args.vit_lr:
                        scale_lr = True
                        _lr_mult = args.vit_lr / lr
                    elif is_aligner and args.aligner_lr:
                        scale_lr = True
                        _lr_mult = args.aligner_lr / lr

                if not no_wd and not scale_lr:
                    wd_mult, _lr_mult = 1.0, 1.0
                elif not no_wd and scale_lr:
                    wd_mult, _lr_mult = 1.0, _lr_mult
                elif no_wd and not scale_lr:
                    wd_mult, _lr_mult = 0.0, 1.0
                else:
                    wd_mult, _lr_mult = 0.0, _lr_mult

                is_decoupled_lr = False
                # For input/embedding and output layer: embedding.word_embeddings.weight /
                # output_layer.weight.
                if use_decoupled_learning_rate and getattr(param, 'is_embedding_or_output_parameter', False):
                    is_decoupled_lr = True

                key = (wd_mult, _lr_mult, is_expert_parallel, is_decoupled_lr)
                if key not in params_map:
                    params_map[key] = []
                params_map[key].append(param)

        # Distributed checkpoint requires all ranks to have the same param groups,
        # so we need to align the param groups across ranks, otherwise we may have
        # runtime error when loading the checkpoint or numerical error when resuming training.
        params_key = list(params_map.keys())
        gathered_params_key = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered_params_key, params_key)
        for keys in gathered_params_key:
            for key in keys:
                if key not in params_key:
                    params_key.append(key)

        param_groups = []
        for key in params_key:
            wd_mult, _lr_mult, is_expert_parallel, is_decoupled_lr = key
            params = params_map[key] if key in params_map else []
            param_group = {
                'params': params,
                'wd_mult': wd_mult,
                'lr_mult': _lr_mult,
                'is_expert_parallel': is_expert_parallel,
                'is_decoupled_lr': is_decoupled_lr,
            }
            # Ensure param_group has required keys for matching when loading optimizer state
            # See MegatronOptimizer._filter_and_reorder_param_groups.
            if param_group_identifier_keys is not None:
                assert set(param_group.keys()) - set(param_group_identifier_keys) == {'params'}
            param_groups.append(param_group)

        # Update min and max lr in param groups
        # These changes are compatible with mcore 0.16.
        for param_group in param_groups:
            if param_group['is_decoupled_lr']:
                assert decoupled_lr is not None
                param_group['max_lr'] = decoupled_lr
                param_group['min_lr'] = decoupled_min_lr
            else:
                param_group['max_lr'] = lr
                param_group['min_lr'] = min_lr
            lr_mult = param_group.pop('lr_mult')
            param_group['max_lr'] *= lr_mult
            param_group['min_lr'] *= lr_mult
        return param_groups

    @contextmanager
    def _patch_get_param_groups(self):
        from megatron.core import optimizer
        _get_param_groups = optimizer._get_param_groups
        if mcore_016:
            optimizer._get_param_groups = self._get_param_groups_mcore_016
        else:
            optimizer._get_param_groups = self._get_param_groups
        try:
            yield
        finally:
            optimizer._get_param_groups = _get_param_groups

    def _load_iteration(self):
        args = self.args
        ckpt_dir = None
        if args.tuner_type == 'full':
            ckpt_dir = args.model
        else:
            ckpt_dir = args.adapters[0] if args.adapters else args.model
        if ckpt_dir is None:
            return 0
        logger.info(f'checkpoint_dir: {ckpt_dir}')
        tracker_path = os.path.join(ckpt_dir, 'latest_checkpointed_iteration.txt')
        if not os.path.exists(tracker_path):
            return 0
        with open(tracker_path, 'r') as f:
            iteration = int(f.read())

        common_path = os.path.join(ckpt_dir, f'iter_{iteration:07d}', 'common.pt')
        if not os.path.exists(common_path):
            return iteration

        state_dict = torch.load(common_path)
        if 'args' not in state_dict:
            return iteration
        self.args.consumed_train_samples = getattr(state_dict['args'], 'consumed_train_samples', 0)

        return iteration

    def _prepare_vit_gradient_checkpointing(self, model):
        visual = model.visual
        if visual is None:
            return
        for vision_tower in visual._vision_tower:
            module = deep_getattr(visual, vision_tower)
            if self.args.vit_gradient_checkpointing:
                dynamic_gradient_checkpointing(module, False)
                try:
                    module.gradient_checkpointing_enable(**(self.args.vit_gradient_checkpointing_kwargs or {}))
                    module.enable_input_require_grads()
                except AttributeError:
                    pass

    @staticmethod
    def _initialize_embedding(model):
        # compat new_special_tokens
        init_method = model.config.init_method
        if hasattr(model, 'language_model'):
            model = model.language_model
        for key in ['embedding.word_embeddings', 'output_layer']:
            if key == 'output_layer' and model.share_embeddings_and_output_weights:
                continue
            module = deep_getattr(model, key)
            if module is None:
                continue
            initialize_mask = (module.weight == 0).all(dim=-1)
            num_to_initialize = initialize_mask.sum().item()
            if num_to_initialize == 0:
                continue
            logger.info_if(f'num_to_initialize: {num_to_initialize}', cond=mpu.get_data_parallel_rank() == 0)
            tensor = module.weight.new_empty(num_to_initialize, module.weight.shape[1])
            module.weight.data[initialize_mask] = init_method(tensor)
            if getattr(module.weight, 'main_param', None) is not None:
                module.weight.main_param.copy_(module.weight.view(-1))

    def _all_reduce_metric(self,
                           metric: Dict[str, torch.Tensor],
                           reduction=torch.distributed.ReduceOp.AVG,
                           group=None) -> Dict[str, torch.Tensor]:
        if group is None:
            group = mpu.get_data_parallel_group()
        reporting_metric = torch.stack(list(metric.values()), dim=0)
        torch.distributed.all_reduce(reporting_metric, reduction, group=group)
        return {k: reporting_metric[i] for i, k in enumerate(metric.keys())}

    def merge_lora_adapters(self, adapter_name='default'):
        """Merge LoRA adapters into base model weights for vLLM inference."""
        with torch.no_grad():
            for model in self.unwrapped_models:
                for module in model.modules():
                    if isinstance(module, LoraParallelLinear):
                        # Merge all active adapters
                        module.merge(adapter_names=[adapter_name])

    def unmerge_lora_adapters(self):
        """Unmerge LoRA adapters to restore training state."""
        with torch.no_grad():
            for model in self.unwrapped_models:
                for module in model.modules():
                    if isinstance(module, LoraParallelLinear):
                        # Unmerge to restore separate LoRA weights for training
                        module.unmerge()

    @staticmethod
    def copy_path(src_path: str, tgt_path: str):
        if not is_master():
            return
        if not os.path.exists(src_path):
            raise FileNotFoundError(f'Source path does not exist: {src_path}')

        if os.path.isfile(src_path):
            os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
            shutil.copy(src_path, tgt_path)
        elif os.path.isdir(src_path):
            shutil.copytree(src_path, tgt_path, dirs_exist_ok=True)
        else:
            raise ValueError(f'Source path is neither a file nor a directory: {src_path}')

    def _copy_hf_metadata_to_checkpoint(self, output_dir: str):
        if not is_master():
            return
        model_dir = Path(getattr(self.args, 'model', '') or '')
        if not model_dir.is_dir():
            return
        skip_names = {
            'args.json',
            'config.json',
            'model.safetensors.index.json',
            'pytorch_model.bin.index.json',
            'latest_checkpointed_iteration.txt',
            'trainer_state.json',
        }
        output_config = Path(output_dir) / 'config.json'
        is_safin1 = False
        if output_config.is_file():
            with output_config.open() as stream:
                is_safin1 = json.load(stream).get('model_type') in {'safin1_4b', 'safin1_35b_a3b'}
        # Safin-1 remote code is emitted by the current bridge finalizer. Input
        # checkpoint Python files can predate this code and must not replace it.
        patterns = ('*.json', '*.jinja', '*.model', '*.txt', '*.tiktoken')
        if not is_safin1:
            patterns += ('*.py',)
        for pattern in patterns:
            for src_path in model_dir.glob(pattern):
                if src_path.name in skip_names:
                    continue
                self.copy_path(str(src_path), os.path.join(output_dir, src_path.name))

    def _prepare_data_iterator(self, train_dataset, val_dataset=None, use_origin_cyclic: bool = False):
        train_dataloader, val_dataloader = self._prepare_dataloader(train_dataset, val_dataset)
        train_data_iterator = iter(self.cyclic_iter(train_dataloader, use_origin_cyclic=use_origin_cyclic))
        val_data_iterator = None
        if val_dataset is not None:
            val_data_iterator = iter(self.cyclic_iter(val_dataloader, use_origin_cyclic=use_origin_cyclic))
        return train_data_iterator, val_data_iterator

    def train(self, train_dataset, val_dataset):
        args = self.args
        config = self.config
        state = self.state
        for m in self.wrapped_models:
            m.train()

        if args.is_multimodal:
            for m in self.unwrapped_models:
                self._prepare_vit_gradient_checkpointing(m)

        config.grad_scale_func = self.optimizer.scale_loss
        if isinstance(self.wrapped_models[0], DDP) and args.overlap_grad_reduce:
            assert config.no_sync_func is None, ('When overlap_grad_reduce is True, config.no_sync_func must be None; '
                                                 'a custom no_sync_func is not supported when overlapping grad-reduce')
            config.no_sync_func = [model_chunk.no_sync for model_chunk in self.wrapped_models]
            if len(self.wrapped_models) == 1:
                config.no_sync_func = config.no_sync_func[0]
            if args.align_grad_reduce:
                config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in self.wrapped_models]
                if len(self.wrapped_models) == 1:
                    config.grad_sync_func = config.grad_sync_func[0]
        if args.overlap_param_gather and args.align_param_gather:
            config.param_sync_func = [model_chunk.start_param_sync for model_chunk in self.wrapped_models]
            if len(self.wrapped_models) == 1:
                config.param_sync_func = config.param_sync_func[0]
        config.finalize_model_grads_func = self._wrap_precision_finalize_model_grads(finalize_model_grads)
        start_iteration = state.iteration
        pre_hook_enabled = False
        # Disable forward pre-hook to start training to ensure that errors in checkpoint loading
        # or random initialization don't propagate to all ranks in first all-gather (which is a
        # no-op if things work correctly).
        if should_disable_forward_pre_hook(args):
            disable_forward_pre_hook(self.wrapped_models, param_sync=False)
            # Also remove param_sync_func temporarily so that sync calls made in
            # `forward_backward_func` are no-ops.
            param_sync_func = config.param_sync_func
            config.param_sync_func = None
            pre_hook_enabled = False

        self.call_event('on_train_begin')
        train_metrics = {}
        if args.virtual_pipeline_model_parallel_size is not None:
            train_data_iterator, val_data_iterator = [], []
            for _ in range(args.virtual_pipeline_model_parallel_size):
                train_it, val_it = self._prepare_data_iterator(train_dataset, val_dataset)
                train_data_iterator.append(train_it)
                val_data_iterator.append(val_it)
        else:
            train_data_iterator, val_data_iterator = self._prepare_data_iterator(train_dataset, val_dataset)
        while state.iteration < args.train_iters:
            self.call_event('on_step_begin')
            maybe_finalize_async_save(args, blocking=False)
            metrics, grad_norm, update_successful = self.train_step(train_data_iterator)
            if state.iteration == start_iteration:
                if update_successful:
                    # Enable forward pre-hook after training step has successfully run. All subsequent
                    # forward passes will use the forward pre-hook / `param_sync_func` in
                    # `forward_backward_func`.
                    if should_disable_forward_pre_hook(args):
                        enable_forward_pre_hook(self.wrapped_models)
                        config.param_sync_func = param_sync_func
                        pre_hook_enabled = True
                else:
                    start_iteration = state.iteration + 1

            state.iteration += 1
            self.call_event('on_step_end')
            self._aggregated_metrics(metrics, train_metrics)
            train_metrics['grad_norm'] = grad_norm
            learning_rate = None
            for param_group in self.optimizer.param_groups:
                if len(param_group['params']) == 0:
                    continue
                learning_rate = param_group['lr']
            if learning_rate is not None:
                train_metrics['learning_rate'] = learning_rate
            if state.should_log:
                state.should_log = False
                self.on_log(logs=train_metrics)
                train_metrics = {}

            eval_metrics = None
            if state.should_eval:
                state.should_eval = False
                if should_disable_forward_pre_hook(args):
                    disable_forward_pre_hook(self.wrapped_models)
                    pre_hook_enabled = False
                eval_metrics = self.evaluate(val_data_iterator)
                for m in self.wrapped_models:
                    m.train()
                if should_disable_forward_pre_hook(args):
                    enable_forward_pre_hook(self.wrapped_models)
                    pre_hook_enabled = True

            if state.should_save:
                self._determine_best_metric(eval_metrics)
                if should_disable_forward_pre_hook(args):
                    disable_forward_pre_hook(self.wrapped_models)
                state.should_save = False
                self.save_checkpoint()
                self.call_event('on_save', output_dir=self.state.last_model_checkpoint)
                if should_disable_forward_pre_hook(args):
                    enable_forward_pre_hook(self.wrapped_models)

        self.call_event('on_train_end')
        self._close_torch_profiler()
        # Close out pre-hooks if using distributed optimizer and overlapped param gather.
        if pre_hook_enabled:
            disable_forward_pre_hook(self.wrapped_models)
        maybe_finalize_async_save(args, blocking=True, terminate=True)

    def _determine_best_metric(self, metrics) -> bool:
        args = self.args
        state = self.state
        if (args.metric_for_best_model is None or metrics is None or not is_last_rank()
                or args.metric_for_best_model not in metrics):
            return False
        metric_value = metrics[args.metric_for_best_model]
        op = operator.ge if args.greater_is_better else operator.le
        if state.best_metric is None:
            state.best_metric = float('-inf') if args.greater_is_better else float('inf')

        is_new_best_metric = False
        if op(metric_value, state.best_metric):
            state.best_metric = metric_value
            state.best_global_step = state.global_step
            is_new_best_metric = True
        return is_new_best_metric

    def save_checkpoint(self):
        args = self.args
        state = self.state
        args.consumed_train_samples = state.consumed_train_samples
        iteration = state.iteration
        output_dir = os.path.join(args.output_dir, f'checkpoint-{iteration}')
        os.makedirs(output_dir, exist_ok=True)
        args_path = os.path.join(os.path.dirname(output_dir), 'args.json')
        self.copy_path(args_path, os.path.join(output_dir, 'args.json'))
        save_peft_format = args.tuner_type == 'lora' and not args.merge_lora
        if args.save_safetensors and args.no_save_optim:
            model = []
        else:
            model = self.wrapped_models
        release_optim = (
            os.getenv('SWIFT_RELEASE_OPTIM_BEFORE_FINAL_SAVE', '0').lower() in {'1', 'true', 'yes', 'on'}
            and args.no_save_optim and args.save_safetensors and args.train_iters is not None
            and iteration >= args.train_iters)
        if release_optim:
            # Final-step smoke/pilot checkpoints do not resume training and do
            # not save optimizer state.  EP2 keeps 128 experts per rank, so the
            # Adam state otherwise leaves no room for HF TP/EP all-gathers.
            # Drop those states only after the last optimizer step and before
            # conversion; formal intermediate epoch checkpoints never set this
            # opt-in environment variable.
            self.optimizer = None
            self.opt_param_scheduler = None
            gc.collect()
            torch.cuda.empty_cache()
            logger.info('Released optimizer state before final safetensors export.')
        save_mcore_checkpoint(
            args,
            model,
            self.optimizer,
            self.opt_param_scheduler,
            iteration=iteration,
            peft_format=args.tuner_type == 'lora',
            output_dir=output_dir)
        state.last_model_checkpoint = output_dir
        if state.best_global_step:
            best_model_checkpoint = os.path.join(args.output_dir, f'checkpoint-{state.best_global_step}')
            if os.path.exists(best_model_checkpoint):
                state.best_model_checkpoint = best_model_checkpoint
        # safetensors
        if args.save_safetensors:
            # merge-lora does not store lora, lora saving may report an error (Qwen3-VL-Moe)
            if args.tuner_type != 'full' and args.merge_lora:
                origin_output_dir = output_dir
                output_dir = f'{output_dir}-merged'
                os.makedirs(output_dir, exist_ok=True)
                for fname in ['latest_checkpointed_iteration.txt', 'args.json']:
                    src_path = os.path.join(origin_output_dir, fname)
                    self.copy_path(src_path, os.path.join(output_dir, fname))
                # common.pt
                common_path = os.path.join(origin_output_dir, f'iter_{iteration:07d}', 'common.pt')
                tgt_common_path = os.path.join(output_dir, f'iter_{iteration:07d}', 'common.pt')
                os.makedirs(os.path.dirname(tgt_common_path), exist_ok=True)
                self.copy_path(common_path, tgt_common_path)
            merge_context = (temporary_merged_lora(self.unwrapped_models, LoraParallelLinear)
                             if args.tuner_type != 'full' and args.merge_lora else nullcontext())
            with merge_context:
                self.bridge.save_weights(
                    self.unwrapped_models,
                    output_dir,
                    peft_format=save_peft_format,
                    args=args,
                    processor=self.template.processor,
                )
                self._copy_hf_metadata_to_checkpoint(output_dir)

        if is_master():
            self._rotate_checkpoints(args.output_dir)

    def _rotate_checkpoints(self, output_dir: str):
        # Code borrowed from huggingface/transformers
        args = self.args
        if args.save_total_limit is None or args.save_total_limit <= 0:
            return

        checkpoints_sorted = self._sorted_checkpoints(output_dir)
        if len(checkpoints_sorted) <= args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            shutil.rmtree(checkpoint, ignore_errors=True)
            if os.path.exists(f'{checkpoint}-merged'):
                shutil.rmtree(f'{checkpoint}-merged', ignore_errors=True)

    def _sorted_checkpoints(self, output_dir: str):
        # Code borrowed from huggingface/transformers
        state = self.state
        glob_checkpoints = [
            str(p) for p in Path(output_dir).glob('checkpoint-*') if p.is_dir() and not p.name.endswith('-merged')
        ]
        # Sort by modification time
        checkpoints_sorted = sorted(glob_checkpoints, key=os.path.getmtime)

        # Make sure we don't delete the best model.
        if state.best_model_checkpoint is not None and state.best_model_checkpoint in checkpoints_sorted:
            best_model_index = checkpoints_sorted.index(state.best_model_checkpoint)
            checkpoints_sorted.pop(best_model_index)
            checkpoints_sorted.append(state.best_model_checkpoint)
        return checkpoints_sorted

    def training_log(self, metrics, grad_norm):
        learning_rate = None
        for param_group in self.optimizer.param_groups:
            if len(param_group['params']) == 0:
                continue
            learning_rate = param_group['lr']
        logger.info(f'metrics: {metrics}, grad_norm: {grad_norm}, learning_rate: {learning_rate}')

    @staticmethod
    def _hsd_debug_enabled() -> bool:
        return os.environ.get('HSD_DEBUG_STATS', '').lower() in {'1', 'true', 'yes', 'on'}

    @staticmethod
    def _hsd_debug_verbose_enabled() -> bool:
        return os.environ.get('HSD_DEBUG_VERBOSE_STATS', '').lower() in {'1', 'true', 'yes', 'on'}

    @staticmethod
    def _hsd_profile_enabled() -> bool:
        return os.environ.get('HSD_PROFILE_TIMING', '').lower() in {'1', 'true', 'yes', 'on'}

    def _clear_hsd_debug_metrics(self):
        if not self._hsd_debug_enabled():
            return
        for model in self.unwrapped_models:
            for module in model.modules():
                if hasattr(module, '_hsd_debug_stats'):
                    module._hsd_debug_stats = {}
                if hasattr(module, '_hsd_debug_stat_counts'):
                    module._hsd_debug_stat_counts = {}

    def _clear_hsd_profile_metrics(self):
        if not self._hsd_profile_enabled():
            return
        for model in self.unwrapped_models:
            for module in model.modules():
                if hasattr(module, '_hsd_profile_stats'):
                    module._hsd_profile_stats = {}
        try:
            from mcore_bridge.model.modules.march_chunk_gdn import clear_hsd_chunk_profile_stats
        except Exception:
            return
        clear_hsd_chunk_profile_stats()

    @staticmethod
    def _normalize_hsd_debug_key(key: str) -> str:
        return re.sub(r'^hsd/layer_\d+/', 'hsd/', key)

    @staticmethod
    def _hsd_debug_aggregation_mode(key: str) -> str:
        metric_name = key.rsplit('/', 1)[-1]
        if metric_name.endswith('_finite'):
            return 'min'
        if metric_name.endswith('_absmax') or metric_name.endswith('_max'):
            return 'max'
        return 'mean'

    def _collect_hsd_debug_metrics(self):
        if not self._hsd_debug_enabled():
            return []
        anchor_embedding_metrics = self._collect_hsd_anchor_embedding_metrics()
        aggregated = {}
        counts = {}
        debug_layer_count = 0
        for model in self.unwrapped_models:
            for module in model.modules():
                hsd_stats = getattr(module, '_hsd_debug_stats', None)
                if not hsd_stats:
                    continue
                debug_layer_count += 1
                for key, value in hsd_stats.items():
                    if isinstance(value, torch.Tensor):
                        value = value.detach().float()
                        if value.numel() != 1:
                            value = value.mean()
                        value = value.reshape(())
                    else:
                        value = torch.tensor(float(value), dtype=torch.float32, device=torch.cuda.current_device())
                    key = self._normalize_hsd_debug_key(key)
                    mode = self._hsd_debug_aggregation_mode(key)
                    if key not in aggregated:
                        aggregated[key] = value
                        counts[key] = 1
                    elif mode == 'min':
                        aggregated[key] = torch.minimum(aggregated[key], value)
                    elif mode == 'max':
                        aggregated[key] = torch.maximum(aggregated[key], value)
                    else:
                        aggregated[key] = aggregated[key] + value
                        counts[key] += 1
                module._hsd_debug_stats = {}
                if hasattr(module, '_hsd_debug_stat_counts'):
                    module._hsd_debug_stat_counts = {}
        if not debug_layer_count:
            return anchor_embedding_metrics

        row = {}
        for key, value in aggregated.items():
            if self._hsd_debug_aggregation_mode(key) == 'mean':
                value = value / max(counts[key], 1)
            row[key] = value
        if self._hsd_debug_verbose_enabled():
            row['hsd/debug_layer_count'] = torch.tensor(
                float(debug_layer_count), dtype=torch.float32, device=torch.cuda.current_device())
        return [row, *anchor_embedding_metrics]

    def _collect_hsd_anchor_embedding_metrics(self):
        stats = []
        for model in self.unwrapped_models:
            for name, param in model.named_parameters():
                if not name.endswith('anchor_embedding.weight'):
                    continue
                with torch.no_grad():
                    weight = param.detach().float()
                    flat = weight.reshape(-1)
                    token_weight = weight.reshape(weight.shape[0], -1) if weight.ndim >= 2 else weight.reshape(1, -1)
                    token_norm = torch.linalg.vector_norm(token_weight, dim=1)
                    stats.append({
                        'norm': torch.linalg.vector_norm(flat),
                        'rms': torch.sqrt(torch.mean(flat * flat)) if flat.numel() else flat.new_tensor(0.0),
                        'absmax': flat.abs().max() if flat.numel() else flat.new_tensor(0.0),
                        'mean': flat.mean() if flat.numel() else flat.new_tensor(0.0),
                        'std': flat.std(unbiased=False) if flat.numel() else flat.new_tensor(0.0),
                        'finite': torch.isfinite(flat).all().float(),
                        'token_norm_mean': token_norm.mean() if token_norm.numel() else flat.new_tensor(0.0),
                        'token_norm_min': token_norm.min() if token_norm.numel() else flat.new_tensor(0.0),
                        'token_norm_max': token_norm.max() if token_norm.numel() else flat.new_tensor(0.0),
                        'num_tokens': flat.new_tensor(float(weight.shape[0] if weight.ndim >= 1 else 0)),
                        'hidden_size': flat.new_tensor(float(weight.shape[-1] if weight.ndim >= 2 else weight.numel())),
                    })

        if not stats:
            return []

        def combine(key: str, mode: str = 'mean') -> torch.Tensor:
            values = torch.stack([item[key].detach().float().reshape(()) for item in stats])
            if mode == 'min':
                return values.min()
            if mode == 'max':
                return values.max()
            return values.mean()

        row = {
            'hsd/anchor_embedding_norm': combine('norm'),
            'hsd/anchor_embedding_rms': combine('rms'),
            'hsd/anchor_embedding_absmax': combine('absmax', 'max'),
            'hsd/anchor_embedding_mean': combine('mean'),
            'hsd/anchor_embedding_std': combine('std'),
            'hsd/anchor_embedding_finite': combine('finite', 'min'),
            'hsd/anchor_embedding_token_norm_mean': combine('token_norm_mean'),
            'hsd/anchor_embedding_token_norm_min': combine('token_norm_min', 'min'),
            'hsd/anchor_embedding_token_norm_max': combine('token_norm_max', 'max'),
            'hsd/anchor_embedding_num_tokens': combine('num_tokens'),
            'hsd/anchor_embedding_hidden_size': combine('hidden_size'),
        }

        if not getattr(self, '_hsd_anchor_embedding_init_logged', False):
            self._hsd_anchor_embedding_init_logged = True
            row.update({
                'hsd/anchor_embedding_init_norm': row['hsd/anchor_embedding_norm'],
                'hsd/anchor_embedding_init_rms': row['hsd/anchor_embedding_rms'],
                'hsd/anchor_embedding_init_absmax': row['hsd/anchor_embedding_absmax'],
                'hsd/anchor_embedding_init_mean': row['hsd/anchor_embedding_mean'],
                'hsd/anchor_embedding_init_std': row['hsd/anchor_embedding_std'],
                'hsd/anchor_embedding_init_finite': row['hsd/anchor_embedding_finite'],
                'hsd/anchor_embedding_init_token_norm_mean': row['hsd/anchor_embedding_token_norm_mean'],
                'hsd/anchor_embedding_init_token_norm_min': row['hsd/anchor_embedding_token_norm_min'],
                'hsd/anchor_embedding_init_token_norm_max': row['hsd/anchor_embedding_token_norm_max'],
            })
        return [row]

    def _collect_hsd_profile_metrics(self):
        if not self._hsd_profile_enabled():
            return []
        aggregated = {}
        counts = {}
        for model in self.unwrapped_models:
            for module in model.modules():
                hsd_stats = getattr(module, '_hsd_profile_stats', None)
                if not hsd_stats:
                    continue
                for key, value in hsd_stats.items():
                    if isinstance(value, torch.Tensor):
                        value = value.detach().float()
                        if value.numel() != 1:
                            value = value.mean()
                        value = value.reshape(())
                    else:
                        value = torch.tensor(float(value), dtype=torch.float32, device=torch.cuda.current_device())
                    key = self._normalize_hsd_debug_key(key)
                    if key not in aggregated:
                        aggregated[key] = value
                        counts[key] = 1
                    else:
                        aggregated[key] = aggregated[key] + value
                        counts[key] += 1
                module._hsd_profile_stats = {}

        try:
            from mcore_bridge.model.modules.march_chunk_gdn import pop_hsd_chunk_profile_stats
        except Exception:
            chunk_stats = {}
        else:
            chunk_stats = pop_hsd_chunk_profile_stats()
        for key, value in chunk_stats.items():
            metric_key = f'hsd/profile/{key}'
            tensor_value = torch.tensor(float(value), dtype=torch.float32, device=torch.cuda.current_device())
            if metric_key not in aggregated:
                aggregated[metric_key] = tensor_value
                counts[metric_key] = 1
            else:
                aggregated[metric_key] = aggregated[metric_key] + tensor_value
                counts[metric_key] += 1

        if not aggregated:
            return []
        return [{key: value / max(counts[key], 1) for key, value in aggregated.items()}]

    def _clear_dataloader_timing_metrics(self):
        self._dataloader_wait_sum_s = 0.0
        self._dataloader_wait_max_s = 0.0
        self._dataloader_prepare_sum_s = 0.0
        self._dataloader_prepare_max_s = 0.0
        self._dataloader_wait_count = 0
        self._dataloader_payload_cpu_bytes_sum = 0.0
        self._dataloader_payload_media_bytes_sum = 0.0
        self._dataloader_payload_input_tokens_sum = 0.0
        self._dataloader_payload_media_grid_tokens_sum = 0.0
        self._dataloader_payload_media_items_sum = 0.0

    def _collect_dataloader_timing_metrics(self):
        count = int(getattr(self, '_dataloader_wait_count', 0))
        if count <= 0:
            return []
        device = torch.cuda.current_device()
        wait_sum_ms = float(self._dataloader_wait_sum_s) * 1000.0
        wait_max_ms = float(self._dataloader_wait_max_s) * 1000.0
        prepare_sum_ms = float(getattr(self, '_dataloader_prepare_sum_s', 0.0)) * 1000.0
        prepare_max_ms = float(getattr(self, '_dataloader_prepare_max_s', 0.0)) * 1000.0
        sum_stats = torch.tensor([wait_sum_ms, float(count)], dtype=torch.float32, device=device)
        prepare_sum_stats = torch.tensor([prepare_sum_ms, float(count)], dtype=torch.float32, device=device)
        max_stats = torch.tensor([wait_max_ms, wait_sum_ms, prepare_max_ms, prepare_sum_ms],
                                 dtype=torch.float32,
                                 device=device)
        payload_stats = torch.tensor([
            float(getattr(self, '_dataloader_payload_cpu_bytes_sum', 0.0)) / 1024.0**2,
            float(getattr(self, '_dataloader_payload_media_bytes_sum', 0.0)) / 1024.0**2,
            float(getattr(self, '_dataloader_payload_input_tokens_sum', 0.0)),
            float(getattr(self, '_dataloader_payload_media_grid_tokens_sum', 0.0)),
            float(getattr(self, '_dataloader_payload_media_items_sum', 0.0)),
            float(count),
        ],
                                     dtype=torch.float32,
                                     device=device)
        payload_count = payload_stats[-1].clamp_min(1.0)
        payload_avg_stats = payload_stats[:-1] / payload_count
        payload_max_stats = payload_avg_stats.clone()
        payload_min_stats = payload_avg_stats.clone()
        world_size = 1.0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(sum_stats, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(prepare_sum_stats, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(max_stats, op=torch.distributed.ReduceOp.MAX)
            torch.distributed.all_reduce(payload_max_stats, op=torch.distributed.ReduceOp.MAX)
            torch.distributed.all_reduce(payload_min_stats, op=torch.distributed.ReduceOp.MIN)
            torch.distributed.all_reduce(payload_stats, op=torch.distributed.ReduceOp.SUM)
            world_size = float(torch.distributed.get_world_size())
        media_ratio = payload_max_stats[1] / payload_min_stats[1].clamp_min(1.0)
        grid_ratio = payload_max_stats[3] / payload_min_stats[3].clamp_min(1.0)
        return [{
            'data/dataloader_wait_ms':
            sum_stats,
            'data/dataloader_wait_step_ms':
            torch.stack([sum_stats[0], sum_stats.new_tensor(world_size)]),
            'data/dataloader_wait_step_max_ms':
            torch.stack([max_stats[1], max_stats.new_tensor(1.0)]),
            'data/dataloader_wait_max_ms':
            torch.stack([max_stats[0], max_stats.new_tensor(1.0)]),
            'data/dataloader_wait_count':
            torch.stack([sum_stats[1], sum_stats.new_tensor(world_size)]),
            'data/prepare_batch_ms':
            prepare_sum_stats,
            'data/prepare_batch_step_ms':
            torch.stack([prepare_sum_stats[0], prepare_sum_stats.new_tensor(world_size)]),
            'data/prepare_batch_step_max_ms':
            torch.stack([max_stats[3], max_stats.new_tensor(1.0)]),
            'data/prepare_batch_max_ms':
            torch.stack([max_stats[2], max_stats.new_tensor(1.0)]),
            'data/batch_cpu_tensor_mib':
            torch.stack([payload_stats[0], payload_stats[5].clamp_min(1.0)]),
            'data/batch_media_tensor_mib':
            torch.stack([payload_stats[1], payload_stats[5].clamp_min(1.0)]),
            'data/batch_media_tensor_mib_step_max':
            torch.stack([payload_max_stats[1], payload_max_stats.new_tensor(1.0)]),
            'data/batch_media_tensor_mib_step_min':
            torch.stack([payload_min_stats[1], payload_min_stats.new_tensor(1.0)]),
            'data/batch_media_tensor_mib_step_max_min_ratio':
            torch.stack([media_ratio, media_ratio.new_tensor(1.0)]),
            'data/batch_input_tokens':
            torch.stack([payload_stats[2], payload_stats[5].clamp_min(1.0)]),
            'data/batch_input_tokens_step_max':
            torch.stack([payload_max_stats[2], payload_max_stats.new_tensor(1.0)]),
            'data/batch_input_tokens_step_min':
            torch.stack([payload_min_stats[2], payload_min_stats.new_tensor(1.0)]),
            'data/batch_media_grid_tokens':
            torch.stack([payload_stats[3], payload_stats[5].clamp_min(1.0)]),
            'data/batch_media_grid_tokens_step_max':
            torch.stack([payload_max_stats[3], payload_max_stats.new_tensor(1.0)]),
            'data/batch_media_grid_tokens_step_min':
            torch.stack([payload_min_stats[3], payload_min_stats.new_tensor(1.0)]),
            'data/batch_media_grid_tokens_step_max_min_ratio':
            torch.stack([grid_ratio, grid_ratio.new_tensor(1.0)]),
            'data/batch_media_items':
            torch.stack([payload_stats[4], payload_stats[5].clamp_min(1.0)]),
        }]

    def _perf_monitor_enabled(self) -> bool:
        return self._env_flag('SWIFT_PERF_MONITOR', False)

    def _perf_sync_timing_enabled(self) -> bool:
        return self._env_flag('SWIFT_PERF_SYNC_TIMING', False)

    def _perf_mark(self):
        if self._perf_sync_timing_enabled() and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    @staticmethod
    def _read_int_file(path: str):
        try:
            with open(path, 'r') as f:
                value = f.read().strip()
            if not value or value == 'max':
                return None
            return int(value)
        except (FileNotFoundError, PermissionError, ValueError, OSError):
            return None

    @staticmethod
    def _read_proc_rss_bytes(pid: int):
        try:
            with open(f'/proc/{pid}/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        parts = line.split()
                        if len(parts) >= 2:
                            return int(parts[1]) * 1024
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, OSError):
            return 0
        return 0

    @classmethod
    def _read_process_tree_rss_bytes(cls):
        pid = os.getpid()
        total = cls._read_proc_rss_bytes(pid)
        try:
            with open(f'/proc/{pid}/task/{pid}/children', 'r') as f:
                child_pids = [int(x) for x in f.read().split() if x.strip()]
        except (FileNotFoundError, PermissionError, ValueError, OSError):
            child_pids = []
        for child_pid in child_pids:
            total += cls._read_proc_rss_bytes(child_pid)
        return total

    @classmethod
    def _resource_snapshot_gib(cls):
        cgroup_current = (cls._read_int_file('/sys/fs/cgroup/memory.current')
                          or cls._read_int_file('/sys/fs/cgroup/memory/memory.usage_in_bytes'))
        cgroup_peak = (cls._read_int_file('/sys/fs/cgroup/memory.peak')
                       or cls._read_int_file('/sys/fs/cgroup/memory/memory.max_usage_in_bytes'))
        snapshot = {
            'sys/process_rss_gib': cls._read_proc_rss_bytes(os.getpid()) / 1024.0**3,
            'sys/process_tree_rss_gib': cls._read_process_tree_rss_bytes() / 1024.0**3,
        }
        if cgroup_current is not None:
            snapshot['sys/cgroup_mem_gib'] = cgroup_current / 1024.0**3
        if cgroup_peak is not None:
            snapshot['sys/cgroup_mem_peak_gib'] = cgroup_peak / 1024.0**3
        if torch.cuda.is_available():
            snapshot['gpu/allocated_gib'] = torch.cuda.memory_allocated() / 1024.0**3
            snapshot['gpu/reserved_gib'] = torch.cuda.memory_reserved() / 1024.0**3
        return snapshot

    def _collect_perf_monitor_metrics(self, timings_s):
        if not self._perf_monitor_enabled():
            return []
        device = torch.cuda.current_device()
        timing_names = [
            'zero_grad',
            'forward_backward',
            'collect_metrics',
            'optimizer_step',
            'post_optimizer',
            'step_wall',
        ]
        timing_values = torch.tensor([float(timings_s.get(name, 0.0)) * 1000.0 for name in timing_names],
                                     dtype=torch.float32,
                                     device=device)
        resource_snapshot = self._resource_snapshot_gib()
        resource_names = [
            'sys/process_rss_gib',
            'sys/process_tree_rss_gib',
            'sys/cgroup_mem_gib',
            'sys/cgroup_mem_peak_gib',
            'gpu/allocated_gib',
            'gpu/reserved_gib',
        ]
        resource_values = torch.tensor([float(resource_snapshot.get(name, 0.0)) for name in resource_names],
                                       dtype=torch.float32,
                                       device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(timing_values, op=torch.distributed.ReduceOp.MAX)
            if resource_values.numel() > 0:
                torch.distributed.all_reduce(resource_values, op=torch.distributed.ReduceOp.MAX)
        one = timing_values.new_tensor(1.0)
        metrics = {f'perf/{name}_ms': torch.stack([timing_values[i], one]) for i, name in enumerate(timing_names)}
        for i, name in enumerate(resource_names):
            metrics[name] = torch.stack([resource_values[i], one])
        return [metrics]

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        value = os.environ.get(name)
        if value in {None, ''}:
            return default
        try:
            return int(value)
        except ValueError:
            logger.warning(f'Invalid integer env {name}={value!r}; using {default}.')
            return default

    def _torch_profiler_enabled(self) -> bool:
        if not self._env_flag('SWIFT_TORCH_PROFILER', False):
            return False
        return self._rank_allowed_from_env('SWIFT_TORCH_PROFILER_RANKS', '0')

    def _current_rank(self) -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
        return int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0)) or 0)

    def _get_torch_profiler(self):
        if not self._torch_profiler_enabled():
            return None
        if self._torch_profiler is not None:
            return self._torch_profiler
        try:
            from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
        except Exception as e:
            logger.warning(f'SWIFT_TORCH_PROFILER=1 but torch.profiler is unavailable: {e}')
            return None

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        wait = max(0, self._env_int('SWIFT_TORCH_PROFILER_WAIT', 2))
        warmup = max(0, self._env_int('SWIFT_TORCH_PROFILER_WARMUP', 2))
        active = max(1, self._env_int('SWIFT_TORCH_PROFILER_ACTIVE', 4))
        repeat = max(1, self._env_int('SWIFT_TORCH_PROFILER_REPEAT', 1))
        trace_root = (
            os.environ.get('SWIFT_TORCH_PROFILER_DIR')
            or os.path.join(os.environ.get('RUN_DIR') or self.args.output_dir, 'torch_profiler'))
        rank = self._current_rank()
        rank_dir = os.path.join(trace_root, f'rank{rank:05d}')
        os.makedirs(rank_dir, exist_ok=True)

        self._torch_profiler_rank = rank
        self._torch_profiler = profile(
            activities=activities,
            schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
            on_trace_ready=tensorboard_trace_handler(rank_dir, worker_name=f'rank{rank}'),
            record_shapes=self._env_flag('SWIFT_TORCH_PROFILER_RECORD_SHAPES', False),
            profile_memory=True,
            with_stack=self._env_flag('SWIFT_TORCH_PROFILER_WITH_STACK', False),
            with_flops=self._env_flag('SWIFT_TORCH_PROFILER_WITH_FLOPS', False),
        )
        self._torch_profiler.__enter__()
        logger.info(
            'torch profiler enabled: '
            f'rank={rank} dir={rank_dir} wait={wait} warmup={warmup} active={active} repeat={repeat}')
        return self._torch_profiler

    def _torch_profiler_step(self):
        if self._torch_profiler is not None:
            self._torch_profiler.step()

    def _close_torch_profiler(self):
        if self._torch_profiler is None:
            return
        try:
            self._torch_profiler.__exit__(None, None, None)
            logger.info(f'torch profiler closed: rank={self._torch_profiler_rank}')
        finally:
            self._torch_profiler = None
            self._torch_profiler_rank = None

    @classmethod
    def _tensor_payload_stats(cls, value, key_path=''):
        stats = {'cpu_bytes': 0, 'media_bytes': 0, 'input_tokens': 0, 'media_grid_tokens': 0, 'media_items': 0}
        if isinstance(value, torch.Tensor):
            nbytes = value.numel() * value.element_size()
            if not value.is_cuda:
                stats['cpu_bytes'] += nbytes
            lowered_key = key_path.lower()
            if any(token in lowered_key for token in ('pixel', 'image', 'video', 'frame')):
                stats['media_bytes'] += nbytes
            if lowered_key.endswith('input_ids'):
                stats['input_tokens'] += value.numel()
            if lowered_key.endswith('image_grid_thw') or lowered_key.endswith('video_grid_thw'):
                if value.dim() >= 2 and value.shape[-1] == 3:
                    grid = value.detach()
                    stats['media_grid_tokens'] += int(grid.to(dtype=torch.float32).prod(dim=-1).sum().item())
                    stats['media_items'] += int(grid.reshape(-1, 3).shape[0])
            return stats
        if isinstance(value, dict):
            for key, item in value.items():
                child = cls._tensor_payload_stats(item, f'{key_path}.{key}' if key_path else str(key))
                for stat_key, stat_value in child.items():
                    stats[stat_key] += stat_value
            return stats
        if isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                child = cls._tensor_payload_stats(item, f'{key_path}.{i}' if key_path else str(i))
                for stat_key, stat_value in child.items():
                    stats[stat_key] += stat_value
        return stats

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        value = os.environ.get(name)
        if value in {None, ''}:
            return default
        return value.lower() in {'1', 'true', 'yes', 'on'}

    def _grad_debug_enabled(self) -> bool:
        return self._env_flag('HSD_GRAD_DEBUG', False)

    @staticmethod
    def _grad_debug_rank_allowed() -> bool:
        return BaseMegatronTrainer._rank_allowed_from_env('HSD_GRAD_DEBUG_RANKS', '0')

    @staticmethod
    def _rank_allowed_from_env(env_name: str, default: str = '0') -> bool:
        ranks = os.environ.get(env_name, default).strip().lower()
        if ranks in {'all', '*'}:
            return True
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return ranks in {'', '0', 'rank0'}
        rank = torch.distributed.get_rank()
        if ranks in {'last', 'last_rank'}:
            return is_last_rank()
        allowed = {item.strip() for item in ranks.split(',') if item.strip()}
        return str(rank) in allowed or f'rank{rank}' in allowed

    @staticmethod
    def _param_grad(param):
        grad = getattr(param, 'main_grad', None)
        if grad is None:
            grad = param.grad
        return grad

    @staticmethod
    def _grad_stats_for_debug(grad):
        if grad is None:
            return 'None'
        with torch.no_grad():
            grad_f = grad.detach().float()
            finite = torch.isfinite(grad_f)
            is_finite = bool(finite.all().item())
            norm = torch.linalg.vector_norm(grad_f).item()
            max_abs = grad_f.abs().max().item() if grad_f.numel() else 0.0
            mean_abs = grad_f.abs().mean().item() if grad_f.numel() else 0.0
        return (
            f'finite={is_finite} norm={norm:.6e} max_abs={max_abs:.6e} '
            f'mean_abs={mean_abs:.6e} dtype={grad.dtype} shape={tuple(grad.shape)}'
        )

    @staticmethod
    def _format_float_list(values):
        return '[' + ', '.join(f'{value:.6e}' for value in values) + ']'

    @staticmethod
    def _emit_grad_debug(message: str):
        # The swift logger suppresses INFO logs on LOCAL_RANK != 0. Grad debug may
        # explicitly target all ranks, so bypass the logger on those workers.
        local_rank = int(os.environ.get('LOCAL_RANK', '-1'))
        if local_rank in {-1, 0}:
            logger.info(message)
        else:
            print(f'[INFO:swift] {message}', flush=True)

    @staticmethod
    def _iter_optimizer_param_groups(optimizer):
        stack = [optimizer]
        seen = set()
        while stack:
            current = stack.pop()
            if current is None or id(current) in seen:
                continue
            seen.add(id(current))
            for child in getattr(current, 'chained_optimizers', []) or []:
                stack.append(child)
            try:
                inner = getattr(current, 'optimizer', None)
            except (AssertionError, AttributeError):
                inner = None
            if inner is not None and inner is not current:
                stack.append(inner)
            try:
                param_groups = getattr(current, 'param_groups', None)
            except (AssertionError, AttributeError):
                param_groups = None
            if param_groups is None:
                continue
            for group_idx, param_group in enumerate(param_groups):
                yield type(current).__name__, group_idx, param_group

    @staticmethod
    def _iter_optimizer_objects(optimizer):
        stack = [optimizer]
        seen = set()
        while stack:
            current = stack.pop()
            if current is None or id(current) in seen:
                continue
            seen.add(id(current))
            yield current
            for child in getattr(current, 'chained_optimizers', []) or []:
                stack.append(child)
            try:
                inner = getattr(current, 'optimizer', None)
            except (AssertionError, AttributeError):
                inner = None
            if inner is not None and inner is not current:
                stack.append(inner)

    def _optimizer_model_param_locations(self, optimizer):
        locations = {}
        for current in self._iter_optimizer_objects(optimizer):
            opt_name = type(current).__name__
            for attr in ('model_float16_groups', 'model_fp32_groups', 'float16_groups', 'fp32_from_fp32_groups'):
                groups = getattr(current, attr, None)
                if not groups:
                    continue
                for group_idx, group in enumerate(groups):
                    for param in group:
                        if param is not None:
                            locations.setdefault(id(param), []).append(f'{opt_name}.{attr}[{group_idx}]')
            model_group_map = getattr(current, 'model_param_group_index_map', None)
            if model_group_map:
                for param, index in model_group_map.items():
                    locations.setdefault(id(param), []).append(f'{opt_name}.model_param_group_index_map{index}')
            model_gbuf_map = getattr(current, 'model_param_gbuf_map', None)
            if model_gbuf_map:
                for param, index in model_gbuf_map.items():
                    locations.setdefault(id(param), []).append(f'{opt_name}.model_param_gbuf_map{index}')
        return locations

    def _ddp_model_param_locations(self):
        locations = {}
        for model_idx, model in enumerate(self.wrapped_models):
            param_to_bucket_group = getattr(model, 'param_to_bucket_group', None)
            if param_to_bucket_group:
                for param in param_to_bucket_group:
                    locations.setdefault(id(param), []).append(f'DDP[{model_idx}].param_to_bucket_group')
            for attr in ('buffers', 'expert_parallel_buffers'):
                for buffer_idx, buffer in enumerate(getattr(model, attr, []) or []):
                    param_index_map = getattr(buffer, 'param_index_map', None)
                    if param_index_map:
                        for param in param_index_map:
                            locations.setdefault(id(param), []).append(
                                f'DDP[{model_idx}].{attr}[{buffer_idx}].param_index_map')
                    for bucket_idx, bucket in enumerate(getattr(buffer, 'buckets', []) or []):
                        for param in getattr(bucket, 'params_list', []) or []:
                            locations.setdefault(id(param), []).append(
                                f'DDP[{model_idx}].{attr}[{buffer_idx}].bucket[{bucket_idx}]')
        return locations

    def _log_hsd_optimizer_debug(self, optimizer):
        if not self._env_flag('HSD_OPTIM_DEBUG', False):
            return
        optim_rank_default = os.environ.get('HSD_GRAD_DEBUG_RANKS', '0')
        if not self._rank_allowed_from_env('HSD_OPTIM_DEBUG_RANKS', optim_rank_default):
            return
        pattern_text = os.environ.get(
            'HSD_OPTIM_DEBUG_REGEX',
            r'(anchor_embedding\.weight|anchor_q_proj\.weight|anchor_k_proj\.weight|router_q_proj\.weight|'
            r'null_gate_proj\.weight|hsd_null_read_gate_bias|hsd_history_gate_logit|'
            r'hsd_history_out_rmsnorm_weight)$',
        )
        try:
            pattern = re.compile(pattern_text)
        except re.error as e:
            self._emit_grad_debug(f'[hsd-optim-debug] invalid regex={pattern_text!r}: {e}')
            return

        param_locations = {}
        for opt_name, group_idx, param_group in self._iter_optimizer_param_groups(optimizer):
            group_desc = (
                f'{opt_name}[{group_idx}]'
                f' lr={param_group.get("lr", param_group.get("max_lr", "<unset>"))}'
                f' wd_mult={param_group.get("wd_mult", "<unset>")}'
                f' expert={param_group.get("is_expert_parallel", "<unset>")}'
            )
            for param in param_group.get('params', []):
                param_locations.setdefault(id(param), []).append(group_desc)
        model_param_locations = self._optimizer_model_param_locations(optimizer)
        ddp_param_locations = self._ddp_model_param_locations()

        rows = []
        matched = 0
        registered = 0
        trainable = 0
        for model_idx, model in enumerate(self.unwrapped_models):
            for name, param in model.named_parameters():
                if not pattern.search(name):
                    continue
                matched += 1
                if param.requires_grad:
                    trainable += 1
                locations = param_locations.get(id(param), [])
                model_locations = model_param_locations.get(id(param), [])
                ddp_locations = ddp_param_locations.get(id(param), [])
                is_registered = bool(locations or model_locations or ddp_locations)
                if is_registered:
                    registered += 1
                rows.append(
                    f'  model={model_idx} requires_grad={param.requires_grad} '
                    f'registered_for_distributed_optimizer={is_registered} '
                    f'in_optimizer_shard_map={bool(locations or model_locations)} '
                    f'allreduce={getattr(param, "allreduce", True)} '
                    f'in_ddp_buffer={bool(ddp_locations)} has_main_grad={hasattr(param, "main_grad")} '
                    f'tensor_model_parallel={getattr(param, "tensor_model_parallel", False)} '
                    f'average_gradients_across_tp_domain='
                    f'{getattr(param, "average_gradients_across_tp_domain", False)} '
                    f'dtype={param.dtype} shape={tuple(param.shape)} name={name}\n'
                    f'    optimizer_groups={locations if locations else "[]"}\n'
                    f'    model_param_groups={model_locations if model_locations else "[]"}\n'
                    f'    ddp_buffers={ddp_locations if ddp_locations else "[]"}'
                )

        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized() else 0)
        if not rows:
            rows.append(f'  no params matched regex={pattern_text}')
        self._emit_grad_debug(
            f'[hsd-optim-debug] rank={rank} matched={matched} trainable={trainable} '
            f'registered_for_distributed_optimizer={registered} regex={pattern_text}\n' + '\n'.join(rows)
        )

    def _log_anchor_embedding_grad_debug(self):
        if not self._grad_debug_enabled() or not self._grad_debug_rank_allowed():
            return

        rows = []
        max_tokens = int(os.environ.get('HSD_GRAD_DEBUG_ANCHOR_TOKENS', '16'))
        for model_idx, model in enumerate(self.unwrapped_models):
            for name, param in model.named_parameters():
                if not name.endswith('anchor_embedding.weight'):
                    continue
                grad = self._param_grad(param)
                if grad is None:
                    rows.append(f'  model={model_idx} grad=None shape={tuple(param.shape)} name={name}')
                    continue
                with torch.no_grad():
                    grad_f = grad.detach().float()
                    finite = torch.isfinite(grad_f)
                    is_finite = bool(finite.all().item())
                    norm = torch.linalg.vector_norm(grad_f).item()
                    max_abs = grad_f.abs().max().item() if grad_f.numel() else 0.0
                    mean_abs = grad_f.abs().mean().item() if grad_f.numel() else 0.0
                    token_norms = torch.linalg.vector_norm(grad_f.reshape(grad_f.shape[0], -1),
                                                           dim=1).tolist() if grad_f.ndim >= 2 else []
                token_norm_text = ''
                if token_norms:
                    shown_norms = token_norms[:max_tokens] if max_tokens > 0 else []
                    token_norm_text = f' token_grad_norms={self._format_float_list(shown_norms)}'
                    if len(token_norms) > len(shown_norms):
                        token_norm_text += f' ...(+{len(token_norms) - len(shown_norms)})'
                rows.append(
                    f'  model={model_idx} finite={is_finite} grad_norm={norm:.6e} max_abs={max_abs:.6e} '
                    f'mean_abs={mean_abs:.6e} dtype={param.dtype} shape={tuple(param.shape)} '
                    f'name={name}{token_norm_text}')

        if not rows:
            return
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized() else 0)
        self._emit_grad_debug(f'[hsd-grad-debug-anchor-embedding] step={self.state.iteration + 1} rank={rank}\n' +
                              '\n'.join(rows))

    def _log_hsd_grad_buffer_debug(self):
        if not self._env_flag('HSD_GRAD_DEBUG_BUFFERS', False):
            return

        pattern_text = os.environ.get(
            'HSD_GRAD_DEBUG_BUFFER_REGEX',
            r'(anchor_embedding\.weight|anchor_q_proj\.weight|anchor_k_proj\.weight|router_q_proj\.weight|'
            r'null_gate_proj\.weight|hsd_null_read_gate_bias|hsd_history_gate_logit|'
            r'hsd_history_out_rmsnorm_weight)$',
        )
        try:
            pattern = re.compile(pattern_text)
        except re.error as e:
            self._emit_grad_debug(f'[hsd-grad-debug-buffers] invalid regex={pattern_text!r}: {e}')
            return

        limit = int(os.environ.get('HSD_GRAD_DEBUG_BUFFER_LIMIT', '80'))
        rows = []
        matched = 0
        for model_idx, model in enumerate(self.unwrapped_models):
            for name, param in model.named_parameters():
                if not pattern.search(name):
                    continue
                matched += 1
                main_grad = getattr(param, 'main_grad', None)
                param_grad = param.grad
                rows.append(
                    f'  model={model_idx} requires_grad={param.requires_grad} dtype={param.dtype} '
                    f'shape={tuple(param.shape)} name={name}\n'
                    f'    main_grad={self._grad_stats_for_debug(main_grad)}\n'
                    f'    param_grad={self._grad_stats_for_debug(param_grad)}'
                )
                if limit > 0 and len(rows) >= limit:
                    break
            if limit > 0 and len(rows) >= limit:
                break

        if not rows:
            return
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized() else 0)
        suffix = ''
        if limit > 0 and matched > len(rows):
            suffix = f' shown={len(rows)}'
        self._emit_grad_debug(
            f'[hsd-grad-debug-buffers] step={self.state.iteration + 1} rank={rank} '
            f'matched={matched}{suffix} regex={pattern_text}\n' + '\n'.join(rows)
        )

    @staticmethod
    def _grad_debug_group_name(name: str) -> str:
        if 'anchor_embedding.weight' in name:
            return 'hsd_anchor_embedding'
        if 'anchor_q_proj.weight' in name:
            return 'hsd_anchor_q_proj'
        if 'anchor_k_proj.weight' in name:
            return 'hsd_anchor_k_proj'
        if 'router_q_proj.weight' in name:
            return 'hsd_router_q_proj'
        if 'null_gate_proj.weight' in name or name.endswith('hsd_null_read_gate_bias'):
            return 'hsd_null_read_gate'
        if name.endswith('hsd_history_gate_logit'):
            return 'hsd_history_gate'
        if name.endswith('hsd_history_out_rmsnorm_weight'):
            return 'hsd_history_out_rmsnorm'
        if '.self_attention.' in name and re.search(
                r'\.(in_proj|in_proj_qkvz|in_proj_ba|conv1d|out_proj)\.|'
                r'\.(A_log|dt_bias|D)$',
                name,
        ):
            return 'hsd_core_self_attention'
        if 'embedding' in name or name.endswith('output_layer.weight'):
            return 'embedding_or_output'
        if '.mlp.' in name or '.moe' in name or '.router.' in name:
            return 'mlp_moe_or_router'
        return 'other'

    def _log_grad_debug_groups(self):
        if not self._grad_debug_enabled() or not self._grad_debug_rank_allowed():
            return
        if not self._env_flag('HSD_GRAD_DEBUG_GROUPS', False):
            return

        def _new_group():
            return {
                'params': 0,
                'sumsq': 0.0,
                'max_abs': 0.0,
                'nonfinite': 0,
                'top_param_norm': 0.0,
                'top_param_name': '',
            }

        groups = {}

        def _add(group_name, norm, sumsq, max_abs, is_finite, name):
            group = groups.setdefault(group_name, _new_group())
            group['params'] += 1
            group['sumsq'] += sumsq
            group['max_abs'] = max(group['max_abs'], max_abs)
            if not is_finite:
                group['nonfinite'] += 1
            if norm >= group['top_param_norm'] or not group['top_param_name']:
                group['top_param_norm'] = norm
                group['top_param_name'] = name

        total_params = 0
        params_with_grad = 0
        nonfinite_count = 0
        for model in self.unwrapped_models:
            for name, param in model.named_parameters():
                total_params += 1
                grad = self._param_grad(param)
                if grad is None:
                    continue
                params_with_grad += 1
                with torch.no_grad():
                    grad_f = grad.detach().float()
                    finite = torch.isfinite(grad_f)
                    is_finite = bool(finite.all().item())
                    norm = torch.linalg.vector_norm(grad_f).item()
                    sumsq = grad_f.pow(2).sum().item()
                    max_abs = grad_f.abs().max().item() if grad_f.numel() else 0.0
                if not is_finite:
                    nonfinite_count += 1
                primary_group = self._grad_debug_group_name(name)
                _add('all', norm, sumsq, max_abs, is_finite, name)
                _add(primary_group, norm, sumsq, max_abs, is_finite, name)
                if primary_group in {
                        'hsd_anchor_embedding',
                        'hsd_anchor_q_proj',
                        'hsd_anchor_k_proj',
                        'hsd_router_q_proj',
                        'hsd_null_read_gate',
                        'hsd_history_gate',
                        'hsd_history_out_rmsnorm',
                }:
                    _add('hsd_new_total', norm, sumsq, max_abs, is_finite, name)
                if primary_group in {
                        'hsd_anchor_k_proj',
                        'hsd_router_q_proj',
                        'hsd_null_read_gate',
                        'hsd_history_gate',
                        'hsd_history_out_rmsnorm',
                }:
                    _add('hsd_read_route_total', norm, sumsq, max_abs, is_finite, name)

        if not groups:
            return
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized() else 0)
        total_sumsq = groups.get('all', {}).get('sumsq', 0.0)
        total_norm = math.sqrt(total_sumsq) if total_sumsq >= 0.0 else float('nan')
        limit = int(os.environ.get('HSD_GRAD_DEBUG_GROUP_LIMIT', '20'))
        ordered = sorted(groups.items(), key=lambda item: (item[0] != 'all', item[1]['sumsq']), reverse=True)
        ordered = sorted(ordered, key=lambda item: item[0] != 'all')
        if limit > 0:
            ordered = ordered[:limit]

        lines = [
            f'[hsd-grad-debug-groups] step={self.state.iteration + 1} rank={rank} '
            f'local_total_norm={total_norm:.6e} params_with_grad={params_with_grad}/{total_params} '
            f'nonfinite_params={nonfinite_count}'
        ]
        for group_name, group in ordered:
            group_norm = math.sqrt(group['sumsq']) if group['sumsq'] >= 0.0 else float('nan')
            if total_sumsq > 0.0 and math.isfinite(total_sumsq):
                sumsq_pct = 100.0 * group['sumsq'] / total_sumsq
            else:
                sumsq_pct = float('nan')
            lines.append(
                f'  group={group_name} params={group["params"]} nonfinite={group["nonfinite"]} '
                f'norm={group_norm:.6e} sumsq_pct={sumsq_pct:.6f} max_abs={group["max_abs"]:.6e} '
                f'top_param_norm={group["top_param_norm"]:.6e} top_param={group["top_param_name"]}'
            )
        self._emit_grad_debug('\n'.join(lines))

    def _log_grad_debug_topk(self):
        if not self._grad_debug_enabled() or not self._grad_debug_rank_allowed():
            return
        self._log_anchor_embedding_grad_debug()
        self._log_hsd_grad_buffer_debug()
        self._log_grad_debug_groups()
        topk = int(os.environ.get('HSD_GRAD_DEBUG_TOPK', '20'))
        if topk <= 0:
            return
        min_norm = float(os.environ.get('HSD_GRAD_DEBUG_MIN_NORM', '0'))
        only_nonfinite = self._env_flag('HSD_GRAD_DEBUG_ONLY_NONFINITE', False)

        rows = []
        total_params = 0
        params_with_grad = 0
        nonfinite_count = 0
        for model_idx, model in enumerate(self.unwrapped_models):
            for name, param in model.named_parameters():
                total_params += 1
                grad = self._param_grad(param)
                if grad is None:
                    continue
                params_with_grad += 1
                with torch.no_grad():
                    grad_f = grad.detach().float()
                    finite = torch.isfinite(grad_f)
                    is_finite = bool(finite.all().item())
                    if not is_finite:
                        nonfinite_count += 1
                    norm = torch.linalg.vector_norm(grad_f).item()
                    max_abs = grad_f.abs().max().item() if grad_f.numel() else 0.0
                    mean_abs = grad_f.abs().mean().item() if grad_f.numel() else 0.0
                if norm < min_norm and is_finite:
                    continue
                if only_nonfinite and is_finite:
                    continue
                rows.append((norm, max_abs, mean_abs, is_finite, model_idx, name, tuple(param.shape), str(param.dtype)))

        if not rows:
            return
        rows.sort(key=lambda item: (not item[3], item[0], item[1]), reverse=True)
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized() else 0)
        header = (
            f'[hsd-grad-debug] step={self.state.iteration + 1} rank={rank} '
            f'params_with_grad={params_with_grad}/{total_params} nonfinite_params={nonfinite_count} '
            f'topk={min(topk, len(rows))}'
        )
        lines = [header]
        for idx, (norm, max_abs, mean_abs, is_finite, model_idx, name, shape, dtype) in enumerate(rows[:topk], start=1):
            lines.append(
                f'  #{idx} model={model_idx} finite={is_finite} grad_norm={norm:.6e} '
                f'max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} dtype={dtype} '
                f'shape={shape} name={name}'
            )
        self._emit_grad_debug('\n'.join(lines))

    def _get_precision_tracer(self):
        if hasattr(self, '_precision_tracer'):
            return self._precision_tracer
        try:
            from march.diagnostics.precision_trace import PrecisionTracer
        except Exception as e:
            if self._env_flag('MEGATRON_PRECISION_TRACE', False) or os.environ.get('MEGATRON_PRECISION_TRACE_STEPS'):
                logger.warning(f'Megatron precision trace requested but unavailable: {e}')
            self._precision_tracer = None
            return None

        default_dir = os.path.join(self.args.output_dir, 'precision_trace')
        self._precision_tracer = PrecisionTracer.from_env(default_dir)
        if self._precision_tracer is not None:
            info = self._precision_tracer.info
            logger.info(
                f"[megatron-precision-trace] enabled dir={self._precision_tracer.trace_dir} "
                f"rank={info['rank']} dp_rank={info['data_parallel_rank']} "
                f"tp_rank={info['tensor_model_parallel_rank']} pp_rank={info['pipeline_model_parallel_rank']}")
        return self._precision_tracer

    def _precision_trace_step(self) -> int:
        return int(self.state.iteration) + 1

    def _trace_precision_batch(self, data, *, labels=None, loss_scale=None):
        tracer = self._get_precision_tracer()
        if tracer is None:
            return
        tracer.trace_hsd_runtime_metadata(self.args, step=self._precision_trace_step())
        tracer.trace_batch(data, labels=labels, loss_scale=loss_scale, step=self._precision_trace_step())

    def _trace_precision_output(self, output_tensor):
        tracer = self._get_precision_tracer()
        if tracer is None:
            return
        tracer.trace_output_tensor(output_tensor, step=self._precision_trace_step())

    @staticmethod
    def _first_precision_tensor(value):
        if torch.is_tensor(value):
            return value
        if isinstance(value, dict):
            for item in value.values():
                tensor = BaseMegatronTrainer._first_precision_tensor(item)
                if tensor is not None:
                    return tensor
        if isinstance(value, (list, tuple)):
            for item in value:
                tensor = BaseMegatronTrainer._first_precision_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    def _trace_precision_inference_logits(self, model, data):
        if not self._env_flag('MEGATRON_PRECISION_TRACE_INFERENCE_LOGITS', False):
            return
        tracer = self._get_precision_tracer()
        if tracer is None or not tracer.should_trace(self._precision_trace_step()):
            return

        trace_data = {key: value for key, value in data.items() if key != 'labels'}
        was_training = bool(getattr(model, 'training', False))
        rng_devices = []
        if torch.cuda.is_available():
            try:
                rng_devices = [torch.cuda.current_device()]
            except Exception:
                rng_devices = []
        try:
            model.eval()
            with torch.no_grad(), torch.random.fork_rng(devices=rng_devices, enabled=True):
                output = model(**trace_data)
            logits = self._first_precision_tensor(output)
            if logits is not None:
                tracer.trace_tensor(
                    'forward/logits',
                    logits,
                    step=self._precision_trace_step(),
                    category='forward_logits',
                    extra={'layout': 'batch_sequence_vocab'},
                )
        finally:
            if was_training:
                model.train()

    def _trace_precision_param_grads(self):
        tracer = self._get_precision_tracer()
        if tracer is None:
            return
        tracer.trace_model_gradients(self.unwrapped_models, self._param_grad, step=self._precision_trace_step())

    def _trace_precision_param_chain(self, stage, *, include_meta=False, extra=None):
        tracer = self._get_precision_tracer()
        if tracer is None:
            return
        tracer.trace_param_chain_models(
            self.unwrapped_models,
            stage=stage,
            step=self._precision_trace_step(),
            include_meta=include_meta,
            extra=extra,
        )

    def _register_precision_param_chain_hooks(self):
        tracer = self._get_precision_tracer()
        if tracer is None:
            return []
        return tracer.register_param_chain_hooks(self.unwrapped_models, step=self._precision_trace_step())

    def _wrap_precision_finalize_model_grads(self, finalize_func):

        def wrapped_finalize_model_grads(model, num_tokens=None, pg_collection=None, force_all_reduce=False):
            tracer = self._get_precision_tracer()
            trace_enabled = (
                tracer is not None
                and getattr(tracer, 'param_chain_enabled', False)
                and tracer.should_trace(self._precision_trace_step())
            )
            if not trace_enabled:
                return finalize_func(
                    model,
                    num_tokens=num_tokens,
                    pg_collection=pg_collection,
                    force_all_reduce=force_all_reduce,
                )

            finalize_module = importlib.import_module(finalize_func.__module__)
            originals = []

            def restore_originals():
                for obj, attr, original in reversed(originals):
                    setattr(obj, attr, original)

            def patch_attr(obj, attr, replacement):
                original = getattr(obj, attr)
                setattr(obj, attr, replacement)
                originals.append((obj, attr, original))
                return original

            self._trace_precision_param_chain(
                'pre_finalize',
                extra={'num_tokens_present': num_tokens is not None},
            )

            model_chunks = list(model) if isinstance(model, (list, tuple)) else [model]
            for chunk_idx, model_chunk in enumerate(model_chunks):
                if not hasattr(model_chunk, 'finish_grad_sync'):
                    continue
                original_finish_grad_sync = getattr(model_chunk, 'finish_grad_sync')

                def finish_grad_sync_with_trace(*args, _original=original_finish_grad_sync, _idx=chunk_idx, **kwargs):
                    result = _original(*args, **kwargs)
                    self._trace_precision_param_chain(
                        'post_finish_grad_sync',
                        extra={
                            'num_tokens_present': num_tokens is not None,
                            'model_chunk_idx': _idx,
                        },
                    )
                    return result

                patch_attr(model_chunk, 'finish_grad_sync', finish_grad_sync_with_trace)

            def patch_finalize_helper(helper_name, chain_stage):
                if not hasattr(finalize_module, helper_name):
                    return
                original_helper = getattr(finalize_module, helper_name)

                def helper_with_trace(*args, _original=original_helper, _helper_name=helper_name, **kwargs):
                    result = _original(*args, **kwargs)
                    self._trace_precision_param_chain(
                        chain_stage,
                        extra={
                            'num_tokens_present': num_tokens is not None,
                            'finalize_helper': _helper_name,
                        },
                    )
                    return result

                patch_attr(finalize_module, helper_name, helper_with_trace)

            patch_finalize_helper('_allreduce_non_tensor_model_parallel_grads', 'post_non_tp_grad_allreduce')
            patch_finalize_helper('_allreduce_position_embedding_grads', 'post_embedding_grad_allreduce')

            try:
                result = finalize_func(
                    model,
                    num_tokens=num_tokens,
                    pg_collection=pg_collection,
                    force_all_reduce=force_all_reduce,
                )
                self._trace_precision_param_chain(
                    'post_num_token_scale',
                    extra={'num_tokens_present': num_tokens is not None},
                )
                return result
            finally:
                restore_originals()

        return wrapped_finalize_model_grads

    def evaluate(self, val_data_iterator):
        args = self.args
        for m in self.wrapped_models:
            m.eval()
        eval_metrics = {}
        forward_backward_func = get_forward_backward_func()
        self.call_event('on_eval_begin')
        with torch.no_grad():
            for _ in range(args.eval_iters):
                data_iterator = self._replace_data_iterator(val_data_iterator)
                metrics = forward_backward_func(
                    forward_step_func=self.forward_step,
                    data_iterator=data_iterator,
                    model=self.wrapped_models,
                    num_microbatches=self.args.num_microbatches,
                    seq_length=args.seq_length,
                    micro_batch_size=args.micro_batch_size,
                    forward_only=True,
                )
                self.call_event('on_eval_step')
                self._aggregated_metrics(metrics, eval_metrics)
        self.compute_eval_metrics(eval_metrics)
        self.on_log(logs=eval_metrics, prefix='eval_')
        self.call_event('on_eval_end')
        return eval_metrics

    def compute_eval_metrics(self, metrics):
        if self.eval_metrics is not None:
            metric = self.eval_metrics.compute()
            for k, v in metric.items():
                metrics[k] = v if isinstance(v, torch.Tensor) else torch.tensor(v)
            self.eval_metrics.reset()

    def _replace_data_iterator(self, data_iterator):
        return data_iterator

    def train_step(self, train_data_iterator):
        args = self.args
        forward_backward_func = get_forward_backward_func()
        perf_enabled = self._perf_monitor_enabled()
        step_timings = {}
        step_start = self._perf_mark() if perf_enabled else time.perf_counter()
        zero_grad_start = self._perf_mark() if perf_enabled else time.perf_counter()
        self._trace_precision_param_chain('pre_zero_grad', include_meta=True)
        for m in self.wrapped_models:
            m.zero_grad_buffer()
        self.optimizer.zero_grad()
        self._trace_precision_param_chain('post_zero_grad')
        step_timings['zero_grad'] = (self._perf_mark() if perf_enabled else time.perf_counter()) - zero_grad_start
        # TODO: refactor _replace_data_iterator
        data_iterator = self._replace_data_iterator(train_data_iterator)

        if self.enable_routing_replay:
            RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

        self._clear_hsd_debug_metrics()
        self._clear_hsd_profile_metrics()
        self._clear_dataloader_timing_metrics()
        param_chain_handles = self._register_precision_param_chain_hooks()
        torch_profiler = self._get_torch_profiler()
        forward_backward_start = self._perf_mark() if perf_enabled else time.perf_counter()
        try:
            metrics = forward_backward_func(
                forward_step_func=self.forward_step,
                data_iterator=data_iterator,
                model=self.wrapped_models,
                num_microbatches=args.num_microbatches,
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                forward_only=False,
            )
        finally:
            if torch_profiler is not None:
                self._torch_profiler_step()
            for handle in param_chain_handles:
                handle.remove()
        step_timings['forward_backward'] = (
            self._perf_mark() if perf_enabled else time.perf_counter()) - forward_backward_start
        collect_metrics_start = self._perf_mark() if perf_enabled else time.perf_counter()
        hsd_debug_metrics = self._collect_hsd_debug_metrics()
        if hsd_debug_metrics:
            if isinstance(metrics, list):
                metrics.extend(hsd_debug_metrics)
            else:
                metrics = [metrics, *hsd_debug_metrics]
        hsd_profile_metrics = self._collect_hsd_profile_metrics()
        if hsd_profile_metrics:
            if isinstance(metrics, list):
                metrics.extend(hsd_profile_metrics)
            else:
                metrics = [metrics, *hsd_profile_metrics]
        dataloader_timing_metrics = self._collect_dataloader_timing_metrics()
        if dataloader_timing_metrics:
            if isinstance(metrics, list):
                metrics.extend(dataloader_timing_metrics)
            else:
                metrics = [metrics, *dataloader_timing_metrics]

        self._trace_precision_param_chain('post_forward_backward_pre_optimizer')
        self._log_grad_debug_topk()
        self._trace_precision_param_grads()
        step_timings['collect_metrics'] = (
            self._perf_mark() if perf_enabled else time.perf_counter()) - collect_metrics_start
        optimizer_start = self._perf_mark() if perf_enabled else time.perf_counter()
        update_successful, grad_norm, _ = self.optimizer.step()
        step_timings['optimizer_step'] = (self._perf_mark() if perf_enabled else time.perf_counter()) - optimizer_start
        post_optimizer_start = self._perf_mark() if perf_enabled else time.perf_counter()
        self._trace_precision_param_chain('post_optimizer_step')
        update_successful = logical_and_across_model_parallel_group(update_successful)
        grad_norm = reduce_max_stat_across_model_parallel_group(grad_norm)
        if update_successful:
            self.opt_param_scheduler.step(increment=args.global_batch_size)
        step_timings['post_optimizer'] = (
            self._perf_mark() if perf_enabled else time.perf_counter()) - post_optimizer_start

        if self.enable_routing_replay:
            RouterReplay.clear_global_router_replay_action()
            RouterReplay.clear_global_indices()

        step_timings['step_wall'] = (self._perf_mark() if perf_enabled else time.perf_counter()) - step_start
        perf_monitor_metrics = self._collect_perf_monitor_metrics(step_timings)
        if perf_monitor_metrics:
            if isinstance(metrics, list):
                metrics.extend(perf_monitor_metrics)
            else:
                metrics = [metrics, *perf_monitor_metrics]

        return metrics, grad_norm, update_successful

    def _aggregated_metrics(self, metrics, total_metrics):
        if 'n_steps' not in total_metrics:
            total_metrics['n_steps'] = 0
        total_metrics['n_steps'] += 1
        if not metrics:
            return
        metrics = RowPreprocessor.rows_to_batched(metrics)
        for key, val in metrics.items():
            val = torch.stack([v for v in val if v is not None], dim=0)
            if val[0].numel() == 2:
                val = val.sum(dim=0)
                if val[1] == 0:
                    continue
            elif val[0].numel() == 1:
                val = val.new_tensor([val.sum(), val.shape[0]])
            else:
                raise ValueError(f'Invalid value shape: {val[0].shape} for key {key}')
            if key not in total_metrics:
                total_metrics[key] = torch.tensor([0.0, 0.0], dtype=torch.float32, device=torch.cuda.current_device())
            total_metrics[key] += val

    def _prepare_dataloader(self, train_dataset, val_dataset=None):
        args = self.args
        val_dataloader = None
        if args.streaming:
            train_dataloader = build_streaming_dataloader(args, train_dataset, self.data_collator)
            if val_dataset is not None:
                val_dataloader = build_streaming_dataloader(args, val_dataset, self.data_collator)
            return train_dataloader, val_dataloader
        train_batch_sampler = MegatronPretrainingRandomSampler(
            train_dataset,
            total_samples=len(train_dataset),
            consumed_samples=self.state.consumed_train_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding,
            shuffle=args.train_dataloader_shuffle,
            group_by_length=args.group_by_length,
        )
        train_dataloader = self._create_dataloader(train_dataset, train_batch_sampler)
        if val_dataset is not None:
            val_batch_sampler = MegatronPretrainingSampler(
                total_samples=len(val_dataset),
                consumed_samples=0,
                micro_batch_size=args.micro_batch_size,
                data_parallel_rank=mpu.get_data_parallel_rank(),
                data_parallel_size=mpu.get_data_parallel_world_size(),
            )
            val_dataloader = self._create_dataloader(val_dataset, val_batch_sampler)
        return train_dataloader, val_dataloader

    def _create_dataloader(self, dataset, batch_sampler):
        args = self.args

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=args.dataloader_num_workers,
            pin_memory=args.dataloader_pin_memory,
            persistent_workers=args.dataloader_persistent_workers if args.dataloader_num_workers > 0 else False,
            prefetch_factor=args.dataloader_prefetch_factor if args.dataloader_num_workers > 0 else None,
            collate_fn=self.data_collator,
        )
        return dataloader

    @abstractmethod
    def forward_step(self, data_iterator, model):
        pass

    def _prepare_batch(self, data, vp_stage=None, num_samples=None):
        batch = get_batch_on_this_pp_rank(self.args, data, vp_stage=vp_stage)
        if num_samples is None:
            num_samples = batch.pop('num_samples')
        args = self.args
        text_position_ids = batch.pop('text_position_ids', None)
        batch.pop('attention_mask_2d', None)
        if text_position_ids is None:
            text_position_ids = batch.get('position_ids')
        if args.padding_free and text_position_ids is not None:
            batch['packed_seq_params'] = get_packed_seq_params(text_position_ids)
            batch['packed_seq_params'].num_samples = num_samples
        # slice batch along sequence dimension for context parallelism
        batch = get_batch_on_this_cp_rank(args, batch)
        return batch

    def get_batch(self, data_iterator, vp_stage=None):
        """Generate a batch."""
        start_time = time.perf_counter()
        data = next(data_iterator)
        wait_s = time.perf_counter() - start_time
        self._dataloader_wait_sum_s += wait_s
        self._dataloader_wait_max_s = max(self._dataloader_wait_max_s, wait_s)
        self._dataloader_wait_count += 1
        if self._perf_monitor_enabled():
            payload_stats = self._tensor_payload_stats(data)
            self._dataloader_payload_cpu_bytes_sum += float(payload_stats['cpu_bytes'])
            self._dataloader_payload_media_bytes_sum += float(payload_stats['media_bytes'])
            self._dataloader_payload_input_tokens_sum += float(payload_stats['input_tokens'])
            self._dataloader_payload_media_grid_tokens_sum += float(payload_stats['media_grid_tokens'])
            self._dataloader_payload_media_items_sum += float(payload_stats['media_items'])
        if self._perf_monitor_enabled() and self._perf_sync_timing_enabled() and torch.cuda.is_available():
            torch.cuda.synchronize()
        prepare_start = time.perf_counter()
        batch = self._prepare_batch(data, vp_stage)
        if self._perf_monitor_enabled() and self._perf_sync_timing_enabled() and torch.cuda.is_available():
            torch.cuda.synchronize()
        prepare_s = time.perf_counter() - prepare_start
        self._dataloader_prepare_sum_s += prepare_s
        self._dataloader_prepare_max_s = max(self._dataloader_prepare_max_s, prepare_s)
        return batch

    def _collect_config_info(self) -> Dict[str, str]:
        """
        Collects trainer-specific configuration details.

        Subclasses can override this method to provide additional configuration
        information for model compatibility verification.

        Returns:
            Dict[str, str]: Configuration parameters as key-value pairs.
        """
        if self.__class__.__name__ == 'MegatronTrainer':
            if not self.template.use_chat_template:
                return {
                    'seq2seq_mode': 'pt',
                }
            else:
                return {
                    'seq2seq_mode': 'sft',
                }
        return {}

    def get_last_tokens(self, output_tensor, packed_seq_params=None, attention_mask=None, num_samples=None):
        if packed_seq_params is None:
            last_token_idx = get_last_valid_indices((~attention_mask[:, 0, -1]).long())
            last_tokens = output_tensor[torch.arange(output_tensor.shape[0]), last_token_idx]
        else:
            num_samples = num_samples or packed_seq_params.num_samples
            last_token_idx = packed_seq_params.cu_seqlens_q[1:num_samples + 1] - 1
            last_tokens = output_tensor[0, last_token_idx]
        return last_tokens

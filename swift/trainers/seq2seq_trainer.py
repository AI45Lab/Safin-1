# Copyright (c) ModelScope Contributors. All rights reserved.
# Part of the implementation is borrowed from huggingface/transformers.
import inspect
import json
import os
import time
import torch
from contextlib import contextmanager, nullcontext
from peft import PeftModel
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_peft_available
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
from swift.sequence_parallel import sequence_parallel
from swift.utils import HfConfigFactory, JsonlWriter, Serializer, gc_collect, get_logger, unwrap_model_for_generation
from .arguments import Seq2SeqTrainingArguments
from .mixin import DataLoaderMixin, SwiftMixin
from .utils import per_token_loss_func, per_token_loss_func_sp

logger = get_logger()


def _stacked_monitor_rows_to_cpu(tensor: torch.Tensor):
    """Synchronize one packed monitoring tensor and return ordinary rows."""
    return tensor.detach().cpu().tolist()


class Seq2SeqTrainer(SwiftMixin, DataLoaderMixin, HfSeq2SeqTrainer):
    args: Seq2SeqTrainingArguments

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = True  # fix transformers>=4.46.2
        if self.args.predict_with_generate:
            self.infer_engine = TransformersEngine(
                self.model, template=self.template, max_batch_size=self.args.per_device_eval_batch_size)
        self.jsonl_writer = JsonlWriter(os.path.join(self.args.output_dir, 'predict.jsonl'))
        self._raven_perf_last_step = None
        self._raven_perf_last_time = None

    def _collect_raven_performance_log(self, logs: Dict[str, float]) -> None:
        """Add measured text throughput and CUDA peaks for Raven training."""
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        config = getattr(unwrapped_model, 'config', None)
        model_type = getattr(config, 'model_type', None)
        if model_type not in {'raven', 'raven_hils'} or not self.model.training:
            return
        metric_prefix = 'raven_hils' if model_type == 'raven_hils' else 'raven'

        step = int(self.state.global_step)
        now = time.perf_counter()
        previous_step = self._raven_perf_last_step
        previous_time = self._raven_perf_last_time
        self._raven_perf_last_step = step
        self._raven_perf_last_time = now

        sample_tensor = next(unwrapped_model.parameters())
        peak_allocated = 0.0
        peak_reserved = 0.0
        if sample_tensor.device.type == 'cuda':
            peak_allocated = float(torch.cuda.max_memory_allocated(sample_tensor.device))
            peak_reserved = float(torch.cuda.max_memory_reserved(sample_tensor.device))

        elapsed = 0.0
        step_delta = 0
        if previous_step is not None and previous_time is not None and step > previous_step:
            elapsed = now - previous_time
            step_delta = step - previous_step

        # Report the slowest-rank interval and largest rank-local CUDA peak.
        # Every training rank reaches Trainer.log at the same logging cadence.
        measured = torch.tensor(
            (elapsed, peak_allocated, peak_reserved),
            dtype=torch.float64,
            device=sample_tensor.device,
        )
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if distributed:
            torch.distributed.all_reduce(measured, op=torch.distributed.ReduceOp.MAX)
        elapsed, peak_allocated, peak_reserved = (float(value) for value in measured.tolist())

        logs[f'{metric_prefix}/perf/peak_memory_allocated_gib'] = peak_allocated / 2**30
        logs[f'{metric_prefix}/perf/peak_memory_reserved_gib'] = peak_reserved / 2**30
        if step_delta > 0 and elapsed > 0.0:
            # ``max_length`` is a template/preprocessing argument in ms-swift,
            # not a Transformers TrainingArguments field. Prefer the active
            # template, with config as a defensive fallback for custom users.
            sequence_length = getattr(getattr(self, 'template', None), 'max_length', None)
            if sequence_length is None:
                sequence_length = getattr(config, 'max_position_embeddings', None)
            if sequence_length is None:
                logger.warning_once('Skipping Raven text throughput: sequence length is unavailable.')
                return
            tokens_per_step = (
                int(self.args.world_size)
                * int(self.args.per_device_train_batch_size)
                * int(self.args.gradient_accumulation_steps)
                * int(sequence_length)
            )
            logs[f'{metric_prefix}/perf/text_tokens_per_second'] = step_delta * tokens_per_step / elapsed

    def _collect_raven_router_log(self, logs: Dict[str, float]) -> None:
        """Drain Raven's on-device router accumulators at the logging cadence.

        Scalar summaries are added to the normal Trainer logs (and therefore
        TensorBoard). The full per-layer, per-head slot histograms are written
        to a compact JSONL sidecar on world rank zero. No load-balancing loss is
        introduced; this method is observability-only.
        """
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        config = getattr(unwrapped_model, 'config', None)
        model_type = getattr(config, 'model_type', None)
        if model_type not in {'raven', 'raven_hils'} or not hasattr(unwrapped_model, 'get_router_stats'):
            return
        metric_prefix = 'raven_hils' if model_type == 'raven_hils' else 'raven'
        raven_layers = getattr(getattr(unwrapped_model, 'model', None), 'layers', None)
        if raven_layers and not any(
                bool(getattr(layer.attn, 'router_stats_enabled', False)) for layer in raven_layers):
            # Benchmark runs disable router monitoring uniformly on every rank.
            # Return before allocating reduction buffers or issuing the three
            # otherwise-empty collectives at every logging step.
            return
        layer_stats = unwrapped_model.get_router_stats(clear=True)
        if not layer_stats:
            return

        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        json_layers = {}
        scalar_rows = []
        topk = int(getattr(config, 'topk', 1))
        # The histogram tracks the forget support.  In bounded-decoupled
        # Raven this is intentionally larger than the write support/topk.
        selection_topk = int(getattr(config, 'forget_topk', topk))
        num_heads = int(getattr(config, 'num_heads', 1))
        num_slots = int(getattr(config, 'num_slots', 1))
        ordered = sorted(layer_stats.items())
        if not ordered:
            return
        sample_tensor = next(unwrapped_model.parameters())
        histograms = []
        sums_per_layer = []
        maxima_per_layer = []
        for _, stats in ordered:
            if stats is None:
                histograms.append(torch.zeros(
                    num_heads, num_slots, dtype=torch.long, device=sample_tensor.device))
                sums_per_layer.append(torch.zeros(3, dtype=torch.float32, device=sample_tensor.device))
                maxima_per_layer.append(torch.full(
                    (2,), -torch.inf, dtype=torch.float32, device=sample_tensor.device))
                continue
            histograms.append(stats['selection_histogram'].detach().clone())
            sums_per_layer.append(torch.stack((
                stats['selection_entropy_sum'].float(),
                stats['total_write_sum'].float(),
                stats['decay_sum'].float(),
            )))
            maxima_per_layer.append(torch.stack((
                stats['decay_max'].float(),
                stats['decay_abs_max'].float(),
            )))
        histogram_stack = torch.stack(histograms)
        sums_stack = torch.stack(sums_per_layer)
        maxima_stack = torch.stack(maxima_per_layer)
        if distributed:
            # Every rank executes exactly these three collectives, including a
            # rank that happened to collect no local observations.
            torch.distributed.all_reduce(histogram_stack, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(sums_stack, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(maxima_stack, op=torch.distributed.ReduceOp.MAX)

        # Pack every layer's scalars and histogram into one FP64 tensor before
        # crossing the device boundary. Histogram counts remain exact for any
        # realistic logging window (well below FP64's 2**53 integer range).
        observation_counts = histogram_stack.sum(dim=(1, 2), dtype=torch.long) // selection_topk
        denominators = observation_counts.clamp_min(1).to(dtype=torch.float32)
        scalar_stack = torch.stack((
            observation_counts.to(dtype=torch.float64),
            histogram_stack.eq(0).float().mean(dim=(1, 2)).to(dtype=torch.float64),
            (sums_stack[:, 0] / denominators).to(dtype=torch.float64),
            (sums_stack[:, 1] / denominators).to(dtype=torch.float64),
            (sums_stack[:, 2] / denominators).to(dtype=torch.float64),
            maxima_stack[:, 0].to(dtype=torch.float64),
            maxima_stack[:, 1].to(dtype=torch.float64),
        ), dim=1)
        monitor_stack = torch.cat((
            scalar_stack,
            histogram_stack.reshape(len(ordered), -1).to(dtype=torch.float64),
        ), dim=1)
        monitor_rows = _stacked_monitor_rows_to_cpu(monitor_stack)

        for position, (layer_idx, _) in enumerate(ordered):
            monitor_row = monitor_rows[position]
            observation_count = int(monitor_row[0])
            if observation_count == 0:
                continue
            prefix = f'{metric_prefix}/layer_{int(layer_idx):02d}'
            metric_values = monitor_row[1:7]
            row = {
                f'{prefix}/dead_slot_fraction': metric_values[0],
                f'{prefix}/selection_entropy': metric_values[1],
                f'{prefix}/total_write': metric_values[2],
                f'{prefix}/decay_mean': metric_values[3],
                f'{prefix}/decay_max': metric_values[4],
                f'{prefix}/decay_abs_max': metric_values[5],
            }
            logs.update(row)
            scalar_rows.append(row)
            if self.is_world_process_zero():
                flat_histogram = monitor_row[7:]
                selection_histogram = [
                    [int(value) for value in flat_histogram[head * num_slots:(head + 1) * num_slots]]
                    for head in range(num_heads)
                ]
                json_layers[str(layer_idx)] = {
                    'selection_histogram': selection_histogram,
                    'observation_count': observation_count,
                    'dead_slot_fraction': row[f'{prefix}/dead_slot_fraction'],
                    'selection_entropy': row[f'{prefix}/selection_entropy'],
                    'total_write': row[f'{prefix}/total_write'],
                    'decay_mean': row[f'{prefix}/decay_mean'],
                    'decay_max': row[f'{prefix}/decay_max'],
                    'decay_abs_max': row[f'{prefix}/decay_abs_max'],
                }

        if scalar_rows:
            metric_names = (
                'dead_slot_fraction', 'selection_entropy', 'total_write',
                'decay_mean', 'decay_max', 'decay_abs_max',
            )
            for metric_name in metric_names:
                values = [
                    value
                    for row in scalar_rows
                    for key, value in row.items()
                    if key.endswith(f'/{metric_name}')
                ]
                logs[f'{metric_prefix}/global/{metric_name}'] = sum(values) / len(values)

        if json_layers and self.is_world_process_zero():
            os.makedirs(self.args.output_dir, exist_ok=True)
            raven_core = getattr(getattr(unwrapped_model, 'model', None), 'layers', None)
            if raven_core:
                forward_sample_interval = int(raven_core[0].attn.router_stats_interval)
            else:
                forward_sample_interval = int(getattr(config, 'router_stats_interval', 0))
            record = {
                'step': int(self.state.global_step),
                'mode': 'train' if self.model.training else 'eval',
                'world_size': torch.distributed.get_world_size() if distributed else 1,
                'router_alpha': float(getattr(config, 'router_alpha', 1.0)),
                'topk': topk,
                'memory_update_mode': getattr(config, 'memory_update_mode', 'tied'),
                'write_topk': int(getattr(config, 'write_topk', topk)),
                'forget_topk': selection_topk,
                'forward_sample_interval': forward_sample_interval,
                'layers': json_layers,
            }
            path = os.path.join(self.args.output_dir, f'{metric_prefix}_router_stats.jsonl')
            with open(path, 'a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, separators=(',', ':')) + '\n')

    def _collect_raven_hils_memory_log(self, logs: Dict[str, float]) -> None:
        """Reduce and persist dense-memory statistics on an identical rank path."""
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        config = getattr(unwrapped_model, 'config', None)
        if getattr(config, 'model_type', None) != 'raven_hils' or not hasattr(
                unwrapped_model, 'get_memory_stats'):
            return
        raven_layers = getattr(getattr(unwrapped_model, 'model', None), 'layers', None)
        if raven_layers and not any(
                bool(getattr(layer.attn, 'memory_stats_enabled', False)) for layer in raven_layers):
            # Stats-off throughput runs must not pay for two empty distributed
            # reductions plus a device-to-host drain merely because the model
            # exposes the monitoring API.
            return
        layer_stats = unwrapped_model.get_memory_stats(clear=True)
        layer_count = int(getattr(config, 'num_hidden_layers', 0))
        if layer_count <= 0:
            return
        metric_names = (
            'base_mass',
            'history_mass',
            'visible_snapshot_count',
            'snapshot_topk_count',
            'snapshot_topk_fraction',
            'snapshot_discarded_history_mass',
            'snapshot_preprune_history_mass',
            'route_top1_selected_fraction',
            'route_top4_selected_fraction',
            'route_entropy',
            'effective_candidates',
            # Conditional history-route concentration, excluding the base
            # candidate.  These are emitted by Raven-HiLS per query and must
            # be whitelisted here so they reach the JSONL monitor artifacts.
            'conditional_history_entropy',
            'conditional_history_effective_candidates',
            'history_valid_fraction',
            'age_mass',
            'oldest_history_mass',
            'latest_history_mass',
            'history_distance_mass',
            'history_mean_distance_chunks',
            'base_lse_mean',
            'base_lse_variance',
            'base_lse_q10',
            'base_lse_q50',
            'base_lse_q90',
            'landmark_entropy_mean',
            'landmark_anchor_key_rms',
            'state_anchor_slot_lse_correlation',
            'state_anchor_slot_lse_top4_recall',
            'landmark_relevance_mean',
            'history_score_mean',
            'history_score_variance',
            'history_score_rms',
            'base_output_rms',
            'history_raw_output_rms',
            'history_output_rms',
            'base_history_cosine',
            'mixed_output_rms',
            'fusion_relative_base_change',
            'snapshot_key_rms',
            'snapshot_value_rms',
            'snapshot_state_rms',
            'snapshot_count',
            'snapshot_payload_elements',
            'snapshot_fp32_cache_gib',
            'snapshot_grad_norm',
            'route_adapter_grad_norm',
            'anchor_adapter_grad_norm',
            'stats_interval',
        )
        sample_tensor = next(unwrapped_model.parameters())
        values = torch.zeros(
            layer_count, len(metric_names), dtype=torch.float64, device=sample_tensor.device)
        counts = torch.zeros_like(values)
        for layer_idx in range(layer_count):
            stats = None if not layer_stats else layer_stats.get(layer_idx)
            if stats is None:
                continue
            for metric_idx, metric_name in enumerate(metric_names):
                value = stats.get(metric_name)
                if value is None:
                    continue
                tensor = torch.as_tensor(value, dtype=torch.float64, device=sample_tensor.device)
                if tensor.numel() != 1:
                    continue
                values[layer_idx, metric_idx] = tensor.reshape(())
                counts[layer_idx, metric_idx] = 1

        # Top-K selection histograms are variable-width (the number of
        # snapshots depends on the current context/cache), so keep them out
        # of the long-standing scalar reduction tensor.  The configured
        # maximum gives every rank a common collective shape while the model
        # accumulator pads shorter documents on demand.
        configured_heads = int(getattr(config, 'num_heads', 0) or 0)
        configured_max_snapshots = 0
        try:
            configured_max_snapshots = int(getattr(config, 'max_text_length')) // int(
                getattr(config, 'memory_chunk_size')
            )
        except (TypeError, ValueError, AttributeError, ZeroDivisionError):
            configured_max_snapshots = 0
        histogram_width = max(1, configured_max_snapshots)
        histogram_heads = max(1, configured_heads)
        histogram_values = torch.zeros(
            layer_count,
            histogram_heads,
            histogram_width,
            dtype=torch.float64,
            device=sample_tensor.device,
        )
        histogram_present = torch.zeros(
            layer_count, dtype=torch.float64, device=sample_tensor.device
        )
        for layer_idx in range(layer_count):
            stats = None if not layer_stats else layer_stats.get(layer_idx)
            if not stats:
                continue
            histogram = stats.get('selected_snapshot_index_histogram')
            if histogram is None:
                continue
            histogram = torch.as_tensor(
                histogram, dtype=torch.float64, device=sample_tensor.device
            )
            if histogram.ndim != 2:
                continue
            heads = min(histogram_heads, int(histogram.shape[0]))
            width = min(histogram_width, int(histogram.shape[1]))
            if heads and width:
                histogram_values[layer_idx, :heads, :width] = histogram[:heads, :width]
                histogram_present[layer_idx] = 1

        # Every Raven-HiLS rank reaches precisely these two collectives at each
        # Trainer logging event, including ranks with no sampled observation.
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if distributed:
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(
                histogram_values, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(
                histogram_present, op=torch.distributed.ReduceOp.SUM
            )
        means = values / counts.clamp_min(1)
        # One packed transfer replaces per-layer/per-metric item() calls.
        monitor_rows = _stacked_monitor_rows_to_cpu(torch.stack((counts, means), dim=-1))

        json_layers = {}
        for layer_idx in range(layer_count):
            row = {}
            for metric_idx, metric_name in enumerate(metric_names):
                count, value = monitor_rows[layer_idx][metric_idx]
                if count <= 0:
                    continue
                value = float(value)
                row[metric_name] = value
                logs[f'raven_hils/memory/layer_{layer_idx:02d}/{metric_name}'] = value
            if row:
                if bool(histogram_present[layer_idx].item() > 0):
                    row['selected_snapshot_index_histogram'] = (
                        histogram_values[layer_idx].detach().cpu().to(torch.int64).tolist()
                    )
                json_layers[str(layer_idx)] = row

        global_row = {}
        additive_metrics = {'snapshot_payload_elements', 'snapshot_fp32_cache_gib'}
        for metric_idx, metric_name in enumerate(metric_names):
            valid_values = [
                float(layer[metric_idx][1])
                for layer in monitor_rows
                if layer[metric_idx][0] > 0
            ]
            if not valid_values:
                continue
            value = sum(valid_values)
            if metric_name not in additive_metrics:
                value /= len(valid_values)
            global_row[metric_name] = value
            logs[f'raven_hils/memory/global/{metric_name}'] = value

        if self.is_world_process_zero() and (json_layers or global_row):
            os.makedirs(self.args.output_dir, exist_ok=True)
            record = {
                'step': int(self.state.global_step),
                'mode': 'train' if self.model.training else 'eval',
                'world_size': torch.distributed.get_world_size() if distributed else 1,
                'memory_topk': int(getattr(config, 'memory_topk', 0)),
                'layers': json_layers,
                'global': global_row,
            }
            path = os.path.join(self.args.output_dir, 'raven_hils_memory_stats.jsonl')
            with open(path, 'a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, separators=(',', ':')) + '\n')

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        self._collect_raven_performance_log(logs)
        self._collect_raven_router_log(logs)
        self._collect_raven_hils_memory_log(logs)
        return super().log(logs, *args, **kwargs)

    @staticmethod
    def _predict_data_collator(batch):
        return {'_data': batch}

    @contextmanager
    def _patch_predict_with_generate(self):
        origin_data_collator = self.data_collator
        self.data_collator = self._predict_data_collator
        packing = self.template.packing
        padding_free = self.template.padding_free
        self.template.packing = False
        self.template.padding_free = False
        try:
            yield
        finally:
            self.template.packing = packing
            self.template.padding_free = padding_free
            self.data_collator = origin_data_collator

    def evaluate(self, *args, **kwargs):
        context = self._patch_predict_with_generate() if self.args.predict_with_generate else nullcontext()
        with context:
            res = super().evaluate(*args, **kwargs)
            gc_collect()
            return res

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.args.predict_with_generate or prediction_loss_only:
            with self.template.forward_context(self.model, inputs):
                return super().prediction_step(
                    model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)
        data_list = inputs['_data']
        labels_list = [InferRequest.remove_response(data['messages']) for data in data_list]
        with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation), self.template.generate_context():
            resp_list = self.infer_engine.infer(
                data_list,
                RequestConfig(max_tokens=self.model.generation_config.max_new_tokens),
                use_tqdm=False,
            )

        response_list = []
        jsonl_cache = []
        device = self.args.device
        for data, resp, labels in zip(data_list, resp_list, labels_list):
            response = resp.choices[0].message.content
            jsonl_cache.append({'response': response, 'labels': labels, **data})
            response_list.append(Serializer.to_tensor(resp.choices[0].message.content).to(device=device))
        self.jsonl_writer.append(jsonl_cache, gather_obj=True)
        labels_list = [Serializer.to_tensor(labels).to(device=device) for labels in labels_list]
        response_list = pad_sequence(response_list, batch_first=True, padding_value=0)
        labels_list = pad_sequence(labels_list, batch_first=True, padding_value=0)
        return None, response_list, labels_list

    def _prepare_inputs(self, inputs):
        args = self.args
        inputs = super()._prepare_inputs(inputs)
        if self.template.sequence_parallel_size > 1:
            sequence_parallel.prepare_inputs(inputs)

        use_logits_to_keep = self.get_use_logits_to_keep(self.template.sequence_parallel_size == 1)
        if use_logits_to_keep:
            self.prepare_logits_to_keep(inputs)
            if args.tuner_backend == 'unsloth' and isinstance(inputs['logits_to_keep'], torch.Tensor):
                inputs['logits_to_keep'] = int(inputs['logits_to_keep'].sum())

        base_model = self.template.get_base_model(self.model)
        forward_params = inspect.signature(base_model.forward).parameters
        if self.model.model_info.is_moe_model and any(key in forward_params
                                                      for key in ['output_router_logits', 'kwargs']):
            HfConfigFactory.set_config_attr(base_model.config, 'router_aux_loss_coef', args.router_aux_loss_coef)
            base_model.router_aux_loss_coef = args.router_aux_loss_coef
            logger.info_once(f'router_aux_loss_coef: {args.router_aux_loss_coef}')
            if args.router_aux_loss_coef > 0:
                inputs['output_router_logits'] = True
        inputs['compute_loss_func'] = self.compute_loss_func
        return inputs

    @staticmethod
    def _hsd_debug_enabled():
        return os.environ.get('HSD_DEBUG_STATS', '').lower() in {'1', 'true', 'yes', 'on'}

    def _clear_hsd_debug_metrics(self, model):
        if not self._hsd_debug_enabled():
            return
        unwrapped_model = self.accelerator.unwrap_model(model)
        for module in unwrapped_model.modules():
            if hasattr(module, '_hsd_debug_stats'):
                module._hsd_debug_stats = {}
            if hasattr(module, '_hsd_debug_stat_counts'):
                module._hsd_debug_stat_counts = {}

    def _collect_hsd_debug_metrics(self, model):
        if not self._hsd_debug_enabled():
            return
        unwrapped_model = self.accelerator.unwrap_model(model)
        mode = 'train' if self.model.training else 'eval'
        metrics = self.custom_metrics[mode]
        self._collect_hsd_anchor_embedding_metrics(unwrapped_model, metrics)
        debug_layer_count = 0
        for module in unwrapped_model.modules():
            hsd_stats = getattr(module, '_hsd_debug_stats', None)
            if not hsd_stats:
                continue
            debug_layer_count += 1
            for key, value in hsd_stats.items():
                metrics[key].update(value.detach() if isinstance(value, torch.Tensor) else value)
            module._hsd_debug_stats = {}
            if hasattr(module, '_hsd_debug_stat_counts'):
                module._hsd_debug_stat_counts = {}
        if debug_layer_count:
            metrics['hsd/debug_layer_count'].update(debug_layer_count)

    def _collect_hsd_anchor_embedding_metrics(self, unwrapped_model, metrics):
        stats = []
        for name, param in unwrapped_model.named_parameters():
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
            return

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
        for key, value in row.items():
            metrics[key].update(value)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = None
        compute_loss_func: Callable = inputs.pop('compute_loss_func', None)
        loss_scale = inputs.pop('loss_scale', None)
        text_position_ids = inputs.pop('text_position_ids', None)
        if text_position_ids is None:
            text_position_ids = inputs.get('position_ids')
        channels = inputs.pop('channel', None)

        if (self.label_smoother is not None or compute_loss_func is not None or loss_scale is not None
                or self.args.enable_dft_loss or self.args.enable_channel_loss
                or self.template.sequence_parallel_size > 1) and 'labels' in inputs:
            if self.args.use_liger_kernel:
                logger.warning_once('The cross_entropy loss function defined in Liger Kernel will not '
                                    'take effect, potentially leading to increased GPU memory consumption.')
            labels = inputs.pop('labels')
        self._clear_hsd_debug_metrics(model)
        outputs = model(**inputs)
        self._collect_hsd_debug_metrics(model)
        if getattr(outputs, 'aux_loss', None) is not None:
            mode = 'train' if self.model.training else 'eval'
            self.custom_metrics[mode]['aux_loss'].update(outputs.aux_loss)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if hasattr(self.args, 'past_index') and self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is None:
            labels = inputs['labels']
            outputs.loss = outputs.loss.to(labels.device)
            # fix https://github.com/huggingface/transformers/issues/34263
            if num_items_in_batch is not None:
                outputs.loss = outputs.loss * ((labels[:, 1:] != -100).sum() / num_items_in_batch)

            if isinstance(outputs, dict) and 'loss' not in outputs:
                raise ValueError(
                    'The model did not return a loss from the inputs, only the following keys: '
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}.")
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        else:
            outputs.loss = None
            if (self.args.enable_dft_loss or loss_scale is not None or self.args.enable_channel_loss
                    or self.template.sequence_parallel_size > 1):
                if self.template.sequence_parallel_size > 1:
                    outputs.loss = per_token_loss_func_sp(outputs, labels, enable_dft_loss=self.args.enable_dft_loss)
                else:
                    outputs.loss = per_token_loss_func(outputs, labels, enable_dft_loss=self.args.enable_dft_loss)

                if loss_scale is not None:
                    loss_scale = torch.roll(loss_scale, shifts=-1, dims=-1).view(-1)
                    outputs.loss = outputs.loss * loss_scale

                if self.args.enable_channel_loss:
                    mode = 'train' if self.model.training else 'eval'
                    metrics = self.custom_metrics[mode]
                    masks = torch.roll(labels, shifts=-1, dims=-1).view(-1) != -100
                    if self.template.padding_free:
                        cu_seqlens = self.get_cu_seqlens(text_position_ids, inputs.get('logits_to_keep'))
                    else:
                        cu_seqlens = torch.arange(0, labels.shape[0] + 1) * labels.shape[1]
                    for i in range(cu_seqlens.shape[0] - 1):
                        channel = None if channels is None else channels[i]
                        slice_ = slice(cu_seqlens[i], cu_seqlens[i + 1])
                        metrics[f'loss_{channel}'].update(outputs.loss[slice_][masks[slice_]])

            unwrapped_model = self.accelerator.unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if compute_loss_func is not None:
                loss = compute_loss_func(
                    outputs, labels, num_items_in_batch=num_items_in_batch, loss_scale=loss_scale, trainer=self)
            elif self.label_smoother is None:
                # Handle the outputs.loss generated by loss_scale.
                if num_items_in_batch is None:
                    num_items_in_batch = (labels[:, 1:] != -100).sum()
                loss = outputs.loss.sum() / num_items_in_batch
            else:
                if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    loss = self.label_smoother(outputs, labels)

            if self.model.model_info.is_moe_model and self.args.router_aux_loss_coef is not None:
                aux_loss = outputs.get('aux_loss')
                if aux_loss is not None:
                    if num_items_in_batch is not None:
                        aux_loss = aux_loss * ((labels[:, 1:] != -100).sum() / num_items_in_batch)
                    loss = loss + self.args.router_aux_loss_coef * aux_loss.to(loss.device)

        if getattr(self.args, 'average_tokens_across_devices',
                   False) and self.model_accepts_loss_kwargs and num_items_in_batch is not None:
            loss *= self.accelerator.num_processes

        if (outputs.logits is not None and labels is not None and self.args.tuner_backend != 'unsloth'):
            cu_seqlens = None
            if self.template.padding_free and self.args.acc_strategy == 'seq':
                cu_seqlens = self.get_cu_seqlens(text_position_ids, inputs.get('logits_to_keep'))
            # Liger does not have logits
            # Unsloth has a bug with output logits
            self._compute_acc(outputs, labels, cu_seqlens=cu_seqlens)
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, *args, **kwargs):
        with self.template.forward_context(self.model, inputs):
            return super().training_step(model, inputs, *args, **kwargs)

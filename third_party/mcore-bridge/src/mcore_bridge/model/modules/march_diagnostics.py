# Copyright (c) ModelScope Contributors. All rights reserved.
"""Opt-in training diagnostics; parameters and mutable metric stores stay on attention."""

from contextlib import contextmanager
import math
import os
import re
from typing import Callable, Optional

import torch
from megatron.core import parallel_state

from .march_runtime import (
    _env_bool,
    _parse_layer_selector,
    _precision_trace_internal_layer_allowed,
    _rank_allowed_from_env,
)


class MarchDiagnosticsMixin:
    @staticmethod
    def _hsd_debug_enabled() -> bool:
        return _env_bool("HSD_DEBUG_STATS", False)

    @staticmethod
    def _hsd_debug_verbose_enabled() -> bool:
        return _env_bool("HSD_DEBUG_VERBOSE_STATS", False)

    @staticmethod
    def _hsd_production_health_enabled() -> bool:
        """Enable scalar-only health metrics without large-tensor reductions."""
        return _env_bool("HSD_PRODUCTION_HEALTH_STATS", False)

    @staticmethod
    def _hsd_profile_enabled() -> bool:
        return _env_bool("HSD_PROFILE_TIMING", False)

    def _debug_key(self, name: str) -> str:
        layer_number = getattr(self, "layer_number", None)
        if layer_number is None:
            return f"hsd/{name}"
        return f"hsd/layer_{layer_number}/{name}"

    @staticmethod
    def _debug_scalar_aggregation_mode(name: str) -> str:
        metric_name = name.rsplit("/", 1)[-1]
        if metric_name.endswith("_finite"):
            return "min"
        if metric_name.endswith("_absmax") or metric_name.endswith("_max"):
            return "max"
        return "mean"

    def _record_debug_scalar(self, name: str, value) -> None:
        if not self._hsd_debug_enabled():
            return
        if isinstance(value, torch.Tensor):
            value = value.detach()
            if value.numel() != 1:
                value = value.float().mean()
            value = value.float().reshape(())
        else:
            value = torch.tensor(
                float(value), dtype=torch.float32, device=self.A_log.device
            )
        key = self._debug_key(name)
        old_value = self._hsd_debug_stats.get(key)
        mode = self._debug_scalar_aggregation_mode(name)
        if isinstance(old_value, torch.Tensor):
            old_value = old_value.to(device=value.device, dtype=value.dtype)
            if mode == "min":
                value = torch.minimum(old_value, value)
            elif mode == "max":
                value = torch.maximum(old_value, value)
            else:
                counts = getattr(self, "_hsd_debug_stat_counts", None)
                if counts is None:
                    counts = {}
                    self._hsd_debug_stat_counts = counts
                count = int(counts.get(key, 1))
                value = old_value + (value - old_value) / float(count + 1)
                counts[key] = count + 1
        else:
            counts = getattr(self, "_hsd_debug_stat_counts", None)
            if counts is None:
                counts = {}
                self._hsd_debug_stat_counts = counts
            counts[key] = 1
        self._hsd_debug_stats[key] = value

    def _profile_key(self, name: str) -> str:
        return self._debug_key(f"profile/{name}")

    def _record_profile_scalar(self, name: str, value) -> None:
        if not self._hsd_profile_enabled():
            return
        if isinstance(value, torch.Tensor):
            value = value.detach()
            if value.numel() != 1:
                value = value.float().mean()
            value = value.float().reshape(())
        else:
            value = torch.tensor(
                float(value), dtype=torch.float32, device=self.A_log.device
            )
        key = self._profile_key(name)
        old_value = self._hsd_profile_stats.get(key)
        if isinstance(old_value, torch.Tensor):
            old_value = old_value.to(device=value.device, dtype=value.dtype)
            value = old_value + value
        self._hsd_profile_stats[key] = value

    @contextmanager
    def _profile_cuda_timer(self, name: str):
        if not self._hsd_profile_enabled() or not torch.cuda.is_available():
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            end.synchronize()
            self._record_profile_scalar(f"{name}_ms", float(start.elapsed_time(end)))

    @staticmethod
    def _tensor_rms(value: torch.Tensor) -> torch.Tensor:
        if value.numel() == 0:
            return value.new_tensor(0.0)
        return value.pow(2).mean().sqrt()

    def _tensor_debug_stats(self, prefix: str, tensor: torch.Tensor) -> None:
        if not self._hsd_debug_enabled() or not self._hsd_debug_verbose_enabled():
            return
        with torch.no_grad():
            value = tensor.detach().float()
            self._record_debug_scalar(f"{prefix}_norm", value.norm())
            self._record_debug_scalar(f"{prefix}_rms", self._tensor_rms(value))
            self._record_debug_scalar(f"{prefix}_absmax", value.abs().max())
            self._record_debug_scalar(
                f"{prefix}_finite", value.isfinite().all().float()
            )

    def _register_debug_grad(
        self,
        name: str,
        tensor: Optional[torch.Tensor],
        transform: Optional[Callable[[torch.Tensor], Optional[torch.Tensor]]] = None,
    ) -> None:
        if not isinstance(tensor, torch.Tensor) or not tensor.requires_grad:
            return
        verbose_debug = self._hsd_debug_enabled() and self._hsd_debug_verbose_enabled()
        activation_grad_debug = _env_bool("HSD_GRAD_DEBUG_ACTIVATIONS", False)
        if not verbose_debug and not activation_grad_debug:
            return
        activation_grad_layers = _parse_layer_selector(
            os.environ.get("HSD_GRAD_DEBUG_ACTIVATION_LAYERS")
        )
        activation_grad_regex = os.environ.get("HSD_GRAD_DEBUG_ACTIVATION_REGEX")
        activation_grad_pattern = (
            re.compile(activation_grad_regex) if activation_grad_regex else None
        )

        def _hook(grad):
            with torch.no_grad():
                grad_to_record = transform(grad) if transform is not None else grad
                if grad_to_record is None:
                    return
                value = grad_to_record.detach().float()
                if verbose_debug:
                    self._record_debug_scalar(f"grad/{name}_norm", value.norm())
                    self._record_debug_scalar(
                        f"grad/{name}_rms", self._tensor_rms(value)
                    )
                    self._record_debug_scalar(f"grad/{name}_absmax", value.abs().max())
                    self._record_debug_scalar(
                        f"grad/{name}_finite", value.isfinite().all().float()
                    )
                layer_number = getattr(self, "layer_number", "unknown")
                layer_allowed = activation_grad_layers is None or (
                    layer_number != "unknown"
                    and int(layer_number) in activation_grad_layers
                )
                name_allowed = (
                    activation_grad_pattern is None
                    or activation_grad_pattern.search(name)
                )
                if (
                    activation_grad_debug
                    and layer_allowed
                    and name_allowed
                    and _rank_allowed_from_env("HSD_GRAD_DEBUG_ACTIVATION_RANKS")
                ):
                    rank = (
                        torch.distributed.get_rank()
                        if torch.distributed.is_available()
                        and torch.distributed.is_initialized()
                        else 0
                    )
                    print(
                        f"[INFO:swift] [hsd-activation-grad] layer={layer_number} rank={rank} name={name} "
                        f"norm={value.norm().item():.6e} rms={self._tensor_rms(value).item():.6e} "
                        f"absmax={value.abs().max().item() if value.numel() else 0.0:.6e} "
                        f"finite={bool(value.isfinite().all().item())} shape={tuple(grad.shape)}",
                        flush=True,
                    )
            return grad

        tensor.register_hook(_hook)

    def _trace_precision_layer_tensor(
        self, name: str, tensor: Optional[torch.Tensor], category: str = "layer_io"
    ) -> None:
        if not isinstance(tensor, torch.Tensor):
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
        layer_number = int(getattr(self, "layer_number", 0) or 0)
        tracer.trace_layer_tensor(
            f"layer_{layer_number}/{name}",
            tensor,
            step=precision_trace_runtime_step(),
            category=category,
            layer_number=layer_number,
        )

    def _trace_precision_internal_tensor(
        self, name: str, tensor: Optional[torch.Tensor]
    ) -> None:
        if not isinstance(tensor, torch.Tensor) or not _env_bool(
            "MEGATRON_PRECISION_TRACE_GDN_INTERNALS", False
        ):
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
        layer_number = int(getattr(self, "layer_number", 0) or 0)
        if not _precision_trace_internal_layer_allowed(layer_number):
            return
        tracer.trace_tensor(
            f"layer_{layer_number}/{name}",
            tensor,
            step=precision_trace_runtime_step(),
            category="hsd_internal",
            extra={"layer_number": layer_number},
        )

    def _trace_precision_internal_backward_tensor(
        self,
        name: str,
        tensor: Optional[torch.Tensor],
        transform: Optional[Callable[[torch.Tensor], Optional[torch.Tensor]]] = None,
    ) -> None:
        if (
            not isinstance(tensor, torch.Tensor)
            or not _env_bool("MEGATRON_PRECISION_TRACE_GDN_INTERNALS", False)
            or not _env_bool("MEGATRON_PRECISION_TRACE_INTERNAL_BACKWARD", False)
        ):
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
        layer_number = int(getattr(self, "layer_number", 0) or 0)
        if not _precision_trace_internal_layer_allowed(layer_number):
            return
        tracer.trace_tensor_backward(
            f"layer_{layer_number}/{name}",
            tensor,
            step=precision_trace_runtime_step(),
            category="hsd_internal_bwd",
            layer_number=layer_number,
            transform=transform,
        )

    @staticmethod
    def _select_bld_masked(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if tensor.ndim < 2 or mask.ndim != 2:
            raise ValueError(
                f"Expected tensor [batch, seq, ...] and mask [batch, seq], got {tensor.shape}, {mask.shape}."
            )
        batch, seq_len = mask.shape
        if tuple(tensor.shape[:2]) != (batch, seq_len):
            raise ValueError(
                f"Tensor prefix {tuple(tensor.shape[:2])} must match mask {(batch, seq_len)}."
            )
        count = int(mask[0].sum().item())
        return tensor[mask].view(batch, count, *tensor.shape[2:])

    def _record_hsd_history_debug_stats(
        self,
        text_o: torch.Tensor,
        history_out: torch.Tensor,
        history_scale: torch.Tensor,
        anchor_states: torch.Tensor,
        router_q: torch.Tensor,
        anchor_k: torch.Tensor,
    ) -> None:
        if not self._hsd_debug_enabled():
            return
        with torch.no_grad():
            history_scale_f = history_scale.detach().float()
            if (
                self._hsd_production_health_enabled()
                and not self._hsd_debug_verbose_enabled()
            ):
                # Production health uses three Boolean-only reductions over
                # the actual local output, global-read contribution, and HSD
                # state.  It deliberately avoids float copies and all
                # norm/RMS/router diagnostics; the trainer performs periodic
                # full HSD parameter/gradient scans separately.
                self._record_debug_scalar("history_scale", history_scale_f)
                self._record_debug_scalar(
                    "history_scale_finite",
                    history_scale_f.isfinite().all().float(),
                )
                self._record_debug_scalar(
                    "text_o_finite",
                    text_o.detach().isfinite().all().float(),
                )
                self._record_debug_scalar(
                    "history_out_finite",
                    history_out.detach().isfinite().all().float(),
                )
                self._record_debug_scalar(
                    "anchor_states_finite",
                    anchor_states.detach().isfinite().all().float(),
                )
                return
            history_gate_param = self._history_gate_param_tensor(
                device=history_out.device
            )
            history_gate_schedule = self._history_gate_schedule_tensor(
                device=history_out.device
            )
            history_out_grad_scale = self._history_out_grad_scale_tensor(
                device=history_out.device
            )
            self._record_debug_scalar(
                "history_base_scale", float(self.hsd_history_scale)
            )
            self._record_debug_scalar(
                "history_out_rmsnorm_enabled",
                float(self.hsd_history_out_rmsnorm_weight is not None),
            )
            self._record_debug_scalar(
                "history_gate", self._history_gate_tensor(device=history_out.device)
            )
            self._record_debug_scalar("history_scale", history_scale_f)
            self._record_debug_scalar(
                "learnable_history_scale_enabled",
                float(self.hsd_history_gate_logit is not None),
            )
            self._record_debug_scalar(
                "history_gate_schedule_enabled",
                float(self.hsd_history_gate_schedule_enabled),
            )
            self._record_debug_scalar(
                "history_out_grad_scale_enabled",
                float(history_out_grad_scale is not None),
            )
            if history_gate_param is not None:
                self._record_debug_scalar("history_gate_param", history_gate_param)
            if history_gate_schedule is not None:
                self._record_debug_scalar(
                    "history_gate_schedule", history_gate_schedule
                )
            if history_out_grad_scale is not None:
                self._record_debug_scalar(
                    "history_out_grad_scale", history_out_grad_scale
                )
            text = text_o.detach().float()
            history_raw = history_out.detach().float()
            history = history_raw * history_scale_f
            eps = torch.finfo(torch.float32).eps
            text_plus_history = text + history
            text_norm = text.norm()
            history_norm = history.norm()
            sum_norm = text_plus_history.norm()
            text_rms = self._tensor_rms(text)
            text_plus_history_rms = self._tensor_rms(text_plus_history)
            self._record_debug_scalar(
                "history_text_norm_ratio", history_norm / text_norm.clamp_min(eps)
            )
            self._record_debug_scalar(
                "history_to_sum_norm_ratio", history_norm / sum_norm.clamp_min(eps)
            )
            self._record_debug_scalar(
                "text_plus_history_text_rms_ratio",
                text_plus_history_rms / text_rms.clamp_min(eps),
            )
            history_tokens = history.flatten(2)
            text_tokens = text.flatten(2)
            token_history_norm = history_tokens.norm(dim=-1)
            token_text_norm = text_tokens.norm(dim=-1)
            history_token_ratio = token_history_norm / token_text_norm.clamp_min(eps)
            self._record_debug_scalar(
                "history_effective_token_share_gt_0p10",
                (history_token_ratio > 0.10).float().mean(),
            )
            flat_history_token_ratio = history_token_ratio.flatten()
            p90_idx = min(
                flat_history_token_ratio.numel() - 1,
                max(0, math.ceil(0.90 * flat_history_token_ratio.numel()) - 1),
            )
            self._record_debug_scalar(
                "history_token_ratio_p90",
                flat_history_token_ratio.kthvalue(p90_idx + 1).values,
            )
            if not self._hsd_debug_verbose_enabled():
                return
            self._record_debug_scalar("history_raw_norm", history_raw.norm())
            self._record_debug_scalar("history_raw_rms", self._tensor_rms(history_raw))
            self._record_debug_scalar("history_norm", history_norm)
            self._record_debug_scalar("history_rms", self._tensor_rms(history))
            self._record_debug_scalar("text_o_norm", text_norm)
            self._record_debug_scalar("text_o_rms", text_rms)
            self._record_debug_scalar("text_plus_history_norm", sum_norm)
            self._record_debug_scalar("text_plus_history_rms", text_plus_history_rms)
            self._record_debug_scalar("history_absmax", history.abs().max())
            self._record_debug_scalar("text_o_absmax", text.abs().max())

            history_flat = history.reshape(-1)
            text_flat = text.reshape(-1)
            self._record_debug_scalar(
                "history_text_cos",
                torch.dot(history_flat, text_flat)
                / history_norm.clamp_min(eps)
                / text_norm.clamp_min(eps),
            )
            sum_tokens = text_plus_history.flatten(2)
            token_cos = (history_tokens * text_tokens).sum(dim=-1)
            token_cos = (
                token_cos
                / token_history_norm.clamp_min(eps)
                / token_text_norm.clamp_min(eps)
            )
            self._record_debug_scalar("history_token_cos_mean", token_cos.mean())
            self._record_debug_scalar(
                "history_token_norm_mean", token_history_norm.mean()
            )
            self._record_debug_scalar(
                "history_token_norm_max", token_history_norm.max()
            )
            self._record_debug_scalar("text_o_token_norm_mean", token_text_norm.mean())
            self._record_debug_scalar("text_o_token_norm_max", token_text_norm.max())
            token_history_rms = history_tokens.pow(2).mean(dim=-1).sqrt()
            token_text_rms = text_tokens.pow(2).mean(dim=-1).sqrt()
            token_sum_rms = sum_tokens.pow(2).mean(dim=-1).sqrt()
            self._record_debug_scalar(
                "history_token_rms_mean", token_history_rms.mean()
            )
            self._record_debug_scalar("history_token_rms_max", token_history_rms.max())
            self._record_debug_scalar("text_o_token_rms_mean", token_text_rms.mean())
            self._record_debug_scalar("text_o_token_rms_max", token_text_rms.max())
            self._record_debug_scalar(
                "text_plus_history_token_rms_mean", token_sum_rms.mean()
            )
            self._record_debug_scalar(
                "text_plus_history_token_rms_max", token_sum_rms.max()
            )

            self._tensor_debug_stats("anchor_states", anchor_states)
            self._tensor_debug_stats("router_q", router_q)
            self._tensor_debug_stats("anchor_k", anchor_k)

    def _record_hsd_router_debug_stats(
        self,
        weights: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        anchor_k: Optional[torch.Tensor] = None,
        valid_prefix: Optional[torch.Tensor] = None,
        has_valid: Optional[torch.Tensor] = None,
        boundaries: Optional[torch.Tensor] = None,
        token_index: Optional[torch.Tensor] = None,
        null_prob: Optional[torch.Tensor] = None,
        anchor_mass: Optional[torch.Tensor] = None,
        record_debug_stats: Optional[bool] = None,
        scalar_stats: Optional[dict[str, torch.Tensor]] = None,
        monitor_contract: Optional[dict[str, object]] = None,
    ) -> None:
        if record_debug_stats is False or not self._hsd_debug_enabled():
            return
        # Production health normally rejects router diagnostics because the
        # legacy path materializes [T,H,M] tensors.  The TileOnline monitor is
        # different: it supplies an explicitly contracted dictionary of
        # zero-dimensional, blockwise-reduced scalars.  Admit only that form
        # while keeping every materialized diagnostic disabled.
        if (
            self._hsd_production_health_enabled()
            and not self._hsd_debug_verbose_enabled()
            and scalar_stats is None
        ):
            return
        if scalar_stats is not None:
            # The tile-online production collector owns the sampling and
            # blockwise reductions.  Accept scalars only at this boundary so
            # a model layer cannot accidentally retain [T,H,M] diagnostics.
            if not isinstance(monitor_contract, dict):
                raise RuntimeError(
                    "HSD scalar router debug stats require a monitor contract."
                )
            expected_contract = {
                "contract_version": 1,
                "sampling_policy": "uniform_adjacent_pairs_v1",
                "sampling_policy_id": 1,
                "scalar_only": True,
            }
            mismatches = {
                key: (monitor_contract.get(key), value)
                for key, value in expected_contract.items()
                if monitor_contract.get(key) != value
            }
            if mismatches:
                raise RuntimeError(
                    f"HSD scalar router monitor contract mismatch: {mismatches}"
                )
            for name, value in scalar_stats.items():
                if (
                    not isinstance(name, str)
                    or not isinstance(value, torch.Tensor)
                    or value.ndim != 0
                ):
                    raise RuntimeError(
                        "HSD scalar router monitor accepts only named zero-dimensional tensors; "
                        f"got {name!r}={type(value).__name__} "
                        f"shape={getattr(value, 'shape', None)}."
                    )
                self._record_debug_scalar(name, value)
            return
        required_tensors = {
            "weights": weights,
            "logits": logits,
            "anchor_k": anchor_k,
            "valid_prefix": valid_prefix,
            "has_valid": has_valid,
            "boundaries": boundaries,
            "token_index": token_index,
        }
        missing = [
            name
            for name, value in required_tensors.items()
            if not isinstance(value, torch.Tensor)
        ]
        if missing:
            raise RuntimeError(
                "HSD materialized router debug stats are missing tensors: "
                + ", ".join(missing)
            )
        # Type narrowing for the legacy/reference debug implementation below.
        assert weights is not None
        assert logits is not None
        assert anchor_k is not None
        assert valid_prefix is not None
        assert has_valid is not None
        assert boundaries is not None
        assert token_index is not None
        with torch.no_grad():
            weights_f = weights.detach().float()
            valid_mask = has_valid.squeeze(-1).expand_as(weights_f[..., 0])
            if not bool(valid_mask.any().item()):
                return
            eps = torch.finfo(torch.float32).tiny
            anchor_mass_f = None
            if null_prob is not None:
                null_prob_f = null_prob.detach().float()
                if null_prob_f.ndim == 4 and null_prob_f.shape[-1] == 1:
                    null_prob_f = null_prob_f.squeeze(-1)
                if tuple(null_prob_f.shape) != tuple(valid_mask.shape):
                    raise RuntimeError(
                        "HSD router debug null probability shape mismatch: "
                        f"got {tuple(null_prob_f.shape)}, expected {tuple(valid_mask.shape)} "
                        "or the same shape with a trailing singleton dimension."
                    )
                if anchor_mass is None:
                    anchor_mass_f = weights_f.sum(dim=-1, keepdim=True)
                else:
                    anchor_mass_f = anchor_mass.detach().float()
                valid_null_prob = null_prob_f[valid_mask]
                valid_anchor_mass = anchor_mass_f.squeeze(-1)[valid_mask]
                self._record_debug_scalar("router_null_prob", valid_null_prob.mean())
                self._record_debug_scalar("router_null_prob_max", valid_null_prob.max())
                self._record_debug_scalar(
                    "router_anchor_mass", valid_anchor_mass.mean()
                )
                self._record_debug_scalar(
                    "router_history_head_share_gt_0p10",
                    (valid_anchor_mass > 0.10).float().mean(),
                )
                token_valid_mask = valid_mask.any(dim=-1)
                if bool(token_valid_mask.any().item()):
                    anchor_mass_bth = anchor_mass_f.squeeze(-1)
                    valid_mask_f = valid_mask.float()
                    token_anchor_mass_max = (
                        anchor_mass_bth.masked_fill(~valid_mask, 0.0).max(dim=-1).values
                    )
                    token_anchor_mass_mean = (anchor_mass_bth * valid_mask_f).sum(
                        dim=-1
                    ) / valid_mask_f.sum(dim=-1).clamp_min(1.0)
                    self._record_debug_scalar(
                        "router_history_token_share_any_head_gt_0p10",
                        (token_anchor_mass_max[token_valid_mask] > 0.10).float().mean(),
                    )
                    self._record_debug_scalar(
                        "router_history_token_share_mean_head_gt_0p10",
                        (token_anchor_mass_mean[token_valid_mask] > 0.10)
                        .float()
                        .mean(),
                    )
            anchor_k_norm = torch.linalg.vector_norm(
                anchor_k.detach().float(), dim=-1
            ).mean(dim=(0, 2))
            if anchor_k_norm.numel() > 0:
                self._record_debug_scalar(
                    "router_anchor_k_norm_oldest", anchor_k_norm[0]
                )
                self._record_debug_scalar(
                    "router_anchor_k_norm_latest", anchor_k_norm[-1]
                )
                self._record_debug_scalar(
                    "router_anchor_k_norm_max_over_mean",
                    anchor_k_norm.max() / anchor_k_norm.mean().clamp_min(eps),
                )
            logits_f = logits.detach().float()
            pre_weights = torch.softmax(
                torch.where(has_valid, logits_f, torch.zeros_like(logits_f)), dim=-1
            )
            pre_weights = torch.where(
                has_valid, pre_weights, torch.zeros_like(pre_weights)
            )
            top4 = min(4, weights_f.shape[-1])
            top4_mass = pre_weights.topk(k=top4, dim=-1).values.sum(dim=-1)
            pre_entropy = -(pre_weights * pre_weights.clamp_min(eps).log()).sum(dim=-1)
            pre_top1_prob = pre_weights.max(dim=-1).values
            top2_logits = logits_f.topk(k=min(2, logits_f.shape[-1]), dim=-1).values
            if top2_logits.shape[-1] == 1:
                top1_margin = torch.zeros_like(top2_logits[..., 0])
            else:
                top1_margin = top2_logits[..., 0] - top2_logits[..., 1]
            weighted_boundary = (weights_f * boundaries.float().view(1, 1, 1, -1)).sum(
                dim=-1
            )
            weighted_age = token_index.float().view(1, -1, 1) - weighted_boundary
            entropy = -(weights_f * weights_f.clamp_min(eps).log()).sum(dim=-1)
            top1_prob = weights_f.max(dim=-1).values
            top1_idx = weights_f.argmax(dim=-1)
            if null_prob is not None:
                anchor_top1_prob = top1_prob
                if anchor_mass_f is not None:
                    anchor_top1_prob = anchor_top1_prob * anchor_mass_f.squeeze(-1)
                null_top1 = null_prob_f >= anchor_top1_prob
                self._record_debug_scalar(
                    "router_null_top1_share", null_top1[valid_mask].float().mean()
                )
            valid_entropy = entropy[valid_mask]
            valid_top1_prob = top1_prob[valid_mask]
            valid_top4_mass = top4_mass[valid_mask]
            valid_pre_entropy = pre_entropy[valid_mask]
            valid_pre_top1_prob = pre_top1_prob[valid_mask]
            valid_weighted_age = weighted_age[valid_mask]
            valid_counts_per_token = valid_prefix.sum(dim=-1)
            valid_counts = valid_counts_per_token.float()
            margin_mask = valid_mask & (
                valid_counts.to(device=valid_mask.device).view(1, -1, 1) >= 2
            )
            valid_counts = valid_counts[valid_counts > 0]
            valid_top1_idx = top1_idx[valid_mask].long()
            valid_row_count = valid_top1_idx.numel()
            if valid_row_count > 0:
                row_count = weights_f.new_tensor(float(valid_row_count))
                top1_usage = (
                    torch.bincount(
                        valid_top1_idx,
                        minlength=weights_f.shape[-1],
                    ).to(device=weights_f.device, dtype=weights_f.dtype)
                    / row_count
                )
                top1_usage_entropy = -(
                    top1_usage * top1_usage.clamp_min(eps).log()
                ).sum()
                mass_usage = weights_f[valid_mask].sum(dim=0) / row_count
                mass_usage_entropy = -(
                    mass_usage * mass_usage.clamp_min(eps).log()
                ).sum()
                valid_counts_bth = (
                    valid_counts_per_token.to(device=top1_idx.device, dtype=torch.long)
                    .view(1, -1, 1)
                    .expand_as(top1_idx)
                )
                valid_latest_idx = valid_counts_bth[valid_mask] - 1
                self._record_debug_scalar(
                    "router_top1_usage_effective_anchor_count",
                    top1_usage_entropy.exp(),
                )
                self._record_debug_scalar("router_top1_max_share", top1_usage.max())
                self._record_debug_scalar(
                    "router_mass_usage_effective_anchor_count",
                    mass_usage_entropy.exp(),
                )
                self._record_debug_scalar("router_mass_max_share", mass_usage.max())
                self._record_debug_scalar(
                    "router_top1_oldest_share", (valid_top1_idx == 0).float().mean()
                )
                self._record_debug_scalar(
                    "router_top1_latest_share",
                    (valid_top1_idx == valid_latest_idx).float().mean(),
                )
                token_valid_mask = valid_mask.any(dim=-1)
                if bool(token_valid_mask.any().item()):
                    head_all_same = (top1_idx == top1_idx[..., :1]).all(dim=-1)
                    self._record_debug_scalar(
                        "router_head_all_same_share",
                        head_all_same[token_valid_mask].float().mean(),
                    )
                if top1_idx.shape[1] > 1:
                    switch_mask = valid_mask[:, 1:, :] & valid_mask[:, :-1, :]
                    if bool(switch_mask.any().item()):
                        switched = top1_idx[:, 1:, :] != top1_idx[:, :-1, :]
                        self._record_debug_scalar(
                            "router_top1_switch_rate",
                            switched[switch_mask].float().mean(),
                        )
            self._record_debug_scalar("router_entropy", valid_entropy.mean())
            self._record_debug_scalar("router_top1_prob", valid_top1_prob.mean())
            self._record_debug_scalar("router_valid_anchor_count", valid_counts.mean())
            self._record_debug_scalar(
                "router_effective_anchor_count", valid_entropy.exp().mean()
            )
            self._record_debug_scalar("router_top4_mass", valid_top4_mass.mean())
            self._record_debug_scalar(
                "router_pre_topk_entropy", valid_pre_entropy.mean()
            )
            self._record_debug_scalar(
                "router_pre_topk_effective_anchor_count", valid_pre_entropy.exp().mean()
            )
            self._record_debug_scalar(
                "router_pre_topk_top1_prob", valid_pre_top1_prob.mean()
            )
            self._record_debug_scalar("router_weighted_age", valid_weighted_age.mean())
            if bool(margin_mask.any().item()):
                self._record_debug_scalar(
                    "router_top1_margin", top1_margin[margin_mask].mean()
                )

    def _record_persistent_memory_state_stats(
        self,
        anchor_states: torch.Tensor,
        persistent_anchor_slots: Optional[torch.Tensor],
    ) -> None:
        if persistent_anchor_slots is None or self.persistent_memory_state is None:
            return
        with torch.no_grad():
            state_rms = self._tensor_rms(self.persistent_memory_state.detach().float())
            self._set_persistent_memory_metric("state_rms", state_rms)
            if self._hsd_debug_enabled():
                self._record_debug_scalar("persistent_memory_state_rms", state_rms)
            dynamic = anchor_states[
                :, ~persistent_anchor_slots.to(device=anchor_states.device)
            ]
            if dynamic.numel() > 0:
                dynamic_rms = (
                    dynamic.detach().float().pow(2).mean(dim=(-3, -2, -1)).sqrt()
                )
                dynamic_median = dynamic_rms.flatten().median()
                self._set_persistent_memory_metric(
                    "dynamic_state_rms_median", dynamic_median
                )
                if self._hsd_debug_enabled():
                    self._record_debug_scalar(
                        "dynamic_cached_state_rms_median", dynamic_median
                    )

    def _record_persistent_memory_route_stats(
        self,
        *,
        memory_logits: torch.Tensor,
        memory_prob: torch.Tensor,
        weighted_memory_read: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            memory_logit = memory_logits.detach().float().mean()
            memory_probability = memory_prob.detach().float().mean()
            memory_read_rms = self._tensor_rms(weighted_memory_read.detach().float())
            self._set_persistent_memory_metric("memory_logit", memory_logit)
            self._set_persistent_memory_metric("memory_probability", memory_probability)
            self._set_persistent_memory_metric("memory_read_rms", memory_read_rms)
            if self._hsd_debug_enabled():
                self._record_debug_scalar("persistent_memory_logit", memory_logit)
                self._record_debug_scalar(
                    "persistent_memory_probability", memory_probability
                )
                self._record_debug_scalar("persistent_memory_read_rms", memory_read_rms)

    def _record_layout_debug_stats(
        self,
        local_seq_len: int,
        full_seq_len: int,
        full_hidden_seq_len: int,
        anchor_mask: torch.Tensor,
        text_len: int,
        anchor_count: int,
        projected_seq_len: Optional[int] = None,
    ) -> None:
        if not self._hsd_debug_enabled() or (
            not self._hsd_debug_verbose_enabled()
            and not self._hsd_production_health_enabled()
        ):
            return
        self._record_debug_scalar("layout/local_seq_len", local_seq_len)
        self._record_debug_scalar("layout/full_seq_len", full_seq_len)
        self._record_debug_scalar("layout/full_hidden_seq_len", full_hidden_seq_len)
        self._record_debug_scalar("layout/text_len", text_len)
        self._record_debug_scalar("layout/anchor_count", anchor_count)
        self._record_debug_scalar("layout/sp_size", self.sp_size)
        self._record_debug_scalar("layout/tp_size", self.tp_size)
        self._record_debug_scalar(
            "layout/tp_rank", parallel_state.get_tensor_model_parallel_rank()
        )
        if projected_seq_len is not None:
            self._record_debug_scalar("layout/projected_seq_len", projected_seq_len)
        if self.sp_size > 1 and full_seq_len % self.sp_size == 0:
            shard_len = full_seq_len // self.sp_size
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            local_mask = anchor_mask[:, tp_rank * shard_len : (tp_rank + 1) * shard_len]
            self._record_debug_scalar(
                "layout/local_text_count", int((~local_mask[0]).sum().item())
            )
            self._record_debug_scalar(
                "layout/local_anchor_count", int(local_mask[0].sum().item())
            )

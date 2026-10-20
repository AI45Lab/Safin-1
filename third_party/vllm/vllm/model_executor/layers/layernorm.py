# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom normalization layers."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import kernels
import vllm.kernels  # noqa: F401
from vllm import envs, ir
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.batch_invariant import (
    rms_norm_batch_invariant,
    use_hsd_moe_batch_invariant,
)

logger = init_logger(__name__)

_RMSNORM_OVERRIDE_LOGGED: set[str] = set()


def _force_torch_rms_norm() -> bool:
    return os.environ.get("VLLM_FORCE_TORCH_RMSNORM", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _force_batch_invariant_gemma_rms_norm() -> bool:
    return (
        envs.VLLM_BATCH_INVARIANT
        or use_hsd_moe_batch_invariant("decoder_norm")
        or os.environ.get(
            "VLLM_FORCE_BATCH_INVARIANT_GEMMA_RMSNORM", ""
        ).lower()
        in {
            "1",
            "true",
            "yes",
            "on",
        }
    )


def _log_rms_norm_override_once(op_name: str) -> None:
    if torch.compiler.is_compiling():
        return
    if op_name in _RMSNORM_OVERRIDE_LOGGED:
        return
    _RMSNORM_OVERRIDE_LOGGED.add(op_name)
    logger.info("Using debug RMSNorm override path for %s", op_name)


def _torch_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    variance_epsilon: float,
    variance_size: int | None = None,
) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.float()
    x_var = x if variance_size is None else x[..., :variance_size]
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + variance_epsilon)
    if weight is not None:
        x = x.to(weight.dtype) * weight
    return x.to(orig_dtype)


def _torch_fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None,
    variance_epsilon: float,
    variance_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = x.dtype
    x = x.float() + residual.float()
    new_residual = x.to(orig_dtype)
    x_var = x if variance_size is None else x[..., :variance_size]
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + variance_epsilon)
    if weight is not None:
        x = x.to(weight.dtype) * weight
    return x.to(orig_dtype), new_residual


def _torch_gemma_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = x.dtype
    weight = weight.float() + 1.0
    if residual is not None:
        x = (
            x.float() + residual.float()
            if orig_dtype == torch.float16
            else x + residual
        )
        residual = x
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    out = x_float * torch.rsqrt(variance + variance_epsilon)
    out = out * weight
    return out.to(orig_dtype) if residual is None else (out.to(orig_dtype), residual)


def _batch_invariant_gemma_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = x.dtype
    if residual is not None:
        x = (
            x.float() + residual.float()
            if orig_dtype == torch.float16
            else x + residual
        )
        residual = x
    out = rms_norm_batch_invariant(x, weight.float() + 1.0, variance_epsilon)
    return out.to(orig_dtype) if residual is None else (out.to(orig_dtype), residual)


def poly_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, variance_epsilon: float
) -> torch.Tensor:
    from vllm import _custom_ops as ops

    out = torch.empty_like(x)
    ops.poly_norm(  # type: ignore[attr-defined]
        out,
        x,
        weight,
        bias,
        variance_epsilon,
    )
    return out


# --8<-- [start:rms_norm]
@CustomOp.register("rms_norm")
class RMSNorm(CustomOp):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    # --8<-- [end:rms_norm]

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: int | None = None,
        has_weight: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )
        weight_dtype = dtype or torch.get_default_dtype()
        self.has_weight = has_weight
        self.weight = torch.ones(hidden_size, dtype=weight_dtype)
        if self.has_weight:
            self.weight = nn.Parameter(self.weight)

        # Do not pass identity weight to native implementation (causes issue on TPU).
        # Other implementations require weight to be passed even if all ones.
        # Cheat and predict if native will be dispatched to:
        #  1) if native is first in priority list
        #  2) if variance_size_override is given (only supported by native impl)
        # TODO(luka): address weight passing inconsistency:
        # https://github.com/vllm-project/vllm/issues/39370
        priority = get_current_vllm_config().kernel_config.ir_op_priority
        var_override = self.variance_size_override is not None
        native_rms_norm = priority.rms_norm[0] == "native" or var_override
        native_add_rms_norm = priority.fused_add_rms_norm[0] == "native" or var_override
        self.pass_weight = self.has_weight or not native_rms_norm
        self.pass_weight_add = self.has_weight or not native_add_rms_norm

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        if residual is None:
            return ir.ops.rms_norm(
                x,
                self.weight.data if self.pass_weight else None,
                self.variance_epsilon,
                self.variance_size_override,
            )
        else:
            return ir.ops.fused_add_rms_norm.maybe_inplace(
                x,
                residual,
                self.weight.data if self.pass_weight_add else None,
                self.variance_epsilon,
                self.variance_size_override,
            )

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if _force_torch_rms_norm():
            _log_rms_norm_override_once("rms_norm.forward_cuda.torch")
            weight = self.weight.data if self.has_weight else None
            if residual is None:
                return _torch_rms_norm(
                    x,
                    weight,
                    self.variance_epsilon,
                    self.variance_size_override,
                )
            return _torch_fused_add_rms_norm(
                x,
                residual,
                weight,
                self.variance_epsilon,
                self.variance_size_override,
            )
        if (
            envs.VLLM_BATCH_INVARIANT
            and residual is None
            and self.variance_size_override is None
        ):
            return rms_norm_batch_invariant(x, self.weight.data, self.variance_epsilon)

        return self.forward_native(x, residual)

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_cuda(x, residual)

    def extra_repr(self) -> str:
        s = f"hidden_size={self.weight.data.size(0)}"
        s += f", eps={self.variance_epsilon}"
        return s


# --8<-- [start:gemma_rms_norm]
@CustomOp.register("gemma_rms_norm")
class GemmaRMSNorm(CustomOp):
    """RMS normalization for Gemma.

    Two differences from the above RMSNorm:
        1. x * (1 + w) instead of x * w.
        2. (x * w).to(orig_dtype) instead of x.to(orig_dtype) * w.
    """

    # --8<-- [end:gemma_rms_norm]

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        if _force_batch_invariant_gemma_rms_norm():
            _log_rms_norm_override_once(
                "gemma_rms_norm.forward_native.batch_invariant"
            )
            return _batch_invariant_gemma_rms_norm(
                x, residual, self.weight.data, self.variance_epsilon
            )
        if _force_torch_rms_norm():
            _log_rms_norm_override_once("gemma_rms_norm.forward_native.torch")
            return _torch_gemma_rms_norm(
                x, residual, self.weight.data, self.variance_epsilon
            )
        orig_dtype = x.dtype
        weight = self.weight.data.float() + 1.0
        if residual is not None:
            x = (
                x.float() + residual.float()
                if orig_dtype == torch.float16
                else x + residual
            )
            residual = x
        # ir.ops.rms_norm handles fp32 upcast internally
        out = ir.ops.rms_norm(x, weight, self.variance_epsilon)
        return (
            out.to(orig_dtype) if residual is None else (out.to(orig_dtype), residual)
        )

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if _force_batch_invariant_gemma_rms_norm():
            _log_rms_norm_override_once(
                "gemma_rms_norm.forward_cuda.batch_invariant"
            )
            return _batch_invariant_gemma_rms_norm(
                x, residual, self.weight.data, self.variance_epsilon
            )
        if _force_torch_rms_norm():
            _log_rms_norm_override_once("gemma_rms_norm.forward_cuda.torch")
            return _torch_gemma_rms_norm(
                x, residual, self.weight.data, self.variance_epsilon
            )
        return self.forward_native(x, residual)


# --8<-- [start:rms_norm_gated]
@CustomOp.register("rms_norm_gated")
class RMSNormGated(CustomOp):
    """RMS Normalization with optional gating.

    This is a native PyTorch implementation that supports:
    - Standard RMS normalization
    - Group RMS normalization
    - Optional gating with SiLU activation
    """

    # --8<-- [end:rms_norm_gated]

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        group_size: int | None = None,
        norm_before_gate: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        activation: str = "swish",
    ):
        """Initialize RMSNormGated.

        Args:
            hidden_size: Size of the hidden dimension
            eps: Epsilon for numerical stability
            group_size: If not None, do GroupNorm with each group
                        having group_size elements.
                        group_size=None is equivalent to group_size=hidden_size
                        (i.e. there's only 1 group).
            norm_before_gate: If True and z is provided: out = norm(x) * silu(z)
                              If False and z is provided: out = norm(x * silu(z))
            device: Device to create parameters on
            dtype: Data type for parameters
            activation: Activation function name for gating
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.activation = activation
        self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward_native(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Native PyTorch implementation of RMS normalization with gating.

        Args:
            x: Input tensor
            z: Optional gating tensor

        Returns:
            Normalized (and optionally gated) tensor

        If z is not None:
            - norm_before_gate=True: out = norm(x) * silu(z)
            - norm_before_gate=False: out = norm(x * silu(z))
        """
        orig_dtype = x.dtype
        x = x.float()
        weight = self.weight.float()
        z = z.float() if z is not None else None

        assert self.activation in ["silu", "sigmoid", "swish"]
        act_fn = F.sigmoid if self.activation == "sigmoid" else F.silu

        # Apply gating before normalization if needed
        if z is not None and not self.norm_before_gate:
            x = x * act_fn(z)

        # RMS Normalization
        if self.group_size is None:
            # Standard RMS norm across the last dimension
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x * torch.rsqrt(variance + self.eps)
            out = x_normed * weight
        else:
            # Group RMS norm
            from einops import rearrange

            x_group = rearrange(x, "... (g d) -> ... g d", d=self.group_size)
            variance = x_group.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x_group * torch.rsqrt(variance + self.eps)
            out = rearrange(x_normed, "... g d -> ... (g d)") * weight

        # Apply gating after normalization if needed
        if z is not None and self.norm_before_gate:
            out = out * act_fn(z)

        return out.to(orig_dtype)

    def forward_cuda(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fla.ops.layernorm_guard import rmsnorm_fn

        return rmsnorm_fn(
            x,
            self.weight,
            self.bias,
            z=z,
            eps=self.eps,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            activation=self.activation,
        )

    def forward_cuda_batch_invariant(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fla.ops.layernorm_guard import (
            rmsnorm_fn_batch_invariant,
        )

        return rmsnorm_fn_batch_invariant(
            x,
            self.weight,
            self.bias,
            z=z,
            eps=self.eps,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            activation=self.activation,
        )

    def forward_xpu(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.forward_cuda(x, z)


class LayerNorm(nn.Module):
    """
    Layer Normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(
            x.float(), (self.dim,), self.weight, self.bias, self.eps
        ).type_as(x)

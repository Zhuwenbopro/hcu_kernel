"""Attention output sigmoid-gate plus per-token INT8 quantization."""

from __future__ import annotations

import torch

from . import op


def fused_sigmoid_mul_per_token_quant(
    attn_output: torch.Tensor,
    gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``attn_output * sigmoid(gate)`` and per-token INT8 quantize it.

    ``gate`` may be a flat ``[tokens, hidden]`` tensor or the production
    strided ``[tokens, heads, head_dim]`` view used by Qwen3.5 attention.
    The fused kernel preserves the BF16/FP16 intermediate rounding performed
    by the original sigmoid-mul followed by ``per_token_quant_int8`` path.
    FP16 sigmoid approximation can differ from Triton at rounding midpoints;
    the FP16 regression checks rare one-bin INT8 differences explicitly.
    Aligned 4096-wide inputs with flat or 256-dimensional heads use a Vec8
    specialization; other shapes and unaligned views use the generic kernel.
    """
    if attn_output.dim() != 2 or not attn_output.is_contiguous():
        raise ValueError("attn_output must be a contiguous [tokens, hidden] tensor")
    if attn_output.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("attn_output must be torch.bfloat16 or torch.float16")
    if gate.dim() not in (2, 3) or gate.device != attn_output.device:
        raise ValueError("gate must be a 2D/3D tensor on attn_output.device")
    if gate.dtype != attn_output.dtype or gate.stride(-1) != 1:
        raise ValueError("gate dtype must match and its last dimension must be contiguous")
    if gate.dim() == 2 and gate.shape != attn_output.shape:
        raise ValueError("flat gate shape must match attn_output")
    if gate.dim() == 3:
        if gate.shape[0] != attn_output.shape[0] or gate.shape[1] * gate.shape[2] != attn_output.shape[1]:
            raise ValueError("3D gate shape must flatten to attn_output shape")

    output = torch.empty_like(attn_output, dtype=torch.int8)
    scales = torch.empty(
        (attn_output.shape[0], 1), device=attn_output.device, dtype=torch.float32
    )
    op.fused_sigmoid_mul_per_token_quant(output, attn_output, gate, scales)
    return output, scales


__all__ = ["fused_sigmoid_mul_per_token_quant"]

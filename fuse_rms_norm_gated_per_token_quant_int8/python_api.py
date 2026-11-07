"""GDN RMSNorm-gated output fused with per-token INT8 quantization."""

from __future__ import annotations

import torch

from . import op


def fuse_rms_norm_gated_per_token_quant_int8(
    x: torch.Tensor,
    weight: torch.Tensor,
    z: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize each 128-dim head; quantize across all heads per token.

    Inputs have shape ``[tokens, heads * 128]``; scales have shape
    ``[tokens, 1]`` for direct consumption by the output projection.
    """
    if x.dim() != 2 or x.shape[1] % 128 or not 128 <= x.shape[1] <= 8192 or not x.is_contiguous():
        raise ValueError("x must be contiguous [tokens, heads * 128], 1..64 heads")
    if z.shape != x.shape or z.dtype != x.dtype or not z.is_contiguous():
        raise ValueError("z must be contiguous with x's shape and dtype")
    if weight.shape != (128,) or weight.dtype != x.dtype or not weight.is_contiguous():
        raise ValueError("weight must be contiguous [128] with x's dtype")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("x must be torch.bfloat16 or torch.float16")
    output = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32)
    op.fuse_rms_norm_gated_per_token_quant_int8(output, x, weight, z, scales, eps)
    return output, scales


__all__ = ["fuse_rms_norm_gated_per_token_quant_int8"]

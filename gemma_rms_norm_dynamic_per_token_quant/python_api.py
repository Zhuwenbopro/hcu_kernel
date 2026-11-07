from __future__ import annotations

from typing import Optional
import torch
from lightop import op

def gemma_rms_norm_dynamic_per_token_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-5,
    residual: Optional[torch.Tensor] = None,
    update_input: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gemma RMSNorm + per-token symmetric int8 quantization.

    weight is the raw Gemma RMSNorm weight; the kernel applies
    gamma = 1 + weight in FP32. The return values follow
    rms_norm_dynamic_per_token_quant: int8 output and [tokens, 1] scales.
    If residual is supplied it is updated with input + residual. If
    update_input is true, input is updated with the BF16 normalized result.
    """
    output = torch.empty(input.shape, device=input.device, dtype=torch.int8)
    scales = torch.empty((input.numel() // input.shape[-1], 1),
                         device=input.device,
                         dtype=torch.float32)
    op.gemma_rms_norm_dynamic_per_token_quant(
        output, input, weight, scales, epsilon, residual, update_input
    )
    return output, scales




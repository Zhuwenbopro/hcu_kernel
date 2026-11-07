# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from lightop import (
    gemma_fused_add_rmsnorm,
    gemma_rms_norm_dynamic_per_token_quant,
    gemma_rmsnorm,
)


def quant_reference(norm):
    absmax = norm.float().abs().amax(dim=-1, keepdim=True)
    scale = torch.maximum(
        absmax / 127.0,
        torch.full_like(absmax, 1.19209e-7),
    )
    quant = torch.round(norm.float() / scale).clamp(-128, 127).to(torch.int8)
    return quant, scale


@pytest.mark.parametrize("num_tokens", [1, 7, 257])
@pytest.mark.parametrize("with_residual", [False, True])
@pytest.mark.parametrize("update_input", [False, True])
def test_gemma_rms_norm_dynamic_per_token_quant(
    num_tokens, with_residual, update_input
):
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA/HIP device")

    torch.manual_seed(20261106 + num_tokens)
    device = "cuda"
    hidden = 2048
    epsilon = 1e-5
    x = torch.randn((num_tokens, hidden), device=device, dtype=torch.bfloat16)
    weight = torch.randn((hidden,), device=device, dtype=torch.bfloat16) * 0.02
    residual = torch.randn_like(x) if with_residual else None

    x_original = x.clone()
    x_ref = x.clone()
    residual_ref = residual.clone() if residual is not None else None
    if with_residual:
        gemma_fused_add_rmsnorm(x_ref, residual_ref, weight, epsilon)
    else:
        norm_ref = torch.empty_like(x_ref)
        gemma_rmsnorm(x_ref, weight, epsilon, out=norm_ref)
        x_ref = norm_ref
    quant_ref, scale_ref = quant_reference(x_ref)

    quant, scale = gemma_rms_norm_dynamic_per_token_quant(
        x, weight, epsilon, residual=residual, update_input=update_input
    )
    torch.cuda.synchronize()

    # The fused kernel preserves the existing LightOp Gemma norm and scale
    # semantics. A few values can land exactly on an int8 rounding boundary
    # across the separate quantization launch.
    torch.testing.assert_close(scale, scale_ref, rtol=1e-5, atol=1e-7)
    q_mismatch = int((quant != quant_ref).sum().item())
    assert q_mismatch <= max(8, num_tokens * hidden // 1000), q_mismatch

    if with_residual:
        torch.testing.assert_close(residual, residual_ref, rtol=0, atol=0)
    if update_input:
        torch.testing.assert_close(x, x_ref, rtol=0, atol=0)
    else:
        torch.testing.assert_close(x, x_original, rtol=0, atol=0)

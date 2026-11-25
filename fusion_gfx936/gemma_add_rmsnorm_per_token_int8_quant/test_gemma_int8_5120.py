import pytest
import torch

from fusion_gfx936.gemma_add_rmsnorm_per_token_int8_quant.python_api import (
    WIDTH,
    gemma_fused_add_rmsnorm_int8_fp32,
)


@pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available()
    or torch.cuda.get_device_properties().gcnArchName.split(":")[0] != "gfx936",
    reason="requires gfx936",
)
@pytest.mark.parametrize("tokens", [0, 1, 63, 64, 159, 160, 351, 352, 703, 704, 1407, 1408, 4096, 16384])
def test_fp32_joint_reduction(tokens):
    torch.manual_seed(71107 + tokens)
    x = torch.randn((tokens, WIDTH), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(WIDTH, device=x.device, dtype=x.dtype) * 0.1
    if tokens:
        x[0].zero_()
        residual[0].zero_()
    if tokens > 1:
        x[1].mul_(2.0**-42)
        residual[1].mul_(2.0**-42)
    original = x.clone()
    total = x.float() + residual.float()
    weighted = total * (1 + weight.float())
    rstd = (total.square().mean(dim=-1, keepdim=True) + 1e-6).rsqrt()
    normalized = weighted * rstd
    absmax = normalized.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    expected_scale = absmax / 127
    expected_q = torch.round(normalized * (127 / absmax)).to(torch.int8)

    q, scale = gemma_fused_add_rmsnorm_int8_fp32(x, residual, weight)
    assert q.dtype == torch.int8 and q.shape == x.shape
    assert scale.dtype == torch.float32 and scale.shape == (tokens, 1)
    assert torch.equal(x.view(torch.int16), original.view(torch.int16))
    assert torch.equal(residual.view(torch.int16), total.to(torch.bfloat16).view(torch.int16))
    torch.testing.assert_close(scale, expected_scale, rtol=2e-6, atol=0)
    delta = (q.int() - expected_q.int()).abs()
    assert torch.all(delta <= 1)
    assert (delta != 0).sum().item() <= max(1, x.numel() // 100_000)

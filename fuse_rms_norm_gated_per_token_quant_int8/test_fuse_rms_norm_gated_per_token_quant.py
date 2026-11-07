"""Compare head-local GDN normalization plus token-wide INT8 quantization."""

import pytest
import torch

import lightop
import lightop.norm
from lightop import fuse_rms_norm_gated_per_token_quant_int8
from lightop.norm import layer_norm_fwd_1pass_opt
from lightop.quant import per_token_quant_int8


def reference(x, weight, z, eps):
    if x.shape[0] == 0:
        return torch.empty_like(x, dtype=torch.int8), torch.empty(
            (0, 1), device=x.device, dtype=torch.float32
        )
    head_x = x.view(-1, 128)
    head_z = z.view(-1, 128)
    head_out = torch.empty_like(head_x)
    rstd = torch.empty(head_x.shape[0], device=x.device, dtype=torch.float32)
    layer_norm_fwd_1pass_opt(
        head_x, head_out, weight, None, head_z, None, rstd,
        128, 128, 128, head_x.shape[0], 128, eps, 128, 16,
        False, True, True, True, "swish",
    )
    return per_token_quant_int8(head_out.view_as(x))


def assert_bitwise_equal(actual, expected):
    q, scale = actual
    ref_q, ref_scale = expected
    torch.testing.assert_close(q, ref_q, rtol=0, atol=0)
    torch.testing.assert_close(
        scale.view(torch.int32), ref_scale.view(torch.int32), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
@pytest.mark.parametrize("rows", [0, 1, 7, 26, 104, 257, 1024])
@pytest.mark.parametrize("heads", [1, 2, 7, 16, 32, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seed", [0, 20261107])
def test_gdn_quant_matches_existing_lightop(rows, heads, dtype, seed):
    torch.manual_seed(seed)
    x = torch.randn((rows, heads * 128), device="cuda", dtype=dtype)
    z = torch.randn_like(x)
    weight = torch.randn(128, device=x.device, dtype=dtype)
    assert_bitwise_equal(
        fuse_rms_norm_gated_per_token_quant_int8(x, weight, z, 1e-6),
        reference(x, weight, z, 1e-6),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
@pytest.mark.parametrize("heads", [1, 32])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("magnitude", [0.0, 1e-4, 128.0])
def test_gdn_quant_input_magnitudes(heads, dtype, magnitude):
    torch.manual_seed(7)
    x = torch.randn((26, heads * 128), device="cuda", dtype=dtype) * magnitude
    z = torch.randn_like(x)
    weight = torch.randn(128, device=x.device, dtype=dtype)
    assert_bitwise_equal(
        fuse_rms_norm_gated_per_token_quant_int8(x, weight, z, 1e-5),
        reference(x, weight, z, 1e-5),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
@pytest.mark.parametrize("hidden", [0, 127, 129, 8320])
def test_gdn_quant_rejects_unsupported_head_shapes(hidden):
    x = torch.empty((1, hidden), device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(128, device=x.device, dtype=x.dtype)
    with pytest.raises(ValueError, match="1..64 heads"):
        fuse_rms_norm_gated_per_token_quant_int8(x, weight, x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_gdn_quant_rejects_misaligned_vector_loads():
    x = torch.empty(129, device="cuda", dtype=torch.bfloat16)[1:].view(1, 128)
    z = torch.zeros_like(x)
    weight = torch.ones(128, device=x.device, dtype=x.dtype)
    with pytest.raises(RuntimeError, match="aligned to 16 bytes"):
        fuse_rms_norm_gated_per_token_quant_int8(x, weight, z)


def test_gdn_quant_public_exports():
    assert lightop.norm.fuse_rms_norm_gated_per_token_quant_int8 is (
        lightop.fuse_rms_norm_gated_per_token_quant_int8
    )
    assert "fuse_rms_norm_gated_per_token_quant_int8" in lightop.norm.__all__

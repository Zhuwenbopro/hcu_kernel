"""Triton reference for the original attention gate, then LightOp quantization.

The gate computation follows SGLang's fused_sigmoid_mul: load to FP32,
evaluate tl.sigmoid and multiplication, and store to the input dtype.
"""

import torch
import triton
import triton.language as tl

from lightop.quant import per_token_quant_int8


@triton.jit
def _sigmoid_mul_reference(
    output, attn, gate, gate_stride_row, gate_stride_head,
    hidden: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    offsets = block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offsets < hidden
    head = offsets // HEAD_DIM
    dim = offsets - head * HEAD_DIM
    x = tl.load(attn + row * hidden + offsets, mask=mask, other=0).to(tl.float32)
    g = tl.load(
        gate + row * gate_stride_row + head * gate_stride_head + dim,
        mask=mask, other=0,
    ).to(tl.float32)
    tl.store(output + row * hidden + offsets, x * tl.sigmoid(g), mask=mask)


def sigmoid_mul_reference(attn, gate):
    output = torch.empty_like(attn)
    if attn.shape[0] == 0:
        return output
    head_dim = gate.shape[-1]
    stride_head = gate.stride(1) if gate.ndim == 3 else head_dim
    block = 1024 if attn.shape[0] < 1024 else 2048
    _sigmoid_mul_reference[(attn.shape[0], triton.cdiv(attn.shape[1], block))](
        output, attn, gate, gate.stride(0), stride_head,
        attn.shape[1], HEAD_DIM=head_dim, BLOCK_H=block, num_warps=4,
    )
    return output


def quant_reference(attn, gate):
    if attn.shape[0] == 0:
        return torch.empty_like(attn, dtype=torch.int8), torch.empty(
            (0, 1), device=attn.device, dtype=torch.float32
        )
    return per_token_quant_int8(sigmoid_mul_reference(attn, gate))


def comparison_stats(actual, expected):
    q, scale = actual
    ref_q, ref_scale = expected
    error = (q.to(torch.int16) - ref_q.to(torch.int16)).abs()
    return {
        "int8_elements": q.numel(), "scale_elements": scale.numel(),
        "int8_mismatches": int((q != ref_q).sum()),
        "scale_bit_mismatches": int((scale.view(torch.int32) != ref_scale.view(torch.int32)).sum()),
        "max_int8_error": int(error.max()) if error.numel() else 0,
        "max_scale_relative_error": float(((scale - ref_scale).abs() / ref_scale.clamp_min(1e-30)).max()) if scale.numel() else 0.0,
    }


def assert_accuracy(stats, dtype):
    if dtype == torch.bfloat16:
        assert stats["int8_mismatches"] == 0, stats
        assert stats["scale_bit_mismatches"] == 0, stats
    else:
        # HIP and Triton sigmoid implementations can straddle an FP16 rounding
        # midpoint. Explicitly check the resulting rare, one-bin INT8 errors.
        assert stats["max_int8_error"] <= 1, stats
        # The existing HIP fallback and Triton reference differ at a small
        # number of FP16 rounding midpoints for a non-packed head stride.
        # The unaligned fallback regression records that known case with a
        # slightly wider bound; exact candidate-vs-original checks cover it.
        mismatch_limit = 2e-4 if stats.get("layout") == "unaligned_gate_head" else 1e-4
        assert stats["int8_mismatches"] / max(stats["int8_elements"], 1) <= mismatch_limit, stats
        assert stats["max_scale_relative_error"] <= 1e-3, stats

# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Check the HCU gated normalization and quantization against PyTorch."""

import pytest
import torch
import triton
import triton.language as tl
from torch.nn import functional as F

from gated_rms_norm_per_token_int8_quant import gated_rms_norm_per_token_int8_quant
from gated_rms_norm_per_token_int8_quant import python_api as impl

EPS = 1e-6
HEAD_DIM = 128
pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(), reason="requires HIP"
)


def _inputs(tokens, heads, strided=False):
    x = torch.randn(
        (tokens * heads, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )
    weight = torch.randn(HEAD_DIM, dtype=x.dtype, device=x.device) * 0.25 + 1
    if not strided:
        return x, torch.randn_like(x), weight
    width = heads * HEAD_DIM
    projection = torch.randn((tokens, width * 3), dtype=x.dtype, device=x.device)
    gate = projection[:, -width:].view(tokens, heads, HEAD_DIM)
    return x, gate, weight


def _reference(x, gate, weight, heads):
    x = x.float()
    norm = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + EPS)
    gated = (norm * weight.float()) * F.silu(gate.reshape_as(x).float())
    rounded = gated.to(torch.bfloat16).float().reshape(-1, heads * HEAD_DIM)
    absmax = rounded.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    return (rounded * (127.0 / absmax)).round().to(torch.int8), absmax / 127.0


def _assert_matches(actual, reference):
    q, scale = actual
    expected_q, expected_scale = reference
    assert q.shape == expected_q.shape and q.dtype == torch.int8
    assert scale.shape == expected_scale.shape and scale.dtype == torch.float32
    torch.testing.assert_close(scale, expected_scale, rtol=8e-3, atol=0)
    assert (q.to(torch.int16) - expected_q.to(torch.int16)).abs().max().item() <= 1
    reference_values = expected_q.float() * expected_scale
    error = (q.float() * scale - reference_values).norm()
    assert error <= 1.5e-3 * reference_values.norm()


@pytest.mark.parametrize(
    "tokens,heads,strided",
    [(1, 24, False), (4, 24, False), (64, 24, True), (128, 24, True),
     (1024, 24, True), (2048, 24, True), (4, 12, False), (4, 48, False)],
)
def test_matches_pytorch(tokens, heads, strided):
    torch.manual_seed(19)
    x, gate, weight = _inputs(tokens, heads, strided)
    actual = gated_rms_norm_per_token_int8_quant(x, gate, weight, EPS, num_heads=heads)
    _assert_matches(actual, _reference(x, gate, weight, heads))


def test_zero_scale_floor_and_empty_batch():
    x, gate, weight = _inputs(2, 24)
    gate.zero_()
    q, scale = gated_rms_norm_per_token_int8_quant(x, gate, weight, EPS, num_heads=24)
    assert torch.count_nonzero(q).item() == 0
    torch.testing.assert_close(scale, torch.full_like(scale, 1e-10 / 127), rtol=1e-7, atol=0)
    q, scale = gated_rms_norm_per_token_int8_quant(x[:0], gate[:0], weight, EPS, num_heads=24)
    assert q.shape == (0, 3072)
    assert scale.shape == (0, 1)


def test_rejects_incomplete_token_and_mismatched_gate():
    x, gate, weight = _inputs(2, 24)
    with pytest.raises(ValueError, match="complete tokens"):
        gated_rms_norm_per_token_int8_quant(x[:-1], gate[:-1], weight, EPS, num_heads=24)
    with pytest.raises(ValueError, match="Gate shape"):
        gated_rms_norm_per_token_int8_quant(x, gate[:-1], weight, EPS, num_heads=24)


def test_graph_replays_new_input():
    torch.manual_seed(23)
    x, gate, weight = _inputs(4, 24)
    before, _ = gated_rms_norm_per_token_int8_quant(x, gate, weight, EPS, num_heads=24)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gated_rms_norm_per_token_int8_quant(x, gate, weight, EPS, num_heads=24)
    x.neg_()
    gate.mul_(0.5)
    graph.replay()
    _assert_matches(actual, _reference(x, gate, weight, 24))
    assert not torch.equal(before, actual[0])


@triton.jit
def _check_gfx936_math(gate_ref, gate_opt, inv_ref, inv_opt):
    bits = tl.program_id(0) * 256 + tl.arange(0, 256)
    z = (bits << 16).to(tl.float32, bitcast=True)
    amax = tl.maximum(tl.abs(z), 1e-10)
    tl.store(gate_ref + bits, z * tl.sigmoid(z))
    tl.store(gate_opt + bits, impl._gfx936_silu(z))
    tl.store(inv_ref + bits, 127.0 / amax)
    tl.store(inv_opt + bits, impl._gfx936_quant_inverse(amax))


def test_gfx936_finite_bf16_gate_and_scale_bits():
    device = torch.device("cuda", torch.cuda.current_device())
    if not impl._is_gfx936(device):
        pytest.skip("gfx936 arithmetic")
    tensors = [torch.empty(65536, dtype=torch.float32, device=device) for _ in range(4)]
    _check_gfx936_math[(256,)](*tensors)
    bits = torch.arange(65536, device=device, dtype=torch.int32)
    finite = (bits & 0x7f80) != 0x7f80
    for reference, actual in ((tensors[0], tensors[1]), (tensors[2], tensors[3])):
        assert torch.equal(reference.view(torch.int32)[finite], actual.view(torch.int32)[finite])


@pytest.mark.parametrize("tokens,heads", [(65, 24), (513, 24), (2049, 24), (128, 48)])
def test_gfx936_projected_layout_matches_original_bits(tokens, heads, monkeypatch):
    if not impl._is_gfx936(torch.device("cuda", torch.cuda.current_device())):
        pytest.skip("gfx936 fast path")
    monkeypatch.setattr(impl, "_gate_tables", {})
    monkeypatch.setattr(impl, "_pending_gate_tables", {})
    torch.manual_seed(8741)
    x, _, w = _inputs(tokens, heads)
    stride = 8192 if heads == 24 else 16384
    z = torch.randn(tokens, stride, dtype=x.dtype, device=x.device)[:, -heads * 128 :]
    z = z.view(tokens, heads, 128)
    q = torch.empty(tokens, heads * 128, dtype=torch.int8, device=x.device)
    s = torch.empty(tokens, 1, dtype=torch.float32, device=x.device)
    if heads == 24:
        warps = 4 if tokens <= 512 else 2
        impl._gated_exact_heads_kernel[(tokens,)](x, z, w, q, s, EPS, 128, stride, 128, num_warps=warps)
    else:
        impl._gated_rmsnorm_int8_kernel[(tokens,)](x, z, w, q, s, EPS, 128, stride, 128, heads, 64, 128, num_warps=4)
    for _ in range(2):
        actual_q, actual_s = impl.gated_rms_norm_per_token_int8_quant(x, z, w, EPS, num_heads=heads)
        assert torch.equal(actual_q, q)
        assert torch.equal(actual_s.view(torch.int32), s.view(torch.int32))

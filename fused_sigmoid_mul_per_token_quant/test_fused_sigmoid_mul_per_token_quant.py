import json
import os
from pathlib import Path

import pytest
import torch

import lightop
import lightop.activation
from lightop import fused_sigmoid_mul_per_token_quant

from attention_sigmoid_int8_reference import assert_accuracy, comparison_stats, quant_reference


_RESULTS = []


@pytest.fixture(scope="module", autouse=True)
def record_accuracy_results():
    yield
    output = os.getenv("LIGHTOP_ATTENTION_ACCURACY_JSON")
    if output:
        Path(output).write_text(json.dumps(_RESULTS, indent=2) + "\n")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
@pytest.mark.parametrize("tokens", [0, 1, 4, 26, 104, 257])
@pytest.mark.parametrize(
    "heads,head_dim",
    [(1, 128), (2, 125), (4, 128), (8, 128), (8, 256), (1, 4096), (16, 256), (32, 256), (65, 128)],
)
@pytest.mark.parametrize("layout", ["flat", "flat_strided", "heads", "heads_strided"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seed", [0, 20261107])
def test_attention_quant_matches_original_gate(tokens, heads, head_dim, layout, dtype, seed):
    torch.manual_seed(seed)
    hidden = heads * head_dim
    x = torch.randn((tokens, hidden), device="cuda", dtype=dtype)
    if layout == "flat":
        gate = torch.randn_like(x)
    elif layout == "flat_strided":
        gate = torch.randn((tokens, 2, hidden), device=x.device, dtype=dtype)[:, 1, :]
    elif layout == "heads":
        gate = torch.randn((tokens, heads, head_dim), device=x.device, dtype=dtype)
    else:
        gate = torch.randn((tokens, heads, 2, head_dim), device=x.device, dtype=dtype)[:, :, 1, :]
    stats = comparison_stats(
        fused_sigmoid_mul_per_token_quant(x, gate), quant_reference(x, gate)
    )
    stats.update(tokens=tokens, heads=heads, head_dim=head_dim, layout=layout, dtype=str(dtype), seed=seed)
    _RESULTS.append(stats)
    assert_accuracy(stats, dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("magnitude", [0.0, 1e-4, 128.0])
def test_attention_quant_input_magnitudes(dtype, magnitude):
    torch.manual_seed(7)
    x = torch.randn((26, 4096), device="cuda", dtype=dtype) * magnitude
    gate = torch.randn((26, 16, 256), device=x.device, dtype=dtype) * 128
    stats = comparison_stats(
        fused_sigmoid_mul_per_token_quant(x, gate), quant_reference(x, gate)
    )
    stats.update(tokens=26, heads=16, head_dim=256, layout="heads", dtype=str(dtype), magnitude=magnitude)
    _RESULTS.append(stats)
    assert_accuracy(stats, dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
@pytest.mark.parametrize("invalid", ["dtype", "shape", "gate_stride", "empty_hidden"])
def test_attention_quant_rejects_invalid_inputs(invalid):
    x = torch.ones((2, 256), device="cuda", dtype=torch.bfloat16)
    gate = torch.ones_like(x)
    if invalid == "dtype":
        x = x.float()
        gate = gate.float()
    elif invalid == "shape":
        gate = gate[:1]
    elif invalid == "gate_stride":
        gate = torch.ones((2, 512), device=x.device, dtype=x.dtype)[:, ::2]
    else:
        x = x[:, :0].contiguous()
        gate = gate[:, :0].contiguous()
    with pytest.raises((ValueError, RuntimeError)):
        fused_sigmoid_mul_per_token_quant(x, gate)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("unaligned", ["x", "gate", "gate_row", "gate_head", "output"])
def test_attention_quant_unaligned_fallback(dtype, unaligned):
    torch.manual_seed(17)
    tokens = 104
    x = torch.randn((tokens, 4096), device="cuda", dtype=dtype)
    gate = torch.randn((tokens, 16, 256), device=x.device, dtype=dtype)
    if unaligned == "x":
        x = torch.randn(tokens * 4096 + 1, device=x.device, dtype=dtype)[1:].view(tokens, 4096)
    elif unaligned == "gate":
        gate = torch.randn(tokens * 4096 + 1, device=x.device, dtype=dtype)[1:].view(tokens, 16, 256)
    elif unaligned == "gate_row":
        gate = torch.randn((tokens, 4097), device=x.device, dtype=dtype)[:, :4096]
    elif unaligned == "gate_head":
        gate = torch.randn((tokens, 16, 257), device=x.device, dtype=dtype)[:, :, :256]
    if unaligned == "output":
        q = torch.empty(tokens * 4096 + 1, device=x.device, dtype=torch.int8)[1:].view(tokens, 4096)
        scale = torch.empty((tokens, 1), device=x.device, dtype=torch.float32)
        lightop.op.fused_sigmoid_mul_per_token_quant(q, x, gate, scale)
        actual = q, scale
    else:
        actual = fused_sigmoid_mul_per_token_quant(x, gate)
    stats = comparison_stats(actual, quant_reference(x, gate))
    stats.update(tokens=tokens, heads=16, head_dim=256, layout="unaligned_" + unaligned, dtype=str(dtype))
    _RESULTS.append(stats)
    assert_accuracy(stats, dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
@pytest.mark.parametrize("tokens", [32, 33, 511, 512, 513])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_attention_quant_dispatch_sizes(tokens, dtype):
    torch.manual_seed(31)
    x = torch.randn((tokens, 4096), device="cuda", dtype=dtype)
    gate = torch.randn((tokens, 16, 2, 256), device=x.device, dtype=dtype)[:, :, 1, :]
    stats = comparison_stats(fused_sigmoid_mul_per_token_quant(x, gate), quant_reference(x, gate))
    stats.update(tokens=tokens, heads=16, head_dim=256, layout="dispatch_sizes", dtype=str(dtype))
    _RESULTS.append(stats)
    assert_accuracy(stats, dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
@pytest.mark.parametrize("tokens", [104, 257])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_attention_quant_graph_replay(tokens, dtype):
    torch.manual_seed(37)
    x = torch.randn((tokens, 4096), device="cuda", dtype=dtype)
    gate = torch.randn((tokens, 16, 2, 256), device=x.device, dtype=dtype)[:, :, 1, :]
    for _ in range(3):
        fused_sigmoid_mul_per_token_quant(x, gate)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = fused_sigmoid_mul_per_token_quant(x, gate)
    graph.replay()
    stats = comparison_stats(result, quant_reference(x, gate))
    stats.update(tokens=tokens, heads=16, head_dim=256, layout="graph_replay", dtype=str(dtype))
    _RESULTS.append(stats)
    assert_accuracy(stats, dtype)


def test_attention_quant_public_exports():
    assert lightop.activation.fused_sigmoid_mul_per_token_quant is (
        lightop.fused_sigmoid_mul_per_token_quant
    )
    assert "fused_sigmoid_mul_per_token_quant" in lightop.activation.__all__

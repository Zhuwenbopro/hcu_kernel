import pytest
import torch
import triton
import triton.language as tl

from ._gfx936 import _sigmoid
from .python_api import _is_gfx936, _launch_original, fuse_sigmoid_mul_int8_quant


def _require_gfx936():
    if not torch.cuda.is_available() or not _is_gfx936(torch.device("cuda", torch.cuda.current_device())):
        pytest.skip("requires gfx936")


@triton.jit
def _all_gates(actual, expected, IS_BF16: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    raw = i.to(tl.uint16)
    if IS_BF16:
        gate = raw.to(tl.bfloat16, bitcast=True).to(tl.float32)
    else:
        gate = raw.to(tl.float16, bitcast=True).to(tl.float32)
    tl.store(actual + i, _sigmoid(gate))
    tl.store(expected + i, tl.sigmoid(gate))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sigmoid_all_encodings(dtype):
    _require_gfx936()
    actual = torch.empty(65536, device="cuda", dtype=torch.float32)
    expected = torch.empty_like(actual)
    _all_gates[(256,)](actual, expected, dtype == torch.bfloat16)
    valid = ~expected.isnan()
    assert torch.equal(actual.isnan(), expected.isnan())
    assert torch.equal(actual.view(torch.int32)[valid], expected.view(torch.int32)[valid])


@pytest.mark.parametrize("projected", [False, True])
@pytest.mark.parametrize(
    "width,dtype", [(3072, torch.bfloat16), (4096, torch.bfloat16), (6144, torch.bfloat16),
                    (3072, torch.float16), (4096, torch.float16)],
)
def test_projected_quantization_and_rounding(projected, width, dtype):
    _require_gfx936()
    rows = 257
    gen = torch.Generator(device="cuda").manual_seed(9287)
    attention = torch.randn(rows, width, generator=gen, device="cuda", dtype=dtype)
    if projected:
        row_stride = {3072: 7168, 4096: 8192, 6144: 14336}[width]
        storage = torch.randn(rows, row_stride, generator=gen, device="cuda", dtype=dtype)
        gate = storage.as_strided((rows, width // 256, 256), (row_stride, 512, 1), 256)
    else:
        gate = torch.randn(rows, width, generator=gen, device="cuda", dtype=dtype)
    attention[0].zero_()
    attention[1].mul_(1e-16)
    gate[2].fill_(-88.5)
    gate[3].fill_(float("inf"))
    gate[4].fill_(-float("inf"))
    # sigmoid(0) gives 0.5. A row max of 127 gives an exact unit quantizer:
    # positive and negative half-integers then exercise nearest-even rounding.
    gate[5].zero_()
    attention[5].copy_(((torch.arange(width, device="cuda") % 255) - 127).to(attention.dtype))
    attention[5, 0] = 254
    q, scale = fuse_sigmoid_mul_int8_quant(attention, gate)
    ref_q, ref_s = torch.empty_like(q), torch.empty_like(scale)
    _launch_original(attention, gate, ref_q, ref_s)
    assert torch.equal(q, ref_q)
    assert torch.equal(scale.view(torch.int32), ref_s.view(torch.int32))


def test_graph_replay_reads_updated_inputs():
    _require_gfx936()
    x = torch.randn(19, 3072, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(x)
    fuse_sigmoid_mul_int8_quant(x, gate)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, scale = fuse_sigmoid_mul_int8_quant(x, gate)
    gate.neg_()
    graph.replay()
    ref_q, ref_s = torch.empty_like(q), torch.empty_like(scale)
    _launch_original(x, gate, ref_q, ref_s)
    assert torch.equal(q, ref_q)
    assert torch.equal(scale.view(torch.int32), ref_s.view(torch.int32))

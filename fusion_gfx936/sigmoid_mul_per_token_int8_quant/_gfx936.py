# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import triton
import triton.language as tl


@triton.jit
def _sigmoid(gate):
    exp_neg = tl.inline_asm_elementwise(
        "v_exp_f32 $0, $1", constraints="=v,v", args=[-gate * 1.4426950408889634],
        dtype=tl.float32, is_pure=True, pack=1,
    )
    denominator = 1.0 + exp_neg
    # The scaled denominator keeps the approximate reciprocal normal. Rescale
    # before correcting on the original denominator to round subnormals once.
    approx = tl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1", constraints="=v,v", args=[denominator * 0.125],
        dtype=tl.float32, is_pure=True, pack=1,
    ) * 0.125
    reciprocal = tl.fma(tl.fma(-denominator, approx, 1.0), approx, approx)
    return tl.where(denominator == float("inf"), 0.0, reciprocal)


@triton.jit
def _values(attn, gate, col, HEAD_DIM: tl.constexpr, HEAD_STRIDE: tl.constexpr):
    if HEAD_STRIDE == HEAD_DIM:
        gate_col = col
    else:
        gate_col = (col // HEAD_DIM) * HEAD_STRIDE + col % HEAD_DIM
    a = tl.load(attn + col).to(tl.float32)
    g = tl.load(gate + gate_col).to(tl.float32)
    return (a * _sigmoid(g)).to(attn.dtype.element_ty).to(tl.float32)


@triton.jit
def _to_int8(scaled):
    # |scaled| <= 127 (up to multiplication rounding). At 1.5 * 2**23,
    # FP32 addition rounds to the nearest even integer. Its low byte
    # encodes the signed INT8 result. Keep the product and addition separate.
    bits = tl.inline_asm_elementwise(
        "v_add_f32 $0, 0x4b400000, $1", constraints="=v,v", args=[scaled],
        dtype=tl.int32, is_pure=True, pack=1,
    )
    return bits.to(tl.int8)


@triton.jit
def _kernel(
    attention, gate, quant, scale, GATE_ROW_STRIDE: tl.constexpr,
    HEAD_STRIDE: tl.constexpr, HEAD_DIM: tl.constexpr, WIDTH: tl.constexpr,
    STEP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    attention += row * WIDTH
    quant += row * WIDTH
    gate += row * GATE_ROW_STRIDE
    col = tl.arange(0, STEP)
    x0 = _values(attention, gate, col, HEAD_DIM, HEAD_STRIDE)
    if WIDTH == 3 * STEP:
        x1 = _values(attention, gate, col + STEP, HEAD_DIM, HEAD_STRIDE)
        x2 = _values(attention, gate, col + 2 * STEP, HEAD_DIM, HEAD_STRIDE)
        largest = tl.maximum(tl.maximum(tl.abs(x0), tl.abs(x1)), tl.abs(x2))
    else:
        tl.static_assert(WIDTH == STEP)
        largest = tl.abs(x0)
    amax = tl.maximum(tl.max(largest, 0), 1e-10)
    inverse = 127.0 / amax
    tl.store(quant + col, _to_int8(x0 * inverse))
    if WIDTH == 3 * STEP:
        tl.store(quant + col + STEP, _to_int8(x1 * inverse))
        tl.store(quant + col + 2 * STEP, _to_int8(x2 * inverse))
    tl.store(scale + row, amax / 127.0)

# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import math
import threading
from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _fill_gfx936_gate_table(table):
    bits = tl.program_id(0) * 256 + tl.arange(0, 256)
    z = (bits << 16).to(tl.float32, bitcast=True)
    tl.store(table + bits, z * tl.sigmoid(z))


_gate_tables = {}
_pending_gate_tables = {}
_gate_table_lock = threading.Lock()


def _get_gate_table(device):
    table = _gate_tables.get(device)
    if table is not None:
        return table
    with _gate_table_lock:
        table = _gate_tables.get(device)
        if table is not None:
            return table
        # A cold graph uses equivalent arithmetic and owns no table allocation.
        if torch.cuda.is_current_stream_capturing():
            return None
        pending = _pending_gate_tables.get(device)
        if pending is None:
            table = torch.empty(65536, device=device, dtype=torch.float32)
            _fill_gfx936_gate_table[(256,)](table)
            event = torch.cuda.Event()
            event.record()
            _pending_gate_tables[device] = (table, event)
        elif pending[1].query():
            table = _gate_tables[device] = pending[0]
            del _pending_gate_tables[device]
            return table
        # A first caller on any stream can proceed while initialization is pending.
        return None


@triton.jit
def _gfx936_silu(z):
    # Scale down before the reciprocal so large finite denominators stay in the
    # hardware reciprocal's normal range. Refine in the original FP32 scale.
    den = 1.0 + tl.exp(-z)
    r = tl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1", constraints="=v,v", args=[den * 0.125],
        dtype=tl.float32, is_pure=True, pack=1,
    ) * 0.125
    sigmoid = tl.fma(tl.fma(-den, r, 1.0), r, r)
    return z * tl.where(den == float("inf"), 0.0, sigmoid)


@triton.jit
def _gfx936_quant_inverse(amax):
    den = amax * 0.125
    r = tl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1", constraints="=v,v", args=[den],
        dtype=tl.float32, is_pure=True, pack=1,
    )
    q = 15.875 * r
    return tl.where(amax == float("inf"), 0.0, tl.fma(tl.fma(-q, den, 15.875), r, q))


@triton.jit
def _gated_rmsnorm_int8_kernel(
    X, Z, W, Q, S, eps,
    X_ROW: tl.constexpr,
    Z_TOKEN: tl.constexpr,
    Z_HEAD: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    FAST_GATE: tl.constexpr = False, gate_table=None,
):
    token = tl.program_id(0)
    heads = tl.arange(0, BLOCK_HEADS)
    cols = tl.arange(0, HEAD_DIM)
    mask = heads[:, None] < HEADS
    row = token * HEADS + heads
    x = tl.load(X + row[:, None] * X_ROW + cols[None, :], mask, 0).to(tl.float32)
    z = tl.load(Z + token * Z_TOKEN + heads[:, None] * Z_HEAD + cols[None, :],
                mask, 0).to(tl.float32)
    weight = tl.load(W + cols).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=1) / HEAD_DIM + eps)
    if gate_table is not None:
        gate = tl.load(gate_table + (z.to(tl.uint32, bitcast=True) >> 16))
    elif FAST_GATE:
        gate = _gfx936_silu(z)
    else:
        gate = z * tl.sigmoid(z)
    gated = (x * rstd[:, None] * weight[None, :]) * gate
    gated = gated.to(tl.bfloat16).to(tl.float32)
    row_absmax = tl.max(tl.abs(gated), axis=1)
    absmax = tl.maximum(tl.max(row_absmax, axis=0), 1e-10)
    quantized = libdevice.nearbyint(gated * (127.0 / absmax)).to(tl.int8)
    tl.store(Q + row[:, None] * HEAD_DIM + cols[None, :], quantized, mask)
    tl.store(S + token, absmax / 127.0)


@triton.jit
def _gated_tile(X, Z, w, token, head, col, eps,
                X_ROW: tl.constexpr, Z_TOKEN: tl.constexpr, Z_HEAD: tl.constexpr,
                FAST_GATE: tl.constexpr, gate_table):
    x = tl.load(X + (token * 24 + head[:, None]) * X_ROW + col[None, :]).to(tl.float32)
    z = tl.load(Z + token * Z_TOKEN + head[:, None] * Z_HEAD + col[None, :]).to(tl.float32)
    if gate_table is not None:
        bits = z.to(tl.uint32, bitcast=True) >> 16
        gate = tl.load(gate_table + bits)
    elif FAST_GATE:
        gate = _gfx936_silu(z)
    else:
        gate = z * tl.sigmoid(z)
    inv = tl.rsqrt(tl.sum(x * x, axis=1) * 0.0078125 + eps)
    return ((x * inv[:, None] * w[None, :]) * gate).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _quantize_store(Y, out, inv, head, col, BIAS_ROUND: tl.constexpr):
    scaled = Y * inv
    if BIAS_ROUND:
        # Normalized values lie in [-127, 127]. A FP32 add at this exponent
        # rounds to nearest even; its low byte is the signed INT8 result.
        bits = tl.inline_asm_elementwise(
            "v_add_f32 $0, 0x4b400000, $1", constraints="=v,v", args=[scaled],
            dtype=tl.int32, is_pure=True, pack=1,
        )
        q = bits.to(tl.int8)
    else:
        q = libdevice.nearbyint(scaled).to(tl.int8)
    tl.store(out + head[:, None] * 128 + col[None, :], q)


@triton.jit
def _gated_exact_heads_kernel(X, Z, W, Q, S, eps,
                              X_ROW: tl.constexpr, Z_TOKEN: tl.constexpr, Z_HEAD: tl.constexpr,
                              FAST_GATE: tl.constexpr = False, FAST_QUANT: tl.constexpr = False,
                              gate_table=None, BIAS_ROUND: tl.constexpr = False):
    token = tl.program_id(0)
    h = tl.arange(0, 8)
    c = tl.arange(0, 128)
    w = tl.load(W + c).to(tl.float32)
    y0 = _gated_tile(X, Z, w, token, h, c, eps, X_ROW, Z_TOKEN, Z_HEAD, FAST_GATE, gate_table)
    y1 = _gated_tile(X, Z, w, token, h + 8, c, eps, X_ROW, Z_TOKEN, Z_HEAD, FAST_GATE, gate_table)
    y2 = _gated_tile(X, Z, w, token, h + 16, c, eps, X_ROW, Z_TOKEN, Z_HEAD, FAST_GATE, gate_table)
    largest = tl.maximum(tl.maximum(tl.abs(y0), tl.abs(y1)), tl.abs(y2))
    amax = tl.maximum(tl.max(tl.max(largest, 1), 0), 1e-10)
    if FAST_QUANT:
        inv = _gfx936_quant_inverse(amax)
    else:
        inv = 127.0 / amax
    out = Q + token * 3072
    _quantize_store(y0, out, inv, h, c, BIAS_ROUND)
    _quantize_store(y1, out, inv, h + 8, c, BIAS_ROUND)
    _quantize_store(y2, out, inv, h + 16, c, BIAS_ROUND)
    tl.store(S + token, amax / 127.0)


def _launch_gfx936_24(x, z, weight, q, scale, epsilon, gate_token_stride, gate_head_stride):
    tokens = q.shape[0]
    if not tokens:
        return
    # Preserve the original per-head sum tree at its warp-count boundaries.
    warps = 8 if tokens <= 64 else 4 if tokens <= 512 else 2
    # Larger token batches amortize arithmetic better than indexed loads.
    table = _get_gate_table(x.device) if tokens <= 2048 else None
    _gated_exact_heads_kernel[(tokens,)](
        x, z, weight, q, scale, epsilon, x.stride(0), gate_token_stride,
        gate_head_stride, True, True, table, True, num_warps=warps,
    )


@lru_cache(maxsize=None)
def _is_gfx936(device):
    if torch.version.hip is None:
        return False
    properties = torch.cuda.get_device_properties(device)
    return getattr(properties, "gcnArchName", "").split(":")[0] == "gfx936"


def gated_rms_norm_per_token_int8_quant(
    input: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    *,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply per-head RMSNorm and SiLU gate, then quantize each token to INT8.

    Input has shape `(tokens * num_heads, 128)`. Gate has the same shape
    or shape `(tokens, num_heads, 128)` with a unit last-dimension stride.
    Weight is shared across heads and has shape `(128,)`. All three tensors
    use BF16 on the same HCU device. Quantization observes the BF16-rounded
    gated output and returns INT8 activations of shape
    `(tokens, num_heads * 128)` and FP32 scales of shape `(tokens, 1)`.
    """
    if input.ndim != 2 or input.shape[1] != 128 or not 1 <= num_heads <= 64:
        raise ValueError("Expected input with 128 columns and 1 to 64 value heads")
    rows, head_dim = input.shape
    if rows % num_heads:
        raise ValueError("Input rows must contain complete tokens")
    tokens = rows // num_heads
    if (
        torch.version.hip is None
        or not input.is_cuda
        or input.dtype != torch.bfloat16
        or input.stride(-1) != 1
        or gate.device != input.device
        or gate.dtype != input.dtype
        or gate.ndim not in (2, 3)
        or gate.stride(-1) != 1
        or weight.device != input.device
        or weight.dtype != input.dtype
        or weight.shape != (head_dim,)
        or not weight.is_contiguous()
    ):
        raise ValueError("Expected BF16 GPU tensors on one device with a contiguous last dimension")
    if gate.shape == input.shape:
        gate_token_stride, gate_head_stride = num_heads * gate.stride(0), gate.stride(0)
    elif gate.shape == (tokens, num_heads, head_dim):
        gate_token_stride, gate_head_stride = gate.stride(0), gate.stride(1)
    else:
        raise ValueError("Gate shape must match input rows or be [tokens, num_heads, 128]")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Expected a finite positive epsilon")

    quantized = torch.empty((tokens, num_heads * head_dim), dtype=torch.int8, device=input.device)
    scale = torch.empty((tokens, 1), dtype=torch.float32, device=input.device)
    if tokens:
        num_warps = 8 if num_heads == 24 and tokens <= 128 else 4
        with torch.cuda.device(input.device):
            if num_heads == 24 and _is_gfx936(input.device):
                _launch_gfx936_24(input, gate, weight, quantized, scale, epsilon,
                                 gate_token_stride, gate_head_stride)
            else:
                fast_gate = num_heads == 48 and _is_gfx936(input.device)
                table = _get_gate_table(input.device) if fast_gate else None
                _gated_rmsnorm_int8_kernel[(tokens,)](
                    input, gate, weight, quantized, scale, epsilon, input.stride(0),
                    gate_token_stride, gate_head_stride, num_heads, triton.next_power_of_2(num_heads),
                    head_dim, fast_gate, table, num_warps=num_warps,
                )
    return quantized, scale

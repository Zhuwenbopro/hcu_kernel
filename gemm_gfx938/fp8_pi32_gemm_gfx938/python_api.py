"""Launch tuned gfx938 Channel FP8 GEMM with one prepacked weight."""

import ctypes as ct
import json
from functools import cache
from pathlib import Path
from typing import NamedTuple

import torch

ROOT = Path(__file__).resolve().parent
SHAPES = {
    "gdn_qkvz": (5120, 16384),
    "gdn_out": (6144, 5120),
    "attention_qkv": (5120, 14336),
    "attention_out": (6144, 5120),
    "gate_up": (5120, 34816),
    "down": (17408, 5120),
}
# The serving path uses the same schedule for both Out projections.
_DEFAULT_ROLES = {dims: role for role, dims in SHAPES.items() if role != "attention_out"}
_FP8 = torch.float8_e4m3fn


class Config(NamedTuple):
    provider: str
    splits: int
    depth: int
    tile_n: int


@cache
def _dispatch():
    policy = json.loads((ROOT / "dispatch.json").read_text(encoding="utf-8"))
    choices = {}
    for role, rows in policy.items():
        for m, case in rows.items():
            provider, splits, depth, *tile = case.split(":")
            choices[role, int(m)] = Config(
                provider, int(splits), int(depth), int(tile[0]) if tile else 256
            )
    return choices


def get_config(m, k, n, role=None):
    """Return the measured GEMM configuration for M rows and shape (K,N)."""
    if not 1 <= m <= 128:
        raise ValueError("PI32 GEMM requires 1 <= M <= 128")
    if role is None:
        role = _DEFAULT_ROLES.get((k, n))
    if SHAPES.get(role) != (k, n):
        raise ValueError(f"unsupported PI32 GEMM role or (K,N) shape: {role}, {(k, n)}")
    return _dispatch()[role, m]


@cache
def _gfx938(device):
    if device.type != "cuda":
        return False
    return getattr(torch.cuda.get_device_properties(device), "gcnArchName", "").startswith("gfx938")


def _check_buffers(first, *others):
    device = first.device
    if not _gfx938(device):
        raise ValueError("PI32 kernels require a gfx938 HIP tensor")
    if any(t.device != device or not t.is_contiguous() or t.data_ptr() % 16
           for t in (first, *others)):
        raise ValueError("buffers must be contiguous and 16-byte aligned on the same device")


def _check_disjoint(writable, *other_buffers):
    start = writable.data_ptr()
    end = start + writable.numel() * writable.element_size()
    for other in other_buffers:
        other_start = other.data_ptr()
        other_end = other_start + other.numel() * other.element_size()
        if start < other_end and other_start < end:
            raise ValueError("writable buffers must use disjoint storage")


@cache
def _load_library(name):
    path = ROOT / "build" / name
    if not path.is_file():
        raise RuntimeError(f"missing {path}; run python -m gemm_gfx938.fp8_pi32_gemm_gfx938.build")
    return ct.CDLL(str(path))


@cache
def _gemm_launcher(provider, tile_n):
    libraries = {
        "ring": ("exp4_ring_fp8", "exp4_ring_launch"),
        "vec": ("exp4_ring_vec_reduce", "exp4_ring_vec_launch"),
    }
    stem, symbol = libraries[provider]
    suffix = "_bn128" if tile_n == 128 else ""
    launch = getattr(_load_library(f"lib{stem}{suffix}.so"), symbol)
    launch.argtypes = [ct.c_void_p] * 6 + [ct.c_int] * 6 + [ct.c_void_p]
    launch.restype = ct.c_int
    return launch


@cache
def _relayout_launcher():
    launch = _load_library("libpi32_relayout.so").pi32_relayout_launch
    launch.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int, ct.c_int, ct.c_void_p]
    launch.restype = ct.c_int
    return launch


def _packed_shape(n, k):
    return (k // 64, n // 16, 4, 16, 16)


def _dimensions(packed):
    if packed.ndim != 5 or packed.shape[2:] != (4, 16, 16) or packed.dtype != torch.uint8:
        raise ValueError("packed weight must be uint8 [K/64,N/16,4,16,16]")
    return packed.shape[1] * 16, packed.shape[0] * 64


def _relayout(src, dst, n, k, *, to_nk):
    _check_buffers(src, dst)
    _check_disjoint(dst, src)
    with torch.cuda.device(src.device):
        stream = torch.cuda.current_stream(src.device)
        status = _relayout_launcher()(src.data_ptr(), dst.data_ptr(), n, k, int(to_nk), stream.cuda_stream)
    if status:
        raise RuntimeError(f"PI32 relayout failed with HIP error {status}")
    return dst


def pack_weight(weight, *, out=None):
    """Permute checkpoint E4M3FN [N,K] bytes into the shared PI32 layout."""
    if weight.ndim != 2 or weight.dtype != _FP8:
        raise ValueError("weight must be E4M3FN [N,K]")
    n, k = weight.shape
    if n <= 0 or n % 32 or k <= 0 or k % 64:
        raise ValueError("PI32 requires positive N divisible by 32 and K divisible by 64")
    shape = _packed_shape(n, k)
    if out is None:
        out = torch.empty(shape, device=weight.device, dtype=torch.uint8)
    if out.shape != shape or out.dtype != torch.uint8:
        raise ValueError("out must be uint8 [K/64,N/16,4,16,16]")
    return _relayout(weight, out, n, k, to_nk=False)


def unpack_weight(packed, *, out=None):
    """Recover [N,K] E4M3FN bytes, optionally into a transient buffer."""
    n, k = _dimensions(packed)
    if n <= 0 or n % 32 or k <= 0:
        raise ValueError("invalid PI32 dimensions")
    if out is None:
        out = torch.empty((n, k), device=packed.device, dtype=_FP8)
    if out.shape != (n, k) or out.dtype != _FP8:
        raise ValueError("out must be E4M3FN [N,K]")
    return _relayout(packed, out, n, k, to_nk=True)


def gemm(x, packed, x_scale, weight_scale, *, role=None, out=None, workspace=None):
    """Compute scaled E4M3FN [M,K] 脳 PI32 W as BF16 [M,N] on the current stream."""
    if x.ndim != 2 or x.dtype != _FP8:
        raise ValueError("X must be row-major E4M3FN [M,K]")
    m, k = x.shape
    n, packed_k = _dimensions(packed)
    if packed_k != k:
        raise ValueError("X and packed weight must agree on K")
    config = get_config(m, k, n, role)
    if x_scale.dtype != torch.float32 or x_scale.numel() != m:
        raise ValueError("X scale must have one FP32 value per row")
    if weight_scale.dtype != torch.float32 or weight_scale.numel() != n:
        raise ValueError("W scale must have one FP32 value per output column")
    if out is None:
        out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    if out.shape != (m, n) or out.dtype != torch.bfloat16:
        raise ValueError("out must be BF16 [M,N]")

    inputs = (x, packed, x_scale, weight_scale)
    buffers = [*inputs, out]
    if config.splits > 1:
        if workspace is None:
            workspace = torch.empty((config.splits, m, n), device=x.device, dtype=torch.float32)
        if workspace.dtype != torch.float32 or workspace.numel() < config.splits * m * n:
            raise ValueError("split-K workspace must fit splits 脳 M 脳 N FP32 values")
        buffers.append(workspace)
    _check_buffers(*buffers)
    _check_disjoint(out, *inputs)
    if config.splits > 1:
        _check_disjoint(workspace, *buffers[:-1])

    with torch.cuda.device(x.device):
        stream = torch.cuda.current_stream(x.device)
        launch = _gemm_launcher(config.provider, config.tile_n)
        status = launch(
            x.data_ptr(), packed.data_ptr(), x_scale.data_ptr(), weight_scale.data_ptr(),
            out.data_ptr(), workspace.data_ptr() if config.splits > 1 else 0,
            m, n, k, config.splits, config.depth, 0, stream.cuda_stream,
        )
    if status:
        raise RuntimeError(f"PI32 GEMM failed with HIP error {status}")
    return out

"""Prepacked INT8 X @ W.T -> INT32 on gfx936; packing is separate from kernel timing."""
from dataclasses import dataclass
import ctypes
from functools import lru_cache
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
VARIANTS = json.loads((ROOT / "variants.json").read_text())
DISPATCH = json.loads((ROOT / "dispatch.json").read_text())["policies"]

@dataclass(frozen=True)
class PackedActivation:
    data: torch.Tensor
    m: int
    k: int
    block_m: int

@dataclass(frozen=True)
class PackedWeight:
    data: torch.Tensor
    n: int
    k: int

def _check_input(t):
    if t.ndim != 2 or not t.is_cuda or t.dtype != torch.int8 or not t.is_contiguous():
        raise ValueError("expected contiguous 2D HCU torch.int8 tensor")
    if t.shape[0] < 1 or t.shape[1] < 64 or t.shape[1] % 64:
        raise ValueError("rows must be positive and K a positive multiple of 64")

def _pack(t, block_m=16):
    rows, k = t.shape
    padded = (rows + block_m - 1) // block_m * block_m
    if rows != padded:
        t = torch.cat([t, torch.zeros((padded - rows, k), device=t.device, dtype=t.dtype)], dim=0)
    return t.view(padded // 16, 16, k // 64, 4, 16).permute(2, 0, 3, 1, 4).contiguous()

def pack_weight(w):
    """Pack row-major W[N,K] once, including the PI16 output-column permutation."""
    _check_input(w)
    n, k = w.shape
    if n % 16:
        raise ValueError("N must be a multiple of 16")
    p = torch.arange(16, device=w.device)
    perm = 4 * (p % 4) + p // 4
    reordered = w.view(n // 16, 16, k)[:, perm, :].reshape(n, k)
    return PackedWeight(_pack(reordered), n, k)

def choose_variant(m, n):
    """Conservative measured-shape policy; select a variant explicitly for tuning."""
    if m < 1:
        raise ValueError("M must be positive")
    rules = DISPATCH.get(str(n))
    if rules is None:
        raise ValueError("auto supports N=" + ",".join(DISPATCH) + "; specify a variant for other N")
    for rule in rules:
        if rule["max_M"] is None or m <= rule["max_M"]:
            return rule["variant"]
    raise RuntimeError("dispatch policy must have a fallback")

def pack_activation(x, variant):
    _check_input(x)
    if variant not in VARIANTS:
        raise ValueError("unknown variant: " + str(variant))
    bm = VARIANTS[variant]["validated_resources"][3]
    return PackedActivation(_pack(x, bm), x.shape[0], x.shape[1], bm)

@lru_cache(None)
def _kernel(variant, device):
    if variant not in VARIANTS:
        raise ValueError("unknown variant: " + str(variant))
    props = torch.cuda.get_device_properties(device)
    if "gfx936" not in props.gcnArchName:
        raise RuntimeError("these kernels are validated only for gfx936")
    path = ROOT / "build" / (variant + ".so")
    if not path.exists():
        raise FileNotFoundError(str(path) + "; run python build.py first")
    handle = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
    launch = handle.zmoe_launch
    launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    launch.restype = ctypes.c_int
    info = handle.zmoe_info
    info.argtypes = [ctypes.POINTER(ctypes.c_int)]
    info.restype = ctypes.c_int
    resources = (ctypes.c_int * 7)()
    if info(resources):
        raise RuntimeError("hipFuncGetAttributes failed")
    if resources[2]:
        raise RuntimeError("kernel reports register spills")
    return handle, launch, list(resources)

def kernel_info(variant, device=None):
    device = torch.cuda.current_device() if device is None else int(device)
    return _kernel(variant, device)[2][:]

def gemm_packed(x, w, variant, out=None):
    """Asynchronous launch on the current Torch stream; Y has int32 dtype."""
    if not isinstance(x, PackedActivation) or not isinstance(w, PackedWeight):
        raise TypeError("use pack_activation() and pack_weight()")
    m, n, k = x.m, w.n, w.k
    if variant not in VARIANTS:
        raise ValueError("unknown variant: " + str(variant))
    spec = VARIANTS[variant]["validated_resources"]
    if m < 1 or n < 1 or k < 64 or k % 64:
        raise ValueError("invalid M/N/K")
    padded_m = (m + spec[3] - 1) // spec[3] * spec[3]
    if x.data.numel() != padded_m * k or w.data.numel() != n * k:
        raise ValueError("packed tensor storage size does not match M/N/K")
    if x.k != k or x.block_m != spec[3] or n % spec[4]:
        raise ValueError("packed layout or dimensions do not match the selected tile")
    if x.data.device != w.data.device or not x.data.is_cuda:
        raise ValueError("packed tensors must be on the same HCU")
    if x.data.dtype != torch.int8 or w.data.dtype != torch.int8:
        raise ValueError("packed inputs must have int8 dtype")
    if not x.data.is_contiguous() or not w.data.is_contiguous():
        raise ValueError("packed inputs must be contiguous")
    if out is None:
        out = torch.empty((m, n), dtype=torch.int32, device=x.data.device)
    if out.shape != (m, n) or out.dtype != torch.int32 or out.device != x.data.device or not out.is_contiguous():
        raise ValueError("out must be contiguous int32 [M,N] on the input device")
    with torch.cuda.device(x.data.device):
        _, launch, _ = _kernel(variant, x.data.device.index)
        stream = torch.cuda.current_stream(x.data.device).cuda_stream
        rc = launch(x.data.data_ptr(), w.data.data_ptr(), out.data_ptr(), m, n, k, stream)
        if rc:
            raise RuntimeError("HIP launch failed: " + str(rc))
    return out

def decode_gemm_int8(x, w, variant="auto", out=None):
    """Convenience API; packs X each call. Keep the PackedWeight across calls."""
    _check_input(x)
    if not isinstance(w, PackedWeight):
        raise TypeError("W must be packed once with pack_weight(W[N,K])")
    variant = choose_variant(x.shape[0], w.n) if variant == "auto" else variant
    return gemm_packed(pack_activation(x, variant), w, variant, out)

__all__ = ["PackedActivation", "PackedWeight", "VARIANTS", "pack_activation",
           "pack_weight", "choose_variant", "kernel_info", "gemm_packed", "decode_gemm_int8"]

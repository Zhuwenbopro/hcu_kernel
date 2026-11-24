"""Standalone 5120-wide BF16 residual add, Gemma RMSNorm and INT8 quantization."""

import ctypes
from functools import lru_cache
import math
from pathlib import Path

import torch


WIDTH = 5120


@lru_cache(maxsize=1)
def _launcher():
    path = Path(__file__).resolve().parent / "build" / "libgemma_int8_5120.so"
    library = ctypes.CDLL(str(path))
    launch = library.launch_gemma_int8_5120
    launch.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int, ctypes.c_float, ctypes.c_void_p]
    launch.restype = ctypes.c_int
    return launch


def gemma_fused_add_rmsnorm_int8_fp32(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return INT8 [tokens,5120] and FP32 [tokens,1], updating BF16 residual in place."""
    if x.ndim != 2 or x.shape[1] != WIDTH or residual.shape != x.shape or weight.shape != (WIDTH,):
        raise ValueError("expected x and residual [tokens,5120] and weight [5120]")
    if not x.is_cuda or any(
        t.device != x.device or t.dtype != torch.bfloat16 or not t.is_contiguous() or t.data_ptr() % 16
        for t in (x, residual, weight)
    ):
        raise ValueError("expected contiguous 16-byte-aligned BF16 tensors on one HCU")
    if torch.version.hip is None or torch.cuda.get_device_properties(x.device).gcnArchName.split(":")[0] != "gfx936":
        raise ValueError("this kernel requires gfx936")
    eps = ctypes.c_float(epsilon).value
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("epsilon must be positive and finite in FP32")

    tokens = x.shape[0]
    if tokens > 2**31 - 1:
        raise ValueError("too many tokens for the HIP launch")
    out = torch.empty(x.shape, device=x.device, dtype=torch.int8)
    scale = torch.empty((tokens, 1), device=x.device, dtype=torch.float32)
    if tokens == 0:
        return out, scale
    start = residual.data_ptr()
    end = start + residual.numel() * residual.element_size()
    for tensor in (x, weight):
        if tensor.data_ptr() < end and tensor.data_ptr() + tensor.numel() * tensor.element_size() > start:
            raise ValueError("residual must not overlap x or weight")

    with torch.cuda.device(x.device):
        stream = torch.cuda.current_stream()
        status = _launcher()(x.data_ptr(), residual.data_ptr(), weight.data_ptr(),
                             out.data_ptr(), scale.data_ptr(), tokens, eps, stream.cuda_stream)
        if status:
            raise RuntimeError(f"HIP kernel launch failed with status {status}")
    return out, scale

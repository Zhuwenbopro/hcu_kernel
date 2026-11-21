"""Qwen3.8-27B TP2 local logits with contiguous BF16 W[K,N]."""

import json
from functools import lru_cache
from pathlib import Path

import torch

from . import rolling
from .native import launch_gemv, launch_indexed, launch_packed_x

K = 5120
N = 124160
MAX_M = 20_480


@lru_cache(maxsize=1)
def _small_policy():
    with Path(__file__).with_name("small-policy.json").open() as file:
        return {int(m): config for m, config in json.load(file).items()}


@lru_cache(maxsize=1)
def _medium_policy():
    with Path(__file__).with_name("medium-policy.json").open() as file:
        rows = json.load(file)
    result = {}
    for key, value in rows.items():
        p, tiles, index = int(key), tuple(value["tiles"]), value["solution"]
        if not (256 <= p <= 1024 and p % 16 == 0 and tiles and
                min(tiles) > 0 and 256 * sum(tiles) == N and index >= 0):
            raise RuntimeError(f"Invalid packed-X route at P={p}")
        result[p] = tiles, index
    return result


def route(m, *, tuned=False):
    """Return the selected family, output tile sizes and executed row count."""
    if not 1 <= m <= MAX_M:
        raise ValueError(f"Expected 1 <= M <= {MAX_M}")
    kernel = rolling.route(m) if tuned else None
    if kernel is not None:
        return "rolling", kernel.config.bm, kernel.config.bn, m
    if m > 128:
        p = (m + 15) // 16 * 16
        family = "segmented" if tuned and m >= 256 and p in _medium_policy() else "large"
        return family, 256, 256, p
    config = _small_policy()[m]
    return config["method"], config["tm"], config["tn"], m


def _check_buffer(tensor, shape, device, name):
    if (tensor.shape != shape or tensor.device != device or tensor.dtype != torch.bfloat16
            or not tensor.is_contiguous() or tensor.data_ptr() % 16):
        raise ValueError(f"Expected 16-byte aligned contiguous BF16 {name}{list(shape)} on {device}")


def _pack_x(x, workspace, rows):
    if workspace is None:
        workspace = x.new_empty((K, rows))
    _check_buffer(workspace, (K, rows), x.device, "workspace")
    workspace[:, :x.shape[0]].copy_(x.T)
    if x.shape[0] < rows:
        workspace[:, x.shape[0]:].zero_()
    return workspace


class LMHead:
    """Hold one caller-owned W[K,N] and compute BF16 local logits.

    Prepare W before using the operator on a different stream. Callers that pass
    output or scratch buffers own their contents and cross-stream dependencies.
    """

    def __init__(self, weight_kn, *, tuned=False):
        _check_buffer(weight_kn, (K, N), weight_kn.device, "W")
        if not weight_kn.is_cuda:
            raise ValueError("W must be on an HCU device")
        properties = torch.cuda.get_device_properties(weight_kn.device)
        if not getattr(properties, "gcnArchName", "").startswith("gfx936"):
            raise ValueError("Expected a gfx936 device")
        self.weight = weight_kn
        self.tuned = tuned
        if tuned:
            rolling.preload(weight_kn.device.index)

    @torch.no_grad()
    def __call__(self, x, out=None, workspace=None):
        """Compute X @ W into out[:M], optionally reusing disjoint buffers."""
        if x.ndim != 2:
            raise ValueError(f"Expected X[M,{K}]")
        m = x.shape[0]
        family, _, _, p = route(m, tuned=self.tuned)
        _check_buffer(x, (m, K), self.weight.device, "X")

        if out is None:
            out = x.new_empty((p, N))
        _check_buffer(out, (p, N), x.device, "out")

        with torch.cuda.device(x.device):
            stream = torch.cuda.current_stream(x.device)
            if family in ("large", "segmented"):
                workspace = _pack_x(x, workspace, p)
                if family == "segmented":
                    tiles, index = _medium_policy()[p]
                    launch_packed_x(index, workspace, self.weight, out, tiles, stream.cuda_stream)
                else:
                    torch.mm(workspace.T, self.weight, out=out)
                workspace.record_stream(stream)
            elif family == "rolling":
                kernel = rolling.route(m)
                operand = x
                if kernel.config.xt:
                    operand = _pack_x(x, workspace, kernel.config.bm)
                rolling.launch(m, operand, self.weight, out, kernel, stream.cuda_stream)
                if kernel.config.xt:
                    operand.record_stream(stream)
            elif family == "gemv":
                launch_gemv(x, self.weight, out, stream.cuda_stream)
            else:
                config = _small_policy()[m]
                if family == "triton":
                    from .triton_kernel import launch
                    tile = (config["tm"], config["tn"], config["tk"], config["waves"])
                    launch(x, self.weight, out, tile)
                else:
                    launch_indexed(config["index"], x, self.weight, out, stream.cuda_stream)
            for tensor in (x, self.weight, out):
                tensor.record_stream(stream)
        return out[:m]

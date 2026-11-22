"""Unfused BF16 GateUp from one contiguous W[K,N]."""

import ctypes as C
from functools import lru_cache
from threading import Lock

import torch
import triton
import triton.language as tl

from .policy import K, SHARDS, SMALL_M, check_m, routes
from .runtime import (
    INT,
    PTR,
    build_catalog,
    check_device,
    check_hip,
    check_tensor,
    checked_path,
    get_workspace,
    load_assembly,
    record_stream,
)


@triton.jit
def _transpose(X, T, M, Pitch, Block: tl.constexpr):
    rows = tl.program_id(0) * Block + tl.arange(0, Block)
    inner = tl.program_id(1) * Block + tl.arange(0, Block)
    values = tl.load(X + rows[:, None] * 5120 + inner[None, :], rows[:, None] < M, 0)
    tl.store(T + inner[None, :] * Pitch + rows[:, None], values, rows[:, None] < Pitch)


@lru_cache(maxsize=None)
def _load_blas(path):
    api = C.CDLL(str(path))
    api.gate_up_blas_init.argtypes = [C.POINTER(PTR)]
    api.gate_up_blas_destroy.argtypes = [PTR]
    api.gate_up_kn_gemm.argtypes = [PTR] * 5 + [INT] * 6
    return api


@lru_cache(maxsize=None)
def _load_gemv(path):
    fn = C.CDLL(str(path)).run
    fn.argtypes = [PTR, PTR, PTR, PTR, INT, INT, PTR]
    return fn


def _check_blas(status):
    if status:
        raise RuntimeError(f"rocBLAS error {status}")


class BF16GateUpGemm:
    """Compute BF16 Y[M,N] with caller-owned, contiguous BF16 W[5120,N]."""

    def __init__(self, weight_kn):
        if weight_kn.ndim != 2 or weight_kn.shape[1] not in SHARDS:
            raise ValueError("Expected BF16 W[5120,N] where N is 17408 or 34816")
        self.device = weight_kn.device
        self.n = weight_kn.shape[1]
        check_tensor(weight_kn, (K, self.n), torch.bfloat16, self.device, 16, "W")
        check_device(self.device)
        self.weight = weight_kn
        self._default = routes("bf16", self.n)
        self._exact = routes("bf16", self.n, "exact")
        self._handles = {}
        self._lock = Lock()

        catalog = build_catalog("bf16")
        self._blas = _load_blas(checked_path("bf16", catalog["blas"]))
        self._rolling = {}
        for name in set(self._default[1:]) | set(self._exact[1:]):
            if name.startswith("roll:"):
                _, key, parts = name.split(":")
                spec = catalog["rolling"][f"{self.n}/{key}"]
                self._rolling[name] = (
                    load_assembly("bf16", self.device, spec),
                    spec["config"],
                    int(parts),
                )
        self._gemv = None
        if str(self.n) in catalog["gemv"]:
            self._gemv = _load_gemv(checked_path("bf16", catalog["gemv"][str(self.n)]))

    def route(self, m, *, exact=False):
        check_m(m)
        return self._exact[m] if exact and m <= SMALL_M else self._default[m]

    def _geometry(self, name, m):
        if name.startswith("roll:"):
            _, q, _ = self._rolling[name]
            if m > q["bm"]:
                raise ValueError(f"{name} does not cover M={m}")
            return (q["bm"] if q["xt"] else 0), m
        if name.startswith("kn-pad-"):
            alignment = int(name.split("-")[2])
            rows = triton.cdiv(m, alignment) * alignment
            return rows, rows
        if name.startswith("kn-blas-1-"):
            return triton.cdiv(m, 16) * 16, m
        return 0, m

    def output_rows(self, m, *, exact=False):
        return self._geometry(self.route(m, exact=exact), m)[1]

    def workspace_elements(self, m, *, exact=False):
        return self._geometry(self.route(m, exact=exact), m)[0] * K

    def _blas_gemm(self, x, y, active_m, parts, transposed, stream):
        width = self.n // parts
        pitch = x.shape[1] if transposed else K
        with self._lock:
            handle = self._handles.get(stream)
            if handle is None:
                handle = PTR()
                _check_blas(self._blas.gate_up_blas_init(C.byref(handle)))
                self._handles[stream] = handle
            for column in range(0, self.n, width):
                offset = column * 2
                _check_blas(
                    self._blas.gate_up_kn_gemm(
                        handle,
                        stream,
                        x.data_ptr(),
                        self.weight.data_ptr() + offset,
                        y.data_ptr() + offset,
                        active_m,
                        width,
                        K,
                        self.n,
                        pitch,
                        transposed,
                    )
                )

    def _launch(self, name, x, out, operand, rows, stream):
        m = x.shape[0]
        if name == "kn-torch":
            torch.mm(x, self.weight, out=out)
        elif name == "gemv:cached:9":
            check_hip(
                self._gemv(
                    x.data_ptr(),
                    self.weight.data_ptr(),
                    out.data_ptr(),
                    None,
                    m,
                    9,
                    stream,
                )
            )
        elif name.startswith("roll:"):
            kernel, q, parts = self._rolling[name]
            tiles = self.n // q["bn"]
            for part in range(parts):
                first, last = part * tiles // parts, (part + 1) * tiles // parts
                offset = q["bn"] * first * 2
                args = (
                    PTR(operand.data_ptr()),
                    PTR(self.weight.data_ptr() + offset),
                    PTR(out.data_ptr() + offset),
                    INT(m),
                    INT(0),
                    INT(K // q["bk"]),
                    INT(0),
                )
                kernel.launch((last - first, 1, 1), q["nw"] * 64, args, stream)
        elif name.startswith("kn-pad-"):
            parts = int(name.split("-")[-1])
            self._blas_gemm(operand, out, rows, parts, 1, stream)
        elif name.startswith("kn-blas-"):
            transposed, parts, solution = map(int, name.split("-")[2:])
            if solution:
                raise ValueError(
                    "BF16 GateUp uses automatic rocBLAS solution selection"
                )
            self._blas_gemm(operand, out, m, parts, transposed, stream)
        else:
            raise ValueError(f"Unknown BF16 route {name}")

    @torch.inference_mode()
    def __call__(self, x, *, out=None, workspace=None):
        m = x.shape[0] if x.ndim == 2 else 0
        check_tensor(x, (m, K), torch.bfloat16, self.device, 16, "X")
        exact = out is not None and out.shape == (m, self.n)
        name = self.route(m, exact=exact)
        pitch, rows = self._geometry(name, m)
        with torch.cuda.device(self.device):
            if out is None:
                out = x.new_empty((rows, self.n))
            check_tensor(out, (rows, self.n), torch.bfloat16, self.device, 16, "Y")
            live = (self.weight, x, out)
            operand = x
            if pitch:
                workspace = get_workspace(x, workspace, pitch * K)
                operand = workspace[: pitch * K].view(K, pitch)
                live += (workspace,)
            if len({tensor.untyped_storage().data_ptr() for tensor in live}) != len(
                live
            ):
                raise ValueError("W, X, output and workspace require separate storage")
            if pitch:
                _transpose[(triton.cdiv(pitch, 32), K // 32)](
                    x, operand, m, pitch, 32, num_warps=4
                )
            active = torch.cuda.current_stream(self.device)
            self._launch(name, x, out, operand, rows, active.cuda_stream)
            record_stream(active, *live)
        return out if rows == m else out[:m]

    def close(self):
        """Release rocBLAS handles after the caller has synchronized outstanding work."""
        with torch.cuda.device(self.device), self._lock:
            while self._handles:
                stream, handle = next(iter(self._handles.items()))
                _check_blas(self._blas.gate_up_blas_destroy(handle))
                del self._handles[stream]

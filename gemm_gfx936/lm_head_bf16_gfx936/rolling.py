"""Load and launch the fixed-register gfx936 rolling GEMM."""

import ctypes as c
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from threading import Lock

import torch

from .kernels.rolling_asm import Config, K, N


_DATA = Path(__file__).resolve().parent
_BUILD = _DATA / "build" / "rolling"
_PTR = c.c_void_p
_UINT = c.c_uint32
_LOAD_LOCK = Lock()
_MODULES = {}


@dataclass(frozen=True)
class Kernel:
    config: Config
    parts: int
    reference_sha256: str

    @property
    def stem(self):
        return self.config.name + "-rolling"


@lru_cache(maxsize=1)
def policy():
    with (_DATA / "rolling-policy.json").open() as file:
        raw = json.load(file)
    if (raw["k"], raw["n"]) != (K, N):
        raise RuntimeError("Unexpected rolling LM-head weight shape")
    kernels = {}
    for name, fields in raw["kernels"].items():
        args = dict(fields)
        parts = args.pop("parts")
        digest = args.pop("reference_sha256")
        cfg = Config(**args)
        if K % cfg.bk or N % cfg.bn or not 1 <= parts <= N // cfg.bn:
            raise RuntimeError(f"Invalid rolling LM-head geometry for {name}")
        kernels[name] = Kernel(cfg, parts, digest)

    routes = {}
    for first, last, name in raw["ranges"]:
        kernel = kernels[name]
        for m in range(first, last + 1):
            if m in routes or m > kernel.config.bm:
                raise RuntimeError(f"Invalid rolling LM-head route at M={m}")
            routes[m] = kernel
    if set(routes) != set(range(2, 193)):
        raise RuntimeError("Incomplete rolling LM-head policy")
    return tuple(kernels.values()), routes


def route(m):
    return policy()[1].get(m) if 2 <= m <= 192 else None


@lru_cache(maxsize=1)
def _runtime():
    hip = c.CDLL("libamdhip64.so")
    hip.hipGetErrorString.argtypes = [c.c_int]
    hip.hipGetErrorString.restype = c.c_char_p
    hip.hipModuleLoad.argtypes = [c.POINTER(_PTR), c.c_char_p]
    hip.hipModuleLoad.restype = c.c_int
    hip.hipModuleUnload.argtypes = [_PTR]
    hip.hipModuleUnload.restype = c.c_int
    hip.hipModuleGetFunction.argtypes = [c.POINTER(_PTR), _PTR, c.c_char_p]
    hip.hipModuleGetFunction.restype = c.c_int
    hip.hipExtModuleLaunchKernel.argtypes = (
        [_PTR] + [_UINT] * 6 +
        [c.c_size_t, _PTR, c.POINTER(_PTR), c.POINTER(_PTR), _PTR, _PTR, _UINT]
    )
    hip.hipExtModuleLaunchKernel.restype = c.c_int
    return hip


def _check(hip, status, operation):
    if status:
        error = hip.hipGetErrorString(status)
        message = error.decode() if error else str(status)
        raise RuntimeError(f"gfx936 rolling LM head {operation} failed: HIP {status} {message}")


def preload(device):
    """Load every kernel on the correct device before Graph capture."""
    with _LOAD_LOCK:
        cached = _MODULES.get(device)
        if cached is not None:
            return cached

        audit = _BUILD / "audit.json"
        if not audit.is_file():
            raise RuntimeError(f"Run python -m {__package__}.build --tuned")
        hashes = json.loads(audit.read_text())
        paths = {}
        for kernel in policy()[0]:
            path = _BUILD / f"{kernel.stem}.hsaco"
            if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[kernel.stem]["sha256"]:
                raise RuntimeError(f"Code object digest mismatch: {path.name}")
            paths[kernel] = path

        hip = _runtime()
        loaded = {}
        objects = []
        with torch.cuda.device(device):
            try:
                for kernel, path in paths.items():
                    module, function = _PTR(), _PTR()
                    _check(hip, hip.hipModuleLoad(c.byref(module), bytes(path)), path.name)
                    objects.append(module)
                    _check(hip, hip.hipModuleGetFunction(c.byref(function), module, b"rolling_mmac"),
                           f"{path.name} symbol lookup")
                    loaded[kernel] = module, function
            except RuntimeError:
                for module in objects:
                    hip.hipModuleUnload(module)
                raise
        _MODULES[device] = loaded
        return loaded


def launch(m, x, w, out, kernel, stream):
    cfg = kernel.config
    hip = _runtime()
    _, function = preload(x.device.index)[kernel]
    tiles = N // cfg.bn
    threads = cfg.nw * 64
    for part in range(kernel.parts):
        first = part * tiles // kernel.parts
        last = (part + 1) * tiles // kernel.parts
        offset = cfg.bn * first * 2
        args = (
            _PTR(x.data_ptr()), _PTR(w.data_ptr() + offset), _PTR(out.data_ptr() + offset),
            _UINT(m), _UINT(0), _UINT(K // cfg.bk), _UINT(0),
        )
        params = (_PTR * len(args))(*(c.cast(c.byref(arg), _PTR) for arg in args))
        _check(hip, hip.hipExtModuleLaunchKernel(
            function, (last - first) * threads, 1, 1, threads, 1, 1, 0, stream,
            params, None, None, None, 0,
        ), f"M={m} segment {part + 1}/{kernel.parts}")

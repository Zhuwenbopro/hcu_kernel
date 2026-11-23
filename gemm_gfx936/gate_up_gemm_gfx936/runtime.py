"""Load fixed-register HIP code objects and check shared GEMM buffers."""

import ctypes as C
from functools import lru_cache
import hashlib
import json

import torch

from .policy import ROOT

PTR = C.c_void_p
INT = C.c_int
BUILD = ROOT / "build"


@lru_cache(maxsize=1)
def hip():
    lib = C.CDLL("libamdhip64.so")
    lib.hipGetErrorString.argtypes = [INT]
    lib.hipGetErrorString.restype = C.c_char_p
    lib.hipModuleLoad.argtypes = [C.POINTER(PTR), C.c_char_p]
    lib.hipModuleUnload.argtypes = [PTR]
    lib.hipModuleGetFunction.argtypes = [C.POINTER(PTR), PTR, C.c_char_p]
    lib.hipModuleLaunchKernel.argtypes = (
        [PTR] + [C.c_uint] * 7 + [PTR, C.POINTER(PTR), PTR]
    )
    lib.hipFuncGetAttribute.argtypes = [C.POINTER(INT), INT, PTR]
    return lib


def check_hip(status):
    if status:
        raise RuntimeError(
            f"HIP error {status}: {hip().hipGetErrorString(status).decode()}"
        )


def build_catalog(kind):
    path = BUILD / kind / "kernels.json"
    if not path.is_file():
        raise RuntimeError(f"Run python -m {__package__}.build_{kind} first")
    return json.loads(path.read_text(encoding="utf-8"))


def checked_path(kind, record):
    path = BUILD / kind / record["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise RuntimeError(f"Unexpected code object digest for {path}")
    return path


class AssemblyKernel:
    def __init__(self, path, symbol, lds, vgprs):
        self.module = PTR()
        self.function = PTR()
        api = hip()
        check_hip(api.hipModuleLoad(C.byref(self.module), bytes(path)))
        try:
            check_hip(
                api.hipModuleGetFunction(
                    C.byref(self.function), self.module, symbol.encode()
                )
            )
            # HIP function attributes for static shared memory, local memory, and registers.
            for attribute, expected in ((1, lds), (3, 0), (4, vgprs)):
                actual = INT()
                check_hip(
                    api.hipFuncGetAttribute(C.byref(actual), attribute, self.function)
                )
                if actual.value != expected:
                    raise RuntimeError(
                        f"{symbol}: attribute {attribute} is {actual.value}, expected {expected}"
                    )
        except Exception:
            api.hipModuleUnload(self.module)
            raise

    def launch(self, grid, threads, args, stream):
        pointers = (PTR * len(args))(*(C.addressof(arg) for arg in args))
        check_hip(
            hip().hipModuleLaunchKernel(
                self.function,
                *grid,
                threads,
                1,
                1,
                0,
                stream,
                pointers,
                None,
            )
        )


@lru_cache(maxsize=None)
def _load_assembly(kind, device, filename, digest, symbol, lds, vgprs):
    path = checked_path(kind, {"file": filename, "sha256": digest})
    with torch.cuda.device(device):
        return AssemblyKernel(path, symbol, lds, vgprs)


def load_assembly(kind, device, record):
    return _load_assembly(
        kind,
        device,
        record["file"],
        record["sha256"],
        record["symbol"],
        record["lds"],
        record["vgprs"],
    )


def check_device(device):
    if device.type != "cuda":
        raise ValueError("GateUp requires a gfx936 BW1000 device")
    props = torch.cuda.get_device_properties(device)
    if (
        not getattr(props, "gcnArchName", "").startswith("gfx936")
        or props.multi_processor_count != 80
    ):
        raise ValueError("GateUp requires a gfx936 BW1000 device with 80 CUs")


def check_tensor(tensor, shape, dtype, device, alignment, label):
    if (
        tensor is None
        or tensor.shape != shape
        or tensor.dtype != dtype
        or tensor.device != device
        or not tensor.is_contiguous()
        or tensor.data_ptr() % alignment
    ):
        raise ValueError(
            f"Expected contiguous {dtype} {label}{shape} aligned to {alignment} bytes on {device}"
        )


def get_workspace(x, workspace, elements):
    if workspace is None:
        return x.new_empty(elements)
    if (
        workspace.numel() < elements
        or workspace.dtype != x.dtype
        or workspace.device != x.device
        or not workspace.is_contiguous()
        or workspace.data_ptr() % 16
    ):
        raise ValueError(
            f"Expected aligned contiguous X workspace of at least {elements} elements"
        )
    return workspace.view(-1)


def record_stream(active, *tensors):
    for tensor in tensors:
        tensor.record_stream(active)

"""Single-node gfx936 TP8 BF16 IPC for AR + residual add + Gemma RMSNorm.

Each rank uses one ordered HIP stream. All ranks must issue the same calls
(shape and algorithm) in the same order, including during Graph replay.
Construction and close are collective on the supplied eight-rank Gloo group.
Automatic dispatch defaults to the measured cutoff of 12. Override direct_through or use
"direct" or "sharded" to benchmark before the crossover is known.
"""
import ctypes
import socket
from pathlib import Path

import torch
import torch.distributed as dist


_WIDTH = 5120
_ALGORITHMS = {"direct": 0, "sharded": 1}
_IPCHandle = ctypes.c_ubyte * 64
_PointerTable = ctypes.c_void_p * 8


def _load_library(path):
    lib = ctypes.CDLL(str(path))
    p, i, z = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
    for name, returns, args in (
        ("tp8_signal_bytes", z, [i]),
        ("tp8_data_bytes", z, [z]),
        ("tp8_alloc", p, [z]),
        ("tp8_free", i, [p]),
        ("tp8_handle", i, [p, p]),
        ("tp8_open", p, [p]),
        ("tp8_close", i, [p]),
        ("tp8_create", p, [i, ctypes.POINTER(p), ctypes.POINTER(p), z, i]),
        ("tp8_destroy", i, [p]),
        ("tp8_run", i, [p, i, p, p, p, p, p, i, ctypes.c_float, p]),
    ):
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = returns, args
    return lib


class TP8Allreduce:
    def __init__(self, group, device, max_rows=4096, blocks=64, *, library=None, direct_through=12):
        if dist.get_world_size(group) != 8 or dist.get_backend(group) != "gloo":
            raise ValueError("TP8 IPC requires a eight-rank Gloo control group")
        self.group = group
        self.rank = dist.get_rank(group)
        self.max_rows = max_rows
        self.blocks = blocks
        self.direct_through = direct_through
        self._ctx = None
        self._lib = None
        self._local = []
        self._opened = []
        self._ready = False
        self._peer_phase = False
        self.device = None
        error, handles = None, []
        try:
            if direct_through is not None and (type(direct_through) is not int or not 0 <= direct_through <= 4096):
                raise ValueError("direct_through must be an integer in 0..4096")
            if not (1 <= max_rows <= 4096 and 1 <= blocks <= 80):
                raise ValueError("max_rows must be in 1..4096 and blocks in 1..80")
            self.device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
            if self.device.type != "cuda":
                raise ValueError("a HIP device is required")
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            with torch.cuda.device(self.device):
                props = torch.cuda.get_device_properties(self.device)
                if props.gcnArchName.split(":")[0] != "gfx936":
                    raise ValueError(f"unsupported GPU architecture {props.gcnArchName}")
                self.library = Path(library or Path(__file__).resolve().parent / "native" / "libtp8.so")
                self._lib = _load_library(self.library)
                self.max_bytes = max_rows * _WIDTH * 2
                self.signal_bytes = self._lib.tp8_signal_bytes(blocks)
                self.data_bytes = self._lib.tp8_data_bytes(self.max_bytes)
                for size in (self.signal_bytes, self.data_bytes):
                    if not size:
                        raise RuntimeError("invalid communication allocation size")
                    ptr = self._lib.tp8_alloc(size)
                    if not ptr:
                        raise RuntimeError(f"TP8 IPC allocation failed for {size} bytes")
                    self._local.append(ptr)
                    handle = _IPCHandle()
                    rc = self._lib.tp8_handle(ptr, handle)
                    if rc:
                        raise RuntimeError(f"tp8_handle returned {rc}")
                    handles.append(bytes(handle))
        except Exception as exc:
            error = repr(exc)

        try:
            offers = [None] * 8
            setup = (socket.gethostname(), max_rows, blocks, direct_through)
            dist.all_gather_object(offers, (error, setup, handles), group=group)
            if any(o[0] for o in offers):
                raise RuntimeError(f"TP8 IPC export failed: {[o[0] for o in offers]}")
            if any(o[1] != setup for o in offers):
                raise ValueError("TP8 requires all ranks on one host with identical workspace settings")
            self._peer_phase = True
            with torch.cuda.device(self.device):
                try:
                    tables = []
                    for kind in range(2):
                        table = _PointerTable()
                        for rank, (_, _, exported) in enumerate(offers):
                            if rank == self.rank:
                                table[rank] = self._local[kind]
                            else:
                                handle = _IPCHandle.from_buffer_copy(exported[kind])
                                peer = self._lib.tp8_open(handle)
                                if not peer:
                                    raise RuntimeError(f"cannot map IPC allocation {kind} from rank {rank}")
                                self._opened.append(peer)
                                table[rank] = peer
                        tables.append(table)
                    self._ctx = self._lib.tp8_create(self.rank, tables[0], tables[1], self.max_bytes, blocks)
                    if not self._ctx:
                        raise RuntimeError("tp8_create rejected pointers or grid occupancy")
                except Exception as exc:
                    error = repr(exc)
                errors = [None] * 8
                dist.all_gather_object(errors, error, group=group)
                if any(errors):
                    raise RuntimeError(f"TP8 IPC initialization failed: {errors}")
                self._ready = True
        except BaseException:
            self.close()
            raise

    def fused_into(self, x, residual, weight, out, updated, eps, algorithm="auto"):
        """Launch into separate, aligned output tensors on the current HIP stream."""
        if not self._ready:
            raise RuntimeError("TP8 communicator is closed")
        if x.ndim != 2 or not 1 <= x.shape[0] <= self.max_rows or x.shape[1] != _WIDTH:
            raise ValueError(f"expected BF16 [1..{self.max_rows},{_WIDTH}]")
        if algorithm == "auto":
            if self.direct_through is None:
                raise ValueError("auto requires a TP8 pilot cutoff in the direct_through constructor argument")
            algorithm = "direct" if x.shape[0] <= self.direct_through else "sharded"
        if algorithm not in _ALGORITHMS:
            raise ValueError(f"unknown TP8 algorithm {algorithm!r}")
        if any(t.shape != x.shape for t in (residual, out, updated)) or weight.shape != (_WIDTH,):
            raise ValueError("TP8 residual, output or weight shape mismatch")
        tensors = (x, residual, weight, out, updated)
        if any(t.device != self.device or t.dtype != torch.bfloat16 or not t.is_contiguous() for t in tensors):
            raise ValueError("TP8 requires contiguous BF16 tensors on the communicator device")
        addresses = [t.data_ptr() for t in tensors]
        if any(addr % 16 for addr in addresses):
            raise ValueError("TP8 requires 16-byte aligned tensors")
        ranges = [(addr, addr + t.numel()*2) for addr,t in zip(addresses,tensors)]
        for i in (3,4):
            start,end = ranges[i]
            if any(start < other_end and other_start < end for other_start,other_end in ranges[:i]):
                raise ValueError("TP8 outputs must not overlap the inputs or each other")
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device).cuda_stream
            rc = self._lib.tp8_run(self._ctx, _ALGORITHMS[algorithm], *addresses, x.shape[0], eps, stream)
            if rc:
                raise RuntimeError(f"tp8_run({algorithm}, M={x.shape[0]}) returned {rc}")
        return out, updated

    def fused(self, x, residual, weight, eps, algorithm="auto"):
        out, updated = torch.empty_like(x), torch.empty_like(residual)
        return self.fused_into(x, residual, weight, out, updated, eps, algorithm=algorithm)

    def close(self):
        """Synchronize users, close peer mappings, then release their backing allocations."""
        if not self._peer_phase and not self._local:
            return
        with torch.cuda.device(self.device):
            if self._ready:
                torch.cuda.synchronize(self.device)
                dist.barrier(group=self.group)
            errors = []
            if self._ctx:
                rc = self._lib.tp8_destroy(self._ctx)
                if rc:
                    errors.append(("destroy", rc))
                self._ctx = None
            for ptr in self._opened:
                rc = self._lib.tp8_close(ptr)
                if rc:
                    errors.append(("close", rc))
            self._opened.clear()
            if self._peer_phase:
                dist.barrier(group=self.group)
            self._ready = self._peer_phase = False
            for ptr in self._local:
                rc = self._lib.tp8_free(ptr)
                if rc:
                    errors.append(("free", rc))
            self._local.clear()
            if errors:
                raise RuntimeError(f"TP8 IPC release error: {errors}")

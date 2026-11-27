"""TP2 BF16 IPC collectives for a single, stream-ordered gfx936 device group."""

import ctypes
import socket
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist


class _OsarLib:
    def __init__(self, library, tuned):
        lib = ctypes.CDLL(str(library))
        p, i, f = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
        for name, result, args in (
            ("osar_meta_alloc", p, [i, i]),
            ("osar_meta_free", None, [p]),
            ("osar_get_handle", i, [p, ctypes.c_char_p, i]),
            ("osar_open", p, [ctypes.c_char_p]),
            ("osar_close_handle", i, [p]),
            ("osar_init", p, [i, i, p, ctypes.POINTER(p), i]),
            ("osar_set_regbuf_sized", i, [p, p, ctypes.POINTER(p), i]),
            ("osar_run_ci", i, [p, p, p, i, i, p]),
            ("osar_run_fused", i, [p, p, p, p, p, p, i, i, i, f, p]),
            ("osar_dispose", i, [p]),
        ):
            entry = getattr(lib, name)
            entry.restype, entry.argtypes = result, args
        if tuned:
            lib.osar_run_fused_grid.restype = i
            lib.osar_run_fused_grid.argtypes = [p, p, p, p, p, p, i, i, i, f, p, i]
        self.run_fused_int8 = getattr(lib, "osar_run_fused_int8", None)
        if self.run_fused_int8 is not None:
            self.run_fused_int8.restype = i
            self.run_fused_int8.argtypes = [p, p, p, p, p, p, p, i, i, i, f, p, i, i]
        self.fused_int8_block_limit = getattr(lib, "osar_fused_int8_block_limit", None)
        if self.fused_int8_block_limit is not None:
            self.fused_int8_block_limit.restype = i
            self.fused_int8_block_limit.argtypes = [p, i]
        self.has_dual_staging = hasattr(lib, "osar_staging_alloc") and hasattr(lib, "osar_set_regbuf_dual")
        if self.has_dual_staging:
            lib.osar_staging_alloc.argtypes = [i]
            lib.osar_staging_alloc.restype = p
            lib.osar_set_regbuf_dual.argtypes = [p, p, ctypes.POINTER(p), i]
            lib.osar_set_regbuf_dual.restype = i
        self.set_staging = lib.osar_set_regbuf_dual if self.has_dual_staging else lib.osar_set_regbuf_sized
        self.lib = lib
        self.variant = 0

    def allocate_staging(self, size):
        if self.has_dual_staging:
            return self.lib.osar_staging_alloc(size)
        return self.lib.osar_meta_alloc(size, 1)

    def handle(self, ptr):
        buf = ctypes.create_string_buffer(64)
        if not ptr or self.lib.osar_get_handle(ptr, buf, len(buf)):
            raise RuntimeError("failed to export an uncached HIP IPC allocation")
        return buf.raw


class HipOneShotAllreduce:
    """Use one ordered stream per rank. Both ranks create and close the context
    collectively while the CPU group is alive, after their final submission.
    """

    _kernel = "fixed"

    def __init__(self, group=None, device=None, max_size=4096 * 5120 * 2, variant=0, blocks=64, *, library=None):
        self.group = group if group is not None else dist.new_group(backend="gloo")
        self.rank = dist.get_rank(self.group)
        self.world_size = dist.get_world_size(self.group)
        if self.world_size != 2 or dist.get_backend(self.group) != "gloo":
            raise ValueError("gfx936 OSAR requires a two-rank CPU Gloo group")
        self.device = None
        self.max_size = max_size
        self._block_cap = blocks
        self._ctx = ctypes.c_void_p()
        self._local = []
        self._opened = []
        self._osar = None
        self._ready = False
        self._int8_block_limits = None
        self._close_mappings_together = False
        local_error = None
        handles = []
        try:
            self._osar = _OsarLib(library or Path(__file__).parent / "lib" / f"libosar_{self._kernel}.so",
                                  self._kernel == "tuned")
            if isinstance(device, int):
                device = torch.device("cuda", device)
            self.device = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
            if self.device.index is None:
                self.device = torch.device(self.device.type, torch.cuda.current_device())
            if self.device.type != "cuda":
                raise ValueError(f"unsupported device {self.device}")
            with torch.cuda.device(self.device):
                arch = getattr(torch.cuda.get_device_properties(self.device), "gcnArchName", "")
                if arch.split(":")[0] != "gfx936":
                    raise ValueError(f"unsupported gfx architecture {arch}")
                if variant not in (None, 0) or not (16 <= max_size <= 2**31-1) or not (1 <= blocks <= 128):
                    raise ValueError("unsupported variant, staging size or block count")
                meta = self._osar.lib.osar_meta_alloc(1 << 16, 1)
                if meta:
                    self._local.append(meta)
                handles.append(self._osar.handle(meta))
                data = self._osar.allocate_staging(max_size)
                if data:
                    self._local.append(data)
                handles.append(self._osar.handle(data))
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"

        # Even a rank with a missing library or an invalid device participates in
        # the CPU agreement. No rank tries opening a peer handle before it passes.
        offers = [None, None]
        try:
            dist.all_gather_object(offers, (local_error, handles, blocks, max_size, socket.gethostname(), self._kernel,
                                            bool(self._osar and self._osar.has_dual_staging)), group=self.group)
            if any(offer[0] for offer in offers):
                raise RuntimeError(f"IPC export failed: {[offer[0] for offer in offers]}")
            if offers[0][2:] != offers[1][2:]:
                raise RuntimeError(f"incompatible IPC group settings: {[offer[2:] for offer in offers]}")
            lib = self._osar.lib
            with torch.cuda.device(self.device):
                local_int8_limits = None
                try:
                    tables = []
                    for idx in range(2):
                        peer_ptr = lib.osar_open(offers[1-self.rank][1][idx])
                        if not peer_ptr:
                            raise RuntimeError("hipIpcOpenMemHandle failed")
                        self._opened.append(peer_ptr)
                        table = (ctypes.c_void_p * 2)()
                        table[self.rank], table[1-self.rank] = self._local[idx], peer_ptr
                        tables.append(table)
                    self._ctx = ctypes.c_void_p(lib.osar_init(self.rank, 2, self._local[0], tables[0], blocks))
                    if not self._ctx.value:
                        raise RuntimeError("osar_init failed, check occupancy and grid cap")
                    self._opened.pop(0)  # the native context owns the peer signal
                    if self._osar.set_staging(self._ctx, self._local[1], tables[1], max_size):
                        raise RuntimeError("osar_set_regbuf_sized failed")
                    self._opened.pop()  # the context now owns the peer staging mapping
                    int8_block_limit = self._osar.fused_int8_block_limit
                    if self._osar.run_fused_int8 is not None and int8_block_limit is not None:
                        local_int8_limits = (int8_block_limit(self._ctx, 0), int8_block_limit(self._ctx, 1))
                except Exception as error:
                    local_error = str(error)
                reports = [None, None]
                dist.all_gather_object(reports, (local_error, local_int8_limits), group=self.group)
                # Both ranks have now attempted to map peer memory. Even a
                # partially initialized mapping must close before the exporter
                # releases that allocation.
                self._close_mappings_together = True
                errors, int8_limits = zip(*reports)
                if any(errors):
                    raise RuntimeError(f"IPC group setup failed: {errors}")
                if all(limits is not None for limits in int8_limits):
                    self._int8_block_limits = tuple(min(limits) for limits in zip(*int8_limits))
                self._ready = True
        except BaseException:
            self.close()
            raise

    def should_custom_ar(self, inp):
        return (
            self._ctx.value is not None and inp.device == self.device and inp.dtype == torch.bfloat16
            and inp.is_contiguous() and inp.data_ptr() % 16 == 0
            and 0 < inp.numel()*2 <= self.max_size and inp.numel() % 8 == 0
        )

    def custom_all_reduce(self, input):
        if not self.should_custom_ar(input):
            return None
        with torch.cuda.device(self.device):
            out = torch.empty_like(input)
            rc = self._osar.lib.osar_run_ci(self._ctx, input.data_ptr(), out.data_ptr(), input.numel()*2, 0,
                                            torch.cuda.current_stream(self.device).cuda_stream)
        if rc:
            raise RuntimeError(f"osar_run_ci failed ({rc})")
        return out

    def all_reduce(self, inp):
        out = self.custom_all_reduce(inp)
        if out is None:
            raise ValueError("ineligible BF16 IPC input tensor")
        return out

    def _check_fused_args(self, inp, residual, weight, blocks):
        if blocks is not None and self._kernel != "tuned":
            raise ValueError("the fixed-grid library does not support a per-call grid cap")
        if not (
            self.should_custom_ar(inp) and inp.ndim == 2 and 1 <= inp.shape[0] <= 4096 and inp.shape[1] == 5120
            and residual.shape == inp.shape and residual.device == self.device
            and residual.dtype == torch.bfloat16 and residual.is_contiguous() and residual.data_ptr() % 16 == 0
            and weight.shape == (5120,) and weight.device == self.device and weight.is_contiguous()
            and weight.dtype in (torch.bfloat16, torch.float32) and weight.data_ptr() % 16 == 0
        ):
            raise ValueError("ineligible gfx936 fused AR/add/Gemma RMSNorm arguments")

    def fused(self, inp, residual, weight, eps, *, blocks=None):
        self._check_fused_args(inp, residual, weight, blocks)
        with torch.cuda.device(self.device):
            out, updated = torch.empty_like(inp), torch.empty_like(residual)
            launch = self._osar.lib.osar_run_fused if blocks is None else self._osar.lib.osar_run_fused_grid
            extra = () if blocks is None else (blocks,)
            rc = launch(
                self._ctx, inp.data_ptr(), residual.data_ptr(), weight.data_ptr(), out.data_ptr(), updated.data_ptr(),
                inp.shape[0], inp.shape[1], 2 if weight.dtype == torch.bfloat16 else 1, eps,
                torch.cuda.current_stream(self.device).cuda_stream, *extra,
            )
        if rc:
            raise RuntimeError(f"osar_run_fused failed ({rc})")
        return out, updated

    def fused_int8(self, inp, residual, weight, eps, *, fast=False, blocks=None):
        """Return INT8 output, FP32 per-token scales and the updated BF16 residual.

        Both ranks must use the same mode, grid and input shape at each call.
        """
        if self._int8_block_limits is None:
            raise RuntimeError("fused_int8 requires rebuilt libraries on both ranks")
        if type(fast) is not bool:
            raise TypeError("fast must be a bool")
        self._check_fused_args(inp, residual, weight, blocks)
        cap = self._block_cap if blocks is None else blocks
        if not 1 <= cap <= self._block_cap:
            raise ValueError(f"fused_int8 grid cap must be in 1..{self._block_cap}")
        mode = int(fast)
        grid = min(cap, inp.shape[0])
        if grid > self._int8_block_limits[mode]:
            raise RuntimeError(
                f"fused_int8 needs {grid} resident CTAs, but the group supports "
                f"{self._int8_block_limits[mode]} in this mode"
            )
        with torch.cuda.device(self.device):
            q = torch.empty_like(inp, dtype=torch.int8)
            scale = torch.empty((inp.shape[0], 1), device=self.device, dtype=torch.float32)
            updated = torch.empty_like(residual)
            rc = self._osar.run_fused_int8(
                self._ctx, inp.data_ptr(), residual.data_ptr(), weight.data_ptr(), q.data_ptr(),
                scale.data_ptr(), updated.data_ptr(), inp.shape[0], inp.shape[1],
                2 if weight.dtype == torch.bfloat16 else 1, eps,
                torch.cuda.current_stream(self.device).cuda_stream, 0 if blocks is None else blocks, mode,
            )
        if rc:
            raise RuntimeError(f"osar_run_fused_int8 failed ({rc})")
        return q, scale, updated

    @contextmanager
    def capture(self):
        yield

    def close(self):
        """Collective on a ready group's two ranks. Subsequent calls are local no-ops."""
        if not self._ctx.value and not self._opened and not self._local:
            return
        with torch.cuda.device(self.device):
            lib = self._osar.lib
            if self._ready:
                # Returning from the final local launch need not mean the peer
                # finished reading our last published slot.
                torch.cuda.synchronize(self.device)
                dist.barrier(group=self.group)
            if self._ctx.value:
                lib.osar_dispose(self._ctx)
                self._ctx = ctypes.c_void_p()
            for ptr in self._opened:
                lib.osar_close_handle(ptr)
            self._opened.clear()
            if self._close_mappings_together:
                dist.barrier(group=self.group)
            self._ready = False
            self._int8_block_limits = None
            self._close_mappings_together = False
            for ptr in self._local:
                lib.osar_meta_free(ptr)
            self._local.clear()


def fused_grid(rows):
    if 65 <= rows <= 96:
        return 128
    if 129 <= rows <= 160:
        return 80
    if 193 <= rows <= 216:
        return 72
    if 257 <= rows <= 272:
        return 96
    return 64

class TunedOneShotAllreduce(HipOneShotAllreduce):
    """BW47 gfx936 TP2, BF16 [M, 5120], one dual-slot communication buffer."""

    _kernel = "tuned"

    def __init__(self, group=None, device=None, max_size=4096 * 5120 * 2, *, library=None):
        super().__init__(group=group, device=device, max_size=max_size, blocks=128, library=library)

    def fused(self, inp, residual, weight, eps):
        rows = inp.shape[0] if inp.ndim == 2 else 0
        return super().fused(inp, residual, weight, eps, blocks=fused_grid(rows))

    def fused_int8(self, inp, residual, weight, eps, *, fast=False):
        rows = inp.shape[0] if inp.ndim == 2 else 0
        return super().fused_int8(inp, residual, weight, eps, fast=fast, blocks=fused_grid(rows))


class RangeSelector:
    """Route fused calls to independent original and tuned communicators."""

    def __init__(self, original, tuned, *, windows_only=False):
        self.original = original
        self.tuned = tuned
        self.windows_only = windows_only

    def communicator(self, rows):
        if self.windows_only:
            return self.tuned if fused_grid(rows) != 64 else self.original
        return self.tuned if 65 <= rows <= 512 else self.original

    def fused(self, inp, residual, weight, eps):
        rows = inp.shape[0] if inp.ndim == 2 else 0
        return self.communicator(rows).fused(inp, residual, weight, eps)

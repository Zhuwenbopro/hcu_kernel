"""Unfused INT8 GateUp with fixed-register assembly and one packed weight."""

import torch
import triton
import triton.language as tl

from .policy import K, SHARDS, check_m, routes
from .runtime import (
    INT,
    PTR,
    build_catalog,
    check_device,
    check_tensor,
    get_workspace,
    load_assembly,
    record_stream,
)

STAGE_K = 64


@triton.jit
def _pack(
    Src,
    Dst,
    M,
    Pad,
    Stage: tl.constexpr,
    Width: tl.constexpr,
    Weight: tl.constexpr,
    Block: tl.constexpr,
):
    z = tl.program_id(0) * Block + tl.arange(0, Block)
    intra = z // Width % 16
    if Weight:
        intra = 4 * (intra % 4) + intra // 4
    row = z // (16 * Stage) * 16 + intra
    stage = tl.program_id(1)
    col = stage * Stage + (z // (16 * Width) % 4) * Width + z % Width
    value = tl.load(Src + row * 5120 + col, (z < Pad * Stage) & (row < M), 0)
    tl.store(Dst + stage * Pad * Stage + z, value, z < Pad * Stage)


@triton.jit
def _pack_x_tile(Src, Dst, M, Pitch):
    # Read 16 contiguous row segments and reorder four K64 stages to match
    # the assembly kernel's packed-X fragment order.
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    cols = tl.program_id(1) * 256 + tl.arange(0, 256)
    tile = tl.load(Src + rows[:, None] * 5120 + cols[None, :], rows[:, None] < M, 0)
    tile = tl.reshape(tile, (16, 4, 4, 16))
    tile = tl.permute(tile, (1, 2, 0, 3))
    tile = tl.reshape(tile, (4, 1024))
    stages = tl.program_id(1) * 4 + tl.arange(0, 4)
    offsets = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    tl.store(Dst + stages[:, None] * (Pitch * 64) + offsets[None, :], tile)


@triton.jit
def _pack_x_linear(SRC, DST, rows, PAD: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    # Traverse packed output contiguously; PAD is part of the tuned launch contract.
    z = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = ((z // 1024) % (PAD // 16)) * 16 + (z // 16) % 16
    col = (z // (PAD * 64)) * 64 + ((z // 256) % 4) * 16 + z % 16
    value = tl.load(SRC + row * K + col, (z < PAD * K) & (row < rows), 0)
    tl.store(DST + z, value, z < PAD * K)


def _pack_into(source, dest, rows, *, weight=False):
    _pack[(triton.cdiv(rows * STAGE_K, 4096), K // STAGE_K)](
        source,
        dest,
        source.shape[0],
        rows,
        STAGE_K,
        STAGE_K // 4,
        weight,
        4096,
        num_warps=4,
    )


@torch.inference_mode()
def pack_int8_weight(weight_nk):
    """Prepare opaque packed storage from a contiguous signed INT8 W[N,5120]."""
    if weight_nk.ndim != 2 or weight_nk.shape[0] not in SHARDS:
        raise ValueError("Expected INT8 W[N,5120] where N is 17408 or 34816")
    n, device = weight_nk.shape[0], weight_nk.device
    check_tensor(weight_nk, (n, K), torch.int8, device, 1, "W")
    check_device(device)
    with torch.cuda.device(device):
        packed = weight_nk.new_empty(n * K)
        _pack_into(weight_nk, packed, n, weight=True)
        record_stream(torch.cuda.current_stream(device), weight_nk)
    return packed


class Int8GateUpGemm:
    """Compute BF16 Y[M,N] using signed INT8 X, packed W and FP32 scales."""

    def __init__(self, weight_packed, scale_w):
        self.n = weight_packed.numel() // K
        self.device = weight_packed.device
        if self.n not in SHARDS:
            raise ValueError("Expected packed W for N=17408 or N=34816")
        check_tensor(
            weight_packed, (self.n * K,), torch.int8, self.device, 16, "packed W"
        )
        check_tensor(scale_w, (self.n, 1), torch.float32, self.device, 4, "scale_w")
        check_device(self.device)
        self.weight = weight_packed
        self.scale_w = scale_w
        self._routes = routes("int8", self.n)
        catalog = build_catalog("int8")
        self._kernels = {}
        for name in set(self._routes[1:]):
            family, _, skew = name.partition("+")
            spec = catalog[family]
            bias = int(skew) if skew else 0
            if bias not in range(0, 257, 16):
                raise ValueError(f"Unsupported X pitch skew in {name}")
            self._kernels[name] = (load_assembly("int8", self.device, spec), spec, bias)

    def route(self, m):
        check_m(m)
        return self._routes[m]

    def workspace_elements(self, m):
        _, spec, skew = self._kernels[self.route(m)]
        bm = spec["bm"]
        return (triton.cdiv(m, bm) * bm + skew) * K

    @torch.inference_mode()
    def __call__(self, x, scale_x, *, out=None, workspace=None):
        m = x.shape[0] if x.ndim == 2 else 0
        check_tensor(x, (m, K), torch.int8, self.device, 1, "X")
        name = self.route(m)
        check_tensor(scale_x, (m, 1), torch.float32, self.device, 4, "scale_x")
        kernel, spec, skew = self._kernels[name]
        mt = triton.cdiv(m, spec["bm"])
        nt = self.n // spec["bn"]
        pitch = mt * spec["bm"] + skew
        grid = (mt, nt, 1) if spec["order"] == "m" else (nt, mt, 1)
        with torch.cuda.device(self.device):
            if out is None:
                out = x.new_empty((m, self.n), dtype=torch.bfloat16)
            check_tensor(out, (m, self.n), torch.bfloat16, self.device, 8, "Y")
            workspace = get_workspace(x, workspace, pitch * K)
            operands = (x, self.weight, scale_x, self.scale_w)
            addresses = {tensor.untyped_storage().data_ptr() for tensor in operands}
            out_addr, ws_addr = (
                out.untyped_storage().data_ptr(),
                workspace.untyped_storage().data_ptr(),
            )
            if out_addr == ws_addr or out_addr in addresses or ws_addr in addresses:
                raise ValueError(
                    "Output and workspace require separate storage from each other and the inputs"
                )
            if spec.get("x_packer") == "linear":
                _pack_x_linear[(triton.cdiv(pitch * K, 4096),)](
                    x, workspace, m, pitch, K, 4096, num_warps=4
                )
            elif self.n == 17408 and m >= 257:
                _pack_x_tile[(pitch // 16, K // 256)](x, workspace, m, pitch, num_warps=4)
            else:
                _pack_into(x, workspace, pitch)
            active = torch.cuda.current_stream(self.device)
            args = (
                PTR(workspace.data_ptr()),
                PTR(self.weight.data_ptr()),
                PTR(out.data_ptr()),
                INT(m),
                INT(self.n),
                INT(K),
                INT(pitch),
                PTR(scale_x.data_ptr()),
                PTR(self.scale_w.data_ptr()),
            )
            kernel.launch(grid, spec.get("waves", 4) * 64, args, active.cuda_stream)
            record_stream(active, *operands, out, workspace)
        return out

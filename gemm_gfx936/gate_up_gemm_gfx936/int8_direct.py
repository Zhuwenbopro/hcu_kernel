"""INT8 GateUp with ordinary contiguous X and the existing packed W."""

import torch
import triton

from .int8_direct_policy import MAX_M, N, check_m, direct_policy
from .policy import K
from .runtime import (
    INT,
    PTR,
    build_catalog,
    check_device,
    check_tensor,
    load_assembly,
    record_stream,
)


class DirectXInt8GateUpGemm:
    """Compute BF16 Y without a separate X conversion or X workspace.

    Supports M=1..128, N=17408 and K=5120. X is contiguous and 16-byte
    aligned. Weights use int8.pack_int8_weight as in Int8GateUpGemm.
    The independent policy selects the measured direct-X variant per M.
    """

    def __init__(self, weight_packed, scale_w):
        self.n = weight_packed.numel() // K
        self.device = weight_packed.device
        if self.n != N:
            raise ValueError(f"Direct-X kernels require N={N}, K={K}, M=1..{MAX_M}")
        check_tensor(
            weight_packed, (self.n * K,), torch.int8, self.device, 16, "packed W"
        )
        check_tensor(scale_w, (self.n, 1), torch.float32, self.device, 4, "scale_w")
        check_device(self.device)
        self.weight = weight_packed
        self.scale_w = scale_w
        self._routes, _ = direct_policy()
        kind = "int8_direct"
        catalog = build_catalog(kind)
        self._kernels = {}
        for route in set(self._routes[1:]):
            spec = catalog[route]
            self._kernels[route] = (
                load_assembly(kind, self.device, spec),
                spec,
            )

    def route(self, m):
        check_m(m)
        return self._routes[m]

    def workspace_elements(self, m):
        self.route(m)
        return 0

    @torch.inference_mode()
    def __call__(self, x, scale_x, *, out=None):
        m = x.shape[0] if x.ndim == 2 else 0
        name = self.route(m)
        check_tensor(x, (m, K), torch.int8, self.device, 16, "X")
        check_tensor(scale_x, (m, 1), torch.float32, self.device, 4, "scale_x")
        kernel, spec = self._kernels[name]
        mt, nt = triton.cdiv(m, spec["bm"]), self.n // spec["bn"]
        grid = (mt, nt, 1) if spec["order"] == "m" else (nt, mt, 1)
        with torch.cuda.device(self.device):
            if out is None:
                out = x.new_empty((m, self.n), dtype=torch.bfloat16)
            check_tensor(out, (m, self.n), torch.bfloat16, self.device, 8, "Y")
            operands = (x, self.weight, scale_x, self.scale_w)
            if out.untyped_storage().data_ptr() in {
                tensor.untyped_storage().data_ptr() for tensor in operands
            }:
                raise ValueError("Output requires separate storage from inputs")
            active = torch.cuda.current_stream(self.device)
            # Retain the packed kernel ABI. The pitch slot is unused by the
            # direct-X generated kernels; ordinary X always has stride K.
            args = (
                PTR(x.data_ptr()),
                PTR(self.weight.data_ptr()),
                PTR(out.data_ptr()),
                INT(m),
                INT(self.n),
                INT(K),
                INT(m),
                PTR(scale_x.data_ptr()),
                PTR(self.scale_w.data_ptr()),
            )
            kernel.launch(grid, spec["waves"] * 64, args, active.cuda_stream)
            record_stream(active, *operands, out)
        return out

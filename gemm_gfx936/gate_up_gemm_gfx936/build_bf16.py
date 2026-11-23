"""Build the rolling-MMAC, GEMV, and rocBLAS entry points for BF16 GateUp."""

import json
import re

from ..lm_head_bf16_gfx936.kernels import rolling_asm

from .build_tools import (
    BUILD,
    DTK,
    SOURCE,
    assemble,
    compile_hip,
    describe_file,
    write_catalog,
)
from .policy import ROOT, SHARDS, routes


def build():
    dest = BUILD / "bf16"
    dest.mkdir(parents=True, exist_ok=True)
    configs = json.loads((ROOT / "bf16-rolling.json").read_text(encoding="utf-8"))
    catalog = {"rolling": {}, "gemv": {}}
    for n in SHARDS:
        names = set(routes("bf16", n)[1:]) | set(routes("bf16", n, "exact")[1:])
        selected = sorted(
            {name.split(":")[1] for name in names if name.startswith("roll:")}
        )
        rolling_asm.N, rolling_asm.ROW_W = n, 2 * n
        for key in selected:
            q = rolling_asm.Config(**configs[key])
            if q.name != key or n % q.bn or rolling_asm.K % q.bk:
                raise ValueError(f"Invalid rolling kernel {n}/{key}")
            symbol = f"gate_kn_{n}_{key.replace('-', '_')}"
            path = dest / str(n) / f"{key}.s"
            path.parent.mkdir(exist_ok=True)
            text = re.sub(r"\brolling_mmac\b", symbol, rolling_asm.render(q))
            code = assemble(path, text, SOURCE)
            catalog["rolling"][f"{n}/{key}"] = {
                **describe_file(code, dest),
                "symbol": symbol,
                "config": configs[key],
                "lds": q.lds,
                "vgprs": q.vgprs,
            }
        gemv_routes = {name for name in names if name.startswith("gemv:")}
        if gemv_routes:
            if n != 34816 or gemv_routes != {"gemv:cached:9"}:
                raise ValueError(f"Unknown GEMV configuration for N={n}")
            binary = dest / "gemv-cached.so"
            compile_hip(SOURCE / "gate_up_gemv.hip", binary)
            catalog["gemv"][str(n)] = describe_file(binary, dest)
    blas = dest / "kn_blas.so"
    compile_hip(
        SOURCE / "kn_blas.cpp",
        blas,
        "-I",
        DTK / "include",
        "-L",
        DTK / "lib",
        "-lrocblas",
    )
    catalog["blas"] = describe_file(blas, dest)
    write_catalog(dest, catalog)
    print(
        f"Built {len(catalog['rolling'])} BF16 rolling kernels, GEMV, and rocBLAS wrapper"
    )


if __name__ == "__main__":
    build()

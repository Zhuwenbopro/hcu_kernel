"""Build the selected packed-W INT8 GateUp code objects."""

import json
import re

from .build_tools import BUILD, assemble, describe_file, write_catalog
from .kernels.int8_output import metadata, patch_generated
from .kernels.packed_asm import render_kernel
from .policy import K, MAX_M, ROOT, SHARDS, routes

INCLUDES = ROOT.parent.parent / "fused_gate_up_gemm_gfx936" / "sources"
FAMILY = re.compile(r"gu_int8_(\d+)_(n|m)(\d+)x(\d+)_(n|m)(?:_sp(\d+))?")
TUNED = json.loads((ROOT / "kernels" / "int8_tuned.json").read_text(encoding="utf-8"))


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected a single {old!r} in generated assembly")
    return text.replace(old, new, 1)


def retarget(text, spec, name, n, bn, order, *, max_m=MAX_M):
    """Set the shard strides and CTA traversal without changing the K loop."""
    text = replace_once(text, "s_mov_b32 s17, 0x110000", f"s_mov_b32 s17, {n * 64}")
    rows = spec["PF"] if spec["direction"] == "M" else spec["SF"]
    if text.count(", 0x11000,") != rows:
        raise ValueError(f"Unexpected epilogue row count in {name}")
    text = text.replace(", 0x11000,", f", {n * 4},")
    id_m, id_n = ("s7", "s6") if order == "n" else ("s6", "s7")
    text, count = re.subn(
        r"^(s_(?:lshl_b32|mul_i32) s(?:21|22), )s7,",
        r"\g<1>" + id_m + ",",
        text,
        flags=re.M,
    )
    if count != 2 or n % bn or n * K * 2 >= 2**31 or max_m * n * 4 >= 2**31:
        raise ValueError(f"Unsupported tile or output stride for {name}")
    text = replace_once(
        text,
        "s_lshl_b32 s23, s6, 9",
        f"s_lshl_b32 s23, {id_n}, {(bn * 4).bit_length() - 1}",
    )
    text = replace_once(
        text,
        "s_lshl_b32 s20, s6, 13",
        f"s_lshl_b32 s20, {id_n}, {(bn * 64).bit_length() - 1}",
    )
    return text


def build_tuned(name, n, dest):
    if n != 17408:
        raise ValueError(f"Tuned packed INT8 kernel {name} requires N=17408")
    record = TUNED[name]
    if "source" in record:
        # E047 retains its compiler-assigned registers and both entry points.
        text = (ROOT / "kernels" / record["source"]).read_text(encoding="utf-8")
        folder = dest / "tuned_e047"
    else:
        text, spec = render_kernel(**record["generate"])
        spec["direction"] = record["generate"]["direction"]
        if spec["family"] != record["symbol"]:
            raise ValueError(f"Unexpected generated symbol for {name}")
        if (spec["LDS"], spec["VGPR"]) != (record["lds"], record["vgprs"]):
            raise ValueError(f"Unexpected register or LDS allocation for {name}")
        text = metadata(patch_generated(text, spec, epilogue=record["epilogue"])[0])
        text = retarget(text, spec, name, n, record["bn"], record["order"])
        folder = dest / name
    folder.mkdir(exist_ok=True)
    code = assemble(folder / "kernel.s", text, INCLUDES)
    return {
        **describe_file(code, dest),
        **{
            key: record[key]
            for key in (
                "symbol", "bm", "bn", "order", "lds", "vgprs", "waves", "x_packer"
            )
        },
    }


def build_family(name, n, dest):
    if name in TUNED:
        return build_tuned(name, n, dest)
    match = FAMILY.fullmatch(name)
    if not match or int(match[1]) != n:
        raise ValueError(f"Invalid packed INT8 kernel {name} for N={n}")
    direction, bm, bn, order, prefetch = (
        match[2].upper(),
        int(match[3]),
        int(match[4]),
        match[5],
        int(match[6] or 0),
    )
    phase = "split_private_then_shared" if prefetch else "legacy"
    text, spec = render_kernel(direction, bm, prefetch, "both", phase, bn=bn)
    spec["direction"] = direction
    # Use ordinary global stores for TP2 M-major 256-column tiles.
    store_flags = "" if n == 17408 and bn == 256 and order == "m" else "glc slc"
    text = metadata(patch_generated(text, spec, store_flags)[0])
    text = re.sub(r"\b" + re.escape(spec["family"]) + r"\b", name, text)
    text = retarget(text, spec, name, n, bn, order)

    folder = dest / name
    folder.mkdir(exist_ok=True)
    code = assemble(folder / "kernel.s", text, INCLUDES)
    return {
        **describe_file(code, dest),
        "symbol": name,
        "bm": bm,
        "bn": bn,
        "order": order,
        "lds": spec["LDS"],
        "vgprs": spec["VGPR"],
    }


def build():
    dest = BUILD / "int8"
    dest.mkdir(parents=True, exist_ok=True)
    catalog = {}
    for n in SHARDS:
        families = {route.partition("+")[0] for route in routes("int8", n)[1:]}
        for name in sorted(families):
            catalog[name] = build_family(name, n, dest)
    write_catalog(dest, catalog)
    print(f"Built {len(catalog)} packed INT8 kernels")


if __name__ == "__main__":
    build()

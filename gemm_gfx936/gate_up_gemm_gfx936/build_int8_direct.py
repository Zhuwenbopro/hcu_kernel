"""Build the selected direct-X kernels without packed-X code objects."""

import re

from .build_int8 import INCLUDES, retarget
from .build_tools import BUILD, assemble, describe_file, write_catalog
from .int8_direct_policy import MAX_M, N, direct_policy
from .kernels.direct_x import (
    coalesced_x_source,
    direct_x_source,
    register_x_copy_source,
)
from .kernels.int8_output import metadata, patch_generated
from .kernels.lds_window import apply_lds_window
from .kernels.packed_asm import render_kernel


def build_family(name, record, dest):
    text, layout = render_kernel(**record["generate"])
    layout["direction"] = record["generate"]["direction"]
    text = metadata(
        patch_generated(text, layout, record["store_flags"], record["epilogue"])[0]
    )
    text = retarget(text, layout, name, N, layout["BN"], record["order"])
    if record["mapping"] == "coalesced":
        text, audit = coalesced_x_source(
            text,
            direction=layout["direction"],
            vgprs=layout["VGPR"],
            waves=layout["NW"],
            reuse_lds_address=record["reuse_lds_address"],
            lds_xor_shift=record["lds_xor_shift"],
        )
    elif record["mapping"] == "gather":
        text, audit = direct_x_source(
            text, direction=layout["direction"], vgprs=layout["VGPR"], waves=layout["NW"]
        )
    else:
        raise ValueError(f"Unknown direct-X mapping for {name}")
    vgprs = layout["VGPR"] + audit.get("extra_vgprs", 0)
    if record["x_copy"] == "reg_lds":
        text, copy_audit = register_x_copy_source(
            text,
            vgprs=vgprs,
            address_register=layout["address_registers"][0],
            shared_fragments=layout["SF"],
            waves=layout["NW"],
            flags=record["x_load_flags"],
        )
        vgprs += copy_audit["extra_vgprs"]
        audit["register_copy"] = copy_audit
    elif record["x_copy"] != "direct_lds":
        raise ValueError(f"Unknown direct-X copy path for {name}")
    if window := record.get("lds_window"):
        text, saved = apply_lds_window(text, layout, window)
        vgprs -= saved
        audit.update(lds_window=window, saved_lds_vgprs=saved,
                     same_mmac_lds_wait_barrier_sequence=False)
    if (vgprs, layout["LDS"]) != (record["vgprs"], record["lds"]):
        raise ValueError(f"Unexpected register or LDS allocation for {name}")
    audit["extra_vgprs"] = vgprs - layout["VGPR"]
    audit["same_vgprs_and_lds"] = vgprs == layout["VGPR"]
    audit["X_copy"] = record["x_copy"]
    symbol = record["symbol"]
    text = re.sub(r"\b" + re.escape(layout["family"]) + r"\b", symbol, text)
    phase = record.get("loop_phase_bytes", 0)
    if phase:
        entry = ".p2align 5\n.L_pair:"
        if phase not in range(4, 32, 4) or text.count(entry) != 1:
            raise ValueError(f"Invalid K-loop phase or entry for {name}")
        # Only the initial fall-through executes these NOPs. The loop branch
        # targets .L_pair after the padding on every subsequent iteration.
        text = text.replace(entry, ".p2align 5\n" + "s_nop 0\n" * (phase // 4) + ".L_pair:")
        audit["loop_phase_bytes"] = phase
    folder = dest / name
    folder.mkdir(exist_ok=True)
    code = assemble(folder / "kernel.s", text, INCLUDES)
    return {
        **describe_file(code, dest),
        "symbol": symbol,
        "bm": layout["BM"],
        "bn": layout["BN"],
        "waves": layout["NW"],
        "order": record["order"],
        "lds": layout["LDS"],
        "vgprs": vgprs,
        "input_layout": "contiguous_row_major",
        "audit": audit,
    }


def build():
    _, families = direct_policy()
    dest = BUILD / "int8_direct"
    dest.mkdir(parents=True, exist_ok=True)
    catalog = {name: build_family(name, rec, dest) for name, rec in sorted(families.items())}
    write_catalog(dest, catalog)
    print(f"Built {len(catalog)} direct-X INT8 kernels for M=1..{MAX_M}")


if __name__ == "__main__":
    build()

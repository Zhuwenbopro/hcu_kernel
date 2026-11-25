"""Build the selected direct-X kernels without packed-X code objects."""

import re

from .build_int8 import INCLUDES, replace_once, retarget
from .build_tools import BUILD, assemble, describe_file, write_catalog
from .int8_direct_policy import MAX_M, N, direct_policy, grouped_policy
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
    text = retarget(text, layout, name, N, layout["BN"], record["order"], max_m=MAX_M)
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


def build_grouped_family(dest, base):
    """Add a guarded M-band grid to the measured M-first 128x512 kernel."""
    if (base["order"], base["bm"], base["bn"]) != ("m", 128, 512):
        raise ValueError("The direct-X grouped prologue requires M-first BM128 x BN512")
    text = (dest / base["file"]).with_suffix(".s").read_text()
    text = replace_once(
        text, ".amdhsa_system_sgpr_workgroup_id_z 0", ".amdhsa_system_sgpr_workgroup_id_z 1"
    )
    # Z arrives in s8. Preserve it before the existing kernarg load clobbers s8.
    first_load = "s_load_dwordx4 s[8:11], s[4:5], 0"
    text = replace_once(text, first_load, "s_mov_b32 s26, s8\n" + first_load)
    # After lgkmcnt(0), s15 holds the band width and s14 the input row count.
    # Exit unused CTAs in the last band before any data access or workgroup barrier.
    grouped_prologue = """s_mul_i32 s26, s26, s15
s_add_u32 s6, s6, s26
s_lshl_b32 s26, s6, 7
s_cmp_lt_u32 s26, s14
s_cbranch_scc1 .L_valid_m_block
s_endpgm
.L_valid_m_block:
s_mov_b32 s16, 64"""
    text = replace_once(text, "s_mov_b32 s16, 64", grouped_prologue)
    folder = dest / "grouped"
    folder.mkdir(exist_ok=True)
    code = assemble(folder / "group.s", text, INCLUDES)
    return {**base, **describe_file(code, dest), "order": "grouped_m"}


def build():
    _, families = direct_policy()
    dest = BUILD / "int8_direct"
    dest.mkdir(parents=True, exist_ok=True)
    catalog = {name: build_family(name, rec, dest) for name, rec in sorted(families.items())}
    family, _ = grouped_policy()
    catalog[family + "_grouped"] = build_grouped_family(dest, catalog[family])
    write_catalog(dest, catalog)
    print(f"Built {len(catalog)} direct-X INT8 kernels for M=1..{MAX_M}")


if __name__ == "__main__":
    build()

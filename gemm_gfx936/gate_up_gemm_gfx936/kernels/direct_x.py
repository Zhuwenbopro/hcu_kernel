"""Adapt the packed kernels to read ordinary contiguous INT8 X.

The gather path retains the original fragment mapping. The coalesced path
changes X load and LDS read addresses together. MMAC order and the output
epilogue are preserved. W retains the existing packed layout.
"""

import re
import yaml


def _once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one occurrence of {old!r}")
    return source.replace(old, new, 1)


def _decode_fragment(offset, first, second):
    # packed fragment: [M/16, K64/16, M%16, K%16].
    # All source addresses are 16-byte aligned; the final K%16 is zero.
    return "\n".join(
        [
            f"v_lshrrev_b32_e32 v{first}, 10, v{offset}",
            f"v_lshlrev_b32_e32 v{first}, 4, v{first}",
            f"v_bfe_u32 v{second}, v{offset}, 4, 4",
            f"v_add_u32_e32 v{first}, v{first}, v{second}",
            f"v_mul_u32_u24_e32 v{first}, 5120, v{first}",
            f"v_bfe_u32 v{second}, v{offset}, 8, 2",
            f"v_lshlrev_b32_e32 v{second}, 4, v{second}",
            f"v_add_u32_e32 v{offset}, v{first}, v{second}",
            "",
        ]
    )


def _protected(source):
    prefixes = (
        "GEMM_MMA_I8 ",
        "GEMM_LDS_TO_REG ",
        "GEMM_WAIT_VM ",
        "GEMM_WAIT_LGKM ",
        "GEMM_CTA_BARRIER",
    )
    return [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith(prefixes)
    ]


def _masked_load(instruction, address, scalar_offset, limit, scratch):
    # Packed kernels may read padded rows. Ordinary X has no such allocation.
    # Explicit EXEC masking also avoids relying on the SRD stride-dependent
    # interpretation of NUM_RECORDS for direct-to-LDS instructions.
    out = []
    effective = address
    if scalar_offset != "0":
        out.append(f"v_add_u32_e32 v{scratch}, {scalar_offset}, {address}")
        effective = f"v{scratch}"
    out.extend(
        [
            f"v_cmp_gt_u32_e32 vcc, {limit}, {effective}",
            "s_and_saveexec_b64 s[30:31], vcc",
            instruction,
            "s_or_b64 exec, exec, s[30:31]",
        ]
    )
    return "\n".join(out)


def direct_x_source(source, *, direction, vgprs, waves=4, masked=False):
    """Return row-major-X assembly and a structural preservation audit.

    Supported inputs are the scalar-address generated kernels, K=5120 and
    waves=4/8. Raw buffer bounds exclude M padding, so callers do not need an X
    workspace or readable storage beyond their actual contiguous [M,5120].
    """
    if direction not in ("M", "N") or waves not in (4, 8):
        raise ValueError("Unsupported wave layout")
    original = source
    t = vgprs - 12
    source = _once(
        source, "s_lshl_b32 s16, s15, 6", "s_mov_b32 s16, 64"
    )
    if direction == "N":
        # The existing LDS copy still scatters each lane to its old LDS slot.
        # Change its global address to row*K + K16-quarter instead.
        line = f"v_add_u32_e32 v{t + 3}, s20, v{t + 3}\n"
        source = _once(
            source, line, line + _decode_fragment(t + 3, t + 5, t + 6)
        )
        source = _once(
            source, "s_mov_b32 s2, 0x80000000", "s_mul_i32 s2, s14, 5120"
        )
        if not masked:
            source = _once(
                source,
                "s_add_u32 s1, s9, 0x40800000",
                "s_mov_b32 s1, s9",
            )
        for q in (1, 2, 3):
            source = _once(
                source,
                f"s_mov_b32 s{26 + q}, {q * waves * 1024}",
                f"s_mov_b32 s{26 + q}, {q * waves * 16 * 5120}",
            )
        if masked:
            source = re.sub(
                r"GEMM_X_GLOBAL_TO_LDS (v\d+), s\[0:3\], (0|s\d+)",
                lambda m: _masked_load(m[0], m[1], m[2], "s2", t + 10),
                source,
            )
        changed_loads = source.count("GEMM_X_GLOBAL_TO_LDS ")
    else:
        # s22 already holds the logical CTA M origin. The old packed stride
        # BM*64 becomes the ordinary row stride BM*K.
        source, count = re.subn(
            r"^s_(?:lshl_b32|mul_i32) s21, [^\n]+$",
            "s_mul_i32 s21, s22, 5120",
            source,
            flags=re.M,
        )
        if count != 1:
            raise ValueError("Missing X CTA offset")
        line = f"v_add_u32_e32 v{t + 2}, v{t + 2}, v{t}\n"
        source = _once(
            source, line, line + _decode_fragment(t + 2, t + 5, t + 6)
        )
        # Shared W already owns its descriptor in s0:s3. Its original pointer
        # in s10:s11 is now dead, allowing s8:s11 to become a bounded X SRD.
        # s15/s20/s21 are likewise dead after coordinate initialization.
        line = f"v_mov_b32_e32 v{t + 11}, s9\n"
        setup = (
            "s_sub_u32 s10, s14, s22\n"
            "s_mul_i32 s10, s10, 5120\n"
            "s_mov_b32 s11, 0x20000\n"
            "s_mov_b32 s15, 81920\n"
            "s_mov_b32 s20, 163840\n"
            "s_mov_b32 s21, 245760\n"
        )
        if masked:
            setup = "s_add_u32 s9, s9, 0x40800000\n" + setup
        source = _once(source, line, line + setup)
        pattern = (
            r"GEMM_X_GLOBAL_SADDR_TO_REG (v\[\d+:\d+\]), "
            + re.escape(f"v{t + 2}")
            + r", s\[8:9\], (\d+)"
        )

        def replace_load(match):
            fragment = int(match[2]) // 1024
            if int(match[2]) != fragment * 1024 or fragment not in range(4):
                raise ValueError("Unsupported private X fragment")
            offset = ("0", "s15", "s20", "s21")[fragment]
            instruction = (
                f"buffer_load_dwordx4 {match[1]}, v{t + 2}, "
                f"s[8:11], {offset} offen"
            )
            if masked:
                return _masked_load(instruction, f"v{t + 2}", offset, "s10", t + 10)
            return instruction

        source, changed_loads = re.subn(pattern, replace_load, source)
        head, separator, tail = source.partition(setup)
        tail, advances = re.subn(
            r"(s_addc_u32 s9, s9, 0\n)",
            r"\1s_sub_u32 s10, s10, 64\n",
            tail,
        )
        source = head + separator + tail
        if advances != 3 or changed_loads == 0:
            raise ValueError("Unexpected private-X pipeline")
    if _protected(source) != _protected(original):
        raise ValueError("MMAC, LDS reads, waits or barriers changed")
    epilogue = "s_load_dwordx4 s[0:3], s[4:5], 0x28"
    if original.partition(epilogue)[2] != source.partition(epilogue)[2]:
        raise ValueError("Output epilogue changed")
    return source, {
        "direction": direction,
        "waves": waves,
        "same_mmac_lds_wait_barrier_sequence": True,
        "same_epilogue": True,
        "same_vgprs_and_lds": True,
        "X_load_sites": changed_loads,
        "input_layout": "contiguous_row_major",
        "K": 5120,
        "bounds": "explicit_exec" if masked else "stride_zero_raw_srd_bytes",
    }


def coalesced_x_source(
    source, *, direction, vgprs, waves=4, masked=False,
    reuse_lds_address=False, lds_xor_shift=1,
):
    """Coalesce ordinary X loads and retain a bank-distributed LDS read.

    A group of four lanes loads all four K16 quarters of one row. The quarter
    is XORed with two row bits before landing in LDS. The inverse read mapping
    restores the original MMAC operands. The LDS-read instruction sequence and
    scheduling stay intact. A dead address register can hold the LDS mapping
    until the epilogue instead of increasing the VGPR declaration.
    """
    original = source
    source, audit = direct_x_source(
        source, direction=direction, vgprs=vgprs, waves=waves, masked=masked
    )
    if direction == "M":
        return source, audit
    if lds_xor_shift not in (1, 2):
        raise ValueError("LDS XOR shift must be 1 or 2 for a 16-row fragment")
    t = vgprs - 12
    lds = t + 7 if reuse_lds_address else vgprs
    if reuse_lds_address:
        before = source.partition("s_load_dwordx4 s[0:3], s[4:5], 0x28")[0]
        if re.search(r"\bv" + str(lds) + r"\b", before):
            raise ValueError("The selected LDS address register is still live")
    z, row, quarter, temp = t + 3, t + 5, t + 6, t + 10
    decode = "\n".join(
        [
            f"v_lshrrev_b32_e32 v{row}, 6, v{z}",
            f"v_lshrrev_b32_e32 v{quarter}, {lds_xor_shift}, v{row}",
            f"v_lshrrev_b32_e32 v{temp}, 4, v{z}",
            f"v_xor_b32_e32 v{quarter}, v{quarter}, v{temp}",
            f"v_and_b32_e32 v{quarter}, 3, v{quarter}",
            f"v_lshlrev_b32_e32 v{quarter}, 4, v{quarter}",
            f"v_mul_u32_u24_e32 v{row}, 5120, v{row}",
            f"v_add_u32_e32 v{z}, v{row}, v{quarter}",
            f"v_lshrrev_b32_e32 v{lds}, {lds_xor_shift}, v0",
            f"v_lshrrev_b32_e32 v{temp}, 4, v0",
            f"v_xor_b32_e32 v{lds}, v{lds}, v{temp}",
            f"v_and_b32_e32 v{lds}, 3, v{lds}",
            f"v_lshlrev_b32_e32 v{lds}, 4, v{lds}",
            f"v_and_b32_e32 v{temp}, 15, v0",
            f"v_lshlrev_b32_e32 v{temp}, 6, v{temp}",
            f"v_or_b32_e32 v{lds}, v{lds}, v{temp}",
            "",
        ]
    )
    source = _once(source, _decode_fragment(z, row, quarter), decode)
    source, count = re.subn(
        rf"(v_add_u32_e32 v{t + 9}, \d+, )v{t}\b",
        rf"\g<1>v{lds}",
        source,
    )
    if count != 2:
        raise ValueError("Missing the two LDS slot addresses")
    declared = vgprs if reuse_lds_address else vgprs + 1
    source = _once(
        source,
        f".amdhsa_next_free_vgpr {vgprs}",
        f".amdhsa_next_free_vgpr {declared}",
    )
    before, doc = source.split(".amdgpu_metadata", 1)
    doc, after = doc.split(".end_amdgpu_metadata", 1)
    meta = yaml.safe_load(doc.expandtabs(8))
    for kernel in meta["amdhsa.kernels"]:
        kernel[".vgpr_count"] = declared
    source = (
        before + ".amdgpu_metadata\n---\n"
        + yaml.safe_dump(meta, sort_keys=False)
        + "...\n.end_amdgpu_metadata" + after
    )
    assert _protected(source) == _protected(original)
    audit.update(
        coalesced_global_X=True,
        LDS_quarter_xor=f"(row >> {lds_xor_shift}) & 3",
        extra_vgprs=declared - vgprs,
        same_vgprs_and_lds=reuse_lds_address,
        same_lds=True,
    )
    return source, audit


def register_x_copy_source(
    source, *, vgprs, address_register, shared_fragments, waves=4, flags=""
):
    """Stage X through registers and write LDS at the next existing ready point.

    The global addresses, double-buffered LDS layout and MMAC order stay
    unchanged. Extra registers retain the prefetched X fragments until vmcnt=0.
    Saved lane/wave offsets form the LDS write address: v0 is an accumulator
    after initialization and cannot be used as the workitem ID in the K loop.
    """
    if waves not in (4, 8) or not 1 <= shared_fragments <= 16:
        raise ValueError("Unsupported register-to-LDS copy geometry")
    if flags not in ("", "glc"):
        raise ValueError("Unsupported X load flags")
    original = source
    rounds = (shared_fragments + waves - 1) // waves
    extra = 4 * rounds
    if vgprs + extra > 256:
        raise ValueError("Register-to-LDS copy exceeds the VGPR limit")
    t = address_register
    sites = [0]

    def load(match):
        q = sites[0] % rounds
        sites[0] += 1
        instruction = (
            f"buffer_load_dwordx4 v[{vgprs + 4 * q}:{vgprs + 4 * q + 3}], "
            f"{match[1]}, s[0:3], {match[2]} offen"
        )
        return instruction + (" " + flags if flags else "")

    source, count = re.subn(
        r"GEMM_X_GLOBAL_TO_LDS (v\d+), s\[0:3\], (0|s\d+)", load, source
    )
    if count != 3 * rounds:
        raise ValueError("Unexpected X prefetch pipeline")
    slots = [0]

    def store(match):
        slot = slots[0]
        slots[0] += 1
        lines = ["GEMM_WAIT_VM 0", f"v_add_u32_e32 v{t + 10}, v{t}, v{t + 4}"]
        for q in range(rounds):
            active = min(waves, shared_fragments - q * waves)
            label = f".L_reglds_{slot}_{q}"
            if active < waves:
                lines.extend([f"s_cmp_lt_u32 s25, {active}", f"s_cbranch_scc0 {label}"])
            offset = slot * shared_fragments * 1024 + q * waves * 1024
            lines.extend([
                f"v_add_u32_e32 v{t + 9}, {offset}, v{t + 10}",
                f"ds_write_b128 v{t + 9}, v[{vgprs + 4 * q}:{vgprs + 4 * q + 3}]",
            ])
            if active < waves:
                lines.append(label + ":")
        return "\n".join(lines + ["GEMM_WAIT_LGKM 0", "GEMM_CTA_BARRIER"])

    source, count = re.subn(
        r"GEMM_WAIT_VM 0\nGEMM_WAIT_LGKM 0\nGEMM_CTA_BARRIER", store, source
    )
    if count != 2 or _protected(source) != _protected(original):
        raise ValueError("MMAC, LDS reads, waits or barriers changed")
    source = _once(
        source, f".amdhsa_next_free_vgpr {vgprs}", f".amdhsa_next_free_vgpr {vgprs + extra}"
    )
    before, doc = source.split(".amdgpu_metadata", 1)
    doc, after = doc.split(".end_amdgpu_metadata", 1)
    meta = yaml.safe_load(doc.expandtabs(8))
    for kernel in meta["amdhsa.kernels"]:
        kernel[".vgpr_count"] = vgprs + extra
    source = (
        before + ".amdgpu_metadata\n---\n" + yaml.safe_dump(meta, sort_keys=False)
        + "...\n.end_amdgpu_metadata" + after
    )
    return source, {
        "X_copy": "buffer_reg_then_lds",
        "extra_vgprs": extra,
        "same_mmac_lds_wait_barrier_sequence": True,
        "same_lds": True,
        "X_load_flags": flags,
    }

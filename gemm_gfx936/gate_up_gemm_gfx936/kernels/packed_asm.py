"""Generate physical-register INT8 MMAC kernels and their launch metadata."""

from pathlib import Path
import re

METADATA = Path(__file__).with_name("int8_metadata.yaml").read_text()


def scale_u32(dst, src, factor):
    """Emit an exact constant multiply for the wave/CTA coordinates used here."""
    if type(factor) is not int or factor <= 0:
        raise ValueError("scale factor must be a positive integer")
    vector = dst.startswith("v")
    if not (
        vector and src.startswith("v") or dst.startswith("s") and src.startswith("s")
    ):
        raise ValueError("scale operands must be vector/vector or scalar/scalar")
    if factor & (factor - 1) == 0:
        shift = factor.bit_length() - 1
        return (
            f"v_lshlrev_b32_e32 {dst}, {shift}, {src}"
            if vector
            else f"s_lshl_b32 {dst}, {src}, {shift}"
        )
    # Vector callers supply only wave_id in0..7; both operands fit unsigned24bit.
    if vector:
        if factor >= 1 << 24:
            raise ValueError("vector wave-coordinate scale must fit unsigned24bit")
        return f"v_mul_u32_u24_e32 {dst}, {factor}, {src}"
    return f"s_mul_i32 {dst}, {src}, {factor}"


def shared_copy_rounds(sf, waves=4):
    """Each active wave copies one 1KiB fragment per round."""
    return [(q, min(waves, sf - waves * q)) for q in range((sf + waves - 1) // waves)]


ADDRESS_MODES = ("none", "private", "shared", "both")
PREFETCH_ORDERS = ("legacy", "private_first", "split_private_then_shared")


def kernel_layout(
    direction,
    bm,
    read_first=0,
    scalar_address="none",
    prefetch_order="legacy",
    bn=128,
    waves=4,
    mmac_group=1,
):
    if direction not in ("M", "N"):
        raise ValueError("direction must be M or N")
    if type(bm) is not int or bm <= 0:
        raise ValueError("BM must be a positive integer")
    if type(waves) is not int or waves not in (4, 8):
        raise ValueError("waves must be 4 or 8")
    align = 16 * waves if direction == "M" else 16
    if bm % align:
        raise ValueError(
            f"{direction}-split BM must be a multiple of {align}: waves must own whole 16-row MMAC fragments"
        )
    supported_bn = (64, 128, 256, 512) if direction == "N" and waves == 8 else (64, 128, 256)
    if bn not in supported_bn or bn % (16 * waves):
        raise ValueError("Unsupported BN")
    pf = bm // (16 * waves) if direction == "M" else bn // (16 * waves)
    sf = bn // 16 if direction == "M" else bm // 16
    if type(read_first) is not int or not 0 <= read_first <= sf:
        raise ValueError(f"read_first must be an integer in0..{sf}")
    if sf > 16:
        raise ValueError(
            "current full LDS-read window supports at most16 fragments; change the read-window schedule before increasing this bound"
        )
    if type(mmac_group) is not int or not 1 <= mmac_group <= sf:
        raise ValueError(f"mmac_group must be an integer in 1..{sf}")
    if scalar_address not in ADDRESS_MODES:
        raise ValueError(f"scalar_address must be one of {ADDRESS_MODES}")
    if prefetch_order not in PREFETCH_ORDERS:
        raise ValueError(f"prefetch_order must be one of {PREFETCH_ORDERS}")
    if prefetch_order == "private_first" and read_first:
        raise ValueError(
            "private_first has a full LDS read batch; specify read_first=0"
        )
    if scalar_address in ("private", "both") and pf > 4:
        raise ValueError("scalar private SADDR supports at most4 fragments per wave")
    ac = 4 * pf * sf
    p0 = ac
    p1 = p0 + 4 * pf
    ds = p1 + 4 * pf
    t = ds + 4 * sf
    vgpr = t + 12
    lds = 2 * sf * 1024
    if vgpr > 256:
        raise ValueError(
            f"VGPR budget{vgpr} exceeds256; reduce the operand/read window or tile"
        )
    if lds > 65536:
        raise ValueError(f"LDS budget{lds} exceeds64KiB")
    by_wave = [len(range(wave, sf, waves)) for wave in range(waves)]
    fam = f"{direction}{bm}" + (
        f"_SP{read_first}"
        if prefetch_order == "split_private_then_shared"
        else f"_RF{read_first}"
        if read_first
        else ""
    )
    if prefetch_order == "private_first":
        fam += "_PF"
    if scalar_address != "none":
        fam += "_SA" + {"private": "P", "shared": "S", "both": "B"}[scalar_address]
    if mmac_group != 1:
        fam += f"_G{mmac_group}"
    if waves != 4:
        fam += f"_BN{bn}_W{waves}"
    return dict(
        family=fam,
        mmac_group=mmac_group,
        read_first_fragments=read_first,
        scalar_address=scalar_address,
        prefetch_order=prefetch_order,
        current_reads_before_private=0
        if prefetch_order == "private_first"
        else read_first,
        current_reads_before_shared=sf
        if prefetch_order in ("private_first", "split_private_then_shared")
        else read_first,
        private_operand="X" if direction == "M" else "W",
        shared_operand="W" if direction == "M" else "X",
        BM=bm,
        BN=bn,
        NW=waves,
        PF=pf,
        SF=sf,
        VGPR=vgpr,
        SGPR=36,
        LDS=lds,
        scratch=0,
        accumulator_VGPR=ac,
        private_slot0=[p0, p1 - 1],
        private_slot1=[p1, ds - 1],
        shared_fragments=[ds, t - 1],
        address_registers=[t, t + 11],
        MMAC_per_wave_K64=2 * pf * sf,
        DS_read_per_wave_K64=sf,
        direct_load_per_wave_K64=pf,
        LDS_global_load_per_wave_K64=by_wave[0] if len(set(by_wave)) == 1 else by_wave,
        LDS_global_loads_by_wave_K64=by_wave,
        LDS_copy_rounds=shared_copy_rounds(sf, waves),
    )


def _advance_source(cfg):
    t = cfg["address_registers"][0]
    if cfg["scalar_address"] in ("private", "both"):
        lo, hi = (8, 9) if cfg["family"].startswith("M") else (10, 11)
        return f"s_add_u32 s{lo}, s{lo}, s18\ns_addc_u32 s{hi}, s{hi}, 0\nv_add_u32_e32 v{t + 3}, s19, v{t + 3}\n"
    return f"v_add_u32_e32 v{t + 2}, s18, v{t + 2}\nv_add_u32_e32 v{t + 3}, s19, v{t + 3}\n"


def _private_first_source(src, cfg):
    """P(next), D(current), S(next), compute; last D is unconditional."""
    sf = cfg["SF"]
    t = cfg["address_registers"][0]
    pitch = sf * 1024
    advance = _advance_source(cfg)
    pat = (
        r"; issue private and shared stage to slot([01])\n(.*?)("
        + re.escape(advance)
        + ")"
    )
    stages = list(re.finditer(pat, src, re.S))
    assert len(stages) == 3
    packets = []
    for m in stages:
        body = m[2]
        loads = list(re.finditer(r"GEMM_[XW]_GLOBAL_(?:SADDR_)?TO_REG [^\n]+\n", body))
        assert len(loads) == cfg["PF"]
        cut = loads[-1].end()
        packets.append({"P": body[:cut], "S": body[cut:], "advance": m[3]})
    pat = rf"v_add_u32_e32 v{t + 9}, (0|{pitch}), v{t}\n(?:(?:GEMM_LDS_TO_REG|GEMM_WAIT_LGKM|GEMM_MMA_I8)[^\n]*\n)+"
    cons = list(re.finditer(pat, src))
    assert len(cons) == 2
    reads = []
    maths = []
    for m in cons:
        ls = m[0].splitlines(keepends=True)
        assert all("GEMM_LDS_TO_REG" in x for x in ls[1 : sf + 1])
        reads.append("".join(ls[: sf + 1]))
        maths.append("".join(ls[sf + 1 :]))
    ready = "GEMM_WAIT_VM 0\nGEMM_WAIT_LGKM 0\nGEMM_CTA_BARRIER\n"
    out = [
        ready,
        packets[1]["P"],
        reads[0],
        packets[1]["S"],
        packets[1]["advance"],
        maths[0],
        ready,
        "s_sub_u32 s24, s24, 1\n",
    ]
    for token in ["P", "D", "S"]:
        if token == "D":
            out.append(reads[1])
            continue
        label = ".L_last_skip_" + token
        out.extend(
            [
                f"s_cmp_eq_u32 s24, 0\ns_cbranch_scc1 {label}\n",
                packets[2][token],
                label + ":\n",
            ]
        )
    out.extend(
        [
            packets[2]["advance"],
            maths[1],
            "s_cmp_lg_u32 s24, 0\ns_cbranch_scc1 .L_pair\nGEMM_WAIT_VM 0",
        ]
    )
    a = src.index(".L_pair:\n") + len(".L_pair:\n")
    marker = "s_cmp_lg_u32 s24, 0\ns_cbranch_scc1 .L_pair\nGEMM_WAIT_VM 0"
    b = src.index(marker, a) + len(marker)
    return src[:a] + "".join(out) + src[b:]


def _split_prefetch_source(src, cfg):
    """D_prefix(current), P(next), D_rest(current), S(next), compute.

    Both current-slot read groups remain unconditional in the final K64 pair.
    P and S use the existing, disjoint private/address/shared register windows.
    """
    sf = cfg["SF"]
    k = cfg["read_first_fragments"]
    t = cfg["address_registers"][0]
    advance = _advance_source(cfg)
    stages = list(
        re.finditer(
            r"; issue private and shared stage to slot([01])\n(.*?)("
            + re.escape(advance)
            + ")",
            src,
            re.S,
        )
    )
    assert len(stages) == 3
    packets = []
    for m in stages:
        body = m[2]
        loads = list(re.finditer(r"GEMM_[XW]_GLOBAL_(?:SADDR_)?TO_REG [^\n]+\n", body))
        assert len(loads) == cfg["PF"]
        cut = loads[-1].end()
        packets.append(dict(P=body[:cut], S=body[cut:], advance=m[3]))
    pattern = rf"v_add_u32_e32 v{t + 9}, (0|{sf * 1024}), v{t}\n(?:(?:GEMM_LDS_TO_REG|GEMM_WAIT_LGKM|GEMM_MMA_I8)[^\n]*\n)+"
    consumers = list(re.finditer(pattern, src))
    assert len(consumers) == 2
    prefix = []
    rest = []
    math = []
    for m in consumers:
        ls = m[0].splitlines(keepends=True)
        assert all("GEMM_LDS_TO_REG" in z for z in ls[1 : sf + 1])
        prefix.append("".join(ls[: k + 1]))
        rest.append("".join(ls[k + 1 : sf + 1]))
        math.append("".join(ls[sf + 1 :]))
    ready = "GEMM_WAIT_VM 0\nGEMM_WAIT_LGKM 0\nGEMM_CTA_BARRIER\n"
    out = [
        ready,
        prefix[0],
        packets[1]["P"],
        rest[0],
        packets[1]["S"],
        packets[1]["advance"],
        math[0],
        ready,
        "s_sub_u32 s24, s24, 1\n",
        prefix[1],
        "s_cmp_eq_u32 s24, 0\ns_cbranch_scc1 .L_custom_skip_P\n",
        packets[2]["P"],
        ".L_custom_skip_P:\n",
        rest[1],
        "s_cmp_eq_u32 s24, 0\ns_cbranch_scc1 .L_custom_skip_S\n",
        packets[2]["S"],
        packets[2]["advance"],
        ".L_custom_skip_S:\n",
        math[1],
        "s_cmp_lg_u32 s24, 0\ns_cbranch_scc1 .L_pair\nGEMM_WAIT_VM 0",
    ]
    a = src.index(".L_pair:\n") + len(".L_pair:\n")
    marker = "s_cmp_lg_u32 s24, 0\ns_cbranch_scc1 .L_pair\nGEMM_WAIT_VM 0"
    b = src.index(marker, a) + len(marker)
    return src[:a] + "".join(out) + src[b:]


def render_kernel(
    direction,
    bm,
    read_first=0,
    scalar_address="none",
    prefetch_order="legacy",
    bn=128,
    waves=4,
    mmac_group=1,
):
    cfg = kernel_layout(
        direction, bm, read_first, scalar_address, prefetch_order,
        bn=bn, waves=waves, mmac_group=mmac_group,
    )
    if prefetch_order == "split_private_then_shared":
        source, parent = render_kernel(
            direction, bm, 0, scalar_address, "legacy",
            bn=bn, waves=waves, mmac_group=mmac_group,
        )
        source = re.sub(
            r"\b" + re.escape(parent["family"]) + r"\b", cfg["family"], source
        )
        return _split_prefetch_source(source, cfg), cfg
    sp = scalar_address in ("private", "both")
    ss = scalar_address in ("shared", "both")
    fam = cfg["family"]
    pf = cfg["PF"]
    sf = cfg["SF"]
    ac = cfg["accumulator_VGPR"]
    p0 = cfg["private_slot0"][0]
    p1 = cfg["private_slot1"][0]
    ds = cfg["shared_fragments"][0]
    t = cfg["address_registers"][0]
    vgpr = cfg["VGPR"]
    lds = cfg["LDS"]
    slotbytes = sf * 1024
    L = []

    def w(s):
        L.append(s)

    def vr(i):
        return f"v{i}"

    def v4(i):
        return f"v[{i}:{i + 3}]"

    def v2(i):
        return f"v[{i}:{i + 1}]"

    w(
        "; Physical-register gfx936 kernel, generated deterministically. No device compiler."
    )
    w(
        '.include "common_gfx936.inc"\n.text\n.amdgcn_target "amdgcn-amd-amdhsa--gfx936:sramecc+"'
    )
    w(
        f".section .text.{fam},#alloc,#execinstr\n.protected {fam}\n.globl {fam}\n.p2align 8\n.type {fam},@function\n{fam}:"
    )
    w(
        "s_load_dwordx4 s[8:11], s[4:5], 0\ns_load_dwordx2 s[12:13], s[4:5], 0x10\ns_load_dword s14, s[4:5], 0x18\ns_load_dword s15, s[4:5], 0x24"
    )
    w(
        f"v_and_b32_e32 {vr(t)}, 63, v0\nv_lshlrev_b32_e32 {vr(t)}, 4, {vr(t)}\nv_lshrrev_b32_e32 {vr(t + 1)}, 6, v0\nv_lshlrev_b32_e32 {vr(t + 3)}, 4, v0\nv_lshlrev_b32_e32 {vr(t + 4)}, 10, {vr(t + 1)}"
    )
    w("GEMM_WAIT_LGKM 0\ns_lshl_b32 s16, s15, 6\ns_mov_b32 s17, 0x110000")
    # packedX CTA offset(tileM*BM*64), W CTA offset(tileN*128*64)
    w(scale_u32("s22", "s7", bm))
    w("s_lshl_b32 s23, s6, 9")
    w("s_lshl_b32 s20, s6, 13")
    w(scale_u32("s21", "s7", bm * 64))
    if direction == "M":
        w(
            "s_mov_b32 s0, s10\ns_add_u32 s1, s11, 0x40800000\ns_mov_b32 s18, s16\ns_mov_b32 s19, s17"
        )
        plo, phi = 8, 9
    else:
        w(
            "s_mov_b32 s0, s8\ns_add_u32 s1, s9, 0x40800000\ns_mov_b32 s18, s17\ns_mov_b32 s19, s16"
        )
        # swap static CTA offsets; sharedX in s20, privateW in s21
        w("s_mov_b32 s26, s20\ns_mov_b32 s20, s21\ns_mov_b32 s21, s26")
        plo, phi = 10, 11
    w("s_mov_b32 s2, 0x80000000\ns_mov_b32 s3, 0x20000")
    w(scale_u32(vr(t + 2), vr(t + 1), pf * 1024))
    if sp:
        w(f"s_add_u32 s{plo}, s{plo}, s21\ns_addc_u32 s{phi}, s{phi}, 0")
    else:
        w(f"v_add_u32_e32 {vr(t + 2)}, s21, {vr(t + 2)}")
    w(
        f"v_add_u32_e32 {vr(t + 2)}, {vr(t + 2)}, {vr(t)}\nv_add_u32_e32 {vr(t + 3)}, s20, {vr(t + 3)}"
    )
    if sf % waves:
        w(f"v_readfirstlane_b32 s25, {vr(t + 1)}")
    issue_serial = [0]

    def issue(slot):
        serial = issue_serial[0]
        issue_serial[0] += 1
        pr = p0 if slot == 0 else p1
        w(f"; issue private and shared stage to slot{slot}")
        if sp:
            macro = (
                "GEMM_X_GLOBAL_SADDR_TO_REG"
                if direction == "M"
                else "GEMM_W_GLOBAL_SADDR_TO_REG"
            )
            for p in range(pf):
                w(f"{macro} {v4(pr + 4 * p)}, {vr(t + 2)}, s[{plo}:{phi}], {p * 1024}")
        else:
            w(
                f"v_add_co_u32_e32 {vr(t + 5)}, vcc, s{plo}, {vr(t + 2)}\nv_addc_co_u32_e32 {vr(t + 6)}, vcc, 0, {vr(t + 11)}, vcc"
            )
            for p in range(pf):
                ad = t + 5
                if p:
                    w(
                        f"v_add_co_u32_e32 {vr(t + 7)}, vcc, {p * 1024}, {vr(t + 5)}\nv_addc_co_u32_e32 {vr(t + 8)}, vcc, 0, {vr(t + 6)}, vcc"
                    )
                    ad = t + 7
                macro = (
                    "GEMM_X_GLOBAL_TO_REG"
                    if direction == "M"
                    else "GEMM_W_GLOBAL_TO_REG"
                )
                w(f"{macro} {v4(pr + 4 * p)}, {v2(ad)}")
        for q, active_waves in shared_copy_rounds(sf, waves):
            label = f".L_copy_skip_{serial}_{q}"
            if active_waves < waves:
                w(f"s_cmp_lt_u32 s25, {active_waves}\ns_cbranch_scc0 {label}")
            macro = (
                "GEMM_W_GLOBAL_TO_LDS" if direction == "M" else "GEMM_X_GLOBAL_TO_LDS"
            )
            if ss:
                dest = slot * slotbytes + q * waves * 1024
                w("s_mov_b32 m0, s26" if dest == 0 else f"s_add_u32 m0, s26, {dest}")
                off = "0" if q == 0 else f"s{26 + q}"
                w(f"{macro} {vr(t + 3)}, s[0:3], {off}")
            else:
                w(
                    f"v_add_u32_e32 {vr(t + 7)}, {slot * slotbytes + q * waves * 1024}, {vr(t + 4)}\nv_readfirstlane_b32 s26, {vr(t + 7)}\ns_mov_b32 m0, s26"
                )
                w(f"v_add_u32_e32 {vr(t + 10)}, {q * waves * 1024}, {vr(t + 3)}")
                w(f"{macro} {vr(t + 10)}, s[0:3], 0")
            if active_waves < waves:
                w(label + ":")
        w(_advance_source(cfg).rstrip("\n"))

    def ready():
        w("GEMM_WAIT_VM 0\nGEMM_WAIT_LGKM 0\nGEMM_CTA_BARRIER")

    def read_shared(slot, first, stop):
        # Prefix loads are issued only after this slot has passed ready().
        if first == 0:
            w(f"v_add_u32_e32 {vr(t + 9)}, {slot * slotbytes}, {vr(t)}")
        for sh in range(first, stop):
            w(f"GEMM_LDS_TO_REG {v4(ds + 4 * sh)}, {vr(t + 9)}, {sh * 1024}")

    def consume(slot, already_read=0):
        pr = p0 if slot == 0 else p1
        if already_read < sf:
            read_shared(slot, already_read, sf)
        for first in range(0, sf, mmac_group):
            group = range(first, min(sf, first + mmac_group))
            for h in range(2):
                for sh in group:
                    if h == 0:
                        w(f"GEMM_WAIT_LGKM {sf - 1 - sh}")
                    for pp in range(pf):
                        acc = 4 * (sh * pf + pp)
                        xs = (
                            pr + 4 * pp + 2 * h if direction == "M" else ds + 4 * sh + 2 * h
                        )
                        ws = (
                            ds + 4 * sh + 2 * h if direction == "M" else pr + 4 * pp + 2 * h
                        )
                        w(f"GEMM_MMA_I8 {v4(acc)}, {v2(xs)}, {v2(ws)}, {v4(acc)}")

    w(f"v_mov_b32_e32 {vr(t + 11)}, s{phi}")
    if ss:
        w(
            f"v_readfirstlane_b32 s26, {vr(t + 4)}\ns_mov_b32 s27, {waves * 1024}\ns_mov_b32 s28, {waves * 2048}\ns_mov_b32 s29, {waves * 3072}"
        )
    issue(0)
    for a in range(ac):
        w(f"v_mov_b32_e32 v{a}, 0")
    w("s_mov_b32 s24, 40\n.p2align 5\n.L_pair:")
    ready()
    if read_first:
        read_shared(0, 0, read_first)
    issue(1)
    consume(0, read_first)
    ready()
    if read_first:
        read_shared(1, 0, read_first)
    w("s_sub_u32 s24, s24, 1\ns_cmp_eq_u32 s24, 0\ns_cbranch_scc1 .L_skip_next")
    issue(0)
    w(".L_skip_next:")
    consume(1, read_first)
    w("s_cmp_lg_u32 s24, 0\ns_cbranch_scc1 .L_pair\nGEMM_WAIT_VM 0")
    # Epilogue: row strip for M direction, column strip for N direction.
    w(f"v_mov_b32_e32 {vr(t + 11)}, s13")
    w(
        f"v_lshrrev_b32_e32 {vr(t + 2)}, 4, {vr(t)}\nv_and_b32_e32 {vr(t + 2)}, 15, {vr(t + 2)}\nv_lshrrev_b32_e32 {vr(t + 3)}, 8, {vr(t)}\nv_lshlrev_b32_e32 {vr(t + 3)}, 4, {vr(t + 3)}"
    )
    w(scale_u32(vr(t + 4), vr(t + 1), pf * 16))
    if direction == "M":
        w(f"v_add_u32_e32 {vr(t + 2)}, {vr(t + 4)}, {vr(t + 2)}")
    else:
        w(
            f"v_lshlrev_b32_e32 {vr(t + 4)}, 2, {vr(t + 4)}\nv_add_u32_e32 {vr(t + 3)}, {vr(t + 4)}, {vr(t + 3)}"
        )
    w(
        f"v_add_u32_e32 {vr(t + 2)}, s22, {vr(t + 2)}\nv_add_u32_e32 {vr(t + 3)}, s23, {vr(t + 3)}"
    )
    rows = pf if direction == "M" else sf
    for row in range(rows):
        w(
            f"v_add_u32_e32 {vr(t + 7)}, {row * 16}, {vr(t + 2)}\nv_cmp_gt_u32_e32 vcc, s14, {vr(t + 7)}\ns_and_saveexec_b64 s[28:29], vcc\ns_cbranch_execz .L_store_skip_{row}"
        )
        w(
            f"v_mul_u32_u24_e32 {vr(t + 8)}, 0x11000, {vr(t + 7)}\nv_add_u32_e32 {vr(t + 8)}, {vr(t + 3)}, {vr(t + 8)}\nv_add_co_u32_e32 {vr(t + 5)}, vcc, s12, {vr(t + 8)}\nv_addc_co_u32_e32 {vr(t + 6)}, vcc, 0, {vr(t + 11)}, vcc"
        )
        cols = sf if direction == "M" else pf
        for col in range(cols):
            sh, pr = (col, row) if direction == "M" else (row, col)
            acc = 4 * (sh * pf + pr)
            w(f"GEMM_EPILOGUE_STORE_GLC_SLC {v2(t + 5)}, {v4(acc)}, {col * 64}")
        w(f".L_store_skip_{row}:\ns_or_b64 exec, exec, s[28:29]")
    w("s_endpgm\n.L_func_end:")
    w(
        f".size {fam}, .L_func_end-{fam}\n.section .rodata,#alloc\n.p2align 6\n.amdhsa_kernel {fam}"
    )
    directives = {
        "group_segment_fixed_size": lds,
        "private_segment_fixed_size": 0,
        "kernarg_size": 40,
        "user_sgpr_count": 6,
        "user_sgpr_private_segment_buffer": 1,
        "user_sgpr_dispatch_ptr": 0,
        "user_sgpr_queue_ptr": 0,
        "user_sgpr_kernarg_segment_ptr": 1,
        "user_sgpr_dispatch_id": 0,
        "user_sgpr_flat_scratch_init": 0,
        "user_sgpr_private_segment_size": 0,
        "uses_dynamic_stack": 0,
        "system_sgpr_private_segment_wavefront_offset": 0,
        "system_sgpr_workgroup_id_x": 1,
        "system_sgpr_workgroup_id_y": 1,
        "system_sgpr_workgroup_id_z": 0,
        "system_sgpr_workgroup_info": 0,
        "system_vgpr_workitem_id": 0,
        "next_free_vgpr": vgpr,
        "next_free_sgpr": 32,
        "reserve_flat_scratch": 0,
        "reserve_xnack_mask": 1,
        "float_round_mode_32": 0,
        "float_round_mode_16_64": 0,
        "float_denorm_mode_32": 3,
        "float_denorm_mode_16_64": 3,
        "dx10_clamp": 1,
        "ieee_mode": 1,
        "fp16_overflow": 0,
    }
    for k, v in directives.items():
        w(f".amdhsa_{k} {v}")
    w(".end_amdhsa_kernel")
    metadata = METADATA.format(symbol=fam, lds=lds, vgpr=vgpr, threads=waves * 64)
    w(".amdgpu_metadata\n---\n" + metadata + "...\n.end_amdgpu_metadata")
    src = "\n".join(L) + "\n"
    if prefetch_order == "private_first":
        src = _private_first_source(src, cfg)
    return src, cfg

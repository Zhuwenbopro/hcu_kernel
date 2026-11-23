import re
import yaml


def metadata(s, extra_sgpr=False):
    s = re.sub(r"(\.amdhsa_kernarg_size\s+)40\b", r"\g<1>56", s)
    if extra_sgpr:
        s = re.sub(r"(\.amdhsa_next_free_sgpr\s+)\d+", r"\g<1>28", s)
    if ".amdgpu_metadata" in s:
        a, rest = s.split(".amdgpu_metadata", 1)
        doc, b = rest.split(".end_amdgpu_metadata", 1)
        m = yaml.safe_load(doc.expandtabs(8))
        for rec in m["amdhsa.kernels"]:
            for key in [".args", ".unfolded_args"]:
                if key in rec:
                    rec[key] += [
                        {
                            ".address_space": "global",
                            ".offset": o,
                            ".size": 8,
                            ".value_kind": "global_buffer",
                        }
                        for o in [40, 48]
                    ]
            rec[".kernarg_segment_size"] = 56
            if extra_sgpr:
                rec[".sgpr_count"] = 32
        s = (
            a
            + ".amdgpu_metadata\n---\n"
            + yaml.safe_dump(m, sort_keys=False)
            + "...\n.end_amdgpu_metadata"
            + b
        )
    return s


def conversion(a, sx, sw, tmp, addr, offset, flags="glc slc"):
    out = []
    for e in range(4):
        v = a + e
        out += [
            f"v_cvt_f32_i32 v{v}, v{v}",
            f"v_mul_f32_e32 v{tmp}, v{sx}, v{sw + e}",
            f"v_mul_f32_e32 v{v}, v{tmp}, v{v}",
            f"v_lshrrev_b32_e32 v{tmp}, 16, v{v}",
            f"v_and_b32_e32 v{tmp}, 1, v{tmp}",
            f"v_add_u32_e32 v{v}, 0x7fff, v{v}",
            f"v_add_u32_e32 v{v}, v{tmp}, v{v}",
        ]
    out += [
        f"v_lshrrev_b32_e32 v{a}, 16, v{a}",
        f"v_and_b32_e32 v{a + 1}, 0xffff0000, v{a + 1}",
        f"v_or_b32_e32 v{a}, v{a}, v{a + 1}",
        f"v_lshrrev_b32_e32 v{a + 2}, 16, v{a + 2}",
        f"v_and_b32_e32 v{a + 3}, 0xffff0000, v{a + 3}",
        f"v_or_b32_e32 v{a + 1}, v{a + 2}, v{a + 3}",
        f"global_store_dwordx2 v[{addr}:{addr + 1}], v[{a}:{a + 1}], off offset:{offset // 2} {flags}",
    ]
    return "\n".join(out) + "\n"


def patch_generated(source, rec):
    # All K-loop instructions remain verbatim. Only the final output region changes.
    addr = list(range(rec["address_registers"][0], rec["address_registers"][-1] + 1))
    col, row, offset = addr[3], addr[7], addr[8]
    t = rec["accumulator_VGPR"]
    stores = re.findall(
        r"GEMM_EPILOGUE_STORE_GLC_SLC v\[(\d+):\d+\], v\[(\d+):\d+\], (\w+)", source
    )
    offsets = sorted(set(int(z[2], 0) for z in stores))
    sx = t + 4 * len(offsets)
    to = sx + 1
    tmp = sx + 2
    assert tmp < addr[0], (rec["family"], tmp, addr)
    pos = source.rfind("GEMM_WAIT_VM 0\nv_mov_b32_e32")
    assert pos >= 0
    pre, ep = source[:pos], source[pos:]
    ep = ep.replace(
        "GEMM_WAIT_VM 0\n",
        "GEMM_WAIT_VM 0\ns_load_dwordx4 s[0:3], s[4:5], 0x28\nGEMM_WAIT_LGKM 0\n",
        1,
    )
    line = f"v_add_u32_e32 v{col}, s23, v{col}\n"
    assert ep.count(line) == 1
    loads = "".join(
        f"global_load_dwordx4 v[{t + 4 * i}:{t + 4 * i + 3}], v{col}, s[2:3] offset:{o}\n"
        for i, o in enumerate(offsets)
    )
    ep = ep.replace(line, line + loads, 1)
    pattern = rf"(v_mul_u32_u24_e32 v{offset}, 0x11000, v{row}\n)"
    ep, num = re.subn(
        pattern,
        f"v_lshlrev_b32_e32 v{to}, 2, v{row}\nglobal_load_dword v{sx}, v{to}, s[0:1]\n"
        + r"\1",
        ep,
    )
    assert num == rec["BM"] // 16 if rec["direction"] == "N" else num == rec["BM"] // 64
    line = f"v_add_u32_e32 v{offset}, v{col}, v{offset}\n"
    assert ep.count(line) == num
    ep = ep.replace(line, line + f"v_lshrrev_b32_e32 v{offset}, 1, v{offset}\n")
    pat = r"GEMM_EPILOGUE_STORE_GLC_SLC v\[(\d+):\d+\], v\[(\d+):\d+\], (\w+)\n"
    call = [0]

    def repl(m):
        ptr, a, o = int(m[1]), int(m[2]), int(m[3], 0)
        prefix = "GEMM_WAIT_VM 0\n" if call[0] % len(offsets) == 0 else ""
        call[0] += 1
        return prefix + conversion(a, sx, t + 4 * offsets.index(o), tmp, ptr, o)

    ep = re.sub(pat, repl, ep)
    assert call[0] == len(stores)
    return pre + ep, "GEMM_WAIT_VM 0\nv_mov_b32_e32", pos

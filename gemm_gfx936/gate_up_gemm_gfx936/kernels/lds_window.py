"""Reuse the shared-X operand registers of N-split direct-X kernels."""

import re

import yaml


LDS_READ = re.compile(r"GEMM_LDS_TO_REG v\[(\d+):(\d+)\], v\d+, (\d+)")
MMAC_X = re.compile(r"GEMM_MMA_I8 v\[\d+:\d+\], v\[(\d+):\d+\],")
LDS_WAIT = re.compile(r"GEMM_WAIT_LGKM (\d+)")
VGPR = re.compile(r"\bv\[(\d+):(\d+)\]|\bv(\d+)\b")


def _schedule_stage(lines, begin, layout, window):
    count = layout["SF"]
    base = layout["shared_fragments"][0]
    reads = {}
    last_use = {}
    # Every X fragment has two K halves and is used with each private W
    # fragment. Grouping can interleave the last MMACs for adjacent X rows.
    remaining_mmacs = 2 * count * layout["PF"]
    for end in range(begin, len(lines)):
        if match := LDS_READ.fullmatch(lines[end]):
            slot = int(match[3]) // 1024
            if int(match[1]) != base + 4 * slot or int(match[2]) != base + 4 * slot + 3:
                raise ValueError("Unexpected LDS destination registers")
            reads[slot] = lines[end]
        if match := MMAC_X.match(lines[end]):
            slot = (int(match[1]) - base) // 4
            last_use[slot] = end
            remaining_mmacs -= 1
            if remaining_mmacs == 0:
                break
    if remaining_mmacs or set(reads) != set(range(count)) or set(last_use) != set(reads):
        raise ValueError("Incomplete shared-X fragment sequence")

    # Refill a slot immediately after its previous occupant's final use. The
    # DS read address for this LDS buffer stays live through the last MMAC.
    refill = {pos: slot + window for slot, pos in last_use.items() if slot + window < count}
    issued = 0
    waited = 0
    stage = []
    for i in range(begin, end + 1):
        line = lines[i]
        if match := LDS_READ.fullmatch(line):
            if int(match[3]) // 1024 >= window:
                continue
            issued += 1
        if match := LDS_WAIT.fullmatch(line):
            slot = count - 1 - int(match[1])
            if slot != waited or slot >= issued:
                raise ValueError("Unexpected LDS wait order")
            # Permit precisely the already-issued reads younger than this slot.
            line = f"GEMM_WAIT_LGKM {issued - 1 - slot}"
            waited += 1
        stage.append(line)
        if i in refill:
            if refill[i] != issued:
                raise ValueError("Refill would change LDS issue order")
            stage.append(reads[issued])
            issued += 1
    if waited != count or issued != count:
        raise ValueError("Incomplete LDS window")
    lines[begin : end + 1] = stage


def apply_lds_window(source, layout, window):
    """Return assembly with a bounded register window and the VGPR savings."""
    count = layout["SF"]
    group = layout["mmac_group"]
    if (layout["direction"] != "N" or not group <= window < count or window % group):
        raise ValueError("The N-split LDS window must fit whole MMAC groups")
    lines = source.splitlines()
    stages = [
        i for i, line in enumerate(lines)
        if (match := LDS_READ.fullmatch(line)) and int(match[3]) == 0
    ]
    if len(stages) != 2:
        raise ValueError("Expected two LDS buffers")
    for begin in reversed(stages):
        _schedule_stage(lines, begin, layout, window)

    base = layout["shared_fragments"][0]
    saved = 4 * (count - window)

    def compact(index):
        if index < base:
            return index
        if index < base + 4 * count:
            return base + (index - base) % (4 * window)
        return index - saved

    def rename(match):
        if match[3]:
            return f"v{compact(int(match[3]))}"
        return f"v[{compact(int(match[1]))}:{compact(int(match[2]))}]"

    code, metadata = ("\n".join(lines) + "\n").split(".amdgpu_metadata", 1)
    code = VGPR.sub(rename, code)
    code, updated = re.subn(
        r"(\.amdhsa_next_free_vgpr )(\d+)",
        lambda match: match[1] + str(int(match[2]) - saved),
        code,
    )
    if updated != 1:
        raise ValueError("Missing VGPR declaration")
    doc, suffix = metadata.split(".end_amdgpu_metadata", 1)
    meta = yaml.safe_load(doc.expandtabs(8))
    for kernel in meta["amdhsa.kernels"]:
        kernel[".vgpr_count"] -= saved
    source = code + ".amdgpu_metadata\n---\n" + yaml.safe_dump(meta, sort_keys=False)
    return source + "...\n.end_amdgpu_metadata" + suffix, saved

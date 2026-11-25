import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
WEIGHT_SHAPE = (248320, 5120)
HALF_BYTES = WEIGHT_SHAPE[0] // 2 * WEIGHT_SHAPE[1] * 2
BLOCK_COUNTS = (64, 65, 72, 73, 80, 128, 144, 400)
DEPTHS = (8, 12)
TIMESTAMP_ARG = """      - .address_space: global
        .offset: 16
        .size: 8
        .value_kind: global_buffer
"""


def module_path(blocks, depth, timestamps=False):
    suffix = "_timestamps" if timestamps else ""
    return BUILD / f"read_b{blocks}_q{depth}{suffix}.hsaco"


def build_module(llvm, source, blocks, depth, timestamps):
    module = module_path(blocks, depth, timestamps)
    assembly = module.with_suffix(".s")
    obj = module.with_suffix(".o")
    fields = {
        "KERNARG_SIZE": 24 if timestamps else 16,
        "VGPRS": 4 + 4 * depth,
        "NEXT_SGPR": 25 if timestamps else 14,
        "SGPRS": 29 if timestamps else 18,
        "TIMESTAMP_ARG": TIMESTAMP_ARG if timestamps else "",
    }
    for field, value in fields.items():
        source = source.replace(f"@{field}@", str(value))
    assembly.write_text(source, encoding="utf-8")
    subprocess.run([
        llvm / "llvm-mc", "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx938", "-mattr=+sramecc",
        "-filetype=obj", f"--defsym=BLOCKS={blocks}", f"--defsym=TOTAL_BYTES={HALF_BYTES}",
        f"--defsym=DEPTH={depth}", f"--defsym=TIMESTAMPS={int(timestamps)}", assembly, "-o", obj,
    ], check=True)
    subprocess.run([llvm / "ld.lld", "-shared", obj, "-o", module], check=True)


def main():
    llvm = Path(os.environ.get("DTK", "/opt/dtk")) / "aillvm/bin"
    source = (ROOT / "read_xr.s").read_text(encoding="utf-8")
    BUILD.mkdir(exist_ok=True)
    for blocks in BLOCK_COUNTS:
        for depth in DEPTHS:
            for timestamps in (False, True):
                build_module(llvm, source, blocks, depth, timestamps)
    print(f"Built {len(BLOCK_COUNTS) * len(DEPTHS) * 2} gfx938 code objects in {BUILD}")


if __name__ == "__main__":
    main()

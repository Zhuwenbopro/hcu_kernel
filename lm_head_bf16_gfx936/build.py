"""Build gfx936 LM-head native libraries and optional rolling code objects."""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def audit_rolling(image, prefix, cfg, dtk):
    from .audit import dtk_tool, dump_isa

    info = dump_isa(image, "rolling_mmac", prefix, dtk)
    resources = info["resources"]
    expected = {
        "vgpr_count": cfg.vgprs,
        "sgpr_count": 47,
        "group_segment_fixed_size": cfg.lds,
        "private_segment_fixed_size": 0,
        "vgpr_spill_count": 0,
        "sgpr_spill_count": 0,
    }
    if resources != expected:
        raise RuntimeError(f"{image.name}: unexpected code-object resources {resources}")
    symbols = subprocess.check_output([dtk_tool(dtk, "llvm-nm"), "-a", str(image)], text=True)
    bounds = {fields[2]: int(fields[0], 16) for line in symbols.splitlines()
              if len(fields := line.split()) == 3 and fields[2] in (".Lroll0", ".Lroll1")}
    dis = prefix.with_suffix(".isa").read_text()
    body = [op.strip() for op, address in re.findall(r"^(.+?)//\s+([0-9a-fA-F]+):", dis, re.M)
            if bounds[".Lroll0"] <= int(address, 16) < bounds[".Lroll1"]]
    wait = next(i for i, op in enumerate(body) if "vmcnt(0)" in op)
    barrier = body.index("s_barrier")
    w_loads = [i for i, op in enumerate(body)
               if op.startswith("buffer_load_dwordx4") and "s[16:19]" in op and "lds" in op]
    mmacs = [i for i, op in enumerate(body) if op.startswith("v_mmac_f32_16x16x16_bf16")]
    if (len(w_loads) != cfg.wloads or len(mmacs) != cfg.mb * cfg.nb * 2
            or not wait < barrier < w_loads[0] <= w_loads[-1] < mmacs[0]
            or not any("lgkmcnt(0)" in op for op in body[w_loads[-1] + 1:mmacs[0]])):
        raise RuntimeError(f"{image.name}: unexpected rolling load or MMAC order")
    info["rolling_loop"] = {
        "w_loads": [body[i] for i in w_loads],
        "mmacs": [body[i] for i in mmacs],
    }
    return info


def build_rolling(out, dtk):
    from .audit import dtk_tool
    from .kernels.rolling_asm import render
    from .rolling import policy

    out.mkdir()
    llvm_mc = dtk_tool(dtk, "llvm-mc")
    linker = dtk_tool(dtk, "ld.lld")
    records = {}
    for kernel in policy()[0]:
        prefix = out / kernel.stem
        asm, obj, image = (prefix.with_suffix(suffix) for suffix in (".s", ".o", ".hsaco"))
        asm.write_text(render(kernel.config))
        subprocess.run([llvm_mc, "--save-temp-labels", "-triple=amdgcn-amd-amdhsa",
                        "-mcpu=gfx936", "-mattr=+sramecc", "-filetype=obj",
                        str(asm), "-o", str(obj)], check=True)
        subprocess.run([linker, "-shared", str(obj), "-o", str(image)], check=True)
        info = audit_rolling(image, prefix, kernel.config, dtk)
        digest = info["code_object_sha256"]
        records[kernel.stem] = info | {"sha256": digest,
                                      "matches_reference": digest == kernel.reference_sha256}
        print(f"Audited {image.name} (reference match: {records[kernel.stem]['matches_reference']})",
              flush=True)
    (out / "audit.json").write_text(json.dumps(records, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtk-root", type=Path,
                        default=os.environ.get("DTKROOT") or os.environ.get("ROCM_PATH") or "/opt/dtk")
    parser.add_argument("--hipcc", default=os.environ.get("HIPCC"))
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--tuned", action="store_true", help="Build and audit the rolling MMAC kernels")
    args = parser.parse_args()
    dtk = args.dtk_root.resolve()
    hipcc = args.hipcc or str(dtk / "bin" / "hipcc")
    if shutil.which(hipcc) is None:
        parser.error(f"Cannot find hipcc at {hipcc}. Set --dtk-root or --hipcc.")

    source = ROOT / "kernels"
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="native-", dir=build) as tmp:
        out = Path(tmp)
        for name in ("single", "scalar"):
            command = [hipcc, "-O3", "-std=c++17", "--offload-arch=gfx936", "-fPIC", "-shared",
                       str(source / f"{name}.hip"), "-o", str(out / f"{name}.so")]
            subprocess.run(command, check=True)

        command = [args.cxx, "-O2", "-std=c++17", "-fPIC", "-shared", "-D__HIP_PLATFORM_AMD__",
                   f"-I{dtk / 'include'}", f"-I{dtk / 'hip' / 'include'}",
                   str(source / "indexed.cpp"), "-o", str(out / "indexed.so"),
                   f"-L{dtk / 'lib'}", f"-L{dtk / 'hip' / 'lib'}", "-lrocblas", "-lamdhip64"]
        subprocess.run(command, check=True)
        if args.tuned:
            built = out / "rolling"
            build_rolling(built, dtk)
            installed = build / "rolling"
            installed.mkdir(exist_ok=True)
            for path in sorted(built.iterdir(), key=lambda path: path.name == "audit.json"):
                if path.suffix != ".o":
                    path.replace(installed / path.name)
        for name in ("single.so", "scalar.so", "indexed.so"):
            (out / name).replace(build / name)
            print(f"Built {build.relative_to(ROOT) / name}")


if __name__ == "__main__":
    main()

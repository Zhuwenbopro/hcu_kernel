"""Build the three native gfx936 libraries in build/."""

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtk-root", type=Path,
                        default=os.environ.get("DTKROOT") or os.environ.get("ROCM_PATH") or "/opt/dtk")
    parser.add_argument("--hipcc", default=os.environ.get("HIPCC"))
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
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
        for name in ("single.so", "scalar.so", "indexed.so"):
            (out / name).replace(build / name)
            print(f"Built {build.relative_to(ROOT) / name}")


if __name__ == "__main__":
    main()

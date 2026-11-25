"""Build the gfx938 GEMM and relayout shared libraries with DTK hipcc."""

import argparse
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"


def build(hipcc):
    BUILD.mkdir(exist_ok=True)
    flags = [
        "-O3", "-std=c++17", "-Wno-return-type-void-func",
        "--offload-arch=gfx938", "-fPIC", "-shared",
    ]
    variants = (
        ("libexp4_ring_fp8.so", []),
        ("libexp4_ring_fp8_bn128.so", ["-DEXP4_NBW=2"]),
        ("libexp4_ring_vec_reduce.so", ["-DEXP4_VEC_REDUCE"]),
        ("libexp4_ring_vec_reduce_bn128.so", ["-DEXP4_VEC_REDUCE", "-DEXP4_NBW=2"]),
    )
    for name, defines in variants:
        subprocess.run(
            [hipcc, *flags, *defines, "kernels/fp8_pi32_gemm.cu", "-o", f"build/{name}"],
            cwd=ROOT,
            check=True,
        )
    subprocess.run(
        [hipcc, *flags, "kernels/pi32_relayout.cu", "-o", "build/libpi32_relayout.so"],
        cwd=ROOT,
        check=True,
    )
    print(f"Built five gfx938 libraries in {BUILD}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hipcc", default=os.environ.get("HIPCC", "hipcc"))
    build(parser.parse_args().hipcc)

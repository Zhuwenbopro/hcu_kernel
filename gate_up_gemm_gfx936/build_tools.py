"""DTK build helpers for the generated GateUp code objects."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from .policy import ROOT

SOURCE = ROOT / "kernels"
BUILD = ROOT / "build"
DTK = Path(os.environ.get("DTK_PATH", "/opt/dtk"))


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)


def assemble(source, output, include_dir):
    source.write_text(output, encoding="utf-8")
    obj = source.with_suffix(".o")
    code = source.with_suffix(".hsaco")
    tools = DTK / "aillvm" / "bin"
    run(
        tools / "llvm-mc",
        "--save-temp-labels",
        "-I",
        include_dir,
        "-triple=amdgcn-amd-amdhsa",
        "-mcpu=gfx936",
        "-mattr=+sramecc",
        "-filetype=obj",
        source,
        "-o",
        obj,
    )
    run(tools / "ld.lld", "-shared", obj, "-o", code)
    return code


def compile_hip(source, output, *flags):
    hipcc = shutil.which("hipcc") or DTK / "bin" / "hipcc"
    run(
        hipcc,
        "-O3",
        "-std=c++17",
        "--offload-arch=gfx936",
        "-shared",
        "-fPIC",
        source,
        "-o",
        output,
        *flags,
    )


def describe_file(path, dest):
    return {
        "file": path.relative_to(dest).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def write_catalog(dest, catalog):
    path = dest / "kernels.json"
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)

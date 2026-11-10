"""Build the archived gfx936 HIP kernels without a LightOp dependency."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=["all"])
    parser.add_argument("--hipcc", default=os.environ.get("HIPCC"))
    args = parser.parse_args()
    catalog = json.loads((ROOT / "variants.json").read_text())
    ids = list(catalog) if args.variants == ["all"] else args.variants
    if not ids or any(key not in catalog for key in ids):
        parser.error("variants must be all or IDs from variants.json")
    hipcc = args.hipcc or shutil.which("hipcc") or "/opt/dtk/bin/hipcc"
    out = ROOT / "build"
    out.mkdir(exist_ok=True)
    builds = {}
    for key in ids:
        spec = catalog[key]
        src = ROOT / spec["source"]
        target = out / (key + ".so")
        flags = ["-D" + name + "=" + str(value) for name, value in spec["config"].items()]
        command = [hipcc, "-O3", "-std=c++17", "--offload-arch=gfx936",
                   "-shared", "-fPIC", "-Dreg_kernel=reg_" + key, *flags,
                   str(src), "-o", str(target)]
        with (out / (key + ".build.log")).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        builds[key] = {"source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                       "binary_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                       "arch": "gfx936", "config": spec["config"]}
        print("built", key, flush=True)
    (out / "build_manifest.json").write_text(json.dumps(builds, indent=2) + "\n")

if __name__ == "__main__":
    main()

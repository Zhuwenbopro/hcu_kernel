"""Trace dispatched gfx936 kernels and disassemble the code objects they use."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from . import K, MAX_M, N, LMHead, rolling, route, workspace_rows
from .lm_head import _active_x_policy, _medium_policy


PACKAGE = Path(__file__).resolve().parent
BUILD = PACKAGE / "build"
DEFAULT_M = "1,2,3,4,15,64,65,128,129,256,257,4096,8192,16384,20000,20479,20480"


def parse_m(value):
    try:
        sizes = sorted({int(size) for size in value.split(",")})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from exc
    if not sizes or not 1 <= sizes[0] <= sizes[-1] <= MAX_M:
        raise argparse.ArgumentTypeError(f"Each M must be in [1, {MAX_M}]")
    return sizes


def launches(trace_path):
    trace = json.loads(trace_path.read_text())
    kernels = (event for event in trace["traceEvents"] if event.get("cat") == "kernel")
    return sorted(kernels, key=lambda event: event["ts"])


def capture(sizes, device, trace_path, tuned=False):
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    torch.manual_seed(723)
    rows = max(route(m, tuned=tuned)[-1] for m in sizes)
    x = torch.randn((rows, K), device=device, dtype=torch.bfloat16)
    weight = torch.randn((K, N), device=device, dtype=torch.bfloat16)
    out = x.new_empty((rows, N))
    max_work = max(workspace_rows(m, tuned=tuned) for m in sizes)
    workspace = x.new_empty((K * max_work,))
    head = LMHead(weight, tuned=tuned)
    # Shapes and strides determine dispatch. Both layouts are viewed in one
    # synthetic allocation for this structural audit.
    baseline_weight = weight.view(N, K).T

    def call(mode, m):
        if mode == "baseline":
            torch.mm(x[:m], baseline_weight, out=out[:m])
        else:
            p = route(m, tuned=tuned)[-1]
            work_rows = workspace_rows(m, tuned=tuned)
            work = workspace[:K * work_rows].view(K, work_rows) if work_rows else None
            head(x[:m], out=out[:p], workspace=work)

    for mode in ("baseline", "optimized"):
        for m in sizes:
            call(mode, m)
    torch.cuda.synchronize(device)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        for mode in ("baseline", "optimized"):
            for m in sizes:
                with record_function(f"{mode} M={m}"):
                    call(mode, m)
        torch.cuda.synchronize(device)
    trace.export_chrome_trace(str(trace_path))


def native_symbol(m):
    return f"_Z9vector_knILi{m}ELi8ELi2ELi4EEvPKtS1_Pt"



def launch_grid(kernel, expected_wgs, label):
    grid = kernel["args"]["grid"]
    block = kernel["args"]["block"]
    raw_grid, block_size = math.prod(grid), math.prod(block)
    if raw_grid == expected_wgs:
        unit = "workgroups"
    elif raw_grid == expected_wgs * block_size:
        unit = "work-items"
    else:
        raise RuntimeError(f"{label}: grid {grid}, block {block}, expected {expected_wgs} workgroups")
    return {"grid": grid, "block": block, "grid_unit": unit, "workgroups": expected_wgs}


def tuned_dispatches(events, m, p, family):
    if family == "rolling":
        obj = rolling.route(m)
        cfg = obj.config
        ntiles = N // cfg.bn
        parts = [ntiles * (i + 1) // obj.parts - ntiles * i // obj.parts
                 for i in range(obj.parts)]
        widths = [cfg.bn * count for count in parts]
        prepared = (1 + int(m != cfg.bm)) if cfg.xt else 0
    elif family == "active_x":
        parts = _active_x_policy()[m]
        widths = [N // parts] * parts
        prepared = 1
    else:
        tiles, _ = _medium_policy()[p]
        widths = [256 * count for count in tiles]
        prepared = 1 + int(m != p)

    count = prepared + len(widths)
    if len(events) < count:
        raise RuntimeError(f"M={m}: incomplete {family} GPU trace")
    auxiliary = [e["name"] for e in events[:prepared]]
    packing_kernel = "_transpose_active_x" if family == "active_x" else "copy"
    if auxiliary and packing_kernel not in auxiliary[0].lower():
        raise RuntimeError(f"M={m}: expected X packing before {family} GEMM")
    rows = []
    for part, (event, width) in enumerate(zip(events[prepared:count], widths)):
        symbol = event["name"]
        if family == "rolling":
            if symbol != "rolling_mmac" or math.prod(event["args"]["block"]) != cfg.nw * 64:
                raise RuntimeError(f"M={m}: unexpected rolling kernel {symbol}")
            tm, tn, tk = cfg.bm, cfg.bn, cfg.bk
        else:
            match = re.search(r"_MT(\d+)x(\d+)x(\d+)_", symbol)
            if not symbol.startswith("Cijk_") or not match or "_GSU1_" not in symbol:
                raise RuntimeError(f"M={m}: unexpected packed-X kernel {symbol}")
            tn, tm, tk = (int(value) for value in match.groups())
        grid = launch_grid(event, math.ceil(p / tm) * math.ceil(width / tn), f"M={m} {family}")
        rows.append(dict(mode="optimized", m=m, p=p, family=family, part=part, columns=width,
                         tile=[tm, tn, tk], kernel=symbol, auxiliary=auxiliary if part == 0 else [],
                         **grid))
        if family == "rolling":
            rows[-1]["rolling_stem"] = obj.stem
    return rows, count


def audit_dispatches(sizes, events, policy, tuned=False):
    by_shape = []
    cursor = 0
    for mode in ("baseline", "optimized"):
        for m in sizes:
            family, tm, tn, p = route(m, tuned=tuned)
            if mode == "optimized" and family in ("rolling", "segmented", "active_x"):
                rows, count = tuned_dispatches(events[cursor:], m, p, family)
                cursor += count
                by_shape.extend(rows)
                continue
            if mode == "baseline":
                family, tm, tn, tk, p = "rocblas", 256, 256, 32, m
            elif family == "large":
                tk = 16
            else:
                tk = policy[str(m)].get("tk")
            count = 1 if family != "large" else 2 + (m != p)
            call = events[cursor:cursor + count]
            if len(call) != count:
                raise RuntimeError(f"{mode} M={m}: missing kernels in profiler trace")
            cursor += count
            kernel = call[-1]
            name = kernel["name"]
            short = name.split("(")[0].strip()
            auxiliary = [event["name"] for event in call[:-1]]
            if family == "gemv":
                signature = f"vector_kn<{m},8,2,4>"
                matches = short == native_symbol(m) or signature in short.replace(" ", "")
            elif family == "triton":
                matches = short.endswith("_head")
            else:
                # Tensile writes its column-major N tile before M in the symbol.
                tile = f"_MT{tn}x{tm}x{tk}_"
                matches = name.startswith("Cijk_") and tile in name and "_GSU1_" in name
                if family == "vendor":
                    matches = matches and name == policy[str(m)]["symbol"]
            if not matches:
                raise RuntimeError(f"{mode} M={m}: unexpected kernel {name!r} for {family}")
            if family == "large" and "copy" not in auxiliary[0].lower():
                raise RuntimeError(f"{mode} M={m}: expected an X copy before GEMM")
            grid = launch_grid(kernel, math.ceil(p / tm) * math.ceil(N / tn), f"{mode} M={m}")
            by_shape.append({
                "mode": mode, "m": m, "p": p, "family": family, "tile": [tm, tn, tk],
                "kernel": name, "auxiliary": auxiliary, **grid,
            })
    if cursor != len(events):
        raise RuntimeError(f"Profiler trace contains {len(events) - cursor} unexpected kernels")
    return by_shape


def dtk_tool(root, name):
    for rel in ("bin", "aillvm/bin", "llvm/bin"):
        candidate = root / rel / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    fallback = shutil.which(name)
    if fallback:
        return fallback
    raise RuntimeError(f"Cannot find {name} under {root} or on PATH")


def dump_isa(code_object, symbol, prefix, root):
    symbols = subprocess.check_output(
        [dtk_tool(root, "llvm-nm"), "-S", "--defined-only", str(code_object)], text=True,
    )
    text_symbols = {
        fields[3]: (int(fields[0], 16), int(fields[1], 16))
        for line in symbols.splitlines() if len(fields := line.split()) == 4
        and fields[2] in {"T", "t"}
    }
    if symbol not in text_symbols:
        raise RuntimeError(f"Cannot find {symbol} in {code_object}")
    start, size = text_symbols[symbol]
    if size:
        end = start + size
    else:
        # Tensile's zero-sized function symbols are followed by local labels
        # inside the same kernel. Use the next kernel or the end of .text.
        following = [addr for name, (addr, _) in text_symbols.items()
                     if name.startswith("Cijk_") and addr > start]
        if following:
            end = min(following)
        else:
            sections = subprocess.check_output(
                [dtk_tool(root, "llvm-readelf"), "-SW", str(code_object)], text=True,
            )
            section = re.search(r"\.text\s+PROGBITS\s+([0-9a-fA-F]+)\s+\S+\s+([0-9a-fA-F]+)", sections)
            if not section:
                raise RuntimeError(f"Cannot find .text in {code_object}")
            end = int(section[1], 16) + int(section[2], 16)
    command = [dtk_tool(root, "llvm-objdump"), "-d", "--mcpu=gfx936",
               f"--start-address={start}", f"--stop-address={end}", str(code_object)]
    dis = subprocess.check_output(command, text=True)
    if f"<{symbol}>:" not in dis:
        raise RuntimeError(f"Disassembler cannot find {symbol}")
    if (symbol.startswith("Cijk_") or symbol == "_head") and "v_mmac_f32_16x16x16_bf16" not in dis:
        raise RuntimeError(f"{symbol} lacks the expected BF16-to-FP32 MMAC")
    notes = subprocess.check_output(
        [dtk_tool(root, "llvm-readelf"), "-n", str(code_object)], text=True,
    )
    sections = re.split(r"(?=\n  - \.args:)", notes)
    note = next((entry for entry in sections if re.search(
        rf"(?m)^\s*\.name:\s*['\"]?{re.escape(symbol)}['\"]?\s*$", entry,
    )), "")
    if not note:
        raise RuntimeError(f"Cannot find metadata for {symbol}")
    digest = hashlib.sha256(code_object.read_bytes()).hexdigest()
    prefix.with_suffix(".isa").write_text(dis)
    prefix.with_suffix(".notes").write_text(note)
    resources = {}
    for key in ("vgpr_count", "sgpr_count", "group_segment_fixed_size", "private_segment_fixed_size",
                "vgpr_spill_count", "sgpr_spill_count"):
        value = re.search(rf"(?m)^\s*\.{key}:\s*(\d+)", note)
        if value:
            resources[key] = int(value[1])
    return {
        "symbol": symbol, "code_object_sha256": digest, "resources": resources,
        "text_start": start, "text_end": end,
        "static_bf16_mmac": dis.count("v_mmac_f32_16x16x16_bf16"),
        "isa": prefix.with_suffix(".isa").name,
    }


def native_isa(m, output, root):
    library = "single" if m == 1 else "scalar"
    source = BUILD / f"{library}.so"
    env = dict(os.environ)
    bins = [str(root / rel) for rel in ("bin", "aillvm/bin", "llvm/bin")]
    env["PATH"] = os.pathsep.join([*bins, env.get("PATH", "")])
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        copy = shutil.copy2(source, tmp)
        subprocess.run([dtk_tool(root, "extractkernel"), "-i", Path(copy).name],
                       cwd=tmp, env=env, check=True, capture_output=True, text=True)
        device_images = sorted(Path(tmp).glob("*gfx936*.hsaco"))
        if len(device_images) != 1:
            raise RuntimeError(f"Expected one gfx936 image in {library}.so, found {device_images}")
        image = output / f"{library}.hsaco"
        shutil.copy2(device_images[0], image)
    return dump_isa(image, native_symbol(m), output / library, root)


def rocblas_isa(rows, output, root, library_dir):
    directories = [library_dir] if library_dir else [
        *sorted((root / "lib" / "rocblas").glob("library*")),
        *sorted((root / "lib64" / "rocblas").glob("library*")),
    ]
    results = []
    names = sorted({row["kernel"] for row in rows if row["kernel"].startswith("Cijk_")})
    for i, name in enumerate(names):
        layout = "_".join(name.split("_")[1:3])
        pattern = f"*BB_HPA*_{layout}_*gfx936*"
        candidates = (path for directory in directories for path in directory.glob(pattern)
                      if path.suffix in {".co", ".hsaco"})
        image = next((path for path in candidates if name.encode() + b"\x00" in path.read_bytes()), None)
        if image is None:
            raise RuntimeError(f"Cannot resolve rocBLAS code object for {name}; set --rocblas-library-dir")
        result = dump_isa(image, name, output / f"rocblas-{i}", root)
        result["code_object"] = str(image)
        results.append(result)
    return results


def collect_isa(rows, output, root, library_dir, triton_cache):
    gemv_shapes = sorted({row["m"] for row in rows if row["family"] == "gemv"})
    native = [native_isa(m, output, root) for m in gemv_shapes]
    triton = []
    if any(row["family"] == "triton" for row in rows):
        objects = sorted(triton_cache.rglob("_head.hsaco"))
        if not objects:
            raise RuntimeError("No gfx936 _head.hsaco in the Triton cache")
        unique = set()
        for obj in objects:
            binary = obj.read_bytes()
            digest = hashlib.sha256(binary).hexdigest()
            if digest in unique:
                continue
            unique.add(digest)
            prefix = output / f"triton-{digest[:12]}"
            prefix.with_suffix(".hsaco").write_bytes(binary)
            triton.append(dump_isa(obj, "_head", prefix, root))
    stems = sorted({row["rolling_stem"] for row in rows if "rolling_stem" in row})
    rolling_isa = [dump_isa(BUILD / "rolling" / f"{stem}.hsaco", "rolling_mmac", output / stem, root)
                   for stem in stems]
    return {"native": native, "triton": triton, "rolling": rolling_isa,
            "rocblas": rocblas_isa(rows, output, root, library_dir)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=parse_m, default=DEFAULT_M, help="Comma-separated M values")
    parser.add_argument("--device", type=int, default=0, help="Logical HIP device index")
    parser.add_argument("--dtk-root", type=Path, default=os.environ.get("DTKROOT") or os.environ.get("ROCM_PATH"),
                        help="DTK SDK directory used for disassembly")
    parser.add_argument("--rocblas-library-dir", type=Path, default=os.environ.get("ROCBLAS_TENSILE_LIBPATH"),
                        help="Optional rocBLAS gfx936 code-object directory")
    parser.add_argument("--dispatch-only", action="store_true", help="Check launch symbols and grids without disassembly")
    parser.add_argument("--tuned", action="store_true", help="Audit the opt-in merged TP2 paths")
    args = parser.parse_args()
    if not args.dispatch_only and args.dtk_root is None:
        parser.error("Set --dtk-root or DTKROOT/ROCM_PATH, or use --dispatch-only")
    output = BUILD / ("audit-tuned" if args.tuned else "audit")
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "trace.json"
    policy = json.loads((PACKAGE / "small-policy.json").read_text())

    import torch

    if not torch.cuda.is_available() or not 0 <= args.device < torch.cuda.device_count():
        parser.error("The requested logical HIP device is unavailable")
    device = torch.device("cuda", args.device)
    props = torch.cuda.get_device_properties(device)
    if not getattr(props, "gcnArchName", "").startswith("gfx936"):
        parser.error(f"Expected a gfx936 device, found {props}")
    with tempfile.TemporaryDirectory(prefix="triton-", dir=output) as tmp:
        os.environ["TRITON_CACHE_DIR"] = tmp
        with torch.cuda.device(device):
            capture(args.m, device, trace_path, tuned=args.tuned)
        rows = audit_dispatches(args.m, launches(trace_path), policy, tuned=args.tuned)
        report = {"device": str(props), "torch_version": str(torch.__version__),
                  "hip": str(torch.version.hip), "launches": rows}
        if not args.dispatch_only:
            report["isa"] = collect_isa(rows, output, args.dtk_root,
                                        args.rocblas_library_dir, Path(tmp))
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {len(rows)} launches. See {output / 'audit.json'}")


if __name__ == "__main__":
    main()

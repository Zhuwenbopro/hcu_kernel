"""Rebuild regression: exact results, INT8 extremes, K tails, M boundaries, and guards."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import traceback

import torch

try:
    from .python_api import VARIANTS, decode_gemm_int8, gemm_packed, kernel_info, pack_activation, pack_weight
    from .validation_utils import SENTINEL, cpu_check, guarded_output, guards_ok, reference
except ImportError:
    from python_api import VARIANTS, decode_gemm_int8, gemm_packed, kernel_info, pack_activation, pack_weight
    from validation_utils import SENTINEL, cpu_check, guarded_output, guards_ok, reference

ROOT = Path(__file__).resolve().parent

def run_validation(variants=None, output=None):
    variants = list(VARIANTS) if variants is None else variants
    torch.set_num_threads(8)
    props = torch.cuda.get_device_properties(0)
    assert "gfx936" in props.gcnArchName
    assert torch.cuda.mem_get_info()[0] > 4 * (1 << 30), "need at least 4 GiB free"
    state = {"status": "running", "records": [], "resources": {v: kernel_info(v, 0) for v in variants},
             "environment": {"gpu": props.name, "arch": props.gcnArchName, "cu": props.multi_processor_count,
                             "torch": torch.__version__, "hip": torch.version.hip}}
    output = Path(output) if output else ROOT / "build" / "validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    cases = [(m, 256, 576, "random") for m in (1, 3, 16, 24, 32, 33, 64, 127, 128, 129, 193, 256, 257, 512, 1023, 1024)]
    cases += [(32, 256, k, "random") for k in (64, 128, 192, 256, 320, 448, 512, 576, 960, 1024, 1088, 5056, 5120, 5184)]
    cases += [(32, 256, 576, pattern) for pattern in ("zero", "minmin", "minmax", "maxmax", "onehot")]
    cases += [(m, n, 5120, "random") for m in (32, 128, 129, 256, 512, 1024) for n in (17408, 34816)]
    cases = list(dict.fromkeys(cases))
    try:
        for ci, (m, n, k, pattern) in enumerate(cases):
            torch.manual_seed(202611110 + ci)
            x = torch.randint(-128, 128, (m, k), dtype=torch.int8, device="cuda")
            w = torch.randint(-128, 128, (n, k), dtype=torch.int8, device="cuda")
            if pattern == "zero":
                x.zero_()
            elif pattern == "minmin":
                x.fill_(-128); w.fill_(-128)
            elif pattern == "minmax":
                x.fill_(-128); w.fill_(127)
            elif pattern == "maxmax":
                x.fill_(127); w.fill_(127)
            elif pattern == "onehot":
                x.zero_(); rows = torch.arange(m, device=x.device); x[rows, (rows * 17 + 3) % k] = 1
            wp = pack_weight(w)
            expected = reference(x, w)
            ref_bad, cpu_count = cpu_check(x, w, expected)
            assert ref_bad == 0, "GPU reference disagrees with independent CPU INT64 samples"
            backing, y = guarded_output(m, n, x.device)
            for variant in variants + (["auto"] if n in (17408, 34816) else []):
                backing.fill_(SENTINEL)
                if variant == "auto":
                    decode_gemm_int8(x, wp, out=y)
                else:
                    gemm_packed(pack_activation(x, variant), wp, variant, out=y)
                torch.cuda.synchronize()
                bad = int((y != expected).sum().item())
                guard = guards_ok(backing)
                rec = {"variant": variant, "M": m, "N": n, "K": k, "pattern": pattern,
                       "checked": m * n, "bad": bad, "guard": guard, "cpu_reference_bad": ref_bad, "cpu_checked": cpu_count}
                state["records"].append(rec)
                assert bad == 0 and guard, rec
            del x, w, wp, expected, backing, y
        for key in variants:
            spec = VARIANTS[key]
            assert hashlib.sha256((ROOT / spec["source"]).read_bytes()).hexdigest() == spec["original_source_sha256"]
        state["status"] = "passed"
        state["summary"] = {"checks": len(state["records"]), "per_variant": dict(Counter(r["variant"] for r in state["records"])),
                            "output_elements_checked": sum(r["checked"] for r in state["records"]),
                            "bad": 0, "guards_all_pass": True, "source_hashes_match_validated_originals": True,
                            "M_range": [1, 1024], "K_tail_cases": sorted({r["K"] for r in state["records"]})}
    except Exception:
        state["status"] = "failed"
        state["traceback"] = traceback.format_exc()
        raise
    finally:
        output.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state["summary"]), flush=True)
    return state

def test_decode_gemm_int8():
    run_validation()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=str(ROOT / "build" / "validation.json"))
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    a = p.parse_args()
    run_validation(a.variants, a.output)

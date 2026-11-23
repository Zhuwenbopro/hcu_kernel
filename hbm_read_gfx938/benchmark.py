import argparse
from collections import Counter
from contextlib import ExitStack
import ctypes as c
import json
from pathlib import Path
import random
from statistics import median

import torch

from build import BLOCK_COUNTS, DEPTHS, HALF_BYTES, WEIGHT_SHAPE, module_path

POINTER = c.c_void_p
POINTER_PTR = c.POINTER(POINTER)
UINT = c.c_uint32
UINT_PTR = c.POINTER(UINT)
THREADS = 256
WAVE_SIZE = 64
EVENT_DISABLE_SYSTEM_FENCE = 0x20000000


def get_hip():
    hip = c.CDLL("libamdhip64.so")
    hip.hipGetErrorString.restype = c.c_char_p
    signatures = {
        "hipGetErrorString": [c.c_int],
        "hipDeviceGetPCIBusId": [c.POINTER(c.c_char), c.c_int, c.c_int],
        "hipExtStreamCreateWithCUMask": [POINTER_PTR, UINT, UINT_PTR],
        "hipExtStreamGetCUMask": [POINTER, UINT, UINT_PTR],
        "hipStreamDestroy": [POINTER],
        "hipStreamSynchronize": [POINTER],
        "hipModuleLoad": [POINTER_PTR, c.c_char_p],
        "hipModuleGetFunction": [POINTER_PTR, POINTER, c.c_char_p],
        "hipModuleUnload": [POINTER],
        "hipEventCreateWithFlags": [POINTER_PTR, UINT],
        "hipEventDestroy": [POINTER],
        "hipEventSynchronize": [POINTER],
        "hipEventElapsedTime": [c.POINTER(c.c_float), POINTER, POINTER],
        "hipExtModuleLaunchKernel": [POINTER] + [UINT] * 6 + [
            c.c_size_t, POINTER, POINTER_PTR, POINTER_PTR, POINTER, POINTER, UINT,
        ],
    }
    for name, argtypes in signatures.items():
        getattr(hip, name).argtypes = argtypes
    return hip


def check(hip, status):
    if status != 0:
        raise RuntimeError(hip.hipGetErrorString(status).decode())


def open_stream(hip, available_cus, requested_cus, cleanup):
    words = (available_cus + 31) // 32
    word_array = UINT * words
    default_mask = word_array()
    check(hip, hip.hipExtStreamGetCUMask(None, words, default_mask))
    enabled_bits = [bit for bit in range(words * 32) if default_mask[bit // 32] & (1 << (bit % 32))]
    if requested_cus is not None and requested_cus > len(enabled_bits):
        raise ValueError(f"Requested {requested_cus} CUs, but the default HIP stream exposes {len(enabled_bits)}")
    mask = word_array()
    for bit in enabled_bits[:requested_cus]:
        mask[bit // 32] |= 1 << (bit % 32)
    if not any(mask):
        raise RuntimeError("HIP returned an empty default CU mask")
    stream = POINTER()
    check(hip, hip.hipExtStreamCreateWithCUMask(c.byref(stream), words, mask))
    cleanup.callback(lambda: check(hip, hip.hipStreamDestroy(stream)))
    actual = word_array()
    check(hip, hip.hipExtStreamGetCUMask(stream, words, actual))
    if list(actual) != list(mask):
        raise RuntimeError(f"Requested CU mask {list(mask)}; read back {list(actual)}")
    return stream, list(actual)


def kernel(hip, path, cleanup):
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing. Run build.py first.")
    module, fn = POINTER(), POINTER()
    check(hip, hip.hipModuleLoad(c.byref(module), bytes(path)))
    cleanup.callback(lambda: check(hip, hip.hipModuleUnload(module)))
    check(hip, hip.hipModuleGetFunction(c.byref(fn), module, b"read_xr"))
    return fn


def launch(hip, fn, blocks, args, stream, start=None, stop=None):
    params = (POINTER * len(args))(*(c.cast(c.byref(arg), POINTER) for arg in args))
    check(hip, hip.hipExtModuleLaunchKernel(fn, blocks * THREADS, 1, 1, THREADS, 1, 1,
                                            0, stream, params, None, start, stop, 0))


def timed_launch(hip, fn, blocks, args, stream, start, stop):
    launch(hip, fn, blocks, args, stream, start, stop)
    check(hip, hip.hipEventSynchronize(stop))
    elapsed = c.c_float()
    check(hip, hip.hipEventElapsedTime(c.byref(elapsed), start, stop))
    if elapsed.value <= 0:
        raise RuntimeError("HIP reported a nonpositive dispatch duration")
    return elapsed.value * 1000


def prepare_weight(buffer, model_dir, half):
    if model_dir is None:
        buffer.uniform_(-1, 1)
    else:
        from safetensors import safe_open

        name = "lm_head.weight"
        index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        shard = model_dir / index["weight_map"][name]
        with safe_open(str(shard), framework="pt", device="cpu") as archive:
            tensor_slice = archive.get_slice(name)
            if tuple(tensor_slice.get_shape()) != WEIGHT_SHAPE:
                raise ValueError(f"Expected {name} with shape {WEIGHT_SHAPE}")
            rows = WEIGHT_SHAPE[0] // 2
            weight = tensor_slice[half * rows:(half + 1) * rows]
            if weight.dtype != torch.bfloat16 or not weight.is_contiguous():
                raise ValueError(f"Expected contiguous BF16 weights for {name}")
            buffer.copy_(weight.flatten())
    torch.cuda.synchronize()


def decode_timestamps(records):
    rows = []
    for block, waves in enumerate(records):
        starts, ends, cu_ids = [], [], set()
        for start_lo, start_hi, end_lo, end_hi, hw_id, *_ in waves:
            start = ((start_hi & 0xffffffff) << 32) | (start_lo & 0xffffffff)
            end = ((end_hi & 0xffffffff) << 32) | (end_lo & 0xffffffff)
            if hw_id == -1 or not 0 < start < end:
                raise RuntimeError(f"Invalid timestamps for CTA {block}")
            starts.append(start)
            ends.append(end)
            cu_ids.add((hw_id >> 8) & 255)
        if len(cu_ids) != 1:
            raise RuntimeError(f"Expected waves of CTA {block} on the same CU")
        rows.append({"cta": block, "cu": cu_ids.pop(), "start_tick": min(starts), "end_tick": max(ends)})
    origin = min(row["start_tick"] for row in rows)
    counts = Counter(row["cu"] for row in rows)
    for row in rows:
        row["start_tick"] -= origin
        row["end_tick"] -= origin
        row["ctas_on_cu"] = counts[row["cu"]]
    return rows


def measure(hip, args, stream, ptrs, kernels, events):
    samples = {blocks: [] for blocks in args.blocks}
    order = list(args.blocks)
    rng = random.Random(4421)
    for rep in range(args.warmup + args.samples):
        rng.shuffle(order)
        for blocks in order:
            elapsed = timed_launch(hip, kernels[blocks], blocks, ptrs, stream, *events)
            if rep >= args.warmup:
                samples[blocks].append(elapsed)
    return samples


def sample_timestamps(hip, args, stream, ptrs, probes):
    samples = {}
    for blocks, probe in probes.items():
        records = torch.empty((blocks, THREADS // WAVE_SIZE, 8), dtype=torch.int32, device="cuda")
        probe_args = ptrs + [POINTER(records.data_ptr())]
        trials = []
        for _ in range(9):
            records.fill_(-1)
            torch.cuda.synchronize()
            launch(hip, probe, blocks, probe_args, stream)
            check(hip, hip.hipStreamSynchronize(stream))
            trials.append(decode_timestamps(records.tolist()))
        samples[blocks] = trials
    return samples


def main():
    parser = argparse.ArgumentParser(description="gfx938 BF16 LM-head read-only bandwidth")
    parser.add_argument("--model-dir", type=Path, help="Local checkpoint. Omit for synthetic BF16 data.")
    parser.add_argument("--blocks", type=int, nargs="+", choices=BLOCK_COUNTS, default=(64, 72, 400))
    parser.add_argument("--depth", type=int, choices=DEPTHS, default=8)
    parser.add_argument("--cu-count", type=int, choices=(64, 72), help="Number of CUs to enable on the benchmark stream")
    parser.add_argument("--half", type=int, choices=(0, 1), help="Read only one half of the checkpoint")
    parser.add_argument("--samples", type=int, default=61)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timestamps", action="store_true", help="Collect separate instrumented CTA samples")
    parser.add_argument("--output", type=Path, help="Save raw samples and benchmark configuration as JSON")
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0 or len(set(args.blocks)) != len(args.blocks):
        parser.error("Use positive samples, nonnegative warmup and unique CTA counts")
    if torch.cuda.device_count() != 1:
        parser.error("Expose exactly one device with HIP_VISIBLE_DEVICES")
    props = torch.cuda.get_device_properties(0)
    if props.gcnArchName.split(":")[0] != "gfx938" or props.warp_size != WAVE_SIZE:
        parser.error("This benchmark requires gfx938 with wave64")
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < HALF_BYTES:
        parser.error(f"Visible device has {free_bytes} bytes free; one half requires {HALF_BYTES} bytes")

    hip = get_hip()
    bdf = c.create_string_buffer(32)
    check(hip, hip.hipDeviceGetPCIBusId(bdf, len(bdf), 0))
    halves = (0, 1) if args.half is None else (args.half,)
    torch.manual_seed(4421)
    buffer = torch.empty(HALF_BYTES // 2, dtype=torch.bfloat16, device="cuda")
    base = buffer.data_ptr()
    ptrs = [POINTER(base), POINTER(base + HALF_BYTES // 2)]
    results = []
    timestamps = []
    with ExitStack() as cleanup:
        stream, mask = open_stream(hip, props.multi_processor_count, args.cu_count, cleanup)
        events = (POINTER(), POINTER())
        for event in events:
            check(hip, hip.hipEventCreateWithFlags(c.byref(event), EVENT_DISABLE_SYSTEM_FENCE))
            cleanup.callback(lambda ev=event: check(hip, hip.hipEventDestroy(ev)))
        kernels = {n: kernel(hip, module_path(n, args.depth), cleanup) for n in args.blocks}
        probes = {n: kernel(hip, module_path(n, args.depth, True), cleanup) for n in args.blocks} if args.timestamps else {}
        for half in halves:
            prepare_weight(buffer, args.model_dir, half)
            samples = measure(hip, args, stream, ptrs, kernels, events)
            results.append({"half": half, "cases": {
                n: {"p50_us": median(s), "samples_us": s} for n, s in samples.items()
            }})
            if probes:
                timestamps.append({"half": half, "cases": sample_timestamps(hip, args, stream, ptrs, probes)})

    summary = []
    for blocks in args.blocks:
        total_us = sum(half["cases"][blocks]["p50_us"] for half in results)
        row = {"ctas": blocks, "threads": THREADS, "depth": args.depth, "cu_count": sum(w.bit_count() for w in mask),
               "read_bytes": HALF_BYTES * len(halves), "sum_half_p50_us": total_us}
        row["payload_TBps"] = row["read_bytes"] / total_us / 1e6
        print(json.dumps(row))
        summary.append(row)
    if args.output:
        data = {"gpu": props.name, "arch": props.gcnArchName, "pci_bdf": bdf.value.decode(),
                "torch_version": str(torch.__version__), "hip_version": torch.version.hip,
                "cu_mask": [f"0x{w:08x}" for w in mask], "weight_shape": WEIGHT_SHAPE,
                "model_dir": str(args.model_dir) if args.model_dir else None,
                "warmup": args.warmup, "samples": args.samples, "stage_bytes_per_cta": THREADS * 16,
                "summary": summary, "halves": results, "timestamp_traces": timestamps}
        args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

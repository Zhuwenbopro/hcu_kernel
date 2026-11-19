import argparse
import ctypes as c
import json
from pathlib import Path
from statistics import median

import torch


ROOT = Path(__file__).resolve().parent
WIDTH = 5120
WAVE_READ_CASES = ("xr_plain_window", "xr_c3_window", "xr_c3_batch", "xr_c3_serial")
EVENT_DISABLE_SYSTEM_FENCE = 0x20000000
POINTER = c.c_void_p
UINT = c.c_uint32
PP = c.POINTER(POINTER)


def get_hip():
    hip = c.CDLL("libamdhip64.so")
    hip.hipGetErrorString.argtypes = [c.c_int]
    hip.hipGetErrorString.restype = c.c_char_p
    hip.hipModuleLoad.argtypes = [PP, c.c_char_p]
    hip.hipModuleGetFunction.argtypes = [PP, POINTER, c.c_char_p]
    hip.hipModuleUnload.argtypes = [POINTER]
    hip.hipEventCreateWithFlags.argtypes = [PP, UINT]
    hip.hipEventDestroy.argtypes = [POINTER]
    hip.hipEventSynchronize.argtypes = [POINTER]
    hip.hipEventElapsedTime.argtypes = [c.POINTER(c.c_float), POINTER, POINTER]
    hip.hipExtModuleLaunchKernel.argtypes = [POINTER] + [UINT] * 6 + [
        c.c_size_t, POINTER, PP, PP, POINTER, POINTER, UINT,
    ]
    return hip


def check(hip, status):
    if status:
        raise RuntimeError(hip.hipGetErrorString(status).decode())


def get_module(hip, path):
    result = POINTER()
    check(hip, hip.hipModuleLoad(c.byref(result), bytes(path)))
    return result


def get_function(hip, module, symbol):
    result = POINTER()
    check(hip, hip.hipModuleGetFunction(c.byref(result), module, symbol.encode()))
    return result


def run_case(hip, fn, args, start, stop):
    blocks, threads = 80, 256
    params = (POINTER * len(args))(*(c.cast(c.byref(arg), POINTER) for arg in args))
    check(hip, hip.hipExtModuleLaunchKernel(fn, blocks * threads, 1, 1, threads, 1, 1,
                                            0, None, params, None, start, stop, 0))
    check(hip, hip.hipEventSynchronize(stop))
    result_ms = c.c_float()
    check(hip, hip.hipEventElapsedTime(c.byref(result_ms), start, stop))
    return result_ms.value * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=61)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--case", choices=WAVE_READ_CASES)
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0:
        parser.error("require samples >= 1 and warmup >= 0")

    props = torch.cuda.get_device_properties(0)
    if props.gcnArchName.split(":")[0] != "gfx936" or props.warp_size != 64:
        parser.error("require gfx936 wave64")
    torch.manual_seed(4421)
    inputs = [torch.randint(0, 65536, (4096, WIDTH), device="cuda", dtype=torch.int32)
              .to(torch.int16) for _ in range(2)]
    torch.cuda.synchronize()
    pointers = [POINTER(t.data_ptr()) for t in inputs]
    names = [args.case] if args.case else list(WAVE_READ_CASES)
    hip = get_hip()
    modules = {name: get_module(hip, ROOT / "build" / f"{name}.hsaco") for name in names}
    kernels = {name: get_function(hip, module, "read_xr") for name, module in modules.items()}
    start, stop = POINTER(), POINTER()
    for event in (start, stop):
        check(hip, hip.hipEventCreateWithFlags(c.byref(event), EVENT_DISABLE_SYSTEM_FENCE))
    samples = {name: [] for name in names}
    for rep in range(args.warmup + args.samples):
        shift = rep % len(names)
        for name in names[shift:] + names[:shift]:
            elapsed = run_case(hip, kernels[name], pointers, start, stop)
            if rep >= args.warmup:
                samples[name].append(elapsed)
    payload = sum(t.numel() * t.element_size() for t in inputs)
    for name, times in samples.items():
        us = median(times)
        print(json.dumps({"case": name, "p50_us": us, "read_bytes": payload,
                          "payload_TBps": payload / us / 1e6}))
    for event in (start, stop):
        check(hip, hip.hipEventDestroy(event))
    for module in modules.values():
        check(hip, hip.hipModuleUnload(module))


if __name__ == "__main__":
    main()

"""Run both gfx936 SiLU kernels on equivalent split and interleaved BF16 inputs."""
import argparse
import ctypes
from pathlib import Path

import torch


def load_kernels(path):
    library = ctypes.CDLL(str(path.resolve()))
    library.silu_mul_int8_init.argtypes = [ctypes.c_void_p]
    library.silu_mul_int8_init.restype = ctypes.c_int
    for launch in (library.silu_mul_int8_split, library.silu_mul_int8_interleaved):
        launch.argtypes = [ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 4
        launch.restype = ctypes.c_int
    return library


def check_status(status):
    if status != 0:
        raise RuntimeError(f"HIP call returned status {status}")


def run(launch, data, rows, width, stream):
    quantized = torch.empty((rows, width), device=data.device, dtype=torch.int8)
    scales = torch.empty((rows, 1), device=data.device, dtype=torch.float32)
    check_status(launch(rows, width, data.data_ptr(), quantized.data_ptr(), scales.data_ptr(), stream))
    return quantized, scales


def compare(actual, expected):
    quantized, scales = actual
    if not torch.equal(quantized, expected[0]):
        raise AssertionError("INT8 output mismatch")
    if not torch.equal(scales.view(torch.int32), expected[1].view(torch.int32)):
        raise AssertionError("FP32 scale bit mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path(__file__).resolve().parent / "build/libsilu_mul_int8.so")
    parser.add_argument("--rows", type=int, default=257)
    parser.add_argument("--width", type=int, choices=(5120, 8704, 17408), default=17408)
    parser.add_argument("--check-lightop", action="store_true", help="Also compare against the installed LightOP operator.")
    args = parser.parse_args()
    if args.rows < 0:
        parser.error("--rows must be nonnegative")

    torch.manual_seed(2026)
    gate = torch.randn(args.rows, args.width, device="cuda", dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    split = torch.cat((gate, up), dim=1)
    interleaved = torch.stack((gate, up), dim=-1).view(args.rows, 2 * args.width)

    kernels = load_kernels(args.library)
    stream = torch.cuda.current_stream().cuda_stream
    check_status(kernels.silu_mul_int8_init(stream))
    a = run(kernels.silu_mul_int8_split, split, args.rows, args.width, stream)
    b = run(kernels.silu_mul_int8_interleaved, interleaved, args.rows, args.width, stream)
    compare(b, a)
    if args.check_lightop and args.rows > 0:
        from lightop.fuse_silu_mul_quant import fuse_silu_mul_quant
        compare(a, fuse_silu_mul_quant(split))
    print(f"Bitwise checks passed for M={args.rows}, D={args.width}")


if __name__ == "__main__":
    main()

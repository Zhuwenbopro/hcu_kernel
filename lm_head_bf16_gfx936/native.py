"""Load the HIP GEMV and indexed rocBLAS wrappers built from kernels/."""

import ctypes
from functools import lru_cache
from pathlib import Path

_BUILD = Path(__file__).resolve().parent / "build"
_PTR = ctypes.c_void_p
_INT = ctypes.c_int


@lru_cache(maxsize=None)
def _entry(name, symbol, argtypes):
    library = _BUILD / f"{name}.so"
    if not library.is_file():
        raise RuntimeError(f"{library.name} is missing. Run python -m lm_head_bf16_gfx936.build")
    fn = getattr(ctypes.CDLL(str(library)), symbol)
    fn.argtypes = argtypes
    fn.restype = _INT
    return fn


def launch_gemv(x, w, y, stream):
    m = x.shape[0]
    name = "single" if m == 1 else "scalar"
    fn = _entry(name, "run", (_PTR, _PTR, _PTR, _PTR, _INT, _INT, _PTR))
    status = fn(x.data_ptr(), w.data_ptr(), y.data_ptr(), None, m, 4, stream)
    if status:
        raise RuntimeError(f"HIP GEMV M={m} returned status {status}")


def launch_indexed(index, x, w, y, stream):
    m = x.shape[0]
    fn = _entry("indexed", "head_indexed", (_INT, _INT, _INT, _PTR, _PTR, _PTR, _PTR))
    status = fn(index, m, x.device.index, stream, x.data_ptr(), w.data_ptr(), y.data_ptr())
    if status:
        raise RuntimeError(f"rocBLAS solution {index} for M={m} returned status {status}")


def launch_packed_x(index, x, w, y, tiles, stream):
    """Run disjoint 256-column groups directly into the output storage."""
    p = x.shape[1]
    fn = _entry("indexed", "head_packed_x", (_INT, _INT, _INT, _INT, _PTR, _PTR, _PTR, _PTR))
    columns = 0
    for tile_count in tiles:
        width = 256 * tile_count
        offset = columns * 2
        status = fn(index, p, width, x.device.index, stream, x.data_ptr(),
                    w.data_ptr() + offset, y.data_ptr() + offset)
        if status:
            raise RuntimeError(f"rocBLAS packed-X solution {index} for P={p} columns "
                               f"[{columns}, {columns + width}) returned status {status}")
        columns += width

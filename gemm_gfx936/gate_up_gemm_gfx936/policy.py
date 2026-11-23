"""Load the measured routing tables for K=5120 GateUp shards."""

from functools import lru_cache
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
K = 5120
SHARDS = (17408, 34816)
MAX_M = 4096
SMALL_M = 256


def check_m(m):
    if not 1 <= m <= MAX_M:
        raise ValueError(f"Expected 1 <= M <= {MAX_M}")


@lru_cache(maxsize=None)
def policy_sections(kind, n):
    path = ROOT / f"{kind}-policy.json"
    return json.loads(path.read_text(encoding="utf-8"))[str(n)]


@lru_cache(maxsize=None)
def routes(kind, n, mode="default"):
    """Expand [inclusive last M, route] records into an M-indexed tuple."""
    sections = policy_sections(kind, n)
    limit = SMALL_M if mode == "exact" else MAX_M
    result = [None]
    for end, name in sections[mode]:
        if not isinstance(end, int) or not len(result) <= end <= limit:
            raise ValueError(f"Invalid M range in {kind}-policy.json for N={n}, {mode}")
        result.extend([name] * (end + 1 - len(result)))
    if len(result) != limit + 1:
        raise ValueError(f"Incomplete policy in {kind}-policy.json for N={n}, {mode}")
    return tuple(result)


def bf16_override_ranges(n, section="overrides"):
    ranges = policy_sections("bf16", n).get(section, ())
    previous = 0
    for first, last, _ in ranges:
        if not previous < first <= last <= MAX_M:
            raise ValueError(f"Invalid BF16 {section} interval {first}..{last} for N={n}")
        previous = last
    return ranges


@lru_cache(maxsize=None)
def bf16_padded_routes(n):
    """Apply the measured active-M overrides to the default output path."""
    table = list(routes("bf16", n))
    for key, alignment in policy_sections("bf16", n).get("tail", {}).items():
        m = int(key)
        if (
            not SMALL_M < m <= MAX_M
            or alignment not in (2, 8, 16)
            or m % alignment == 0
        ):
            raise ValueError(f"Invalid BF16 tail padding for N={n}, M={m}")
        original = table[m]
        if not original.startswith("kn-blas-1-") or not original.endswith("-0"):
            raise ValueError(
                f"BF16 tail padding requires the transposed-X rocBLAS route for N={n}, M={m}"
            )
        parts = original.split("-")[3]
        table[m] = f"kn-tail-{alignment}-{parts}"
    for first, last, route in bf16_override_ranges(n):
        table[first : last + 1] = [route] * (last - first + 1)
    return tuple(table)

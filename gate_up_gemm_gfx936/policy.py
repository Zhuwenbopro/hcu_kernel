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
def routes(kind, n, mode="default"):
    """Expand [inclusive last M, route] records into an M-indexed tuple."""
    path = ROOT / f"{kind}-policy.json"
    sections = json.loads(path.read_text(encoding="utf-8"))
    limit = SMALL_M if mode == "exact" else MAX_M
    result = [None]
    for end, name in sections[str(n)][mode]:
        if not isinstance(end, int) or not len(result) <= end <= limit:
            raise ValueError(f"Invalid M range in {path.name} for N={n}, {mode}")
        result.extend([name] * (end + 1 - len(result)))
    if len(result) != limit + 1:
        raise ValueError(f"Incomplete policy in {path.name} for N={n}, {mode}")
    return tuple(result)

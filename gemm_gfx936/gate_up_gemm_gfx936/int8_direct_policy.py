"""Independent direct-X routes for the K=5120 TP2 shard."""

from functools import lru_cache

from .policy import policy_sections

N = 17408
MAX_M = 20480


def check_m(m):
    if not 1 <= m <= MAX_M:
        raise ValueError(f"Direct-X requires 1 <= M <= {MAX_M}")


@lru_cache(maxsize=1)
def direct_policy():
    section = policy_sections("int8-direct", N)
    routes = [None]
    for end, name in section["default"]:
        if not isinstance(end, int) or not len(routes) <= end <= MAX_M:
            raise ValueError("Invalid M interval in int8-direct-policy.json")
        if name not in section["kernels"]:
            raise ValueError(f"Unknown direct-X family {name}")
        routes.extend([name] * (end + 1 - len(routes)))
    if len(routes) != MAX_M + 1:
        raise ValueError(f"Incomplete direct-X M=1..{MAX_M} policy")
    if set(routes[1:]) != set(section["kernels"]):
        raise ValueError("Direct-X policy contains unused kernel families")
    return tuple(routes), section["kernels"]

"""Independent direct-X routes for the K=5120 TP2 shard."""

from functools import lru_cache

from .policy import policy_sections

N = 17408
MAX_M = 40960


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


@lru_cache(maxsize=1)
def grouped_policy():
    """Expand inclusive M ranges to band widths measured in BM128 tiles."""
    section = policy_sections("int8-direct", N)["grouped"]
    family = section["family"]
    routes, families = direct_policy()
    if family not in families:
        raise ValueError(f"Unknown grouped direct-X family {family}")
    widths = [0]
    for end, width in section["widths"]:
        first = len(widths)
        if not isinstance(end, int) or not first <= end <= MAX_M:
            raise ValueError("Invalid grouped M interval in int8-direct-policy.json")
        if not isinstance(width, int) or not 0 <= width <= (first + 127) // 128:
            raise ValueError(f"Invalid direct-X band width {width} at M={first}")
        if width and any(name != family for name in routes[first:end + 1]):
            raise ValueError("A grouped M interval selects an incompatible direct-X kernel")
        widths.extend([width] * (end - first + 1))
    if len(widths) != MAX_M + 1:
        raise ValueError("Incomplete direct-X band-width policy")
    return family, tuple(widths)

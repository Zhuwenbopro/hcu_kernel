"""Exact INT32 reference, independent CPU checks, and output overwrite guards."""
import numpy as np
import torch

SENTINEL = 0x5A5A5A5A

def reference(x, w):
    m, k = x.shape
    padded = max(32, (m + 31) // 32 * 32)
    xx = torch.cat([x, torch.zeros((padded - m, k), dtype=x.dtype, device=x.device)]) if padded != m else x
    return torch._int_mm(xx, w.t())[:m]

def cpu_check(x, w, y):
    m, _ = x.shape
    n = w.shape[0]
    rows = sorted({0, min(15, m - 1), min(16, m - 1), m // 2, m - 1})
    cols = sorted({min(c, n - 1) for c in (0, 1, 15, 16, 63, 64, n // 2, n - 1)})
    expected = x[rows].cpu().numpy().astype(np.int64) @ w[cols].cpu().numpy().astype(np.int64).T
    return int(np.count_nonzero(expected != y[rows][:, cols].cpu().numpy())), len(rows) * len(cols)

def guarded_output(m, n, device):
    backing = torch.full((m * n + 2048,), SENTINEL, device=device, dtype=torch.int32)
    return backing, backing[1024:-1024].view(m, n)

def guards_ok(backing):
    return bool((backing[:1024] == SENTINEL).all().item() and (backing[-1024:] == SENTINEL).all().item())

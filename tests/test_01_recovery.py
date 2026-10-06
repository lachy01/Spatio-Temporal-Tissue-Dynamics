"""Test 01 -- simulation recovery (spec 6.1).

Pass: every scalar parameter within 3 Monte Carlo standard errors of truth
across the replicates, and ||B_hat - B||_F / ||B||_F < 0.15.
"""
import numpy as np
import pytest
from mixture_report import fit_summary, size, write

from tissue_dynamics.mixture.data import load_config
from tissue_dynamics.mixture.fit import multistart
from tissue_dynamics.mixture.model import B_matrix
from tissue_dynamics.mixture.simulate import simulate_stage_b

SCALARS = ("b0", "theta", "delta", "a0", "psi")


@pytest.mark.slow
def test_01_recovery():
    cfg = load_config()
    n, reps = size(100_000, 30_000), size(20, 8)
    est, truths, bfro, last = {k: [] for k in SCALARS}, {}, [], None

    for rep in range(reps):
        data, truth, spec = simulate_stage_b(n=n, seed=rep, e_delta=np.log(4.0),
                                             mean_pi=0.15, rank=2)
        best, _ = multistart(data, spec, cfg, n_restarts=size(5, 3),
                             seed=100 * rep, verbose=False, maxiter=2000)
        for k in SCALARS:
            if k not in truth:              # psi: absent when the config drops oxygen
                continue
            est[k].append(float(best.params[k]))
            truths[k] = float(truth[k])
        bfro.append(float(np.linalg.norm(B_matrix(best.params) - truth["B"])
                          / np.linalg.norm(truth["B"])))
        last = (best, best.params, data, spec)

    lines, ok = [], True
    lines.append("| parameter | truth | mean estimate | MC SE | bias / SE |")
    lines.append("|---|---:|---:|---:|---:|")
    for k in SCALARS:
        if not est[k]:
            continue
        v = np.array(est[k])
        se = v.std(ddof=1) / np.sqrt(len(v))
        z = abs(v.mean() - truths[k]) / max(se, 1e-12)
        ok &= bool(z < 3.0)
        lines.append(f"| `{k}` | {truths[k]:.4f} | {v.mean():.4f} | {se:.4f} | {z:.2f} |")
    bf = np.array(bfro)
    b_ok = bool(bf.mean() < 0.15)
    ok &= b_ok
    lines += ["", f"- replicates: **{reps}** at n = **{n:,}**",
              f"- `||B_hat - B||_F / ||B||_F`: mean **{bf.mean():.4f}**, "
              f"max {bf.max():.4f} (threshold 0.15 -> {'pass' if b_ok else 'FAIL'})"]
    prov, _e, _f = fit_summary(last[0], last[1], last[2], last[3])
    lines += [""] + [f"{k}: **{v}**" for k, v in prov]
    write("Test 01 -- simulation recovery (spec 6.1)", lines, passed=ok)
    assert ok, "parameter recovery outside 3 MC SE, or B recovery worse than 0.15"

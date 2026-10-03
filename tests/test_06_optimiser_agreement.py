"""Test 06 -- optimiser agreement (spec 6.6).

EM to convergence versus L-BFGS on the same Stage A problem.
Pass: log-likelihoods agree to 6 significant figures; parameters to 4.
"""
import numpy as np
from mixture_report import write

from tissue_dynamics.mixture.data import load_config
from tissue_dynamics.mixture.em import run_em
from tissue_dynamics.mixture.fit import fit_lbfgs, poisson_start
from tissue_dynamics.mixture.model import ModelSpec, init_params
from tissue_dynamics.mixture.simulate import simulate_stage_b


def test_06_optimiser_agreement():
    cfg = load_config()
    data, _t, _sb = simulate_stage_b(n=20_000, seed=0, weight_ridge=0.0)
    spec = ModelSpec(stage="A", theta_free=True, d_own=0, d_nbr=0, df_area=4)
    ps = poisson_start(data, spec)
    p0 = init_params(spec, cfg, data=data, poisson_start=ps)

    pe, _hist, info = run_em(data, spec, p0, max_iter=500)
    rl = fit_lbfgs(data, spec, p0, maxiter=2000)

    rel_ll = abs(info["loglik"] - rl.loglik) / max(abs(rl.loglik), 1e-12)
    ll_ok = bool(rel_ll < 1e-6)                       # 6 significant figures

    rows, par_ok = [], True
    for k in ("b0", "theta", "delta"):
        a, b = float(pe[k]), float(rl.params[k])
        rel = abs(a - b) / max(abs(b), 1e-12)
        par_ok &= bool(rel < 1e-4)                    # 4 significant figures
        rows.append(f"| `{k}` | {a:.8f} | {b:.8f} | {rel:.2e} |")
    for j in range(len(np.asarray(pe["s"]))):
        a, b = float(np.asarray(pe["s"])[j]), float(np.asarray(rl.params["s"])[j])
        rel = abs(a - b) / max(abs(b), 1e-12)
        par_ok &= bool(rel < 1e-4)
        rows.append(f"| `s[{j}]` | {a:.8f} | {b:.8f} | {rel:.2e} |")

    lines = [f"- EM: loglik **{info['loglik']:.8f}** in {info['n_iter']} iterations",
             f"- L-BFGS: loglik **{rl.loglik:.8f}** in {rl.n_iter} iterations "
             f"(||grad||inf {rl.grad_inf:.1e})",
             f"- relative log-likelihood difference: **{rel_ll:.2e}** "
             f"(threshold 1e-6 -> {'pass' if ll_ok else 'FAIL'})", "",
             "| parameter | EM | L-BFGS | relative difference |", "|---|---:|---:|---:|"] + rows
    write("Test 06 -- optimiser agreement (spec 6.6)", lines, passed=bool(ll_ok and par_ok))
    assert ll_ok, f"log-likelihoods differ by {rel_ll:.2e}"
    assert par_ok, "parameters differ by more than 4 significant figures"

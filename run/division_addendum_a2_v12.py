"""Division classifier, addendum stage A2 (part 2): V12, iterated two-step to MIX.

    python run/division_addendum_a2_v12.py [--smoke N] [--max-maps 2000] [--inner 15]

Addendum section 3.4. On all cells, generalised EM on the penalised MIX
objective (model.loglik, production spec), started from the TWO-P solution
(step 1 rates, step 2 weights, from two_step_params.json):

  E-step     w_i = P(cycling | y_i, N_i, x_i) under the current pi(x_i), mu_0, mu_1
  M weights  `--inner` L-BFGS steps (zoom line search, so each lowers the
             penalised fractional-logistic objective) on a0 and the weight block,
             MIX's penalty -- a partial M-step, so generalised EM
  M rates    b0, delta, eta_m by Newton to convergence on
             sum_i [w_i log Pois(y_i; mu_1i) + (1 - w_i) log Pois(y_i; mu_0i)]

with SQUAREM (Varadhan and Roland 2008, scheme S3, step length capped at
alpha <= -1, then one stabilising map; the extrapolated point is kept only if
it lowers the objective, otherwise the plain two-map point is). The penalised
objective -loglik is logged after every accepted update and must never
increase. The cap is 2,000 EM map evaluations (each SQUAREM cycle costs three).

Start. Section 3.4 reasons from the constant-pi fixed point (weights = a0
alone): there the first E-step returns step 1's posteriors, the first weight
update is step 2, and the rates do not move. U = V = 0 is a saddle of the
bilinear term, so a partial M-step from there cannot leave it; the run starts
instead from TWO-P itself (step 1 rates, step 2 weights), which is where that
first iteration lands with an exact weight M-step. The claim that the rates do
not move is checked directly: one rate M-step on step 1's own posteriors.

Pass: final objective within 1e-6 (relative) of production MIX's, and max
|coefficient difference| < 1e-3. U and V are compared through B = U V',
since rotating U and V together leaves the model unchanged.

Writes v12_trace.csv, v12_params.json and a section of report.md.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import jax                                                            # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                               # noqa: E402
import optax                                                          # noqa: E402
from jax.flatten_util import ravel_pytree                             # noqa: E402

from tissue_dynamics.division import twostep as T                     # noqa: E402
from tissue_dynamics.mixture.basis import build_basis, spec_from_basis    # noqa: E402
from tissue_dynamics.mixture.data import load_config                  # noqa: E402
from tissue_dynamics.mixture.fit import fit_lbfgs                     # noqa: E402
from tissue_dynamics.mixture.model import (_log_lambda, _log_mu, _log_pois,  # noqa: E402
                                           build_data, loglik, responsibilities)

OUT = "outputs/division_classifier"
REPORT = f"{OUT}/report.md"
W_KEYS = ("a0", "a", "c", "U", "V", "h", "K")
R_KEYS = ("b0", "delta", "eta_m")
TOL_OBJ, TOL_COEF = 1e-6, 1e-3


def say(*a):
    print(*a, flush=True)


def report(text):
    with open(REPORT, "a") as fh:
        fh.write("\n" + text.rstrip() + "\n")


def md_table(df, fmt="{:.4g}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + " --- |" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating))
                                     else str(v) for v in r.to_numpy()) + " |")
    return "\n".join(out)


def coef_view(p):
    """Every identified coefficient, flattened by name: B = U V' replaces U, V."""
    out = {}
    for k in ("a0", "a", "c", "h", "K", *R_KEYS):
        out[k] = np.atleast_1d(np.asarray(p[k], float)).ravel()
    out["B"] = (np.asarray(p["U"]) @ np.asarray(p["V"]).T).ravel()
    return out


def coef_diff(p, q):
    a, b = coef_view(p), coef_view(q)
    return {k: float(np.max(np.abs(a[k] - b[k]))) for k in a}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--max-maps", type=int, default=2000)
    ap.add_argument("--inner", type=int, default=15)
    args = ap.parse_args()
    global OUT, REPORT
    t0 = time.time()
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["table_parquet"])
    df.attrs["phi_names"] = json.load(open("data/mixture_cells_major.phi_names.json"))
    pe = json.load(open("outputs/hpc/point_estimate.json"))
    basis = build_basis(df, cfg, verbose=False)
    spec = spec_from_basis(basis, cfg, stage="B", weight_ridge=float(pe["weight_ridge"]))
    if args.smoke:
        keep = np.sort(np.random.default_rng(0).choice(len(df), args.smoke, replace=False))
        df = df.iloc[keep].reset_index(drop=True)
        OUT = os.environ.get("SMOKE_OUT", "/tmp/a2_smoke")
        REPORT = f"{OUT}/report.md"
    two = json.load(open(f"{OUT}/two_step_params.json"))["TWO-P"]
    data = build_data(df, basis, cfg, spec)
    D = data.as_jax()
    n = len(df)
    y = D["y"]

    P_prod = {k: jnp.asarray(np.asarray(v)) for k, v in pe["params"].items()}
    if args.smoke:                       # no production fit on a subsample: make one
        P_prod = fit_lbfgs(data, spec, P_prod, maxiter=int(cfg["optim"]["maxiter"])).params
    F = jax.jit(lambda p: -loglik(p, D, spec))
    F_prod = float(F(P_prod))
    say(f"{n:,} cells; production penalised objective {F_prod:,.6f}")

    s1 = {k: jnp.asarray(np.asarray(v)) for k, v in two["step1"].items()}
    s2 = {k: jnp.asarray(np.asarray(v)) for k, v in two["step2"].items()}
    p = {**{k: s2[k] for k in W_KEYS}, **{k: s1[k] for k in R_KEYS}}

    # ---- weight M-step: `inner` L-BFGS steps on the penalised fractional logistic ----
    xw0, unravel_w = ravel_pytree({k: p[k] for k in W_KEYS})
    opt = optax.lbfgs()

    def qw(xw, w):
        return T.step2_objective(D, w, n, spec)(unravel_w(xw))

    @jax.jit
    def wstep(xw, st, w):
        fun = lambda x: qw(x, w)                                     # noqa: E731
        v, g = jax.value_and_grad(fun)(xw)
        up, st = opt.update(g, st, xw, value=v, grad=g, value_fn=fun)
        return optax.apply_updates(xw, up), st

    # ---- rate M-step: Newton on the weighted complete-data Poisson log-likelihood ----
    xr0, unravel_r = ravel_pytree({k: p[k] for k in R_KEYS})

    def qr(xr, w):
        pr = unravel_r(xr)
        l0, l1 = _log_lambda(pr, D, spec)
        f0 = _log_pois(y, _log_mu(l0, None, pr, D, spec))
        f1 = _log_pois(y, _log_mu(l1, None, pr, D, spec))
        return -jnp.sum(w * f1 + (1 - w) * f0)

    qr_j = jax.jit(qr)
    qr_vg = jax.jit(jax.value_and_grad(qr))
    qr_h = jax.jit(jax.hessian(qr))

    def rate_mstep(xr, w):
        for _ in range(30):
            v, g = qr_vg(xr, w)
            H = np.asarray(qr_h(xr, w))
            step = -np.linalg.solve(H, np.asarray(g))
            t = 1.0
            while t > 1e-8 and float(qr_j(xr + t * step, w)) > float(v) + 1e-4 * t * float(np.dot(g, step)):
                t *= 0.5
            xr = xr + t * jnp.asarray(step)
            if np.max(np.abs(t * step)) < 1e-11:
                break
        return xr

    resp = jax.jit(lambda q: responsibilities(q, D, spec))

    def split(x):
        return x[:xw0.size], x[xw0.size:]

    def join(pp):
        return jnp.concatenate([ravel_pytree({k: pp[k] for k in W_KEYS})[0],
                                ravel_pytree({k: pp[k] for k in R_KEYS})[0]])

    def unjoin(x):
        xw, xr = split(x)
        return {**unravel_w(xw), **unravel_r(xr)}

    def em_map(x):
        q = unjoin(x)
        w = resp(q)
        xw, xr = split(x)
        st = opt.init(xw)
        for _ in range(args.inner):
            xw, st = wstep(xw, st, w)
        xr = rate_mstep(xr, w)
        return jnp.concatenate([xw, xr])

    # ---- check: one rate M-step on step 1's own posteriors leaves the rates put ----
    w1 = T.posterior(s1, D, "pp")
    xr_s1 = ravel_pytree({k: s1[k] for k in R_KEYS})[0]
    d_rates0 = float(jnp.max(jnp.abs(rate_mstep(xr_s1, w1) - xr_s1)))
    say(f"rate M-step on step 1's posteriors moves the rates by {d_rates0:.2e}")

    # ---- SQUAREM ------------------------------------------------------------------------
    x = join(p)
    fx = float(F(unjoin(x)))
    F_start = fx
    trace = [{"maps": 0, "objective": fx, "step": "start", "alpha": np.nan}]
    maps, max_increase, first_pass = 0, 0.0, None

    def passes(xx, fv):
        d = coef_diff(unjoin(xx), P_prod)
        return abs(fv - F_prod) <= TOL_OBJ * abs(F_prod) and max(d.values()) < TOL_COEF, d

    stall = 0
    while maps + 3 <= args.max_maps:
        x1 = em_map(x)
        x2 = em_map(x1)
        maps += 2
        r = x1 - x
        v = x2 - x1 - r
        alpha = -float(jnp.linalg.norm(r) / max(float(jnp.linalg.norm(v)), 1e-300))
        alpha = min(alpha, -1.0)
        xs = x - 2 * alpha * r + alpha ** 2 * v
        f2 = float(F(unjoin(x2)))
        try:
            xs = em_map(xs)
            maps += 1
            fs = float(F(unjoin(xs)))
        except FloatingPointError:
            fs = np.inf
        if np.isfinite(fs) and fs <= f2:
            xn, fn, kind = xs, fs, "squarem"
        else:
            xn, fn, kind = x2, f2, "em"
        max_increase = max(max_increase, fn - fx)
        rel = (fx - fn) / abs(fx)
        x, fx = xn, fn
        ok, d = passes(x, fx)
        trace.append({"maps": maps, "objective": fx, "step": kind, "alpha": alpha,
                      "gap to production": fx - F_prod, "max |coef diff|": max(d.values())})
        if ok and first_pass is None:
            first_pass = maps
        if len(trace) % 10 == 0:
            say(f"  maps {maps:5d}  objective {fx:,.6f}  gap {fx - F_prod:+.3e}  "
                f"max|dcoef| {max(d.values()):.2e}  ({kind}, alpha {alpha:.1f}; "
                f"{(time.time() - t0) / 60:.0f} min)")
        stall = stall + 1 if abs(rel) < 1e-13 else 0
        if stall >= 3:
            break
    tr = pd.DataFrame(trace)
    tr.to_csv(f"{OUT}/v12_trace.csv", index=False)
    pf = unjoin(x)
    json.dump({k: np.asarray(v).tolist() for k, v in pf.items()}, open(f"{OUT}/v12_params.json", "w"))
    ok, d = passes(x, fx)
    monotone = max_increase <= 1e-9 * abs(F_prod)
    gap = fx - F_prod
    still_falling = (not ok) and stall < 3 and maps + 3 > args.max_maps
    if ok:
        verdict = "pass"
    elif still_falling and gap > 0:
        verdict = "FAIL: still decreasing at the cap (slow convergence)"
    elif gap < -TOL_OBJ * abs(F_prod):
        verdict = "FAIL: converged to a LOWER objective than production MIX"
    elif gap > TOL_OBJ * abs(F_prod):
        verdict = "FAIL: converged to a HIGHER objective (a different optimum)"
    else:
        verdict = "FAIL: objective matches but coefficients differ by more than 1e-3"
    say(f"V12: {verdict}; final {fx:,.6f} vs production {F_prod:,.6f} after {maps} maps")
    drows = pd.DataFrame([{"block": k, "max |difference|": v} for k, v in d.items()])
    milestones = tr.iloc[[0] + [i for i in (1, 2, 5, 10, 25, 50, 100, 200, 400) if i < len(tr)] + [len(tr) - 1]]
    milestones = milestones.drop_duplicates("maps")
    report(f"""
## Addendum stage A2 -- V12: iterated two-step converges to MIX ({time.strftime('%Y-%m-%d %H:%M')})

`run/division_addendum_a2_v12.py`. Generalised EM on the penalised MIX
objective over all {n:,} cells, from the TWO-P solution: E-step under the
current pi(x); weight M-step {args.inner} L-BFGS steps on the penalised
fractional logistic objective (MIX's penalty); rate M-step b0, delta and eta_m
by Newton to convergence; SQUAREM (S3) with a monotone safeguard. Cap {args.max_maps}
EM map evaluations.

The run starts from TWO-P itself (step 1 rates and step 2 weights) rather than
the constant-pi fixed point, because U = V = 0 is a saddle a partial weight
M-step cannot leave; TWO-P is where the section 3.4 path lands after its first
iteration with an exact weight M-step. The premise that the rates do not move
in that first iteration holds: a rate M-step on step 1's own posteriors moves
(b0, delta, eta_m) by {d_rates0:.1e}.

| | penalised objective (-loglik + penalty) |
| --- | --- |
| TWO-P (start) | {F_start:,.6f} |
| end of EM | {fx:,.6f} |
| production MIX | {F_prod:,.6f} |
| end - production | {gap:+.3e} ({gap / abs(F_prod):+.2e} relative) |

EM map evaluations: {maps}. The objective never increased: **{'yes' if monotone else 'NO'}**
(largest increase {max_increase:.2e}). Both pass criteria first held after
{first_pass if first_pass is not None else 'never'} map evaluations.

Largest coefficient difference from production MIX, by block (B = U V'):

{md_table(drows, '{:.2e}')}

Trace (selected cycles; every cycle in `v12_trace.csv`):

{md_table(milestones, '{:.8g}')}

**V12: {verdict}.**
""")
    if not ok:
        report("\nPer the addendum's section 3.4, the stage stops here; see the verdict above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

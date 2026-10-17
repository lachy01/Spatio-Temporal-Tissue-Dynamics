"""BASE-T-trunc calibrated against simulated truth, at a target real-data rate.

    python run/division_trunc_calibration.py [--reps 3] [--n 200000] [--target 0.10]

BASE-T has two settings: the truncation level of the technical null
(alpha_trim; "naive" = Stage 1's iteratively trimmed null) and the tail level
alpha_B (label = P(Y >= y | null) < alpha_B). This chooses them by how well the
label recovers the true cycling state Z in simulated sections, subject to the
label calling `--target` of the REAL cells cycling (user, 2026-10-17: ~10%).

Generators (cells, state, neighbourhood and technical covariates are the real
ones; pi_true and the component means mu_0, mu_1 are the production mixture's,
Z ~ Bernoulli(pi_true)):
  G1  y | Z ~ Poisson(mu_Z)                          (the production mixture)
  G2  as G1, cycling component NB(mu_1, alpha_1), alpha_1 matching the
      observed variance-to-mean ratio in the first focus type (Stage 6's G2)
  G4  y | Z ~ NB(mu_Z, alpha_Z), the two dispersions fitted to the real counts
      by maximum likelihood with pi, mu_0 and mu_1 held at the production fit.
G1 and G2 have Poisson non-cycling cells, so a truncated null fitted to them
collapses towards Poisson (A5: alpha0 ~0.04 at alpha_trim 0.05, against 0.30
on the real counts) and thresholds tuned there do not transfer. G4 is the
primary generator; the fidelity table checks it reproduces the real nulls'
alpha0 and label rates across alpha_trim.

For each generator and replicate (a random `--n` cells; Z shared by the three
generators of a replicate) every null is refitted on the simulated counts:
the naive robust null and the truncated null at each alpha_trim (from the
untrimmed fit, `param_tol` 1e-3). The label at each alpha_B is scored against
Z: rate, sensitivity, precision, MCC (the primary), F1, per-type fraction
error (mean over types of |log(rate / true fraction)|), contrast bias and
depth bias. Contrast bias: per type, the logit label-rate difference between
cells at or above / at or below the 90th / 10th percentile of neighbourhood
tumour score, minus the same difference in Z, averaged over types (0 = the
label carries the true neighbourhood effect; a ratio is unstable where the
true effect is near 0, as in the tumour type itself). Depth bias: sensitivity
in the top tercile of depth N over that in the bottom tercile (1 = a label
equally able to see cycling at every depth).

Selection rule (fixed before the results are read):
  1. per null, alpha_B* = the level at which the real label rate is `--target`
     (the target quantile of the real cells' tail probabilities);
  2. the null with the highest mean G4 MCC at its alpha_B*, the label then
     being that null at alpha_B*; nulls within two Monte Carlo standard
     errors of the best tie, and the smallest alpha_trim of the tied wins
     (the least aggressive truncation);
  3. reported beside it, not chosen: the unconstrained G4 MCC optimum over
     (null, alpha_B) and its real-data rate, and G1 / G2 at the same settings.

Writes outputs/division_classifier[_<ds>]/trunc_calibration/{real.csv,
sim.csv, choice.json, mcc_vs_rate.png} and a section of report.md.
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
from scipy.optimize import minimize                                   # noqa: E402
from scipy.special import gammaln                                     # noqa: E402

from tissue_dynamics.division import labels as L                      # noqa: E402
from tissue_dynamics.division import paths as DP                      # noqa: E402
from tissue_dynamics.division.simulation import g2_alpha              # noqa: E402
from tissue_dynamics.division.survival import nb_p_ge                 # noqa: E402
from tissue_dynamics.mixture.basis import spec_from_basis             # noqa: E402
from tissue_dynamics.mixture.data import load_config                  # noqa: E402
from tissue_dynamics.mixture.model import build_data, component_means, pi_hat  # noqa: E402
from tissue_dynamics.mixture.tasks import _load                       # noqa: E402

TRIMS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3)
NULLS = ("naive",) + tuple(f"trunc {t:g}" for t in TRIMS)
AB_GRID = np.unique(np.round(np.geomspace(0.002, 0.5, 81), 6))
PARAM_TOL = 1e-3
SEED = 20261017
GENS = ("G1", "G2", "G4")

say = lambda *a: print(*a, flush=True)                                # noqa: E731


def md_table(df, fmt="{:.4g}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating))
                                     else str(v) for v in r.to_numpy()) + " |")
    return "\n".join(out)


def log_nb(y, mu, a):
    r = 1.0 / a
    return (gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * np.log(r / (r + mu))
            + y * np.log(mu / (r + mu)))


def fit_g4(y, pi, m0, m1):
    """(alpha_0, alpha_1) maximising the two-component NB likelihood with the
    production pi and component means held fixed."""
    def nll(q):
        a0, a1 = np.exp(q)
        return -np.logaddexp(np.log1p(-pi) + log_nb(y, m0, a0), np.log(pi) + log_nb(y, m1, a1)).sum()
    r = minimize(nll, np.log([0.1, 0.1]), method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-4})
    pois = -np.logaddexp(np.log1p(-pi) + y * np.log(m0) - m0 - gammaln(y + 1),
                         np.log(pi) + y * np.log(m1) - m1 - gammaln(y + 1)).sum()
    return float(np.exp(r.x[0])), float(np.exp(r.x[1])), float(pois - r.fun)


def fit_nulls(y, X, off, m, tag=""):
    """Every null on these counts: {name: (mu, alpha, info)}."""
    out = {}
    nv = L.robust_null(y, X, off, m=m)
    out["naive"] = (nv["mu"], float(nv["alpha"]), {"theta_m": float(nv["theta_m"]),
                                                   "iterations": len(nv["trace"])})
    full = L.fit_nb_additive_null(y, X, off, m)
    for t in TRIMS:
        ft = L.truncated_null(y, X, off, m, t, full, param_tol=PARAM_TOL)
        out[f"trunc {t:g}"] = (ft["mu"], float(ft["alpha"]),
                               {"theta_m": float(ft["theta_m"]), "iterations": len(ft["trace"]),
                                "stopped_on": ft["stopped_on"],
                                "trimmed": float(ft["trace"][-1]["trimmed_frac"])})
        say(f"    {tag}trunc {t:g}: alpha0 {ft['alpha']:.4f}, {len(ft['trace'])} it ({ft['stopped_on']})")
    return out


def logit_rate(v):
    p = np.clip(np.mean(v), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def score(lab, Z, ct, types, tails, deep, shallow):
    tp = float(np.sum(lab & Z))
    pos, tru = float(lab.sum()), float(Z.sum())
    n = len(Z)
    sens = tp / tru
    prec = tp / pos if pos else np.nan
    fp, fn = pos - tp, tru - tp
    tn = n - tp - fp - fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    errs = []
    for t in types:
        s = ct == t
        if s.sum() >= 200 and Z[s].mean() > 0:
            errs.append(abs(np.log(max(lab[s].mean(), 1e-4) / Z[s].mean())))
    cb = [(logit_rate(lab[hi]) - logit_rate(lab[lo])) - (logit_rate(Z[hi]) - logit_rate(Z[lo]))
          for hi, lo in tails]
    return {"rate": pos / n, "sensitivity": sens, "precision": prec,
            "MCC": (tp * tn - fp * fn) / den if den > 0 else np.nan,
            "F1": 2 * tp / (pos + tru), "type error": float(np.mean(errs)),
            "contrast bias": float(np.mean(cb)),
            "depth bias": float(lab[deep & Z].mean() / max(lab[shallow & Z].mean(), 1e-6))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--target", type=float, default=0.10)
    ap.add_argument("--reuse-real", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config()
    OUT = f"{DP.out_dir()}/trunc_calibration"
    os.makedirs(OUT, exist_ok=True)
    df, basis = _load(cfg)
    pe = json.load(open(f"{DP.hpc_dir()}/point_estimate.json"))
    spec = spec_from_basis(basis, cfg, stage="B", weight_ridge=float(pe["weight_ridge"]))
    D = build_data(df, basis, cfg, spec).as_jax()
    P = {k: jnp.asarray(np.asarray(v)) for k, v in pe["params"].items()}
    pi = np.asarray(pi_hat(P, D, spec))
    m0, m1 = (np.asarray(v) for v in component_means(P, D, spec))
    y_real = np.asarray(D["y"], float)
    X, off, m = L.null_design(df, basis)
    names = df.attrs["phi_names"]
    types = DP.type_names(cfg)
    ct_all = L.cell_type(df, names).astype(str)
    focus = DP.focus_types(cfg)[0]
    tum_all = df[f"nbr_phi_{names.index(focus):02d}"].to_numpy(float)
    n_all = len(df)
    N_all = df["N"].to_numpy(float)
    say(f"{n_all:,} cells; mean pi {pi.mean():.4f}; focus type {focus}")

    # ---- generators ------------------------------------------------------------------
    a0_g4, a1_g4, gain = fit_g4(y_real, pi, m0, m1)
    fc = ct_all == focus
    vmr = float(y_real[fc].var() / y_real[fc].mean())
    a1_g2 = g2_alpha(pi[fc], m0[fc], m1[fc], vmr)
    say(f"G4 dispersions: non-cycling {a0_g4:.4f}, cycling {a1_g4:.4f} (log-lik +{gain:,.0f} "
        f"over Poisson + Poisson); G2 alpha_1 {a1_g2:.4f} ({focus} VMR {vmr:.3f})")

    # ---- real data -------------------------------------------------------------------
    real_path = f"{OUT}/real_nulls.npz"
    if a.reuse_real and os.path.exists(real_path):
        z = np.load(real_path, allow_pickle=True)
        p_real = z["p"].item()
        info_real = json.loads(str(z["info"]))
    else:
        say("real counts: fitting every null on all cells ...")
        nr = fit_nulls(y_real, X, off, m, tag="real ")
        p_real = {k: nb_p_ge(y_real, mu, al) for k, (mu, al, _) in nr.items()}
        info_real = {k: {"alpha0": al, **inf} for k, (_, al, inf) in nr.items()}
        np.savez(real_path, p=np.array(p_real, dtype=object), info=json.dumps(info_real))
    ab_star = {k: float(np.quantile(p, a.target)) for k, p in p_real.items()}
    rows = []
    for k in NULLS:
        p = p_real[k]
        r = {"null": k, "alpha0": info_real[k]["alpha0"], "alpha_B*": ab_star[k],
             "rate at alpha_B*": float(np.mean(p < ab_star[k]))}
        for ab in (0.01, 0.05, 0.1, 0.2):
            r[f"rate {ab:g}"] = float(np.mean(p < ab))
        for t in types:
            r[f"{t} at alpha_B*"] = float(np.mean(p[ct_all == t] < ab_star[k]))
        rows.append(r)
    real = pd.DataFrame(rows)
    real.to_csv(f"{OUT}/real.csv", index=False)
    say(real[["null", "alpha0", "alpha_B*", "rate at alpha_B*", "rate 0.1"]].to_string(index=False))

    # ---- simulation ------------------------------------------------------------------
    srows, fid = [], []
    for rep in range(a.reps):
        rng = np.random.default_rng(SEED + rep)
        idx = np.sort(rng.choice(n_all, min(a.n, n_all), replace=False))
        Z = rng.uniform(size=len(idx)) < pi[idx]
        ct, tum = ct_all[idx], tum_all[idx]
        tails = []
        for t in types:
            f = ct == t
            if f.sum() >= 2000:
                hq, lq = np.quantile(tum[f], [0.9, 0.1])
                tails.append((f & (tum >= hq), f & (tum <= lq)))
        Ns = N_all[idx]
        n_lo, n_hi = np.quantile(Ns, [1 / 3, 2 / 3])
        deep, shallow = Ns >= n_hi, Ns <= n_lo
        mu0, mu1 = m0[idx], m1[idx]
        ys = {"G1": np.where(Z, rng.poisson(mu1), rng.poisson(mu0)).astype(float),
              "G2": np.where(Z, L.sim_nb(mu1, a1_g2, rng) if a1_g2 > 0 else rng.poisson(mu1),
                             rng.poisson(mu0)).astype(float),
              "G4": np.where(Z, L.sim_nb(mu1, a1_g4, rng), L.sim_nb(mu0, a0_g4, rng)).astype(float)}
        for g in GENS:
            y = ys[g]
            say(f"  rep {rep} {g}: true fraction {Z.mean():.4f} ({time.time() - t0:.0f}s)")
            nulls = fit_nulls(y, X[idx], off[idx], m[idx], tag=f"{g} r{rep} ")
            for k, (mu, al, inf) in nulls.items():
                p = nb_p_ge(y, mu, al)
                fid.append({"generator": g, "rep": rep, "null": k, "alpha0": al,
                            **{f"rate {ab:g}": float(np.mean(p < ab)) for ab in (0.01, 0.05, 0.1, 0.2)}})
                levels = [(ab, "grid") for ab in AB_GRID] + [(ab_star[k], "alpha_B*")]
                for ab, kind in levels:
                    srows.append({"generator": g, "rep": rep, "null": k, "alpha_B": float(ab),
                                  "kind": kind, "true fraction": float(Z.mean()),
                                  **score(p < ab, Z, ct, types, tails, deep, shallow)})
        pd.DataFrame(srows).to_csv(f"{OUT}/sim.csv", index=False)
    sim = pd.DataFrame(srows)
    fid = pd.DataFrame(fid)

    # ---- selection -------------------------------------------------------------------
    metrics = ["rate", "sensitivity", "precision", "MCC", "F1", "type error", "contrast bias", "depth bias"]
    st = sim[sim["kind"] == "alpha_B*"]
    summ = st.groupby(["generator", "null"], sort=False)[metrics].agg(["mean", "std"])
    g4 = summ.loc["G4"]
    se = g4[("MCC", "std")] / np.sqrt(a.reps)
    best = g4[("MCC", "mean")].idxmax()
    tie_band = 2 * float(np.hypot(se[best], se).max())
    tied = [k for k in NULLS if g4.loc[k, ("MCC", "mean")] >= g4.loc[best, ("MCC", "mean")] - tie_band]
    order = {k: (-1 if k == "naive" else TRIMS.index(float(k.split()[1]))) for k in NULLS}
    tied_trunc = [k for k in tied if k != "naive"]
    pick = min(tied_trunc, key=order.get) if tied_trunc else best
    grid = sim[(sim["kind"] == "grid") & (sim["generator"] == "G4")]
    gm = grid.groupby(["null", "alpha_B"])[["MCC", "rate"]].mean().reset_index()
    u = gm.loc[gm["MCC"].idxmax()]
    u_real = float(np.mean(p_real[u["null"]] < u["alpha_B"]))
    choice = {"target real rate": a.target, "null": pick, "alpha_trim": None if pick == "naive" else float(pick.split()[1]),
              "alpha_B": ab_star[pick], "real rate": float(np.mean(p_real[pick] < ab_star[pick])),
              "G4 MCC": float(g4.loc[pick, ("MCC", "mean")]), "best by MCC": best,
              "tied within 2 MC SE": tied, "tie band": tie_band,
              "unconstrained G4 optimum": {"null": u["null"], "alpha_B": float(u["alpha_B"]),
                                           "MCC": float(u["MCC"]), "sim rate": float(u["rate"]),
                                           "real rate": u_real},
              "generators": {"G4 alpha_0": a0_g4, "G4 alpha_1": a1_g4, "G4 loglik gain": gain,
                             "G2 alpha_1": a1_g2}, "reps": a.reps, "n": a.n}
    json.dump(choice, open(f"{OUT}/choice.json", "w"), indent=1)
    say(json.dumps(choice, indent=1))

    # ---- figure: G4 MCC against REAL rate, one line per null ------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2, GRIDC = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
    SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b", "#07244a"]
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True, facecolor=SURF)
    ax.set_facecolor(SURF)
    for k in NULLS:
        s = gm[gm["null"] == k]
        xr = [np.mean(p_real[k] < ab) for ab in s["alpha_B"]]
        col = "#898781" if k == "naive" else SEQ[TRIMS.index(float(k.split()[1]))]
        ax.plot(xr, s["MCC"], color=col, lw=2.5 if k == pick else 1.5,
                ls="--" if k == "naive" else "-", label=k + ("  (chosen)" if k == pick else ""))
    ax.axvline(a.target, color=INK2, lw=1, ls=":")
    ax.plot([choice["real rate"]], [choice["G4 MCC"]], "o", ms=8, mfc=SEQ[5], mec=SURF, mew=2)
    ax.set_xlim(0, 0.3)
    ax.set_xlabel("label rate on the REAL section")
    ax.set_ylabel("MCC with the true state, G4 simulation")
    ax.grid(color=GRIDC, lw=0.8)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    ax.set_title("BASE-T: recovery of the simulated state against the real label rate", loc="left", fontsize=11)
    fig.savefig(f"{OUT}/mcc_vs_rate.png", dpi=170, facecolor=SURF)

    # ---- report ----------------------------------------------------------------------
    fidm = fid.groupby(["generator", "null"], sort=False)[["alpha0", "rate 0.05", "rate 0.1"]].mean().reset_index()
    fw = real[["null", "alpha0", "rate 0.05", "rate 0.1"]].rename(
        columns={"alpha0": "real alpha0", "rate 0.05": "real rate 0.05", "rate 0.1": "real rate 0.1"})
    for g in GENS:
        s = fidm[fidm["generator"] == g].set_index("null")
        fw[f"{g} alpha0"] = fw["null"].map(s["alpha0"])
        fw[f"{g} rate 0.1"] = fw["null"].map(s["rate 0.1"])
    tab = []
    for k in NULLS:
        r = {"null": k, "alpha_B*": ab_star[k]}
        for g in GENS:
            r[f"{g} MCC"] = f"{summ.loc[(g, k), ('MCC', 'mean')]:.3f} ± {summ.loc[(g, k), ('MCC', 'std')]:.3f}"
        for met in ("rate", "sensitivity", "precision", "type error", "contrast bias", "depth bias"):
            r[f"G4 {met}"] = float(summ.loc[("G4", k), (met, "mean")])
        tab.append(r)
    tab = pd.DataFrame(tab)
    per_type = real[["null"] + [f"{t} at alpha_B*" for t in types]].rename(
        columns={f"{t} at alpha_B*": t for t in types})
    ut = choice["unconstrained G4 optimum"]
    text = f"""
## BASE-T-trunc calibrated against simulated truth ({time.strftime('%Y-%m-%d %H:%M')})

`run/division_trunc_calibration.py`. Requested by the user (2026-10-17): choose BASE-T-trunc's
settings by how well the label recovers simulated truth, with about {a.target:.0%} of real cells
labelled. alpha_trim grid {', '.join(f'{t:g}' for t in TRIMS)} plus the naive Stage 1 null; each
null's alpha_B* puts the real label rate at {a.target:.0%}. {a.reps} replicates per generator of
{a.n:,} cells; the truncated nulls stop when the parameters settle (`param_tol` {PARAM_TOL:g}).
The production mixture's mean pi is {pi.mean():.4f}, so the target rate is close to its cycling
fraction.

**Generators.** G1 is the production Poisson mixture; G2 makes its cycling component NB
(alpha_1 {a1_g2:.3f}); G4 makes both components NB, with dispersions fitted to the real counts by
maximum likelihood with pi and the component means held fixed (non-cycling {a0_g4:.3f}, cycling
{a1_g4:.3f}; log-likelihood +{gain:,.0f} over the Poisson mixture).

### Fidelity: do the simulated sections reproduce the real nulls?

Mean over replicates. Only G4 reproduces the real alpha0 across alpha_trim: G1 and G2 have Poisson
non-cycling cells, so their truncated nulls collapse towards Poisson and their labels behave
differently. G4 is the primary generator.

{md_table(fw)}

### Recovery of Z at alpha_B* (real rate {a.target:.0%})

MCC mean ± sd over replicates for each generator; the remaining columns are G4 means. Type error:
mean over types of |log(label rate / true fraction)|. Contrast bias: label minus true logit
neighbourhood contrast (top against bottom decile of neighbourhood {focus} score), averaged over
types with at least 2,000 cells (0 = unbiased). Depth bias: sensitivity in the deepest third of
cells over the shallowest third (1 = unbiased).

{md_table(tab, fmt="{:.4f}")}

**Choice by the rule.** Best G4 MCC: {best}. Tied within two Monte Carlo SE (band
{tie_band:.4f}): {', '.join(tied)}. Chosen (smallest alpha_trim of the tied): **{pick}, alpha_B
{ab_star[pick]:.4f}**, real rate {choice['real rate']:.4f}, G4 MCC {choice['G4 MCC']:.3f}.

Unconstrained G4 optimum, for comparison: {ut['null']} at alpha_B {ut['alpha_B']:.4g}, MCC
{ut['MCC']:.3f}, simulated rate {ut['sim rate']:.4f}, real rate {ut['real rate']:.4f}.

![MCC against real rate]({os.path.relpath(OUT, DP.out_dir())}/mcc_vs_rate.png)

### Real label rate by type at alpha_B*

{md_table(per_type)}

Done in {(time.time() - t0) / 60:.0f} min. Not yet propagated to Stages 3-7; the user decides.
"""
    with open(f"{DP.out_dir()}/report.md", "a") as fh:
        fh.write(text)
    say(text)


if __name__ == "__main__":
    main()

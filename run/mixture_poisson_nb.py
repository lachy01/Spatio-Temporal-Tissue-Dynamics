"""Over-dispersed mixtures: MIX with NB components (sensitivities).

    MIX_DATASET=ovarian python run/mixture_poisson_nb.py [--restarts 3]               # Poisson + NB
    MIX_DATASET=ovarian python run/mixture_poisson_nb.py --family nbnb [--restarts 3] # NB + NB

--family nbnb (2026-10-16): both components NB, each with its own dispersion,

    y ~ pi(x) NB(rho + lambda_1, alpha_1) + (1 - pi(x)) NB(rho + lambda_0, alpha_0),

started from the Poisson + NB fit (outputs/poisson_nb[_<ds>]/params.json) at three
alpha_0 and from fresh starts; compared with all three others below, and written
to outputs/nb_nb[_<dataset>]/. Where the Xenium bundle has 10x's own annotation
(cell_groups.csv), the cycling calls are checked against its Proliferative cluster.

Poisson + NB (the default family):

    y ~ pi(x) NB(rho + lambda_1, alpha_1) + (1 - pi(x)) Pois(rho + lambda_0)

Everything else is the production specification (`model.cycling_nb`: rank,
penalty, MisTIC technical terms, theta = 1); alpha_1 is one shared, unpenalised
dispersion. Fitted on all cells from the production Poisson estimate at three
starting alpha_1 (0.05, 0.3, 1.0) and from `--restarts` fresh multistart starts;
the best is kept and agreement reported.

Compared, on the same cells, with the production Poisson mixture
(<hpc>/point_estimate.json) and the division comparison's full-data NB-c
(nb_params_nbc_full.json, the matched NB with MIX's penalty):

* log-likelihood (data term and penalised), the likelihood-ratio statistic for
  alpha_1 (boundary at 0: no p-value), AIC;
* hanging-rootogram misfit on 0-29 and 30+, expected zeros, randomised-PIT
  maximum deviation, overall and by type;
* e^delta, rate ratio, alpha_1, theta_m, cycling fraction overall and by type;
* push-through calibration, in sample: mean P(Y >= c_i) against the label rate
  on every BASE-T and BASE-O threshold of the division comparison's labels;
* the per-type neighbourhood contrast (logit pi, top against bottom decile of
  neighbourhood Tumour score) for both mixtures.

Writes outputs/poisson_nb[_<dataset>]/{report.md, comparison.json, params.json,
rootograms.png}.
"""
import argparse
import dataclasses
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
import matplotlib                                                     # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from scipy import stats                                               # noqa: E402
from scipy.stats import spearmanr                                     # noqa: E402

from tissue_dynamics.division import arms                             # noqa: E402
from tissue_dynamics.division import labels as L                      # noqa: E402
from tissue_dynamics.division import paths as DP                      # noqa: E402
from tissue_dynamics.division.survival import nb_p_ge, pois_p_ge      # noqa: E402
from tissue_dynamics.mixture.basis import spec_from_basis             # noqa: E402
from tissue_dynamics.mixture.data import load_config, phi_cell_types  # noqa: E402
from tissue_dynamics.mixture.fit import fit_lbfgs, multistart         # noqa: E402
from tissue_dynamics.mixture.model import (build_data, component_means,  # noqa: E402
                                           loglik, loglik_terms, pi_hat,
                                           responsibilities)
from tissue_dynamics.mixture.tasks import _load                       # noqa: E402

KMAX = 30
COL = {"Poisson mixture": "#2a78d6", "Poisson + NB mixture": "#eb6834", "NB-c (matched NB)": "#1baf7a",
       "NB + NB mixture": "#eda100"}
SHORT = {"Poisson mixture": "P", "Poisson + NB mixture": "PNB", "NB-c (matched NB)": "NB",
         "NB + NB mixture": "NBNB"}
INK, INK2 = "#0b0b0b", "#52514e"


def say(*a):
    print(*a, flush=True)


def md_table(df, fmt="{:.4g}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + " --- |" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating))
                                     else str(v) for v in r.to_numpy()) + " |")
    return "\n".join(out)


def mixture_pmf(pi, mu0, mu1, alpha1, kmax, alpha0=None):
    k = np.arange(kmax + 1)[None, :]
    if alpha0 is None:
        f0 = stats.poisson.pmf(k, mu0[:, None])
    else:
        r0 = 1.0 / alpha0
        f0 = stats.nbinom.pmf(k, r0, r0 / (r0 + mu0[:, None]))
    if alpha1 is None:
        f1 = stats.poisson.pmf(k, mu1[:, None])
    else:
        r = 1.0 / alpha1
        f1 = stats.nbinom.pmf(k, r, r / (r + mu1[:, None]))
    return pi[:, None] * f1 + (1 - pi[:, None]) * f0


def nb_pmf(mu, alpha, kmax):
    k = np.arange(kmax + 1)[None, :]
    r = 1.0 / alpha
    return stats.nbinom.pmf(k, r[:, None], (r / (r + mu))[:, None])


def rootogram(y, pmf):
    obs = np.array([np.sum(y == k) for k in range(KMAX)] + [np.sum(y >= KMAX)], float)
    exp_ = pmf[:, :KMAX].sum(0)                     # pmf holds 0..KMAX; tail = n - the rest
    exp_ = np.append(exp_, max(len(y) - exp_.sum(), 0.0))
    return obs, exp_, float(np.sum((np.sqrt(obs) - np.sqrt(exp_)) ** 2))


def pit_maxdev(y, cdf_y, cdf_ym1, rng, bins=10):
    u = cdf_ym1 + rng.uniform(size=len(y)) * (cdf_y - cdf_ym1)
    h, _ = np.histogram(u, bins=bins, range=(0, 1))
    return float(np.max(np.abs(h / (len(y) / bins) - 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--family", choices=("pnb", "nbnb"), default="pnb")
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config()
    ds = cfg.get("dataset")
    stem = {"pnb": "poisson_nb", "nbnb": "nb_nb"}[a.family]
    out = f"outputs/{stem}_{ds}" if ds else f"outputs/{stem}"
    pnb_dir = f"outputs/poisson_nb_{ds}" if ds else "outputs/poisson_nb"
    new_name = {"pnb": "Poisson + NB mixture", "nbnb": "NB + NB mixture"}[a.family]
    os.makedirs(out, exist_ok=True)
    df, basis = _load(cfg)
    pe = json.load(open(f"{DP.hpc_dir()}/point_estimate.json"))
    spec_p = spec_from_basis(basis, cfg, stage="B", weight_ridge=float(pe["weight_ridge"]))
    spec_pnb = dataclasses.replace(spec_p, cycling_nb=True)
    spec_nb = dataclasses.replace(spec_p, cycling_nb=True, noncycling_nb=a.family == "nbnb")
    data = build_data(df, basis, cfg, spec_p)
    D = data.as_jax()
    n = len(df)
    y = np.asarray(D["y"])
    maxiter = int(cfg["optim"]["maxiter"])
    say(f"{n:,} cells; {spec_nb.describe()}")

    # ---- fits ---------------------------------------------------------------------
    P0 = {k: jnp.asarray(np.asarray(v)) for k, v in pe["params"].items()}
    fits = []
    Ppnb = None
    if a.family == "nbnb":
        Ppnb = {k: jnp.asarray(np.asarray(v)) for k, v in json.load(open(f"{pnb_dir}/params.json")).items()}
        warm = [(f"warm from Poisson + NB, alpha0 start {a0:g}", {**Ppnb, "la0": jnp.asarray(float(np.log(a0)))})
                for a0 in (0.02, 0.1, 0.3)]
    else:
        warm = [(f"warm, alpha1 start {a1:g}", {**P0, "la1": jnp.asarray(float(np.log(a1)))})
                for a1 in (0.05, 0.3, 1.0)]
    for nm_, st in warm:
        r = fit_lbfgs(data, spec_nb, st, maxiter=maxiter, rel_tol=float(cfg["optim"]["rel_tol_loglik"]),
                      grad_tol=float(cfg["optim"]["grad_inf_tol"]))
        fits.append((nm_, r))
        extra = (f"; alpha0 {float(np.exp(r.params['la0'])):.4f}" if "la0" in r.params else "")
        say(f"  {nm_}: {r.summary()}; alpha1 {float(np.exp(r.params['la1'])):.4f}{extra}")
    if a.restarts:
        best_ms, res = multistart(data, spec_nb, cfg, n_restarts=a.restarts, seed=7, verbose=True,
                                  maxiter=maxiter)
        fits += [(f"multistart {i}", r) for i, r in enumerate(res)]
    ok = [(nm, r) for nm, r in fits if np.isfinite(r.loglik)]
    best_name, best = max(ok, key=lambda t: t[1].loglik)
    lls = np.array([r.loglik for _, r in ok])
    agree = float(np.mean(lls > lls.max() - 1e-6 * abs(lls.max())))
    Pn = best.params
    json.dump({k: np.asarray(v).tolist() for k, v in Pn.items()}, open(f"{out}/params.json", "w"))

    # ---- quantities per model -------------------------------------------------------
    names = df.attrs["phi_names"]
    cols = sorted(c for c in df.columns if c.startswith("phi_") and c[4:].isdigit())
    ct = phi_cell_types(df[cols].to_numpy(float), names, cfg["data"]).astype(str)
    types = DP.type_names(cfg)
    first_type = DP.focus_types(cfg)[0]                    # breast Cancer Epithelial, ovarian Tumour
    tum_nbr = df[f"nbr_phi_{names.index(first_type):02d}"].to_numpy(float)
    rng = np.random.default_rng(0)
    models = {}
    mods = [("Poisson mixture", P0, spec_p)]
    if Ppnb is not None:
        mods.append(("Poisson + NB mixture", Ppnb, spec_pnb))
    mods.append((new_name, Pn, spec_nb))
    for nm, P, spec in mods:
        pi = np.asarray(pi_hat(P, D, spec))
        w = np.asarray(responsibilities(P, D, spec))
        mu0, mu1 = (np.asarray(v) for v in component_means(P, D, spec))
        a1 = float(np.exp(np.asarray(P["la1"]))) if spec.cycling_nb else None
        a0 = float(np.exp(np.asarray(P["la0"]))) if spec.noncycling_nb else None
        pmf = mixture_pmf(pi, mu0, mu1, a1, KMAX, a0)

        def cdf(q, pi=pi, mu0=mu0, mu1=mu1, a1=a1, a0=a0):
            c1 = (stats.poisson.cdf(q, mu1) if a1 is None
                  else stats.nbinom.cdf(q, 1 / a1, (1 / a1) / (1 / a1 + mu1)))
            c0 = (stats.poisson.cdf(q, mu0) if a0 is None
                  else stats.nbinom.cdf(q, 1 / a0, (1 / a0) / (1 / a0 + mu0)))
            return pi * c1 + (1 - pi) * c0
        cy, cym1 = cdf(y), np.where(y > 0, cdf(y - 1), 0.0)
        models[nm] = dict(pi=pi, w=w, mu0=mu0, mu1=mu1, alpha1=a1, alpha0=a0, pmf=pmf, cy=cy, cym1=cym1,
                          ll=float(np.sum(np.asarray(loglik_terms(P, D, spec)))),
                          pll=float(loglik(P, D, spec)),
                          e_delta=float(np.exp(np.asarray(P["delta"]))),
                          theta_m=float(np.exp(np.asarray(P["eta_m"]))),
                          k=int(sum(np.size(np.asarray(v)) for v in P.values())))
    # matched NB (division NB-c, full data)
    nbc_path = f"{DP.out_dir()}/nb_params_nbc_full.json"
    if os.path.exists(nbc_path):
        Pc = {k: jnp.asarray(np.asarray(v)) for k, v in json.load(open(nbc_path)).items()}
        mu = np.exp(np.asarray(arms.nb_log_mu(Pc, D)))
        al = np.exp(np.asarray(arms.nb_log_alpha(Pc, D))) * np.ones(n)
        pmf = nb_pmf(mu, al, KMAX)
        r_ = 1 / al
        ncdf = lambda q: stats.nbinom.cdf(q, r_, r_ / (r_ + mu))        # noqa: E731
        llv = (stats.nbinom.logpmf(y, r_, r_ / (r_ + mu)))
        models["NB-c (matched NB)"] = dict(mu=mu, alpha=al, pmf=pmf, cy=ncdf(y),
                                           cym1=np.where(y > 0, ncdf(y - 1), 0.0),
                                           ll=float(llv.sum()), pll=np.nan, e_delta=np.nan, theta_m=float(np.exp(np.asarray(Pc["eta_m"]))),
                                           k=int(sum(np.size(np.asarray(v)) for v in Pc.values())))

    glob_rows, roots = [], {}
    for nm, m in models.items():
        obs, exp_, mis = rootogram(y, m["pmf"])
        roots[nm] = (obs, exp_)
        r = {"model": nm, "parameters": m["k"], "log-likelihood (data)": m["ll"],
             "AIC": 2 * m["k"] - 2 * m["ll"], "rootogram misfit": mis,
             "zeros expected": float(m["pmf"][:, 0].mean()), "zeros observed": float(np.mean(y == 0)),
             "PIT max deviation": pit_maxdev(y, m["cy"], m["cym1"], rng)}
        if "pi" in m:
            r.update({"e^delta": m["e_delta"], "rate ratio": float(np.exp(m["e_delta"])),
                      "alpha0": m["alpha0"] if m["alpha0"] is not None else np.nan,
                      "alpha1": m["alpha1"] if m["alpha1"] is not None else np.nan,
                      "theta_m": m["theta_m"], "cycling fraction (mean w)": float(m["w"].mean()),
                      "mean pi": float(m["pi"].mean())})
        glob_rows.append(r)
    glob = pd.DataFrame(glob_rows)
    ref = "Poisson + NB mixture" if a.family == "nbnb" else "Poisson mixture"
    lr = 2 * (models[new_name]["pll"] - models[ref]["pll"])

    by_type = []
    for t in types:
        mm = ct == t
        if not mm.any():
            continue
        row = {"type": t, "cells": int(mm.sum())}
        for nm, m in models.items():
            short = SHORT[nm]
            _, _, mis = rootogram(y[mm], m["pmf"][mm])
            row[f"misfit {short}"] = mis
            if "w" in m:
                row[f"w {short}"] = float(m["w"][mm].mean())
                lp = np.log(m["pi"] / (1 - m["pi"]))
                lo_q, hi_q = np.quantile(tum_nbr[mm], [0.1, 0.9])
                row[f"nbr contrast {short}"] = float(lp[mm & (tum_nbr >= hi_q)].mean() - lp[mm & (tum_nbr <= lo_q)].mean())
        by_type.append(row)
    by_type = pd.DataFrame(by_type)

    # ---- push-through calibration, in sample ---------------------------------------------
    lab = pd.read_parquet(f"{DP.out_dir()}/labels.parquet")
    if not np.array_equal(lab["cell_id"].astype(str).to_numpy(), df["cell_id"].astype(str).to_numpy()):
        sys.exit("labels.parquet is in a different cell order")
    cal = []
    thr = [(f"BASE-T alpha_B {a_:g}", f"cT_{L.ab_tag(a_)}", f"LT_{L.ab_tag(a_)}") for a_ in L.ALPHA_B_GRID] + \
          [(f"BASE-O Td {t_:g}", f"cO_{L.td_tag(t_)}", f"LO_{L.td_tag(t_)}") for t_ in L.TD_GRID]
    for name, ccol, lcol in thr:
        c = lab[ccol].to_numpy(float)
        rate = float(lab[lcol].mean())
        row = {"label": name, "label rate": rate}
        for nm, m in models.items():
            short = SHORT[nm]
            if "pi" in m:
                p1 = pois_p_ge(c, m["mu1"]) if m["alpha1"] is None else nb_p_ge(c, m["mu1"], m["alpha1"])
                p0 = pois_p_ge(c, m["mu0"]) if m["alpha0"] is None else nb_p_ge(c, m["mu0"], m["alpha0"])
                p = m["pi"] * p1 + (1 - m["pi"]) * p0
            else:
                p = nb_p_ge(c, m["mu"], m["alpha"])
            row[f"{short} p / rate"] = float(p.mean() / rate) if rate > 0 else np.nan
        cal.append(row)
    cal = pd.DataFrame(cal)

    # ---- figure -------------------------------------------------------------------------------
    ks = np.arange(KMAX + 1)
    fig, axes = plt.subplots(1, len(roots), figsize=(4.8 * len(roots), 4.2), sharey=True)
    for ax, (nm, (obs, exp_)) in zip(np.atleast_1d(axes), roots.items()):
        so, se = np.sqrt(obs), np.sqrt(exp_)
        ax.bar(ks, so, bottom=se - so, width=0.8, color=COL[nm], alpha=0.85, edgecolor="white", linewidth=1)
        ax.plot(ks, se, color=INK, lw=1.6, marker="o", ms=3)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_title(f"{nm}\nmisfit {np.sum((so - se) ** 2):,.0f}", color=INK, fontsize=10)
        ax.set_xlabel(f"division count (last bar = {KMAX}+)")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.grid(True, axis="y", color="#e1e0d9", lw=0.6)
    np.atleast_1d(axes)[0].set_ylabel("sqrt(frequency)")
    fig.suptitle("Hanging rootograms (bars hang from the expected frequency; a bar ending off zero is misfit)",
                 fontsize=10, color=INK2, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{out}/rootograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    tenx = ""
    cg = f"{cfg['data']['xenium_dir']}/cell_groups.csv"
    if os.path.exists(cg):
        from sklearn.metrics import roc_auc_score
        g = pd.read_csv(cg).set_index("cell_id")["group"].reindex(df["cell_id"].astype(str)).to_numpy()
        mix_models = [nm for nm in models if "pi" in models[nm]]
        rows = []
        for grp in pd.Series(g).value_counts().index:
            mm = g == grp
            rows.append({"10x group": grp, "cells": int(mm.sum()),
                         **{f"w {SHORT[nm]}": float(models[nm]["w"][mm].mean()) for nm in mix_models}})
        grp_tab = pd.DataFrame(rows)
        # 10x's tumour clusters (not "Tumor Associated" fibroblasts or endothelium)
        tum = np.isin(g, ["Tumor Cells", "Proliferative Tumor Cells", "SOX2-OT+ Tumor Cells",
                          "VEGFA+ Tumor Cells", "Malignant Cells Lining Cyst", "Inflammatory Tumor Cells"])
        truth = (g == "Proliferative Tumor Cells")[tum]
        chk = []
        for nm in mix_models:
            wv = models[nm]["w"][tum]
            call = wv > 0.5
            chk.append({"model": nm, "AUROC of w (Proliferative)": roc_auc_score(truth, wv),
                        "tumour cells called (w > 0.5)": float(call.mean()),
                        "calls in Proliferative": float(truth[call].mean()) if call.any() else np.nan,
                        "Proliferative called": float(call[truth].mean())})
        tenx = f"""
## Check against 10x's own annotation (cell_groups.csv)

Mean posterior cycling probability w by 10x group:

{md_table(grp_tab, '{:.3f}')}

Within 10x's tumour groups ({int(tum.sum()):,} cells), against its Proliferative cluster:

{md_table(pd.DataFrame(chk), '{:.3f}')}
"""
    fit_tab = pd.DataFrame([{"start": nm, "converged": r.converged, "penalised loglik": r.loglik,
                             "grad_inf": r.grad_inf, "iterations": r.n_iter,
                             "alpha1": float(np.exp(np.asarray(r.params["la1"]))) if np.isfinite(r.loglik) else np.nan,
                             "alpha0": (float(np.exp(np.asarray(r.params["la0"]))) if "la0" in r.params
                                        and np.isfinite(r.loglik) else np.nan),
                             "minutes": r.wall_time / 60} for nm, r in fits])
    json.dump({"global": glob_rows, "by_type": by_type.to_dict(orient="records"),
               "calibration": cal.to_dict(orient="records"), "lr": lr, "best_start": best_name,
               "agreement": agree}, open(f"{out}/comparison.json", "w"), indent=1, default=float)
    with open(f"{out}/report.md", "w") as fh:
        desc = ("its cycling component NB2(rho + lambda_1, alpha_1) instead of Poisson "
                "(`ModelSpec.cycling_nb`); one shared, unpenalised alpha_1" if a.family == "pnb" else
                "both components NB2, each with its own unpenalised dispersion (alpha_0 non-cycling, alpha_1 "
                "cycling; `cycling_nb`, `noncycling_nb`)")
        fh.write(f"""# {new_name} ({ds or 'breast'} section, {time.strftime('%Y-%m-%d %H:%M')})

`run/mixture_poisson_nb.py --family {a.family}`. The production mixture with {desc}; everything
else as production. All {n:,} cells. Best of {len(ok)} fits: {best_name}; {100 * agree:.0f}% of fits
within 1e-6 (relative) of the best penalised log-likelihood.

{md_table(fit_tab, '{:.6g}')}

## Fit

Likelihood-ratio statistic for {'alpha_0' if a.family == 'nbnb' else 'alpha_1'}, 2 x (penalised loglik
{SHORT[new_name]} - {SHORT[ref]}) = **{lr:,.1f}** on one parameter (boundary at 0, so no p-value). NB-c is the division comparison's full-data matched NB
(MIX's covariates and penalty in an NB mean, one dispersion).

{md_table(glob)}

Rootograms: `rootograms.png` (0-29 and a pooled 30+).

## By type

Misfit: rootogram misfit within the type. w: mean posterior cycling probability. Neighbourhood
contrast: mean logit pi in the type's top decile of neighbourhood {first_type} score minus its bottom decile.

{md_table(by_type, '{:.3f}')}

## Push-through calibration, in sample

Mean P(Y >= c_i) over the label rate, on the division comparison's labels (1 = calibrated). Fitted
on all cells (not out of fold), for all three models alike.

{md_table(cal, '{:.3f}')}
{tenx}""")
    say(open(f"{out}/report.md").read())
    say(f"done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()

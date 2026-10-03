"""Head-to-head: the negative binomial division model vs the Poisson mixture.

    python run/compare_nb_mixture.py

Three models on the SAME 678,885 cells, the SAME response (`y_count`, summed
Tirosh S+G2M counts) and the SAME frozen bases, so the only things that differ
are the ones being compared:

  nb_prev     the previous recommended NB division model: own + neighbour
              state, free log library size, an area spline and a density
              spline, all in the mean; one constant dispersion.
  nb_matched  every covariate the mixture uses -- including the rank-2 bilinear
              term, oxygen and the per-type density curves -- placed in the NB
              mean. Separates "richer covariates" from "mixture likelihood".
  mixture     the fitted Stage B mixture, loaded from point_estimate.json.

Likelihoods are compared on the DATA term only. The mixture estimate is
penalised MLE, so its unpenalised log-likelihood sits slightly below what an
unpenalised mixture could reach -- the comparison is conservative against it.
The NB fits are unpenalised.

Writes outputs/comparison/{metrics.csv, by_type.csv, summary.json, *.png}.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import jax                                                            # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                               # noqa: E402
import optax                                                          # noqa: E402
from jax.scipy.special import gammaln                                 # noqa: E402

import matplotlib                                                     # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

from tissue_dynamics.mixture.basis import build_basis, spec_from_basis    # noqa: E402
from tissue_dynamics.mixture.data import load_config                      # noqa: E402
from tissue_dynamics.mixture.model import (build_data, loglik_terms,      # noqa: E402
                                           pi_hat, rates)

OUT = "outputs/comparison"
KMAX = 30                      # rootogram support; the tail is pooled into "KMAX+"

# Reference palette slots 1-3: the three that validate all-pairs in both modes.
# Aqua sits below 3:1 on white, so every chart carries a legend AND direct labels.
COL = {"mixture": "#2a78d6", "nb_prev": "#eb6834", "nb_matched": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABEL = {"mixture": "Poisson mixture", "nb_prev": "NB (previous)",
         "nb_matched": "NB (matched covariates)"}
ORDER = ["nb_prev", "nb_matched", "mixture"]


# --------------------------------------------------------------------------
# NB fits
# --------------------------------------------------------------------------
def _nb_logpmf(y, mu, log_alpha):
    r = jnp.exp(-log_alpha)                                   # size = 1/alpha
    return (gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
            + r * (jnp.log(r) - jnp.log(r + mu)) + y * (jnp.log(mu) - jnp.log(r + mu)))


def _nb_logmu(p, D, matched):
    eta = p["b0"] + p["theta"] * D["logN"] + D["area"] @ p["s"]
    eta = eta + D["own"] @ p["a"] + D["nbr"] @ p["c"] + D["hbasis"] @ p["h"]
    if matched:
        eta = (eta + jnp.sum((D["own"] @ p["U"]) * (D["nbr"] @ p["V"]), axis=-1)
               + p["psi"] * D["oxygen"]
               + jnp.einsum("ij,jm,im->i", D["own"], p["K"], D["hbasis"]))
    return jnp.clip(eta, -30.0, 30.0)


def fit_nb(D, matched, maxiter=20000, seed=0):
    rng = np.random.default_rng(seed)
    d_own, d_nbr = D["own"].shape[1], D["nbr"].shape[1]
    n_h, n_s = D["hbasis"].shape[1], D["area"].shape[1]
    p = {"b0": jnp.asarray(float(np.log(np.mean(np.asarray(D["y"]))))),
         "theta": jnp.asarray(1.0), "s": jnp.zeros(n_s), "a": jnp.zeros(d_own),
         "c": jnp.zeros(d_nbr), "h": jnp.zeros(n_h), "log_alpha": jnp.asarray(0.0)}
    if matched:
        # U = V = 0 is a saddle (each block's gradient vanishes), so start small
        p["U"] = jnp.asarray(rng.normal(0, 0.01, (d_own, 2)))
        p["V"] = jnp.asarray(rng.normal(0, 0.01, (d_nbr, 2)))
        p["psi"] = jnp.asarray(0.0)
        p["K"] = jnp.zeros((d_own, n_h))

    def nll(q):
        return -jnp.sum(_nb_logpmf(D["y"], jnp.exp(_nb_logmu(q, D, matched)),
                                   q["log_alpha"]))

    vg = jax.value_and_grad(nll)
    opt = optax.lbfgs()
    state = opt.init(p)

    @jax.jit
    def step(q, st):
        v, g = vg(q)
        up, st = opt.update(g, st, q, value=v, grad=g, value_fn=nll)
        return optax.apply_updates(q, up), st, v, g

    prev, conv, t0 = np.inf, False, time.time()
    for it in range(maxiter):
        p, state, v, g = step(p, state)
        v = float(v)
        gi = float(jnp.max(jnp.abs(jax.flatten_util.ravel_pytree(g)[0])))
        if np.isfinite(prev) and abs(prev - v) < 1e-10 * abs(prev) and gi < 1e-4:
            conv = True
            break
        prev = v
    ll = -float(nll(p))
    k = sum(int(np.size(np.asarray(x))) for x in p.values())
    print(f"  {'nb_matched' if matched else 'nb_prev':<11} loglik {ll:,.2f}  k={k}  "
          f"alpha={float(np.exp(p['log_alpha'])):.4f}  iters={it + 1}  "
          f"|g|inf={gi:.2e}  converged={conv}  ({time.time() - t0:.0f}s)", flush=True)
    mu = np.asarray(jnp.exp(_nb_logmu(p, D, matched)))
    ll_i = np.asarray(_nb_logpmf(D["y"], jnp.asarray(mu), p["log_alpha"]))
    return dict(ll=ll, k=k, mu=mu, alpha=float(np.exp(p["log_alpha"])),
                converged=conv, iters=it + 1, grad_inf=gi, ll_i=ll_i)


# --------------------------------------------------------------------------
# Per-cell predictive distributions
# --------------------------------------------------------------------------
def nb_pmf_cdf(y, mu, alpha):
    r = 1.0 / alpha
    pr = r / (r + mu)
    ks = np.arange(KMAX)
    pmf = stats.nbinom.pmf(ks[None, :], r, pr[:, None])               # (n, KMAX)
    cdf_y = stats.nbinom.cdf(y, r, pr)
    cdf_ym1 = np.where(y > 0, stats.nbinom.cdf(y - 1, r, pr), 0.0)
    mean, var = mu, mu + alpha * mu ** 2
    return pmf, cdf_y, cdf_ym1, mean, var


def mix_pmf_cdf(y, mu0, mu1, pi):
    ks = np.arange(KMAX)
    pmf = ((1 - pi)[:, None] * stats.poisson.pmf(ks[None, :], mu0[:, None])
           + pi[:, None] * stats.poisson.pmf(ks[None, :], mu1[:, None]))
    cdf = lambda q: (1 - pi) * stats.poisson.cdf(q, mu0) + pi * stats.poisson.cdf(q, mu1)
    cdf_y = cdf(y)
    cdf_ym1 = np.where(y > 0, cdf(y - 1), 0.0)
    mean = (1 - pi) * mu0 + pi * mu1
    ey2 = (1 - pi) * (mu0 + mu0 ** 2) + pi * (mu1 + mu1 ** 2)
    return pmf, cdf_y, cdf_ym1, mean, ey2 - mean ** 2


def rootogram_counts(y, pmf):
    """Observed and expected frequencies on 0..KMAX-1, tail pooled into KMAX."""
    obs = np.array([np.sum(y == k) for k in range(KMAX)] + [np.sum(y >= KMAX)], float)
    exp_ = pmf.sum(0)
    # the pooled tail is n minus the rest; float error can leave it at -1e-12,
    # and sqrt of that is NaN
    exp_ = np.append(exp_, max(len(y) - exp_.sum(), 0.0))
    return obs, exp_


def rootogram_misfit(obs, exp_):
    return float(np.sum((np.sqrt(obs) - np.sqrt(exp_)) ** 2))


def randomised_pit(cdf_y, cdf_ym1, rng):
    # Non-randomised PIT is conservative for discrete outcomes; randomise.
    return cdf_ym1 + rng.uniform(size=len(cdf_y)) * (cdf_y - cdf_ym1)


# --------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["table_parquet"])
    df.attrs["phi_names"] = json.load(open("data/mixture_cells_major.phi_names.json"))
    basis = build_basis(df, cfg, verbose=False)
    pe = json.load(open("outputs/hpc/point_estimate.json"))
    spec = spec_from_basis(basis, cfg, stage="B", weight_ridge=float(pe["weight_ridge"]))
    data = build_data(df, basis, cfg, spec)
    D = data.as_jax()
    y = np.asarray(data.y)
    arche = df["archetype"].astype(str).to_numpy()
    print(f"{len(y):,} cells, mean count {y.mean():.3f}, var/mean {y.var() / y.mean():.2f}, "
          f"zeros {100 * np.mean(y == 0):.1f}%")

    # ---- mixture: data term only ----------------------------------------
    P = {k: jnp.asarray(np.asarray(v)) for k, v in pe["params"].items()}
    ll_mix_i = np.asarray(loglik_terms(P, D, spec))
    ll_mix = float(ll_mix_i.sum())
    k_mix = sum(int(np.size(np.asarray(v))) for v in pe["params"].values())
    lam0, lam1, _rho = (np.asarray(x) for x in rates(P, D, spec))
    pi = np.asarray(pi_hat(P, D, spec))
    print(f"  {'mixture':<11} loglik {ll_mix:,.2f}  k={k_mix}  "
          f"(penalised objective {pe['loglik']:,.2f})")

    # ---- NB fits ---------------------------------------------------------
    fits = {"nb_prev": fit_nb(D, matched=False), "nb_matched": fit_nb(D, matched=True)}
    fits["mixture"] = dict(ll=ll_mix, k=k_mix, converged=bool(pe["converged"]),
                           ll_i=ll_mix_i)

    # ---- predictive distributions ---------------------------------------
    rng = np.random.default_rng(0)
    pred = {}
    for m in ("nb_prev", "nb_matched"):
        pred[m] = nb_pmf_cdf(y, fits[m]["mu"], fits[m]["alpha"])
    pred["mixture"] = mix_pmf_cdf(y, lam0, lam1, pi)

    n = len(y)
    rows, by_type, pits, roots = [], [], {}, {}
    for m in ORDER:
        pmf, cy, cym1, mean, var = pred[m]
        u = randomised_pit(cy, cym1, rng)
        pits[m] = u
        hist = np.histogram(u, bins=20, range=(0, 1))[0] / (n / 20)
        obs, exp_ = rootogram_counts(y, pmf)
        roots[m] = (obs, exp_)
        ll, k = fits[m]["ll"], fits[m]["k"]
        rows.append(dict(model=m, loglik=ll, k=k, aic=-2 * ll + 2 * k,
                         bic=-2 * ll + k * np.log(n),
                         pit_max_dev=float(np.max(np.abs(hist - 1))),
                         pit_ks=float(stats.kstest(u, "uniform").statistic),
                         rootogram_misfit=rootogram_misfit(obs, exp_),
                         p0_obs=float(np.mean(y == 0)), p0_exp=float(pmf[:, 0].mean()),
                         converged=fits[m]["converged"]))
        for t in sorted(set(arche)):
            g = arche == t
            o_t, e_t = rootogram_counts(y[g], pmf[g])
            # law of total variance across the cells of a type
            v_model = float(np.mean(var[g]) + np.var(mean[g]))
            by_type.append(dict(model=m, archetype=t, n=int(g.sum()),
                                mean_obs=float(y[g].mean()), mean_model=float(mean[g].mean()),
                                vm_obs=float(y[g].var() / y[g].mean()),
                                vm_model=v_model / float(mean[g].mean()),
                                p0_obs=float(np.mean(y[g] == 0)),
                                p0_model=float(pmf[g, 0].mean()),
                                rootogram_misfit=rootogram_misfit(o_t, e_t),
                                loglik=float(fits[m]["ll_i"][g].sum())))

    # Within-component dispersion. A two-component Poisson mixture forces
    # var = mean INSIDE each component. Among cells the fit is confident are not
    # cycling (w < 0.05), the Pearson statistic sum((y - mu0)^2 / mu0) / n should
    # be ~1 if that holds; > 1 is extra-Poisson variation the mixture cannot
    # represent and an NB absorbs everywhere.
    from tissue_dynamics.mixture.model import responsibilities
    w = np.asarray(responsibilities(P, D, spec))
    disp = []
    for t in sorted(set(arche)) + ["all"]:
        g = (arche == t) if t != "all" else np.ones(n, bool)
        for comp, sel, mu_c in (("non-cycling (w<0.05)", g & (w < 0.05), lam0),
                                ("cycling (w>0.95)", g & (w > 0.95), lam1)):
            if sel.sum() < 200:
                continue
            pear = float(np.sum((y[sel] - mu_c[sel]) ** 2 / mu_c[sel]) / sel.sum())
            disp.append(dict(archetype=t, component=comp, n=int(sel.sum()),
                             pearson_dispersion=pear))
    dp = pd.DataFrame(disp)
    dp.to_csv(f"{OUT}/within_component_dispersion.csv", index=False)
    print("\nwithin-component Pearson dispersion (1 = Poisson holds):")
    print(dp.round(3).to_string(index=False))

    met = pd.DataFrame(rows)
    met["delta_aic_vs_mixture"] = met["aic"] - float(met.loc[met.model == "mixture", "aic"].iloc[0])
    met["delta_bic_vs_mixture"] = met["bic"] - float(met.loc[met.model == "mixture", "bic"].iloc[0])
    met.to_csv(f"{OUT}/metrics.csv", index=False)
    bt = pd.DataFrame(by_type)
    bt.to_csv(f"{OUT}/by_type.csv", index=False)
    print("\n" + met.round(4).to_string(index=False))
    print("\n" + bt.pivot(index="archetype", columns="model",
                          values="rootogram_misfit")[ORDER].round(1).to_string())
    lp = bt.pivot(index="archetype", columns="model", values="loglik")[ORDER]
    lp["nb_prev - mixture"] = lp["nb_prev"] - lp["mixture"]
    lp["nb_matched - mixture"] = lp["nb_matched"] - lp["mixture"]
    print("\nlog-likelihood by cell type (positive gap = NB fits better):")
    print(lp.round(1).to_string())

    summary = dict(n=n, y_mean=float(y.mean()), y_var_mean=float(y.var() / y.mean()),
                   zeros=float(np.mean(y == 0)),
                   alpha={m: fits[m]["alpha"] for m in ("nb_prev", "nb_matched")},
                   nb_converged={m: fits[m]["converged"] for m in ("nb_prev", "nb_matched")},
                   nb_iters={m: fits[m]["iters"] for m in ("nb_prev", "nb_matched")},
                   mixture_penalised_loglik=pe["loglik"],
                   mixture_rate_ratio=float(np.exp(pe["e_delta"])),
                   mixture_mean_pi=float(pi.mean()))
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)

    plot_all(y, arche, roots, pits, bt, met)
    print(f"\nwrote {OUT}/")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_all(y, arche, roots, pits, bt, met):
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelcolor": INK2,
                         "text.color": INK, "figure.facecolor": "white"})

    # 1. hanging rootograms ------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    ks = np.arange(KMAX + 1)
    for ax, m in zip(axes, ORDER):
        obs, exp_ = roots[m]
        so, se = np.sqrt(obs), np.sqrt(exp_)
        ax.bar(ks, so, bottom=se - so, width=0.8, color=COL[m], alpha=0.85,
               edgecolor="white", linewidth=1)
        ax.plot(ks, se, color=INK, lw=1.6, marker="o", ms=3)
        ax.axhline(0, color=INK2, lw=1)
        _style(ax)
        mis = float(met.loc[met.model == m, "rootogram_misfit"].iloc[0])
        ax.set_title(f"{LABEL[m]}\nmisfit {mis:,.0f}", color=INK)
        ax.set_xlabel(f"division count (last bar = {KMAX}+)")
    axes[0].set_ylabel("sqrt(frequency)")
    fig.suptitle("Hanging rootograms: bars hang from the model's expected frequency; "
                 "a bar ending off zero is misfit", fontsize=10, color=INK2, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/rootograms.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 2. rootogram residual on one axis -----------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = 0.27
    for i, m in enumerate(ORDER):
        obs, exp_ = roots[m]
        ax.bar(ks + (i - 1) * w, np.sqrt(obs) - np.sqrt(exp_), width=w,
               color=COL[m], label=LABEL[m], edgecolor="white", linewidth=0.6)
    ax.axhline(0, color=INK2, lw=1)
    _style(ax)
    ax.set_xlabel(f"division count (last group = {KMAX}+)")
    ax.set_ylabel("sqrt(observed) - sqrt(expected)")
    ax.set_title("Rootogram residuals, all three models on one scale", color=INK)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/rootogram_residuals.png", dpi=160)
    plt.close(fig)

    # 3. randomised PIT ----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), sharey=True)
    for ax, m in zip(axes, ORDER):
        h = np.histogram(pits[m], bins=20, range=(0, 1))[0] / (len(pits[m]) / 20)
        ax.bar(np.arange(20) / 20 + 0.025, h, width=0.045, color=COL[m],
               edgecolor="white", linewidth=1)
        ax.axhline(1, color=INK, lw=1.3, ls="--")
        _style(ax)
        dev = float(met.loc[met.model == m, "pit_max_dev"].iloc[0])
        ax.set_title(f"{LABEL[m]}\nmax |density - 1| = {dev:.3f}", color=INK)
        ax.set_xlabel("randomised PIT")
    axes[0].set_ylabel("density (uniform = 1)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/pit.png", dpi=160)
    plt.close(fig)

    # 4. dispersion by cell type -------------------------------------------
    types = (bt[bt.model == "mixture"].sort_values("vm_obs")["archetype"].tolist())
    fig, ax = plt.subplots(figsize=(9, 5.2))
    yy = np.arange(len(types))
    obs_vm = [float(bt[(bt.model == "mixture") & (bt.archetype == t)]["vm_obs"].iloc[0])
              for t in types]
    ax.scatter(obs_vm, yy, s=90, facecolor="white", edgecolor=INK, linewidth=2,
               zorder=5, label="observed")
    for j, m in enumerate(ORDER):
        v = [float(bt[(bt.model == m) & (bt.archetype == t)]["vm_model"].iloc[0])
             for t in types]
        ax.scatter(v, yy + (j - 1) * 0.18, s=46, color=COL[m], zorder=4,
                   edgecolor="white", linewidth=1, label=LABEL[m])
    ax.axvline(1, color=INK2, lw=1, ls=":")
    ax.text(1.02, len(types) - 0.45, "Poisson (var = mean)", color=INK2, fontsize=8)
    ax.set_xscale("log")
    ax.set_yticks(yy)
    ax.set_yticklabels(types)
    _style(ax)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("variance / mean of the division count (log scale)")
    ax.set_title("Where the over-dispersion lives, and which model reproduces it",
                 color=INK)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/dispersion_by_type.png", dpi=160)
    plt.close(fig)

    # 5. rootogram misfit by cell type -------------------------------------
    piv = bt.pivot(index="archetype", columns="model", values="rootogram_misfit")[ORDER]
    piv = piv.loc[types]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for j, m in enumerate(ORDER):
        ax.barh(yy + (j - 1) * 0.26, piv[m].to_numpy(), height=0.24, color=COL[m],
                label=LABEL[m], edgecolor="white", linewidth=0.6)
    ax.set_yticks(yy)
    ax.set_yticklabels(types)
    ax.set_xscale("log")
    _style(ax)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("rootogram misfit within cell type (log scale; lower is better)")
    ax.set_title("Fit to the count distribution, by cell type", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/rootogram_by_type.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

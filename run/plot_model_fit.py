"""Goodness-of-fit figures for the count models, one section at a time.

    python run/plot_model_fit.py                         # breast
    MIX_DATASET=ovarian python run/plot_model_fit.py     # ovarian

Models (all on the section's analysis cells, the same y = summed Tirosh
counts):

  NB            the better (by AIC) of the division comparison's two full-data
                NB fits: NB-c (the mixture's covariates and penalty in an NB
                mean, one dispersion) and NB-d (the same with a dispersion
                submodel on the projected own state). Read from
                outputs/division_classifier[_<ds>]/nb_params_nb{c,d}_full.json.
  Poisson mix.  the production point estimate (outputs/hpc[_<ds>]/point_estimate.json).
  NB + NB mix.  where fitted (ovarian only, outputs/nb_nb_<ds>/params.json): the
                production mixture with both components NB2.

Figures (outputs/figures/model_fit/<ds>_*.png), each with the models side by side:

  rootogram     hanging rootogram on 0..29 and a pooled 30+: bars of sqrt(observed)
                hang from the curve of sqrt(expected); a bar ending above or below
                the zero line is a count the model under- or over-predicts.
  pit           randomised PIT histogram (20 bins; uniform if calibrated), with the
                KS distance. The same uniform draws for every model.
  residuals     randomised quantile residuals r = Phi^-1(PIT) (Dunn and Smyth):
                normal QQ plot, mean residual by depth (N) ventile, and mean
                residual by cell type, each with +/- 2 SE.
  fitted_map    observed count and each model's fitted mean E[y | x], averaged
                in 50 um squares on one colour scale.
  residual_map  mean quantile residual in 50 um squares (diverging, centred at 0),
                with Moran's I of the squares (rook neighbours).

Also writes outputs/figures/model_fit/<ds>_metrics.csv. Colours: the reference
palette's first three categorical slots, its blue sequential ramp, and a
blue-grey-red diverging scale.
"""
import dataclasses
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import jax                                                            # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                               # noqa: E402
import matplotlib                                                     # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.colors import LinearSegmentedColormap                 # noqa: E402

from tissue_dynamics.division import arms                             # noqa: E402
from tissue_dynamics.division import paths as DP                      # noqa: E402
from tissue_dynamics.mixture.basis import build_basis, spec_from_basis  # noqa: E402
from tissue_dynamics.mixture.data import load_config, phi_cell_types  # noqa: E402
from tissue_dynamics.mixture.model import build_data, component_means, pi_hat  # noqa: E402

OUT = "outputs/figures/model_fit"
KMAX = 30
BIN_UM = 50.0
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
COL = {"NB": "#eb6834", "Poisson mixture": "#2a78d6", "NB + NB mixture": "#1baf7a"}
SEQ = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV = ["#0d366b", "#256abf", "#6da7ec", "#cde2fb", "#f0efec", "#fbd5d3", "#f29b98", "#e34948", "#9e2423"]
TITLE = {"breast": "Breast section", "ovarian": "Ovarian section"}


def say(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------
# predictive distributions: chunked, so no n x KMAX matrix is ever held
# --------------------------------------------------------------------------
def nb_parts(mu, alpha):
    r = 1.0 / np.broadcast_to(alpha, mu.shape)
    return r, r / (r + mu)


def dist_summaries(y, comps, rng_u, chunk=100_000):
    """comps: list of (weight, family, mean, alpha) with per-cell arrays; the
    predictive pmf is sum_k weight_k f_k. Returns expected frequencies on
    0..KMAX-1 plus the pooled tail, randomised PIT, and the mean."""
    n = len(y)
    expected = np.zeros(KMAX)
    cdf_y, cdf_ym1 = np.zeros(n), np.zeros(n)
    ks = np.arange(KMAX)
    for s in range(0, n, chunk):
        sl = slice(s, min(s + chunk, n))
        ys = y[sl]
        for w, fam, m, a in comps:
            ws, ms = w[sl], m[sl]
            if fam == "pois":
                pmf = stats.poisson.pmf(ks[None, :], ms[:, None])
                c_y, c_m = stats.poisson.cdf(ys, ms), stats.poisson.cdf(ys - 1, ms)
            else:
                r, p = nb_parts(ms, a[sl] if np.ndim(a) else a)
                pmf = stats.nbinom.pmf(ks[None, :], r[:, None], p[:, None])
                c_y, c_m = stats.nbinom.cdf(ys, r, p), stats.nbinom.cdf(ys - 1, r, p)
            expected += (ws[:, None] * pmf).sum(0)
            cdf_y[sl] += ws * c_y
            cdf_ym1[sl] += ws * np.where(ys > 0, c_m, 0.0)
    expected = np.append(expected, max(n - expected.sum(), 0.0))
    u = cdf_ym1 + rng_u * (cdf_y - cdf_ym1)
    mean = sum(w * m for w, _, m, _ in comps)
    return expected, np.clip(u, 1e-12, 1 - 1e-12), mean


def log_lik(y, comps):
    lp = None
    for w, fam, m, a in comps:
        if fam == "pois":
            l_ = stats.poisson.logpmf(y, m)
        else:
            r, p = nb_parts(m, a)
            l_ = stats.nbinom.logpmf(y, r, p)
        t = np.log(np.clip(w, 1e-300, None)) + l_
        lp = t if lp is None else np.logaddexp(lp, t)
    return float(lp.sum())


# --------------------------------------------------------------------------
def grid_mean(v, xb, yb, shape, min_cells=3):
    flat = yb * shape[1] + xb
    cnt = np.bincount(flat, minlength=shape[0] * shape[1])
    s = np.bincount(flat, weights=v, minlength=shape[0] * shape[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.where(cnt >= min_cells, s / cnt, np.nan)
    return g.reshape(shape)


def morans_i(g):
    """Moran's I of a grid with NaN holes, rook neighbours, row-standardised off."""
    v = g - np.nanmean(g)
    ok = np.isfinite(v)
    num, wsum = 0.0, 0.0
    for dy, dx in ((0, 1), (1, 0)):
        a = v[: v.shape[0] - dy, : v.shape[1] - dx]
        b = v[dy:, dx:]
        m = np.isfinite(a) & np.isfinite(b)
        num += 2 * np.sum(a[m] * b[m])
        wsum += 2 * m.sum()
    return float(ok.sum() / wsum * num / np.nansum(v[ok] ** 2))


def style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURF)


def map_style(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")
    ax.set_facecolor(SURF)


def main():
    cfg = load_config()
    ds = cfg.get("dataset", "breast")
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK2, "figure.facecolor": SURF,
                         "savefig.facecolor": SURF})
    df = pd.read_parquet(cfg["data"]["table_parquet"])
    names = json.load(open(DP.phi_names_path(cfg)))
    df.attrs["phi_names"] = names
    basis = build_basis(df, cfg, verbose=False)
    pe = json.load(open(f"{DP.hpc_dir()}/point_estimate.json"))
    spec = spec_from_basis(basis, cfg, stage="B", weight_ridge=float(pe["weight_ridge"]))
    data = build_data(df, basis, cfg, spec)
    D = data.as_jax()
    y = np.asarray(data.y, float)
    n = len(y)
    N = df["N"].to_numpy(float)
    cols = sorted(c for c in df.columns if c.startswith("phi_") and c[4:].isdigit())
    ctype = phi_cell_types(df[cols].to_numpy(float), names, cfg["data"]).astype(str)
    xs, ys_ = df["x"].to_numpy(float), df["y"].to_numpy(float)
    say(f"{TITLE[ds]}: {n:,} cells, mean y {y.mean():.3f}")

    # ---- models -------------------------------------------------------------------
    models, meta = {}, {}
    nb_choice = []
    for tag in ("nbc", "nbd"):
        p = {k: jnp.asarray(np.asarray(v)) for k, v in
             json.load(open(f"{DP.out_dir()}/nb_params_{tag}_full.json")).items()}
        mu = np.exp(np.asarray(arms.nb_log_mu(p, D)))
        al = np.exp(np.asarray(arms.nb_log_alpha(p, D))) * np.ones(n)
        comps = [(np.ones(n), "nb", mu, al)]
        ll = log_lik(y, comps)
        k = sum(int(np.size(np.asarray(v))) for v in p.values())
        nb_choice.append((-2 * ll + 2 * k, tag, comps, ll, k))
        say(f"  NB-{tag[-1]}: loglik {ll:,.1f}, k {k}")
    aic, tag, comps, ll, k = min(nb_choice, key=lambda t: t[0])
    models["NB"] = comps
    meta["NB"] = {"source": f"{DP.out_dir()}/nb_params_{tag}_full.json", "variant": f"NB-{tag[-1]}",
                  "loglik": ll, "k": k}

    P = {k_: jnp.asarray(np.asarray(v)) for k_, v in pe["params"].items()}
    pi = np.asarray(pi_hat(P, D, spec))
    m0, m1 = (np.asarray(v) for v in component_means(P, D, spec))
    models["Poisson mixture"] = [(1 - pi, "pois", m0, None), (pi, "pois", m1, None)]
    meta["Poisson mixture"] = {"source": f"{DP.hpc_dir()}/point_estimate.json",
                               "k": sum(int(np.size(np.asarray(v))) for v in pe["params"].values())}

    nbnb_path = f"outputs/nb_nb_{ds}/params.json"
    if os.path.exists(nbnb_path):
        Q = {k_: jnp.asarray(np.asarray(v)) for k_, v in json.load(open(nbnb_path)).items()}
        s2 = dataclasses.replace(spec, cycling_nb=True, noncycling_nb=True)
        pi2 = np.asarray(pi_hat(Q, D, s2))
        q0, q1 = (np.asarray(v) for v in component_means(Q, D, s2))
        a0, a1 = float(np.exp(np.asarray(Q["la0"]))), float(np.exp(np.asarray(Q["la1"])))
        models["NB + NB mixture"] = [(1 - pi2, "nb", q0, a0), (pi2, "nb", q1, a1)]
        meta["NB + NB mixture"] = {"source": nbnb_path, "alpha0": a0, "alpha1": a1,
                                   "k": sum(int(np.size(np.asarray(v))) for v in Q.values())}
    else:
        say(f"  no {nbnb_path}: NB + NB mixture not fitted on this section")
    names_m = list(models)
    label = {m: (f"NB ({meta[m]['variant']}, matched covariates)" if m == "NB" else m) for m in names_m}

    # ---- per-model summaries ---------------------------------------------------------
    u_draw = np.random.default_rng(0).uniform(size=n)
    obs = np.array([np.sum(y == k_) for k_ in range(KMAX)] + [np.sum(y >= KMAX)], float)
    res = {}
    rows = []
    for m in names_m:
        say(f"  {m}: predictive distribution ...")
        exp_, u, mean = dist_summaries(y, models[m], u_draw)
        r = stats.norm.ppf(u)
        ll = meta[m].get("loglik") or log_lik(y, models[m])
        res[m] = dict(expected=exp_, u=u, r=r, mean=mean)
        rows.append({"model": m, "detail": meta[m].get("variant", ""), "source": meta[m]["source"],
                     "loglik": ll, "parameters": meta[m]["k"], "AIC": -2 * ll + 2 * meta[m]["k"],
                     "rootogram misfit": float(np.sum((np.sqrt(obs) - np.sqrt(exp_)) ** 2)),
                     "zeros observed": float(obs[0] / n), "zeros expected": float(exp_[0] / n),
                     "PIT KS": float(stats.kstest(u, "uniform").statistic),
                     "PIT max bin deviation": float(np.max(np.abs(
                         np.histogram(u, 20, (0, 1))[0] / (n / 20) - 1))),
                     "residual sd": float(r.std())})

    # ---- 1. rootogram --------------------------------------------------------------
    nm = len(names_m)
    fig, axes = plt.subplots(1, nm, figsize=(4.6 * nm, 3.8), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    kk = np.arange(KMAX + 1)
    for ax, m in zip(axes, names_m):
        e = np.sqrt(res[m]["expected"])
        o = np.sqrt(obs)
        ax.bar(kk, o, bottom=e - o, width=0.8, color=GRID, edgecolor=MUTED, lw=0.4)
        ax.plot(kk, e, "-", color=COL[m], lw=2)
        ax.plot(kk, e, "o", ms=3.5, color=COL[m], mec=SURF, mew=0.8)
        ax.axhline(0, color=INK2, lw=0.9)
        style(ax)
        ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
        ax.set_xticklabels(["0", "5", "10", "15", "20", "25", "30+"])
        ax.set_xlabel("Tirosh S + G2M count")
        mis = rows[names_m.index(m)]["rootogram misfit"]
        ax.set_title(f"{label[m]}\nmisfit {mis:,.0f}", loc="left", fontsize=10, color=INK)
    axes[0].set_ylabel("sqrt(frequency)")
    fig.suptitle(f"{TITLE[ds]}: hanging rootograms (bars: observed, hanging from the expected curve)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(f"{OUT}/{ds}_rootogram.png", dpi=180)
    plt.close(fig)

    # ---- 2. PIT ------------------------------------------------------------------------
    fig, axes = plt.subplots(1, nm, figsize=(4.0 * nm, 3.3), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, names_m):
        h = np.histogram(res[m]["u"], 20, (0, 1))[0] / (n / 20)
        ax.bar(np.arange(20) / 20 + 0.025, h, width=0.045, color=COL[m])
        ax.axhline(1, color=INK2, lw=1, ls="--")
        style(ax)
        ax.set_xlim(0, 1)
        ax.set_xlabel("randomised PIT")
        ks = rows[names_m.index(m)]["PIT KS"]
        ax.set_title(f"{label[m]}\nKS distance {ks:.3f}", loc="left", fontsize=10, color=INK)
    axes[0].set_ylabel("density (1 = calibrated)")
    fig.suptitle(f"{TITLE[ds]}: probability integral transform", x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(f"{OUT}/{ds}_pit.png", dpi=180)
    plt.close(fig)

    # ---- 3. residual diagnostics -----------------------------------------------------
    types = pd.Series(ctype).value_counts().index.tolist()
    dec = pd.qcut(np.log(N), 20, labels=False, duplicates="drop")
    fig, axes = plt.subplots(3, nm, figsize=(4.4 * nm, 10.2), constrained_layout=True, squeeze=False)
    qn = stats.norm.ppf((np.arange(1, 2001) - 0.5) / 2000)
    for j, m in enumerate(names_m):
        r = res[m]["r"]
        ax = axes[0, j]
        qs = np.quantile(r, (np.arange(1, 2001) - 0.5) / 2000)
        ax.plot([-4, 4], [-4, 4], color=INK2, lw=1, ls="--")
        ax.plot(qn, qs, color=COL[m], lw=2)
        style(ax, grid_axis="both")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_xlabel("normal quantile")
        ax.set_title(f"{label[m]}\nQQ plot (sd {r.std():.3f})", loc="left", fontsize=10)
        ax = axes[1, j]
        g = pd.DataFrame({"d": dec, "r": r, "N": N}).groupby("d").agg(
            r=("r", "mean"), se=("r", lambda v: 2 * v.std() / np.sqrt(len(v))), N=("N", "median"))
        ax.axhline(0, color=INK2, lw=1)
        ax.fill_between(g["N"], g["r"] - g["se"], g["r"] + g["se"], color=COL[m], alpha=0.2, lw=0)
        ax.plot(g["N"], g["r"], "-o", color=COL[m], ms=4, mec=SURF, mew=0.8, lw=1.8)
        ax.set_xscale("log")
        style(ax)
        ax.set_xlabel("depth N (ventile median, log scale)")
        ax.set_title("mean residual by depth", loc="left", fontsize=10)
        ax = axes[2, j]
        t = pd.DataFrame({"t": ctype, "r": r}).groupby("t")["r"].agg(
            ["mean", lambda v: 2 * v.std() / np.sqrt(len(v))]).reindex(types)
        yy = np.arange(len(types))[::-1]
        ax.axvline(0, color=INK2, lw=1)
        ax.errorbar(t["mean"], yy, xerr=t.iloc[:, 1], fmt="o", color=COL[m], ms=5, mec=SURF,
                    mew=0.8, elinewidth=1.5, capsize=0)
        ax.set_yticks(yy)
        ax.set_yticklabels(types if j == 0 else [""] * len(types), fontsize=9)
        style(ax, grid_axis="x")
        ax.set_xlabel("mean residual (+/- 2 SE)")
        ax.set_title("mean residual by cell type", loc="left", fontsize=10)
    for i in (1, 2):
        lo = min(a.get_xlim()[0] if i == 2 else a.get_ylim()[0] for a in axes[i])
        hi = max(a.get_xlim()[1] if i == 2 else a.get_ylim()[1] for a in axes[i])
        for a in axes[i]:
            (a.set_xlim if i == 2 else a.set_ylim)(lo, hi)
    axes[0, 0].set_ylabel("residual quantile")
    axes[1, 0].set_ylabel("mean quantile residual")
    fig.suptitle(f"{TITLE[ds]}: randomised quantile residuals (standard normal if the model holds)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(f"{OUT}/{ds}_residuals.png", dpi=180)
    plt.close(fig)

    # ---- 4. spatial: fitted mean and residual maps -----------------------------------
    xb = ((xs - xs.min()) // BIN_UM).astype(int)
    yb = ((ys_ - ys_.min()) // BIN_UM).astype(int)
    shape = (yb.max() + 1, xb.max() + 1)
    ext = [xs.min(), xs.min() + shape[1] * BIN_UM, ys_.min() + shape[0] * BIN_UM, ys_.min()]
    asp = (ys_.max() - ys_.min()) / (xs.max() - xs.min())
    panels = [("Observed", y)] + [(m, res[m]["mean"]) for m in names_m]
    grids = [grid_mean(v, xb, yb, shape) for _, v in panels]
    vmax = float(np.nanquantile(grids[0], 0.99))
    seq = LinearSegmentedColormap.from_list("seq", SEQ)
    seq.set_bad(SURF)
    pw = 4.4
    fig, axes = plt.subplots(1, len(panels), figsize=(pw * len(panels), pw * asp + 1.3),
                             constrained_layout=True)
    for ax, (ttl, _), g in zip(axes, panels, grids):
        im = ax.imshow(np.ma.masked_invalid(g), cmap=seq, vmin=0, vmax=vmax, extent=ext,
                       interpolation="nearest")
        map_style(ax)
        ax.set_title("Observed" if ttl == "Observed" else f"{label[ttl]}\nfitted mean", loc="left", fontsize=10)
    cb = fig.colorbar(im, ax=axes, location="bottom", shrink=0.35, aspect=40)
    cb.set_label(f"mean Tirosh count per {BIN_UM:g} \u00b5m square", color=INK2)
    cb.outline.set_visible(False)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.94))
    fig.suptitle(f"{TITLE[ds]}: observed counts and fitted means", x=0.01, y=0.995, ha="left", fontsize=11)
    fig.savefig(f"{OUT}/{ds}_fitted_map.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    rg = {m: grid_mean(res[m]["r"], xb, yb, shape) for m in names_m}
    lim = float(np.nanmax([np.nanquantile(np.abs(g), 0.98) for g in rg.values()]))
    div = LinearSegmentedColormap.from_list("div", DIV)
    div.set_bad(SURF)
    fig, axes = plt.subplots(1, nm, figsize=(pw * nm + 0.4, pw * asp + 1.3), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, names_m):
        I = morans_i(rg[m])
        rows[names_m.index(m)]["Moran's I (residual squares)"] = I
        im = ax.imshow(np.ma.masked_invalid(rg[m]), cmap=div, vmin=-lim, vmax=lim, extent=ext,
                       interpolation="nearest")
        map_style(ax)
        ax.set_title(f"{label[m]}\nMoran's I {I:.3f}", loc="left", fontsize=10)
    cb = fig.colorbar(im, ax=axes, location="bottom", shrink=0.35, aspect=40)
    cb.set_label(f"mean quantile residual per {BIN_UM:g} \u00b5m square (red: more counts than predicted)",
                 color=INK2)
    cb.outline.set_visible(False)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.94))
    fig.suptitle(f"{TITLE[ds]}: spatial structure left in the residuals", x=0.01, y=0.995, ha="left",
                 fontsize=11)
    fig.savefig(f"{OUT}/{ds}_residual_map.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    met = pd.DataFrame(rows)
    met.to_csv(f"{OUT}/{ds}_metrics.csv", index=False)
    say(met.drop(columns=["source"]).round(4).to_string(index=False))
    say(f"wrote {OUT}/{ds}_{{rootogram,pit,residuals,fitted_map,residual_map}}.png")


if __name__ == "__main__":
    main()

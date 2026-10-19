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
  Poisson + NB  where fitted (outputs/poisson_nb[_<ds>]/params.json): the production
                mixture with an NB2 cycling component.
  NB + NB mix.  where fitted (outputs/nb_nb[_<ds>]/params.json): both components NB2.

Figures: outputs/figures/model_fit/<ds>_<figure>.png with the models side by side,
and outputs/figures/model_fit/isolated/<ds>_<model>_<figure>.png with one model each:

  rootogram     hanging rootogram on 0..29 and a pooled 30+: bars of sqrt(observed)
                hang from the curve of sqrt(expected); a bar ending above or below
                the zero line is a count the model under- or over-predicts.
  pit           randomised PIT histogram (20 bins; uniform if calibrated), with the
                KS distance. The same uniform draws for every model.
  residuals     randomised quantile residuals r = Phi^-1(PIT) (Dunn and Smyth):
                normal QQ plot, mean residual by depth (N) ventile, and mean
                residual by cell type, each with +/- 2 block-cluster SE (250 spatial blocks).
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

from tissue_dynamics.count_diagnostics import (KMAX, cluster_mean_se, dist_summaries,  # noqa: E402
                                               grid_mean, log_lik, morans_i)
from tissue_dynamics.division import arms                             # noqa: E402
from tissue_dynamics.division import paths as DP                      # noqa: E402
from tissue_dynamics.mixture.basis import build_basis, spec_from_basis  # noqa: E402
from tissue_dynamics.mixture.data import load_config, phi_cell_types  # noqa: E402
from tissue_dynamics.mixture.model import build_data, component_means, pi_hat  # noqa: E402

OUT = "outputs/figures/model_fit"
BIN_UM = 50.0
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
COL = {"NB": "#eb6834", "Poisson mixture": "#2a78d6", "Poisson + NB mixture": "#6250d6",
       "NB + NB mixture": "#1baf7a"}
SLUG = {"NB": "nb", "Poisson mixture": "poisson_mixture", "Poisson + NB mixture": "poisson_nb_mixture",
        "NB + NB mixture": "nb_nb_mixture"}
SEQ = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV = ["#0d366b", "#256abf", "#6da7ec", "#cde2fb", "#f0efec", "#fbd5d3", "#f29b98", "#e34948", "#9e2423"]
TITLE = {"breast": "Breast section", "ovarian": "Ovarian section"}


def say(*a):
    print(*a, flush=True)


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
    iso = f"{OUT}/isolated"
    os.makedirs(iso, exist_ok=True)
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
    fm = pd.read_parquet(f"{DP.out_dir()}/folds.parquet")
    if not np.array_equal(fm["cell_id"].astype(str).to_numpy(), df["cell_id"].astype(str).to_numpy()):
        sys.exit("folds.parquet lists cells in a different order from the table")
    BLOCK = fm["block"].to_numpy()          # the 250 spatial blocks: SEs are block-cluster
    say(f"{TITLE[ds]}: {n:,} cells, mean y {y.mean():.3f}")
    nparams = lambda q: sum(int(np.size(np.asarray(v))) for v in q.values())   # noqa: E731

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
        nb_choice.append((-2 * ll + 2 * nparams(p), tag, comps, ll, nparams(p)))
        say(f"  NB-{tag[-1]}: loglik {ll:,.1f}, k {nparams(p)}")
    _, tag, comps, ll, k = min(nb_choice, key=lambda t: t[0])
    models["NB"] = comps
    meta["NB"] = {"source": f"{DP.out_dir()}/nb_params_{tag}_full.json", "variant": f"NB-{tag[-1]}",
                  "loglik": ll, "k": k}

    P = {k_: jnp.asarray(np.asarray(v)) for k_, v in pe["params"].items()}
    pi = np.asarray(pi_hat(P, D, spec))
    m0, m1 = (np.asarray(v) for v in component_means(P, D, spec))
    models["Poisson mixture"] = [(1 - pi, "pois", m0, None), (pi, "pois", m1, None)]
    meta["Poisson mixture"] = {"source": f"{DP.hpc_dir()}/point_estimate.json", "k": nparams(pe["params"]),
                               "mean pi": float(pi.mean())}

    sfx = f"_{ds}" if ds != "breast" else ""
    for m, path, c_nb, n_nb in (("Poisson + NB mixture", f"outputs/poisson_nb{sfx}/params.json", True, False),
                                ("NB + NB mixture", f"outputs/nb_nb{sfx}/params.json", True, True)):
        if not os.path.exists(path):
            say(f"  no {path}: {m} not fitted on this section")
            continue
        Q = {k_: jnp.asarray(np.asarray(v)) for k_, v in json.load(open(path)).items()}
        s2 = dataclasses.replace(spec, cycling_nb=c_nb, noncycling_nb=n_nb)
        pq = np.asarray(pi_hat(Q, D, s2))
        q0, q1 = (np.asarray(v) for v in component_means(Q, D, s2))
        a1 = float(np.exp(np.asarray(Q["la1"])))
        a0 = float(np.exp(np.asarray(Q["la0"]))) if n_nb else None
        models[m] = [(1 - pq, "nb" if n_nb else "pois", q0, a0), (pq, "nb", q1, a1)]
        meta[m] = {"source": path, "alpha1": a1, "alpha0": a0, "k": nparams(Q), "mean pi": float(pq.mean())}
    names_m = list(models)
    label = {m: (f"NB ({meta[m]['variant']}, matched covariates)" if m == "NB" else m) for m in names_m}

    # ---- per-model summaries ---------------------------------------------------------
    u_draw = np.random.default_rng(0).uniform(size=n)
    obs = np.array([np.sum(y == k_) for k_ in range(KMAX)] + [np.sum(y >= KMAX)], float)
    res, rows = {}, {}
    for m in names_m:
        say(f"  {m}: predictive distribution ...")
        exp_, u, mean = dist_summaries(y, models[m], u_draw)
        r = stats.norm.ppf(u)
        ll = meta[m].get("loglik") or log_lik(y, models[m])
        res[m] = dict(expected=exp_, u=u, r=r, mean=mean)
        rows[m] = {"model": m, "detail": meta[m].get("variant", ""), "source": meta[m]["source"],
                   "loglik": ll, "parameters": meta[m]["k"], "AIC": -2 * ll + 2 * meta[m]["k"],
                   "alpha0": meta[m].get("alpha0"), "alpha1": meta[m].get("alpha1"),
                   "mean pi": meta[m].get("mean pi"),
                   "rootogram misfit": float(np.sum((np.sqrt(obs) - np.sqrt(exp_)) ** 2)),
                   "zeros observed": float(obs[0] / n), "zeros expected": float(exp_[0] / n),
                   "PIT KS": float(stats.kstest(u, "uniform").statistic),
                   "PIT max bin deviation": float(np.max(np.abs(
                       np.histogram(u, 20, (0, 1))[0] / (n / 20) - 1))),
                   "residual sd": float(r.std())}

    xb = ((xs - xs.min()) // BIN_UM).astype(int)
    yb = ((ys_ - ys_.min()) // BIN_UM).astype(int)
    shape = (yb.max() + 1, xb.max() + 1)
    ext = [xs.min(), xs.min() + shape[1] * BIN_UM, ys_.min() + shape[0] * BIN_UM, ys_.min()]
    asp = (ys_.max() - ys_.min()) / (xs.max() - xs.min())
    obs_grid = grid_mean(y, xb, yb, shape)
    fit_grid = {m: grid_mean(res[m]["mean"], xb, yb, shape) for m in names_m}
    res_grid = {m: grid_mean(res[m]["r"], xb, yb, shape) for m in names_m}
    for m in names_m:
        rows[m]["Moran's I (residual squares)"] = morans_i(res_grid[m])
    # one colour scale per figure kind, shared by the combined and isolated versions
    vmax = float(np.nanquantile(obs_grid, 0.99))
    rlim = float(np.nanmax([np.nanquantile(np.abs(g), 0.98) for g in res_grid.values()]))
    types = pd.Series(ctype).value_counts().index.tolist()
    dec = pd.qcut(np.log(N), 20, labels=False, duplicates="drop")
    qn = stats.norm.ppf((np.arange(1, 2001) - 0.5) / 2000)
    kk = np.arange(KMAX + 1)
    seq = LinearSegmentedColormap.from_list("seq", SEQ)
    seq.set_bad(SURF)
    div = LinearSegmentedColormap.from_list("div", DIV)
    div.set_bad(SURF)
    pw = 4.4

    def fname(kind, ms):
        return f"{OUT}/{ds}_{kind}.png" if len(ms) > 1 else f"{iso}/{ds}_{SLUG[ms[0]]}_{kind}.png"

    def fig_rootogram(ms):
        fig, axes = plt.subplots(1, len(ms), figsize=(4.6 * len(ms), 3.8), sharey=True,
                                 constrained_layout=True, squeeze=False)
        for ax, m in zip(axes[0], ms):
            e, o = np.sqrt(res[m]["expected"]), np.sqrt(obs)
            ax.bar(kk, o, bottom=e - o, width=0.8, color=GRID, edgecolor=MUTED, lw=0.4)
            ax.plot(kk, e, "-", color=COL[m], lw=2)
            ax.plot(kk, e, "o", ms=3.5, color=COL[m], mec=SURF, mew=0.8)
            ax.axhline(0, color=INK2, lw=0.9)
            style(ax)
            ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
            ax.set_xticklabels(["0", "5", "10", "15", "20", "25", "30+"])
            ax.set_xlabel("Tirosh S + G2M count")
            ax.set_title(f"{label[m]}\nmisfit {rows[m]['rootogram misfit']:,.0f}", loc="left", fontsize=10)
        axes[0, 0].set_ylabel("sqrt(frequency)")
        fig.suptitle(f"{TITLE[ds]}: hanging rootogram (bars: observed, hanging from the expected curve)",
                     x=0.01, ha="left", fontsize=11)
        fig.savefig(fname("rootogram", ms), dpi=180)
        plt.close(fig)

    def fig_pit(ms):
        fig, axes = plt.subplots(1, len(ms), figsize=(4.0 * len(ms), 3.3), sharey=True,
                                 constrained_layout=True, squeeze=False)
        for ax, m in zip(axes[0], ms):
            h = np.histogram(res[m]["u"], 20, (0, 1))[0] / (n / 20)
            ax.bar(np.arange(20) / 20 + 0.025, h, width=0.045, color=COL[m])
            ax.axhline(1, color=INK2, lw=1, ls="--")
            style(ax)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.35)
            ax.set_xlabel("randomised PIT")
            ax.set_title(f"{label[m]}\nKS distance {rows[m]['PIT KS']:.3f}", loc="left", fontsize=10)
        axes[0, 0].set_ylabel("density (1 = calibrated)")
        fig.suptitle(f"{TITLE[ds]}: probability integral transform", x=0.01, ha="left", fontsize=11)
        fig.savefig(fname("pit", ms), dpi=180)
        plt.close(fig)

    rlims = {}

    def fig_residuals(ms):
        if len(ms) == 1:                    # one model: the three panels side by side
            fig, row = plt.subplots(1, 3, figsize=(14.5, 4.4), constrained_layout=True)
            axes = np.asarray(row).reshape(3, 1)
        else:
            fig, axes = plt.subplots(3, len(ms), figsize=(4.4 * len(ms), 10.2), constrained_layout=True,
                                     squeeze=False)
        for j, m in enumerate(ms):
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
            g = cluster_mean_se(r, dec, BLOCK).rename(columns={"mean": "r"})
            g["se"] *= 2
            g["N"] = pd.Series(N).groupby(dec).median().reindex(g.index).to_numpy()
            ax.axhline(0, color=INK2, lw=1)
            ax.fill_between(g["N"], g["r"] - g["se"], g["r"] + g["se"], color=COL[m], alpha=0.2, lw=0)
            ax.plot(g["N"], g["r"], "-o", color=COL[m], ms=4, mec=SURF, mew=0.8, lw=1.8)
            ax.set_xscale("log")
            style(ax)
            ax.set_xlabel("depth N (ventile median, log scale)")
            ax.set_title("mean residual by depth", loc="left", fontsize=10)
            ax = axes[2, j]
            t = cluster_mean_se(r, ctype, BLOCK).reindex(types)
            t["se"] *= 2
            yy = np.arange(len(types))[::-1]
            ax.axvline(0, color=INK2, lw=1)
            ax.errorbar(t["mean"], yy, xerr=t.iloc[:, 1], fmt="o", color=COL[m], ms=5, mec=SURF,
                        mew=0.8, elinewidth=1.5, capsize=0)
            ax.set_yticks(yy)
            ax.set_yticklabels(types if j == 0 else [""] * len(types), fontsize=9)
            style(ax, grid_axis="x")
            ax.set_xlabel("mean residual (+/- 2 SE)")
            ax.set_title("mean residual by cell type", loc="left", fontsize=10)
        # shared limits: from the combined figure, reused by every isolated one
        if "dep" not in rlims:
            rlims["dep"] = (min(a.get_ylim()[0] for a in axes[1]), max(a.get_ylim()[1] for a in axes[1]))
            rlims["typ"] = (min(a.get_xlim()[0] for a in axes[2]), max(a.get_xlim()[1] for a in axes[2]))
        for a in axes[1]:
            a.set_ylim(*rlims["dep"])
        for a in axes[2]:
            a.set_xlim(*rlims["typ"])
        axes[0, 0].set_ylabel("residual quantile")
        axes[1, 0].set_ylabel("mean quantile residual")
        fig.suptitle(f"{TITLE[ds]}: randomised quantile residuals (standard normal if the model holds)",
                     x=0.01, ha="left", fontsize=11)
        fig.savefig(fname("residuals", ms), dpi=180)
        plt.close(fig)

    def fig_fitted_map(ms):
        panels = [("Observed", obs_grid)] + [(m, fit_grid[m]) for m in ms]
        fig, axes = plt.subplots(1, len(panels), figsize=(pw * len(panels), pw * asp + 1.3),
                                 constrained_layout=True)
        for ax, (ttl, g) in zip(axes, panels):
            im = ax.imshow(np.ma.masked_invalid(g), cmap=seq, vmin=0, vmax=vmax, extent=ext,
                           interpolation="nearest")
            map_style(ax)
            ax.set_title("Observed" if ttl == "Observed" else f"{label[ttl]}\nfitted mean", loc="left",
                         fontsize=10)
        cb = fig.colorbar(im, ax=axes, location="bottom", shrink=0.35 if len(panels) > 2 else 0.6, aspect=40)
        cb.set_label(f"mean Tirosh count per {BIN_UM:g} \u00b5m square", color=INK2)
        cb.outline.set_visible(False)
        fig.get_layout_engine().set(rect=(0, 0, 1, 0.94))
        fig.suptitle(f"{TITLE[ds]}: observed counts and fitted means", x=0.01, y=0.995, ha="left", fontsize=11)
        fig.savefig(fname("fitted_map", ms), dpi=170, bbox_inches="tight")
        plt.close(fig)

    def fig_residual_map(ms):
        fig, axes = plt.subplots(1, len(ms), figsize=(pw * len(ms) + 0.4 + (1.6 if len(ms) == 1 else 0),
                                                      pw * asp + 1.3), constrained_layout=True, squeeze=False)
        for ax, m in zip(axes[0], ms):
            im = ax.imshow(np.ma.masked_invalid(res_grid[m]), cmap=div, vmin=-rlim, vmax=rlim, extent=ext,
                           interpolation="nearest")
            map_style(ax)
            mi = rows[m]["Moran's I (residual squares)"]
            ax.set_title(f"{label[m]}\nMoran's I {mi:.3f}", loc="left", fontsize=10)
        cb = fig.colorbar(im, ax=axes[0], location="bottom", shrink=0.35 if len(ms) > 1 else 0.8, aspect=40)
        cb.set_label(f"mean quantile residual per {BIN_UM:g} \u00b5m square (red: more counts than predicted)",
                     color=INK2)
        cb.outline.set_visible(False)
        fig.get_layout_engine().set(rect=(0, 0, 1, 0.94))
        fig.suptitle(f"{TITLE[ds]}: spatial structure left in the residuals", x=0.01, y=0.995, ha="left",
                     fontsize=11)
        fig.savefig(fname("residual_map", ms), dpi=170, bbox_inches="tight")
        plt.close(fig)

    for draw in (fig_rootogram, fig_pit, fig_residuals, fig_fitted_map, fig_residual_map):
        draw(names_m)                       # combined first: it fixes the shared residual limits
        for m in names_m:
            draw([m])

    met = pd.DataFrame(list(rows.values()))
    met.to_csv(f"{OUT}/{ds}_metrics.csv", index=False)
    say(met.drop(columns=["source"]).round(4).to_string(index=False))
    say(f"wrote {OUT}/{ds}_*.png and {iso}/{ds}_<model>_*.png")


if __name__ == "__main__":
    main()

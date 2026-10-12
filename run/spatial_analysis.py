"""Spatial structure of the cell states and of each division model's residuals.

    python run/spatial_analysis.py

1. PhiSpace correlograms, cells stratified by argmax archetype. For each
   archetype three Moran's I curves against distance:
     - the full PhiSpace vector WITHIN that archetype's cells (Wartenberg's
       multivariate Moran's I: components standardised within the stratum, the
       pair statistic is their inner product);
     - the archetype's own score within its cells;
     - the archetype indicator 1[argmax == k] over ALL cells -- how clustered
       the cell type itself is.
2. Residual correlograms for the previous NB, the matched NB and the Poisson
   mixture: Pearson residuals (comparable with Test 14) and mid-quantile
   residuals Phi^-1((F(y-1) + F(y)) / 2). Mid-quantile rather than randomised
   PIT, because the randomisation jitter is independent noise that would dilute
   Moran's I. The raw count is shown as the reference for how much spatial
   structure each model removes, and a permuted residual as the noise floor.
3. Local Moran's I on residuals binned to a 100 um grid: queen contiguity,
   row-standardised, 9,999 conditional permutations, Benjamini-Hochberg at 5%.
4. The spatial blocks used by the bootstrap and CV, measured.

Correlograms use a random sample of focal cells against ALL cells (or all cells
of the stratum), so every band uses every neighbour of each focal cell. The
estimator is mean(z_i z_j over pairs in the band) / mean(z^2), which is Moran's
I with binary distance-band weights restricted to the sampled pairs.

Writes outputs/spatial/.
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import matplotlib                                                     # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.colors import ListedColormap                          # noqa: E402
import jax                                                            # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                               # noqa: E402

from tissue_dynamics.mixture.basis import build_basis, spec_from_basis    # noqa: E402
from tissue_dynamics.mixture.data import load_config                      # noqa: E402
from tissue_dynamics.mixture.inference import block_side_length, spatial_blocks  # noqa: E402
from tissue_dynamics.mixture.model import build_data, component_means, pi_hat  # noqa: E402

_s = importlib.util.spec_from_file_location("cmp", os.path.join(HERE, "compare_nb_mixture.py"))
cmp = importlib.util.module_from_spec(_s)
_s.loader.exec_module(cmp)

OUT = "outputs/spatial"
EDGES = np.array([0, 15, 25, 35, 50, 75, 100, 150, 200, 300, 400, 500], float)
MID = 0.5 * (EDGES[:-1] + EDGES[1:])
N_FOCAL_ALL = 30_000          # focal cells for the all-cell pass
N_FOCAL_STRATUM = 6_000       # focal cells per archetype for the stratified pass
CHUNK = 250
BIN_UM = 100.0
MIN_CELLS = 20
N_PERM = 9999
MODELS = ["nb_prev", "nb_matched", "mixture"]
LABEL = cmp.LABEL
COL = cmp.COL
INK, INK2, GRID = cmp.INK, cmp.INK2, cmp.GRID
NEUTRAL = "#8a8985"


def say(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------
# Correlogram engine
# --------------------------------------------------------------------------
def band_sums(xy_pool, focal, Z, edges=EDGES, chunk=CHUNK):
    """Sum of z_i . z_j and pair counts per distance band.

    `xy_pool` (m, 2) are the cells pairs may involve; `focal` indexes into it;
    `Z` is (m, p) -- each column a variable, already centred over the pool.
    Returns num (nb, p) and cnt (nb,). A multivariate variable is handled by the
    caller summing its columns' num.
    """
    tree = cKDTree(xy_pool)
    nb = len(edges) - 1
    num = np.zeros((nb, Z.shape[1]))
    cnt = np.zeros(nb)
    rmax = float(edges[-1])
    for s in range(0, len(focal), chunk):
        fc = focal[s:s + chunk]
        sub = cKDTree(xy_pool[fc])
        sp = sub.sparse_distance_matrix(tree, rmax, output_type="ndarray")
        i = fc[sp["i"]]
        j = sp["j"]
        d = sp["v"]
        keep = i != j
        i, j, d = i[keep], j[keep], d[keep]
        b = np.searchsorted(edges, d, side="right") - 1
        ok = (b >= 0) & (b < nb)
        i, j, b = i[ok], j[ok], b[ok]
        cnt += np.bincount(b, minlength=nb)
        prod = Z[i] * Z[j]                                        # (pairs, p)
        for c in range(Z.shape[1]):
            num[:, c] += np.bincount(b, weights=prod[:, c], minlength=nb)
    return num, cnt


def morans_from_sums(num, cnt, denom):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (num / cnt[:, None]) / denom[None, :]


# --------------------------------------------------------------------------
# Residuals
# --------------------------------------------------------------------------
def nb_moments_cdf(y, mu, alpha):
    r = 1.0 / alpha
    p = r / (r + mu)
    F = stats.nbinom.cdf(y, r, p)
    Fm = np.where(y > 0, stats.nbinom.cdf(y - 1, r, p), 0.0)
    return mu, mu + alpha * mu ** 2, F, Fm


def mix_moments_cdf(y, mu0, mu1, pi):
    mean = (1 - pi) * mu0 + pi * mu1
    var = (1 - pi) * (mu0 + mu0 ** 2) + pi * (mu1 + mu1 ** 2) - mean ** 2
    cdf = lambda q: (1 - pi) * stats.poisson.cdf(q, mu0) + pi * stats.poisson.cdf(q, mu1)
    F = cdf(y)
    Fm = np.where(y > 0, cdf(y - 1), 0.0)
    return mean, var, F, Fm


def residuals(y, mean, var, F, Fm):
    pearson = (y - mean) / np.sqrt(np.maximum(var, 1e-12))
    u = np.clip(0.5 * (F + Fm), 1e-12, 1 - 1e-12)
    return pearson, stats.norm.ppf(u)


# --------------------------------------------------------------------------
# Local Moran's I on a grid
# --------------------------------------------------------------------------
def grid_index(xy, size):
    x0, y0 = xy.min(0)
    gx = np.floor((xy[:, 0] - x0) / size).astype(int)
    gy = np.floor((xy[:, 1] - y0) / size).astype(int)
    return gx, gy


def lisa(values, gx, gy, min_cells=MIN_CELLS, n_perm=N_PERM, seed=0):
    """Local Moran's I of bin means, queen contiguity, conditional permutation."""
    nx, ny = gx.max() + 1, gy.max() + 1
    flat = gx * ny + gy
    cnt = np.bincount(flat, minlength=nx * ny)
    tot = np.bincount(flat, weights=values, minlength=nx * ny)
    keep = cnt >= min_cells
    idx = -np.ones(nx * ny, int)
    idx[keep] = np.arange(keep.sum())
    cells = np.where(keep)[0]
    bx, by = cells // ny, cells % ny
    nbr = -np.ones((len(cells), 8), int)
    k = 0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            xx, yy = bx + dx, by + dy
            inside = (xx >= 0) & (xx < nx) & (yy >= 0) & (yy < ny)
            nbr[inside, k] = idx[xx[inside] * ny + yy[inside]]
            k += 1
    kn = (nbr >= 0).sum(1)
    has = kn > 0
    m = tot[cells] / cnt[cells]
    # islands (no kept neighbour) carry no local statistic
    z = (m - m[has].mean()) / m[has].std()
    zp = np.where(nbr >= 0, z[np.maximum(nbr, 0)], 0.0)
    lag = np.where(has, zp.sum(1) / np.maximum(kn, 1), np.nan)
    I = z * lag

    rng = np.random.default_rng(seed)
    pool = np.where(has)[0]
    zpool = z[pool]
    p = np.full(len(z), np.nan)
    rows = np.where(has)[0]
    for s in range(0, len(rows), 400):
        r_ = rows[s:s + 400]
        draws = zpool[rng.integers(0, len(zpool), size=(len(r_), n_perm, 8))]
        mask = (np.arange(8)[None, None, :] < kn[r_][:, None, None])
        lagp = (draws * mask).sum(-1) / kn[r_][:, None]
        Ip = z[r_][:, None] * lagp
        obs = I[r_][:, None]
        more = np.where(obs >= 0, Ip >= obs, Ip <= obs).sum(1)
        p[r_] = (more + 1) / (n_perm + 1)
    q = np.full(len(z), np.nan)
    pv = p[has]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    qv = np.minimum.accumulate(ranked[::-1])[::-1]
    qq = np.empty_like(pv)
    qq[order] = np.minimum(qv, 1.0)
    q[has] = qq
    sig = q < 0.05
    cls = np.full(len(z), "ns", dtype=object)
    cls[sig & (z > 0) & (lag > 0)] = "HH"
    cls[sig & (z < 0) & (lag < 0)] = "LL"
    cls[sig & (z > 0) & (lag < 0)] = "HL"
    cls[sig & (z < 0) & (lag > 0)] = "LH"
    cls[~has] = "island"

    # global Moran's I on the same grid, with a permutation reference
    zh, lh = z[has], lag[has]
    I_glob = float(np.sum(zh * lh) / np.sum(zh ** 2))
    perm = []
    for _ in range(199):
        zs = z.copy()
        zs[pool] = rng.permutation(zpool)
        zsp = np.where(nbr >= 0, zs[np.maximum(nbr, 0)], 0.0)
        ls = zsp.sum(1)[has] / kn[has]
        perm.append(np.sum(zs[has] * ls) / np.sum(zs[has] ** 2))
    perm = np.array(perm)
    return dict(cells=cells, bx=bx, by=by, nx=nx, ny=ny, mean=m, z=z, lag=lag, I=I,
                p=p, q=q, cls=cls, idx=idx, flat=flat, I_global=I_glob,
                I_global_null_sd=float(perm.std()),
                I_global_p=float((np.sum(np.abs(perm) >= abs(I_glob)) + 1) / (len(perm) + 1)))


# --------------------------------------------------------------------------
def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    # Everything read from disk is read here, up front. The repo lives on an
    # iCloud-synced Desktop, and a file evicted mid-run has already killed one
    # nine-hour job by timing out a later read.
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["table_parquet"])
    phi_names = json.load(open("data/mixture_cells_major.phi_names.json"))
    df.attrs["phi_names"] = phi_names
    pe = json.load(open("outputs/hpc/point_estimate.json"))

    basis = build_basis(df, cfg, verbose=False)
    spec = spec_from_basis(basis, cfg, stage="B", weight_ridge=float(pe["weight_ridge"]))
    data = build_data(df, basis, cfg, spec)
    D = data.as_jax()
    y = np.asarray(data.y)
    xy = df[["x", "y"]].to_numpy(float)
    n = len(y)

    # phi_00..phi_08 are the raw celltype_major scores in `phi_names` order.
    # Their scales differ ~18-fold (sd 0.19 to 3.55), so a RAW argmax is
    # dominated by the high-variance columns -- it agrees with the cell-type
    # label for only 36.6% of cells, and sends 90% of Cancer Epithelial cells to
    # Plasmablasts. Stratify on the argmax of the column-standardised scores,
    # exactly as the stored `archetype` label is built (data.py).
    phi_cols = sorted(c for c in df.columns if c.startswith("phi_"))
    PHI = df[phi_cols].to_numpy(float)
    names = list(phi_names)
    Zs = (PHI - PHI.mean(0)) / np.where(PHI.std(0) > 0, PHI.std(0), 1.0)
    am = Zs.argmax(1)
    arg_name = np.array(names)[am]
    agree = float(np.mean(arg_name == df["archetype"].astype(str).to_numpy()))
    raw_agree = float(np.mean(np.array(names)[PHI.argmax(1)]
                              == df["archetype"].astype(str).to_numpy()))
    say(f"{n:,} cells; standardised argmax agrees with the stored label for "
        f"{100 * agree:.2f}% of cells (raw argmax: {100 * raw_agree:.1f}%)")

    # ---------------- 4. the blocks the bootstrap and CV use ----------------
    labels = spatial_blocks(xy, int(cfg["inference"]["n_blocks_target"]))
    sizes = np.bincount(labels)
    blocks = dict(n_blocks=int(len(sizes)), cells_per_block_median=float(np.median(sizes)),
                  cells_per_block_p5=float(np.percentile(sizes, 5)),
                  cells_per_block_p95=float(np.percentile(sizes, 95)),
                  median_block_diameter_um=block_side_length(xy, labels),
                  tissue_extent_um=[float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1]))])
    say(f"blocks: {blocks['n_blocks']} | median {blocks['cells_per_block_median']:.0f} cells | "
        f"median diameter {blocks['median_block_diameter_um']:.0f} um")

    # ---------------- 1. PhiSpace, stratified by argmax ---------------------
    rng = np.random.default_rng(0)
    rows = []
    for k, nm in enumerate(names):
        g = np.where(am == k)[0]
        Zk = PHI[g]
        Zk = (Zk - Zk.mean(0)) / np.where(Zk.std(0) > 0, Zk.std(0), 1.0)
        own = (PHI[g, k] - PHI[g, k].mean()) / PHI[g, k].std()
        V = np.column_stack([Zk, own])
        focal = rng.choice(len(g), min(N_FOCAL_STRATUM, len(g)), replace=False)
        num, cnt = band_sums(xy[g], focal, V)
        I_multi = num[:, :-1].sum(1) / cnt / np.mean((Zk ** 2).sum(1))
        I_own = num[:, -1] / cnt / np.mean(own ** 2)
        for b in range(len(MID)):
            rows.append(dict(archetype=nm, n_cells=len(g), dist_um=MID[b],
                             band_lo=EDGES[b], band_hi=EDGES[b + 1], pairs=int(cnt[b]),
                             I_multivariate_within=float(I_multi[b]),
                             I_own_score_within=float(I_own[b])))
        say(f"  {nm:<20} n={len(g):>7,}  I(0-15um): multivariate {I_multi[0]:.3f}, "
            f"own score {I_own[0]:.3f}  ({time.time() - t0:.0f}s)")
    phi_cg = pd.DataFrame(rows)

    # ---------------- 2. residuals ------------------------------------------
    say("fitting the two NB models (the matched one takes ~25 min)")
    nb = {}
    for m, matched in (("nb_prev", False), ("nb_matched", True)):
        f = cmp.fit_nb(D, matched=matched)
        nb[m] = (f["mu"], f["alpha"])
    np.savez_compressed(f"{OUT}/nb_fits.npz",
                        **{f"{m}_mu": v[0] for m, v in nb.items()},
                        **{f"{m}_alpha": np.array(v[1]) for m, v in nb.items()})
    P = {k: jnp.asarray(np.asarray(v)) for k, v in pe["params"].items()}
    lam0, lam1 = (np.asarray(x) for x in component_means(P, D, spec))
    pi = np.asarray(pi_hat(P, D, spec))

    res = {}
    for m in MODELS:
        mom = (nb_moments_cdf(y, *nb[m]) if m != "mixture"
               else mix_moments_cdf(y, lam0, lam1, pi))
        res[m] = residuals(y, *mom)
        say(f"  {m:<11} Pearson mean {res[m][0].mean():+.4f} sd {res[m][0].std():.3f} | "
            f"mid-quantile mean {res[m][1].mean():+.4f} sd {res[m][1].std():.3f}")

    # all-cell pass: indicators, raw count, residuals, a permuted noise floor
    cols, Vs = [], []
    for k, nm in enumerate(names):
        ind = (am == k).astype(float)
        cols.append(f"indicator:{nm}")
        Vs.append(ind - ind.mean())
    cols.append("raw:count")
    Vs.append((y - y.mean()) / y.std())
    for m in MODELS:
        for t, r in zip(("pearson", "midq"), res[m]):
            cols.append(f"{t}:{m}")
            Vs.append(r - r.mean())
    cols.append("pearson:mixture_permuted")
    Vs.append(rng.permutation(res["mixture"][0] - res["mixture"][0].mean()))
    V = np.column_stack(Vs).astype(np.float64)
    focal = rng.choice(n, N_FOCAL_ALL, replace=False)
    say(f"all-cell correlogram pass: {len(cols)} variables, {N_FOCAL_ALL:,} focal cells")
    num, cnt = band_sums(xy, focal, V)
    I_all = morans_from_sums(num, cnt, np.mean(V ** 2, 0))
    allc = pd.DataFrame(I_all, columns=cols)
    allc.insert(0, "pairs", cnt.astype(int))
    allc.insert(0, "band_hi", EDGES[1:])
    allc.insert(0, "band_lo", EDGES[:-1])
    allc.insert(0, "dist_um", MID)
    allc.to_csv(f"{OUT}/correlogram_all_cells.csv", index=False)

    ind_long = []
    for nm in names:
        for b in range(len(MID)):
            ind_long.append(dict(archetype=nm, dist_um=MID[b],
                                 I_indicator_all=float(allc[f"indicator:{nm}"].iloc[b])))
    phi_cg = phi_cg.merge(pd.DataFrame(ind_long), on=["archetype", "dist_um"])
    phi_cg.to_csv(f"{OUT}/correlogram_phispace.csv", index=False)

    # ---------------- 3. local Moran's I ------------------------------------
    gx, gy = grid_index(xy, BIN_UM)
    L = {}
    lisa_rows, char_rows = [], []
    ctype = df["archetype"].astype(str).to_numpy()
    dens = df["density"].to_numpy(float)
    for m in MODELS:
        L[m] = lisa(res[m][0], gx, gy, seed=1)
        cls = L[m]["cls"]
        say(f"  LISA {m:<11} global I on {BIN_UM:.0f}um bins = {L[m]['I_global']:.4f} "
            f"(null sd {L[m]['I_global_null_sd']:.4f}, p {L[m]['I_global_p']:.3f}) | "
            + ", ".join(f"{c} {np.sum(cls == c)}" for c in ("HH", "LL", "HL", "LH", "ns")))
        lisa_rows.append(dict(model=m, bins=int(np.sum(cls != "island")),
                              I_global=L[m]["I_global"], I_global_null_sd=L[m]["I_global_null_sd"],
                              I_global_p=L[m]["I_global_p"],
                              **{f"n_{c}": int(np.sum(cls == c)) for c in ("HH", "LL", "HL", "LH", "ns")}))
        # characterise the clusters by the cells inside them
        cell_bin = L[m]["idx"][L[m]["flat"]]
        cell_cls = np.where(cell_bin >= 0, cls[np.maximum(cell_bin, 0)], "dropped")
        for c in ("HH", "LL", "ns"):
            sel = cell_cls == c
            if sel.sum() == 0:
                continue
            comp = pd.Series(ctype[sel]).value_counts(normalize=True)
            char_rows.append(dict(model=m, cluster=c, n_cells=int(sel.sum()),
                                  mean_count=float(y[sel].mean()),
                                  median_density=float(np.median(dens[sel])),
                                  mean_residual=float(res[m][0][sel].mean()),
                                  **{f"share:{t}": float(comp.get(t, 0.0)) for t in names}))
    pd.DataFrame(lisa_rows).to_csv(f"{OUT}/lisa_summary.csv", index=False)
    chars = pd.DataFrame(char_rows)
    chars.to_csv(f"{OUT}/lisa_cluster_composition.csv", index=False)
    for m in MODELS:
        pd.DataFrame(dict(bx=L[m]["bx"], by=L[m]["by"], mean_resid=L[m]["mean"],
                          z=L[m]["z"], lag=L[m]["lag"], I=L[m]["I"], p=L[m]["p"],
                          q=L[m]["q"], cls=L[m]["cls"])).to_csv(
            f"{OUT}/lisa_bins_{m}.csv", index=False)

    json.dump(dict(blocks=blocks, argmax_agreement=agree, raw_argmax_agreement=raw_agree, edges_um=EDGES.tolist(),
                   n_focal_all=N_FOCAL_ALL, n_focal_stratum=N_FOCAL_STRATUM,
                   bin_um=BIN_UM, min_cells_per_bin=MIN_CELLS, n_perm=N_PERM,
                   nb_alpha={m: float(nb[m][1]) for m in nb}),
              open(f"{OUT}/summary.json", "w"), indent=2)

    plot_phi(phi_cg, names)
    plot_resid(allc)
    plot_lisa(L, xy)
    print_report(phi_cg, allc, pd.DataFrame(lisa_rows), chars, names, blocks)
    say(f"\ndone in {(time.time() - t0) / 60:.1f} min; wrote {OUT}/")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def plot_phi(phi_cg, names):
    plt.rcParams.update({"font.size": 9.5, "text.color": INK, "axes.labelcolor": INK2})
    order = (phi_cg[phi_cg.dist_um == MID[0]]
             .sort_values("I_indicator_all", ascending=False)["archetype"].tolist())
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 9), sharex=True, sharey=True)
    series = (("I_indicator_all", "cell type clustering (indicator, all cells)", COL["mixture"]),
              ("I_multivariate_within", "PhiSpace vector, within type", COL["nb_prev"]),
              ("I_own_score_within", "own score, within type", COL["nb_matched"]))
    for ax, nm in zip(axes.ravel(), order):
        s = phi_cg[phi_cg.archetype == nm]
        ax.axhline(0, color=INK2, lw=0.8)
        for col, lab, c in series:
            ax.plot(s.dist_um, s[col], color=c, lw=2, marker="o", ms=3.5, label=lab)
        _style(ax)
        ax.set_title(f"{nm}  (n = {int(s.n_cells.iloc[0]):,})", loc="left", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("distance (um)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Moran's I")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("PhiSpace correlograms, cells stratified by argmax archetype",
                 fontsize=11, color=INK, y=0.995)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig.savefig(f"{OUT}/correlogram_phispace.png", dpi=160)
    plt.close(fig)


def plot_resid(allc):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    ax.plot(allc.dist_um, allc["raw:count"], color=NEUTRAL, lw=2, ls="--", marker="o",
            ms=3.5, label="raw division count")
    for m in MODELS:
        ax.plot(allc.dist_um, allc[f"pearson:{m}"], color=COL[m], lw=2, marker="o",
                ms=3.5, label=f"{LABEL[m]} residual")
    ax.set_title("What the models remove (Pearson residuals)", loc="left", fontsize=10.5)
    for ax, t, ttl in ((axes[1], "pearson", "Pearson residuals"),
                       (axes[2], "midq", "Mid-quantile residuals")):
        if ax is not axes[0]:
            for m in MODELS:
                ax.plot(allc.dist_um, allc[f"{t}:{m}"], color=COL[m], lw=2, marker="o",
                        ms=3.5, label=LABEL[m])
            ax.plot(allc.dist_um, allc["pearson:mixture_permuted"], color=NEUTRAL, lw=1.2,
                    ls=":", label="permuted (noise floor)")
            ax.set_title(f"{ttl}, zoomed", loc="left", fontsize=10.5)
    for ax in axes:
        ax.axhline(0, color=INK2, lw=0.8)
        _style(ax)
        ax.set_xlabel("distance (um)")
        ax.legend(frameon=False, fontsize=8.5)
    axes[0].set_ylabel("Moran's I")
    fig.tight_layout()
    fig.savefig(f"{OUT}/correlogram_residuals.png", dpi=160)
    plt.close(fig)


def plot_lisa(L, xy):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5))
    lim = max(np.nanpercentile(np.abs(L[m]["mean"]), 98) for m in MODELS)
    codes = {"ns": 1, "HH": 2, "LL": 3, "HL": 4, "LH": 5, "island": 1}
    cmap = ListedColormap(["#ffffff", "#e4e3df", "#e34948", "#2a78d6", "#f4b4b3", "#a9cbf2"])
    for j, m in enumerate(MODELS):
        Lm = L[m]
        g = np.full((Lm["ny"], Lm["nx"]), np.nan)
        g[Lm["by"], Lm["bx"]] = Lm["mean"]
        ax = axes[0, j]
        im = ax.imshow(g, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                       interpolation="nearest")
        ax.set_title(f"{LABEL[m]}\nmean Pearson residual per {BIN_UM:.0f} um bin",
                     fontsize=10.5, loc="left")
        c = np.zeros((Lm["ny"], Lm["nx"]))
        c[Lm["by"], Lm["bx"]] = [codes[v] for v in Lm["cls"]]
        ax2 = axes[1, j]
        ax2.imshow(c, origin="lower", cmap=cmap, vmin=-0.5, vmax=5.5, interpolation="nearest")
        nhh, nll = int(np.sum(Lm["cls"] == "HH")), int(np.sum(Lm["cls"] == "LL"))
        ax2.set_title(f"local Moran's I clusters (FDR 5%)\nHH {nhh}, LL {nll} bins; "
                      f"global I = {Lm['I_global']:.3f}", fontsize=10.5, loc="left")
        for a in (ax, ax2):
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
    fig.colorbar(im, ax=axes[0, :].tolist(), shrink=0.7, label="mean residual")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#e34948", label="HH: high residual among high"),
                        Patch(color="#2a78d6", label="LL: low among low"),
                        Patch(color="#f4b4b3", label="HL: high among low"),
                        Patch(color="#a9cbf2", label="LH: low among high"),
                        Patch(color="#e4e3df", label="not significant")],
               loc="lower center", ncol=5, frameon=False, fontsize=9.5)
    fig.savefig(f"{OUT}/lisa_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_report(phi_cg, allc, lisa_df, chars, names, blocks):
    say("\n================ PhiSpace, I at the first three bands ================")
    t = phi_cg[phi_cg.dist_um.isin(MID[:3])].pivot_table(
        index="archetype", columns="dist_um",
        values=["I_indicator_all", "I_multivariate_within", "I_own_score_within"])
    say(t.round(3).to_string())
    say("\nrange: first band where I drops below 0.05")
    for nm in names:
        s = phi_cg[phi_cg.archetype == nm]
        out = []
        for col in ("I_indicator_all", "I_multivariate_within", "I_own_score_within"):
            below = s[s[col] < 0.05]
            out.append(f"{col.split('_')[1]} {below.band_lo.iloc[0]:.0f}um" if len(below)
                       else f"{col.split('_')[1]} >500um")
        say(f"  {nm:<20} " + " | ".join(out))
    say("\n================ residual correlograms ================")
    show = ["dist_um", "pairs", "raw:count"] + [f"pearson:{m}" for m in MODELS] + \
        [f"midq:{m}" for m in MODELS] + ["pearson:mixture_permuted"]
    say(allc[show].round(4).to_string(index=False))
    say("\n================ local Moran's I ================")
    say(lisa_df.round(4).to_string(index=False))
    say("\ncluster composition (share of cells by archetype):")
    keep = ["model", "cluster", "n_cells", "mean_count", "median_density", "mean_residual"]
    top = chars[keep].copy()
    top["top types"] = [", ".join(f"{t} {100 * r[f'share:{t}']:.0f}%" for t in
                                  sorted(names, key=lambda t: -r[f"share:{t}"])[:3])
                        for _, r in chars.iterrows()]
    say(top.round(3).to_string(index=False))
    say(f"\nblocks: {json.dumps(blocks)}")


if __name__ == "__main__":
    main()

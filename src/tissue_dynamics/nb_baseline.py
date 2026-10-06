"""Baseline negative binomial for the division counts, and the tests run on it.

The model is the division comparison's NB-c (`division.arms`): every covariate
the production mixture uses, placed in one NB mean,

    log mu_i = b0 + theta log N_i + s(log area_i) + eta(x_i)
    eta(x)   = phi'a + phibar'c + sum_r (phi'U_r)(phibar'V_r) [+ psi ox] + H'h + phi'K H

with the mixture's penalty (`weight_ridge` on every eta coefficient, `ridge_tau`
on U and V). `eta` is `arms.eta_state`, the block shared with the mixture, so a
change of specification here changes both models the same way. A fit takes
about a minute and a half on the full table, against hours for the mixture's
20 restarts, which is why this is the workhorse for specification tests.

Two switches are tuned in notebooks/nb_baseline.ipynb:

    theta       how log N enters:
                "offset"   pinned at 1 (production since 2026-10-06)
                "free"     one estimated exponent
                "spline"   offset plus a natural-spline deviation in log N
                "by_type"  offset plus a per-cell-type slope deviation,
                           (phi_i' t) (log N_i - mean log N)
    dispersion  "const" is one alpha; "state" is log alpha_i = la + phi_i' gamma

Oxygen is in the model only when the design carries an `oxygen` column; the
production config drops it, so `load_design` leaves it out and the oxygen tests
add it back explicitly. The neighbour block can be any width: stacking several
neighbourhood means (`stack_nbr`) gives each its own `c` and rows of `V`.
The neighbour block is removed by zeroing its columns, which holds its
coefficients at 0 through the penalty.

The rest of the module serves the tests: the oxygen field and its decoys,
annulus neighbourhoods, and the calibration and spatial diagnostics.
"""

from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np
import pandas as pd

from .division import arms

CACHE = "outputs/nb_baseline/cache"
DESIGN_KEYS = ("y", "logN", "area", "own", "nbr", "hbasis", "oxygen")
DEPTH_SPLINE_DF = 6

HIF_CORE = ["CA9", "NDRG1", "ADM", "EGLN3", "BNIP3", "ANGPTL4"]
HIF_EXTENDED = ["VEGFA", "SLC2A1", "PGK1", "LDHA", "P4HA1", "PDK1", "ERO1A", "LOX",
                "ALDOA", "ENO1", "MXI1", "KDM3A", "BHLHE40", "PFKFB3", "ANKRD37"]
ENDO_MARKERS = ["PECAM1", "CDH5", "CLDN5"]
OTHER_MARKERS = ["EPCAM", "CDH1", "CD3E", "CD3D", "DCN", "LUM", "CD68", "CSF1R"]
DECOYS = ("rot180", "flip_x", "flip_y")


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------
def cached(name, fn, refresh=False):
    """Load `CACHE/name.pkl`, or compute it with `fn()` and store it.

    Keys are explicit names, not hashes of the inputs: delete the file (or
    pass `refresh=True`) after changing anything upstream of it.
    """
    path = os.path.join(CACHE, name + ".pkl")
    if os.path.exists(path) and not refresh:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    out = fn()
    os.makedirs(CACHE, exist_ok=True)
    with open(path + ".tmp", "wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(path + ".tmp", path)
    return out


# --------------------------------------------------------------------------
# Design
# --------------------------------------------------------------------------
def load_design(cfg=None):
    """The production table, its frozen basis, the model spec and the design.

    Returns `(df, basis, spec, D)`; `D` holds numpy arrays keyed by
    `DESIGN_KEYS` (without `oxygen` when the config drops it), plus the
    depth-test columns `Nbasis` (a natural spline in log N, constant projected
    out) and `logNc` (centred log N). Subsets reuse the full-table basis: rows
    are dropped, the knots and centring constants are not refitted.
    """
    from .mixture.basis import _Spline

    from .mixture.basis import build_basis, spec_from_basis
    from .mixture.data import load_config
    from .mixture.model import build_data

    cfg = cfg or load_config()
    tp = cfg["data"]["table_parquet"]
    df = pd.read_parquet(tp)
    with open(tp.replace(".parquet", ".phi_names.json")) as fh:
        df.attrs["phi_names"] = json.load(fh)
    basis = build_basis(df, cfg, verbose=False)
    spec = spec_from_basis(basis, cfg, stage="B",
                           weight_ridge=float(cfg["model"]["weight_ridge"]))
    data = build_data(df, basis, cfg, spec)
    D = {k: np.asarray(getattr(data, k), dtype=np.float64) for k in DESIGN_KEYS
         if getattr(data, k) is not None}
    D["Nbasis"] = np.asarray(_Spline(D["logN"], DEPTH_SPLINE_DF)(D["logN"]), dtype=np.float64)
    D["logNc"] = D["logN"] - D["logN"].mean()
    return df, basis, spec, D


def subset(D, idx, **replace):
    """Rows `idx` of every design array, with any columns in `replace` swapped
    in (`None` removes a column, e.g. `oxygen=None`)."""
    out = {k: v[idx] for k, v in D.items()}
    for k, v in replace.items():
        if v is None:
            out.pop(k, None)
        else:
            out[k] = np.asarray(v, dtype=np.float64)
    return out


def project_nbr(P, basis):
    """Centre a (n, 9) neighbour-state block at its own mean and project it
    with the basis' sum-to-zero matrix, as `basis.nbr` does for the 80 um disc.
    Centring at its own mean keeps `a` comparable across neighbourhood
    definitions (a shift in phibar moves the bilinear term into `a`)."""
    C = P - P.mean(0)
    return C if basis.Q_nbr is None else C @ basis.Q_nbr


def stack_nbr(blocks, basis):
    """Several neighbourhood means side by side, each projected by `project_nbr`.

    With a stacked block, each neighbourhood gets its own main effect `c` and
    its own rows of `V`, so the bilinear term becomes
    `(phi'U)(phibar_1'V_1 + phibar_2'V_2 + ...)`: one set of own-state
    directions, read against every neighbourhood at once.
    """
    return np.hstack([project_nbr(P, basis) for P in blocks])


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------
def _log_mu(p, D):
    import jax.numpy as jnp

    depth = (p["theta"] if "theta" in p else 1.0) * D["logN"]
    if "tN" in p:                                  # theta="spline"
        depth = depth + D["Nbasis"] @ p["tN"]
    if "tT" in p:                                  # theta="by_type"
        depth = depth + (D["own"] @ p["tT"]) * D["logNc"]
    eta = p["b0"] + depth + D["area"] @ p["s"] + arms.eta_state(p, D)
    return jnp.clip(eta, -30.0, 30.0)


def _log_alpha(p, D):
    return p["la"] + D["own"] @ p["gamma"] if "gamma" in p else p["la"]


THETA_OPTIONS = ("offset", "free", "spline", "by_type")


def _start(D, theta, dispersion, rank, seed, init):
    """Cold start for this design and options, overlaid with `init` wherever a
    block of `init` has the same shape (so a warm start survives a change of
    neighbour-block width or of the depth and oxygen terms)."""
    import jax.numpy as jnp

    if theta not in THETA_OPTIONS:
        raise ValueError(f"theta must be one of {THETA_OPTIONS}")
    p = arms.init_nb(D, "state" if dispersion == "state" else "const", rank, seed,
                     oxygen="oxygen" in D, theta_free=(theta == "free"))
    if theta == "spline":
        p["tN"] = jnp.zeros(D["Nbasis"].shape[1])
    if theta == "by_type":
        p["tT"] = jnp.zeros(D["own"].shape[1])
    if init is not None:
        for k, v in init.items():
            if k in p and np.shape(v) == np.shape(p[k]):
                p[k] = jnp.asarray(np.asarray(v))
    return p


def n_params(params):
    return int(sum(np.size(v) for v in params.values()))


def fit(D, *, theta="offset", dispersion="const", rank=2, init=None, seed=0,
        weight_ridge=10.0, n_reference=678885, ridge_tau=1.0, b_penalty="frobenius",
        se=True, say=None, retries=2):
    """Fit the baseline NB by penalised maximum likelihood.

    `init` warm-starts from another fit's `params`: blocks of matching shape are
    copied, the rest start cold. Do not warm-start from a fit whose neighbour
    block was zeroed: its U and V sit at 0, a saddle for the bilinear term.

    A fit that misses the convergence criteria is retried from a cold start
    (seeds `seed`, `seed + 1`, ...) up to `retries` times, as the division
    comparison's Stage 2 does: the Newton finish occasionally stalls short of
    grad_tol at the optimum the other starts reach. The converged attempt with
    the lowest objective is returned; `info["attempts"]` records how many ran.

    Returns a dict: `params`, `loglik` (data term only), `objective`
    (penalised), `mu`, `alpha` (a float, or per cell when dispersion="state"),
    `info` (convergence), and `cov` when `se` -- the inverse of the penalised
    Hessian, with the U/V rotation direction (exactly flat) projected out.
    """
    kw = dict(theta=theta, dispersion=dispersion, rank=rank, weight_ridge=weight_ridge,
              n_reference=n_reference, ridge_tau=ridge_tau, b_penalty=b_penalty,
              se=se, say=say)
    tries = [_fit_once(D, init=init, seed=seed, **kw)]
    while not tries[-1]["info"]["converged"] and len(tries) <= retries:
        if say:
            say(f"      not converged; retrying from a cold start, seed {seed + len(tries) - 1}")
        tries.append(_fit_once(D, init=None, seed=seed + len(tries) - 1, **kw))
    ok = [t for t in tries if t["info"]["converged"]] or tries
    best = min(ok, key=lambda t: t["objective"])
    best["info"]["attempts"] = len(tries)
    return best


def _fit_once(D, *, theta, dispersion, rank, init, seed, weight_ridge, n_reference,
              ridge_tau, b_penalty, se, say):
    arms._jax()
    import jax.numpy as jnp

    n = len(D["y"])
    jD = {k: jnp.asarray(v) for k, v in D.items()}
    p0 = _start(D, theta, dispersion, rank, seed, init)

    def f(p):
        ll = jnp.sum(arms.nb_logpmf(jD["y"], _log_mu(p, jD), _log_alpha(p, jD)))
        return -ll + arms.penalty(p, n, weight_ridge, n_reference, ridge_tau, b_penalty)

    t0 = time.time()
    p, info = arms.minimise(f, p0, say=say)
    log_mu = _log_mu(p, jD)
    log_al = jnp.broadcast_to(_log_alpha(p, jD), (n,))
    ll = float(jnp.sum(arms.nb_logpmf(jD["y"], log_mu, log_al)))
    out = dict(params={k: np.asarray(v) for k, v in p.items()}, loglik=ll,
               objective=float(info["f"]), mu=np.asarray(jnp.exp(log_mu)),
               alpha=(np.asarray(jnp.exp(log_al)) if dispersion == "state"
                      else float(np.exp(np.asarray(p["la"])))),
               theta=theta, dispersion=dispersion, n=n,
               info={k: v for k, v in info.items() if k != "newton_trace"})
    if se:
        out["cov"] = covariance(f, p)
    out["info"]["minutes"] = (time.time() - t0) / 60.0
    if say:
        say(f"      loglik {ll:,.2f}; converged {info['converged']}; "
            f"{out['info']['minutes']:.1f} min")
    return out


def covariance(f, p, chunk=8):
    """Inverse penalised Hessian, by chunked Hessian-vector products.

    `B = U V'` is unchanged by rotating U and V together, and so is either form
    of the penalty (the ridge on U V' and the balancing term ||U'U - V'V||^2, or
    ||U||^2 + ||V||^2), so the Hessian has an exactly flat direction.
    Eigenvalues below 1e-8 x the largest are dropped (a pseudo-inverse); every
    genuine direction carries data curvature far above that floor. Standard errors from this are model-based: they
    ignore spatial autocorrelation and so are too small (see the notebook).
    """
    jax = arms._jax()
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree

    x, unravel = ravel_pytree(p)
    g = jax.grad(lambda z: f(unravel(z)))
    hv = jax.jit(lambda z, V: jax.vmap(lambda v: jax.jvp(g, (z,), (v,))[1])(V))
    d = x.shape[0]
    E = np.eye(d)
    H = np.vstack([np.asarray(hv(x, jnp.asarray(E[i:i + chunk]))) for i in range(0, d, chunk)])
    H = 0.5 * (H + H.T)
    w, Q = np.linalg.eigh(H)
    keep = w > 1e-8 * w.max()
    C = (Q[:, keep] / w[keep]) @ Q[:, keep].T
    idx = unravel(jnp.arange(d, dtype=jnp.float64))
    index = {k: np.rint(np.asarray(v)).astype(int).reshape(np.shape(v)) for k, v in idx.items()}
    return dict(C=C, index=index, n_dropped=int((~keep).sum()), min_eig=float(w.min()))


def estimate(fitres, block, M=None):
    """Value and covariance of `M @ vec(params[block])` (M defaults to identity)."""
    v = np.asarray(fitres["params"][block], dtype=float).ravel()
    ii = fitres["cov"]["index"][block].ravel()
    C = fitres["cov"]["C"][np.ix_(ii, ii)]
    if M is None:
        return v, C
    M = np.atleast_2d(M)
    return M @ v, M @ C @ M.T


def state_effects(fitres, basis, block):
    """Own (`a`), neighbour (`c`) or dispersion (`gamma`) effects in cell-type
    coordinates.

    The fit sees the state through the sum-to-zero projection, so the effect of
    cell type k is `(Q a)_k`: the change in log mu (or log alpha, for `gamma`)
    per unit of type-k score, relative to the mean over types (the nine effects
    sum to zero).
    """
    Q = basis.Q_nbr if block == "c" else basis.Q_own
    names = basis.nbr_names if block == "c" else basis.own_names
    v, C = estimate(fitres, block, Q)
    se = np.sqrt(np.clip(np.diag(C), 0, None))
    return pd.DataFrame({"estimate": v, "se": se, "z": v / se}, index=names)


# --------------------------------------------------------------------------
# Calibration and spatial diagnostics
# --------------------------------------------------------------------------
def _nb_rp(mu, alpha):
    alpha = np.broadcast_to(np.asarray(alpha, dtype=float), np.shape(mu))
    r = 1.0 / alpha
    return r, r / (r + mu), alpha


def rootogram(y, mu, alpha, groups=None, kmax=30, chunk=50000):
    """Observed and expected frequencies of 0..kmax-1, tail pooled at kmax.

    With `groups`, returns `{group: (obs, exp)}`; otherwise `(obs, exp)`.
    """
    from scipy import stats

    r, pr, _ = _nb_rp(mu, alpha)
    labels = np.zeros(len(y), dtype=int) if groups is None else pd.factorize(groups)[0]
    names = [None] if groups is None else list(pd.factorize(groups)[1])
    G = len(names)
    ks = np.arange(kmax)
    E = np.zeros((G, kmax))
    for s in range(0, len(y), chunk):
        pmf = stats.nbinom.pmf(ks[None, :], r[s:s + chunk, None], pr[s:s + chunk, None])
        lab = labels[s:s + chunk]
        for g in range(G):
            E[g] += pmf[lab == g].sum(0)
    out = {}
    for g, nm in enumerate(names):
        m = labels == g
        obs = np.array([np.sum(y[m] == k) for k in ks] + [np.sum(y[m] >= kmax)], float)
        exp_ = np.append(E[g], max(m.sum() - E[g].sum(), 0.0))
        out[nm] = (obs, exp_)
    return out[None] if groups is None else out


def misfit(obs, exp_):
    """Rootogram misfit: sum over counts of (sqrt obs - sqrt exp)^2."""
    return float(np.sum((np.sqrt(obs) - np.sqrt(exp_)) ** 2))


def pit(y, mu, alpha, seed=0):
    """Randomised PIT (non-randomised PIT is conservative for counts)."""
    from scipy import stats

    r, pr, _ = _nb_rp(mu, alpha)
    F = stats.nbinom.cdf(y, r, pr)
    Fm = np.where(y > 0, stats.nbinom.cdf(y - 1, r, pr), 0.0)
    return Fm + np.random.default_rng(seed).uniform(size=len(y)) * (F - Fm)


def pit_maxdev(u, bins=10):
    h = np.histogram(u, bins=bins, range=(0, 1))[0] / (len(u) / bins)
    return float(np.max(np.abs(h - 1.0)))


def pearson(y, mu, alpha):
    _, _, a = _nb_rp(mu, alpha)
    return (y - mu) / np.sqrt(mu + a * mu ** 2)


def tiles(x, y, tile=100.0, min_cells=5):
    """Tile index per cell and the kept tiles' centres (tiles with >= min_cells)."""
    tx = np.floor((x - x.min()) / tile).astype(np.int64)
    ty = np.floor((y - y.min()) / tile).astype(np.int64)
    key = tx * (ty.max() + 1) + ty
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    keep = cnt >= min_cells
    cx = x.min() + (uk // (ty.max() + 1) + 0.5) * tile
    cy = y.min() + (uk % (ty.max() + 1) + 0.5) * tile
    return dict(inv=inv, keep=keep, n_tiles=len(uk), xy=np.column_stack([cx, cy])[keep],
                count=cnt[keep])


def tile_moran(v, T, permutations=999, seed=0):
    """Moran's I of the tile means of `v` (8-nearest-neighbour weights)."""
    from .evaluate import morans_i

    s = np.bincount(T["inv"], weights=v)
    c = np.bincount(T["inv"])
    m = (s / c)[T["keep"]]
    return morans_i(m, coords=T["xy"], n_neighbors=8, permutations=permutations, seed=seed)


def summarise(fitres, y, T=None, groups=None, kmax=30):
    """One row of fit, calibration and spatial diagnostics for a fit."""
    a = fitres["alpha"]
    obs, exp_ = rootogram(y, fitres["mu"], a, kmax=kmax)
    d = np.sqrt(obs) - np.sqrt(exp_)
    row = dict(k=n_params(fitres["params"]), loglik=fitres["loglik"],
               objective=fitres["objective"],
               theta=float(fitres["params"].get("theta", 1.0)),
               alpha=(a if np.isscalar(a) else float(np.median(a))),
               rootogram_misfit=misfit(obs, exp_), worst_bin=float(np.max(np.abs(d))),
               worst_count=int(np.argmax(np.abs(d))),
               pit_maxdev=pit_maxdev(pit(y, fitres["mu"], a)),
               converged=bool(fitres["info"]["converged"]),
               minutes=float(fitres["info"]["minutes"]))
    if T is not None:
        row["resid_moran"] = float(tile_moran(pearson(y, fitres["mu"], a), T,
                                              permutations=0)["I"])
    return row


# --------------------------------------------------------------------------
# Every cell: position, depth, typing, PhiSpace and marker counts
# --------------------------------------------------------------------------
def all_cells(cfg, genes=None):
    """Every segmented cell with a PhiSpace score, before any filtering.

    Reproduces `mixture.data.build_table`'s library size `N` (total counts minus
    the Tirosh genes and the blocklist) and its `typed` flag (N >=
    min_library_size), so the oxygen source and the annulus neighbour pool can
    be rebuilt exactly. Adds raw counts for `genes` (default: the HIF, endothelial
    and other-lineage marker lists) as `g_<gene>` columns.
    """
    from .io_xenium import read_xenium
    from .mixture.data import _tirosh_genes

    dc = cfg["data"]
    ad = read_xenium(dc["xenium_dir"], sample_id="BC")
    X = ad.layers["counts"] if "counts" in ad.layers else ad.X
    names = np.asarray(ad.var_names)
    pos = {g: i for i, g in enumerate(names)}
    tir = [g for g in _tirosh_genes() if g in pos]
    block = list(pd.read_csv(dc["blocklist_csv"], header=None)[0].astype(str)) \
        if dc.get("blocklist_csv") else []
    excl = sorted({pos[g] for g in tir} | {pos[g] for g in block if g in pos})
    total = np.asarray(X.sum(axis=1)).ravel()
    N = total - np.asarray(X[:, excl].sum(axis=1)).ravel()
    out = pd.DataFrame({"cell_id": np.asarray(ad.obs["cell_id"], dtype=str),
                        "x": np.asarray(ad.obsm["spatial"], dtype=float)[:, 0],
                        "y": np.asarray(ad.obsm["spatial"], dtype=float)[:, 1],
                        "N": np.rint(N).astype(np.int64),
                        "total": np.rint(total).astype(np.int64)})
    genes = genes or (HIF_CORE + HIF_EXTENDED + ENDO_MARKERS + OTHER_MARKERS)
    present = [g for g in genes if g in pos]
    G = X[:, [pos[g] for g in present]]
    G = np.asarray(G.todense() if hasattr(G, "todense") else G)
    for j, g in enumerate(present):
        out[f"g_{g}"] = G[:, j].astype(np.int32)

    P = pd.read_csv(dc["phi_csv"], index_col=0)
    P.index = P.index.astype(str)
    P = P[list(dc["phi_columns"])] if dc.get("phi_columns") else P
    out = out.loc[out["cell_id"].isin(P.index)].reset_index(drop=True)
    Phi = P.reindex(out["cell_id"]).to_numpy(float)
    ok = ~np.isnan(Phi).any(axis=1)
    out, Phi = out.loc[ok].reset_index(drop=True), Phi[ok]
    for j, nm in enumerate(P.columns):
        out[f"phi_{nm}"] = Phi[:, j]
    out["typed"] = out["N"].to_numpy() >= int(dc.get("min_library_size", 0))
    out.attrs["phi_names"] = list(P.columns)
    out.attrs["genes"] = present
    return out


# --------------------------------------------------------------------------
# Oxygen field
# --------------------------------------------------------------------------
def oxygen_source(cells, cfg, consumers="typed"):
    """The field's source exactly as `build_table` builds it.

    The source score is the Endothelial PhiSpace score of typed cells, z-scored,
    minus its (1 - source_quantile) quantile, so the top 2.6% are positive.
    `consumers` decides which cells enter the PDE's density (consumption) map:
    "typed" reproduces the production field, in which shallow cells do not
    consume oxygen; "all" adds them as zero-weight cells.
    """
    oc = cfg["oxygen"]
    t = cells["typed"].to_numpy()
    e = cells["phi_Endothelial"].to_numpy(float)
    z = (e - e[t].mean()) / e[t].std()
    q = float(np.quantile(z[t], 1.0 - float(oc["source_quantile"])))
    score = z - q
    if consumers == "typed":
        return cells.loc[t, ["x", "y"]].to_numpy(float), score[t]
    score = np.where(t, score, -1e3)            # zero weight after the soft threshold
    return cells[["x", "y"]].to_numpy(float), score


def solve_field(xy, score, L0, cfg):
    from .oxygen import build_oxygen_field

    oc = cfg["oxygen"]
    return build_oxygen_field(xy, score, h=float(oc["h_um"]), L0=float(L0),
                              beta=float(oc["beta"]), c=0.0)


def transformed(res, kind=None):
    """The field raster and tissue mask, unchanged or as a decoy.

    Decoys keep the field's smoothness and marginal distribution exactly and
    break its alignment with the tissue: `rot180` is a point reflection through
    the raster centre, `flip_x` and `flip_y` are mirror images.
    """
    sl = {None: (slice(None), slice(None)), "rot180": (slice(None, None, -1),) * 2,
          "flip_x": (slice(None, None, -1), slice(None)),
          "flip_y": (slice(None), slice(None, None, -1))}[kind]
    return res.u[sl], res.tissue[sl]


def sample_field(res, xy, kind=None):
    """`u` at the cells, NaN where a cell falls outside the (transformed) tissue."""
    from .oxygen import sample_at_cells

    u, t = transformed(res, kind)
    v, _, ok = sample_at_cells(u, np.asarray(xy, dtype=float), res.origin, res.h, tissue=t)
    return np.where(ok, v, np.nan)


def field_samples(cells_xy, src_xy, src_score, L0, cfg, keep_raster=False, window=None):
    """Solve at `L0` and sample the field and every decoy at `cells_xy`."""
    t0 = time.time()
    res = solve_field(src_xy, src_score, L0, cfg)
    out = dict(L0=float(L0), u=sample_field(res, cells_xy),
               decoys={k: sample_field(res, cells_xy, k) for k in DECOYS},
               diagnostics=res.diagnostics, origin=res.origin, h=res.h,
               minutes=(time.time() - t0) / 60.0)
    if keep_raster:
        out["raster"] = np.where(res.tissue, res.u, np.nan).astype(np.float32)
    if window is not None:
        (i0, i1), (j0, j1) = window
        out["window"] = np.where(res.tissue, res.u, np.nan)[i0:i1, j0:j1].astype(np.float32)
    return out


# --------------------------------------------------------------------------
# Annulus neighbourhoods
# --------------------------------------------------------------------------
RINGS = ((0.0, 15.0), (15.0, 30.0), (30.0, 50.0), (50.0, 80.0), (80.0, 120.0), (120.0, 200.0))


def ring_counts(pool_xy, focal, r):
    """Number of other pool cells within `r` of each focal cell."""
    from scipy.spatial import cKDTree

    tree = cKDTree(pool_xy)
    return tree.query_ball_point(pool_xy[focal], r, return_length=True) - 1


def ring_means(pool_xy, pool_phi, focal, Z, rings=RINGS, max_neighbors=None, seed=0,
               chunk=6000):
    """Ring sums of the neighbour state, and the pair correlogram, in one pass.

    For each focal cell (an index into the pool) and each ring
    `r_in < d <= r_out`, sums `pool_phi` over the pool cells in the ring
    (excluding the cell itself). With `max_neighbors`, also the count-matched
    version: each ring thinned at random to at most that many neighbours, so
    every ring's mean averages the same number of cells.

    `Z` (pool, p) holds standardised variables; the correlogram accumulates
    `sum z_i z_j` over every focal-pool pair in each ring, so `rho = num / cnt`
    is Moran's I with binary ring weights. It is computed over all pairs, never
    the thinned ones: it describes the tissue, not the estimator.

    Never holds more than one chunk's pairs (~5 x 10^6) in memory.
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    tree = cKDTree(pool_xy)
    nr, d, p = len(rings), pool_phi.shape[1], Z.shape[1]
    edges = np.array([rings[0][0]] + [r[1] for r in rings])
    S = np.zeros((nr, len(focal), d))
    C = np.zeros((nr, len(focal)))
    Sm = np.zeros_like(S) if max_neighbors else None
    Cm = np.zeros_like(C) if max_neighbors else None
    num, cnt = np.zeros((nr, p)), np.zeros(nr)
    for s in range(0, len(focal), chunk):
        fc = focal[s:s + chunk]
        sp = cKDTree(pool_xy[fc]).sparse_distance_matrix(tree, edges[-1], output_type="ndarray")
        i, j, dist = sp["i"], sp["j"], sp["v"]
        ok = j != fc[i]
        i, j, dist = i[ok], j[ok], dist[ok]
        b = np.searchsorted(edges, dist, side="left") - 1      # (r_in, r_out]
        ok = (b >= 0) & (b < nr)
        i, j, b = i[ok], j[ok], b[ok]
        cnt += np.bincount(b, minlength=nr)
        prod = Z[fc[i]] * Z[j]
        for c in range(p):
            num[:, c] += np.bincount(b, weights=prod[:, c], minlength=nr)
        for r in range(nr):
            m = b == r
            ii, jj = i[m], j[m]
            C[r, s:s + len(fc)] = np.bincount(ii, minlength=len(fc))
            for c in range(d):
                S[r, s:s + len(fc), c] = np.bincount(ii, weights=pool_phi[jj, c],
                                                     minlength=len(fc))
            if max_neighbors:
                order = np.lexsort((rng.random(len(ii)), ii))
                ii_o, jj_o = ii[order], jj[order]
                start = np.searchsorted(ii_o, np.arange(len(fc)))
                rank = np.arange(len(ii_o)) - start[ii_o]
                k = rank < max_neighbors
                Cm[r, s:s + len(fc)] = np.bincount(ii_o[k], minlength=len(fc))
                for c in range(d):
                    Sm[r, s:s + len(fc), c] = np.bincount(ii_o[k], weights=pool_phi[jj_o[k], c],
                                                          minlength=len(fc))
    return dict(rings=rings, sum=S, count=C, sum_matched=Sm, count_matched=Cm,
                rho=num / cnt[:, None], n_pairs=cnt, max_neighbors=max_neighbors)


def kernel_means(pool_xy, pool_phi, focal, ells, rmax=200.0, chunk=6000):
    """Distance-weighted neighbour means, `w(d) = exp(-d / ell)` out to `rmax`.

    For each focal cell and each `ell`, returns the weighted sum of `pool_phi`
    over pool cells within `rmax` (the cell itself excluded), the weight sum,
    and the sum of squared weights, so `mean = sum / W` and the effective
    number of neighbours averaged is `W^2 / W2`. All `ells` share one pass.

    `rmax` truncates the kernel; at 200 um an ell of 80 um keeps weight
    exp(-2.5) = 0.08 at the edge, so the largest ells are mildly truncated.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(pool_xy)
    ells = np.asarray(ells, dtype=float)
    ne, d = len(ells), pool_phi.shape[1]
    S = np.zeros((ne, len(focal), d))
    W = np.zeros((ne, len(focal)))
    W2 = np.zeros((ne, len(focal)))
    for s in range(0, len(focal), chunk):
        fc = focal[s:s + chunk]
        sp = cKDTree(pool_xy[fc]).sparse_distance_matrix(tree, rmax, output_type="ndarray")
        i, j, dist = sp["i"], sp["j"], sp["v"]
        ok = j != fc[i]
        i, j, dist = i[ok], j[ok], dist[ok]
        for e, ell in enumerate(ells):
            w = np.exp(-dist / ell)
            W[e, s:s + len(fc)] = np.bincount(i, weights=w, minlength=len(fc))
            W2[e, s:s + len(fc)] = np.bincount(i, weights=w * w, minlength=len(fc))
            for c in range(d):
                S[e, s:s + len(fc), c] = np.bincount(i, weights=w * pool_phi[j, c],
                                                     minlength=len(fc))
    return dict(ells=ells, rmax=float(rmax), sum=S, W=W, W2=W2)


def heldout_loglik(fitres, D):
    """Per-cell NB log-likelihood of `D` under a fit's parameters (constant
    or state-dependent dispersion), for spatial cross-validation."""
    arms._jax()
    import jax.numpy as jnp

    jD = {k: jnp.asarray(v) for k, v in D.items()}
    p = {k: jnp.asarray(v) for k, v in fitres["params"].items()}
    log_al = jnp.broadcast_to(_log_alpha(p, jD), (len(D["y"]),))
    return np.asarray(arms.nb_logpmf(jD["y"], _log_mu(p, jD), log_al))

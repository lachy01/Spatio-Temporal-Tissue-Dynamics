"""Expanding-ring (annulus) neighbourhood analysis: is the neighbourhood effect real?

The fate/count models regress on `phi_bar`, the mean neighbour state inside one
disk. That conflates two very different things:

  * a genuine neighbourhood effect (a cell's fate depends on who surrounds it), and
  * **segmentation spillover** — transcripts from a touching neighbour assigned to
    this cell, so `phi_i` is contaminated by `phi_j` and `phi_bar` merely echoes
    the cell's own (mis-measured) state.

Spillover is a *contact* artefact: it can only act between cells whose boundaries
touch, i.e. at the very shortest distances. A real signalling effect need not be.
So refit the model with the neighbourhood restricted to successive **annuli**
(0-15, 15-30, ... um) and watch the neighbour coefficients `c` (and the
interaction `B`) as the ring moves outward.

The quantitative null: under pure spillover the apparent coefficient at distance
`r` is inherited entirely from how much the ring's mean state still resembles the
cell's own state, i.e.

    c_hat(r)  =  k * rho_phi(r)

where `rho_phi(r)` is the spatial autocorrelation of the Phi-Space vectors at
distance `r` (the correlogram). `spillover_chi2` fits `k` and tests the deviation.
If `c_hat(r)` decays *with* `rho_phi(r)`, the "neighbourhood effect" is consistent
with an artefact; if it persists at ranges where `rho_phi` has decayed away, it is
not.

Both the ring averages and the correlogram are accumulated chunk-wise: a 200um
neighbourhood holds ~1000 cells, so the full pair set for one slide is ~7e8 pairs
and must never be materialized.
"""

import numpy as np

# Contact-scale to signalling-scale rings, in micrometers (r_inner, r_outer].
DEFAULT_RINGS = (
    (0.0, 15.0), (15.0, 30.0), (30.0, 50.0),
    (50.0, 80.0), (80.0, 120.0), (120.0, 200.0),
)


def _subsample_rows(A, max_neighbors, rng):
    """Randomly thin each CSR row to at most `max_neighbors` stored entries.

    Used to hold the number of averaged neighbours *constant* across rings:
    a wide ring averages far more cells than a narrow one, so its `phi_bar` is a
    less noisy regressor and suffers less regression dilution — which would
    inflate the coefficient for reasons that have nothing to do with distance.
    """
    cnt = np.diff(A.indptr)
    if cnt.size == 0 or cnt.max() <= max_neighbors:
        return A
    # rank entries within each row by a random draw, then keep the first k:
    # lexsort groups by row (ascending) and orders by the draw inside each group,
    # so an entry's position minus its row's start is its within-row rank.
    rows = np.repeat(np.arange(cnt.size), cnt)
    order = np.lexsort((rng.random(A.nnz), rows))
    within = np.arange(A.nnz) - A.indptr[:-1][rows[order]]
    keep = np.zeros(A.nnz, dtype=bool)
    keep[order] = within < max_neighbors
    A.data = A.data * keep
    A.eliminate_zeros()
    return A


def annulus_neighbor_stats(adata, r_inner, r_outer, phi_key="phi", spatial_key="spatial",
                            sample_key="sample_id", chunk_size=10000,
                            max_neighbors=None, seed=0):
    """Ring-averaged neighbour state and the Phi-Space correlogram at that ring.

    For every cell, averages `obsm[phi_key]` over the neighbours `j` with
    `r_inner < d(i,j) <= r_outer` (per section; the cell itself is excluded,
    since its own distance is 0). In the same pass it accumulates the exact
    pairwise moments needed for the correlogram, so `rho_phi` is computed over
    *every* pair in the ring without ever holding them in memory.

    `max_neighbors`, if given, randomly thins each cell's ring to at most that
    many neighbours — the count-matched control: it equalizes the averaging
    noise in `phi_bar` across rings, so coefficients can be compared without the
    wider rings gaining an advantage purely from less regression dilution.

    Returns a dict with:
      `phi_bar`   (n, D) ring-mean neighbour state (0 for cells with no neighbour
                  in the ring), `counts` (n,) neighbours per cell in the ring,
      `rho_pair`  (D,) per-dimension correlogram `Corr(phi_i, phi_j)` over ring
                  pairs — the `rho_phi(r)` of the spillover null,
      `rho_own_bar` (D,) `Corr(phi_i, phi_bar_i(r))`, the regression-relevant
                  attenuated version (reported alongside for transparency),
      `n_pairs`, `frac_isolated`.
    """
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors

    coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    phi = np.asarray(adata.obsm[phi_key], dtype=float)
    n, D = phi.shape
    phi_sq = phi ** 2

    phi_bar = np.zeros((n, D))
    counts = np.zeros(n)
    # pairwise moment accumulators (i = focal cell, j = its ring neighbour)
    n_pairs = 0.0
    s_i = np.zeros(D); s_j = np.zeros(D)
    s_ii = np.zeros(D); s_jj = np.zeros(D); s_ij = np.zeros(D)

    samples = (adata.obs[sample_key].to_numpy() if sample_key in adata.obs.columns
               else np.zeros(n, dtype=int))
    rng = np.random.default_rng(seed)

    for s in pd.unique(samples):
        idx = np.flatnonzero(samples == s)
        if len(idx) < 2:
            continue
        c_s = coords[idx]
        phi_s = phi[idx]
        phi_sq_s = phi_sq[idx]
        nn = NearestNeighbors(radius=r_outer).fit(c_s)

        for start in range(0, len(idx), chunk_size):
            sl = slice(start, min(start + chunk_size, len(idx)))
            # distances live in .data, so the ring is a mask on the values;
            # d == 0 (the cell itself) is excluded by the strict > r_inner test
            A = nn.radius_neighbors_graph(c_s[sl], radius=r_outer, mode="distance")
            A.data = (A.data > r_inner).astype(np.float64)
            A.eliminate_zeros()
            if max_neighbors is not None:
                A = _subsample_rows(A, int(max_neighbors), rng)

            cnt = np.asarray(A.sum(axis=1)).ravel()
            S = A @ phi_s                       # sum of neighbour phi over the ring
            Q = A @ phi_sq_s                    # sum of neighbour phi^2 over the ring
            tgt = idx[sl]
            focal = phi_s[sl]

            counts[tgt] = cnt
            nz = cnt > 0
            phi_bar[tgt[nz]] = S[nz] / cnt[nz, None]

            n_pairs += float(cnt.sum())
            s_i += (cnt[:, None] * focal).sum(0)
            s_ii += (cnt[:, None] * focal ** 2).sum(0)
            s_j += S.sum(0)
            s_jj += Q.sum(0)
            s_ij += (focal * S).sum(0)
            del A, S, Q

    if n_pairs == 0:
        raise ValueError(f"no cell pairs fall in the ring ({r_inner}, {r_outer}] um")

    m_i, m_j = s_i / n_pairs, s_j / n_pairs
    v_i = np.maximum(s_ii / n_pairs - m_i ** 2, 0.0)
    v_j = np.maximum(s_jj / n_pairs - m_j ** 2, 0.0)
    cov = s_ij / n_pairs - m_i * m_j
    denom = np.sqrt(v_i * v_j)
    rho_pair = np.divide(cov, denom, out=np.zeros(D), where=denom > 0)

    ok = counts > 0
    rho_own_bar = np.array([
        np.corrcoef(phi[ok, d], phi_bar[ok, d])[0, 1] if ok.sum() > 2 else np.nan
        for d in range(D)
    ])

    return {
        "phi_bar": phi_bar, "counts": counts,
        "rho_pair": rho_pair, "rho_own_bar": rho_own_bar,
        "n_pairs": n_pairs, "frac_isolated": float((~ok).mean()),
    }


def annulus_scan(adata, rings=DEFAULT_RINGS, fit_fn=None, interaction=True,
                  standardize_phi_bar=True, max_neighbors=None, phi_key="phi",
                  phi_bar_key="phi_bar", spatial_key="spatial", sample_key="sample_id",
                  chunk_size=10000, seed=0, verbose=False, **fit_kwargs):
    """Refit the count model once per annulus and tabulate the neighbour effect.

    For each `(r_inner, r_outer)` ring: recompute `obsm[phi_bar_key]` from that
    ring only, refit `fit_fn` (default `count_glm.fit_division_nb`) with the same
    `**fit_kwargs`, and record the neighbour coefficients `c` with their standard
    errors, plus the ring's Phi-Space correlogram.

    `standardize_phi_bar` (default) z-scores each ring's `phi_bar` before fitting.
    This matters: averaging over a wider ring shrinks the variance of `phi_bar`,
    which would inflate raw coefficients purely mechanically. Standardizing makes
    `c(r)` an effect *per SD of the ring's mean state* — comparable across rings,
    and the scale on which the `c ∝ rho_phi` spillover prediction is stated.

    `max_neighbors` runs the **count-matched control**: each ring is thinned to
    the same number of neighbours per cell, so the wider rings no longer benefit
    from averaging away more noise (less regression dilution). Set it to roughly
    the innermost ring's mean neighbour count to compare rings on equal footing.

    With `interaction`, each ring is also fit with the full dense `B`
    (`interaction_rank=None`) and the block's size (Frobenius norm, mean |B|) and
    likelihood-ratio test against the main-effects fit are recorded.

    Returns `(coef, summary)` DataFrames: `coef` has one row per (ring, phi
    dimension) with `c`, `se`, `z`, `rho_pair`, `rho_own_bar`; `summary` has one
    row per ring with the aggregate `c_norm`/`c_mean_abs`, `rho_phi` (mean over
    dimensions), neighbour counts, and — if `interaction` — `B_fro`, `B_mean_abs`
    and `lr_interaction`.
    """
    import pandas as pd
    from scipy import stats as sps

    from .count_glm import fit_division_nb

    fit_fn = fit_fn or fit_division_nb
    phi_names = list(adata.uns.get(f"{phi_key}_names",
                                    [str(j) for j in range(adata.obsm[phi_key].shape[1])]))

    coef_rows, summary_rows = [], []
    for r_in, r_out in rings:
        stats = annulus_neighbor_stats(
            adata, r_in, r_out, phi_key=phi_key, spatial_key=spatial_key,
            sample_key=sample_key, chunk_size=chunk_size,
            max_neighbors=max_neighbors, seed=seed)

        pb = stats["phi_bar"]
        if standardize_phi_bar:
            sd = pb.std(0)
            sd[sd == 0] = 1.0
            pb = (pb - pb.mean(0)) / sd
        adata.obsm[phi_bar_key] = pb

        fit = fit_fn(adata, predictors="own+neighbor", interaction_rank=0, **fit_kwargs)
        ll_main, k_main = fit["loglik"], fit["n_params"]
        cf = fit["coef"].set_index("term")
        del fit

        row = {"r_inner": r_in, "r_outer": r_out,
               "mean_neighbors": float(stats["counts"].mean()),
               "frac_isolated": stats["frac_isolated"],
               "n_pairs": stats["n_pairs"],
               "rho_phi": float(np.mean(stats["rho_pair"])),
               "rho_own_bar": float(np.nanmean(stats["rho_own_bar"])),
               "loglik_main": ll_main}

        for d, nm in enumerate(phi_names):
            term = next((t for t in (f"nbr[{nm}]", f"nbr_dev[{nm}]") if t in cf.index), None)
            if term is None:
                continue
            coef_rows.append({
                "r_inner": r_in, "r_outer": r_out, "dim": nm,
                "c": float(cf.loc[term, "estimate"]), "se": float(cf.loc[term, "std_error"]),
                "z": float(cf.loc[term, "z"]),
                "rho_pair": float(stats["rho_pair"][d]),
                "rho_own_bar": float(stats["rho_own_bar"][d]),
            })

        cvals = np.array([r["c"] for r in coef_rows if r["r_inner"] == r_in])
        row["c_norm"] = float(np.linalg.norm(cvals))
        row["c_mean_abs"] = float(np.abs(cvals).mean())

        if interaction:
            fi = fit_fn(adata, predictors="own+neighbor", interaction_rank=None, **fit_kwargs)
            B = fi["coef"].set_index("term").filter(like="int[", axis=0)["estimate"].to_numpy()
            lr = 2.0 * (fi["loglik"] - ll_main)
            df = fi["n_params"] - k_main
            row["B_fro"] = float(np.linalg.norm(B))
            row["B_mean_abs"] = float(np.abs(B).mean())
            row["lr_interaction"] = float(lr)
            row["lr_interaction_p"] = float(sps.chi2.sf(lr, df)) if df > 0 and lr > 0 else 1.0
            del fi

        summary_rows.append(row)
        if verbose:
            print("ring %3.0f-%3.0f um: mean nbrs=%6.1f  rho_phi=%.3f  |c|=%.3f"
                  % (r_in, r_out, row["mean_neighbors"], row["rho_phi"], row["c_mean_abs"]),
                  flush=True)

    return pd.DataFrame(coef_rows), pd.DataFrame(summary_rows)


def spillover_chi2(coef, rho_col="rho_pair"):
    r"""Test the pure-segmentation-spillover null `c_d(r) = k_d * rho_phi(r)`.

    For each Phi-Space dimension `d`, fits the single scaling `k_d` by weighted
    least squares across rings (weights `1/se^2`, treating `rho_phi` as known —
    it is estimated from ~1e8 pairs, so its own error is negligible beside the
    coefficient standard errors) and accumulates

        chi2 = sum_{r,d} (c_d(r) - k_d * rho(r))^2 / se_d(r)^2 ,  df = D*(R-1).

    A large chi2 means the neighbour coefficients do **not** follow the shape of
    the Phi-Space autocorrelation, i.e. the neighbourhood effect is not explained
    by spillover alone. Also returns per-ring standardized deviations, which show
    *where* the profiles part company.

    Caveat: the rings are fit on the same cells, so the per-ring estimates are
    not independent and the chi2 is an approximation — read it alongside the
    per-ring deviations and the shape comparison, not as an exact p-value.

    Returns a dict with `chi2`, `df`, `p_value`, `k` (per dimension), and
    `deviations` (a DataFrame of observed vs predicted per ring and dimension).
    """
    import pandas as pd
    from scipy import stats as sps

    rows = []
    k_by_dim = {}
    chi2 = 0.0
    for dim, g in coef.groupby("dim", sort=False):
        rho = g[rho_col].to_numpy(dtype=float)
        c = g["c"].to_numpy(dtype=float)
        w = 1.0 / np.clip(g["se"].to_numpy(dtype=float), 1e-12, None) ** 2
        denom = float(np.sum(w * rho ** 2))
        k = float(np.sum(w * rho * c) / denom) if denom > 0 else 0.0
        k_by_dim[dim] = k
        pred = k * rho
        resid = (c - pred) * np.sqrt(w)
        chi2 += float(np.sum(resid ** 2))
        for (_, r), p, z in zip(g.iterrows(), pred, resid):
            rows.append({"dim": dim, "r_inner": r["r_inner"], "r_outer": r["r_outer"],
                         "c": r["c"], "se": r["se"], "rho": r[rho_col],
                         "c_pred_spillover": p, "std_deviation": z})

    n_dims = coef["dim"].nunique()
    n_rings = coef[["r_inner", "r_outer"]].drop_duplicates().shape[0]
    df = n_dims * (n_rings - 1)
    return {
        "chi2": chi2, "df": int(df),
        "p_value": float(sps.chi2.sf(chi2, df)) if df > 0 else np.nan,
        "k": k_by_dim, "deviations": pd.DataFrame(rows),
    }

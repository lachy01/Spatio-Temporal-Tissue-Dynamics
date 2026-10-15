"""Data contract for the concomitant-variable Poisson mixture (spec section 1).

`build_table` assembles the one-row-per-cell parquet from the raw Xenium sample,
the celltype_minor PhiSpace scores and the oxygen field; `validate` runs every
assertion in spec 1.2 and refuses to return a table that violates a MUST.

Two documented deviations from the spec, both recorded in the returned report:

* **Library size `N`** excludes the Tirosh outcome genes but not a "~280-gene
  blocklist" -- no such list exists in this project. The Tirosh set is 0.71% of
  the total library here, so the definitional dependence spec 1.2 item 2 warns
  about is real but small.
* **Density and neighbourhood** keep the radius definitions used by every other
  model in this project (50 um density, 80 um neighbour mean) rather than the
  fixed-k kNN of spec 1.3, so the mixture stays comparable with the existing
  negative-binomial models. Spec 1.3's own first sentence asks for exactly this.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DataReport:
    """Everything spec 1.2 requires to be logged, in one serialisable object."""

    n_cells: int
    n_dropped_zero_N: int
    n_dropped_no_oxygen: int
    tirosh_genes_in_panel: list
    tirosh_library_share: float
    phi_rowsum_mean: float
    phi_rowsum_sd: float
    phi_rowsum_is_constant: bool
    corr_logN_logarea: float
    orthogonalise_applied: bool
    y_count_hist: dict
    zero_fraction: float
    density_radius_um: float
    neighbour_radius_um: float
    deviations: list = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "### Data contract (spec section 1)",
            "",
            f"- cells: **{self.n_cells:,}** (dropped: {self.n_dropped_zero_N} for "
            f"`N == 0`, {self.n_dropped_no_oxygen} outside the oxygen tissue mask)",
            f"- Tirosh genes present in panel: **{len(self.tirosh_genes_in_panel)}**",
            f"- Tirosh share of total library: **{100 * self.tirosh_library_share:.2f}%**",
            f"- `y_count` zero fraction: **{100 * self.zero_fraction:.1f}%**",
            f"- phi rowsum: mean {self.phi_rowsum_mean:.4f}, sd {self.phi_rowsum_sd:.3e} "
            f"-> constant: **{self.phi_rowsum_is_constant}** "
            f"({'spec 3.1 branch (a), Q projection' if self.phi_rowsum_is_constant else 'spec 3.1 branch (b), keep full dimension'})",
            f"- `corr(log N, log area)` = **{self.corr_logN_logarea:.4f}** "
            f"-> orthogonalisation applied: **{self.orthogonalise_applied}**",
            f"- density radius {self.density_radius_um} um; neighbour radius {self.neighbour_radius_um} um",
            "",
            "`y_count` marginal:",
            "",
            "| count | " + " | ".join(str(k) for k in range(21)) + " | 21+ |",
            "|---|" + "---|" * 22,
            "| cells | " + " | ".join(f"{self.y_count_hist.get(k, 0):,}" for k in range(21))
            + f" | {self.y_count_hist.get('21+', 0):,} |",
        ]
        if self.deviations:
            lines += ["", "**Deviations from spec:**", ""]
            lines += [f"{i + 1}. {d}" for i, d in enumerate(self.deviations)]
        return "\n".join(lines)


def _tirosh_genes():
    from ..gene_sets import TIROSH_G2M_GENES, TIROSH_S_GENES

    return list(dict.fromkeys(list(TIROSH_S_GENES) + list(TIROSH_G2M_GENES)))


def split_neighbourhood(xy, phi, pool, contact_um, regional_um, chunk=20000):
    """Mean neighbour state in a contact disc and a regional annulus.

    For every cell (row of `xy`), averages `phi` over the cells flagged in
    `pool` (the typed cells) at distance `d` from it, never itself:

        contact    0 <= d <= contact_um
        regional   contact_um < d <= regional_um

    The two together are exactly the `regional_um` disc that `build_adjacency`
    builds (distance <= radius, self excluded), so
    `(n_c * contact + n_r * regional) / (n_c + n_r)` reproduces the disc mean.

    Gaps are filled, never dropped, so the cell set and the CV folds stay put:
    a cell with no typed contact neighbour (3.5% of analysis cells at 15 um)
    takes its regional mean -- "no information beyond the region" -- and one
    with no typed neighbour within `regional_um` at all takes the pool mean.

    Returns `(contact, n_contact, regional, n_regional)`; the counts are the
    neighbours actually averaged, before any filling.
    """
    from scipy.spatial import cKDTree

    xy = np.asarray(xy, dtype=float)
    phi = np.asarray(phi, dtype=float)
    pidx = np.flatnonzero(np.asarray(pool, dtype=bool))
    tree = cKDTree(xy[pidx])
    n, d = phi.shape
    S = np.zeros((2, n, d))
    C = np.zeros((2, n))
    for s in range(0, n, chunk):
        rows = np.arange(s, min(s + chunk, n))
        sp = cKDTree(xy[rows]).sparse_distance_matrix(tree, regional_um, output_type="ndarray")
        i, j, dist = sp["i"], pidx[sp["j"]], sp["v"]
        keep = j != rows[i]                       # never the cell itself
        i, j, dist = i[keep], j[keep], dist[keep]
        ring = (dist > contact_um).astype(int)    # 0 = contact, 1 = regional
        for r in (0, 1):
            m = ring == r
            C[r, rows] = np.bincount(i[m], minlength=len(rows))
            for k in range(d):
                S[r, rows, k] = np.bincount(i[m], weights=phi[j[m], k], minlength=len(rows))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = S / C[:, :, None]
    pool_mean = phi[pidx].mean(0)
    contact, regional = mean[0], mean[1]
    regional = np.where(C[1][:, None] > 0, regional, np.where(C[0][:, None] > 0, contact, pool_mean))
    contact = np.where(C[0][:, None] > 0, contact, regional)
    return contact, C[0].astype(np.int64), regional, C[1].astype(np.int64)


def build_table(cfg, verbose=True):
    """Assemble the mixture's input table (spec 1.1). Returns a DataFrame.

    Pass the result to `validate` for the spec-1.2 assertions and the report.

    Columns follow spec 1.1 exactly: `cell_id, x, y, y_count, N, area,
    nuc_area, density, oxygen, phi_00..phi_28, nbr_phi_00..nbr_phi_28,
    archetype, ctrl_count`, plus `phi_names` carried in the frame's `attrs` so
    downstream code can name the components without re-reading the CSV.
    """
    from ..io_xenium import read_xenium
    from ..neighbors import average_neighbors, build_adjacency
    from ..covariates import compute_local_density
    from ..oxygen import attach_oxygen_field

    dc = cfg["data"]
    say = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    say("reading Xenium sample ...")
    ad = read_xenium(dc["xenium_dir"], sample_id="BC")

    # --- outcome and library size -------------------------------------------
    counts = ad.layers["counts"] if "counts" in ad.layers else ad.X
    genes = np.asarray(ad.var_names)
    tirosh = [g for g in _tirosh_genes() if g in set(genes)]
    if not tirosh:
        raise ValueError("none of the Tirosh S/G2M genes are present in the panel")
    idx = np.array([int(np.where(genes == g)[0][0]) for g in tirosh])
    # Spec 1.2 item 2: N excludes the outcome genes AND the blocklist (the panel
    # genes correlated with the Tirosh set). The blocklist is a superset of the
    # Tirosh genes and carries 13.47% of the library, so this is not a rounding
    # detail -- an N that included them would be definitionally tied to y_count.
    block = []
    if dc.get("blocklist_csv"):
        block = list(pd.read_csv(dc["blocklist_csv"], header=None)[0].astype(str))
    block = [g for g in dict.fromkeys(block) if g in set(genes)]
    bidx = np.array([int(np.where(genes == g)[0][0]) for g in block], dtype=int)

    def _colsum(mat, cols):
        sub = mat[:, cols]
        return np.asarray(sub.sum(axis=1)).ravel() if hasattr(sub, "sum") else sub.sum(1)

    total = np.asarray(counts.sum(axis=1)).ravel()
    y_count = _colsum(counts, idx)
    excl = np.array(sorted(set(idx.tolist()) | set(bidx.tolist())), dtype=int)
    N = total - _colsum(counts, excl)
    say(f"  y_count from {len(tirosh)} Tirosh genes "
        f"({100 * y_count.sum() / total.sum():.2f}% of library); "
        f"N excludes {len(excl)} genes ({100 * _colsum(counts, excl).sum() / total.sum():.2f}%)")

    # --- per-cell QC columns from cells.csv.gz -------------------------------
    cq = pd.read_csv(
        f"{dc['xenium_dir']}/cells.csv.gz",
        usecols=["cell_id", "cell_area", "nucleus_area", "control_probe_counts",
                 "control_codeword_counts"],
    ).set_index("cell_id")
    cq.index = cq.index.astype(str)
    cq = cq.reindex(np.asarray(ad.obs["cell_id"], dtype=str))

    df = pd.DataFrame({
        "cell_id": np.asarray(ad.obs["cell_id"], dtype=str),
        "x": np.asarray(ad.obsm["spatial"], dtype=float)[:, 0],
        "y": np.asarray(ad.obsm["spatial"], dtype=float)[:, 1],
        "y_count": np.rint(y_count).astype(np.int64),
        "N": np.rint(N).astype(np.int64),
        "area": cq["cell_area"].to_numpy(float),
        "nuc_area": cq["nucleus_area"].to_numpy(float),
        "ctrl_count": np.rint(
            cq["control_probe_counts"].to_numpy(float)
            + cq["control_codeword_counts"].to_numpy(float)).astype(np.int64),
    })

    # --- 29-component celltype_minor PhiSpace --------------------------------
    say(f"reading PhiSpace from {dc['phi_csv']} ...")
    P = pd.read_csv(dc.get("phi_csv", dc.get("phi_minor_csv")), index_col=0)
    P.index = P.index.astype(str)
    if dc.get("phi_columns"):
        P = P[list(dc["phi_columns"])]
    phi_names = list(P.columns)
    say(f"  {len(phi_names)} components: {phi_names if len(phi_names) <= 12 else '...'}")

    keep = df["cell_id"].isin(P.index).to_numpy()
    df, ad = df.loc[keep].reset_index(drop=True), ad[keep].copy()
    Phi = P.reindex(df["cell_id"]).to_numpy(float)
    ok = ~np.isnan(Phi).any(axis=1)
    if (~ok).any():
        say(f"  dropping {int((~ok).sum())} cells with a NaN phi embedding")
        df, ad, Phi = df.loc[ok].reset_index(drop=True), ad[ok].copy(), Phi[ok]

    # Types (and everything else read from the raw scores) use Phi as read.
    Phi_raw = Phi
    if dc.get("phi_depth_residualise"):
        # Sensitivity (ovarian, 2026-10-15): remove each score's within-type
        # linear dependence on log N, keeping the type means, so the state
        # carries no within-type depth signal into pi. Types come from the raw
        # scores. The per-column slopes sum to zero (the rowsum is constant), so
        # the rowsum is preserved. Neighbour means are built from the result.
        min_lib0 = int(dc.get("min_library_size", 0))
        lN = np.log(np.maximum(df["N"].to_numpy(float), 1.0))
        ty = phi_cell_types(Phi, phi_names, dc)
        Phi = Phi.copy()
        for t in np.unique(ty):
            m = ty == t
            fit = m & (df["N"].to_numpy() >= max(min_lib0, 1))
            x = lN[fit] - lN[fit].mean()
            slope = (x @ (Phi[fit] - Phi[fit].mean(0))) / (x @ x)
            Phi[m] -= np.outer(lN[m] - lN[fit].mean(), slope)
        say(f"  scores residualised on log N within type ({len(np.unique(ty))} types)")

    # --- neighbourhood mean, density, oxygen ---------------------------------
    say(f"neighbour mean at radius {dc['neighbour_radius_um']} um ...")
    ad.obsm["phi"] = Phi
    ad.uns["phi_names"] = phi_names
    build_adjacency(ad, radius=float(dc["neighbour_radius_um"]))
    # Shallow cells' scores are noise, so they do not contribute to anyone's
    # neighbour average: zero their columns of the adjacency before it is
    # row-normalised. They still count towards density below.
    min_lib = int(dc.get("min_library_size", 0))
    typed = df["N"].to_numpy() >= min_lib
    if min_lib > 0:
        import scipy.sparse as _sp
        A = ad.obsp["spatial_connectivity"]
        ad.obsp["spatial_connectivity"] = (A @ _sp.diags(typed.astype(float))).tocsr()
        say(f"  neighbour average over cells with N >= {min_lib} "
            f"({100 * typed.mean():.1f}% of cells)")
    average_neighbors(ad)
    nbr = np.asarray(ad.obsm["phi_bar"], dtype=float)
    # Contact (0-15 um) and regional (15-80 um) means: the neighbour block of
    # the weight model when state.neighbourhood is "split" (see basis.py).
    # Computed over the same typed pool and the same disc as `nbr`.
    split = split_neighbourhood(np.asarray(ad.obsm["spatial"], dtype=float), Phi, typed,
                                float(dc.get("contact_radius_um", 15.0)),
                                float(dc["neighbour_radius_um"]))
    n_isolated = int((np.asarray(ad.obsp["spatial_connectivity"].sum(1)).ravel() == 0).sum())

    say(f"local density at radius {dc['density_radius_um']} um ...")
    compute_local_density(ad, radius=float(dc["density_radius_um"]))
    df["density"] = ad.obs["local_density"].to_numpy(float)

    # The oxygen source is blood endothelium only (ACKR1 + RGS5 + CXCL12,
    # excluding lymphatic LYVE1), which exists only in the celltype_minor table.
    # That table is stale, so the field inherits its staleness -- but it is the
    # same field validated in notebook sections 16-20, so results stay comparable.
    oc = cfg["oxygen"]
    if oc.get("source", "minor") == "major":
        ecol = oc.get("endothelial_column", "Endothelial")
        say(f"oxygen field ({ecol} score from the major table, typed cells only) ...")
        E = pd.DataFrame({ecol: Phi_raw[:, phi_names.index(ecol)]},
                         index=df["cell_id"].to_numpy())
        E.loc[~typed, ecol] = np.nan                   # untyped cells are not sources
        blood_cols = [ecol]
    else:
        say("oxygen field (blood-endothelium source from the celltype_minor table) ...")
        E = pd.read_csv(dc["phi_minor_csv"], index_col=0)
        blood_cols = [c for c in oc["blood_endothelium"] if c in E.columns]
    E.index = E.index.astype(str)
    # The minor table is compositional (rowsums constant), so SUMMING the three
    # blood-endothelial columns is not a valid "total endothelium": the rowsum
    # constraint makes them anti-correlated, and the sum ends up correlating
    # -0.95 with ACKR1 while correlating +1.00 with RGS5. Z-score each column
    # against its own distribution and take the max -- "how unusually high is
    # this cell for any blood-endothelial subtype" -- which is well defined on
    # compositional scores. The cutoff is then a quantile of that score, so it
    # carries over unchanged if the normalisation changes again.
    V = E.to_numpy(float)
    Z = (V - np.nanmean(V, axis=0)) / np.nanstd(V, axis=0)
    zi = [E.columns.get_loc(c) for c in blood_cols]
    with np.errstate(invalid="ignore"):
        import warnings as _w
        with _w.catch_warnings():           # rows with no scored endothelial type
            _w.simplefilter("ignore", RuntimeWarning)
            endo = pd.Series(np.nanmax(Z[:, zi], axis=1), index=E.index).dropna()
    q = float(np.quantile(endo.to_numpy(float), 1.0 - float(oc["source_quantile"])))
    endo = endo - q                       # top `source_quantile` become positive
    say(f"  source: max z-score over {blood_cols}; cutoff at the "
        f"{100 * (1 - float(oc['source_quantile'])):.1f}th percentile "
        f"(z = {q:.3f}), retaining {100 * (endo > 0).mean():.2f}% of cells")
    src_xy = (pd.read_csv(f"{dc['xenium_dir']}/cells.csv.gz",
                          usecols=["cell_id", "x_centroid", "y_centroid"])
              .astype({"cell_id": str}).set_index("cell_id").reindex(endo.index).dropna())
    endo = endo.reindex(src_xy.index)
    attach_oxygen_field(
        ad, endo.reindex(df["cell_id"]).fillna(0.0).to_numpy(float),
        source_xy=src_xy[["x_centroid", "y_centroid"]].to_numpy(float),
        source_phi_endo=endo.to_numpy(float),
        h=float(oc["h_um"]), L0=float(oc["L0_um"]),
        beta=float(oc["beta"]), c=0.0)      # cutoff already applied as a quantile shift
    df["oxygen"] = ad.obs["oxygen_u"].to_numpy(float)
    df["oxygen_valid"] = ad.obs["oxygen_valid"].to_numpy(bool)

    # --- archetype (reporting only; spec 1.1 forbids it in the likelihood) ----
    # By the section's typing rule (`phi_cell_types`): on breast a raw argmax
    # is dominated by column scale, so each column is z-scored first -- the
    # same correction notebook section 23 made.
    df["archetype"] = pd.Categorical(phi_cell_types(Phi_raw, phi_names, dc))

    for j, nm in enumerate(phi_names):
        df[f"phi_{j:02d}"] = Phi[:, j]
    for j, nm in enumerate(phi_names):
        df[f"nbr_phi_{j:02d}"] = nbr[:, j]
    add_split_columns(df, *split)
    df.attrs["phi_names"] = phi_names
    if min_lib > 0:
        say(f"dropping {int((~typed).sum()):,} analysis cells with N < {min_lib} "
            f"({100 * (~typed).mean():.1f}%); {n_isolated:,} cells have no typed neighbour")
        df = df.loc[typed].reset_index(drop=True)
        df.attrs["phi_names"] = phi_names
    df.attrs["tirosh_genes"] = tirosh
    df.attrs["tirosh_library_share"] = float(y_count.sum() / total.sum())
    say(f"assembled {len(df):,} cells x {df.shape[1]} columns")
    return df


def add_split_columns(df, contact, n_contact, regional, n_regional):
    """Write `split_neighbourhood`'s output into the table, in place."""
    for j in range(contact.shape[1]):
        df[f"nbrc_phi_{j:02d}"] = contact[:, j]
    for j in range(regional.shape[1]):
        df[f"nbrr_phi_{j:02d}"] = regional[:, j]
    df["n_contact"] = n_contact
    df["n_regional"] = n_regional
    return df


def validate(df, cfg, verbose=True):
    """Run every spec-1.2 assertion. Returns `(clean_df, DataReport)`.

    MUST-level failures raise; SHOULD-level findings are recorded as deviations.
    """
    say = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    dc = cfg["data"]
    phi_names = df.attrs.get("phi_names", [])
    devs = []

    # 1. outcome
    y = df["y_count"].to_numpy()
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("y_count must be an integer dtype")
    if np.isnan(y.astype(float)).any() or (y < 0).any():
        raise ValueError("y_count must be non-negative with no NaN")

    # 2. N excludes the outcome genes -- verified by construction in build_table
    share = float(df.attrs.get("tirosh_library_share", np.nan))
    say(f"  N excludes the outcome genes and the blocklist "
        f"({dc.get('blocklist_csv')})")

    # 3. N > 0
    n_drop = int((df["N"] <= 0).sum())
    if n_drop:
        say(f"  dropping {n_drop} cells with N <= 0")
    out = df.loc[df["N"] > 0].reset_index(drop=True)

    # 4. area, density positive
    for col in ("area", "density"):
        bad = int((out[col] <= 0).sum())
        if bad:
            raise ValueError(f"{col} must be strictly positive; {bad} rows violate this")

    # 5. no NaN in the state blocks
    pcols = [c for c in out.columns if c.startswith("phi_")]
    ncols = [c for c in out.columns if c.startswith("nbr_phi_")]
    split_cols = [c for c in out.columns if c.startswith(("nbrc_phi_", "nbrr_phi_"))]
    for cols, lab in ((pcols, "phi"), (ncols, "nbr_phi"), (split_cols, "contact/regional")):
        n_nan = int(out[cols].isna().to_numpy().sum())
        if n_nan:
            raise ValueError(f"{lab} block contains {n_nan} NaN values")

    # 5b. no NaN in ANY column the model consumes.
    #
    # Spec 1.2 item 5 names only the phi blocks, and implementing it literally
    # left a hole: `oxygen` is NaN for cells outside the PDE's tissue mask
    # (0.14% here), those NaNs reach the linear predictor, and a single one makes
    # the whole log-likelihood NaN. Every fit on real data then "diverged" -- a
    # failure that looked like a modelling pathology and was a missing check.
    # The mask is a property of the field, not of the cell, so these rows are
    # dropped and counted rather than imputed.
    n_ox = int(out["oxygen"].isna().sum())
    if n_ox:
        say(f"  dropping {n_ox} cells with no oxygen-field value "
            f"(outside the tissue mask, {100 * n_ox / len(out):.2f}%)")
        out = out.loc[out["oxygen"].notna()].reset_index(drop=True)
    # Only the columns the likelihood actually reads are allowed to be NaN-free
    # by force. `nuc_area` and `ctrl_count` are reporting/robustness columns per
    # spec 1.1 -- a cell with no detected nucleus is real data, not a defect, and
    # dropping thousands of cells to satisfy a column no model term uses would be
    # the wrong trade.
    for col in ("oxygen", "density", "area", "N", "y_count"):
        if col in out.columns:
            bad = int(out[col].isna().sum())
            if bad:
                raise ValueError(f"model covariate {col!r} still contains {bad} NaN "
                                 "values after cleaning")
    for col in ("nuc_area", "ctrl_count"):
        if col in out.columns and out[col].isna().any():
            say(f"  note: {col!r} has {int(out[col].isna().sum())} NaN "
                "(reporting only, not used by the likelihood)")
    n_dropped_oxygen = n_ox

    # 6. phi rowsum -- determines the spec 3.1 basis branch
    s = out[pcols].to_numpy(float).sum(axis=1)
    const = bool(s.std() < 1e-6)
    say(f"  phi rowsum: mean {s.mean():.4f} sd {s.std():.3e} -> constant={const}")

    # 7. log N vs log area collinearity
    r = float(np.corrcoef(np.log(out["N"].to_numpy(float)),
                          np.log(out["area"].to_numpy(float)))[0, 1])
    ortho = bool(abs(r) > float(dc["corr_logN_logarea_orthogonalise_above"]))
    say(f"  corr(log N, log area) = {r:.4f} -> orthogonalise={ortho}")

    # 8. marginal of the outcome
    yv = out["y_count"].to_numpy()
    hist = {k: int((yv == k).sum()) for k in range(21)}
    hist["21+"] = int((yv >= 21).sum())

    devs.append(
        "**Density and neighbourhood (spec 1.3 / 1.1).** Kept at this project's "
        f"radius definitions ({dc['density_radius_um']} um density, "
        f"{dc['neighbour_radius_um']} um neighbour mean) rather than fixed-k "
        "kNN, so the mixture is comparable with the existing NB models. Spec "
        "1.3's own first sentence asks for the existing definition; its fixed-k "
        "preference is tested as a Stage D sensitivity. User-approved.")
    if cfg["state"]["drop_cycling_from_own"]:
        devs.append(
            "**Cycling archetypes (not in spec).** The four explicitly cycling "
            f"celltype_minor components {cfg['state']['cycling_archetypes']} are "
            "dropped from the cell's OWN state. Their reference centroids are "
            "defined by cell-cycle genes and they correlate 0.47-0.57 with "
            "log1p(y_count), so leaving them in the mixing weight would predict "
            "P(cycling) from a cycling score built on the outcome's own "
            "transcripts. They are kept in the neighbourhood state, where "
            "phi_bar excludes the cell itself. User-approved.")

    rep = DataReport(
        n_cells=len(out), n_dropped_zero_N=n_drop,
        n_dropped_no_oxygen=n_dropped_oxygen,
        tirosh_genes_in_panel=list(df.attrs.get("tirosh_genes", [])),
        tirosh_library_share=share,
        phi_rowsum_mean=float(s.mean()), phi_rowsum_sd=float(s.std()),
        phi_rowsum_is_constant=const, corr_logN_logarea=r,
        orthogonalise_applied=ortho, y_count_hist=hist,
        zero_fraction=float((yv == 0).mean()),
        density_radius_um=float(dc["density_radius_um"]),
        neighbour_radius_um=float(dc["neighbour_radius_um"]),
        deviations=devs,
    )
    out.attrs.update(df.attrs)
    out.attrs["phi_names"] = phi_names
    return out, rep


def _deep_merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_config(path=None):
    """Read `config.yaml` (the only place hyperparameters are defined).

    The environment variable MIX_TECHNICAL, if set, overrides
    `model.technical` -- so a Spartan array can fit the plain baseline
    (`MIX_TECHNICAL=plain`) beside the production model without editing the
    file. Every result records the technical covariates it was fitted with.

    MIX_DATASET, if set (e.g. `ovarian`), layers `config_<dataset>.yaml` from
    the same folder over the production file: that overlay holds only what
    differs for the other section (paths, PhiSpace columns, typing rule), so
    every model setting stays shared. Unset is the breast section.
    """
    import os

    import yaml

    here = os.path.dirname(os.path.abspath(__file__))
    if path is None:
        path = os.path.join(here, "config.yaml")
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    ds = os.environ.get("MIX_DATASET")
    if ds:
        with open(os.path.join(here, f"config_{ds}.yaml")) as fh:
            cfg = _deep_merge(cfg, yaml.safe_load(fh))
        cfg["dataset"] = ds
    if os.environ.get("MIX_TECHNICAL"):
        cfg["model"]["technical"] = os.environ["MIX_TECHNICAL"]
    return cfg


def phi_cell_types(Phi, phi_names, data_cfg):
    """Cell type per row of the PhiSpace scores `Phi`, by the section's rule.

    `data.type_rule`: "standardised" (default; the breast convention, argmax
    of column-standardised scores, since its columns differ ~18-fold in
    scale) or "raw" (argmax of the scores as they are; on the ovarian section
    standardising inflated Granulosa 15-fold with no marker support).
    `data.type_fold` maps type names to a merged label (ovarian: Granulosa
    and Fallopian Tube Epithelium to Other). The fold relabels only: the
    score columns, and so the state block, are unchanged.
    """
    Phi = np.asarray(Phi, float)
    rule = data_cfg.get("type_rule", "standardised")
    if rule == "standardised":
        Phi = (Phi - Phi.mean(0)) / np.where(Phi.std(0) > 0, Phi.std(0), 1.0)
    elif rule != "raw":
        raise ValueError(f"unknown data.type_rule {rule!r}")
    lab = np.asarray(phi_names, dtype=object)[Phi.argmax(1)]
    fold = data_cfg.get("type_fold") or {}
    return np.array([fold.get(t, t) for t in lab], dtype=object)

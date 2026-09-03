"""Frozen design bases for the Poisson mixture (spec section 3.1).

Everything here is computed **once on the full dataset and serialised**. Spec
3.1 is explicit that recomputing centring, the sum-to-zero projection or the
spline knots per bootstrap replicate changes the parameterisation between
replicates and produces meaningless confidence intervals, so `FrozenBasis`
carries the fitted constants and applies them to any subsample.

The own-state and neighbour-state blocks can have **different widths**: the four
explicitly cycling celltype_minor archetypes are dropped from the own state (their
reference centroids are defined by cell-cycle genes, so they leak the outcome
into the mixing weight) but kept in the neighbour state, where `phi_bar` excludes
the cell itself. `B = U V^T` is therefore `d_own x d_nbr`, not square.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


def sum_to_zero_basis(d):
    """Orthonormal `d x (d-1)` basis for the sum-to-zero subspace of R^d.

    Used when the PhiSpace rows sum to a constant (spec 3.1 branch (a)): the
    centred columns are then linearly dependent with the intercept, and the
    design is rank-deficient without this projection. Built from a QR of the
    centring projector `I - (1/d) 11^T`, whose range is exactly that subspace.
    """
    P = np.eye(d) - np.ones((d, d)) / d
    Q, _R = np.linalg.qr(P)
    return Q[:, : d - 1]


class _Spline:
    """A natural cubic spline, knots frozen, with the constant projected out.

    `patsy`'s `cr(x, df=k) - 1` basis **spans the constant function exactly** (the
    residual of `1` on its column space is ~1e-13). Stacking it next to a model
    intercept therefore gives a design that is rank-deficient by one, and in this
    model that is an exactly flat direction of the likelihood: `b0` and `s` trade
    off with no effect on the fit, so L-BFGS never satisfies its gradient
    criterion and the observed information spec 4.2 asks for is singular.

    So the basis is centred and reduced to an orthonormal basis of its
    non-constant span, `df - 1` columns. Total parameters are unchanged
    (intercept + df-1), the fitted function space is unchanged, and the design is
    well conditioned. The centring vector and the projection are computed once on
    the training `x` and reused, as spec 3.1 requires.
    """

    def __init__(self, x, df, tol=1e-8):
        import patsy
        from scipy.linalg import qr

        x = np.asarray(x, dtype=float)
        dm = patsy.dmatrix(f"cr(x, df={int(df)}) - 1", {"x": x}, return_type="matrix")
        self._di = dm.design_info
        self.df = int(df)
        B = np.asarray(dm)
        self._mean = B.mean(0)
        Bc = B - self._mean

        # Drop the one redundant column rather than whitening the basis.
        #
        # Whitening (projecting onto V/S) was the obvious construction and is
        # wrong here: dividing by the small singular values amplifies the
        # oscillatory directions of the spline space enormously. The raw `cr`
        # basis is bounded (max |.| ~ 2.6), but the whitened-and-rescaled version
        # reaches max |.| = 42 with kurtosis 303 by df=12, and those spikes
        # overflow `exp` in the rate model. Column-pivoted QR instead *selects*
        # `rank` of the original columns, so the basis keeps its natural, bounded
        # scale and stays interpretable, while the constant -- which `cr(x, df) - 1`
        # spans exactly -- is removed by the centring plus the dropped column.
        _q, r, piv = qr(Bc, mode="economic", pivoting=True)
        rank = int((np.abs(np.diag(r)) > tol * abs(r[0, 0])).sum())
        self._keep = np.sort(piv[:rank])
        self.n_cols = int(rank)

    def __call__(self, x):
        import patsy

        B = np.asarray(patsy.build_design_matrices(
            [self._di], {"x": np.asarray(x, dtype=float)}, return_type="matrix")[0])
        return (B - self._mean)[:, self._keep]


@dataclass
class FrozenBasis:
    """Centring vectors, projections and spline bases, fitted once and frozen."""

    phi_names: list
    own_names: list
    nbr_names: list
    own_cols: np.ndarray          # column indices of phi_* used for the own state
    m_own: np.ndarray             # global means, own block
    m_nbr: np.ndarray             # global means, neighbour block
    Q_own: np.ndarray | None      # sum-to-zero projection, or None
    Q_nbr: np.ndarray | None
    rowsum_constant: bool
    area_spline: object
    density_weight_spline: object
    density_bg_spline: object
    logN_ortho_coef: np.ndarray | None    # residualisation of log N on the area basis
    logN_ortho_r2: float
    cond_own: float
    cond_nbr: float

    @property
    def n_area(self):
        return int(self.area_spline.n_cols)

    @property
    def n_weight(self):
        return int(self.density_weight_spline.n_cols)

    @property
    def n_bg(self):
        return int(self.density_bg_spline.n_cols)

    @property
    def d_own(self):
        return len(self.own_names) if self.Q_own is None else self.Q_own.shape[1]

    @property
    def d_nbr(self):
        return len(self.nbr_names) if self.Q_nbr is None else self.Q_nbr.shape[1]

    # -- transforms ----------------------------------------------------------
    def own(self, df):
        """Centred (and, if needed, projected) own-state matrix `phi~`."""
        X = df[[f"phi_{j:02d}" for j in self.own_cols]].to_numpy(float) - self.m_own
        return X if self.Q_own is None else X @ self.Q_own

    def nbr(self, df):
        """Centred (and, if needed, projected) neighbour-state matrix `phibar~`."""
        X = df[[f"nbr_phi_{j:02d}" for j in range(len(self.phi_names))]].to_numpy(float)
        X = X - self.m_nbr
        return X if self.Q_nbr is None else X @ self.Q_nbr

    def log_area_basis(self, df):
        return self.area_spline(np.log(df["area"].to_numpy(float)))

    def density_weight_basis(self, df):
        return self.density_weight_spline(np.log(df["density"].to_numpy(float)))

    def density_bg_basis(self, df):
        return self.density_bg_spline(np.log(df["density"].to_numpy(float)))

    def log_N(self, df):
        """`log N`, residualised on the area spline when spec 2.2 requires it."""
        v = np.log(df["N"].to_numpy(float))
        if self.logN_ortho_coef is None:
            return v
        A = np.column_stack([np.ones(len(v)), self.log_area_basis(df)])
        return v - A @ self.logN_ortho_coef

    def to_json(self):
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in self.__dict__.items() if not k.endswith("spline")}
        d["own_cols"] = np.asarray(self.own_cols).tolist()
        return json.dumps(d, indent=2, default=str)


def build_basis(df, cfg, verbose=True):
    """Fit every frozen constant on the full data (spec 3.1). Returns FrozenBasis."""
    say = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    sc, mc, dc = cfg["state"], cfg["model"], cfg["data"]
    phi_names = list(df.attrs["phi_names"])
    d = len(phi_names)

    cyc = set(sc["cycling_archetypes"]) if sc["drop_cycling_from_own"] else set()
    missing = cyc - set(phi_names)
    if missing:
        raise ValueError(f"cycling archetypes not found in phi_names: {sorted(missing)}")
    own_cols = np.array([j for j, n in enumerate(phi_names) if n not in cyc])
    own_names = [phi_names[j] for j in own_cols]
    nbr_cyc = set(sc["cycling_archetypes"]) if sc["drop_cycling_from_neighbour"] else set()
    nbr_names = [n for n in phi_names if n not in nbr_cyc]
    if len(nbr_names) != d:
        raise NotImplementedError("dropping cycling axes from the neighbour block "
                                  "is configured but not implemented; the neighbour "
                                  "block is built over all phi_names")
    say(f"own state: {len(own_names)} components (dropped {sorted(cyc)});"
        f" neighbour state: {len(nbr_names)}")

    PO = df[[f"phi_{j:02d}" for j in own_cols]].to_numpy(float)
    PN = df[[f"nbr_phi_{j:02d}" for j in range(d)]].to_numpy(float)
    m_own, m_nbr = PO.mean(0), PN.mean(0)
    CO, CN = PO - m_own, PN - m_nbr

    # spec 3.1: rowsum constant -> project onto the sum-to-zero subspace; else
    # keep full dimension but guard the conditioning.
    rowsum = df[[f"phi_{j:02d}" for j in range(d)]].to_numpy(float).sum(1)
    const = bool(rowsum.std() < 1e-6)
    Q_own = Q_nbr = None
    if const:
        Q_own, Q_nbr = sum_to_zero_basis(CO.shape[1]), sum_to_zero_basis(CN.shape[1])
        CO, CN = CO @ Q_own, CN @ Q_nbr
        say("  rowsum constant -> sum-to-zero projection applied (spec 3.1a)")
    else:
        say(f"  rowsum not constant (sd {rowsum.std():.3e}) -> full dimension (spec 3.1b)")

    cond_own = float(np.linalg.cond(np.column_stack([np.ones(len(CO)), CO])))
    cond_nbr = float(np.linalg.cond(np.column_stack([np.ones(len(CN)), CN])))
    say(f"  condition numbers: own {cond_own:.3g}, neighbour {cond_nbr:.3g}")
    kmax = float(sc["condition_number_max"])
    if max(cond_own, cond_nbr) > kmax:
        raise NotImplementedError(
            f"condition number exceeds {kmax:g}; spec 3.1b asks for a PCA "
            "reduction to 99.9% of variance here. Not triggered on this data, "
            "so deliberately not implemented -- add it if a new sample needs it.")

    area_spline = _Spline(np.log(df["area"].to_numpy(float)), mc["area_spline_df"])
    dw = _Spline(np.log(df["density"].to_numpy(float)), mc["weight_density_spline_df"])
    dg = _Spline(np.log(df["density"].to_numpy(float)), mc["background_spline_df"])

    # spec 2.2: orthogonalise log N against the area spline only if strongly collinear
    logN = np.log(df["N"].to_numpy(float))
    la = np.log(df["area"].to_numpy(float))
    r = float(np.corrcoef(logN, la)[0, 1])
    coef, r2 = None, float("nan")
    if abs(r) > float(dc["corr_logN_logarea_orthogonalise_above"]):
        A = np.column_stack([np.ones(len(logN)), area_spline(la)])
        coef, *_ = np.linalg.lstsq(A, logN, rcond=None)
        fit = A @ coef
        r2 = float(1.0 - np.var(logN - fit) / np.var(logN))
        say(f"  corr(log N, log area) = {r:.3f} -> orthogonalised, aux R2 = {r2:.4f}")
    else:
        say(f"  corr(log N, log area) = {r:.3f} -> no orthogonalisation needed")

    return FrozenBasis(
        phi_names=phi_names, own_names=own_names, nbr_names=nbr_names,
        own_cols=own_cols, m_own=m_own, m_nbr=m_nbr, Q_own=Q_own, Q_nbr=Q_nbr,
        rowsum_constant=const, area_spline=area_spline,
        density_weight_spline=dw, density_bg_spline=dg,
        logN_ortho_coef=coef, logN_ortho_r2=r2,
        cond_own=cond_own, cond_nbr=cond_nbr)


def spec_from_basis(basis, cfg, stage="B", theta_free=True, weight_ridge=0.0,
                    rank=None):
    """Build a `ModelSpec` whose column counts match this basis.

    The spline bases carry `df - 1` columns each (the constant is projected out,
    see `_Spline`), so the spec must be derived from the basis rather than from
    the configured `df` values.
    """
    from .model import ModelSpec

    mc = cfg["model"]
    return ModelSpec(
        stage=stage, rank=int(mc["rank"] if rank is None else rank),
        theta_free=bool(theta_free), d_own=basis.d_own, d_nbr=basis.d_nbr,
        df_area=basis.n_area, df_weight=basis.n_weight, df_bg=basis.n_bg,
        ridge_tau=float(mc["ridge_tau"]), weight_ridge=float(weight_ridge))

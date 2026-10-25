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

    def __init__(self, x, df, tol=1e-8, knot_quantiles=None, scale=False, drop=None):
        """`knot_quantiles` places interior knots explicitly, e.g. [25, 50, 75, 90].

        Worth doing deliberately rather than relying on `df`. `cr` is a NATURAL
        cubic spline, so the second derivative is forced to zero beyond the
        boundary knots and the fit is linear in both tails. A response that
        turns downward only at high density -- contact inhibition -- is exactly
        the shape that constraint forbids unless an interior knot sits up there.
        Placing them by hand also makes the basis inspectable, which reading
        patsy's internals is not.
        """
        import patsy
        from scipy.linalg import qr

        x = np.asarray(x, dtype=float)
        if knot_quantiles:
            kn = np.unique(np.percentile(x, np.asarray(knot_quantiles, dtype=float)))
            self.knots = kn
            dm = patsy.dmatrix("cr(x, knots=kn) - 1", {"x": x, "kn": kn},
                               return_type="matrix")
        else:
            self.knots = None
            dm = patsy.dmatrix(f"cr(x, df={int(df)}) - 1", {"x": x},
                               return_type="matrix")
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
        # `drop` pins the dropped column. The cr columns sum to 1, so after centring
        # the first and last are tied exactly in the pivoting's final step, and
        # which one goes is decided by rounding -- it differs between LAPACK builds
        # (the 60 um refit dropped column 5 on Spartan, 0 on a Mac; 2026-10-25).
        # A fit is only readable on another machine with the same column dropped.
        if drop is not None:
            if rank != B.shape[1] - 1:
                raise ValueError(f"drop= assumes exactly one redundant column; rank {rank} of {B.shape[1]}")
            self._keep = np.array([j for j in range(B.shape[1]) if j != int(drop)])
        self.n_cols = int(rank)
        # `scale`: divide each kept column by its training SD, so a ridge on the
        # coefficients is a ridge on effects per SD of each basis column -- the
        # same scale the whitened state blocks are on (see build_basis).
        self._sd = Bc[:, self._keep].std(0) if scale else np.ones(rank)

    def __call__(self, x):
        import patsy

        B = np.asarray(patsy.build_design_matrices(
            [self._di], {"x": np.asarray(x, dtype=float)}, return_type="matrix")[0])
        return (B - self._mean)[:, self._keep] / self._sd


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
    mean_log_density: float = 0.0
    # Whitening (state.whiten): W_* = Sigma^-1/2 of the projected block, already
    # folded into Q_own / Q_nbr, which are then Q0 W. Kept for reporting.
    W_own: np.ndarray | None = None
    W_nbr: np.ndarray | None = None
    # Neighbour block layout (state.neighbourhood): the table columns it reads,
    # in order, and [(block name, width in model coordinates)]. None = the
    # single 80 um disc (nbr_phi_*), as before the split.
    nbr_cols: list | None = None
    nbr_blocks: list | None = None
    # Technical covariates (model.technical). "area": s(log area) and raw N.
    # "mistic": no area spline (area_spline is None), library size read from
    # `logN_column` (the MisTIC-weighted N), and `m_column` -- the expected
    # count of misassigned outcome transcripts -- for the additive term.
    technical: str = "area"
    logN_column: str = "N"
    m_column: str | None = None

    @property
    def n_area(self):
        return 0 if self.area_spline is None else int(self.area_spline.n_cols)

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
        """Centred (and, if needed, projected) neighbour-state matrix `phibar~`.

        With the split neighbourhood this is [contact | regional], each block
        centred, projected and whitened on its own (Q_nbr is block-diagonal).
        """
        cols = self.nbr_cols or [f"nbr_phi_{j:02d}" for j in range(len(self.phi_names))]
        X = df[cols].to_numpy(float) - self.m_nbr
        return X if self.Q_nbr is None else X @ self.Q_nbr

    def to_own(self, raw):
        """Own-state coordinates of raw (n, d) compositions, e.g. a neighbourhood mean."""
        X = np.asarray(raw, dtype=float)[:, self.own_cols] - self.m_own
        return X if self.Q_own is None else X @ self.Q_own

    def disc_in_own(self, df):
        """The 80 um disc's mean state (`nbr_phi_*`) in own-state coordinates.

        The phase portrait's neighbourhood composition: its state variable
        n * wbar lives on that disc, whatever the weight model's neighbour block.
        """
        d = len(self.phi_names)
        return self.to_own(df[[f"nbr_phi_{j:02d}" for j in range(d)]].to_numpy(float))

    def nbr_from_own(self):
        """`(offset, M)` such that the neighbour design is `offset + z @ M`
        for a cell whose every neighbourhood block has the mean state with
        own-state coordinates `z`.

        Every map involved is affine -- raw state = m_own + z pinv(Q_own), and
        block b's design = (raw - m_b) Q_b -- so this is exact. The phase
        portrait uses it with the mean-field closure that contact and regional
        neighbourhoods share one composition, and folds it into c and B so the
        whole weight model reads own-state coordinates.
        """
        d = len(self.phi_names)
        if len(self.own_cols) != d:
            raise NotImplementedError("own-state coordinates drop components, so a "
                                      "neighbourhood composition cannot be mapped "
                                      "back to raw scores")
        P = np.eye(d) if self.Q_own is None else np.linalg.pinv(self.Q_own)
        Qn = np.eye(len(self.m_nbr)) if self.Q_nbr is None else self.Q_nbr
        widths = [w for _, w in self.nbr_blocks] if self.nbr_blocks else [Qn.shape[1]]
        offs, Ms, c0 = [], [], 0
        for b, w in enumerate(widths):
            rows = slice(b * d, (b + 1) * d)
            Qb = Qn[rows, c0:c0 + w]                               # this block's columns
            offs.append((self.m_own - self.m_nbr[rows]) @ Qb)
            Ms.append(P @ Qb)
            c0 += w
        return np.concatenate(offs), np.hstack(Ms)

    def log_area_basis(self, df):
        """s(log area) basis; zero columns when the technical block is MisTIC's."""
        if self.area_spline is None:
            return np.zeros((len(df), 0))
        return self.area_spline(np.log(df["area"].to_numpy(float)))

    def mistic_m(self, df):
        """`m_i`, the expected misassigned outcome count (technical: mistic)."""
        if self.m_column is None:
            raise ValueError("the basis was built with technical: area; it has no m")
        return df[self.m_column].to_numpy(float)

    def density_weight_basis(self, df):
        return self.density_weight_spline(np.log(df["density"].to_numpy(float)))

    def density_bg_basis(self, df):
        return self.density_bg_spline(np.log(df["density"].to_numpy(float)))

    def log_N(self, df):
        """`log N` (the configured library size), residualised on the area
        spline when spec 2.2 requires it (technical: area only)."""
        v = np.log(df[self.logN_column].to_numpy(float))
        if self.logN_ortho_coef is None:
            return v
        A = np.column_stack([np.ones(len(v)), self.log_area_basis(df)])
        return v - A @ self.logN_ortho_coef

    def to_json(self):
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in self.__dict__.items() if not k.endswith("spline")}
        d["own_cols"] = np.asarray(self.own_cols).tolist()
        return json.dumps(d, indent=2, default=str)


def _block_diag(mats):
    from scipy.linalg import block_diag
    return block_diag(*mats)


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
    base_names = [n for n in phi_names if n not in nbr_cyc]
    if len(base_names) != d:
        raise NotImplementedError("dropping cycling axes from the neighbour block "
                                  "is configured but not implemented; the neighbour "
                                  "block is built over all phi_names")
    # The neighbour block's layout. "disc": the 80 um mean (nbr_phi_*), as
    # before. "split": contact (0-15 um, nbrc_phi_*) and regional (15-80 um,
    # nbrr_phi_*) means side by side, each with its own c and rows of V.
    structure = sc.get("neighbourhood", "disc")
    layout = {"disc": [("disc", "nbr_phi_")],
              "split": [("contact", "nbrc_phi_"), ("regional", "nbrr_phi_")]}.get(structure)
    if layout is None:
        raise ValueError(f"state.neighbourhood must be 'disc' or 'split', not {structure!r}")
    nbr_cols = [f"{pre}{j:02d}" for _, pre in layout for j in range(d)]
    lacking = [c for c in nbr_cols if c not in df.columns]
    if lacking:
        raise KeyError(f"the table lacks {lacking[:3]}...: run run/add_neighbourhood_split.py "
                       "(or rebuild the table) before using state.neighbourhood: split")
    nbr_names = (base_names if structure == "disc"
                 else [f"{b}:{n}" for b, _ in layout for n in base_names])
    say(f"own state: {len(own_names)} components (dropped {sorted(cyc)});"
        f" neighbour state: {structure}, {len(nbr_names)} components")

    PO = df[[f"phi_{j:02d}" for j in own_cols]].to_numpy(float)
    m_own = PO.mean(0)
    CO = PO - m_own
    PN = [df[[f"{pre}{j:02d}" for j in range(d)]].to_numpy(float) for _, pre in layout]
    m_nbr = np.concatenate([P.mean(0) for P in PN])
    CN = [P - P.mean(0) for P in PN]

    # spec 3.1: rowsum constant -> project onto the sum-to-zero subspace; else
    # keep full dimension but guard the conditioning.
    rowsum = df[[f"phi_{j:02d}" for j in range(d)]].to_numpy(float).sum(1)
    const = bool(rowsum.std() < 1e-6)
    Q_own, Qn = None, [None] * len(PN)
    if const:
        Q_own = sum_to_zero_basis(CO.shape[1])
        Qn = [sum_to_zero_basis(d) for _ in PN]
        CO = CO @ Q_own
        CN = [C @ Q for C, Q in zip(CN, Qn)]
        say("  rowsum constant -> sum-to-zero projection applied (spec 3.1a)")
    else:
        say(f"  rowsum not constant (sd {rowsum.std():.3e}) -> full dimension (spec 3.1b)")

    # Whitening. The weight model's ridge is only commensurate across blocks if
    # their covariates share a scale, and they do not: on the log1p table the
    # projected own coordinates have SD ~0.29, the neighbour means ~0.11, and
    # their products -- B's covariates -- ~0.03, so one lambda shrank each entry
    # of B ~100x harder per coefficient than `a` (10% against 0.1% at
    # lambda = 10). Whitening each block (Sigma^-1/2 of its projected
    # coordinates) puts every own and neighbour direction, and so every entry of
    # B, on a per-SD scale. It is a reparameterisation: unpenalised fits are
    # unchanged, and effects map back to cell types through Q = Q0 W exactly as
    # before. It also matches the simulations, whose covariates are N(0, 1).
    # Each neighbour block is whitened by its own covariance: contact means
    # average ~6 cells and regional ones ~90, so their spreads differ.
    W_own = W_nbr = None
    if sc.get("whiten", False):
        def _inv_sqrt(X):
            w, E = np.linalg.eigh(np.cov(X, rowvar=False))
            return (E / np.sqrt(w)) @ E.T
        W_own = _inv_sqrt(CO)
        Wn = [_inv_sqrt(C) for C in CN]
        CO = CO @ W_own
        CN = [C @ W for C, W in zip(CN, Wn)]
        Q_own = W_own if Q_own is None else Q_own @ W_own
        Qn = [W if Q is None else Q @ W for Q, W in zip(Qn, Wn)]
        W_nbr = _block_diag(Wn)
        say("  own and neighbour blocks whitened (unit variance, uncorrelated)")
    if all(Q is None for Q in Qn) and len(Qn) == 1:
        Q_nbr = None                                      # legacy: raw disc, unprojected
    else:
        Q_nbr = _block_diag([np.eye(d) if Q is None else Q for Q in Qn])
    nbr_blocks = [(b, int(C.shape[1])) for (b, _), C in zip(layout, CN)]
    CN = np.hstack(CN)

    cond_own = float(np.linalg.cond(np.column_stack([np.ones(len(CO)), CO])))
    cond_nbr = float(np.linalg.cond(np.column_stack([np.ones(len(CN)), CN])))
    say(f"  condition numbers: own {cond_own:.3g}, neighbour {cond_nbr:.3g}")
    kmax = float(sc["condition_number_max"])
    if max(cond_own, cond_nbr) > kmax:
        raise NotImplementedError(
            f"condition number exceeds {kmax:g}; spec 3.1b asks for a PCA "
            "reduction to 99.9% of variance here. Not triggered on this data, "
            "so deliberately not implemented -- add it if a new sample needs it.")

    adata_density = df["density"].to_numpy(float)
    # frozen_basis is outside the configuration fingerprint (hpc/worker.py): it
    # records which basis a fit was made with, it does not change the model
    pin = (cfg.get("frozen_basis") or {}).get("density_weight_drop_col")
    dw = _Spline(np.log(df["density"].to_numpy(float)), mc["weight_density_spline_df"],
                 knot_quantiles=mc.get("weight_density_knot_quantiles"),
                 scale=bool(mc.get("scale_weight_splines", False)), drop=pin)
    dg = _Spline(np.log(df["density"].to_numpy(float)), mc["background_spline_df"])

    technical = str(mc.get("technical", "area"))
    coef, r2 = None, float("nan")
    if technical == "mistic":
        # MisTIC replaces area: no area spline, the MisTIC-weighted library
        # size, and m for the additive misassignment term
        logN_col, m_col = str(mc["mistic_library_size"]), str(mc["mistic_m"])
        lacking = [c for c in (logN_col, m_col) if c not in df.columns]
        if lacking:
            raise KeyError(f"technical: mistic, but the table lacks {lacking}; run "
                           "run/add_mistic_columns.py first")
        lib = df[logN_col].to_numpy(float)
        mv = df[m_col].to_numpy(float)
        if not (np.all(lib > 0) and np.all(np.isfinite(lib))):
            raise ValueError(f"{logN_col} must be positive and finite")
        if not (np.all(mv >= 0) and np.all(np.isfinite(mv))):
            raise ValueError(f"{m_col} must be non-negative and finite")
        area_spline = None
        say(f"  technical: mistic -- library size {logN_col}, additive m from {m_col} "
            f"(mean {mv.mean():.4f}); no area term")
    elif technical == "plain":
        # the simple baseline: raw library size as the offset, no area term,
        # no MisTIC term -- state, density and depth only
        logN_col, m_col, area_spline = "N", None, None
        say("  technical: plain -- library size N, no area term, no MisTIC term")
    elif technical == "area":
        logN_col, m_col = "N", None
        area_spline = _Spline(np.log(df["area"].to_numpy(float)), mc["area_spline_df"])
        # spec 2.2: orthogonalise log N against the area spline only if strongly collinear
        logN = np.log(df["N"].to_numpy(float))
        la = np.log(df["area"].to_numpy(float))
        r = float(np.corrcoef(logN, la)[0, 1])
        if abs(r) > float(dc["corr_logN_logarea_orthogonalise_above"]):
            A = np.column_stack([np.ones(len(logN)), area_spline(la)])
            coef, *_ = np.linalg.lstsq(A, logN, rcond=None)
            fit = A @ coef
            r2 = float(1.0 - np.var(logN - fit) / np.var(logN))
            say(f"  corr(log N, log area) = {r:.3f} -> orthogonalised, aux R2 = {r2:.4f}")
        else:
            say(f"  corr(log N, log area) = {r:.3f} -> no orthogonalisation needed")
    else:
        raise ValueError(f"model.technical must be 'mistic', 'plain' or 'area', not {technical!r}")

    return FrozenBasis(
        phi_names=phi_names, own_names=own_names, nbr_names=nbr_names,
        own_cols=own_cols, m_own=m_own, m_nbr=m_nbr, Q_own=Q_own, Q_nbr=Q_nbr,
        rowsum_constant=const, area_spline=area_spline,
        density_weight_spline=dw, density_bg_spline=dg,
        logN_ortho_coef=coef, logN_ortho_r2=r2,
        cond_own=cond_own, cond_nbr=cond_nbr,
        mean_log_density=float(np.log(adata_density).mean()),
        W_own=W_own, W_nbr=W_nbr, nbr_cols=nbr_cols, nbr_blocks=nbr_blocks,
        technical=technical, logN_column=logN_col, m_column=m_col)


def spec_from_basis(basis, cfg, stage="B", theta_free=None, weight_ridge=0.0,
                    rank=None, n_reference=None):
    """Build a `ModelSpec` whose column counts match this basis.

    The spline bases carry `df - 1` columns each (the constant is projected out,
    see `_Spline`), so the spec must be derived from the basis rather than from
    the configured `df` values.
    """
    from .model import ModelSpec

    mc = cfg["model"]
    if theta_free is None:                 # production setting, from the config
        theta_free = bool(mc.get("theta_free", True))
    b_pen = str(mc.get("b_penalty", "nuclear"))
    tau = float(mc["balance_tau"] if b_pen == "frobenius" else mc["ridge_tau"])
    return ModelSpec(
        stage=stage, rank=int(mc["rank"] if rank is None else rank),
        theta_free=bool(theta_free), d_own=basis.d_own, d_nbr=basis.d_nbr,
        df_area=basis.n_area, df_weight=basis.n_weight, df_bg=basis.n_bg,
        ridge_tau=tau, weight_ridge=float(weight_ridge), b_penalty=b_pen,
        density_interaction=bool(mc.get("density_interaction", False)),
        density_interaction_spline=bool(mc.get("density_interaction_spline", False)),
        n_reference=int(mc.get("weight_ridge_n_reference", 0)
                        if n_reference is None else n_reference),
        oxygen=bool(mc.get("oxygen", True)),
        mistic=basis.technical == "mistic", technical=basis.technical)

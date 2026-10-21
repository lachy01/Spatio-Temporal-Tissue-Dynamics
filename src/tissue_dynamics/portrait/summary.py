"""One portrait, reduced to the numbers that uncertainty and sensitivity runs compare.

`summarise(model, arch, ...)` returns, for one set of weight-model parameters and
one death anchor:

  pi_bar        the anchor per archetype
  confinement   d eta / d log n at the 90th percentile of density, per archetype
  own_sweep     every crossing of pi_k = pi_bar_k as total density is swept at the
                archetype's own mean neighbourhood composition (stable = ceiling,
                unstable = escape threshold)
  ce_path       the tumour's ceilings on the path from the q50 neighbourhood to a
                pure tumour one (FINDINGS §4)
  fixed_points  every interior and face fixed point in the priority slices,
                classified, with whether it lies inside the support mask
  topology      the spec 8.3 key: the sorted (slice, classification) of interior
                fixed points inside support; a replicate "preserves topology" when
                its key equals the point estimate's

The slice geometry -- grids, reference states, support masks -- depends only on
the observed states, so `build_slices` computes it once and every replicate is
scored on the same geometry. The fixed-point search, support mask and
classification are the portrait script's (run/portrait_preliminary.py imports
them from here).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import jacobian, net_rate

PAIRS = [("CAFs", "Myeloid"), ("Cancer Epithelial", "CAFs"),
         ("Cancer Epithelial", "Myeloid"), ("Cancer Epithelial", "T-cells")]
MIN_CELLS, NSUPP = 20, 30


def support_mask(GA, GB, obs_ab, held_ok, ka, kb, Xo):
    """Spec 5: per-axis q95 cap, >=20 locally supported cells, convex hull."""
    from scipy.spatial import Delaunay

    pts = np.column_stack([GA.ravel(), GB.ravel()])
    cap = (pts[:, 0] <= np.percentile(Xo[:, ka], 95)) & \
          (pts[:, 1] <= np.percentile(Xo[:, kb], 95))
    hull = Delaunay(obs_ab).find_simplex(pts) >= 0
    # "min_cells within the grid cell" has to be counted on a support-sized bin,
    # not the plotting resolution: 150x150 cells would need 450k conditioned
    # observations to average 20 per bin. Count on a coarse NSUPP x NSUPP grid in
    # log space and map each fine point to its coarse bin.
    ea = np.linspace(np.log(GA[0, 0]), np.log(GA[0, -1]), NSUPP + 1)
    eb = np.linspace(np.log(GB[0, 0]), np.log(GB[-1, 0]), NSUPP + 1)
    sel = np.log(obs_ab[held_ok])
    cnt = np.histogram2d(sel[:, 0], sel[:, 1], bins=[ea, eb])[0] >= MIN_CELLS
    ia = np.clip(np.digitize(np.log(pts[:, 0]), ea) - 1, 0, NSUPP - 1)
    ib = np.clip(np.digitize(np.log(pts[:, 1]), eb) - 1, 0, NSUPP - 1)
    local = cnt[ia, ib]
    return (cap & hull & local).reshape(GA.shape), int(held_ok.sum())


def find_fixed_points(ref, ka, kb, model, arch, ox, lo, hi, F_red, GA, GB, mask, bound=None):
    """Spec 6.1 grid-seeded multistart + 6.2 faces, in log X with a finite guard."""
    from scipy.optimize import root

    lb, ub = np.log([lo[0], lo[1]]) - 2.0, np.log([hi[0], hi[1]]) + 2.0

    def resid(u, free=(True, True)):
        u = np.clip(u, lb[:len(u)] if len(u) == 2 else lb[:1], ub[:len(u)])
        Z = ref.copy()
        vals = np.exp(u)
        idx = [i for i, f in zip((ka, kb), free) if f]
        for i, v in zip(idx, vals):
            Z[i] = v
        for i, f in zip((ka, kb), free):
            if not f:
                Z[i] = 0.0
        rate, _p = net_rate(Z[None], model, arch, ox, bound)
        r = [rate[0, i] for i in idx]
        return np.where(np.isfinite(r), r, 1e3)

    seeds = []
    nrm = np.linalg.norm(F_red, axis=0)
    thr = np.percentile(nrm[np.isfinite(nrm)], 5)
    sy, sx = np.where((nrm <= thr))
    for i, j in list(zip(sy, sx))[::max(1, len(sy) // 60)]:
        seeds.append(np.log([GA[i, j], GB[i, j]]))
    found = []
    for s in seeds:
        r = root(resid, s, method="hybr")
        if not r.success:
            continue
        xf = np.exp(np.clip(r.x, lb, ub))
        if not np.all(np.isfinite(xf)) or np.max(np.abs(resid(r.x))) > 1e-8:
            continue
        if any(np.allclose(xf, s2, rtol=1e-3) for s2 in found):
            continue
        found.append(xf)
    faces = []
    for free in ((True, False), (False, True)):
        idx = ka if free[0] else kb
        s0 = np.log([np.median(GA if free[0] else GB)])
        r = root(lambda u: resid(u, free), s0, method="hybr")
        if r.success and np.max(np.abs(resid(r.x, free))) < 1e-8:
            v = float(np.exp(np.clip(r.x, lb[:1], ub[:1])[0]))
            faces.append((idx, v, free))
    return found, faces


def classify(Z, model, arch, ox, ka, kb, bound=None):
    J = jacobian(Z, model, arch, ox, bound=bound)
    ev_f = np.linalg.eigvals(J)
    ev_r = np.linalg.eigvals(J[np.ix_([ka, kb], [ka, kb])])
    cls = ("stable" if np.all(ev_r.real < 0) else
           "unstable" if np.all(ev_r.real > 0) else "saddle")
    return cls, ev_r, ev_f


@dataclass
class Slice:
    pair: str
    qref: int
    ka: int
    kb: int
    ref: np.ndarray
    ga: np.ndarray
    gb: np.ndarray
    GA: np.ndarray
    GB: np.ndarray
    M: np.ndarray
    lo: np.ndarray
    hi: np.ndarray


def build_slices(Xo, names, pairs=PAIRS, qrefs=(25, 50), ngrid=100):
    """The priority slices' grids, reference states and support masks (spec 5, 7)."""
    out = []
    K = len(names)
    for qref in qrefs:
        ref_all = np.percentile(Xo, qref, axis=0)
        for na, nb in pairs:
            if na not in names or nb not in names:
                continue
            ka, kb = names.index(na), names.index(nb)
            both = (Xo[:, ka] > 0) & (Xo[:, kb] > 0)
            obs = Xo[both][:, [ka, kb]]
            lo = np.array([np.percentile(obs[:, 0], 1), np.percentile(obs[:, 1], 1)])
            hi = np.array([np.percentile(Xo[:, ka], 95), np.percentile(Xo[:, kb], 95)])
            ga = np.logspace(*np.log10([lo[0], hi[0]]), ngrid)
            gb = np.logspace(*np.log10([lo[1], hi[1]]), ngrid)
            GA, GB = np.meshgrid(ga, gb)
            held = [m for m in range(K) if m not in (ka, kb)]
            tol = np.array([0.5 * np.subtract(*np.percentile(Xo[:, m], [75, 25])) for m in held])
            held_ok = np.all(np.abs(Xo[both][:, held] - ref_all[held]) < np.abs(tol), axis=1)
            M, _n = support_mask(GA, GB, obs, held_ok, ka, kb, Xo)
            out.append(Slice(f"{na} x {nb}", qref, ka, kb, ref_all.copy(), ga, gb, GA, GB, M, lo, hi))
    return out


def _crossings(grid, f, dens):
    s = np.where(np.diff(np.sign(f)))[0]
    return [{"n": float(grid[i]), "pct": float(100 * np.mean(dens < grid[i])),
             "kind": "stable" if f[i] > 0 else "unstable"} for i in s]


def summarise(model, arch, ox, slices, dens, nbr_own, arche, bound=None, n_sweep=3000, n_blend=101):
    """The portrait's summary numbers for one parameter set (module docstring)."""
    names = arch.names
    out = {"pi_bar": {n: float(v) for n, v in zip(names, arch.pi_bar)}}
    q90 = float(np.percentile(dens, 90))
    out["confinement"] = {n: float(model.dh_dlogn(arch.A[k], np.array([q90]))[0])
                          for k, n in enumerate(names)}
    grid = np.linspace(np.percentile(dens, .5), np.percentile(dens, 99.9), n_sweep)
    sweep = {}
    for k, n in enumerate(names):
        if not (arche == n).any():
            continue
        pb = nbr_own[arche == n].mean(0)
        if bound is None:
            e = model.eta(np.repeat(arch.A[k][None], len(grid), 0), np.repeat(pb[None], len(grid), 0),
                          grid, ox)
            f = 1 / (1 + np.exp(-e)) - arch.pi_bar[k]
        else:
            # the type's mean loadings in the portrait's own state (type fractions
            # for counts), as run/portrait_dynamics_summary.py; barycentric
            # loadings here put counts-state X_k on the wrong scale for the bound
            Xk = arch.X_obs[arche == n]
            w = (Xk / np.clip(Xk.sum(1, keepdims=True), 1e-12, None)).mean(0)
            f = net_rate(grid[:, None] * w[None, :], model, arch, ox, bound)[0][:, k]
        sweep[n] = {"crossings": _crossings(grid, f, dens), "sign": "grows" if f[0] > 0 else "shrinks"}
    out["own_sweep"] = sweep
    if "Cancer Epithelial" in names:
        k = names.index("Cancer Epithelial")
        Xo = arch.X_obs
        ref = np.percentile(Xo, 50, axis=0)
        w0 = ref / ref.sum()
        w1 = np.zeros(len(names))
        w1[k] = 1.0
        ceil, n_cross = [], []
        for t in np.linspace(0, 1, n_blend):
            w = (1 - t) * w0 + t * w1
            if bound is None:
                e = model.eta(np.repeat(arch.A[k][None], len(grid), 0),
                              np.repeat((w @ arch.A)[None], len(grid), 0), grid, ox)
                f = 1 / (1 + np.exp(-e)) - arch.pi_bar[k]
            else:
                f = net_rate(grid[:, None] * w[None, :], model, arch, ox, bound)[0][:, k]
            cr = _crossings(grid, f, dens)
            n_cross.append(len(cr))
            ceil += [c["n"] for c in cr if c["kind"] == "stable"]
        out["ce_path"] = {"ceiling_min": float(min(ceil)) if ceil else None,
                          "ceiling_max": float(max(ceil)) if ceil else None,
                          "ceiling_pct_min": float(100 * np.mean(dens < min(ceil))) if ceil else None,
                          "ceiling_pct_max": float(100 * np.mean(dens < max(ceil))) if ceil else None,
                          "crossings_per_blend": sorted(set(n_cross)),
                          "folds": int(np.sum(np.diff(n_cross) != 0))}
    fps = []
    for sl in slices:
        X = np.repeat(sl.ref[None], sl.GA.size, axis=0)
        X[:, sl.ka], X[:, sl.kb] = sl.GA.ravel(), sl.GB.ravel()
        R, _Pi = net_rate(X, model, arch, ox, bound)
        U = (R[:, sl.ka] / np.log(2)).reshape(sl.GA.shape)
        V = (R[:, sl.kb] / np.log(2)).reshape(sl.GA.shape)
        found, faces = find_fixed_points(sl.ref, sl.ka, sl.kb, model, arch, ox, sl.lo, sl.hi,
                                         np.stack([U, V]), sl.GA, sl.GB, sl.M, bound)
        for xf in found:
            Z = sl.ref.copy()
            Z[sl.ka], Z[sl.kb] = xf
            cls, ev_r, ev_f = classify(Z, model, arch, ox, sl.ka, sl.kb, bound)
            gi = (np.abs(sl.ga - xf[0]).argmin(), np.abs(sl.gb - xf[1]).argmin())
            fps.append({"slice": sl.pair, "q_ref": sl.qref, "X_a": float(xf[0]), "X_b": float(xf[1]),
                        "n": float(Z.sum()), "classification": cls,
                        "in_support": bool(sl.M[gi[1], gi[0]]),
                        "unstable_held": bool(np.any(ev_f.real > 1e-12)),
                        "max_re_full": float(ev_f.real.max())})
        for idx, v, free in faces:
            fps.append({"slice": sl.pair, "q_ref": sl.qref, "X_a": v if free[0] else 0.0,
                        "X_b": 0.0 if free[0] else v, "n": None, "classification": "face",
                        "in_support": False, "unstable_held": None, "max_re_full": None})
    out["fixed_points"] = fps
    key = sorted(f"{f['slice']}|q{f['q_ref']}|{f['classification']}" for f in fps
                 if f["classification"] != "face" and f["in_support"])
    out["topology"] = key
    out["topology_interior"] = sorted(f"{f['slice']}|q{f['q_ref']}|{f['classification']}"
                                      for f in fps if f["classification"] != "face")
    return out

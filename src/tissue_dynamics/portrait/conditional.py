"""Conditional pairwise slices (FINDINGS §15), and their sensitivity and bootstrap
as Spartan array jobs.

A slice holds two archetypes on the axes, (X_a, X_b), and the other seven at a
reference. Under the conditional reference, the others sit at their kernel-weighted
mean over observed neighbourhoods near (log2 X_a, log2 X_b): a Nadaraya-Watson
estimate of E[X_others | X_a, X_b], binned, Gaussian-smoothed and interpolated.
run/portrait_conditional_slices.py draws the point estimate.

  portrait_cond_sens  one task per (variant, bound, pairs): the point-estimate
                      model with one setting changed (`VARIANTS`); the all-pairs
                      scan runs the base settings one pair per task.
  portrait_cond_boot  one task per replicate source and chunk:
                        param  the portrait_param_boot draws, regenerated from
                               their seeds; reference at the full data
                        block  the portrait_block_boot refits; reference rebuilt
                               from the same resampled blocks (model and data
                               surface both vary)
                        ref    the point-estimate model; reference rebuilt from
                               fresh block resamples (the surface's own noise)
                      Anchors come from the stored portrait_*_boot summaries; the
                      OSDR bound is refitted per replicate.

The slice grid, its limits and the support mask stay at their full-data values in
every replicate (spec 3.1's frozen-basis rule), so replicates share axes.
"""
from __future__ import annotations

import dataclasses
import json
import os

import numpy as np

AREA = np.pi * 80.0 ** 2 / 1e4                 # the 80 um disc in (100 um)^2: count = X * AREA
NGRID = 100
SUB = slice(1, None, 2)                         # stored rate grids: 100 x 100 -> 50 x 50
BOUNDS = ("none", "osdr")
N_PER_CHUNK = 10
N_BLOCK_REPS, N_PARAM_CHUNKS, N_REF_CHUNKS = 50, 50, 20

# sensitivity variants: one setting changed from the base (bandwidth 0.25 log2, 160
# bins, neighbourhoods with both axis types present, production anchor and radius)
VARIANTS = {
    "base": {},
    "bw 0.10": {"bw": 0.10},
    "bw 0.15": {"bw": 0.15},
    "bw 0.40": {"bw": 0.40},
    "bw 0.60": {"bw": 0.60},
    "bins 320": {"nbins": 320},
    "both >= 2 cells": {"min_cells": 2},
    "zeros kept": {"pseudo_cells": 1},
    "anchor x0.95": {"anchor_scale": 0.95},
    "anchor x1.05": {"anchor_scale": 1.05},
    "radius 60 um": {"radius": 60},
    "radius 100 um": {"radius": 100},
}


class Reference:
    """Full state X(X_a, X_b) for one pair under one reference mode.

    mode "q50": the others at their overall medians; "co-present": at their medians
    where both axis types have >= min_x; "conditional": at the kernel-weighted mean
    given (log2 X_a, log2 X_b), bandwidth `bw` log2 units, on `nbins` bins over the
    slice's range +-1 log2. Conditional options: `cond_min` keeps only
    neighbourhoods with both axis densities >= cond_min (default: both > 0);
    `pseudo` conditions on log2(X + pseudo) over every neighbourhood instead.
    """

    def __init__(self, mode, Xo, ka, kb, min_x, bw, lo, hi, nbins=160, cond_min=None, pseudo=None):
        self.mode, self.ka, self.kb, self.pseudo = mode, ka, kb, pseudo
        if mode == "q50":
            self.ref = np.median(Xo, axis=0)
        elif mode == "co-present":
            sel = (Xo[:, ka] >= min_x) & (Xo[:, kb] >= min_x)
            self.ref = np.median(Xo[sel], axis=0)
            self.n_sel = int(sel.sum())
        else:
            if pseudo is not None:
                use = np.ones(len(Xo), bool)
            elif cond_min is not None:
                use = (Xo[:, ka] >= cond_min) & (Xo[:, kb] >= cond_min)
            else:
                use = (Xo[:, ka] > 0) & (Xo[:, kb] > 0)
            self.n_sel = int(use.sum())
            off = 0.0 if pseudo is None else pseudo
            ea = np.linspace(np.log2(lo[0] + off) - 1, np.log2(hi[0] + off) + 1, nbins + 1)
            eb = np.linspace(np.log2(lo[1] + off) - 1, np.log2(hi[1] + off) + 1, nbins + 1)
            La, Lb = np.log2(Xo[use, ka] + off), np.log2(Xo[use, kb] + off)
            ia = np.clip(np.digitize(La, ea) - 1, 0, nbins - 1)
            ib = np.clip(np.digitize(Lb, eb) - 1, 0, nbins - 1)
            self.interp = _smoothed_means(ia * nbins + ib, Xo[use], nbins, ea, eb, bw)

    def state(self, xa, xb):
        xa, xb = np.atleast_1d(xa).astype(float), np.atleast_1d(xb).astype(float)
        if self.mode == "conditional":
            off = 0.0 if self.pseudo is None else self.pseudo
            pts = np.column_stack([np.log2(xa + off), np.log2(xb + off)])
            X = np.column_stack([np.clip(f(pts), 0, None) for f in self.interp])
        else:
            X = np.repeat(self.ref[None], len(xa), axis=0)
        X[:, self.ka], X[:, self.kb] = xa, xb
        return X


def _smoothed_means(flat, X, nbins, ea, eb, bw):
    """Binned Nadaraya-Watson: Gaussian-smoothed per-bin sums over smoothed counts."""
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import gaussian_filter

    sig = bw / (ea[1] - ea[0]), bw / (eb[1] - eb[0])
    cnt = gaussian_filter(np.bincount(flat, minlength=nbins * nbins).reshape(nbins, nbins).astype(float),
                          sig, mode="nearest")
    ca, cb = 0.5 * (ea[1:] + ea[:-1]), 0.5 * (eb[1:] + eb[:-1])
    out = []
    for k in range(X.shape[1]):
        s = np.bincount(flat, weights=X[:, k], minlength=nbins * nbins).reshape(nbins, nbins)
        m = gaussian_filter(s, sig, mode="nearest") / np.clip(cnt, 1e-9, None)
        out.append(RegularGridInterpolator((ca, cb), m, bounds_error=False, fill_value=None))
    return out


def rates(R, xa, xb, model, arch, ox, bound):
    from .core import net_rate

    r, _ = net_rate(R.state(xa, xb), model, arch, ox, bound)
    return r[:, R.ka], r[:, R.kb]


def fixed_points(R, U, V, GA, GB, lo, hi, model, arch, ox, bound):
    """Roots of (r_a, r_b) in log X from the slowest grid points (`hybr`); stability
    from the 2 x 2 Jacobian in log X by central differences."""
    from scipy.optimize import root

    lb, ub = np.log([lo[0], lo[1]]) - 2.0, np.log([hi[0], hi[1]]) + 2.0

    def resid(u):
        u = np.clip(u, lb, ub)
        ra, rb = rates(R, np.exp(u[0]), np.exp(u[1]), model, arch, ox, bound)
        out = np.array([ra[0], rb[0]])
        return np.where(np.isfinite(out), out, 1e3)

    nrm = np.hypot(U, V)
    thr = np.nanpercentile(nrm, 5)
    sy, sx = np.where(nrm <= thr)
    found = []
    for i, j in list(zip(sy, sx))[::max(1, len(sy) // 60)]:
        s = root(resid, np.log([GA[i, j], GB[i, j]]), method="hybr")
        if not s.success or np.max(np.abs(resid(s.x))) > 1e-8:
            continue
        xf = np.exp(np.clip(s.x, lb, ub))
        if np.any(s.x <= lb + 1e-6) or np.any(s.x >= ub - 1e-6):
            continue
        if not any(np.allclose(xf, f, rtol=1e-3) for f in found):
            found.append(xf)
    out = []
    for xf in found:
        u, h = np.log(xf), 1e-4
        J = np.zeros((2, 2))
        for c in range(2):
            e = np.zeros(2)
            e[c] = h
            J[:, c] = (resid(u + e) - resid(u - e)) / (2 * h)
        ev = np.linalg.eigvals(J)
        cls = "stable" if np.all(ev.real < 0) else "unstable" if np.all(ev.real > 0) else "saddle"
        out.append((xf, cls))
    return out


@dataclasses.dataclass
class SliceGeometry:
    """The full-data grid, limits and conditional support of one pair."""
    pair: str
    ka: int
    kb: int
    lo: np.ndarray
    hi: np.ndarray
    ga: np.ndarray
    gb: np.ndarray
    GA: np.ndarray
    GB: np.ndarray
    M: np.ndarray

    @classmethod
    def build(cls, Xo, names, na, nb):
        from .summary import support_mask

        ka, kb = names.index(na), names.index(nb)
        both = (Xo[:, ka] > 0) & (Xo[:, kb] > 0)
        obs = Xo[both][:, [ka, kb]]
        lo = np.percentile(obs, 1, axis=0)
        hi = np.array([np.percentile(Xo[:, ka], 95), np.percentile(Xo[:, kb], 95)])
        ga, gb = np.logspace(*np.log10([lo[0], hi[0]]), NGRID), np.logspace(*np.log10([lo[1], hi[1]]), NGRID)
        GA, GB = np.meshgrid(ga, gb)
        M, _n = support_mask(GA, GB, obs, np.ones(len(obs), bool), ka, kb, Xo)
        return cls(f"{na} x {nb}", ka, kb, lo, hi, ga, gb, GA, GB, M)


def conditional_slice(g, Xref, model, arch, ox, bound, bw=0.25, nbins=160, cond_min=None, pseudo=None):
    """One conditional slice: fixed points (with in-support flags) and the rate grids.
    `Xref` is the set of observed states the reference is built from."""
    R = Reference("conditional", Xref, g.ka, g.kb, None, bw, g.lo, g.hi, nbins=nbins,
                  cond_min=cond_min, pseudo=pseudo)
    ra, rb = rates(R, g.GA.ravel(), g.GB.ravel(), model, arch, ox, bound)
    U, V = (ra / np.log(2)).reshape(g.GA.shape), (rb / np.log(2)).reshape(g.GA.shape)
    fps = []
    for xf, cls in fixed_points(R, U, V, g.GA, g.GB, g.lo, g.hi, model, arch, ox, bound):
        gi = (np.abs(g.ga - xf[0]).argmin(), np.abs(g.gb - xf[1]).argmin())
        on_grid = g.lo[0] <= xf[0] <= g.hi[0] and g.lo[1] <= xf[1] <= g.hi[1]
        Z = R.state(xf[0], xf[1])[0]
        fps.append({"X_a": float(xf[0]), "X_b": float(xf[1]), "n": float(Z.sum()), "classification": cls,
                    "in_support": bool(on_grid and g.M[gi[1], gi[0]])})
    return {"fixed_points": fps, "U": U, "V": V, "n_cond": R.n_sel}


# ----------------------------------------------------------------------------- jobs

def enumerate_cond_tasks(job, cfg):
    from .summary import PAIRS

    if job == "portrait_cond_sens":
        tasks = [{"job": job, "variant": v, "bound": b, "pairs": [list(p) for p in PAIRS]}
                 for v in VARIANTS for b in BOUNDS]
        # the all-pairs scan at the base settings, one pair per task
        names = json.load(open(cfg["data"]["table_parquet"].replace(".parquet", ".phi_names.json")))
        prio = {tuple(p) for p in PAIRS}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if (a, b) in prio or (b, a) in prio:
                    continue
                tasks += [{"job": job, "variant": "all pairs", "bound": bd, "pairs": [[a, b]]} for bd in BOUNDS]
        return tasks
    if job == "portrait_cond_boot":
        return ([{"job": job, "source": "param", "chunk": c} for c in range(N_PARAM_CHUNKS)]
                + [{"job": job, "source": "block", "rep": r} for r in range(N_BLOCK_REPS)]
                + [{"job": job, "source": "ref", "chunk": c} for c in range(N_REF_CHUNKS)])
    raise ValueError(job)


def _context(cfg, hpc):
    from .core import PortraitModel, build_archetypes
    from ..mixture.basis import spec_from_basis
    from ..mixture.model import build_data, responsibilities
    from ..mixture.tasks import _load
    import jax.numpy as jnp

    df, basis = _load(cfg)
    pe = json.load(open(os.path.join(hpc, "point_estimate.json")))
    spec = spec_from_basis(basis, cfg, stage="B", weight_ridge=float(pe["weight_ridge"]))
    data = build_data(df, basis, cfg, spec)
    P0 = {k: jnp.asarray(np.asarray(v)) for k, v in pe["params"].items()}
    w = np.asarray(responsibilities(P0, data.as_jax(), spec))
    arch, _W = build_archetypes(df, basis, pi_hat=w, state="counts",
                                radius_um=float(cfg["data"]["neighbour_radius_um"]))
    return dict(df=df, basis=basis, pe=pe, arch=arch, Xo=np.clip(arch.X_obs, 0, None), spec=spec, data=data,
                arche=df["archetype"].astype(str).to_numpy(), ox=float(np.median(df["oxygen"])),
                model=PortraitModel.from_fit(pe["params"], basis))


def _bound(kind, model, arch, ox, arche):
    from .core import fit_osdr_bound

    return fit_osdr_bound(model, arch, ox, cell_archetype=arche, power=8) if kind == "osdr" else None


def _pack(res):
    """JSON form of one slice: fixed points and the 50 x 50 rate grids (5 decimals)."""
    return {"fixed_points": res["fixed_points"], "n_cond": res["n_cond"],
            "U": np.round(res["U"][SUB, SUB], 5).tolist(), "V": np.round(res["V"][SUB, SUB], 5).tolist()}


def run_cond_sens(task, cfg, say=None):
    say = say or (lambda *a: None)
    v = VARIANTS.get(task["variant"], {})
    if "radius" in v:
        from ..mixture.data import load_config
        old = os.environ.get("MIX_DATASET")
        os.environ["MIX_DATASET"] = f"r{v['radius']}"
        try:
            rcfg = load_config()
        finally:
            if old is None:
                os.environ.pop("MIX_DATASET")
            else:
                os.environ["MIX_DATASET"] = old
        ctx = _context(rcfg, f"outputs/hpc_r{v['radius']}")
    else:
        ctx = _context(cfg, cfg.get("hpc_dir") or "outputs/hpc")
    arch = ctx["arch"]
    if "anchor_scale" in v:
        arch = dataclasses.replace(arch, pi_bar=arch.pi_bar * v["anchor_scale"])
    bound = _bound(task["bound"], ctx["model"], arch, ctx["ox"], ctx["arche"])
    out = {}
    for na, nb in task["pairs"]:
        g = SliceGeometry.build(ctx["Xo"], arch.names, na, nb)
        res = conditional_slice(g, ctx["Xo"], ctx["model"], arch, ctx["ox"], bound,
                                bw=v.get("bw", 0.25), nbins=v.get("nbins", 160),
                                cond_min=(v["min_cells"] / AREA if "min_cells" in v else None),
                                pseudo=(v["pseudo_cells"] / AREA if "pseudo_cells" in v else None))
        out[g.pair] = {**_pack(res), "ga": g.ga[SUB].tolist(), "gb": g.gb[SUB].tolist(),
                       "M": g.M[SUB, SUB].astype(int).tolist()}
        say(f"  {g.pair}: {[(round(f['X_a'], 2), round(f['X_b'], 2), f['classification'], f['in_support']) for f in res['fixed_points']]}")
    return {"slices": out, "pi_bar": dict(zip(arch.names, map(float, arch.pi_bar))),
            "technical": cfg["model"].get("technical", "area"), "rank": int(cfg["model"]["rank"]),
            "weight_ridge": float(ctx["pe"]["weight_ridge"]) if "radius" not in v else None}


def _replicates(task, cfg, ctx):
    """(label, params or None, pi_bar, cell indices or None) for one boot task."""
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree

    from .uncertainty import COV_PATH, N_DRAWS_PER_TASK

    hpc = cfg.get("hpc_dir") or "outputs/hpc"
    names = ctx["arch"].names
    src = task["source"]
    if src == "param":
        z = np.load(COV_PATH)
        P0 = {k: jnp.asarray(np.asarray(v)) for k, v in ctx["pe"]["params"].items()}
        flat0, unravel = ravel_pytree(P0)
        if not np.allclose(np.asarray(flat0), z["theta"]):
            raise RuntimeError("param_cov.npz was computed on a different point estimate")
        stored = json.load(open(f"{hpc}/portrait_param_boot/{task['chunk']:05d}.json"))
        rng = np.random.default_rng(20261020 + int(task["chunk"]))
        for b in range(N_DRAWS_PER_TASK):
            th = z["theta"] + z["evecs"] @ (np.sqrt(np.clip(z["evals"], 0, None))
                                            * rng.standard_normal(len(z["evals"])))
            d = stored["draws"][b]
            pb = np.array([d["summary"]["counts"]["pi_bar"][n] for n in names])
            P = unravel(jnp.asarray(th))
            if b == 0:                  # the regenerated draw must reproduce the stored anchors
                from ..mixture.model import responsibilities
                w = np.asarray(responsibilities(P, ctx["data"].as_jax(), ctx["spec"]))
                chk = np.array([w[ctx["arche"] == n].mean() for n in names])
                if np.max(np.abs(chk - pb)) > 1e-8:
                    raise RuntimeError("regenerated parametric draw does not match portrait_param_boot")
            yield d["draw"], {k: np.asarray(v) for k, v in P.items()}, pb, None
    else:
        from ..mixture.inference import spatial_blocks

        xy = ctx["df"][["x", "y"]].to_numpy(float)
        labels = spatial_blocks(xy, int(cfg["inference"]["n_blocks_target"]))
        blocks = np.unique(labels)
        idx_of = {b: np.where(labels == b)[0] for b in blocks}
        if src == "block":
            r = json.load(open(f"{hpc}/portrait_block_boot/{task['rep']:05d}.json"))
            rng = np.random.default_rng(int(task["rep"]))          # the refit's own resample
            idx = np.concatenate([idx_of[b] for b in rng.choice(blocks, size=len(blocks), replace=True)])
            if len(idx) != r["n_cells"]:
                raise RuntimeError("regenerated block resample does not match the stored refit")
            pb = np.array([r["summary"]["counts"]["pi_bar"][n] for n in names])
            yield int(task["rep"]), r["params"], pb, idx
        else:
            for b in range(N_PER_CHUNK):
                rep = int(task["chunk"]) * N_PER_CHUNK + b
                rng = np.random.default_rng(10_000 + rep)
                idx = np.concatenate([idx_of[k] for k in rng.choice(blocks, size=len(blocks), replace=True)])
                yield rep, None, ctx["arch"].pi_bar, idx


def run_cond_boot(task, cfg, say=None):
    from .core import PortraitModel
    from .summary import PAIRS

    say = say or (lambda *a: None)
    ctx = _context(cfg, cfg.get("hpc_dir") or "outputs/hpc")
    geos = [SliceGeometry.build(ctx["Xo"], ctx["arch"].names, a, b) for a, b in PAIRS]
    reps = []
    for rep, params, pb, idx in _replicates(task, cfg, ctx):
        model = ctx["model"] if params is None else PortraitModel.from_fit(
            {k: np.asarray(v) for k, v in params.items()}, ctx["basis"])
        arch = dataclasses.replace(ctx["arch"], pi_bar=np.asarray(pb, float))
        Xref = ctx["Xo"] if idx is None else ctx["Xo"][idx]
        one = {"rep": int(rep), "slices": {}}
        for bd in BOUNDS:
            bound = _bound(bd, model, arch, ctx["ox"], ctx["arche"])
            one["slices"][bd] = {g.pair: _pack(conditional_slice(g, Xref, model, arch, ctx["ox"], bound))
                                 for g in geos}
        reps.append(one)
        say(f"  replicate {rep} done")
    return {"replicates": reps, "technical": cfg["model"].get("technical", "area"),
            "rank": int(cfg["model"]["rank"]), "weight_ridge": float(ctx["pe"]["weight_ridge"])}

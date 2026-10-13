"""Preliminary phase portrait from the fitted mixture (portrait spec 3-7).

    python run/portrait_preliminary.py                    # unbounded (the default)
    python run/portrait_preliminary.py --bound osdr [--power 8] [--target-at-max]

No refitting and no simulation: the field is the ODE's right-hand side on a grid.
The default is UNBOUNDED -- OSDR's `p_correction = c_k * d_k^n` is not applied.
Spec 5 says to check confinement first and add the correction only as a flagged
sensitivity variant, never as the default. `--bound osdr` is that variant
(`portrait.core.fit_osdr_bound`, OSDR's code and settings); it writes to its own
folder, outputs/portrait/osdr_p<power>[_target_at_max]/, and flags every figure.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import root
from scipy.spatial import Delaunay

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
import jax.numpy as jnp                                              # noqa: E402

from tissue_dynamics.mixture.basis import build_basis, spec_from_basis    # noqa: E402
from tissue_dynamics.mixture.data import load_config                      # noqa: E402
from tissue_dynamics.mixture.model import build_data, responsibilities    # noqa: E402
from tissue_dynamics.portrait.core import (PortraitModel, build_archetypes,  # noqa: E402
                                           covariates_from_state, fit_osdr_bound,
                                           interaction_matrix, jacobian, net_rate)

PAIRS = [("CAFs", "Myeloid"), ("Cancer Epithelial", "CAFs"),
         ("Cancer Epithelial", "Myeloid"), ("Cancer Epithelial", "T-cells")]
NGRID, MIN_CELLS, NSUPP = 150, 20, 30
OUT = "outputs/portrait"
OUT_F = OUT


def support_mask(GA, GB, obs_ab, held_ok, ka, kb, Xo):
    """Spec 5: per-axis q95 cap, >=20 locally supported cells, convex hull."""
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


def main():
    global OUT, OUT_F
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bound", choices=("none", "osdr"), default="none")
    ap.add_argument("--power", type=int, default=8, help="OSDR's published analyses use 8")
    ap.add_argument("--target-at-max", action="store_true",
                    help="hold the target type at its maximal density when setting c_k "
                         "(OSDR's docstring) instead of sampling it (OSDR's code)")
    a = ap.parse_args()
    if a.bound == "osdr":
        OUT = OUT_F = f"outputs/portrait/osdr_p{a.power}{'_target_at_max' if a.target_at_max else ''}"
    flag = ("[UNBOUNDED: no OSDR density correction]" if a.bound == "none" else
            f"[SENSITIVITY: OSDR density correction, power {a.power}"
            f"{', target held at max' if a.target_at_max else ''}]")
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["table_parquet"])
    df.attrs["phi_names"] = json.load(open("data/mixture_cells_major.phi_names.json"))
    basis = build_basis(df, cfg, verbose=False)
    pe = json.load(open("outputs/hpc/point_estimate.json"))
    spec = spec_from_basis(basis, cfg, stage="B", weight_ridge=float(pe["weight_ridge"]))
    data = build_data(df, basis, cfg, spec)
    P0 = {k: jnp.asarray(np.asarray(v)) for k, v in pe["params"].items()}
    pi_hat = np.asarray(responsibilities(P0, data.as_jax(), spec))

    model = PortraitModel.from_fit(pe["params"], basis)
    arch, W = build_archetypes(df, basis, pi_hat=pi_hat)
    names, K, Xo = arch.names, arch.K, arch.X_obs
    ox = float(np.median(df["oxygen"]))
    os.makedirs(OUT, exist_ok=True)
    bound = None
    if a.bound == "osdr":
        bound = fit_osdr_bound(model, arch, ox, cell_archetype=df["archetype"].astype(str).to_numpy(),
                               power=a.power, target_at_max=a.target_at_max)
    man = {"n_cells": int(len(df)), "K": K, "r2_recon": arch.r2_recon,
           "oxygen_held_at": ox, "bounded": bound is not None, "n_grid": NGRID,
           "min_cells": MIN_CELLS, "support_bins": NSUPP,
           "weight_ridge": float(pe["weight_ridge"]),
           "note": ("unbounded: OSDR p_correction deliberately not applied" if bound is None
                    else "SENSITIVITY VARIANT: OSDR maximal-density correction applied"),
           "assumptions": [
               "mean-field closure: E[dX/dt | all cells share neighbourhood N(x)]; "
               "spatial correlation is not represented",
               "existence of an interior fixed point is partly assumed -- anchoring "
               "h_minus[k] = pi_bar[k] forces each nullcline through the data cloud; "
               "the informative content is location, multiplicity and stability",
               "archetype identity is fixed: cells cannot move in PhiSpace without "
               "dividing or dying",
               "the archetype residual is discarded (R2_recon reported)",
               "oxygen and endothelial density are coupled in tissue but decoupled "
               "here; the vascular field is frozen at its median",
               "technical covariates are held at reference and are not on the causal "
               "path of the dynamics",
               "own_state_mode = 'vertex': pi_k is evaluated at phi = A[k]",
               "pi_bar uses the spec 3.2 signed-loading formula; the clipped variant "
               "differs by at most 0.017 and is recorded as pi_bar_clipped",
           ]}

    print(f"{len(df):,} cells | {K} archetypes | reconstruction R2 = {arch.r2_recon:.6f}")
    print(f"loadings outside [0,1]: {100*np.mean((W < 0) | (W > 1)):.1f}% of entries; "
          f"only {100*np.mean(np.all(Xo > 0, axis=1)):.1f}% of cells have every X_k > 0")
    man["frac_cells_all_positive"] = float(np.mean(np.all(Xo > 0, axis=1)))
    print("\ndeath anchor pi_bar (loading-weighted mean of fitted pi):")
    for k, nm in enumerate(names):
        print(f"  {nm:<20} {arch.pi_bar[k]:.4f}")
    man["pi_bar"] = {n: float(v) for n, v in zip(names, arch.pi_bar)}
    man["pi_bar_clipped"] = {n: float(v) for n, v in zip(names, arch.pi_bar_clipped)}

    # ---- spec 4: interaction matrix ---------------------------------------
    At = interaction_matrix(model, arch)
    pd.DataFrame(At, index=names, columns=names).to_csv(f"{OUT}/Atilde.csv")
    sv = np.linalg.svd(At, compute_uv=False)
    man["Atilde_singular_values"] = [float(v) for v in sv]
    print(f"\nAtilde = A B A^T  singular values: {np.round(sv[:4], 4)}  "
          f"(rank {int(np.sum(sv > 1e-8))}, inherited from rank(B))")
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    v = np.abs(At).max()
    im = ax.imshow(At, cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_xticks(range(K)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(K)); ax.set_yticklabels(names, fontsize=8)
    ax.set_title(r"$\tilde{A}=ABA^\top$: effect of column archetype on row's log-odds"
                 f"\n(rank {int(np.sum(sv > 1e-8))}, inherited from rank(B))", fontsize=10)
    fig.colorbar(im); fig.tight_layout(); fig.savefig(f"{OUT}/Atilde_heatmap.png", dpi=150)
    plt.close(fig)

    # ---- spec 5: extrapolation / confinement check -------------------------
    print("\n=== spec 5 confinement check: d(eta)/d(log n) at the 90th pct of n ===")
    q90 = float(np.percentile(df["density"], 90))
    conf = {}
    for k, nm in enumerate(names):
        s = float(model.dh_dlogn(arch.A[k], np.array([q90]))[0])
        conf[nm] = s
        print(f"  {nm:<20}{s:>+8.4f}   {'confining' if s < 0 else 'OUTWARD'}")
    man["density_slope_at_q90"] = conf
    man["confining"] = bool(all(v < 0 for v in conf.values()))
    if bound is not None:
        print(f"\n=== OSDR density correction (power {bound.power}; c_k from "
              f"{'the target held at its max' if bound.target_at_max else 'every type sampled'}) ===")
        print(f"  {'archetype':<20}{'x_max (q95)':>12}{'pi cap':>9}{'max net rate':>14}{'c_k':>12}"
              f"{'corr at x_max':>15}")
        for k, nm in enumerate(names):
            print(f"  {nm:<20}{bound.x_max[k]:>12.2f}{bound.pi_cap[k]:>9.4f}"
                  f"{bound.max_flux[k]:>14.4f}{bound.c[k]:>12.3e}"
                  f"{float(bound.correction(bound.x_max[None])[0, k]):>15.4f}")
        man["osdr_bound"] = {"power": bound.power, "target_at_max": bound.target_at_max,
                             "floor": bound.floor,
                             **{f: {n: float(v) for n, v in zip(names, getattr(bound, f))}
                                for f in ("c", "x_max", "pi_cap", "max_flux")}}

    # ---- spec 6 + 7: slices ------------------------------------------------
    rows, mask_frac = [], {}
    for qref in (25, 50, 75):
        ref_all = np.percentile(Xo, qref, axis=0)
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 11.5))
        for ax, (na, nb) in zip(axes.ravel(), PAIRS):
            ka, kb = names.index(na), names.index(nb)
            both = (Xo[:, ka] > 0) & (Xo[:, kb] > 0)
            obs = Xo[both][:, [ka, kb]]
            lo = np.array([np.percentile(obs[:, 0], 1), np.percentile(obs[:, 1], 1)])
            hi = np.array([np.percentile(Xo[:, ka], 95), np.percentile(Xo[:, kb], 95)])
            ga = np.logspace(*np.log10([lo[0], hi[0]]), NGRID)
            gb = np.logspace(*np.log10([lo[1], hi[1]]), NGRID)
            GA, GB = np.meshgrid(ga, gb)
            ref = ref_all.copy()
            held = [m for m in range(K) if m not in (ka, kb)]
            tol = np.array([0.5 * np.subtract(*np.percentile(Xo[:, m], [75, 25]))
                            for m in held])
            held_ok = np.all(np.abs(Xo[both][:, held] - ref[held]) < np.abs(tol), axis=1)
            M, n_held = support_mask(GA, GB, obs, held_ok, ka, kb, Xo)
            mask_frac[f"{na}|{nb}|q{qref}"] = float(1 - M.mean())
            print(f"  q{qref} {na} x {nb}: {n_held:,} cells in the held-coordinate "
                  f"band, {100 * (1 - M.mean()):.0f}% of the grid masked")

            X = np.repeat(ref[None], GA.size, axis=0)
            X[:, ka], X[:, kb] = GA.ravel(), GB.ravel()
            R, Pi = net_rate(X, model, arch, ox, bound)
            _F = X * R
            U = (R[:, ka] / np.log(2)).reshape(GA.shape)
            V = (R[:, kb] / np.log(2)).reshape(GA.shape)
            La, Lb = np.log2(ga), np.log2(gb)
            ax.pcolormesh(La, Lb, np.where(M, np.nan, 1.0), cmap="Greys",
                          vmin=0, vmax=3, shading="auto", zorder=0, alpha=.55)
            ax.plot(np.log2(obs[:4000, 0]), np.log2(obs[:4000, 1]), ".", ms=1.4,
                    color="C2", alpha=.28, zorder=1)
            ax.set_xlim(La[0], La[-1]); ax.set_ylim(Lb[0], Lb[-1])
            Um, Vm = np.where(M, U, np.nan), np.where(M, V, np.nan)
            ax.streamplot(La, Lb, Um, Vm, density=1.2, color="0.35",
                          linewidth=.75, arrowsize=.9, zorder=2)
            ax.contour(La, Lb, Um, [0], colors="C0", linewidths=2.2, zorder=3)
            ax.contour(La, Lb, Vm, [0], colors="C3", linewidths=2.2, zorder=3)

            if qref == 50:
                pd.DataFrame({f"X_{na}": GA.ravel(), f"X_{nb}": GB.ravel(),
                              "n": X.sum(1), f"pi_{na}": Pi[:, ka],
                              f"pi_{nb}": Pi[:, kb], f"F_{na}": _F[:, ka],
                              f"F_{nb}": _F[:, kb], "in_mask": M.ravel()}).to_parquet(
                    f"{OUT}/grid_velocities_{na}_{nb}_q50.parquet".replace(" ", "-"))
            fps, faces = find_fixed_points(ref, ka, kb, model, arch, ox, lo, hi,
                                           np.stack([U, V]), GA, GB, M, bound)
            for xf in fps:
                Z = ref.copy(); Z[ka], Z[kb] = xf
                cls, ev_r, ev_f = classify(Z, model, arch, ox, ka, kb, bound)
                gi = (np.abs(ga - xf[0]).argmin(), np.abs(gb - xf[1]).argmin())
                inside = bool(M[gi[1], gi[0]])
                hidden = cls == "stable" and bool(np.any(ev_f.real > 1e-12))
                ax.plot(np.log2(xf[0]), np.log2(xf[1]),
                        {"stable": "o", "saddle": "s", "unstable": "^"}[cls],
                        ms=11, mfc="k" if cls == "stable" else "white", mec="k",
                        mew=1.8, alpha=1.0 if inside else 0.35, zorder=6)
                Zc = ref.copy(); Zc[ka], Zc[kb] = xf
                rows.append(dict(q_ref=qref, pair=f"{na} x {nb}", X_a=xf[0], X_b=xf[1],
                                 n=float(Zc.sum()), wbar_a=float(xf[0] / Zc.sum()),
                                 classification=cls, in_support=inside,
                                 stable_in_slice_unstable_held=hidden,
                                 max_re_reduced=float(ev_r.real.max()),
                                 max_re_full=float(ev_f.real.max())))
            for idx, v_, free in faces:
                rows.append(dict(q_ref=qref, pair=f"{na} x {nb}",
                                 X_a=v_ if free[0] else 0.0, X_b=0.0 if free[0] else v_,
                                 n=np.nan, wbar_a=np.nan, classification="boundary(face)",
                                 in_support=False, stable_in_slice_unstable_held=False,
                                 max_re_reduced=np.nan, max_re_full=np.nan))
            ax.set_xlabel(f"log2  {na} density"); ax.set_ylabel(f"log2  {nb} density")
            ax.set_title(f"{na}  x  {nb}   ({len(fps)} interior fixed point"
                         f"{'' if len(fps) == 1 else 's'})", fontsize=11)
        fig.suptitle(f"Phase portraits, held coordinates at q{qref}   "
                     "blue = x-nullcline, red = y-nullcline;  filled circle stable, "
                     "square saddle, triangle unstable;  grey = outside support  "
                     + flag, fontsize=9.5)
        fig.tight_layout(rect=[0, 0, 1, .97])
        fig.savefig(f"{OUT}/portrait_q{qref}.png", dpi=140)
        plt.close(fig)
        print(f"  wrote {OUT_F}/portrait_q{qref}.png")

    # ---- spec 7 secondary output: all K-choose-2 pairs at q50 -------------
    ref = np.percentile(Xo, 50, axis=0)
    combos = [(i, j) for i in range(K) for j in range(i + 1, K)]
    fig, axes = plt.subplots(6, 6, figsize=(22, 21))
    for ax, (ka, kb) in zip(axes.ravel(), combos):
        both = (Xo[:, ka] > 0) & (Xo[:, kb] > 0)
        obs = Xo[both][:, [ka, kb]]
        ga = np.logspace(np.log10(np.percentile(obs[:, 0], 1)),
                         np.log10(np.percentile(Xo[:, ka], 95)), 70)
        gb = np.logspace(np.log10(np.percentile(obs[:, 1], 1)),
                         np.log10(np.percentile(Xo[:, kb], 95)), 70)
        GA, GB = np.meshgrid(ga, gb)
        X = np.repeat(ref[None], GA.size, axis=0)
        X[:, ka], X[:, kb] = GA.ravel(), GB.ravel()
        R, _Pi = net_rate(X, model, arch, ox, bound)
        U = (R[:, ka] / np.log(2)).reshape(GA.shape)
        V = (R[:, kb] / np.log(2)).reshape(GA.shape)
        La, Lb = np.log2(ga), np.log2(gb)
        ax.plot(np.log2(obs[:1500, 0]), np.log2(obs[:1500, 1]), ".", ms=1,
                color="C2", alpha=.2)
        ax.streamplot(La, Lb, U, V, density=.85, color="0.4", linewidth=.6, arrowsize=.7)
        ax.contour(La, Lb, U, [0], colors="C0", linewidths=1.6)
        ax.contour(La, Lb, V, [0], colors="C3", linewidths=1.6)
        ax.set_xlim(La[0], La[-1]); ax.set_ylim(Lb[0], Lb[-1])
        ax.set_title(f"{names[ka]} x {names[kb]}", fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(combos):]:
        ax.axis("off")
    fig.suptitle("All 36 archetype pairs, held coordinates at q50, unmasked "
                 "(secondary output; blue = x-nullcline, red = y-nullcline)  " + flag, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .98])
    fig.savefig(f"{OUT}/portrait_contact_sheet.png", dpi=110)
    plt.close(fig)
    print(f"wrote {OUT_F}/portrait_contact_sheet.png ({len(combos)} pairs)")

    fp = pd.DataFrame(rows)
    fp.to_csv(f"{OUT}/fixed_points.csv", index=False)
    man["masked_fraction"] = mask_frac
    man["n_fixed_points"] = int(np.sum(fp["classification"] != "boundary(face)")) if len(fp) else 0
    json.dump(man, open(f"{OUT}/portrait_manifest.json", "w"), indent=2)
    print("\n=== fixed points ===")
    if len(fp):
        print(fp.round(4).to_string(index=False))
    else:
        print("  none found")
    print(f"\nmasked fraction, median over panels: "
          f"{np.median(list(mask_frac.values())):.2f}")
    print(f"wrote {OUT}/{{Atilde.csv,fixed_points.csv,portrait_manifest.json}} "
          f"and {OUT_F}/Atilde_heatmap.png")


if __name__ == "__main__":
    main()

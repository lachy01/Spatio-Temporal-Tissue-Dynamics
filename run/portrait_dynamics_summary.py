"""Numbers behind outputs/portrait/FINDINGS.md that the portrait scripts do not print.

    python run/portrait_dynamics_summary.py                    # unbounded (the default)
    python run/portrait_dynamics_summary.py --bound osdr [--power 8] [--target-at-max]

From the production point estimate, as run/portrait_preliminary.py builds it:

1. density thresholds per archetype: total density swept at each archetype's
   own mean neighbourhood composition (the 80 um disc in own-state
   coordinates, averaged over that archetype's cells), and every crossing of
   pi_k = pi_bar_k classified as a stable ceiling or an unstable threshold;
2. whether the nullclines are total-density contours: n along each zero
   contour of the priority-pair slices at the q50 reference;
3. the angle between the two nullcline gradients at the 80th percentile
   (0 degrees = degenerate);
4. the largest entries of Atilde = A B A^T, and the loadings' sign structure.

Writes outputs/portrait/dynamics_summary.txt, or with `--bound osdr` (OSDR's
maximal-density correction, a flagged sensitivity variant, spec 5) the same file
in outputs/portrait/osdr_p<power>[_target_at_max]/.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import matplotlib                                                    # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
import jax.numpy as jnp                                              # noqa: E402

from tissue_dynamics.mixture.basis import build_basis, spec_from_basis    # noqa: E402
from tissue_dynamics.mixture.data import load_config                      # noqa: E402
from tissue_dynamics.mixture.model import build_data, responsibilities    # noqa: E402
from tissue_dynamics.portrait.core import (PortraitModel, build_archetypes,  # noqa: E402
                                           fit_osdr_bound, interaction_matrix, net_rate,
                                           simplex_loadings)

OUT = "outputs/portrait"
PAIRS = [("CAFs", "Myeloid"), ("Cancer Epithelial", "CAFs"),
         ("Cancer Epithelial", "Myeloid"), ("Cancer Epithelial", "T-cells")]


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bound", choices=("none", "osdr"), default="none")
    ap.add_argument("--power", type=int, default=8)
    ap.add_argument("--target-at-max", action="store_true")
    a = ap.parse_args()
    if a.bound == "osdr":
        OUT = f"outputs/portrait/osdr_p{a.power}{'_target_at_max' if a.target_at_max else ''}"
    os.makedirs(OUT, exist_ok=True)
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
    nm, Xo = arch.names, arch.X_obs
    ox = float(np.median(df["oxygen"]))
    dens = df["density"].to_numpy(float)
    arche = df["archetype"].astype(str).to_numpy()
    nbr_own = basis.disc_in_own(df)
    bound = (None if a.bound == "none" else
             fit_osdr_bound(model, arch, ox, cell_archetype=arche, power=a.power,
                            target_at_max=a.target_at_max))
    out = []
    say = out.append

    say(f"PORTRAIT DYNAMICS SUMMARY  ({len(df):,} cells; technical: "
        f"{cfg['model'].get('technical', 'area')}, lambda {pe['weight_ridge']:g}, rank "
        f"{cfg['model']['rank']}; spec 3.2 signed anchor)"
        + ("" if bound is None else
           f"\nSENSITIVITY VARIANT: OSDR maximal-density correction, power {bound.power}"
           f"{', target held at its max' if bound.target_at_max else ''}") + "\n")
    say(f"Loadings: {100 * np.mean((W < 0) | (W > 1)):.1f}% of entries outside [0, 1]; "
        f"{100 * np.mean((W > 0).all(1)):.1f}% of cells have every X_k > 0.\n")

    say("1. Density thresholds per archetype, swept at that archetype's own mean")
    say("   neighbourhood composition (stable = ceiling, UNSTABLE = escape threshold).\n")
    grid = np.linspace(np.percentile(dens, .5), np.percentile(dens, 99.9), 6000)
    for k, x in enumerate(nm):
        pb = nbr_own[arche == x].mean(0)
        if bound is None:
            e = model.eta(np.repeat(arch.A[k][None], len(grid), 0),
                          np.repeat(pb[None], len(grid), 0), grid, ox)
            f = 1 / (1 + np.exp(-e)) - arch.pi_bar[k]
        else:      # the correction acts on the type's own density, X_k = n w_k
            w = simplex_loadings(pb[None], arch.A)[0]
            f = net_rate(grid[:, None] * w[None, :], model, arch, ox, bound)[0][:, k]
        s = np.where(np.diff(np.sign(f)))[0]
        if not len(s):
            say(f"   {x:<20} none (always {'grows' if f[0] > 0 else 'shrinks'})")
            continue
        seg = [f"n={grid[i]:.0f} [pct {100 * np.mean(dens < grid[i]):.0f}] "
               + ("stable (grow->shrink)" if f[i] > 0 else "UNSTABLE (shrink->grow)") for i in s]
        say(f"   {x:<20} " + ";  ".join(seg))

    ref = np.percentile(Xo, 50, axis=0)
    say("\n2. Are the nullclines total-density contours?  (n along the zero contour, q50)")
    say(f"   {'pair':<32} {'nullcline':<20} {'mean n':>8} {'sd n':>8} {'cv':>7}")
    for na, nb in PAIRS:
        ka, kb = nm.index(na), nm.index(nb)
        both = (Xo[:, ka] > 0) & (Xo[:, kb] > 0)
        obs = Xo[both][:, [ka, kb]]
        ga = np.logspace(*np.log10([np.percentile(obs[:, 0], 1), np.percentile(Xo[:, ka], 95)]), 200)
        gb = np.logspace(*np.log10([np.percentile(obs[:, 1], 1), np.percentile(Xo[:, kb], 95)]), 200)
        GA, GB = np.meshgrid(ga, gb)
        X = np.repeat(ref[None], GA.size, 0)
        X[:, ka], X[:, kb] = GA.ravel(), GB.ravel()
        R, _Pi = net_rate(X, model, arch, ox, bound)
        for lbl, k in ((na, ka), (nb, kb)):
            Z = R[:, k].reshape(GA.shape)
            segs = [g for g in plt.contour(np.log2(ga), np.log2(gb), Z, [0]).allsegs[0] if len(g)]
            plt.close("all")
            if not segs:
                side = "grows" if np.median(Z) > 0 else "shrinks"
                say(f"   {na + ' x ' + nb:<32} {lbl:<20} no contour in the window ({side} throughout)")
                continue
            pts = np.vstack(segs)
            Zc = np.repeat(ref[None], len(pts), 0)
            Zc[:, ka], Zc[:, kb] = 2.0 ** pts[:, 0], 2.0 ** pts[:, 1]
            n = Zc.sum(1)
            say(f"   {na + ' x ' + nb:<32} {lbl:<20} {n.mean():8.1f} {n.std():8.2f} "
                f"{n.std() / n.mean():7.4f}")

    say("\n3. Angle between the two nullcline gradients at the 80th pct (0 deg = degenerate)")
    for na, nb in PAIRS:
        ka, kb = nm.index(na), nm.index(nb)
        Z = ref.copy()
        Z[ka], Z[kb] = np.percentile(Xo[:, ka], 80), np.percentile(Xo[:, kb], 80)
        g = []
        for k in (ka, kb):
            gr = np.zeros(2)
            for t, j in enumerate((ka, kb)):
                e = 1e-4 * Z[j]
                Zp, Zm = Z.copy(), Z.copy()
                Zp[j] += e
                Zm[j] -= e
                gr[t] = (net_rate(Zp[None], model, arch, ox, bound)[0][0, k]
                         - net_rate(Zm[None], model, arch, ox, bound)[0][0, k]) / (2 * e)
            g.append(gr / np.linalg.norm(gr))
        ang = np.degrees(np.arccos(np.clip(abs(g[0] @ g[1]), 0, 1)))
        say(f"   {na + ' x ' + nb:<32} {ang:6.2f} deg")

    At = interaction_matrix(model, arch)
    sym, anti = (At + At.T) / 2, (At - At.T) / 2
    say(f"\n4. Atilde = A B A^T (row <- column: effect of the column archetype's share on the "
        f"row's log-odds)\n   symmetric / antisymmetric Frobenius norms "
        f"{np.linalg.norm(sym):.3f} / {np.linalg.norm(anti):.3f}; largest entries:")
    flat = sorted(((At[i, j], nm[i], nm[j]) for i in range(len(nm)) for j in range(len(nm))),
                  key=lambda t: -abs(t[0]))
    for v, r, c in flat[:10]:
        say(f"   {r + ' <- ' + c:<44} {v:+.3f}")
    say("   diagonal (self-terms): " + ", ".join(f"{nm[i]} {At[i, i]:+.3f}" for i in range(len(nm))))

    text = "\n".join(out) + "\n"
    open(f"{OUT}/dynamics_summary.txt", "w").write(text)
    print(text)


if __name__ == "__main__":
    main()

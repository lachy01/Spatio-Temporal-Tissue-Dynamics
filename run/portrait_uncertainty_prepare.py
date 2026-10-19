"""Block-cluster sandwich covariance of the production mixture, for the portrait's
parametric bootstrap (portrait spec 8.1; job `portrait_param_boot`).

    python run/portrait_uncertainty_prepare.py

V = H^-1 M H^-1 at the point estimate (outputs/hpc/point_estimate.json):
  H  the Hessian of the penalised negative log-likelihood (`model.neg_loglik`),
     by Hessian-vector products; inverted on its eigenvalues above 1e-8 of the
     largest (the U/V rotation is exactly flat, as in nb_baseline.covariance);
  M  the outer product of the estimating functions summed within each spatial
     block -- the `spatial_blocks` partition the bootstrap and CV use -- with
     each block's share of the penalty gradient removed, times G / (G - 1).
The spec names a Conley HAC covariance; this is its block form, and the analytic
counterpart of the spatial block bootstrap. Also reported: the same sandwich with
every cell its own cluster, so the widening due to spatial dependence can be read.

Writes outputs/portrait/uncertainty/param_cov.npz (theta, V's eigenvalues and
eigenvectors, the parameter layout) and se.csv.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import jax                                                            # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                               # noqa: E402
from jax.flatten_util import ravel_pytree                             # noqa: E402

from tissue_dynamics.mixture.basis import spec_from_basis             # noqa: E402
from tissue_dynamics.mixture.data import load_config                  # noqa: E402
from tissue_dynamics.mixture.inference import spatial_blocks          # noqa: E402
from tissue_dynamics.mixture.model import build_data, loglik_terms, neg_loglik  # noqa: E402
from tissue_dynamics.mixture.tasks import _load                       # noqa: E402

OUT = "outputs/portrait/uncertainty"


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    cfg = load_config()
    df, basis = _load(cfg)
    pe = json.load(open("outputs/hpc/point_estimate.json"))
    spec = spec_from_basis(basis, cfg, stage="B", weight_ridge=float(pe["weight_ridge"]))
    D = build_data(df, basis, cfg, spec).as_jax()
    P = {k: jnp.asarray(np.asarray(v)) for k, v in pe["params"].items()}
    x, unravel = ravel_pytree(P)
    d, n = x.shape[0], len(df)
    print(f"{n:,} cells, {d} parameters", flush=True)

    obj = lambda z: neg_loglik(unravel(z), D, spec)                   # noqa: E731
    ll_cells = lambda z: loglik_terms(unravel(z), D, spec)            # noqa: E731
    g = jax.grad(obj)
    print(f"gradient inf-norm at the estimate: {float(jnp.max(jnp.abs(g(x)))):.2e}", flush=True)
    hv = jax.jit(lambda z, V: jax.vmap(lambda v: jax.jvp(g, (z,), (v,))[1])(V))
    E = np.eye(d)
    H = np.vstack([np.asarray(hv(x, jnp.asarray(E[i:i + 4]))) for i in range(0, d, 4)])
    H = 0.5 * (H + H.T)
    w, Q = np.linalg.eigh(H)
    keep = w > 1e-8 * w.max()
    C = (Q[:, keep] / w[keep]) @ Q[:, keep].T
    print(f"Hessian: {int((~keep).sum())} flat directions dropped; min eigenvalue {w.min():.3e} "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    g_pen = np.asarray(g(x)) + np.asarray(jax.grad(lambda z: jnp.sum(ll_cells(z)))(x))
    labels = spatial_blocks(df[["x", "y"]].to_numpy(float), int(cfg["inference"]["n_blocks_target"]))
    _, b = np.unique(labels, return_inverse=True)
    G = int(b.max()) + 1
    jv = jax.jit(lambda z, v: jax.jvp(ll_cells, (z,), (v,))[1])
    S = np.zeros((G, d))
    M_cell = np.zeros((d, d))
    Gc = np.empty((n, d), np.float32)
    for k in range(d):
        Gc[:, k] = np.asarray(jv(x, jnp.asarray(E[k]))) - g_pen[k] / n
    np.add.at(S, b, Gc)
    for s in range(0, n, 100_000):
        gg = Gc[s:s + 100_000].astype(np.float64)
        M_cell += gg.T @ gg
    del Gc
    M = (S.T @ S) * G / (G - 1)
    V, V_cell = C @ M @ C, C @ (M_cell * n / (n - 1)) @ C
    V = 0.5 * (V + V.T)
    ev, evec = np.linalg.eigh(V)
    ev = np.clip(ev, 0, None)
    print(f"sandwich done ({(time.time() - t0) / 60:.1f} min); {G} blocks", flush=True)

    idx = unravel(jnp.arange(d, dtype=jnp.float64))
    rows = []
    for name, ii in idx.items():
        sel = np.rint(np.asarray(ii)).astype(int).ravel()
        se = np.sqrt(np.diag(V)[sel])
        se_c = np.sqrt(np.clip(np.diag(V_cell)[sel], 1e-300, None))
        rows.append({"block": name, "size": len(sel), "median SE (block-cluster)": float(np.median(se)),
                     "median SE (cell-level)": float(np.median(se_c)),
                     "median inflation": float(np.median(se / se_c))})
    tab = pd.DataFrame(rows)
    # Check against the 300-replicate spatial block bootstrap (`bootstrap` job), for
    # the blocks it kept: the sandwich is its analytic counterpart.
    boot = [json.load(open(f)) for f in sorted(glob.glob("outputs/hpc/bootstrap/*.json"))]
    if boot:
        se_all = np.sqrt(np.diag(V))
        ratio = []
        for name, ii in idx.items():
            key = "e_delta" if name == "delta" else name
            if key not in boot[0]:
                ratio.append(np.nan)
                continue
            bs = np.array([np.log(r[key]) if key == "e_delta" else np.ravel(r[key]) for r in boot], float)
            sel = np.rint(np.asarray(ii)).astype(int).ravel()
            ratio.append(float(np.median(se_all[sel] / bs.reshape(len(boot), -1).std(0, ddof=1))))
        tab[f"median SE / bootstrap SD ({len(boot)} reps)"] = ratio
    tab.to_csv(f"{OUT}/se.csv", index=False)
    print(tab.round(4).to_string(index=False))
    np.savez(f"{OUT}/param_cov.npz", theta=np.asarray(x), evals=ev, evecs=evec,
             layout=json.dumps({k: list(np.shape(v)) for k, v in P.items()}))
    print(f"wrote {OUT}/param_cov.npz ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()

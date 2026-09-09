"""
Soft-label variant of the tissue oxygen solver.

Instead of pinning u = 1 on thresholded vessel pixels (Dirichlet), every pixel
carries a Robin exchange term whose strength is the deposited endothelial score:

    (n_valid + h^2 kappa_ij + beta * Phi_ij) u_ij  -  sum_{nb} u_nb  =  beta * Phi_ij

where Phi_ij is the SUM of phi_endo over cells in that pixel (not a per-area
density, so beta is dimensionless and grid-independent).

Physically this is a mass-transfer boundary: flux into the tissue is proportional
to beta * phi_endo * (1 - u), i.e. it stops when tissue reaches blood saturation.
beta -> infinity recovers Dirichlet u=1; beta -> 0 recovers no source.

There are no Dirichlet pixels at all, so the matrix stays symmetric positive
definite and 0 <= u <= 1 is guaranteed by the discrete maximum principle.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg
from scipy.special import k0


def assemble_robin(tissue, Phi, kappa, h, beta):
    """Build the SPD system. Phi is the per-pixel SUM of phi_endo."""
    H, W = tissue.shape
    idx = -np.ones((H, W), dtype=np.int64)
    idx[tissue] = np.arange(tissue.sum())
    n = int(tissue.sum())

    rows, cols, vals = [], [], []
    n_valid = np.zeros((H, W), dtype=np.int64)

    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb_t = np.roll(tissue, (-di, -dj), axis=(0, 1))
        nb_i = np.roll(idx, (-di, -dj), axis=(0, 1))
        sl = [slice(None), slice(None)]
        if di == 1:   sl[0] = slice(H - 1, H)
        elif di == -1: sl[0] = slice(0, 1)
        elif dj == 1:  sl[1] = slice(W - 1, W)
        else:          sl[1] = slice(0, 1)
        nb_t[tuple(sl)] = False

        n_valid += (tissue & nb_t)
        m = tissue & nb_t
        rows.append(idx[m]); cols.append(nb_i[m]); vals.append(-np.ones(int(m.sum())))

    diag = n_valid[tissue] + (h ** 2) * kappa[tissue] + beta * Phi[tissue]
    b = beta * Phi[tissue]

    if np.any(diag <= 0):
        raise ValueError("singular row: fragment with no consumption and no source")

    rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(diag.astype(float))
    A = sp.csr_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return A, b, idx


def solve_soft(tissue, Phi, kappa, h, beta, tol=1e-11):
    A, b, idx = assemble_robin(tissue, Phi, kappa, h, beta)
    try:
        import pyamg
        x = pyamg.smoothed_aggregation_solver(A).solve(b, tol=tol)
    except ImportError:
        M = sp.diags(1.0 / A.diagonal())
        x, _ = cg(A, b, rtol=tol, maxiter=50000, M=M)
    u = np.full(tissue.shape, np.nan)
    u[tissue] = x
    return u


# ----------------------------------------------------------------------

def _greens_function_test():
    """A single weak point source must reproduce the K_0 kernel.

    Solution of  lap(u) - u/L^2 = -s  is  u = s * K_0(r/L) / (2 pi),
    so the PDE solve and the FFT convolution are the same object on an
    infinite homogeneous domain.
    """
    L, h, half = 100.0, 2.0, 700.0
    N = int(2 * half / h) + 1; c = (N - 1) // 2
    ax = (np.arange(N) - c) * h
    X, Y = np.meshgrid(ax, ax, indexing="ij"); r = np.hypot(X, Y)

    tissue = np.ones((N, N), bool)
    Phi = np.zeros((N, N)); Phi[c, c] = 1.0
    kappa = np.full((N, N), 1.0 / L ** 2)

    u = solve_soft(tissue, Phi, kappa, h, beta=1e-4)   # weak source -> linear regime
    band = (r > 8 * h) & (r < 450)
    shape = u[band] / k0(r[band] / L)
    print(f"Green's function: u/K_0(r/L) over 16-450 um "
          f"-> mean {shape.mean():.4e}, CV {shape.std()/shape.mean():.2e}")
    print("  (constant ratio => the solved field IS the K_0 kernel)")


def _beta_limit_test():
    """beta -> infinity must recover the Dirichlet disc solution."""
    from oxygen_field import solve_field
    L, rho, h, half = 100.0, 25.0, 2.5, 600.0
    N = int(2 * half / h) + 1; c = (N - 1) // 2
    ax = (np.arange(N) - c) * h
    X, Y = np.meshgrid(ax, ax, indexing="ij"); r = np.hypot(X, Y)
    tissue = np.ones((N, N), bool); disc = r <= rho
    kappa = np.full((N, N), 1.0 / L ** 2)

    ref = solve_field(tissue, disc, kappa, h, verbose=False)
    Phi = disc.astype(float)
    print("\nbeta limit (max |u_robin - u_dirichlet| outside the disc):")
    band = r > rho + 3 * h
    for beta in (1e-1, 1e0, 1e1, 1e3, 1e5):
        u = solve_soft(tissue, Phi, kappa, h, beta)
        print(f"  beta = {beta:>8.0e}   {np.abs(u[band] - ref[band]).max():.2e}"
              f"   max u = {np.nanmax(u):.4f}")


def _background_test():
    """The noise floor across many non-endothelial cells can swamp real vessels."""
    rng = np.random.default_rng(1)
    L, h, N = 100.0, 5.0, 300
    tissue = np.ones((N, N), bool)
    kappa = np.full((N, N), 1.0 / L ** 2)

    n_cells = 20000
    xy = rng.uniform(0, (N - 1) * h, size=(n_cells, 2))
    is_endo = rng.random(n_cells) < 0.02
    phi = np.where(is_endo, rng.normal(0.80, 0.10, n_cells),
                            np.abs(rng.normal(0.00, 0.05, n_cells)))
    phi = np.clip(phi, 0, 1)

    print(f"\nbackground swamping ({n_cells} cells, {is_endo.mean():.0%} endothelial):")
    print(f"  total source from true endothelium : {phi[is_endo].sum():8.1f}")
    print(f"  total source from the noise floor  : {phi[~is_endo].sum():8.1f}")

    ij = np.floor(xy / h).astype(int)
    for name, w in [("raw phi", phi),
                    ("soft-threshold max(phi-0.15,0)", np.maximum(phi - 0.15, 0)),
                    ("power phi^3", phi ** 3)]:
        frac = w[is_endo].sum() / max(w.sum(), 1e-12)
        Phi = np.zeros((N, N))
        np.add.at(Phi, (ij[:, 0], ij[:, 1]), w)
        u = solve_soft(tissue, Phi, kappa, h, beta=1.0)
        # does the field track vessels, or just cell density?
        dens = np.zeros((N, N)); np.add.at(dens, (ij[:, 0], ij[:, 1]), 1.0)
        u_dens = solve_soft(tissue, dens, kappa, h, beta=1.0)
        rho_d = np.corrcoef(u.ravel(), u_dens.ravel())[0, 1]
        print(f"  {name:<32} signal share {frac:5.1%}   corr(u, u_density) {rho_d:.3f}")


if __name__ == "__main__":
    _greens_function_test()
    _beta_limit_test()
    _background_test()

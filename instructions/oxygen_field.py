"""
Steady-state oxygen (or any screened-diffusion) field on an irregular tissue domain.

Solves     div( grad u ) = kappa(x) * u
with       u = 1            on vessel pixels          (Dirichlet source)
           du/dn = 0        on the tissue boundary    (no-flux)
           kappa = 0        where there are no live consumers (necrosis)

u is dimensionless relative oxygenation in [0, 1]. kappa(x) = 1 / L(x)^2, where
L = sqrt(D / (n k)) is the screening length at local cell density n.

Discretisation: five-point Laplacian on a uniform grid. The resulting matrix is
sparse, symmetric positive definite and diagonally dominant, so CG or AMG applies.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg
from scipy.special import k0


# ----------------------------------------------------------------------
# assembly
# ----------------------------------------------------------------------

def assemble(tissue, vessel, kappa, h):
    """Build the SPD system for the free (tissue, non-vessel) pixels.

    tissue : (H, W) bool   -- True inside the tissue section
    vessel : (H, W) bool   -- True on Dirichlet source pixels (subset of tissue)
    kappa  : (H, W) float  -- consumption 1/L^2 in um^-2; 0 in necrosis
    h      : grid pitch in um

    For each free pixel the stencil is

        (n_valid + h^2 kappa) u_ij  -  sum_{free nb} u_nb  =  sum_{vessel nb} 1

    where n_valid counts neighbours INSIDE the tissue (free or vessel). Neighbours
    outside the tissue are dropped from both sides -- that single omission is the
    entire no-flux boundary condition (the mirrored ghost node cancels against the
    diagonal). Vessel neighbours count toward n_valid but move to the RHS, which is
    what keeps the matrix symmetric.
    """
    free = tissue & ~vessel
    H, W = tissue.shape

    idx = -np.ones((H, W), dtype=np.int64)
    idx[free] = np.arange(free.sum())
    n = int(free.sum())

    rows, cols, vals = [], [], []
    b = np.zeros(n)
    n_valid = np.zeros((H, W), dtype=np.int64)

    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb_tissue = np.roll(tissue, (-di, -dj), axis=(0, 1))
        nb_vessel = np.roll(vessel, (-di, -dj), axis=(0, 1))
        nb_idx = np.roll(idx, (-di, -dj), axis=(0, 1))

        # rolling wraps around; blank out the wrapped edge so it reads as "outside"
        sl = [slice(None), slice(None)]
        if di == 1:
            sl[0] = slice(H - 1, H)
        elif di == -1:
            sl[0] = slice(0, 1)
        elif dj == 1:
            sl[1] = slice(W - 1, W)
        else:
            sl[1] = slice(0, 1)
        nb_tissue[tuple(sl)] = False
        nb_vessel[tuple(sl)] = False

        n_valid += (free & nb_tissue)

        # free -> free coupling: symmetric off-diagonal -1
        m = free & nb_tissue & ~nb_vessel
        rows.append(idx[m])
        cols.append(nb_idx[m])
        vals.append(-np.ones(int(m.sum())))

        # free -> vessel coupling: known value 1 moves to the RHS
        m = free & nb_vessel
        np.add.at(b, idx[m], 1.0)

    diag = n_valid[free] + (h ** 2) * kappa[free]
    if np.any(diag <= 0):
        raise ValueError(
            "singular row: isolated pixel with no in-tissue neighbours and no "
            "consumption. Label connected components and drop vessel-free, "
            "consumption-free fragments."
        )

    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag.astype(float))

    A = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )
    return A, b, idx, free


def solve_field(tissue, vessel, kappa, h, tol=1e-10, verbose=True):
    """Assemble and solve; returns the (H, W) field with NaN outside the tissue."""
    A, b, idx, free = assemble(tissue, vessel, kappa, h)

    try:
        import pyamg
        ml = pyamg.smoothed_aggregation_solver(A)
        x = ml.solve(b, tol=tol)
        how = "pyamg"
    except ImportError:
        M = sp.diags(1.0 / A.diagonal())          # Jacobi preconditioner
        x, info = cg(A, b, rtol=tol, maxiter=20000, M=M)
        how = f"cg (info={info})"

    if verbose:
        res = np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-30)
        print(f"  solved {A.shape[0]:,} unknowns via {how}, rel residual {res:.2e}")

    u = np.full(tissue.shape, np.nan)
    u[free] = x
    u[vessel] = 1.0
    return u


def sample_at_cells(u, xy, origin, h):
    """Bilinear interpolation of the field at cell centroids, plus |grad u|."""
    gy, gx = np.gradient(np.nan_to_num(u), h)
    gmag = np.hypot(gy, gx)

    p = (xy - origin) / h
    i0 = np.floor(p[:, 0]).astype(int)
    j0 = np.floor(p[:, 1]).astype(int)
    fi = p[:, 0] - i0
    fj = p[:, 1] - j0

    def interp(field):
        return (
            field[i0, j0] * (1 - fi) * (1 - fj)
            + field[i0 + 1, j0] * fi * (1 - fj)
            + field[i0, j0 + 1] * (1 - fi) * fj
            + field[i0 + 1, j0 + 1] * fi * fj
        )

    return interp(np.nan_to_num(u)), interp(gmag)


# ----------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------

def _analytic_test():
    """Single circular vessel, homogeneous consumption.

    Exact solution outside a Dirichlet disc of radius rho in an infinite plane:
        u(r) = K_0(r/L) / K_0(rho/L)
    """
    L, rho, h, half = 100.0, 25.0, 2.5, 600.0
    N = int(2 * half / h) + 1
    c = (N - 1) // 2
    ax = (np.arange(N) - c) * h
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    r = np.hypot(X, Y)

    tissue = np.ones((N, N), dtype=bool)
    vessel = r <= rho
    kappa = np.full((N, N), 1.0 / L ** 2)

    print(f"analytic check: L={L}, rho={rho}, h={h}, grid {N}x{N}")
    u = solve_field(tissue, vessel, kappa, h)

    band = (r > rho + 2 * h) & (r < 500)
    exact = k0(r[band] / L) / k0(rho / L)
    got = u[band]
    err = np.abs(got - exact)
    print(f"  max abs err = {err.max():.2e}   max rel err = "
          f"{(err / exact).max():.2e}   (u range {exact.min():.1e}-{exact.max():.2f})")

    print("  grid convergence:")
    for hh in (5.0, 2.5, 1.25):
        M = int(2 * half / hh) + 1
        cc = (M - 1) // 2
        a2 = (np.arange(M) - cc) * hh
        X2, Y2 = np.meshgrid(a2, a2, indexing="ij")
        r2 = np.hypot(X2, Y2)
        u2 = solve_field(np.ones((M, M), bool), r2 <= rho,
                         np.full((M, M), 1.0 / L ** 2), hh, verbose=False)
        bd = (r2 > rho + 2 * hh) & (r2 < 500)
        ex = k0(r2[bd] / L) / k0(rho / L)
        print(f"    h = {hh:>5} um   max rel err = {np.abs(u2[bd] - ex).max() / ex.max():.2e}")


def _structure_test():
    """No-flux boundary and a zero-consumption necrotic patch."""
    L, h, N = 100.0, 5.0, 200
    tissue = np.ones((N, N), dtype=bool)
    tissue[:40, :40] = False                     # a notch out of the section

    vessel = np.zeros((N, N), dtype=bool)
    vessel[100, 60] = True
    vessel[60, 150] = True

    kappa = np.full((N, N), 1.0 / L ** 2)
    necrotic = np.zeros((N, N), dtype=bool)
    necrotic[120:160, 100:140] = True
    kappa[necrotic] = 0.0

    print("\nstructure check: notched domain, 2 vessels, necrotic patch")
    u = solve_field(tissue, vessel, kappa, h)
    ok = np.isfinite(u)
    print(f"  range [{np.nanmin(u):.2e}, {np.nanmax(u):.2f}]  "
          f"-> maximum principle {'holds' if -1e-12 <= np.nanmin(u) and np.nanmax(u) <= 1 + 1e-12 else 'VIOLATED'}")
    print(f"  NaN outside tissue: {'yes' if np.all(np.isnan(u[~tissue])) else 'NO'}")

    # inside a kappa=0 region the field must be harmonic -> no interior extremum
    ni, nj = 140, 120
    nbmean = (u[ni + 1, nj] + u[ni - 1, nj] + u[ni, nj + 1] + u[ni, nj - 1]) / 4
    print(f"  necrotic interior pixel {u[ni, nj]:.6e} vs neighbour mean "
          f"{nbmean:.6e}  (harmonic, should match)")

    # no-flux: normal gradient at the straight outer edge should vanish
    edge = np.abs(u[-1, 50:150] - u[-2, 50:150]).max()
    print(f"  max |du| across the outer boundary = {edge:.2e} (no-flux -> ~0)")


if __name__ == "__main__":
    _analytic_test()
    _structure_test()

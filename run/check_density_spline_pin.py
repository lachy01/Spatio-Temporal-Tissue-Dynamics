"""Which density-spline column a stored fit was made with (basis._Spline, `drop`).

    [MIX_DATASET=r60] python run/check_density_spline_pin.py [--hpc outputs/hpc_r60]

The centred cr basis has one redundant column, and pivoted QR's choice of which to
drop is a floating-point tie between the first and the last column, broken
differently by different LAPACK builds. A stored fit is only readable with the
column it was fitted with. For the dataset MIX_DATASET selects, this prints QR's
own choice on this machine, the configured pin (`frozen_basis` in the config), and
the log-likelihood of the stored point estimate with each candidate column dropped,
against the stored value. The matching row is the one to pin.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import jax                                                            # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                               # noqa: E402
import patsy                                                          # noqa: E402
from scipy.linalg import qr                                           # noqa: E402

from tissue_dynamics.mixture.basis import spec_from_basis             # noqa: E402
from tissue_dynamics.mixture.data import load_config                  # noqa: E402
from tissue_dynamics.mixture.model import build_data, loglik          # noqa: E402
from tissue_dynamics.mixture.tasks import _load                       # noqa: E402


def main():
    ds = os.environ.get("MIX_DATASET")
    ap = argparse.ArgumentParser()
    ap.add_argument("--hpc", default=f"outputs/hpc_{ds}" if ds else "outputs/hpc")
    a = ap.parse_args()
    cfg = load_config()
    df, basis = _load(cfg)
    pe = json.load(open(f"{a.hpc}/point_estimate.json"))
    s = basis.density_weight_spline
    x = np.log(df["density"].to_numpy(float))
    B = np.asarray(patsy.build_design_matrices([s._di], {"x": x})[0])
    Bc = B - s._mean
    _q, r, piv = qr(Bc, mode="economic", pivoting=True)
    pin = (cfg.get("frozen_basis") or {}).get("density_weight_drop_col")
    print(f"dataset {ds or 'breast'}: QR on this machine drops column {piv[-1]}; configured pin {pin}")
    P = {k: jnp.asarray(np.asarray(v)) for k, v in pe["params"].items()}
    for drop in sorted({0, B.shape[1] - 1, int(piv[-1])}):
        s._keep = np.array([j for j in range(B.shape[1]) if j != drop])
        if not np.all(s._sd == 1):                    # scale_weight_splines
            s._sd = Bc[:, s._keep].std(0)
        spec = spec_from_basis(basis, cfg, stage="B", weight_ridge=float(pe["weight_ridge"]))
        D = build_data(df, basis, cfg, spec).as_jax()
        ll = float(loglik(P, D, spec))
        print(f"  drop column {drop}: log-likelihood {ll:,.2f} (stored {pe['loglik']:,.2f})"
              + ("  <- matches" if abs(ll - pe["loglik"]) < 0.01 else ""))


if __name__ == "__main__":
    main()

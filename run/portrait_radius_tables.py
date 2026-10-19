"""Tables for the portrait's neighbourhood-radius sensitivity (portrait spec 9).

    python run/portrait_radius_tables.py [--radii 60 100]

For each radius, the breast table rebuilt with the neighbourhood disc and density
at that radius (config_r<radius>.yaml), on the same cells as production, with the
MisTIC columns copied across (they do not depend on the radius). Writes
data/radius/mixture_cells_r<radius>.parquet and its .phi_names.json sidecar. Then on
Spartan: `MIX_DATASET=r<radius> ./hpc/submit.sh multistart` (10 restarts).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from tissue_dynamics.mixture.data import build_table, load_config, validate   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", type=int, nargs="+", default=[60, 100])
    a = ap.parse_args()
    prod_cfg = load_config()
    prod = pd.read_parquet(prod_cfg["data"]["table_parquet"])
    for r in a.radii:
        os.environ["MIX_DATASET"] = f"r{r}"
        cfg = load_config()
        out = cfg["data"]["table_parquet"]
        print(f"radius {r} um -> {out}", flush=True)
        clean, _rep = validate(build_table(cfg), cfg)
        if not np.array_equal(clean["cell_id"].to_numpy(), prod["cell_id"].to_numpy()):
            sys.exit(f"radius {r}: the rebuilt table's cells differ from production's")
        add = [c for c in prod.columns if c not in clean.columns]
        for c in add:
            clean[c] = prod[c].to_numpy()
        os.makedirs(os.path.dirname(out), exist_ok=True)
        clean.to_parquet(out, index=False)
        json.dump(list(clean.attrs["phi_names"]), open(out.replace(".parquet", ".phi_names.json"), "w"))
        print(f"  {len(clean):,} cells; copied {len(add)} columns ({', '.join(add[:6])}...); "
              f"median density {clean['density'].median():.1f} (production {prod['density'].median():.1f})",
              flush=True)
    os.environ.pop("MIX_DATASET", None)


if __name__ == "__main__":
    main()

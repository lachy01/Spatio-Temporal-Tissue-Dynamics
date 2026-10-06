"""Add the contact / regional neighbourhood columns to the existing table.

    python run/add_neighbourhood_split.py

`state.neighbourhood: split` reads `nbrc_phi_*` (typed neighbours at 0-15 um)
and `nbrr_phi_*` (15-80 um), plus `n_contact` / `n_regional`. A full rebuild
(run/build_table.py) now writes them, but it would also re-solve the oxygen
field and re-read everything; this adds them to the table in place, so every
other column -- and the cell set the CV folds were built on -- stays identical.

The neighbour pool is reproduced exactly as build_table builds it: every cell
with a PhiSpace score and N >= min_library_size, including the typed cells
later dropped for lying outside the oxygen mask. The script checks that
contact and regional means recombine to the table's 80 um disc mean
(`nbr_phi_*`) for every cell with a typed neighbour, and refuses to write if
they do not.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from tissue_dynamics.mixture.data import (add_split_columns, load_config,  # noqa: E402
                                          split_neighbourhood)
from tissue_dynamics.nb_baseline import all_cells                         # noqa: E402


def main():
    cfg = load_config()
    dc = cfg["data"]
    path = dc["table_parquet"]
    table = pd.read_parquet(path)
    names = json.load(open(os.path.splitext(path)[0] + ".phi_names.json"))
    print(f"table: {len(table):,} cells", flush=True)

    cells = all_cells(cfg, genes=[])
    P = cells[[f"phi_{nm}" for nm in names]].to_numpy(float)
    contact, n_c, regional, n_r = split_neighbourhood(
        cells[["x", "y"]].to_numpy(float), P, cells["typed"].to_numpy(),
        float(dc.get("contact_radius_um", 15.0)), float(dc["neighbour_radius_um"]))
    pos = pd.Series(np.arange(len(cells)), index=cells["cell_id"].astype(str))
    idx = pos.reindex(table["cell_id"].astype(str)).to_numpy()
    if np.isnan(idx).any():
        sys.exit("some table cells are missing from the cell list; rebuild the table instead")
    idx = idx.astype(int)
    contact, n_c, regional, n_r = contact[idx], n_c[idx], regional[idx], n_r[idx]

    # contact (0-15 um) + regional (15-80 um) must recombine to the 80 um disc
    disc = table[[f"nbr_phi_{j:02d}" for j in range(len(names))]].to_numpy(float)
    tot = n_c + n_r
    has = tot > 0
    recon = (n_c[has, None] * contact[has] + n_r[has, None] * regional[has]) / tot[has, None]
    err = float(np.abs(recon - disc[has]).max())
    print(f"recombined contact + regional vs the 80 um disc mean: max |difference| {err:.1e} "
          f"over {has.sum():,} cells ({(~has).sum():,} with no typed neighbour)")
    if err > 1e-8:
        sys.exit("the split does not reproduce the disc mean; not writing")
    print(f"no typed contact neighbour (contact mean set to the regional mean): "
          f"{(n_c == 0).sum():,} cells ({100 * (n_c == 0).mean():.2f}%); "
          f"median contact {np.median(n_c):.0f}, regional {np.median(n_r):.0f} neighbours")

    add_split_columns(table, contact, n_c, regional, n_r)
    tmp = path + ".tmp"
    table.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    print(f"wrote {path}: {table.shape[1]} columns")


if __name__ == "__main__":
    main()

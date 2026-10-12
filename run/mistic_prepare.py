"""MisTIC step 1.2: build MisTIC's three inputs from the Xenium bundle.

    conda activate mistic
    python run/mistic_prepare.py

Runs in the MisTIC env (hpc/mistic_setup_env.sh). The Xenium parquet files are
written by Polars, and pyarrow 22 in the main env cannot read some of their
integer columns; polars reads them cleanly.

Inputs (data/Xenium_Prime_Breast_Cancer_FFPE/): transcripts.parquet,
cells.parquet, cell_boundaries.parquet and cell_feature_matrix.h5 from the
10x bundle, plus the current PhiSpace scores (config `data.phi_csv`).

Writes to `<mistic.dir>/inputs/`:
  detected_transcripts.parquet    GeoParquet, WKB points. qv >= 20, gene features
                                  only, assigned to a cell. Columns x_location,
                                  y_location, feature_name, cell_id, transcript_id.
  cell_metadata.csv (+ .parquet)  cell_id first, x_centroid, y_centroid, cell_type.
                                  MisTIC reads cell metadata only from CSV.
  cell_boundary_polygons.parquet  GeoParquet, one Polygon per cell, index cell_id.
  tiles.json                      the spatial tiles run/mistic_run.py fits.
  prepare_log.json                every filter and drop count.

`cell_type` is the argmax archetype of the current PhiSpace scores, each column
z-scored first, exactly as the mixture table's `archetype` (data.build_table).
These labels are cycle-blind: PhiSpace's input excludes the blocklist.

The three tables are made consistent before writing: a cell is kept only if it
has at least one retained transcript, a valid polygon and a PhiSpace label, and
transcripts of dropped cells are dropped with them. MisTIC asserts that the
three cell sets are identical.

Checks, all fatal: the retained transcripts reproduce cell_feature_matrix.h5
cell by cell and gene by gene; every transcript_id is unique; the labels agree
with the table's `archetype` for every analysis cell.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from tissue_dynamics.mixture.data import load_config                  # noqa: E402

WKB_POINT = np.dtype([("order", "u1"), ("type", "<u4"), ("x", "<f8"), ("y", "<f8")])
GEO_META = {"version": "1.0.0", "primary_column": "geometry",
            "columns": {"geometry": {"encoding": "WKB", "geometry_types": ["Point"],
                                     "crs": None}}}


def say(*a):
    print(*a, flush=True)


def wkb_points(x, y):
    """WKB for 2-D points (little-endian), built in numpy, as a pyarrow binary array.

    Building shapely Points for ~70M transcripts would take ~10 GB; the WKB
    bytes of a point are fixed (byte order, type 1, x, y), so they are written
    directly.
    """
    import pyarrow as pa

    rec = np.empty(len(x), dtype=WKB_POINT)
    rec["order"], rec["type"] = 1, 1
    rec["x"], rec["y"] = np.asarray(x, np.float64), np.asarray(y, np.float64)
    # int32 offsets: callers pass at most a few million points (< 2 GB of WKB)
    offsets = np.arange(0, WKB_POINT.itemsize * (len(x) + 1), WKB_POINT.itemsize, dtype=np.int32)
    return pa.Array.from_buffers(pa.binary(), len(x),
                                 [None, pa.py_buffer(offsets), pa.py_buffer(rec.view(np.uint8))])


def cell_types(cfg, ids):
    """Standardised-argmax archetype over the PhiSpace cells; NaN-free rows only."""
    dc = cfg["data"]
    P = pd.read_csv(dc["phi_csv"], index_col=0)
    P.index = P.index.astype(str)
    P = P[list(dc["phi_columns"])]
    P = P.loc[~P.isna().any(axis=1)]
    V = P.to_numpy(float)
    Z = (V - V.mean(0)) / np.where(V.std(0) > 0, V.std(0), 1.0)
    lab = pd.Series(np.asarray(P.columns)[Z.argmax(1)], index=P.index)
    return lab.reindex(ids), len(P)


def polygons(path, keep_ids):
    """One shapely Polygon per cell from the vertex table, and a validity mask."""
    import polars as pl
    import shapely

    v = pl.read_parquet(path, columns=["cell_id", "vertex_x", "vertex_y"])
    ids = v["cell_id"].to_numpy().astype(str)
    # vertices of a cell must be contiguous for the ragged construction
    change = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    starts = np.concatenate([[0], change])
    uniq = ids[starts]
    if len(np.unique(uniq)) != len(uniq):
        raise SystemExit("cell_boundaries.parquet: a cell's vertices are not contiguous")
    lengths = np.diff(np.concatenate([starts, [len(ids)]]))
    xy = np.column_stack([v["vertex_x"].to_numpy(), v["vertex_y"].to_numpy()]).astype(np.float64)
    first, last = xy[starts], xy[starts + lengths - 1]
    closed = np.all(first == last, axis=1)
    ok = lengths >= 3 + closed                  # three distinct vertices at least
    # ragged construction over the cells with enough vertices; shapely closes
    # any ring whose last vertex does not repeat its first
    rows = np.repeat(ok, lengths)
    rings = shapely.linearrings(xy[rows], indices=np.repeat(np.arange(ok.sum()), lengths[ok]))
    poly = np.full(len(uniq), None, dtype=object)
    poly[ok] = shapely.polygons(rings)
    valid = np.zeros(len(uniq), bool)
    valid[ok] = shapely.is_valid(poly[ok]) & (shapely.area(poly[ok]) > 0)
    s = pd.DataFrame({"cell_id": uniq, "n_vertices": lengths, "valid": valid})
    s["geometry"] = poly
    s = s.set_index("cell_id")
    nv, cnt = np.unique(lengths, return_counts=True)
    return s, dict(cells_with_polygon=int(len(s)), rings_closed_in_file=int(closed.sum()),
                   polygons_too_few_vertices=int((~ok).sum()),
                   polygons_invalid=int((ok & ~valid).sum()),
                   vertices_per_cell={int(k): int(c) for k, c in zip(nv, cnt)},
                   kept_cells_with_polygon=int(s.index.isin(keep_ids).sum()))


def tile_grid(xy, nx, ny, halo):
    """Equal-count tiles: x quantiles into columns, then y quantiles in each column."""
    x, y = xy[:, 0], xy[:, 1]
    xe = np.quantile(x, np.linspace(0, 1, nx + 1))
    xe[0], xe[-1] = -np.inf, np.inf
    tiles = []
    for i in range(nx):
        cm = (x >= xe[i]) & (x < xe[i + 1])
        ye = np.quantile(y[cm], np.linspace(0, 1, ny + 1))
        ye[0], ye[-1] = -np.inf, np.inf
        for j in range(ny):
            core = [float(xe[i]), float(xe[i + 1]), float(ye[j]), float(ye[j + 1])]
            tiles.append({"tile": len(tiles), "core": core,
                          "box": [core[0] - halo, core[1] + halo, core[2] - halo, core[3] + halo]})
    return tiles


def in_rect(xy, r):
    return (xy[:, 0] >= r[0]) & (xy[:, 0] < r[1]) & (xy[:, 1] >= r[2]) & (xy[:, 1] < r[3])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None, help="default: <mistic.dir>/inputs")
    a = ap.parse_args()

    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq

    t0 = time.time()
    cfg = load_config()
    mc, xdir = cfg["mistic"], cfg["data"]["xenium_dir"]
    out = a.out or os.path.join(mc["dir"], "inputs")
    os.makedirs(out, exist_ok=True)
    log = {"xenium_dir": xdir, "min_qv": mc["min_qv"], "gene_categories": mc["gene_categories"]}

    # ---- cells ------------------------------------------------------------------
    cells = pl.read_parquet(f"{xdir}/cells.parquet", columns=["cell_id", "x_centroid", "y_centroid"])
    cell_ids = cells["cell_id"].to_numpy().astype(str)
    log["cells_in_bundle"] = len(cell_ids)
    say(f"{len(cell_ids):,} segmented cells")

    # ---- transcripts: filter ----------------------------------------------------
    tx = (pl.scan_parquet(f"{xdir}/transcripts.parquet")
          .select(["transcript_id", "cell_id", "feature_name", "x_location", "y_location", "qv",
                   "codeword_category"]))
    log["transcripts_in_bundle"] = int(tx.select(pl.len()).collect().item())
    keep = ((pl.col("qv") >= float(mc["min_qv"]))
            & pl.col("codeword_category").is_in(list(mc["gene_categories"]))
            & (pl.col("cell_id") != "UNASSIGNED"))
    T = (tx.filter(keep).select(["transcript_id", "cell_id", "feature_name", "x_location",
                                 "y_location"]).collect(streaming=True))
    log["transcripts_retained_before_cell_filter"] = T.height
    say(f"transcripts: {log['transcripts_in_bundle']:,} in the bundle, {T.height:,} retained "
        f"(qv >= {mc['min_qv']}, gene features, assigned)")
    if T["transcript_id"].n_unique() != T.height:
        raise SystemExit("transcript_id is not unique")

    # cells are coded by their position in cells.parquet; an unknown id raises
    T = T.with_columns(pl.col("cell_id").cast(pl.Enum(list(cell_ids))).to_physical()
                       .alias("cix"))
    n_tx = np.bincount(T["cix"].to_numpy(), minlength=len(cell_ids))

    # ---- check against cell_feature_matrix.h5 ------------------------------------
    import h5py
    import scipy.sparse as sp

    with h5py.File(f"{xdir}/cell_feature_matrix.h5", "r") as fh:
        m = fh["matrix"]
        bc = np.array([b.decode() for b in m["barcodes"][:]])
        fname = np.array([b.decode() for b in m["features/name"][:]])
        ftype = np.array([b.decode() for b in m["features/feature_type"][:]])
        H = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]),
                          shape=tuple(m["shape"][:]))
    gene_rows = np.flatnonzero(ftype == "Gene Expression")
    genes = fname[gene_rows]
    H = H[gene_rows].T.tocsr()                          # cells x genes
    if not np.array_equal(bc, cell_ids):
        raise SystemExit("cell_feature_matrix.h5 barcodes differ from cells.parquet order")
    gix = T["feature_name"].cast(pl.Enum(list(genes))).to_physical().to_numpy()
    C = sp.csr_matrix((np.ones(T.height, np.int32), (T["cix"].to_numpy(), gix)),
                      shape=H.shape)
    C.sum_duplicates()
    diff = (C - H.astype(np.int32))
    diff.eliminate_zeros()
    log["h5_identity"] = {"cells": int(H.shape[0]), "genes": int(H.shape[1]),
                          "nonzero_differences": int(diff.nnz)}
    if diff.nnz:
        raise SystemExit(f"retained transcripts differ from cell_feature_matrix.h5 in "
                         f"{diff.nnz:,} cell-gene entries")
    say(f"  retained transcripts reproduce cell_feature_matrix.h5 exactly "
        f"({H.shape[0]:,} cells x {H.shape[1]:,} genes)")
    del C, diff, H

    # ---- cell types --------------------------------------------------------------
    ctype, n_phi = cell_types(cfg, cell_ids)
    log["cells_with_phispace_label"] = int(ctype.notna().sum())
    tp = cfg["data"]["table_parquet"]
    if os.path.exists(tp):
        tab = pd.read_parquet(tp, columns=["cell_id", "archetype"])
        agree = float(np.mean(ctype.reindex(tab["cell_id"].astype(str)).to_numpy()
                              == tab["archetype"].astype(str).to_numpy()))
        log["label_agreement_with_table_archetype"] = agree
        say(f"  labels agree with the table's archetype for {100 * agree:.3f}% of its "
            f"{len(tab):,} cells")
        if agree < 1.0:
            raise SystemExit("cell_type does not reproduce the table's archetype")

    # ---- polygons ----------------------------------------------------------------
    has_tx = n_tx > 0
    say("building polygons ...")
    polys, plog = polygons(f"{xdir}/cell_boundaries.parquet", cell_ids[has_tx])
    log["polygons"] = plog
    valid = polys["valid"].reindex(cell_ids, fill_value=False).to_numpy(bool)

    # ---- consistency -------------------------------------------------------------
    labelled = ctype.notna().to_numpy()
    final = has_tx & valid & labelled
    log["consistency"] = {
        "cells_with_retained_transcripts": int(has_tx.sum()),
        "dropped_no_retained_transcript": int((~has_tx).sum()),
        "dropped_no_valid_polygon (of cells with transcripts)": int((has_tx & ~valid).sum()),
        "dropped_no_phispace_label (of cells with transcripts and a polygon)":
            int((has_tx & valid & ~labelled).sum()),
        "cells_kept": int(final.sum()),
    }
    say("  " + "; ".join(f"{k} {v:,}" for k, v in log["consistency"].items()))
    keep_ix = np.flatnonzero(final)
    T = T.filter(pl.col("cix").is_in(pl.Series(keep_ix).cast(pl.UInt32)))
    log["transcripts_written"] = T.height
    log["transcripts_dropped_with_their_cells"] = (log["transcripts_retained_before_cell_filter"]
                                                   - T.height)
    say(f"  {T.height:,} transcripts in {len(keep_ix):,} cells")

    # ---- write: cell metadata ----------------------------------------------------
    cx, cy = cells["x_centroid"].to_numpy(), cells["y_centroid"].to_numpy()
    meta = pd.DataFrame({"cell_id": cell_ids[keep_ix], "x_centroid": cx[keep_ix],
                         "y_centroid": cy[keep_ix], "cell_type": ctype.to_numpy()[keep_ix]})
    meta.to_csv(f"{out}/cell_metadata.csv", index=False)
    meta.to_parquet(f"{out}/cell_metadata.parquet", index=False)
    log["cell_type_counts"] = {str(k): int(v) for k, v in meta["cell_type"].value_counts().items()}

    # ---- write: polygons -----------------------------------------------------------
    import geopandas as gpd

    g = polys.loc[meta["cell_id"].to_numpy(), ["geometry"]]
    gpd.GeoDataFrame(g, geometry="geometry").to_parquet(f"{out}/cell_boundary_polygons.parquet")

    # ---- write: transcripts (GeoParquet, in row groups) ---------------------------
    T = T.drop("cix")
    schema = pa.schema([("x_location", pa.float32()), ("y_location", pa.float32()),
                        ("feature_name", pa.string()), ("cell_id", pa.string()),
                        ("transcript_id", pa.uint64()), ("geometry", pa.binary())],
                       metadata={b"geo": json.dumps(GEO_META).encode()})
    path = f"{out}/detected_transcripts.parquet"
    step = 4_000_000
    with pq.ParquetWriter(path, schema, compression="zstd") as w:
        for s in range(0, T.height, step):
            b = T.slice(s, step)
            x, y = b["x_location"].to_numpy(), b["y_location"].to_numpy()
            w.write_table(pa.table({
                "x_location": pa.array(x, pa.float32()), "y_location": pa.array(y, pa.float32()),
                "feature_name": b["feature_name"].to_arrow().cast(pa.string()),
                "cell_id": b["cell_id"].to_arrow().cast(pa.string()),
                "transcript_id": b["transcript_id"].to_arrow(),
                "geometry": wkb_points(x, y)}, schema=schema), row_group_size=step)
    say(f"wrote {path} ({os.path.getsize(path) / 1e9:.2f} GB)")

    # ---- tiles ---------------------------------------------------------------------
    xy = meta[["x_centroid", "y_centroid"]].to_numpy(float)
    tiles = tile_grid(xy, int(mc["tiles_x"]), int(mc["tiles_y"]), float(mc["halo_um"]))
    core_count = np.zeros(len(xy), int)
    for t in tiles:
        t["n_core"] = int(in_rect(xy, t["core"]).sum())
        t["n_cells"] = int(in_rect(xy, t["box"]).sum())
        core_count += in_rect(xy, t["core"])
    if not np.all(core_count == 1):
        raise SystemExit("tile cores do not partition the cells")
    json.dump({"tiles_x": mc["tiles_x"], "tiles_y": mc["tiles_y"], "halo_um": mc["halo_um"],
               "tiles": tiles}, open(f"{out}/tiles.json", "w"), indent=1)
    say(f"{len(tiles)} tiles: core cells {min(t['n_core'] for t in tiles):,}-"
        f"{max(t['n_core'] for t in tiles):,}, with halo {min(t['n_cells'] for t in tiles):,}-"
        f"{max(t['n_cells'] for t in tiles):,}")

    log["seconds"] = round(time.time() - t0, 1)
    json.dump(log, open(f"{out}/prepare_log.json", "w"), indent=1, default=int)
    say(f"done in {log['seconds']:.0f}s -> {out}")


if __name__ == "__main__":
    main()

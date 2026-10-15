"""MisTIC step 1.3: fit MisTIC on one spatial tile and save p per transcript.

    conda activate mistic
    python run/mistic_run.py --tile 7       # one tile; the Slurm array index
    python run/mistic_run.py --tile -1      # the whole section in one fit

Reads `<mistic.dir>/inputs/` (run/mistic_prepare.py). For tile k, the cells
whose centroid lies in the tile's box (its core plus a `halo_um` margin) and all
of their transcripts are written to a scratch directory as MisTIC inputs, and
MisTIC is fitted on them through its Python API, in the spec's order:
import_data (paths, not data frames: it deep-copies), patchfy_data,
initialize_parameters, training_loop (early stopping applies),
compute_reassign_probs. `correct_tx` is never called: the analysis uses the soft
probabilities, not thresholded reassignment.

The probability column is `reassign_probs`: MisTIC's calibrated reassignment
probability, taken for each transcript at its most likely neighbour (up to
`nearest` = 3 candidate cells of a different type within 5 um, mask to mask).
`reassign_probs_raw` is the same before calibration. Transcripts MisTIC does not
evaluate -- no candidate cell of another type nearby -- have no row; the gather
step gives them p = 0.

Writes `<mistic.dir>/tiles/tile_KK/`:
  tx_reassign_info.parquet     MisTIC's table as it computes it: every evaluated
                               transcript of the tile, halo cells included.
                               MisTIC does not save it itself.
  mistic.pt, mistic_meta.json  the fitted model (save_model, no correction result)
  probs_core.parquet           transcript_id, cell_id, gene, p, p_raw,
                               neighbor_cell_id, for the evaluated transcripts of
                               the tile's CORE cells -- what the gather step joins
  summary.json                 sizes, epochs, losses, timings, peak memory
"""
import argparse
import json
import os
import resource
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from tissue_dynamics.mixture.data import load_config                  # noqa: E402

PROB_COL = "reassign_probs"
MIN_TYPE_CELLS = 30      # pydeseq2 builds 3 pseudo-bulk replicates per type
MIN_TYPES_LEFT = 2       # one-vs-rest DE and other-type candidates need two types;
                         # every ovarian 6x6 tile keeps >= 6 after exclusion


def say(*a):
    print(*a, flush=True)


def peak_rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1e9 if sys.platform == "darwin" else 1e6)      # bytes on macOS, KiB on Linux


def in_rect(xy, r):
    return (xy[:, 0] >= r[0]) & (xy[:, 0] < r[1]) & (xy[:, 1] >= r[2]) & (xy[:, 1] < r[3])


def tile_cells(meta, tiles, k, box=None):
    """`(cells in the fit, cells in the core)` for tile k; k = -1 is everything.

    `box` = [x0, x1, y0, y1] fits that rectangle instead, with the halo taken
    off each side for the core: a small region for smoke tests.
    """
    xy = meta[["x_centroid", "y_centroid"]].to_numpy(float)
    if box is not None:
        h = float(tiles["halo_um"])
        return in_rect(xy, box), in_rect(xy, [box[0] + h, box[1] - h, box[2] + h, box[3] - h])
    if k < 0:
        allc = np.ones(len(meta), bool)
        return allc, allc
    t = tiles["tiles"][k]
    return in_rect(xy, t["box"]), in_rect(xy, t["core"])


def write_tile_inputs(inp, meta, fit_mask, scratch):
    """The tile's MisTIC inputs, as files (MisTIC deep-copies data frames)."""
    import geopandas as gpd
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    sub = meta.loc[fit_mask]
    paths = {k: os.path.join(scratch, f) for k, f in
             (("meta", "cell_metadata.csv"), ("poly", "cell_boundary_polygons.parquet"),
              ("tx", "detected_transcripts.parquet"))}
    sub.to_csv(paths["meta"], index=False)
    g = gpd.read_parquet(f"{inp}/cell_boundary_polygons.parquet")
    g.loc[sub["cell_id"].to_numpy()].to_parquet(paths["poly"])
    d = ds.dataset(f"{inp}/detected_transcripts.parquet", format="parquet")
    tab = d.to_table(filter=ds.field("cell_id").isin(sub["cell_id"].to_numpy().tolist()))
    pq.write_table(tab.replace_schema_metadata(d.schema.metadata), paths["tx"],
                   compression="zstd")
    return paths, tab.num_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tile", type=int, default=None, help="tile index; -1 = whole section")
    ap.add_argument("--box", type=float, nargs=4, metavar=("X0", "X1", "Y0", "Y1"),
                    help="fit this rectangle instead (smoke tests); written to tile_box/")
    ap.add_argument("--epochs", type=int, default=None, help="default: mistic.n_epochs")
    ap.add_argument("--threads", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR"))
    ap.add_argument("--keep-scratch", action="store_true")
    a = ap.parse_args()
    if (a.tile is None) == (a.box is None):
        ap.error("give exactly one of --tile and --box")

    t0, times = time.time(), {}
    cfg = load_config()
    mc = cfg["mistic"]
    inp = os.path.join(mc["dir"], "inputs")
    tag = "box" if a.box else ("all" if a.tile < 0 else f"{a.tile:02d}")
    out = os.path.join(mc["dir"], "tiles", f"tile_{tag}")
    os.makedirs(out, exist_ok=True)
    seed = int(mc["seed"]) + max(a.tile or 0, 0)
    n_epochs = int(a.epochs or mc["n_epochs"])

    import polars as pl
    import pyarrow.parquet as pq
    import torch

    # pydeseq2 sizes its worker pool from the node's core count, not the job's
    import MisTIC.generate_tx_feature as gtf
    from MisTIC.__version__ import __version__ as mistic_version
    from MisTIC.mistic_class import mistic
    gtf.get_num_processes = lambda n_cpus=None: int(a.threads)
    torch.set_num_threads(int(a.threads))
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"--device cuda, but torch {torch.__version__} (CUDA build "
                         f"{torch.version.cuda}) cannot see a GPU on {os.uname().nodename}")
    say(f"torch {torch.__version__}; device "
        + (f"cuda ({torch.cuda.get_device_name(0)})" if a.device != "cpu"
           and torch.cuda.is_available() else "cpu"))
    np.random.seed(seed)                   # MisTIC shuffles patches with the global RNGs
    torch.manual_seed(seed)

    meta = pd.read_parquet(f"{inp}/cell_metadata.parquet")
    tiles = json.load(open(f"{inp}/tiles.json"))
    fit_mask, core_mask = tile_cells(meta, tiles, a.tile, a.box)
    types = meta.loc[fit_mask, "cell_type"].value_counts()
    say(f"tile {tag}: {fit_mask.sum():,} cells in the fit, {core_mask.sum():,} in the core; "
        f"types {dict(types)}")
    rare = types[types < MIN_TYPE_CELLS]
    excluded_core = []
    if len(rare):
        if mc.get("rare_type_policy", "stop") != "exclude":
            raise SystemExit(f"tile {tag}: types with fewer than {MIN_TYPE_CELLS} cells "
                             f"({dict(rare)}); MisTIC's pseudo-bulk DE needs more. Use fewer "
                             f"tiles, or set mistic.rare_type_policy: exclude.")
        # `exclude` (ovarian; spatially concentrated types leave a few cells of a
        # type in some tiles): those cells leave THIS tile's fit -- they are neither
        # evaluated nor candidate sources here -- and their transcripts keep p = 0,
        # as for any cell MisTIC does not evaluate. Core cells excluded this way are
        # listed, and run/mistic_gather.py marks them in_mistic = False.
        drop = fit_mask & meta["cell_type"].isin(list(rare.index)).to_numpy()
        excluded_core = meta.loc[core_mask & drop, "cell_id"].astype(str).tolist()
        fit_mask, core_mask = fit_mask & ~drop, core_mask & ~drop
        if len(types) - len(rare) < MIN_TYPES_LEFT:
            raise SystemExit(f"tile {tag}: only {len(types) - len(rare)} types would remain "
                             f"after excluding {dict(rare)}; MisTIC needs at least "
                             f"{MIN_TYPES_LEFT}. Use fewer tiles.")
        say(f"tile {tag}: excluding {int(drop.sum())} cells of types with fewer than "
            f"{MIN_TYPE_CELLS} ({dict(rare)}), {len(excluded_core)} of them in the core")

    scratch = tempfile.mkdtemp(prefix=f"mistic_{tag}_", dir=a.scratch)
    try:
        paths, n_tx = write_tile_inputs(inp, meta, fit_mask, scratch)
        times["inputs"] = time.time() - t0
        say(f"  {n_tx:,} transcripts; inputs in {scratch} ({times['inputs']:.0f}s)")

        m = mistic(cell_centroid_x_col="x_centroid", cell_centroid_y_col="y_centroid",
                   celltype_col="cell_type", tx_x_col="x_location", tx_y_col="y_location",
                   gene_col="feature_name", cell_col="cell_id", seed=seed,
                   model_device=a.device)
        t1 = time.time()
        m.import_data(cell_metadata=paths["meta"], cell_boundary_polygons=paths["poly"],
                      detected_transcripts=paths["tx"])
        times["import_data"] = time.time() - t1
        # MisTIC's one-vs-rest DESeq2 runs on random subsamples of cells, so a gene
        # with a handful of transcripts can drop out of every pseudo-bulk; its log2
        # fold change is then missing and so are the expression features of its
        # transcripts (NaN after the join). MisTIC fills missing p-values but not
        # these, and one NaN feature makes the loss NaN from the first batch (4 of
        # 36 ovarian tiles: genes with 1-5 transcripts in the tile). Fill them with
        # 0 -- the logit of the median rank, "no differential evidence". MisTIC's
        # prior features for the same transcripts are complete. A no-op where
        # nothing is missing, so tiles fitted before this are unchanged.
        exp_cols = ["exp_feature", "neighbor_exp_feature"]
        miss = lambda c: pl.col(c).is_null() | pl.col(c).is_nan()          # noqa: E731
        n_imputed = int(m.intf_tx.select(pl.any_horizontal([miss(c) for c in exp_cols])).to_series().sum())
        if n_imputed:
            m.intf_tx = m.intf_tx.with_columns([pl.when(miss(c)).then(0.0).otherwise(pl.col(c)).alias(c)
                                                for c in exp_cols])
            say(f"  {n_imputed:,} transcripts with a missing DE expression feature set to 0 "
                f"(of {m.intf_tx.height:,})")
        num = [c for c, t in zip(m.intf_tx.columns, m.intf_tx.dtypes) if t in (pl.Float32, pl.Float64)]
        left = int(m.intf_tx.select(pl.any_horizontal([miss(c) for c in num])).to_series().sum())
        if left:
            raise SystemExit(f"tile {tag}: {left:,} transcripts still have a missing feature")
        t1 = time.time()
        m.patchfy_data()
        m.initialize_parameters()
        m.training_loop(n_epochs=n_epochs)
        times["training"] = time.time() - t1
        t1 = time.time()
        m.compute_reassign_probs()
        times["reassign_probs"] = time.time() - t1

        info = m.tx_reassign_info
        info.write_parquet(os.path.join(out, "tx_reassign_info.parquet"))
        m.save_model(dir_name=out, model_name="mistic", save_correction_result=False)
        if PROB_COL not in info.columns:
            raise SystemExit(f"tx_reassign_info has no {PROB_COL!r}: {info.columns}")

        # ---- join back: molecule_id is tx_<row> of the tile's transcript file ----
        ref = pq.read_table(paths["tx"], columns=["transcript_id", "cell_id", "feature_name"])
        row = info["molecule_id"].str.slice(3).cast(pl.Int64).to_numpy()
        tid = ref["transcript_id"].to_numpy()[row]
        ok_cell = np.array_equal(np.asarray(ref["cell_id"].to_numpy(zero_copy_only=False),
                                            dtype=str)[row], info["cell_id"].to_numpy().astype(str))
        ok_gene = np.array_equal(np.asarray(ref["feature_name"].to_numpy(zero_copy_only=False),
                                            dtype=str)[row], info["gene"].to_numpy().astype(str))
        if not (ok_cell and ok_gene):
            raise SystemExit("molecule_id does not map back to the input rows "
                             f"(cell match {ok_cell}, gene match {ok_gene})")
        core_ids = set(meta.loc[core_mask, "cell_id"])
        cell = info["cell_id"].to_numpy().astype(str)
        keep = np.fromiter((c in core_ids for c in cell), bool, len(cell))
        res = pd.DataFrame({
            "transcript_id": tid[keep].astype(np.uint64), "cell_id": cell[keep],
            "gene": info["gene"].to_numpy().astype(str)[keep],
            "p": info[PROB_COL].to_numpy().astype(np.float32)[keep],
            "p_raw": info["reassign_probs_raw"].to_numpy().astype(np.float32)[keep],
            "neighbor_cell_id": info["neighbor_cell_id"].to_numpy().astype(str)[keep]})
        res.to_parquet(os.path.join(out, "probs_core.parquet"), index=False)

        n_core_tx = int(np.isin(np.asarray(ref["cell_id"].to_numpy(zero_copy_only=False), dtype=str),
                                list(core_ids)).sum())
        ce = [float(np.mean(e)) for e in m.CEl_list]
        kl = [np.asarray(e).mean(0).tolist() for e in m.KLD_list]
        p = res["p"].to_numpy()
        summ = {
            "tile": a.tile, "box": a.box, "seed": seed, "device": str(m.model_device),
            "threads": a.threads,
            "cells_fit": int(fit_mask.sum()), "cells_core": int(core_mask.sum()),
            "transcripts_fit": int(n_tx), "transcripts_core": n_core_tx,
            "evaluated_fit": int(info.height), "evaluated_core": int(len(res)),
            "evaluated_core_fraction": float(len(res) / max(n_core_tx, 1)),
            "nearest_used": int(m.generate_feature_par["nearest"]),
            "n_cell_types": int(m.n_leiden), "n_genes": int(m.n_genes),
            "epochs_run": len(ce), "max_epochs": n_epochs,
            "mean_cross_entropy_by_epoch": ce, "mean_kl_by_epoch": kl,
            "reassign_weight": np.asarray(m.reassign_coefficients.weight.detach().cpu()).ravel().tolist(),
            "reassign_bias": np.asarray(m.reassign_coefficients.bias.detach().cpu()).ravel().tolist(),
            "p_core_quantiles": dict(zip(("q10", "q50", "q90", "q99"),
                                         np.quantile(p, [.1, .5, .9, .99]).tolist())) if len(p) else {},
            "p_core_mean_over_all_core_transcripts": float(p.sum() / max(n_core_tx, 1)),
            "p_core_above_0.5": float(np.mean(p > 0.5)) if len(p) else float("nan"),
            "columns_of_tx_reassign_info": list(info.columns), "probability_column": PROB_COL,
            "seconds": {k: round(v, 1) for k, v in times.items()},
            "peak_rss_gb": round(peak_rss_gb(), 2),
            "versions": {"txMisTIC": mistic_version, "torch": torch.__version__,
                         "polars": pl.__version__},
        }
        summ["imputed_exp_features"] = n_imputed
        summ["excluded_rare_types"] = {str(k): int(v) for k, v in rare.items()}
        summ["excluded_core_cells"] = len(excluded_core)
        with open(os.path.join(out, "excluded_core_cells.txt"), "w") as fh:
            fh.write("".join(c + "\n" for c in excluded_core))
        summ["seconds"]["total"] = round(time.time() - t0, 1)
        json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)
        say(f"tile {tag}: {len(ce)} epochs; {len(res):,} of {n_core_tx:,} core transcripts "
            f"evaluated; mean p over core transcripts "
            f"{summ['p_core_mean_over_all_core_transcripts']:.4f}; peak RSS "
            f"{summ['peak_rss_gb']:.1f} GB; {summ['seconds']['total']:.0f}s")
    finally:
        if not a.keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()

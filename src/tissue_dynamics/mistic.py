"""MisTIC-derived technical covariates (instructions/mistic_implementation_spec.md, step 2).

MisTIC (run/mistic_run.py) gives every retained transcript k a probability p_k
that it is misassigned to its host cell; q_k = 1 - p_k. Transcripts MisTIC did
not evaluate -- no candidate cell of another type within reach, or a host cell
left out of its input -- have p_k = 0. Per cell i:

    N_i     raw transcript count, all genes
    Nt_i    sum of q_k over the cell's transcripts: the MisTIC-weighted library
            size (the spec's N-tilde). Also kept over the transcripts whose gene
            is not in the blocklist B, the like-for-like replacement for the
            mixture's N, which excludes B.
    f_i     mean p_k over the cell's transcripts whose gene is not in B: the
            contamination fraction. B contains the outcome genes, so they never
            feed the correction.
    s_i^G   sum_{j in nbr(i)} Y_j^G / sum_{j in nbr(i)} N_j, with nbr(i) the 6
            nearest cells by centroid (i excluded) and raw counts: the share of
            G among the transcripts that spillover into i could come from
    m_i^G   f_i N_i s_i^G: the expected number of misassigned G transcripts in i

`m_gene_set` builds m^G for any gene set, so the main models (G = the Tirosh
S/G2M set) and the negative-control and spillover tests share one function.

Everything is indexed by the cells of `cell_feature_matrix.h5`, in its order.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

N_SOURCES = 6          # spillover sources: the 6 nearest cells by centroid


def paths(cfg):
    d = cfg["mistic"]["dir"]
    return {"inputs": os.path.join(d, "inputs"), "tiles": os.path.join(d, "tiles"),
            "transcripts": os.path.join(d, "transcripts_mistic.parquet"),
            "cells": os.path.join(d, "cell_mistic.parquet"),
            "weighted_h5": os.path.join(d, "cell_feature_matrix_mistic.h5"),
            # core cells a tile left out of its fit (mistic.rare_type_policy: exclude)
            "excluded": os.path.join(d, "excluded_cells.txt")}


# --------------------------------------------------------------------------
# Raw counts and cells
# --------------------------------------------------------------------------
@dataclass
class Counts:
    """Raw counts, cells x genes, in cell_feature_matrix.h5 order."""

    X: object               # scipy CSR, int32
    cell_ids: np.ndarray
    genes: np.ndarray
    xy: np.ndarray          # cell centroids (um), cells.csv.gz

    def gene_index(self, names):
        pos = pd.Index(self.genes).get_indexer(list(names))
        if (pos < 0).any():
            missing = [g for g, p in zip(names, pos) if p < 0]
            raise KeyError(f"genes not on the panel: {missing[:5]}")
        return pos

    def gene_sum(self, names):
        """`Y^G`: raw counts summed over the genes in `names`, per cell."""
        return np.asarray(self.X[:, self.gene_index(names)].sum(1)).ravel().astype(np.float64)


def read_counts(cfg):
    """Gene-expression counts and centroids for every segmented cell."""
    import h5py
    import scipy.sparse as sp

    xdir = cfg["data"]["xenium_dir"]
    with h5py.File(f"{xdir}/cell_feature_matrix.h5", "r") as fh:
        m = fh["matrix"]
        bc = np.array([b.decode() for b in m["barcodes"][:]])
        fname = np.array([b.decode() for b in m["features/name"][:]])
        ftype = np.array([b.decode() for b in m["features/feature_type"][:]])
        H = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]),
                          shape=tuple(m["shape"][:]))
    rows = np.flatnonzero(ftype == "Gene Expression")
    X = H[rows].T.tocsr().astype(np.int32)
    cq = (pd.read_csv(f"{xdir}/cells.csv.gz", usecols=["cell_id", "x_centroid", "y_centroid"])
          .astype({"cell_id": str}).set_index("cell_id").reindex(bc))
    return Counts(X=X, cell_ids=bc, genes=fname[rows],
                  xy=cq[["x_centroid", "y_centroid"]].to_numpy(float))


def blocklist(cfg, genes):
    """The PhiSpace blocklist B (a superset of the Tirosh genes), on the panel."""
    from .mixture.data import _tirosh_genes

    b = set(pd.read_csv(cfg["data"]["blocklist_csv"], header=None)[0].astype(str))
    b |= set(_tirosh_genes())
    return np.isin(np.asarray(genes), sorted(b))


def tirosh_on_panel(genes):
    from .mixture.data import _tirosh_genes

    g = set(np.asarray(genes))
    return [x for x in _tirosh_genes() if x in g]


# --------------------------------------------------------------------------
# Per-transcript p, read back as integer codes
# --------------------------------------------------------------------------
def transcript_codes(path, counts):
    """`(cell index, gene index, p)` per transcript of transcripts_mistic.parquet.

    Strings are read as dictionaries and mapped through the (small) dictionary,
    so 70M cell ids are never materialised as Python objects.
    """
    import pyarrow.parquet as pq

    t = pq.read_table(path, columns=["cell_id", "gene", "p"], read_dictionary=["cell_id", "gene"])
    out = []
    for col, ref in (("cell_id", counts.cell_ids), ("gene", counts.genes)):
        arr = t[col].combine_chunks()
        lut = pd.Index(ref).get_indexer(arr.dictionary.to_pandas().astype(str))
        if (lut < 0).any():
            raise KeyError(f"{col}: {int((lut < 0).sum())} values not in cell_feature_matrix.h5")
        out.append(lut[arr.indices.to_numpy()].astype(np.int32))
    return out[0], out[1], t["p"].to_numpy().astype(np.float64)


# --------------------------------------------------------------------------
# Step 2: per-cell covariates
# --------------------------------------------------------------------------
def source_neighbours(xy, pool, k=N_SOURCES):
    """`(n, k)` indices of the k nearest cells in `pool` to each cell, never itself.

    `pool` marks the cells that can be a source of spillover (those with at
    least one transcript); every cell, in the pool or not, gets k sources.
    """
    from scipy.spatial import cKDTree

    pidx = np.flatnonzero(pool)
    _d, j = cKDTree(xy[pidx]).query(xy, k=k + 1)
    j = pidx[j]
    self_hit = j == np.arange(len(xy))[:, None]
    # drop the cell itself where it is among the k+1; otherwise the farthest
    keep = ~self_hit
    keep[~self_hit.any(1), k] = False
    return j[keep].reshape(len(xy), k)


def spill_share(YG, N, nbr):
    """`s_i^G = sum_{j in nbr(i)} Y_j^G / sum_{j in nbr(i)} N_j` (0 where no source counts)."""
    num = np.asarray(YG, float)[nbr].sum(1)
    den = np.asarray(N, float)[nbr].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, 0.0)


def expected_misassigned(f, N, s):
    """`m_i^G = f_i N_i s_i^G`."""
    return np.asarray(f, float) * np.asarray(N, float) * np.asarray(s, float)


@dataclass
class Covariates:
    """Step 2's per-cell quantities, aligned with `counts.cell_ids`."""

    counts: Counts
    N: np.ndarray            # raw, all genes
    N_nonB: np.ndarray       # raw, genes not in B (the mixture's N)
    Nt: np.ndarray           # sum q_k, all genes (the spec's N-tilde)
    Nt_nonB: np.ndarray      # sum q_k, genes not in B
    f: np.ndarray            # mean p over genes not in B; 0 where the cell has none
    nbr: np.ndarray          # (n, 6) spillover sources
    in_mistic: np.ndarray    # the cell was in MisTIC's input

    def m_gene_set(self, genes):
        """`m^G` for the gene set `genes` (raw counts; outcome-gene p never used)."""
        s = spill_share(self.counts.gene_sum(genes), self.N, self.nbr)
        return expected_misassigned(self.f, self.N, s), s


def cell_covariates(counts, cell, gene, p, isB):
    """Per-cell N, Nt, f and the spillover sources from per-transcript codes.

    Count totals come from the raw matrix, and only the sums of p from the
    transcripts, so a transcript outside MisTIC's input counts with p = 0.
    """
    n = len(counts.cell_ids)
    nb = ~isB[gene]
    psum = np.bincount(cell, weights=p, minlength=n)
    psum_nb = np.bincount(cell[nb], weights=p[nb], minlength=n)
    N = np.asarray(counts.X.sum(1)).ravel().astype(np.float64)
    N_nonB = np.asarray(counts.X[:, np.flatnonzero(~isB)].sum(1)).ravel().astype(np.float64)
    f = np.where(N_nonB > 0, psum_nb / np.maximum(N_nonB, 1.0), 0.0)
    return Covariates(counts=counts, N=N, N_nonB=N_nonB, Nt=N - psum, Nt_nonB=N_nonB - psum_nb,
                      f=f, nbr=source_neighbours(counts.xy, N > 0),
                      in_mistic=np.bincount(cell, minlength=n) > 0)


def weighted_matrix(counts, cell, gene, p):
    """`X~_ig = sum_{k in i, gene g} q_k = X_ig - sum p`, sparse float32, cells x genes."""
    import scipy.sparse as sp

    Psp = sp.csr_matrix((p.astype(np.float32), (cell, gene)), shape=counts.X.shape)
    Psp.sum_duplicates()
    W = (counts.X.astype(np.float32) - Psp).tocsr()
    W.eliminate_zeros()
    return W


def write_10x_h5(path, W, counts, template):
    """Write `W` in cell_feature_matrix.h5's layout (features x cells, CSC), with
    float data, the template's barcodes and its gene-expression features only --
    so whatever reads the original matrix reads this one unchanged."""
    import h5py
    import scipy.sparse as sp

    M = sp.csc_matrix(W.T.astype(np.float32))
    with h5py.File(template, "r") as src, h5py.File(path + ".tmp", "w") as dst:
        s, g = src["matrix"], dst.create_group("matrix")
        ftype = np.array([b.decode() for b in s["features/feature_type"][:]])
        rows = np.flatnonzero(ftype == "Gene Expression")
        g.create_dataset("barcodes", data=s["barcodes"][:])
        g.create_dataset("data", data=M.data, compression="gzip")
        g.create_dataset("indices", data=M.indices.astype(np.int64), compression="gzip")
        g.create_dataset("indptr", data=M.indptr.astype(np.int64))
        g.create_dataset("shape", data=np.array(M.shape, dtype=np.int32))
        fg = g.create_group("features")
        for k in s["features"].keys():
            v = s["features"][k]
            if isinstance(v, h5py.Dataset) and v.shape and v.shape[0] == len(ftype):
                fg.create_dataset(k, data=v[:][rows])
            elif isinstance(v, h5py.Dataset):
                fg.create_dataset(k, data=v[()])
        for k, v in s.attrs.items():
            dst.attrs[k] = v
        dst.attrs["mistic_weighted"] = ("data = sum over the cell's transcripts of the gene "
                                        "of q = 1 - p, p from MisTIC (run/mistic_covariates.py)")
    os.replace(path + ".tmp", path)


# --------------------------------------------------------------------------
# Readers for the models
# --------------------------------------------------------------------------
def load_cell_table(cfg):
    """`cell_mistic.parquet`, indexed by cell_id."""
    return pd.read_parquet(paths(cfg)["cells"]).set_index("cell_id")


def load_covariates(cfg):
    """Rebuild a `Covariates` from the saved per-cell table (no transcript pass),
    for computing m^G of further gene sets."""
    counts = read_counts(cfg)
    t = load_cell_table(cfg).reindex(counts.cell_ids)
    nbr = source_neighbours(counts.xy, t["N"].to_numpy() > 0)
    return Covariates(counts=counts, N=t["N"].to_numpy(float), N_nonB=t["N_nonB"].to_numpy(float),
                      Nt=t["Nt"].to_numpy(float), Nt_nonB=t["Nt_nonB"].to_numpy(float),
                      f=t["f"].to_numpy(float), nbr=nbr, in_mistic=t["in_mistic"].to_numpy(bool))


def summary_json(path, d):
    with open(path, "w") as fh:
        json.dump(d, fh, indent=1, default=float)


def test_setup(technical="mistic"):
    """`(cfg, df, out_dir)` for the MisTIC tests (run/mistic_t*.py).

    The config is the project's with `model.technical` forced to `technical`
    ("mistic", or "plain" for the baseline without area or MisTIC); the table
    carries `phi_names` in its attrs. Two environment variables redirect a dry
    run: MISTIC_TABLE (another table with the MisTIC columns) and MISTIC_OUT
    (another output directory), so test code can be exercised without writing
    into outputs/mistic.
    """
    from .mixture.data import load_config

    cfg = load_config()
    cfg["model"] = {**cfg["model"], "technical": technical}
    if os.environ.get("MISTIC_TABLE"):
        cfg["data"] = {**cfg["data"], "table_parquet": os.environ["MISTIC_TABLE"]}
    path = cfg["data"]["table_parquet"]
    df = pd.read_parquet(path)
    with open(os.path.splitext(path)[0] + ".phi_names.json") as fh:
        df.attrs["phi_names"] = json.load(fh)
    out = os.environ.get("MISTIC_OUT", "outputs/mistic")
    os.makedirs(out, exist_ok=True)
    return cfg, df, out

# MisTIC

Per-transcript misassignment probabilities from MisTIC (Yang et al. 2025,
bioRxiv 10.64898/2025.12.11.693759; `txMisTIC` 0.0.2). They replace cell area as the technical
covariate. Spec: `instructions/mistic_implementation_spec.md`.

## Probability column

`tx_reassign_info` (MisTIC's per-transcript table, saved per tile as
`tx_reassign_info.parquet`) has the columns

`molecule_id, cell_id, gene, neighbor_cell_id, cell_type, neighbor_celltype,
neighbor_index, distance_feature, prior_distance_feature, exp_feature,
prior_exp_feature, neighbor_exp_feature, prior_neighbor_exp_feature,
reassign_probs_raw, reassign_probs`

**p is `reassign_probs`**: MisTIC's calibrated probability that the transcript
belongs to a neighbouring cell rather than its host. MisTIC computes it against
each of up to 3 candidate cells and keeps the most likely one
(`neighbor_cell_id`). `reassign_probs_raw` is the same value before calibration.

A transcript has no row in `tx_reassign_info` when MisTIC has nothing to
compare it against. MisTIC only considers neighbours of a different cell type
whose boundary is within 5 µm of the host cell's boundary. Such transcripts get
**p = 0** in `transcripts_mistic.parquet`.

`molecule_id` is `tx_<row>` of the transcript table MisTIC read. `run/mistic_run.py`
maps it back to `transcript_id`, and checks that cell and gene agree row by row.

## Pipeline

| step | script | env | where |
| --- | --- | --- | --- |
| 1.1 environment | `hpc/mistic_setup_env.sh` | builds `mistic` | once |
| 1.2 inputs | `run/mistic_prepare.py` | mistic | local or Spartan, ~2 min, 8 GB |
| 1.3 fit | `run/mistic_run.py --tile k`, `hpc/mistic.slurm` | mistic | Spartan GPU array, 36 tasks |
| 1.5 join | `run/mistic_gather.py` | either | ~1 min |
| 2 covariates | `run/mistic_covariates.py` | main | ~1 min |

Inputs (`data/mistic/inputs/`, built by step 1.2):

- **`detected_transcripts.parquet`**: GeoParquet with WKB points. Keeps transcripts with
  qv ≥ 20, from `predesigned_gene` or `custom_gene` codewords, assigned to a cell.
  Columns: `x_location, y_location, feature_name, cell_id, transcript_id`. These
  transcripts reproduce `cell_feature_matrix.h5` exactly, cell by cell and gene by gene.
- **`cell_metadata.csv`**: `cell_id, x_centroid, y_centroid, cell_type`. MisTIC
  reads cell metadata only from CSV; a `.parquet` copy is kept alongside.
  `cell_type` is the standardised argmax of the current (log1p) PhiSpace scores.
  It matches the mixture table's `archetype` for all 561,897 analysis cells.
- **`cell_boundary_polygons.parquet`**: GeoParquet, one polygon per cell,
  indexed by `cell_id`.
- **`tiles.json`**: the 6 × 6 tiles that cover the section.

Of the 699,110 segmented cells, 682,705 go to MisTIC:

- 15,312 have no retained transcript;
- 514 have an invalid polygon (self-intersecting);
- 579 have no PhiSpace score.

304 of the dropped cells are analysis cells (0.05%). Their transcripts get p = 0.

## Tiling

The section is fitted as 36 tiles, not in one fit. MisTIC's prior-expression
feature builds a long table with one row per (cell, gene) pair: 3.5 billion rows
for the whole section. It also keeps several dense cells × genes copies. A
single fit would need far more memory than a node has.

The tiles split the cells into equal counts: six columns at x quantiles, then
six rows at y quantiles within each column. Each tile holds about 18,965 core
cells. Each tile is fitted on its core plus a 50 µm halo (20,200 to 22,300
cells). The halo matches MisTIC's own neighbour search, which stops at 50 µm
between centroids. p is kept only for transcripts whose host cell's centroid
lies in the core, so the cores partition the cells and every transcript gets
exactly one value. Every tile has at least 509 cells of its rarest type, so
MisTIC's pseudo-bulk DESeq2 step is well posed.

Each tile's fit is independent: its own cell-type classifier, DE features and
calibration. Training uses MisTIC's own priors and defaults
(`nearest` = 3, `mask_dist_cutoff` = 5 µm, `max_centroid_dist` = 50 µm,
`n_epochs` = 50 with early stopping). Seeds are fixed per tile.

## Outputs (`data/mistic/`, not in git)

- **`transcripts_mistic.parquet`**: `transcript_id, cell_id, gene, p`, one row per
  transcript MisTIC was given.
- **`cell_mistic.parquet`**: one row per segmented cell, in the `.h5` order.
  - `N` and `N_nonB`: raw totals over all genes, and over genes not in the
    blocklist B. `N_nonB` is the mixture's `N`.
  - `Nt` and `Nt_nonB`: the same sums, weighted by q = 1 − p.
  - `f`: mean p over the cell's non-B transcripts.
  - `Y_cyc`, `s_cyc`, `m_cyc`: the Tirosh-set raw count, the neighbour share and
    the expected misassigned count.
  - `in_mistic`: whether the cell was in MisTIC's input.
- **`cell_feature_matrix_mistic.h5`**: X̃ (the sum of q over each cell's transcripts
  of each gene) in the layout of `cell_feature_matrix.h5`, with float data. This is
  the PhiSpace input for spec 3.1, which runs in R.
- **`tiles_summary.csv`**: one row per tile.

## Order of work after the MisTIC array

Steps 1, 4 and 5 run on Spartan; the rest run locally.

1. **Join.** Once all 36 tiles are done, run `python run/mistic_gather.py` (any
   env), then copy back `data/mistic/transcripts_mistic.parquet` and
   `tiles_summary.csv`.
2. **Covariates and T2.** Run `python run/mistic_covariates.py`, then
   `python run/mistic_t2_sanity.py`. T2 is blocking: if the curve is flat, stop here.
3. **Table and the null's tests.** Run `python run/add_mistic_columns.py`, then
   `run/mistic_t1_recovery.py`, `run/mistic_t3_negative_control.py` and
   `run/mistic_t4_spillover.py`. T1 and T3 are blocking.
4. **Switch to MisTIC.**
   - Set `model.technical: mistic`, then commit and push.
   - Copy the updated `data/mixture_cells_major.parquet` to Spartan.
   - Archive the area fits: `mv outputs/hpc outputs/hpc_area`, on Spartan and
     locally (`hpc_area` is gitignored).
5. **Refit the mixture under MisTIC.** Run the multistart job,
   `run/collect_point_estimate.py`, then the bootstrap job, as before.
6. **Compare and report.** Sync `outputs/hpc` back, then run
   `python run/mistic_t6_old_vs_new.py` and `python run/mistic_report.py`.

Run every test locally with `MISTIC_TABLE=<table> MISTIC_OUT=<dir>` to dry-run
it on another table without writing into `outputs/mistic`.

## Known limitation

MisTIC's internal model still sees cycle-gene transcripts when it estimates p
for other genes. They are part of the expression profiles it compares. The
covariates the models read never use p of a cycle-gene transcript:

- f is computed from genes outside B;
- m is built from f and raw counts;
- the library size in the models is `Nt_nonB`, which sums q over genes outside B.
  It is the like-for-like replacement for the mixture's `N`.

`Nt`, which sums q over all genes as the spec's notation table defines it, is
kept in the table but not used. Using it would put cycle-gene probabilities into
the offset, and the spec rules out any use of them.

# What produced what in `outputs/`

Written 2026-10-18 from the scripts, the reports' own headers and git history. Each entry
names the model, the script that writes it, and how current it is.

## The two reference models

Almost everything below is one of these, or is compared against one.

- **Production mixture (breast).** Two-component Poisson mixture, Stage B. 561,897 analysis cells.
  - Outcome y: summed Tirosh S + G2M counts (53 genes).
  - Component means: Nt e^b0 (+ e^delta for cycling) + theta_m m. This is the MisTIC technical
    model: Nt is the MisTIC-weighted depth as an offset (theta = 1), m is the expected misassigned
    Tirosh count. It replaced the cell-area spline on 2026-10-13.
  - Weight: log1p PhiSpace own state, a contact (0-15 um) / regional (15-80 um) neighbourhood
    split, a rank-3 bilinear term, a shared density spline h and per-type density curves K.
    No oxygen.
  - Penalty: lambda 300.
  - Fit: `outputs/hpc/point_estimate.json`, the best of 20 restarts, Spartan job `multistart`
    (2026-10-12).
- **Production mixture (ovarian).** The same specification on 395,522 cells
  (`MIX_DATASET=ovarian`).
  - PhiSpace run B, typed by raw argmax, with Granulosa and Fallopian Tube Epithelium folded
    into Other.
  - Lambda and rank are kept from breast.
  - Fit: `outputs/hpc_ovarian/point_estimate.json` (2026-10-15).

"Matched NB" means an NB with the mixture's covariates in its mean. Two versions exist:
- `nb_matched` in `comparison*/` and `spatial/`: unpenalised, rank-2 bilinear term, refitted each
  time the script runs.
- NB-c / NB-d in `division_classifier*/`: the production penalty and rank 3, saved as
  `nb_params_nb{c,d}_full.json`. NB-d adds a dispersion submodel on the own state. NB-d has the
  best AIC of the two on both sections and is the "best NB" in `figures/model_fit/`.

## Folders

| Path | What it is | Model(s) | Written by | Current? |
| --- | --- | --- | --- | --- |
| `figures/model_fit/` | Rootogram, PIT, quantile residuals, fitted-mean and residual maps, metrics CSV | Best NB (NB-d), production mixture, NB + NB mixture (ovarian only) | `run/plot_model_fit.py` | 2026-10-18 |
| `figures/` (other) | Tissue and cell-type maps; prior pi and posterior w maps (`cycling_*`); density response; Stage B diagnostics; spatial residual maps | Production mixture, each section | `plot_tissue_maps.py`, `plot_density_response.py`, `diagnostics_report.py`, `estimands_report.py` | 2026-10-13 to 10-16 |
| `hpc/` | Spartan per-task JSON (breast): `multistart` (production), `bootstrap` (300), `cv_production` (5 spatial folds), `cv_penalty`, `cv_rank`, `theta_bracket`, `sim_single_component` (Test 03), `sim_single_component_path_full` (lambda path), `division_sim` (Stage 6), `division_sim_addendum` (A5), `division_sim_t10` | Production mixture and variants; division simulations | `hpc/worker.py` + `mixture/tasks.py` | All MisTIC, 2026-10-12 to 10-18 |
| `hpc_plain/` | `multistart`, `bootstrap`, `cv_production`, `sim_single_component` | Plain technical model: raw N offset, no m (the pre-MisTIC comparison) | same | 2026-10-13 |
| `hpc_ovarian/` | `multistart`, `bootstrap` | Ovarian production mixture | same | 2026-10-15 |
| `baseline/` | MisTIC against plain: held-out likelihood, estimands, paired differences | Production mixture vs `hpc_plain` | `run/baseline_comparison.py` | 2026-10-13 |
| `theta_bracket/` | Depth exponent pinned at 1 (production) against free | Production mixture, theta free | `run/theta_bracket_summary.py` | 2026-10-13 |
| `penalty/` | Lambda and rank re-checked under MisTIC (null check, held-out, effects by lambda) | Production mixture along lambda and rank | `run/lambda_sensitivity.py`, `run/lambda_path.py` | 2026-10-13 |
| `interaction/` | Data support for each own x neighbour entry against the fitted bilinear term | Production mixture (+ `cv_rank`, bootstrap) | `run/interaction_support.py` | 2026-10-13 |
| `comparison/` | NB against mixture on breast: metrics, rootograms, PIT, dispersion by type | `nb_prev` (earlier NB division model), `nb_matched` (unpenalised, rank 2), production mixture | `run/compare_nb_mixture.py` | 2026-10-13. **`run.log` is stale** (678,885-cell SpaNorm era); the CSVs and PNGs are current |
| `comparison_ovarian/` | The same on ovarian | same, ovarian production mixture | same, `MIX_DATASET=ovarian` | 2026-10-15 |
| `poisson_nb_ovarian/` | Poisson + NB mixture (cycling component NB) | Production spec + `cycling_nb` | `run/mixture_poisson_nb.py` | 2026-10-16; not adopted |
| `nb_nb_ovarian/` | NB + NB mixture (`params.json` is its fit) | Production spec + `cycling_nb` + `noncycling_nb` | `run/mixture_poisson_nb.py --family nbnb` | 2026-10-16; not adopted |
| `ovarian_depthres/` | State block residualised on depth within type, refitted | Ovarian production spec, residualised table | `run/ovarian_depth_residual_check.py` | 2026-10-15; sensitivity only |
| `spatial/` | PhiSpace and residual correlograms, LISA maps, spatial blocks | `nb_prev`, `nb_matched` (refitted, saved in `nb_fits.npz`), production mixture | `run/spatial_analysis.py` | 2026-10-13 |
| `nb_baseline/` | NB baseline development: neighbourhood models, depth offset tests, oxygen field and sweeps, tuning | Single-NB division models (the NB baseline spec) | `notebooks/nb_baseline.ipynb`, `src/tissue_dynamics/nb_baseline.py` | Rerun on MisTIC 2026-10-13 |
| `mistic/` | MisTIC pipeline and tests T1-T6 (breast); README on the probability column | MisTIC per-transcript probabilities; T6 compares old and new mixtures | `run/mistic_*.py` | 2026-10-13 |
| `mistic_ovarian/` | Covariate summary only | MisTIC on ovarian | `run/mistic_covariates.py` | 2026-10-15 |
| `portrait/` | Phase portrait: archetype interaction matrix, velocities, fixed points, bifurcation; `osdr_p*` = OSDR density-correction variants (power 2, 4, 8, 16) | Production mixture (breast) | `run/portrait_*.py` | `FINDINGS.md` and figures rewritten 2026-10-13 for production. **`run.log` is stale** (678,885-cell fit) |
| `generated/` | Density-response values behind the figure | Production mixture | `run/plot_density_response.py` | 2026-10-13 |
| `division_classifier/` | Division classifier comparison, Stages 1-7 and addendum 1 (A1-A6); `trunc_calibration/` (G4 calibration), `wu_calibration/` (Wu calibration); `report.md` is the run log | Labels: naive BASE-T (alpha_B 0.05, rate 3.3%), BASE-O (per type, Td 0.5), SCORE. Arms: NB-c, NB-d, MIX (production), LOG-T/O/S, TWO-P/PNB | `run/division_*.py` | 2026-10-13 to 10-17 |
| `division_classifier_t10/` | The same Stages 1-7 on the t10 label variant | BASE-T from the alpha_trim 0.2 truncated null at alpha_B 0.1085 (rate 10%); everything else as above | `DIV_VARIANT=t10 run/division_*.py` | 2026-10-18 |
| `division_classifier_ovarian/` | Stages 1-4 on ovarian | Ovarian labels and arms | `MIX_DATASET=ovarian run/division_*.py` | 2026-10-16 |

## Top-level files

| File | What it is | Current? |
| --- | --- | --- |
| `data_report.md`, `data_report_ovarian.md` | Table build and data contract for each section | 2026-10-06 (breast, log1p PhiSpace), 2026-10-15 (ovarian) |
| `diagnostics_stage_b.md` | Tests 09 and 14 on the breast production fit | 2026-10-13 |
| `estimands_8_2.md`, `estimands_8_2_ovarian.md` | Archetype-conditional neighbourhood effects with bootstrap intervals | 2026-10-13, 2026-10-16 |
| `validation_report.md` | Spec tests 01-08, appended per run. It holds several vintages (area era and MisTIC); read the section nearest the end for each test | Mixed; last 2026-10-12 |

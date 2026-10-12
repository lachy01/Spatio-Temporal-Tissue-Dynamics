# Stage B diagnostics (spec 6.9, 6.14, 8.7, 8.8)

Computed on the fitted point estimate, 561,897 cells.

## Test 09 — marginal fit

RMS hanging discrepancy over counts 0-20: single component
**22.1** -> mixture **4.1**
(-81.3%).

Counts 1-2, which spec 6.9 singles out: -30.8 / -38.3
-> -9.3 / -9.4.

Randomised PIT max deviation from uniform: **0.0694**.

### By archetype

| archetype         |      n |   single |   mixture |   change_% |
|:------------------|-------:|---------:|----------:|-----------:|
| Cancer Epithelial | 110805 |    13.6  |      2.11 |     -84.5  |
| T-cells           |  89213 |     5.7  |      3.15 |     -44.77 |
| Myeloid           |  68186 |     9.43 |      3.05 |     -67.67 |
| B-cells           |  57825 |     3.97 |      1.82 |     -54.22 |
| Normal Epithelial |  53149 |     6.84 |      1.38 |     -79.89 |
| CAFs              |  52506 |    10.48 |      3.97 |     -62.11 |
| Endothelial       |  47810 |     8.96 |      3.48 |     -61.22 |
| Plasmablasts      |  42228 |     5.1  |      0.84 |     -83.44 |
| PVL               |  40175 |     6.63 |      2.15 |     -67.65 |

## Test 14 — residual spatial autocorrelation

Moran's I on mixture Pearson residuals (8-NN, 120k subsample):
**0.0056** (p = 0.010).

| distance (um) | Moran's I | pairs |
|---:|---:|---:|
| 25 | +0.0068 | 214,158 |
| 75 | +0.0019 | 596,733 |
| 125 | +0.0031 | 939,710 |
| 175 | +0.0026 | 1,257,611 |
| 225 | +0.0022 | 1,569,263 |
| 275 | +0.0012 | 1,862,414 |
| 325 | +0.0011 | 2,146,401 |
| 375 | +0.0006 | 2,416,529 |
| 425 | +0.0024 | 2,680,089 |
| 475 | +0.0019 | 2,937,661 |

## Spec 8.7 / 8.8 (mandatory with every reported fit)

- normalised entropy **0.7632** (floor 0.30)
- confound R^2: nuisance **0.0477**, state
  **0.0950**, ratio
  **1.99x**
- size/detection classifier: **False**

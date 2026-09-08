# Stage B diagnostics (spec 6.9, 6.14, 8.7, 8.8)

Computed on the fitted point estimate, 678,885 cells.

## Test 09 — marginal fit

RMS hanging discrepancy over counts 0-20: single component
**13.5** -> mixture **2.4**
(-82.6%).

Counts 1-2, which spec 6.9 singles out: -10.1 / -18.0
-> -3.2 / -3.9.

Randomised PIT max deviation from uniform: **0.0389**.

### By archetype

| archetype         |      n |   single |   mixture |   change_% |
|:------------------|-------:|---------:|----------:|-----------:|
| Cancer Epithelial | 129033 |    12.1  |      2.78 |     -77    |
| Plasmablasts      | 101340 |    12.34 |      4.31 |     -65.08 |
| T-cells           |  97783 |     2.38 |      3.47 |      46.26 |
| Normal Epithelial |  95780 |     1.08 |      0.62 |     -42.63 |
| Endothelial       |  60732 |     1.09 |      0.25 |     -76.7  |
| B-cells           |  56237 |     1.28 |      2.17 |      69.46 |
| CAFs              |  52417 |     4.8  |      2.91 |     -39.37 |
| Myeloid           |  49284 |     2.8  |      1.91 |     -31.96 |
| PVL               |  36279 |     1.25 |      0.49 |     -61.12 |

## Test 14 — residual spatial autocorrelation

Moran's I on mixture Pearson residuals (8-NN, 120k subsample):
**0.0142** (p = 0.010).

| distance (um) | Moran's I | pairs |
|---:|---:|---:|
| 25 | +0.0079 | 192,408 |
| 75 | +0.0052 | 537,992 |
| 125 | +0.0022 | 851,073 |
| 175 | +0.0017 | 1,146,632 |
| 225 | +0.0018 | 1,430,520 |
| 275 | +0.0005 | 1,703,006 |
| 325 | +0.0009 | 1,968,218 |
| 375 | +0.0010 | 2,221,881 |
| 425 | +0.0013 | 2,469,261 |
| 475 | +0.0003 | 2,714,118 |

## Spec 8.7 / 8.8 (mandatory with every reported fit)

- normalised entropy **0.6832** (floor 0.30)
- confound R^2: nuisance **0.0229**, state
  **0.0921**, ratio
  **4.02x**
- size/detection classifier: **False**

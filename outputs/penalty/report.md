# Penalty and rank under the production model

`technical: mistic`, `weight_ridge` 300, rank 3. Both were chosen on the area model (2026-10-07); this re-checks them on the model in force and measures what the penalty changes. Fits: `outputs/hpc`; assembled 2026-10-13 14:24.

## Null check (the selection rule's gate)

Data with no class structure at the real n, fitted along a lambda path. A replicate passes Test 03's criterion if the mixture collapses to one component or its two components barely differ and cells are not split between them (rate ratio < 1.5 and normalised entropy < 0.10).

| source | lambda | replicates | pass | collapsed | largest entropy, two components | entropies, two components |
|---|---|---|---|---|---|---|
| lambda path | 30 | 4 | 0 | 0 | 0.848 | 0.524, 0.591, 0.653, 0.848 |
| lambda path | 100 | 4 | 4 | 0 | 0.092 | 0.026, 0.029, 0.048, 0.092 |
| lambda path | 300 | 4 | 4 | 0 | 0.019 | 0.012, 0.015, 0.019, 0.019 |
| lambda path | 1000 | 4 | 4 | 1 | 0.008 | 0.005, 0.006, 0.008 |
| lambda path | 3000 | 4 | 4 | 2 | 0.004 | 0.004, 0.004 |
| Test 03 | 300 | 20 | 20 | 0 | 0.070 | median 0.016 |

Smallest lambda on the path at which every replicate passes: **100**. Test 03 at the production lambda: 20 of 20, largest entropy 0.070.

## Held-out log-likelihood along the lambda grid

cv_penalty: 5 spatial folds, summed over 250 spatial blocks, each lambda paired with the best (0.1) block by block.

| lambda | held-out total | minus best | SE | within 2 SE of best |
|---|---|---|---|---|
| 0 | -600,178.8 | -1.1 | 1.5 | yes |
| 0.1 | -600,177.8 | 0.0 | 0.0 | best |
| 0.3 | -600,178.2 | -0.4 | 0.9 | yes |
| 1 | -600,179.4 | -1.7 | 1.8 | yes |
| 3 | -600,180.3 | -2.6 | 2.2 | yes |
| 10 | -600,180.5 | -2.7 | 2.5 | yes |
| 30 | -600,179.7 | -2.0 | 2.7 | yes |
| 100 | -600,180.2 | -2.5 | 4.5 | yes |
| 300 | -600,201.6 | -23.8 | 10.6 | no |
| 1000 | -600,362.7 | -185.0 | 26.7 | no |
| 3000 | -600,900.0 | -722.3 | 55.2 | no |
| 10000 | -602,177.0 | -1,999.2 | 103.1 | no |

lambda 100 minus lambda 300: **+21.3 +/- 6.6** (+3.8e-05 per cell, over 561,897 cells).

## Rank

cv_rank at lambda 300, paired by spatial block.

| rank | held-out total | gain over the rank below | SE |
|---|---|---|---|
| 1 | -600,452.3 | nan | nan |
| 2 | -600,275.8 | 176.5 | 31.8 |
| 3 | -600,201.6 | 74.3 | 14.9 |
| 5 | -600,182.1 | 19.5 | 8.9 |

Rank 3 is 19.5 +/- 8.9 short of the best-scoring rank 5.

## What lambda changes: 100 against 300

Odds ratio of cycling per IQR of each priority-pair neighbourhood component. The full fit carries its 95% spatial block bootstrap interval. The fold columns average the five cv_penalty fold fits at each lambda; the shift is the within-fold difference in log odds ratio, 100 minus 300, over the bootstrap SD.

| own archetype | neighbour component | lambda 300, full fit | lambda 300, folds | lambda 100, folds | shift (bootstrap SEs) |
|---|---|---|---|---|---|
| CAFs | contact: Myeloid | 0.955 (0.903, 1.002) | 0.952 | 0.942 | -0.42 |
| CAFs | regional: Myeloid | 1.151 (1.048, 1.247) | 1.146 | 1.150 | +0.07 |
| Cancer Epithelial | contact: CAFs | 0.947 (0.914, 0.977) | 0.947 | 0.949 | +0.09 |
| Cancer Epithelial | regional: CAFs | 0.991 (0.927, 1.060) | 0.990 | 0.991 | +0.01 |
| Cancer Epithelial | contact: Myeloid | 0.931 (0.899, 0.970) | 0.931 | 0.928 | -0.19 |
| Cancer Epithelial | regional: Myeloid | 1.078 (0.997, 1.154) | 1.078 | 1.078 | -0.01 |
| Cancer Epithelial | contact: T-cells | 1.319 (1.240, 1.391) | 1.319 | 1.338 | +0.48 |
| Cancer Epithelial | regional: T-cells | 1.079 (0.981, 1.190) | 1.079 | 1.076 | -0.04 |
| Myeloid | contact: CAFs | 0.953 (0.912, 0.987) | 0.953 | 0.946 | -0.35 |
| Myeloid | regional: CAFs | 1.005 (0.942, 1.085) | 1.005 | 1.010 | +0.15 |
| CAFs | contact: Cancer Epithelial | 1.837 (1.690, 2.008) | 1.834 | 1.936 | +1.30 |
| CAFs | regional: Cancer Epithelial | 1.443 (1.213, 1.739) | 1.433 | 1.387 | -0.33 |
| Myeloid | contact: Cancer Epithelial | 2.052 (1.893, 2.248) | 2.048 | 2.178 | +1.33 |
| Myeloid | regional: Cancer Epithelial | 1.020 (0.848, 1.268) | 1.023 | 0.992 | -0.31 |
| T-cells | contact: Cancer Epithelial | 2.208 (2.036, 2.419) | 2.213 | 2.403 | +1.87 |
| T-cells | regional: Cancer Epithelial | 1.257 (1.009, 1.495) | 1.256 | 1.198 | -0.46 |

- At lambda 300 the fold means reproduce the full fit to within 0.6%, so the fold values stand in for full fits at other lambdas.
- 13 of 16 effects move by less than 0.5 SE (median |shift| 0.32); the largest moves 1.87 SE (T-cells <- contact: Cancer Epithelial).
- Effects with |log OR| > 0.2 at lambda 300: log OR at 100 over 300 has median 1.069 (range 0.792 to 1.104).
- The shift is systematic, not noise: its fold-to-fold SD is at most 0.17 SE.
- Intervals that would change side of OR = 1 at lambda 100 (normal intervals from the bootstrap SE): CAFs <- contact: Myeloid, T-cells <- regional: Cancer Epithelial.

Null check at lambda 100: 4 of 4 pass, entropies 0.026, 0.029, 0.048, 0.092.

## The path

Fold-mean odds ratios at a selection of grid values (all of them in `effects_by_lambda.csv`; figure `effects_by_lambda.png`).

| effect | 0 | 10 | 30 | 100 | 300 | 1000 | 3000 |
|---|---|---|---|---|---|---|---|
| CAFs <- contact: Myeloid | 0.937 | 0.936 | 0.937 | 0.942 | 0.952 | 0.971 | 0.979 |
| CAFs <- regional: Myeloid | 1.155 | 1.152 | 1.151 | 1.150 | 1.146 | 1.130 | 1.076 |
| Cancer Epithelial <- contact: CAFs | 0.950 | 0.950 | 0.950 | 0.949 | 0.947 | 0.949 | 0.965 |
| Cancer Epithelial <- regional: CAFs | 0.993 | 0.991 | 0.991 | 0.991 | 0.990 | 0.990 | 0.989 |
| Cancer Epithelial <- contact: Myeloid | 0.926 | 0.926 | 0.926 | 0.928 | 0.931 | 0.940 | 0.956 |
| Cancer Epithelial <- regional: Myeloid | 1.079 | 1.078 | 1.078 | 1.078 | 1.078 | 1.080 | 1.082 |
| Cancer Epithelial <- contact: T-cells | 1.348 | 1.347 | 1.345 | 1.338 | 1.319 | 1.266 | 1.174 |
| Cancer Epithelial <- regional: T-cells | 1.073 | 1.074 | 1.075 | 1.076 | 1.079 | 1.076 | 1.049 |
| Myeloid <- contact: CAFs | 0.942 | 0.942 | 0.943 | 0.946 | 0.953 | 0.965 | 0.972 |
| Myeloid <- regional: CAFs | 1.029 | 1.015 | 1.013 | 1.010 | 1.005 | 0.997 | 0.990 |
| CAFs <- contact: Cancer Epithelial | 2.011 | 2.005 | 1.988 | 1.936 | 1.834 | 1.655 | 1.456 |
| CAFs <- regional: Cancer Epithelial | 1.354 | 1.363 | 1.368 | 1.387 | 1.433 | 1.524 | 1.541 |
| Myeloid <- contact: Cancer Epithelial | 2.256 | 2.258 | 2.239 | 2.178 | 2.048 | 1.802 | 1.539 |
| Myeloid <- regional: Cancer Epithelial | 0.984 | 0.975 | 0.979 | 0.992 | 1.023 | 1.088 | 1.159 |
| T-cells <- contact: Cancer Epithelial | 2.539 | 2.522 | 2.493 | 2.403 | 2.213 | 1.871 | 1.536 |
| T-cells <- regional: Cancer Epithelial | 1.165 | 1.160 | 1.169 | 1.198 | 1.256 | 1.333 | 1.310 |


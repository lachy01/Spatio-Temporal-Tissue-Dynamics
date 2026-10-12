# The MisTIC model against the plain baseline

Two mixtures of the same counts on the same cells, identical except for the technical covariates: **MisTIC** (offset log Nt, the MisTIC-weighted library size, plus theta_m m in every component's mean) and **plain** (offset log N, the raw library size, nothing else). Neither has an area term.

_Fits: MisTIC `outputs/hpc`, plain `outputs/hpc_plain`; assembled 2026-10-13 10:11._

## Held-out log-likelihood (cv_production: 5 spatial folds, production penalty)

_Pending: job cv_production for both models (found 0 and 0 of 5 folds)._

## In sample, at each point estimate

_Pending: point estimates found for ['MisTIC']._

## Estimands, with 95% spatial block bootstrap intervals

| quantity | MisTIC | plain |
|---|---|---|
| e^delta | 1.9020 (1.8831, 1.9251) | pending |
| theta_m | 1.5938 (1.5329, 1.6613) | pending |
| cycling fraction, mean prior pi | 0.0982 | pending |
| cycling fraction, mean posterior w | 0.0982 (0.0934, 0.1029) | pending |
| rate ratio exp(e^delta) | 6.699 | pending |

Odds ratio of cycling per IQR of each neighbourhood component, priority pairs both ways round:

| own archetype | neighbour component | MisTIC | plain |
|---|---|---|---|
| CAFs | contact: Myeloid | 0.9545 (0.9029, 1.0021) | pending |
| CAFs | regional: Myeloid | 1.1510 (1.0478, 1.2466) | pending |
| Cancer Epithelial | contact: CAFs | 0.9474 (0.9145, 0.9768) | pending |
| Cancer Epithelial | regional: CAFs | 0.9912 (0.9270, 1.0603) | pending |
| Cancer Epithelial | contact: Myeloid | 0.9313 (0.8988, 0.9699) | pending |
| Cancer Epithelial | regional: Myeloid | 1.0784 (0.9970, 1.1536) | pending |
| Cancer Epithelial | contact: T-cells | 1.3187 (1.2396, 1.3908) | pending |
| Cancer Epithelial | regional: T-cells | 1.0788 (0.9813, 1.1896) | pending |
| Myeloid | contact: CAFs | 0.9530 (0.9122, 0.9867) | pending |
| Myeloid | regional: CAFs | 1.0049 (0.9425, 1.0847) | pending |
| CAFs | contact: Cancer Epithelial | 1.8371 (1.6904, 2.0085) | pending |
| CAFs | regional: Cancer Epithelial | 1.4425 (1.2132, 1.7386) | pending |
| Myeloid | contact: Cancer Epithelial | 2.0516 (1.8934, 2.2485) | pending |
| Myeloid | regional: Cancer Epithelial | 1.0199 (0.8484, 1.2684) | pending |
| T-cells | contact: Cancer Epithelial | 2.2075 (2.0360, 2.4193) | pending |
| T-cells | regional: Cancer Epithelial | 1.2572 (1.0089, 1.4948) | pending |

## Division labels: call rate per archetype under each null

Plain null: NB, mean N e^b0 (alpha0 0.8641). MisTIC null: mean Nt e^b0 + theta m (theta 2.8185, alpha0 0.7889). BASE-T at alpha_B 0.05, BASE-O at Td 0.5.

| archetype | cells | BASE-T, MisTIC | BASE-T, plain | BASE-O, MisTIC | BASE-O, plain |
|---|---|---|---|---|---|
| all | 561897 | 0.0329 | 0.0344 | 0.1818 | 0.1789 |
| Endothelial | 48729 | 0.0107 | 0.0107 | 0.1242 | 0.1177 |
| CAFs | 52079 | 0.0087 | 0.0082 | 0.1318 | 0.1230 |
| PVL | 42834 | 0.0146 | 0.0173 | 0.1462 | 0.1432 |
| B-cells | 59119 | 0.0198 | 0.0201 | 0.2018 | 0.1997 |
| T-cells | 87004 | 0.0273 | 0.0264 | 0.2286 | 0.2227 |
| Myeloid | 69669 | 0.0129 | 0.0128 | 0.1367 | 0.1329 |
| Normal Epithelial | 54508 | 0.0264 | 0.0339 | 0.1632 | 0.1688 |
| Plasmablasts | 42523 | 0.0329 | 0.0410 | 0.1906 | 0.1968 |
| Cancer Epithelial | 105432 | 0.0912 | 0.0916 | 0.2338 | 0.2301 |

The nulls' calibration and spillover tests run both forms: `outputs/mistic/t3.md` (MisTIC) and `t3_plain.md` (plain) on 20 matched random gene sets; `outputs/mistic/t4.md` sets the plain null's spillover curve beside the MisTIC null's.


# The depth exponent: theta pinned at 1 (F) and free (V)

Mixture under `technical: mistic`, lambda 300, rank 3. F is production; V frees theta, the exponent on the library size. Spec 9.11: the two are a bracket, reported side by side, and are not chosen between on fit.

_Fits: `outputs/hpc/theta_bracket` (V) and `outputs/hpc/point_estimate.json` with its bootstrap (F); assembled 2026-10-13 13:57._

## Restarts

Restarts at the best optimum: F 4 of 5, V 4 of 5. The bracket's best F is +0.00 from the production point estimate (the same optimum).

| arm | restart | theta | penalised objective | e^delta | theta_m | mean w | entropy | grad_inf | converged |
|---|---|---|---|---|---|---|---|---|---|
| F | 0 | 1.0000 | -601,827.8638 | 1.9108 | 1.6765 | 0.0949 | 0.7663 | 0.0000 | True |
| F | 1 | 1.0000 | -600,332.2563 | 1.9020 | 1.5938 | 0.0982 | 0.7632 | 0.0000 | True |
| F | 2 | 1.0000 | -600,332.2563 | 1.9020 | 1.5938 | 0.0982 | 0.7632 | 0.0000 | True |
| F | 3 | 1.0000 | -600,332.2563 | 1.9020 | 1.5938 | 0.0982 | 0.7632 | 0.0000 | True |
| F | 4 | 1.0000 | -600,332.2563 | 1.9020 | 1.5938 | 0.0982 | 0.7632 | 0.0000 | True |
| V | 0 | 1.1058 | -600,973.3702 | 1.8654 | 1.6784 | 0.1232 | 0.6880 | 0.0001 | True |
| V | 1 | 1.1194 | -599,250.7674 | 1.8516 | 1.5843 | 0.1330 | 0.6721 | 0.0001 | True |
| V | 2 | 1.1194 | -599,250.7674 | 1.8516 | 1.5843 | 0.1330 | 0.6721 | 0.0000 | True |
| V | 3 | 1.1194 | -599,250.7674 | 1.8516 | 1.5843 | 0.1330 | 0.6721 | 0.0000 | True |
| V | 4 | 1.1194 | -599,250.7674 | 1.8516 | 1.5843 | 0.1330 | 0.6721 | 0.0000 | True |

## Estimands

F with its 95% spatial block bootstrap interval; V is a point estimate. 'In F SEs' is the shift over the standard error implied by F's interval.

| quantity | F (theta = 1) | V (theta free) | V - F | in F SEs |
|---|---|---|---|---|
| theta (depth exponent) | 1 (fixed) | 1.1194 | 0.1194 | nan |
| e^delta | 1.9020 (1.8831, 1.9251) | 1.8516 | -0.0504 | -4.7147 |
| theta_m | 1.5938 (1.5329, 1.6613) | 1.5843 | -0.0095 | -0.2906 |
| cycling fraction, mean prior pi | 0.0982 | 0.1330 | 0.0348 | nan |
| cycling fraction, mean posterior w | 0.0982 (0.0934, 0.1029) | 0.1330 | 0.0348 | 14.3269 |
| rate ratio exp(e^delta) | 6.699 | 6.370 | -0.3296 | nan |

Data log-likelihood: V minus F **+1,067.7** for one extra parameter.

Odds ratio of cycling per IQR of each neighbourhood component, priority pairs both ways round. The shift is on the log scale.

| own archetype | neighbour component | F (theta = 1) | V (theta free) | shift in F SEs | V inside F's 95% CI |
|---|---|---|---|---|---|
| CAFs | contact: Myeloid | 0.9545 (0.9029, 1.0021) | 0.9534 | -0.0445 | yes |
| CAFs | regional: Myeloid | 1.1510 (1.0478, 1.2466) | 1.1036 | -0.9499 | yes |
| Cancer Epithelial | contact: CAFs | 0.9474 (0.9145, 0.9768) | 0.9515 | 0.2541 | yes |
| Cancer Epithelial | regional: CAFs | 0.9912 (0.9270, 1.0603) | 0.9832 | -0.2374 | yes |
| Cancer Epithelial | contact: Myeloid | 0.9313 (0.8988, 0.9699) | 0.9480 | 0.9159 | yes |
| Cancer Epithelial | regional: Myeloid | 1.0784 (0.9970, 1.1536) | 1.0836 | 0.1285 | yes |
| Cancer Epithelial | contact: T-cells | 1.3187 (1.2396, 1.3908) | 1.3447 | 0.6656 | yes |
| Cancer Epithelial | regional: T-cells | 1.0788 (0.9813, 1.1896) | 0.9546 | -2.4904 | **no** |
| Myeloid | contact: CAFs | 0.9530 (0.9122, 0.9867) | 0.9541 | 0.0603 | yes |
| Myeloid | regional: CAFs | 1.0049 (0.9425, 1.0847) | 0.9944 | -0.2939 | yes |
| CAFs | contact: Cancer Epithelial | 1.8371 (1.6904, 2.0085) | 1.7253 | -1.4268 | yes |
| CAFs | regional: Cancer Epithelial | 1.4425 (1.2132, 1.7386) | 1.2837 | -1.2707 | yes |
| Myeloid | contact: Cancer Epithelial | 2.0516 (1.8934, 2.2485) | 1.9513 | -1.1430 | yes |
| Myeloid | regional: Cancer Epithelial | 1.0199 (0.8484, 1.2684) | 0.9224 | -0.9801 | yes |
| T-cells | contact: Cancer Epithelial | 2.2075 (2.0360, 2.4193) | 2.1221 | -0.8964 | yes |
| T-cells | regional: Cancer Epithelial | 1.2572 (1.0089, 1.4948) | 1.1364 | -1.0064 | yes |

Largest shift 2.49 F SEs; median 0.91; 15 of 16 V estimates inside F's interval.

## Cycling fraction by cell type (mean prior pi, point estimates)

| archetype | mean pi, F | mean pi, V | V / F |
|---|---|---|---|
| all | 0.0982 | 0.1330 | 1.3538 |
| Endothelial | 0.0397 | 0.0610 | 1.5363 |
| CAFs | 0.0236 | 0.0375 | 1.5938 |
| PVL | 0.0594 | 0.0880 | 1.4807 |
| B-cells | 0.0897 | 0.1534 | 1.7103 |
| T-cells | 0.1085 | 0.1782 | 1.6425 |
| Myeloid | 0.0474 | 0.0725 | 1.5283 |
| Normal Epithelial | 0.0972 | 0.1207 | 1.2420 |
| Plasmablasts | 0.1148 | 0.1489 | 1.2976 |
| Cancer Epithelial | 0.1947 | 0.2157 | 1.1080 |


# The MisTIC model against the plain baseline

Two mixtures of the same counts on the same cells, identical except for the technical covariates: **MisTIC** (offset log Nt, the MisTIC-weighted library size, plus theta_m m in every component's mean) and **plain** (offset log N, the raw library size, nothing else). Neither has an area term.

_Fits: MisTIC `outputs/hpc`, plain `outputs/hpc_plain`; assembled 2026-10-13 15:03._

## Held-out log-likelihood (cv_production: 5 spatial folds, production penalty)

| fold | n_test | heldout MisTIC | heldout plain | MisTIC - plain |
|---|---|---|---|---|
| 0.0000 | 118,298.0000 | -126,336.8804 | -127,165.8159 | 828.9355 |
| 1.0000 | 105,904.0000 | -118,158.3146 | -118,848.8051 | 690.4905 |
| 2.0000 | 117,639.0000 | -120,814.7219 | -121,651.8091 | 837.0872 |
| 3.0000 | 112,102.0000 | -112,240.9867 | -112,936.9727 | 695.9860 |
| 4.0000 | 107,954.0000 | -122,650.6685 | -123,518.5553 | 867.8868 |

Total over folds: **+3,920.4** log-likelihood in favour of MisTIC; better on 5 of 5 folds. Paired by spatial block (250 blocks): mean +15.68 per block, SE 0.76, so the total is +3,920.4 +/- 190.3.

## In sample, at each point estimate

| model | data log-likelihood | penalised objective | parameters | AIC | BIC | restart agreement |
|---|---|---|---|---|---|---|
| MisTIC | -600,031.6135 | -600,332.2563 | 145 | 1,200,353.2269 | 1,201,982.8926 | 0.9500 |
| plain | -603,938.3769 | -604,262.8895 | 144 | 1,208,164.7537 | 1,209,783.1804 | 0.9500 |

MisTIC minus plain: data log-likelihood **+3,906.8** with +1 parameters. AIC and BIC use the nominal parameter count; the ridge shrinks the weight model, so the effective count is smaller and these understate both models equally.

## Estimands, with 95% spatial block bootstrap intervals

| quantity | MisTIC | plain |
|---|---|---|
| e^delta | 1.9020 (1.8831, 1.9251) | 1.7572 (1.7397, 1.7777) |
| theta_m | 1.5938 (1.5329, 1.6613) | -- |
| cycling fraction, mean prior pi | 0.0982 | 0.1011 |
| cycling fraction, mean posterior w | 0.0982 (0.0934, 0.1029) | 0.1011 (0.0963, 0.1059) |
| rate ratio exp(e^delta) | 6.699 | 5.796 |

Odds ratio of cycling per IQR of each neighbourhood component, priority pairs both ways round:

| own archetype | neighbour component | MisTIC | plain |
|---|---|---|---|
| CAFs | contact: Myeloid | 0.9545 (0.9029, 1.0021) | 0.9308 (0.8832, 0.9742) |
| CAFs | regional: Myeloid | 1.1510 (1.0478, 1.2466) | 1.1519 (1.0416, 1.2566) |
| Cancer Epithelial | contact: CAFs | 0.9474 (0.9145, 0.9768) | 0.9336 (0.8993, 0.9641) |
| Cancer Epithelial | regional: CAFs | 0.9912 (0.9270, 1.0603) | 0.9919 (0.9185, 1.0686) |
| Cancer Epithelial | contact: Myeloid | 0.9313 (0.8988, 0.9699) | 0.9294 (0.8956, 0.9654) |
| Cancer Epithelial | regional: Myeloid | 1.0784 (0.9970, 1.1536) | 1.0839 (0.9995, 1.1654) |
| Cancer Epithelial | contact: T-cells | 1.3187 (1.2396, 1.3908) | 1.3454 (1.2643, 1.4227) |
| Cancer Epithelial | regional: T-cells | 1.0788 (0.9813, 1.1896) | 1.0810 (0.9697, 1.2100) |
| Myeloid | contact: CAFs | 0.9530 (0.9122, 0.9867) | 0.9312 (0.8891, 0.9663) |
| Myeloid | regional: CAFs | 1.0049 (0.9425, 1.0847) | 1.0058 (0.9383, 1.0905) |
| CAFs | contact: Cancer Epithelial | 1.8371 (1.6904, 2.0085) | 1.9990 (1.8478, 2.1899) |
| CAFs | regional: Cancer Epithelial | 1.4425 (1.2132, 1.7386) | 1.5234 (1.2547, 1.8952) |
| Myeloid | contact: Cancer Epithelial | 2.0516 (1.8934, 2.2485) | 2.1427 (1.9836, 2.3550) |
| Myeloid | regional: Cancer Epithelial | 1.0199 (0.8484, 1.2684) | 1.1004 (0.9013, 1.3704) |
| T-cells | contact: Cancer Epithelial | 2.2075 (2.0360, 2.4193) | 2.3739 (2.1876, 2.6202) |
| T-cells | regional: Cancer Epithelial | 1.2572 (1.0089, 1.4948) | 1.3387 (1.0749, 1.6079) |

## MisTIC against plain, paired by bootstrap replicate

300 replicates. Replicate r draws the same spatial blocks in both bootstraps (seeded by r, same cells), so the difference is computed within each replicate; the intervals are its 2.5th and 97.5th percentiles. For the odds ratios the comparison is their ratio.

| quantity | estimate | 95% low | 95% high | interval excludes no difference |
|---|---|---|---|---|
| e^delta, MisTIC - plain | 0.1448 | 0.1362 | 0.1536 | yes |
| cycling fraction (mean w), MisTIC - plain | -0.0028 | -0.0036 | -0.0022 | yes |
| OR ratio MisTIC / plain: CAFs <- contact: Myeloid | 1.0254 | 1.0156 | 1.0358 | yes |
| OR ratio MisTIC / plain: CAFs <- regional: Myeloid | 0.9992 | 0.9805 | 1.0184 | no |
| OR ratio MisTIC / plain: Cancer Epithelial <- contact: CAFs | 1.0148 | 1.0089 | 1.0201 | yes |
| OR ratio MisTIC / plain: Cancer Epithelial <- regional: CAFs | 0.9993 | 0.9906 | 1.0090 | no |
| OR ratio MisTIC / plain: Cancer Epithelial <- contact: Myeloid | 1.0020 | 0.9971 | 1.0074 | no |
| OR ratio MisTIC / plain: Cancer Epithelial <- regional: Myeloid | 0.9949 | 0.9876 | 1.0025 | no |
| OR ratio MisTIC / plain: Cancer Epithelial <- contact: T-cells | 0.9802 | 0.9720 | 0.9880 | yes |
| OR ratio MisTIC / plain: Cancer Epithelial <- regional: T-cells | 0.9979 | 0.9806 | 1.0147 | no |
| OR ratio MisTIC / plain: Myeloid <- contact: CAFs | 1.0233 | 1.0150 | 1.0329 | yes |
| OR ratio MisTIC / plain: Myeloid <- regional: CAFs | 0.9991 | 0.9857 | 1.0116 | no |
| OR ratio MisTIC / plain: CAFs <- contact: Cancer Epithelial | 0.9190 | 0.8987 | 0.9393 | yes |
| OR ratio MisTIC / plain: CAFs <- regional: Cancer Epithelial | 0.9469 | 0.9049 | 0.9831 | yes |
| OR ratio MisTIC / plain: Myeloid <- contact: Cancer Epithelial | 0.9575 | 0.9415 | 0.9734 | yes |
| OR ratio MisTIC / plain: Myeloid <- regional: Cancer Epithelial | 0.9268 | 0.8922 | 0.9539 | yes |
| OR ratio MisTIC / plain: T-cells <- contact: Cancer Epithelial | 0.9299 | 0.9170 | 0.9456 | yes |
| OR ratio MisTIC / plain: T-cells <- regional: Cancer Epithelial | 0.9391 | 0.9097 | 0.9721 | yes |

12 of 18 intervals exclude no difference.

## Cycling fraction by cell type (mean prior pi, point estimates)

| archetype | mean pi, MisTIC | mean pi, plain | MisTIC / plain |
|---|---|---|---|
| all | 0.0982 | 0.1011 | 0.9719 |
| Endothelial | 0.0397 | 0.0397 | 0.9993 |
| CAFs | 0.0236 | 0.0236 | 0.9982 |
| PVL | 0.0594 | 0.0614 | 0.9672 |
| B-cells | 0.0897 | 0.0888 | 1.0105 |
| T-cells | 0.1085 | 0.1074 | 1.0107 |
| Myeloid | 0.0474 | 0.0483 | 0.9821 |
| Normal Epithelial | 0.0972 | 0.1041 | 0.9336 |
| Plasmablasts | 0.1148 | 0.1228 | 0.9345 |
| Cancer Epithelial | 0.1947 | 0.2028 | 0.9599 |

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


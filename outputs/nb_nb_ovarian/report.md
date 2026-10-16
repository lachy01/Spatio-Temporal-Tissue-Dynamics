# NB + NB mixture (ovarian section, 2026-10-16 11:46)

`run/mixture_poisson_nb.py --family nbnb`. The production mixture with both components NB2, each with its own unpenalised dispersion (alpha_0 non-cycling, alpha_1 cycling; `cycling_nb`, `noncycling_nb`); everything
else as production. All 395,522 cells. Best of 6 fits: multistart 1; 83% of fits
within 1e-6 (relative) of the best penalised log-likelihood.

| start | converged | penalised loglik | grad_inf | iterations | alpha1 | alpha0 | minutes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| warm from Poisson + NB, alpha0 start 0.02 | True | -703047 | 2.56927e-05 | 1410 | 0.343358 | 0.132928 | 1.77045 |
| warm from Poisson + NB, alpha0 start 0.1 | True | -703047 | 6.07816e-06 | 871 | 0.343358 | 0.132928 | 1.29881 |
| warm from Poisson + NB, alpha0 start 0.3 | True | -703047 | 2.78518e-05 | 1137 | 0.343358 | 0.132928 | 1.56779 |
| multistart 0 | True | -705026 | 2.70321e-05 | 220 | 0.341485 | 0.139588 | 0.728265 |
| multistart 1 | True | -703047 | 1.70807e-05 | 1350 | 0.343358 | 0.132928 | 1.72289 |
| multistart 2 | True | -703047 | 1.74707e-05 | 1519 | 0.343358 | 0.132928 | 1.87053 |

## Fit

Likelihood-ratio statistic for alpha_0, 2 x (penalised loglik
NBNB - PNB) = **1,391.5** on one parameter (boundary at 0, so no p-value). NB-c is the division comparison's full-data matched NB
(MIX's covariates and penalty in an NB mean, one dispersion).

| model | parameters | log-likelihood (data) | AIC | rootogram misfit | zeros expected | zeros observed | PIT max deviation | e^delta | rate ratio | alpha0 | alpha1 | theta_m | cycling fraction (mean w) | mean pi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Poisson mixture | 145 | -7.42e+05 | 1.484e+06 | 3060 | 0.3395 | 0.4064 | 0.2896 | 1.745 | 5.725 | nan | nan | 1.399 | 0.181 | 0.1881 |
| Poisson + NB mixture | 146 | -7.03e+05 | 1.406e+06 | 885.8 | 0.3863 | 0.4064 | 0.07238 | 1.741 | 5.704 | nan | 0.3544 | 1.487 | 0.3212 | 0.3212 |
| NB + NB mixture | 147 | -7.022e+05 | 1.405e+06 | 735.9 | 0.3917 | 0.4064 | 0.08098 | 1.732 | 5.654 | 0.1329 | 0.3434 | 1.828 | 0.3037 | 0.3037 |
| NB-c (matched NB) | 144 | -7.044e+05 | 1.409e+06 | 1286 | 0.4132 | 0.4064 | 0.107 | nan | nan | nan | nan | nan | nan | nan |

Rootograms: `rootograms.png` (0-29 and a pooled 30+).

## By type

Misfit: rootogram misfit within the type. w: mean posterior cycling probability. Neighbourhood
contrast: mean logit pi in the type's top decile of neighbourhood Tumour score minus its bottom decile.

| type | cells | misfit P | w P | nbr contrast P | misfit PNB | w PNB | nbr contrast PNB | misfit NBNB | w NBNB | nbr contrast NBNB | misfit NB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tumour | 181539 | 817.029 | 0.362 | 2.664 | 987.353 | 0.625 | 3.134 | 1128.657 | 0.602 | 3.401 | 1353.196 |
| Fibroblasts | 86585 | 2141.734 | 0.022 | 2.821 | 503.376 | 0.057 | 3.294 | 454.718 | 0.042 | 3.207 | 104.520 |
| Endothelial Cells | 21595 | 586.330 | 0.018 | 1.101 | 112.226 | 0.041 | 2.855 | 116.885 | 0.032 | 2.943 | 44.753 |
| Pericytes | 13280 | 284.535 | 0.035 | 0.448 | 51.387 | 0.067 | 3.354 | 56.676 | 0.056 | 3.483 | 42.257 |
| Smooth Muscle Cells | 47877 | 1130.237 | 0.003 | -1.339 | 133.612 | 0.005 | 0.966 | 166.466 | 0.004 | 0.947 | 40.928 |
| Macrophages | 32672 | 371.572 | 0.067 | 2.524 | 67.102 | 0.148 | 4.210 | 68.191 | 0.124 | 4.180 | 61.229 |
| T & NK Cells | 8421 | 33.295 | 0.082 | 1.957 | 24.943 | 0.194 | 2.341 | 26.624 | 0.174 | 2.401 | 36.925 |
| Other | 3553 | 66.658 | 0.013 | 2.385 | 5.422 | 0.032 | 2.558 | 4.027 | 0.024 | 2.672 | 33.705 |

## Push-through calibration, in sample

Mean P(Y >= c_i) over the label rate, on the division comparison's labels (1 = calibrated). Fitted
on all cells (not out of fold), for all three models alike.

| label | label rate | P p / rate | PNB p / rate | NBNB p / rate | NB p / rate |
| --- | --- | --- | --- | --- | --- |
| BASE-T alpha_B 0.05 | 0.054 | 0.974 | 0.895 | 0.897 | 0.790 |
| BASE-T alpha_B 0.01 | 0.008 | 0.194 | 1.206 | 1.220 | 1.319 |
| BASE-T alpha_B 0.001 | 0.000 | 0.029 | 1.859 | 1.885 | 3.906 |
| BASE-T alpha_B 0.0001 | 0.000 | 0.010 | 1.970 | 1.985 | 10.307 |
| BASE-O Td 0 | 0.296 | 1.074 | 1.051 | 1.061 | 1.089 |
| BASE-O Td 0.25 | 0.238 | 1.130 | 1.071 | 1.077 | 1.076 |
| BASE-O Td 0.5 | 0.195 | 1.207 | 1.064 | 1.069 | 1.034 |
| BASE-O Td 0.75 | 0.162 | 1.277 | 1.036 | 1.040 | 0.977 |
| BASE-O Td 1 | 0.134 | 1.322 | 1.002 | 1.006 | 0.924 |

## Check against 10x's own annotation (cell_groups.csv)

Mean posterior cycling probability w by 10x group:

| 10x group | cells | w P | w PNB | w NBNB |
| --- | --- | --- | --- | --- |
| Tumor Cells | 103549 | 0.148 | 0.508 | 0.478 |
| Smooth Muscle Cells | 60057 | 0.006 | 0.008 | 0.007 |
| Proliferative Tumor Cells | 50092 | 0.918 | 0.993 | 0.987 |
| Tumor Associated Fibroblasts | 39280 | 0.030 | 0.077 | 0.055 |
| SOX2-OT+ Tumor Cells | 31159 | 0.059 | 0.141 | 0.121 |
| Macrophages | 27355 | 0.113 | 0.224 | 0.196 |
| Stromal Associated Fibroblasts | 24206 | 0.002 | 0.008 | 0.006 |
| Stromal Associated Endothelial Cells | 17817 | 0.011 | 0.024 | 0.019 |
| VEGFA+ Tumor Cells | 10087 | 0.072 | 0.367 | 0.336 |
| T and NK Cells | 9776 | 0.083 | 0.189 | 0.170 |
| Pericytes | 4826 | 0.054 | 0.137 | 0.106 |
| Tumor Associated Endothelial Cells | 3812 | 0.045 | 0.105 | 0.079 |
| Fallopian Tube Epithelium | 3029 | 0.003 | 0.021 | 0.014 |
| Malignant Cells Lining Cyst | 2525 | 0.050 | 0.120 | 0.093 |
| Inflammatory Tumor Cells | 2157 | 0.398 | 0.721 | 0.696 |
| Ciliated Epithelial Cells | 1985 | 0.133 | 0.386 | 0.349 |
| Unassigned | 1984 | 0.182 | 0.380 | 0.320 |
| Urothelial-like Cells | 1826 | 0.003 | 0.013 | 0.009 |

Within 10x's six tumour clusters (199,569 cells; not the "Tumor Associated" fibroblasts or endothelium), against its Proliferative cluster:

| model | AUROC of w (Proliferative) | tumour cells called (w > 0.5) | calls in Proliferative | Proliferative called |
| --- | --- | --- | --- | --- |
| Poisson mixture | 0.963 | 0.312 | 0.743 | 0.923 |
| Poisson + NB mixture | 0.981 | 0.532 | 0.470 | 0.997 |
| NB + NB mixture | 0.978 | 0.512 | 0.488 | 0.996 |

## Reading

- **NB + NB is the best of the four on the counts, but only just ahead of Poisson + NB.**
  - Data log-likelihood 800 above Poisson + NB; likelihood-ratio statistic for alpha_0 1,392.
  - Rootogram misfit 736, against Poisson + NB's 886.
  - Non-cycling cells are over-dispersed, but mildly (alpha_0 0.133, against alpha_1 0.343).
  - Push-through calibration is as good as Poisson + NB's: BASE-T 0.90-1.99, BASE-O 1.01-1.08.
- **It does not bring the cycling class back.** The cycling fraction is 30.4% (Poisson + NB 32.1%, Poisson 18.1%), and e^delta is unchanged (1.73).
  - Half of 10x's non-proliferative "Tumor Cells" are still called cycling (mean w 0.48; Poisson + NB 0.51).
  - Call precision against the Proliferative cluster: 49% (Poisson + NB 47%, Poisson 74%).
- So the broad class is not mainly non-cycling over-dispersion pushed into the cycling component. With an NB cycling component, cells with intermediate Tirosh counts, between the tight non-cycling NB and the 5.7x rate, fall to the cycling side. A Poisson cycling component reserves the class for the high-count cells.
  - At this depth the cycling fraction depends on the component family (18% against 30-32%): a two-class model is being asked to split what looks more like a continuum.
- **Within type, the mixtures fit stroma worse than the matched NB does.** Fibroblasts: misfit 455 (NB + NB) against 105 (NB). They fit tumour better than the NB: 1,129 against 1,353, though Poisson's 817 is better still.

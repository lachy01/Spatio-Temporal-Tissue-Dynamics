# Poisson + NB mixture (ovarian section, 2026-10-16 11:28)

`run/mixture_poisson_nb.py`. The production mixture with its cycling component NB2(rho + lambda_1,
alpha_1) instead of Poisson (`ModelSpec.cycling_nb`); one shared, unpenalised alpha_1; everything
else as production. All 395,522 cells. Best of 6 fits: warm, alpha1 start 1; 83% of fits
within 1e-6 (relative) of the best penalised log-likelihood.

| start | converged | penalised loglik | grad_inf | iterations | alpha1 | minutes |
| --- | --- | --- | --- | --- | --- | --- |
| warm, alpha1 start 0.05 | True | -703743 | 2.98495e-05 | 1193 | 0.354364 | 1.39626 |
| warm, alpha1 start 0.3 | True | -703743 | 1.18902e-05 | 1189 | 0.354364 | 1.36456 |
| warm, alpha1 start 1 | True | -703743 | 1.23646e-05 | 1128 | 0.354364 | 1.31718 |
| multistart 0 | True | -705789 | 9.57724e-06 | 219 | 0.354346 | 0.68089 |
| multistart 1 | True | -703743 | 1.60101e-05 | 1466 | 0.354364 | 1.57805 |
| multistart 2 | True | -703743 | 1.86356e-05 | 1091 | 0.354364 | 1.25607 |

## Fit

Likelihood-ratio statistic for alpha_1, 2 x (penalised loglik PNB - P) = **77,743.6** on one
parameter (boundary at 0, so no p-value). NB-c is the division comparison's full-data matched NB
(MIX's covariates and penalty in an NB mean, one dispersion).

| model | parameters | log-likelihood (data) | AIC | rootogram misfit | zeros expected | zeros observed | PIT max deviation | e^delta | rate ratio | alpha1 | theta_m | cycling fraction (mean w) | mean pi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Poisson mixture | 145 | -7.42e+05 | 1.484e+06 | 3060 | 0.3395 | 0.4064 | 0.2896 | 1.745 | 5.725 | nan | 1.399 | 0.181 | 0.1881 |
| Poisson + NB mixture | 146 | -7.03e+05 | 1.406e+06 | 885.8 | 0.3863 | 0.4064 | 0.07238 | 1.741 | 5.704 | 0.3544 | 1.487 | 0.3212 | 0.3212 |
| NB-c (matched NB) | 144 | -7.044e+05 | 1.409e+06 | 1286 | 0.4132 | 0.4064 | 0.09903 | nan | nan | nan | nan | nan | nan |

Rootograms: `rootograms.png` (0-29 and a pooled 30+).

## By type

Misfit: rootogram misfit within the type. w: mean posterior cycling probability. Neighbourhood
contrast: mean logit pi in the type's top decile of neighbourhood Tumour score minus its bottom decile.

| type | cells | misfit P | w P | nbr contrast P | misfit PNB | w PNB | nbr contrast PNB | misfit NB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tumour | 181539 | 817.029 | 0.362 | 2.664 | 987.353 | 0.625 | 3.134 | 1353.196 |
| Fibroblasts | 86585 | 2141.734 | 0.022 | 2.821 | 503.376 | 0.057 | 3.294 | 104.520 |
| Endothelial Cells | 21595 | 586.330 | 0.018 | 1.101 | 112.226 | 0.041 | 2.855 | 44.753 |
| Pericytes | 13280 | 284.535 | 0.035 | 0.448 | 51.387 | 0.067 | 3.354 | 42.257 |
| Smooth Muscle Cells | 47877 | 1130.237 | 0.003 | -1.339 | 133.612 | 0.005 | 0.966 | 40.928 |
| Macrophages | 32672 | 371.572 | 0.067 | 2.524 | 67.102 | 0.148 | 4.210 | 61.229 |
| T & NK Cells | 8421 | 33.295 | 0.082 | 1.957 | 24.943 | 0.194 | 2.341 | 36.925 |
| Other | 3553 | 66.658 | 0.013 | 2.385 | 5.422 | 0.032 | 2.558 | 33.705 |

## Push-through calibration, in sample

Mean P(Y >= c_i) over the label rate, on the division comparison's labels (1 = calibrated). Fitted
on all cells (not out of fold), for all three models alike.

| label | label rate | P p / rate | PNB p / rate | NB p / rate |
| --- | --- | --- | --- | --- |
| BASE-T alpha_B 0.05 | 0.054 | 0.974 | 0.895 | 0.790 |
| BASE-T alpha_B 0.01 | 0.008 | 0.194 | 1.206 | 1.319 |
| BASE-T alpha_B 0.001 | 0.000 | 0.029 | 1.859 | 3.906 |
| BASE-T alpha_B 0.0001 | 0.000 | 0.010 | 1.970 | 10.307 |
| BASE-O Td 0 | 0.296 | 1.074 | 1.051 | 1.089 |
| BASE-O Td 0.25 | 0.238 | 1.130 | 1.071 | 1.076 |
| BASE-O Td 0.5 | 0.195 | 1.207 | 1.064 | 1.034 |
| BASE-O Td 0.75 | 0.162 | 1.277 | 1.036 | 0.977 |
| BASE-O Td 1 | 0.134 | 1.322 | 1.002 | 0.924 |

## Check against 10x's own annotation (cell_groups.csv)

10x's "Proliferative Tumor Cells" cluster was defined independently of both fits. Its Tirosh composite is 4.19 per 100 non-blocklist transcripts, against 1.07 in "Tumor Cells".

| 10x group | cells | mean w, Poisson | mean w, Poisson + NB |
| --- | --- | --- | --- |
| Proliferative Tumor Cells | 50,092 | 0.918 | 0.993 |
| Tumor Cells | 103,549 | 0.148 | 0.508 |
| VEGFA+ Tumor Cells | 10,087 | 0.072 | 0.367 |
| SOX2-OT+ Tumor Cells | 31,159 | 0.059 | 0.141 |
| Macrophages | 27,355 | 0.113 | 0.224 |
| Smooth Muscle Cells | 60,057 | 0.006 | 0.008 |

Within 10x's tumour groups:

| | Poisson | Poisson + NB |
| --- | --- | --- |
| AUROC of w for the Proliferative cluster | 0.963 | 0.981 |
| tumour cells called cycling (w > 0.5) | 31.2% | 53.2% |
| of those calls, in the Proliferative cluster | 74.3% | 47.0% |
| of the Proliferative cluster, called | 92.3% | 99.7% |

## Reading

- **Poisson + NB is the better model of the counts.** It has a data log-likelihood 38,900 above the Poisson mixture and 1,400 above the matched NB, the lowest rootogram misfit (886, against 3,060 and 1,286) and the best predictive calibration (randomised PIT, maximum deviation 0.072). Its push-through calibration holds across every threshold: BASE-T 0.90-1.97, BASE-O 1.00-1.07. The Poisson mixture falls to 0.01 at the strictest threshold, and the NB rises to 10.3.
- **Its cycling component is no longer cycling alone.** The cycling fraction rises from 18% to 32% while e^delta stays at 1.74. Half of 10x's non-proliferative "Tumor Cells" are now called cycling, and the precision of the calls against the Proliferative cluster falls from 74% to 47%. The NB component absorbs the over-dispersed upper part of non-cycling tumour expression, as the addendum's known risk for TWO-PNB predicted. Its ranking of cells is better (AUROC 0.981 against 0.963), but pi(x) now estimates the share of a high-expression class, not a cycling fraction.
- **The neighbourhood contrasts move** (Tumour 2.66 to 3.13; Pericytes 0.45 to 3.35; Smooth Muscle -1.34 to +0.97), much as under the depth-residualised state.

# NB + NB mixture (breast section, 2026-10-18 13:57)

`run/mixture_poisson_nb.py --family nbnb`. The production mixture with both components NB2, each with its own unpenalised dispersion (alpha_0 non-cycling, alpha_1 cycling; `cycling_nb`, `noncycling_nb`); everything
else as production. All 561,897 cells. Best of 6 fits: warm from Poisson + NB, alpha0 start 0.02; 83% of fits
within 1e-6 (relative) of the best penalised log-likelihood.

| start | converged | penalised loglik | grad_inf | iterations | alpha1 | alpha0 | minutes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| warm from Poisson + NB, alpha0 start 0.02 | True | -591463 | 8.07572e-06 | 918 | 0.53433 | 0.0743758 | 2.43415 |
| warm from Poisson + NB, alpha0 start 0.1 | True | -591463 | 8.31275e-06 | 805 | 0.53433 | 0.0743758 | 2.15848 |
| warm from Poisson + NB, alpha0 start 0.3 | True | -591463 | 2.23235e-05 | 719 | 0.53433 | 0.0743758 | 1.98788 |
| multistart 0 | True | -592923 | 2.36433e-05 | 144 | 0.569118 | 0.0606656 | 0.895407 |
| multistart 1 | True | -591463 | 1.4233e-05 | 1331 | 0.53433 | 0.0743758 | 5.28729 |
| multistart 2 | True | -591463 | 2.63002e-05 | 1101 | 0.53433 | 0.0743758 | 4.62878 |

## Fit

Likelihood-ratio statistic for alpha_0, 2 x (penalised loglik
NBNB - PNB) = **159.4** on one parameter (boundary at 0, so no p-value). NB-c is the division comparison's full-data matched NB
(MIX's covariates and penalty in an NB mean, one dispersion).

| model | parameters | log-likelihood (data) | AIC | rootogram misfit | zeros expected | zeros observed | PIT max deviation | e^delta | rate ratio | alpha0 | alpha1 | theta_m | cycling fraction (mean w) | mean pi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Poisson mixture | 145 | -6e+05 | 1.2e+06 | 597.8 | 0.6024 | 0.6219 | 0.06938 | 1.902 | 6.699 | nan | nan | 1.594 | 0.09825 | 0.09825 |
| Poisson + NB mixture | 146 | -5.911e+05 | 1.182e+06 | 340.5 | 0.617 | 0.6219 | 0.01834 | 1.425 | 4.157 | nan | 0.5622 | 1.653 | 0.3071 | 0.3071 |
| NB + NB mixture | 147 | -5.91e+05 | 1.182e+06 | 337.6 | 0.6181 | 0.6219 | 0.02 | 1.441 | 4.223 | 0.07438 | 0.5343 | 1.71 | 0.2799 | 0.2799 |
| NB-c (matched NB) | 144 | -5.936e+05 | 1.188e+06 | 1097 | 0.6285 | 0.6219 | 0.04919 | nan | nan | nan | nan | nan | nan | nan |

Rootograms: `rootograms.png` (0-29 and a pooled 30+).

## By type

Misfit: rootogram misfit within the type. w: mean posterior cycling probability. Neighbourhood
contrast: mean logit pi in the type's top decile of neighbourhood Cancer Epithelial score minus its bottom decile.

| type | cells | misfit P | w P | nbr contrast P | misfit PNB | w PNB | nbr contrast PNB | misfit NBNB | w NBNB | nbr contrast NBNB | misfit NB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Endothelial | 48729 | 246.156 | 0.038 | 1.555 | 72.861 | 0.134 | 1.358 | 81.043 | 0.119 | 1.401 | 59.151 |
| CAFs | 52079 | 333.714 | 0.022 | 1.405 | 88.104 | 0.086 | 1.267 | 97.539 | 0.074 | 1.264 | 60.557 |
| PVL | 42834 | 95.529 | 0.059 | 3.192 | 29.688 | 0.198 | 3.227 | 30.900 | 0.178 | 3.334 | 37.667 |
| B-cells | 59119 | 78.015 | 0.090 | 0.913 | 98.370 | 0.320 | 1.083 | 100.010 | 0.290 | 1.081 | 108.212 |
| T-cells | 87004 | 224.988 | 0.110 | 0.929 | 233.100 | 0.379 | 1.491 | 235.825 | 0.346 | 1.463 | 245.548 |
| Myeloid | 69669 | 193.309 | 0.043 | 1.739 | 58.948 | 0.161 | 1.955 | 61.539 | 0.142 | 1.951 | 81.152 |
| Normal Epithelial | 54508 | 47.068 | 0.093 | 1.009 | 28.478 | 0.302 | 0.864 | 29.714 | 0.271 | 0.862 | 160.094 |
| Plasmablasts | 42523 | 28.252 | 0.116 | 0.951 | 19.366 | 0.362 | 0.874 | 20.360 | 0.328 | 0.885 | 66.351 |
| Cancer Epithelial | 105432 | 308.808 | 0.206 | 0.122 | 298.626 | 0.551 | 0.117 | 295.417 | 0.514 | 0.116 | 822.238 |

## Push-through calibration, in sample

Mean P(Y >= c_i) over the label rate, on the division comparison's labels (1 = calibrated). Fitted
on all cells (not out of fold), for all three models alike.

| label | label rate | P p / rate | PNB p / rate | NBNB p / rate | NB p / rate |
| --- | --- | --- | --- | --- | --- |
| BASE-T alpha_B 0.05 | 0.033 | 1.171 | 1.052 | 1.048 | 0.952 |
| BASE-T alpha_B 0.01 | 0.011 | 1.015 | 0.966 | 0.969 | 0.667 |
| BASE-T alpha_B 0.001 | 0.002 | 0.474 | 1.006 | 1.026 | 0.515 |
| BASE-T alpha_B 0.0001 | 0.000 | 0.165 | 1.070 | 1.105 | 0.470 |
| BASE-O Td 0 | 0.278 | 1.016 | 0.999 | 1.001 | 1.029 |
| BASE-O Td 0.25 | 0.227 | 1.006 | 1.003 | 1.006 | 1.051 |
| BASE-O Td 0.5 | 0.182 | 1.010 | 1.007 | 1.008 | 1.058 |
| BASE-O Td 0.75 | 0.142 | 1.031 | 1.015 | 1.016 | 1.059 |
| BASE-O Td 1 | 0.111 | 1.056 | 1.020 | 1.021 | 1.051 |

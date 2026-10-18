# Poisson + NB mixture (breast section, 2026-10-18 13:39)

`run/mixture_poisson_nb.py --family pnb`. The production mixture with its cycling component NB2(rho + lambda_1, alpha_1) instead of Poisson (`ModelSpec.cycling_nb`); one shared, unpenalised alpha_1; everything
else as production. All 561,897 cells. Best of 6 fits: multistart 1; 83% of fits
within 1e-6 (relative) of the best penalised log-likelihood.

| start | converged | penalised loglik | grad_inf | iterations | alpha1 | alpha0 | minutes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| warm, alpha1 start 0.05 | True | -591543 | 8.94291e-06 | 715 | 0.562177 | nan | 1.53026 |
| warm, alpha1 start 0.3 | True | -591543 | 8.86871e-06 | 904 | 0.562177 | nan | 1.72479 |
| warm, alpha1 start 1 | True | -591543 | 1.03878e-05 | 1108 | 0.562177 | nan | 2.02784 |
| multistart 0 | True | -592967 | 3.27149e-06 | 109 | 0.593932 | nan | 0.801784 |
| multistart 1 | True | -591543 | 2.40812e-05 | 1058 | 0.562177 | nan | 2.03645 |
| multistart 2 | True | -591543 | 2.84031e-05 | 1023 | 0.562177 | nan | 2.06206 |

## Fit

Likelihood-ratio statistic for alpha_1, 2 x (penalised loglik
PNB - P) = **17,578.3** on one parameter (boundary at 0, so no p-value). NB-c is the division comparison's full-data matched NB
(MIX's covariates and penalty in an NB mean, one dispersion).

| model | parameters | log-likelihood (data) | AIC | rootogram misfit | zeros expected | zeros observed | PIT max deviation | e^delta | rate ratio | alpha0 | alpha1 | theta_m | cycling fraction (mean w) | mean pi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Poisson mixture | 145 | -6e+05 | 1.2e+06 | 597.8 | 0.6024 | 0.6219 | 0.06938 | 1.902 | 6.699 | nan | nan | 1.594 | 0.09825 | 0.09825 |
| Poisson + NB mixture | 146 | -5.911e+05 | 1.182e+06 | 340.5 | 0.617 | 0.6219 | 0.01834 | 1.425 | 4.157 | nan | 0.5622 | 1.653 | 0.3071 | 0.3071 |
| NB-c (matched NB) | 144 | -5.936e+05 | 1.188e+06 | 1097 | 0.6285 | 0.6219 | 0.04952 | nan | nan | nan | nan | nan | nan | nan |

Rootograms: `rootograms.png` (0-29 and a pooled 30+).

## By type

Misfit: rootogram misfit within the type. w: mean posterior cycling probability. Neighbourhood
contrast: mean logit pi in the type's top decile of neighbourhood Cancer Epithelial score minus its bottom decile.

| type | cells | misfit P | w P | nbr contrast P | misfit PNB | w PNB | nbr contrast PNB | misfit NB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Endothelial | 48729 | 246.156 | 0.038 | 1.555 | 72.861 | 0.134 | 1.358 | 59.151 |
| CAFs | 52079 | 333.714 | 0.022 | 1.405 | 88.104 | 0.086 | 1.267 | 60.557 |
| PVL | 42834 | 95.529 | 0.059 | 3.192 | 29.688 | 0.198 | 3.227 | 37.667 |
| B-cells | 59119 | 78.015 | 0.090 | 0.913 | 98.370 | 0.320 | 1.083 | 108.212 |
| T-cells | 87004 | 224.988 | 0.110 | 0.929 | 233.100 | 0.379 | 1.491 | 245.548 |
| Myeloid | 69669 | 193.309 | 0.043 | 1.739 | 58.948 | 0.161 | 1.955 | 81.152 |
| Normal Epithelial | 54508 | 47.068 | 0.093 | 1.009 | 28.478 | 0.302 | 0.864 | 160.094 |
| Plasmablasts | 42523 | 28.252 | 0.116 | 0.951 | 19.366 | 0.362 | 0.874 | 66.351 |
| Cancer Epithelial | 105432 | 308.808 | 0.206 | 0.122 | 298.626 | 0.551 | 0.117 | 822.238 |

## Push-through calibration, in sample

Mean P(Y >= c_i) over the label rate, on the division comparison's labels (1 = calibrated). Fitted
on all cells (not out of fold), for all three models alike.

| label | label rate | P p / rate | PNB p / rate | NB p / rate |
| --- | --- | --- | --- | --- |
| BASE-T alpha_B 0.05 | 0.033 | 1.171 | 1.052 | 0.952 |
| BASE-T alpha_B 0.01 | 0.011 | 1.015 | 0.966 | 0.667 |
| BASE-T alpha_B 0.001 | 0.002 | 0.474 | 1.006 | 0.515 |
| BASE-T alpha_B 0.0001 | 0.000 | 0.165 | 1.070 | 0.470 |
| BASE-O Td 0 | 0.278 | 1.016 | 0.999 | 1.029 |
| BASE-O Td 0.25 | 0.227 | 1.006 | 1.003 | 1.051 |
| BASE-O Td 0.5 | 0.182 | 1.010 | 1.007 | 1.058 |
| BASE-O Td 0.75 | 0.142 | 1.031 | 1.015 | 1.059 |
| BASE-O Td 1 | 0.111 | 1.056 | 1.020 | 1.051 |

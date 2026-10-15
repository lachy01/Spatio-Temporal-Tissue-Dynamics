# Ovarian: depth-residualised state, against production (2026-10-15 23:08)

`run/ovarian_depth_residual_check.py`. Each PhiSpace score residualised on log N within type
(slope removed, type means kept; neighbour means rebuilt; types from raw scores); production
specification refitted on all 395,522 cells, 4 restarts (best of them; restart
log-likelihoods -742,258.7, -742,258.7, -742,258.7, -744,790.4). The production
column is the Spartan point estimate (20 restarts). Largest within-type |Spearman| of a
residualised score with log N: 0.084.

## Global

| fit | e^delta | rate ratio | theta_m | mean w | sd logit pi | rho(logit pi, log N), all | penalised loglik |
| --- | --- | --- | --- | --- | --- | --- | --- |
| production | 1.745 | 5.725 | 1.399 | 0.181 | 2.738 | 0.656 | -741431.823 |
| depth-residualised | 1.743 | 5.713 | 1.342 | 0.183 | 3.120 | 0.630 | -742258.692 |

## By type

Neighbourhood contrast: mean logit pi at or above the type's 90th percentile of neighbourhood
Tumour score minus at or below its 10th (raw-score neighbourhood, same cells in both fits).

| type | cells | w prod | w resid | rho(pi,logN) prod | rho(pi,logN) resid | nbr contrast prod | nbr contrast resid |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tumour | 181539 | 0.362 | 0.365 | 0.375 | 0.203 | 2.664 | 2.489 |
| Fibroblasts | 86585 | 0.022 | 0.023 | 0.347 | 0.349 | 2.821 | 3.746 |
| Smooth Muscle Cells | 47877 | 0.003 | 0.002 | -0.097 | -0.089 | -1.339 | 1.437 |
| Macrophages | 32672 | 0.067 | 0.075 | 0.226 | 0.177 | 2.524 | 4.689 |
| Endothelial Cells | 21595 | 0.018 | 0.016 | -0.014 | 0.147 | 1.101 | 3.557 |
| Pericytes | 13280 | 0.035 | 0.032 | 0.080 | 0.157 | 0.448 | 4.293 |
| T & NK Cells | 8421 | 0.082 | 0.098 | 0.149 | 0.065 | 1.957 | 2.877 |
| Other | 3553 | 0.013 | 0.010 | -0.460 | -0.347 | 2.385 | 1.885 |

## Reading

- **Global quantities do not move.** e^delta 1.745 to 1.743, theta_m 1.40 to 1.34, cycling fraction 18.1% to 18.3%.
- **The residualised state fits worse** by 827 penalised log-likelihood units. Whatever depth carried in the scores predicted cycling.
- **Residualising removes about half of Tumour's pi-depth association** (0.38 to 0.20), and none of Fibroblasts' (0.35 to 0.35). Within type, depth itself tracks the tumour neighbourhood: Spearman of log N with the neighbourhood Tumour score is 0.44 in Fibroblasts, 0.35 in Macrophages and 0.25 in Tumour. So the remaining association runs through where cells sit (deeper stroma inside tumour regions), not through each cell's own scores.
- **Tumour's neighbourhood contrast is stable** (2.66 to 2.49). The stromal types' contrasts are not: Pericytes 0.45 to 4.29, Endothelial 1.10 to 3.56, Macrophages 2.52 to 4.69, and Smooth Muscle changes sign (-1.34 to +1.44). On this section, the stromal neighbourhood effects depend on how depth enters the state. They are to be read against the bootstrap intervals before anything is concluded.

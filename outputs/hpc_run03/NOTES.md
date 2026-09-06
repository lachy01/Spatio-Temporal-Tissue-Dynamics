# multistart + bootstrap, first full run (bootstrap_maxiter = 300)

## multistart -- clean
20/20 converged, ||grad||inf median 1.16e-05, median 5,649 iterations, 12.4 min
median. Restart agreement **95%** (two distinct optima; 19 restarts on the
better one), so spec 4.4's multimodality caveat does NOT apply.

Point estimate: mean pi = 11.6% cycling, rate ratio 5.63x, entropy 0.683,
psi (oxygen) = +0.840, sigma_1(B) = 1.364. B has exactly two non-zero singular
values (1.364, 0.858), confirming rank 2 independently.

## bootstrap -- intervals provisional
300/300 tasks ran, but **all hit the 300-iteration cap** with ||grad||inf median
105 (max 855). Estimates are well centred on the point estimate (shifts of
5-8% of a CI half-width, and gradient size does not predict the estimate), so
there is no sign of bias -- but under-iterating from a common warm start
shrinks the spread toward that start, so these intervals are likely too narrow:

| estimand | point | CI (300 iterations, provisional) |
|---|---:|---|
| rate ratio | 5.63 | [5.54, 5.74] |
| mean pi | 11.6% | [10.6%, 12.5%] |
| entropy | 0.683 | [0.659, 0.712] |
| sigma_1(B) | 1.364 | [1.239, 1.655] |

sigma_1's lower bound is 19x the Test 02 null reference (0.064), which is a
margin no plausible widening would erase -- the bilinear interaction is real.

Re-run with bootstrap_maxiter = 5000 and compare widths.

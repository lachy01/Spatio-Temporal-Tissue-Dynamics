# cv_penalty, first Spartan array (maxiter = 2000)

Kept as a record: this array is what showed the 2000-iteration cap was too low
and that held-out log-likelihood cannot select `weight_ridge`.

- 40/40 tasks ran, ~7.7 min median on 4 cores.
- **0 of 40 fits converged below `weight_ridge = 100`**; every one stopped at the
  cap. Median `||grad||inf`: 142 at lambda=0, 0.63 at lambda=10, 0.0053 at
  lambda=30, against a 1e-5 threshold.
- Paired by fold, held-out log-likelihood is monotone in lambda and prefers
  **lambda = 0** (paired t from -3.1 at lambda=1 to -17.3 at lambda=300) -- the
  specification where `sigma_1(B) = 30` and the weight model is diverging.
- `e^delta` was 1.726-1.750 across every lambda, so the component separation is
  indifferent to the penalty.

Conclusions carried into config.yaml: `optim.maxiter` raised to 20000, and the
selection rule changed to "smallest lambda giving a converged, non-degenerate
fit", with held-out likelihood only breaking ties.

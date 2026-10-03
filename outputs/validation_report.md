
## Test 04 -- gradient check (spec 6.4)

**PASS**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

| trial | loglik | n params | max relative error |
|---|---:|---:|---:|
| 0 | -6522.11 | 55 | 2.34e-07 |
| 1 | -6541.66 | 55 | 2.07e-07 |
| 2 | -6540.69 | 55 | 1.53e-07 |

- threshold (spec 6.4): **1e-5**; worst observed **2.34e-07**

## Test 04 -- gradient check (spec 6.4)

**PASS**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

| trial | loglik | n params | max relative error |
|---|---:|---:|---:|
| 0 | -6522.11 | 55 | 2.34e-07 |
| 1 | -6541.66 | 55 | 2.07e-07 |
| 2 | -6540.69 | 55 | 1.53e-07 |

- threshold (spec 6.4): **1e-5**; worst observed **2.34e-07**

## Test 05 -- EM monotonicity (spec 6.5)

**PASS**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

| stage | iterations | final loglik | min step | monotone |
|---|---:|---:|---:|---|
| A | 121 | -27947.167556 | +2.394e-09 | True |
| B | 92 | -27242.456650 | +2.492e-09 | True |

## Test 06 -- optimiser agreement (spec 6.6)

**PASS**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

- EM: loglik **-27947.16755611** in 121 iterations
- L-BFGS: loglik **-27947.16755610** in 38 iterations (||grad||inf 3.1e-07)
- relative log-likelihood difference: **4.05e-13** (threshold 1e-6 -> pass)

| parameter | EM | L-BFGS | relative difference |
|---|---:|---:|---:|
| `b0` | -0.50725142 | -0.50725344 | 3.98e-06 |
| `theta` | 0.83129505 | 0.83129572 | 7.98e-07 |
| `delta` | 0.32145760 | 0.32145736 | 7.26e-07 |
| `s[0]` | -0.10715875 | -0.10715856 | 1.86e-06 |
| `s[1]` | -0.11559725 | -0.11559712 | 1.14e-06 |
| `s[2]` | 0.03178474 | 0.03178484 | 3.18e-06 |

## Test 08 -- held-out gene machinery (spec 6.8)

**PASS**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

- real `w`: coef **+1.4145**, delta-deviance **2491.7** vs permuted mean 35.2 (ratio **70.8x**, p = 0.032)
- unrelated `w`: delta-deviance 41.2 vs permuted mean 20.7 (ratio 2.0x, p = 0.161)

Verifies the statistic separates a `w` that drives the held-out counts from one that does not, using the same spatially-blocked permutation null the real-data job uses.

## Test 04 -- gradient check (spec 6.4)

**PASS**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

| trial | loglik | n params | max relative error |
|---|---:|---:|---:|
| 0 | -6522.11 | 55 | 2.34e-07 |
| 1 | -6541.66 | 55 | 2.07e-07 |
| 2 | -6540.69 | 55 | 1.53e-07 |

- threshold (spec 6.4): **1e-5**; worst observed **2.34e-07**

## Test 05 -- EM monotonicity (spec 6.5)

**PASS**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

| stage | iterations | final loglik | min step | monotone |
|---|---:|---:|---:|---|
| A | 121 | -27947.167556 | +2.394e-09 | True |
| B | 92 | -27242.456650 | +2.492e-09 | True |

## Test 06 -- optimiser agreement (spec 6.6)

**PASS**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

- EM: loglik **-27947.16755611** in 121 iterations
- L-BFGS: loglik **-27947.16755610** in 38 iterations (||grad||inf 3.1e-07)
- relative log-likelihood difference: **4.05e-13** (threshold 1e-6 -> pass)

| parameter | EM | L-BFGS | relative difference |
|---|---:|---:|---:|
| `b0` | -0.50725142 | -0.50725344 | 3.98e-06 |
| `theta` | 0.83129505 | 0.83129572 | 7.98e-07 |
| `delta` | 0.32145760 | 0.32145736 | 7.26e-07 |
| `s[0]` | -0.10715875 | -0.10715856 | 1.86e-06 |
| `s[1]` | -0.11559725 | -0.11559712 | 1.14e-06 |
| `s[2]` | 0.03178474 | 0.03178484 | 3.18e-06 |

## Test 04 -- gradient check (spec 6.4)

**PASS**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

| trial | loglik | n params | max relative error |
|---|---:|---:|---:|
| 0 | -6522.11 | 55 | 2.34e-07 |
| 1 | -6541.66 | 55 | 2.07e-07 |
| 2 | -6540.69 | 55 | 1.53e-07 |

- threshold (spec 6.4): **1e-5**; worst observed **2.34e-07**

## Test 05 -- EM monotonicity (spec 6.5)

**PASS**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

| stage | iterations | final loglik | min step | monotone |
|---|---:|---:|---:|---|
| A | 121 | -27947.167556 | +2.394e-09 | True |
| B | 92 | -27242.456650 | +2.492e-09 | True |

## Test 06 -- optimiser agreement (spec 6.6)

**PASS**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

- EM: loglik **-27947.16755611** in 121 iterations
- L-BFGS: loglik **-27947.16755610** in 38 iterations (||grad||inf 3.1e-07)
- relative log-likelihood difference: **4.05e-13** (threshold 1e-6 -> pass)

| parameter | EM | L-BFGS | relative difference |
|---|---:|---:|---:|
| `b0` | -0.50725142 | -0.50725344 | 3.98e-06 |
| `theta` | 0.83129505 | 0.83129572 | 7.98e-07 |
| `delta` | 0.32145760 | 0.32145736 | 7.26e-07 |
| `s[0]` | -0.10715875 | -0.10715856 | 1.86e-06 |
| `s[1]` | -0.11559725 | -0.11559712 | 1.14e-06 |
| `s[2]` | 0.03178474 | 0.03178484 | 3.18e-06 |

## Test 08 -- held-out gene machinery (spec 6.8)

**PASS**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

- real `w`: coef **+1.4145**, delta-deviance **2491.7** vs permuted mean 35.2 (ratio **70.8x**, p = 0.032)
- unrelated `w`: delta-deviance 41.2 vs permuted mean 20.7 (ratio 2.0x, p = 0.161)

Verifies the statistic separates a `w` that drives the held-out counts from one that does not, using the same spatially-blocked permutation null the real-data job uses.

## Test 04 -- gradient check (spec 6.4)

**PASS**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

| trial | loglik | n params | max relative error |
|---|---:|---:|---:|
| 0 | -6522.11 | 55 | 2.34e-07 |
| 1 | -6541.66 | 55 | 2.07e-07 |
| 2 | -6540.69 | 55 | 1.53e-07 |

- threshold (spec 6.4): **1e-5**; worst observed **2.34e-07**

## Test 04 -- gradient check (spec 6.4)

**PASS**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

| trial | loglik | n params | max relative error |
|---|---:|---:|---:|
| 0 | -6522.11 | 55 | 2.34e-07 |
| 1 | -6541.66 | 55 | 2.07e-07 |
| 2 | -6540.69 | 55 | 1.53e-07 |

- threshold (spec 6.4): **1e-5**; worst observed **2.34e-07**

## Test 05 -- EM monotonicity (spec 6.5)

**PASS**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

| stage | iterations | final loglik | min step | monotone |
|---|---:|---:|---:|---|
| A | 121 | -27947.167556 | +2.394e-09 | True |
| B | 92 | -27242.456650 | +2.492e-09 | True |

## Test 06 -- optimiser agreement (spec 6.6)

**PASS**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

- EM: loglik **-27947.16755611** in 121 iterations
- L-BFGS: loglik **-27947.16755610** in 38 iterations (||grad||inf 3.1e-07)
- relative log-likelihood difference: **4.05e-13** (threshold 1e-6 -> pass)

| parameter | EM | L-BFGS | relative difference |
|---|---:|---:|---:|
| `b0` | -0.50725142 | -0.50725344 | 3.98e-06 |
| `theta` | 0.83129505 | 0.83129572 | 7.98e-07 |
| `delta` | 0.32145760 | 0.32145736 | 7.26e-07 |
| `s[0]` | -0.10715875 | -0.10715856 | 1.86e-06 |
| `s[1]` | -0.11559725 | -0.11559712 | 1.14e-06 |
| `s[2]` | 0.03178474 | 0.03178484 | 3.18e-06 |

## Test 08 -- held-out gene machinery (spec 6.8)

**PASS**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

- real `w`: coef **+1.4145**, delta-deviance **2491.7** vs permuted mean 35.2 (ratio **70.8x**, p = 0.032)
- unrelated `w`: delta-deviance 41.2 vs permuted mean 20.7 (ratio 2.0x, p = 0.161)

Verifies the statistic separates a `w` that drives the held-out counts from one that does not, using the same spatially-blocked permutation null the real-data job uses.

## Test 01 -- simulation recovery (spec 6.1)

**FAIL**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

| parameter | truth | mean estimate | MC SE | bias / SE |
|---|---:|---:|---:|---:|
| `b0` | -0.5500 | -0.5564 | 0.0022 | 2.88 |
| `theta` | 0.8500 | 0.8516 | 0.0016 | 1.00 |
| `delta` | 0.3266 | 0.3312 | 0.0020 | 2.33 |
| `a0` | -1.7346 | -1.7388 | 0.0197 | 0.22 |
| `psi` | 0.3500 | 0.3575 | 0.0312 | 0.24 |

- replicates: **8** at n = **30,000**
- `||B_hat - B||_F / ||B||_F`: mean **0.1926**, max 0.2675 (threshold 0.15 -> FAIL)

git: **06a5b70**
loglik: **-40998.720923**
||grad||inf: **4.83e-06**
iterations: **162**
converged: **True**
restart agreement: **67%**
wall time: **4.0s**
normalised entropy (spec 8.7): **0.5911**
rate ratio exp(e^delta): **3.996**
mean w: **0.2407**
degeneracy flags fired: **False**

## Test 02 -- null bilinear (spec 6.2)

**FAIL**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

- replicates: **15** at n = **20,000**, 25 bootstrap resamples each
- median `sigma_1(B_hat)` under the null: **0.1452** (IQR 0.1311-0.1545)
- `sigma_1(B_hat)` under Test 01's alternative: **0.3215**
- ratio null/alternative: **0.452** (threshold 0.20 -> FAIL)
- bootstrap CIs consistent with the null level: **0%** of 15 (threshold 90% -> FAIL)
- true `sigma_1(B)` under the alternative: 0.3807

## Test 03 -- single-component null (spec 6.3)

**FAIL**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

- n = **30,000**, simulated with `pi_i == 0` (no latent class structure)
- `e^delta` = **0.16108** (collapse threshold 0.05)
- `mean(pi_hat)` = **0.385623** (boundary threshold 1e-2)
- component collapse detected: **False**
- normalised entropy = **0.9902** (threshold 0.10 -> FAIL)
- spec 3.3 degeneracy flags fired: **False** (fold_collapsed=False, weight_degenerate=False)
- log-likelihood -31077.2724, 2000 iterations

## Test 04 -- gradient check (spec 6.4)

**PASS**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

| trial | loglik | n params | max relative error |
|---|---:|---:|---:|
| 0 | -6522.11 | 55 | 2.34e-07 |
| 1 | -6541.66 | 55 | 2.07e-07 |
| 2 | -6540.69 | 55 | 1.53e-07 |

- threshold (spec 6.4): **1e-5**; worst observed **2.34e-07**

## Test 05 -- EM monotonicity (spec 6.5)

**PASS**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

| stage | iterations | final loglik | min step | monotone |
|---|---:|---:|---:|---|
| A | 121 | -27947.167556 | +2.394e-09 | True |
| B | 92 | -27242.456650 | +2.492e-09 | True |

## Test 06 -- optimiser agreement (spec 6.6)

**PASS**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

- EM: loglik **-27947.16755611** in 121 iterations
- L-BFGS: loglik **-27947.16755610** in 38 iterations (||grad||inf 3.1e-07)
- relative log-likelihood difference: **4.05e-13** (threshold 1e-6 -> pass)

| parameter | EM | L-BFGS | relative difference |
|---|---:|---:|---:|
| `b0` | -0.50725142 | -0.50725344 | 3.98e-06 |
| `theta` | 0.83129505 | 0.83129572 | 7.98e-07 |
| `delta` | 0.32145760 | 0.32145736 | 7.26e-07 |
| `s[0]` | -0.10715875 | -0.10715856 | 1.86e-06 |
| `s[1]` | -0.11559725 | -0.11559712 | 1.14e-06 |
| `s[2]` | 0.03178474 | 0.03178484 | 3.18e-06 |

## Test 07 -- adversarial size-confound (spec 6.7)

**FAIL**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

- n = **30,000**; simulated with `pi_i == 0` and an area effect built from frequencies a df=8 spline cannot represent
- normalised entropy = **0.9995** (threshold 0.25 -> FAIL)
- fold change `exp(e^delta)` = **1.2701** (threshold 1.5 -> pass)
- `sigma_1(B_hat)` = **776.2441** against a genuine-null reference of 0.0998 (threshold 2x -> FAIL)
- `mean(w)` = 0.1144; log-likelihood -32235.65

**Spec 6.7:** a failure here means the mixture is separating cells on size and segmentation quality rather than biology. It is not a tuning problem -- the area spline must be made flexible enough, the model refit, and this test re-run **before any real-data result is reported**.

## Test 08 -- held-out gene machinery (spec 6.8)

**PASS**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

- real `w`: coef **+1.4145**, delta-deviance **2491.7** vs permuted mean 35.2 (ratio **70.8x**, p = 0.032)
- unrelated `w`: delta-deviance 41.2 vs permuted mean 20.7 (ratio 2.0x, p = 0.161)

Verifies the statistic separates a `w` that drives the held-out counts from one that does not, using the same spatially-blocked permutation null the real-data job uses.

## Test 01 -- simulation recovery (spec 6.1)

**FAIL**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

| parameter | truth | mean estimate | MC SE | bias / SE |
|---|---:|---:|---:|---:|
| `b0` | -0.5500 | -0.5564 | 0.0022 | 2.88 |
| `theta` | 0.8500 | 0.8516 | 0.0016 | 1.00 |
| `delta` | 0.3266 | 0.3312 | 0.0020 | 2.33 |
| `a0` | -1.7346 | -1.7388 | 0.0197 | 0.22 |
| `psi` | 0.3500 | 0.3575 | 0.0312 | 0.24 |

- replicates: **8** at n = **30,000**
- `||B_hat - B||_F / ||B||_F`: mean **0.1926**, max 0.2675 (threshold 0.15 -> FAIL)

git: **06a5b70**
loglik: **-40998.720923**
||grad||inf: **4.83e-06**
iterations: **162**
converged: **True**
restart agreement: **67%**
wall time: **3.8s**
normalised entropy (spec 8.7): **0.5911**
rate ratio exp(e^delta): **3.996**
mean w: **0.2407**
degeneracy flags fired: **False**

## Test 02 -- null bilinear (spec 6.2)

**FAIL**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

- replicates: **15** at n = **20,000**, 25 bootstrap resamples each
- median `sigma_1(B_hat)` under the null: **0.1452** (IQR 0.1311-0.1545)
- `sigma_1(B_hat)` under Test 01's alternative: **0.3215**
- ratio null/alternative: **0.452** (threshold 0.20 -> FAIL)
- bootstrap CIs consistent with the null level: **0%** of 15 (threshold 90% -> FAIL)
- true `sigma_1(B)` under the alternative: 0.3807

## Test 03 -- single-component null (spec 6.3)

**FAIL**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

- n = **30,000**, simulated with `pi_i == 0` (no latent class structure)
- `e^delta` = **0.16108** (collapse threshold 0.05)
- `mean(pi_hat)` = **0.385623** (boundary threshold 1e-2)
- component collapse detected: **False**
- normalised entropy = **0.9902** (threshold 0.10 -> FAIL)
- spec 3.3 degeneracy flags fired: **False** (fold_collapsed=False, weight_degenerate=False)
- log-likelihood -31077.2724, 2000 iterations

## Test 07 -- adversarial size-confound (spec 6.7)

**FAIL**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

- n = **30,000**; simulated with `pi_i == 0` and an area effect built from frequencies a df=8 spline cannot represent
- normalised entropy = **0.9995** (threshold 0.25 -> FAIL)
- fold change `exp(e^delta)` = **1.2701** (threshold 1.5 -> pass)
- `sigma_1(B_hat)` = **776.2441** against a genuine-null reference of 0.0998 (threshold 2x -> FAIL)
- `mean(w)` = 0.1144; log-likelihood -32235.65

**Spec 6.7:** a failure here means the mixture is separating cells on size and segmentation quality rather than biology. It is not a tuning problem -- the area spline must be made flexible enough, the model refit, and this test re-run **before any real-data result is reported**.

## Test 03 -- single-component null (spec 6.3)

**FAIL**

- n = **100,000**, simulated with `pi_i == 0` (no latent class structure)
- `e^delta` = **0.15371** (collapse threshold 0.05)
- `mean(pi_hat)` = **0.099098** (boundary threshold 1e-2)
- component collapse detected: **False**
- normalised entropy = **0.9990** (threshold 0.10 -> FAIL)
- spec 3.3 degeneracy flags fired: **False** (fold_collapsed=False, weight_degenerate=False)
- log-likelihood -104055.4577, 2000 iterations

## Test 07 -- adversarial size-confound (spec 6.7)

**FAIL**

- n = **100,000**; simulated with `pi_i == 0` and an area effect built from frequencies a df=8 spline cannot represent
- normalised entropy = **0.3143** (threshold 0.25 -> FAIL)
- fold change `exp(e^delta)` = **6.3743** (threshold 1.5 -> FAIL)
- `sigma_1(B_hat)` = **0.0448** against a genuine-null reference of 0.0693 (threshold 2x -> pass)
- `mean(w)` = 0.5838; log-likelihood -130148.55

**Spec 6.7:** a failure here means the mixture is separating cells on size and segmentation quality rather than biology. It is not a tuning problem -- the area spline must be made flexible enough, the model refit, and this test re-run **before any real-data result is reported**.

## Test 05 -- EM monotonicity (spec 6.5)

**PASS**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

| stage | iterations | final loglik | min step | monotone |
|---|---:|---:|---:|---|
| A | 121 | -27947.167556 | +2.394e-09 | True |
| B | 92 | -27242.456650 | +2.492e-09 | True |

## Test 06 -- optimiser agreement (spec 6.6)

**PASS**

_Run at reduced scale (set `MIXTURE_FULL=1` for the sizes named in the spec)._

- EM: loglik **-27947.16755611** in 121 iterations
- L-BFGS: loglik **-27947.16755610** in 38 iterations (||grad||inf 3.1e-07)
- relative log-likelihood difference: **4.05e-13** (threshold 1e-6 -> pass)

| parameter | EM | L-BFGS | relative difference |
|---|---:|---:|---:|
| `b0` | -0.50725142 | -0.50725344 | 3.98e-06 |
| `theta` | 0.83129505 | 0.83129572 | 7.98e-07 |
| `delta` | 0.32145760 | 0.32145736 | 7.26e-07 |
| `s[0]` | -0.10715875 | -0.10715856 | 1.86e-06 |
| `s[1]` | -0.11559725 | -0.11559712 | 1.14e-06 |
| `s[2]` | 0.03178474 | 0.03178484 | 3.18e-06 |

## Test 01 -- simulation recovery (spec 6.1)

**PASS**

| parameter | truth | mean estimate | MC SE | bias / SE |
|---|---:|---:|---:|---:|
| `b0` | -0.5500 | -0.5494 | 0.0016 | 0.39 |
| `theta` | 0.8500 | 0.8483 | 0.0008 | 2.15 |
| `delta` | 0.3266 | 0.3268 | 0.0008 | 0.23 |
| `a0` | -1.7346 | -1.7306 | 0.0057 | 0.69 |
| `psi` | 0.3500 | 0.3500 | 0.0093 | 0.00 |

- replicates: **20** at n = **100,000**
- `||B_hat - B||_F / ||B||_F`: mean **0.1156**, max 0.2713 (threshold 0.15 -> pass)

git: **06a5b70**
loglik: **-135065.565413**
||grad||inf: **1.69e-05**
iterations: **310**
converged: **True**
restart agreement: **80%**
wall time: **9.6s**
normalised entropy (spec 8.7): **0.5790**
rate ratio exp(e^delta): **3.991**
mean w: **0.2123**
degeneracy flags fired: **False**

## Test 03 -- single-component null (spec 6.3)

**FAIL**

- n = **100,000**, simulated with `pi_i == 0` (no latent class structure)
- `e^delta` = **0.17760** (collapse threshold 0.05)
- `mean(pi_hat)` = **0.089822** (boundary threshold 1e-2)
- component collapse detected: **False**
- normalised entropy = **0.7490** (threshold 0.10 -> FAIL)
- spec 3.3 degeneracy flags fired: **False** (fold_collapsed=False, weight_degenerate=False)
- log-likelihood -104111.3371, 1290 iterations

## Test 07 -- adversarial size-confound (spec 6.7)

**FAIL**

- n = **100,000**; simulated with `pi_i == 0` and an area effect built from frequencies a df=8 spline cannot represent
- normalised entropy = **0.3143** (threshold 0.25 -> FAIL)
- fold change `exp(e^delta)` = **6.3744** (threshold 1.5 -> FAIL)
- `sigma_1(B_hat)` = **0.0444** against a genuine-null reference of 0.0686 (threshold 2x -> pass)
- `mean(w)` = 0.5838; log-likelihood -130148.79

**Spec 6.7:** a failure here means the mixture is separating cells on size and segmentation quality rather than biology. It is not a tuning problem -- the area spline must be made flexible enough, the model refit, and this test re-run **before any real-data result is reported**.

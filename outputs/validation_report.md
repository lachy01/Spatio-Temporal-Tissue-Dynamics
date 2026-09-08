
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

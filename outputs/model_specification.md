# The fitted model — specification and coefficients

Xenium Prime 5K breast cancer FFPE, **678,885 cells**. Stage B, Configuration V,
`weight_ridge = 10`, `area_spline_df = 8`, `rank = 2`. CIs are percentile
intervals from a 300-replicate spatial block bootstrap.

## The model

Each cell has a latent state `Z_i ∈ {0, 1}`, "cycling" or not:

$$P(Z_i = 1) = \pi_i, \qquad y_i \mid Z_i = k \;\sim\; \text{Poisson}(\lambda_{ki})$$

`y_i` is the summed count over the 53 Tirosh S/G2M genes present in the panel.

**Component rates** — detection covariates only, never cell state:

$$\log \lambda_{0i} = b_0 + \theta \log N_i + s(\log a_i), \qquad
\log \lambda_{1i} = \log \lambda_{0i} + e^{\delta}$$

The two components share everything but the additive constant `e^δ`, so
`λ₁/λ₀ = exp(e^δ)` for every cell — the **proportional-rates restriction**, which
is what makes the model identifiable at this count depth. `e^δ > 0` pins
component 1 as the faster one and removes label switching.

**Mixing weight** — cell state, density and oxygen only, never detection:

$$\operatorname{logit} \pi_i = a_0 + \tilde\phi_i^{\top} a + \tilde{\bar\phi}_i^{\top} c
+ \tilde\phi_i^{\top} B \, \tilde{\bar\phi}_i + h(\log d_i) + \psi \, u_i$$

with `B = U Vᵀ` of rank 2.

| symbol | meaning |
|---|---|
| `N_i` | library size **excluding** the 53 Tirosh genes and the 530-gene blocklist (15.6% of the library) |
| `a_i` | segmented cell area, µm² |
| `d_i` | local density, cells within 50 µm per (100 µm)² |
| `u_i` | oxygen field (vessel proximity / perfusion — it failed its HIF check) |
| `φ̃_i` | own state: 9 celltype_major PhiSpace scores, centred and projected to the 8-dimensional sum-to-zero subspace (the scores are compositional, rowsums ≡ −7) |
| `φ̄̃_i` | the same, averaged over cells within 80 µm, excluding the cell itself |
| `s`, `h` | natural cubic splines (7 and 2 columns; the constant is projected out) |

**33 parameters**: 1 + 1 + 7 (rates) + 1 (δ) + 1 + 8 + 8 + 2 + 1 (weight) + 2×8 + 2×8 (`U`, `V`, of which 30 are identified after the rotational freedom in `B = UVᵀ`).

## Scalar coefficients

| parameter | estimate | 95% CI |
|---|---:|---|
| `b0` — rate intercept | −5.3757 | [−5.4721, −5.2634] |
| `theta` — log library size | **+1.0362** | [+1.0138, +1.0567] |
| `a0` — weight intercept | −2.4381 | — |
| `psi` — oxygen | **+0.8396** | [+0.5107, +1.1311] |
| `e^delta` | +1.7280 | [+1.7111, +1.7478] |
| **fold change** `exp(e^delta)` | **5.630** | [5.535, 5.742] |
| **cycling fraction** `mean(pi)` | **11.60%** | [10.61%, 12.47%] |

`s` (log-area spline) = [+0.3647, −0.1036, −0.0505, +0.0020, −0.0022, +0.0230, −0.1804]
`h` (log-density spline) = [−1.5663, +0.0859]

`theta = 1.036` is essentially the 1.0 a fixed offset assumes, so Configuration F
is a defensible simplification (see the bracket in `results_stage_b.md`).

## Cell-state coefficients

Log-odds of cycling, at the mean state, mapped back from the projected basis to
the nine cell types (`Q a`). **These sum to zero by construction** — the scores
are compositional, so only contrasts between cell types are identified, never
absolute levels.

| cell type | own `a` | 95% CI | neighbour `c` | 95% CI |
|---|---:|---|---:|---|
| Cancer Epithelial | **+2.983** | [+2.846, +3.111] | **+3.117** | [+2.752, +3.477] |
| T-cells | **+2.879** | [+2.774, +2.969] | +0.380 | [−0.001, +0.676] |
| B-cells | **+2.339** | [+2.191, +2.494] | −0.559 | [−0.804, −0.314] |
| PVL | −0.421 | [−0.579, −0.224] | +0.138 | [−0.001, +0.280] |
| Normal Epithelial | −0.566 | [−0.767, −0.355] | −0.680 | [−1.014, −0.326] |
| Plasmablasts | −0.629 | [−0.694, −0.558] | −0.974 | [−1.077, −0.865] |
| Endothelial | −1.071 | [−1.204, −0.924] | −0.756 | [−0.970, −0.527] |
| Myeloid | −1.468 | [−1.648, −1.263] | −0.070 | [−0.318, +0.209] |
| **CAFs** | **−4.046** | [−4.229, −3.863] | −0.596 | [−0.810, −0.383] |

**Own state.** Cancer Epithelial, T-cells and B-cells raise the odds of cycling;
CAFs lower them dramatically. Every interval excludes zero.

**Neighbourhood.** Only `nbr[Cancer Epithelial]` is large (+3.117): *being
surrounded by tumour* raises a cell's odds of cycling, over and above its own
state. T-cells and PVL neighbourhoods are indistinguishable from zero.

## The interaction `B` (spec 3.2: report `B`, never `U` or `V`)

Singular values **1.364, 0.858**, and zero thereafter — genuinely rank 2, against
a null reference of 0.064 from Test 02. Bootstrap CI on `sigma_1`:
[1.194, 1.716].

Rows are own state, columns neighbour state; the entry modifies the neighbour
effect for a cell of that own type.

| own \ nbr | Endo | CAFs | PVL | B | T | Mye | NormEp | Plasma | CancEp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Endothelial | −0.027 | −0.130 | −0.298 | −0.004 | −0.056 | −0.037 | +0.204 | +0.108 | +0.240 |
| CAFs | +0.004 | −0.036 | −0.087 | +0.003 | −0.011 | −0.002 | +0.057 | +0.012 | +0.059 |
| PVL | +0.076 | −0.051 | −0.149 | +0.032 | +0.014 | +0.047 | +0.088 | −0.095 | +0.037 |
| B-cells | −0.190 | −0.119 | −0.217 | −0.067 | −0.120 | −0.151 | +0.174 | +0.359 | +0.332 |
| T-cells | −0.063 | −0.087 | −0.184 | −0.020 | −0.056 | −0.057 | +0.133 | +0.143 | +0.191 |
| Myeloid | +0.039 | −0.064 | −0.165 | +0.018 | −0.005 | +0.019 | +0.105 | −0.030 | +0.083 |
| Normal Epithelial | +0.280 | +0.090 | +0.117 | +0.103 | +0.148 | +0.211 | −0.120 | −0.487 | −0.342 |
| Plasmablasts | −0.208 | +0.199 | +0.544 | −0.089 | −0.019 | −0.121 | −0.333 | +0.229 | −0.201 |
| Cancer Epithelial | +0.089 | +0.198 | +0.439 | +0.025 | +0.105 | +0.090 | −0.307 | −0.239 | −0.399 |

The largest entries say the neighbourhood effect is **context-dependent**: a
B-cell next to plasmablasts or tumour has its odds raised further (+0.359,
+0.332), while normal epithelium next to plasmablasts has them lowered (−0.487).
Per-entry CIs are not yet extracted — treat individual cells as indicative and
read the block through its singular values, which are the identified objects.

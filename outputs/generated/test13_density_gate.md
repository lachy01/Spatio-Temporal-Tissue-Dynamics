# Test 13 — density-in-weight sensitivity (spec 6.13)

Four arms, 20 restarts each, all 80 converged. The density block in the mixing
weight is now 45 parameters (a 5-column spline plus a per-phenotype 8x5
interaction), so spec 3.4's warning applies with force: density and `phi_bar`
are both spatially smooth on similar scales, and a flexible `h()` can absorb
genuine neighbourhood signal.

## Does the neighbourhood effect survive removing density?

| cell type | with density | `h()` dropped | change | |
|---|---:|---:|---:|---|
| **Cancer Epithelial** | **+3.0125** | **+3.1166** | **+3.5%** | |
| Plasmablasts | −0.9535 | −0.9565 | −0.3% | |
| Myeloid | −0.1105 | −0.1437 | −30.1% | |
| PVL | +0.1390 | +0.0824 | −40.7% | |
| Endothelial | −0.6476 | −0.9201 | −42.1% | |
| Normal Epithelial | −0.5858 | −0.8618 | −47.1% | |
| B-cells | −0.5406 | −0.2743 | +49.2% | |
| CAFs | −0.4859 | −0.8796 | −81.0% | flagged |
| T-cells | +0.1723 | +0.8371 | +385.7% | flagged |

`sigma_1(B)` 1.3805 -> 1.3010 (−5.8%); `psi` +0.788 -> +0.775; fold change
5.6294 -> 5.6328; mean `pi` 0.1148 -> 0.1143.

**PASS on the gate.** The headline neighbourhood coefficient moves 3.5%, so the
"being surrounded by tumour raises cycling odds" result does not depend on the
density term. Two small coefficients are flagged — CAFs and T-cells — and both
are near zero in absolute terms, which is where spatial confounding bites.

Making `h()` eightfold more flexible did **not** cost more neighbourhood signal:
the previous 5-parameter density block flagged three coefficients (B-cells,
T-cells, Myeloid); this 45-parameter one flags two.

## How much is the density term worth?

| knots | params | log-likelihood | vs previous | per param |
|---|---:|---:|---:|---:|
| none (`h()` dropped) | 60 | −625,220.0 | — | — |
| `[50]` | 73 | −625,072.0 | +148.0 | 11.4 |
| `[25, 50, 75]` | 97 | −624,942.5 | +129.5 | 5.4 |
| `[25, 50, 75, 90]` | 105 | −624,938.1 | +4.4 | 0.6 |

Density is worth 281.9 log-likelihood for 45 parameters (6.3 per parameter,
against the ~0.5 that pure overfitting yields). The q90 knot earns nothing —
0.6 per parameter is the overfitting floor.

**The contact-inhibition finding does not depend on that knot.** Cancer
Epithelial peaks at the median density and then falls by −0.327 log-odds with
knots at `[25,50,75]` and −0.315 with `[25,50,75,90]`. The turn-down is visible
without the high knot, so it is not an artefact of extra freedom in the tail.

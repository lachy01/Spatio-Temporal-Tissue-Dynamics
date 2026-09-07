# Spec 8.2 — archetype-conditional neighbourhood effects

Odds ratio of cycling per IQR of each neighbourhood component, for a cell whose own state sits at the given archetype centroid. `c + B^T phi_a`, mapped back to cell types. 95% CIs from the spatial block bootstrap; **bold** where the CI excludes 1.

| own archetype | Endothelial | CAFs | PVL | B-cells | T-cells | Myeloid | Normal Epithe | Plasmablasts | Cancer Epithe |
|---|---|---|---|---|---|---|---|---|---|
| Endothelial | 0.85 | **0.70** | **0.33** | 0.87 | 1.14 | 1.15 | **1.18** | **0.02** | **11.44** |
| CAFs | **0.59** | **0.79** | **0.61** | 0.81 | 1.18 | 1.04 | 1.02 | **0.03** | **7.58** |
| PVL | 0.68 | **0.76** | **0.49** | 0.83 | 1.16 | 1.08 | 1.08 | **0.02** | **8.80** |
| B-cells | **0.57** | **0.79** | **0.58** | 0.80 | 1.16 | 1.02 | 1.04 | **0.04** | **8.02** |
| T-cells | **0.55** | **0.80** | **0.65** | 0.80 | 1.17 | 1.01 | 1.01 | **0.04** | **7.38** |
| Myeloid | 0.67 | **0.76** | **0.49** | 0.83 | 1.16 | 1.08 | 1.08 | **0.03** | **8.79** |
| Normal Epithelial | 0.69 | **0.77** | **0.51** | 0.83 | 1.18 | 1.09 | 1.07 | **0.02** | **8.40** |
| Plasmablasts | **0.24** | 1.07 | **2.82** | 0.67 | 1.27 | 0.80 | **0.73** | **0.18** | **2.73** |
| Cancer Epithelial | **0.18** | 1.23 | **5.70** | 0.63 | 1.35 | 0.75 | **0.62** | **0.22** | **1.62** |

42 of 81 entries have a CI excluding 1.

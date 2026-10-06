### Data contract (spec section 1)

- cells: **561,897** (dropped: 0 for `N == 0`, 2107 outside the oxygen tissue mask)
- Tirosh genes present in panel: **53**
- Tirosh share of total library: **0.83%**
- `y_count` zero fraction: **62.2%**
- phi rowsum: mean -7.0000, sd 3.411e-15 -> constant: **True** (spec 3.1 branch (a), Q projection)
- `corr(log N, log area)` = **0.7072** -> orthogonalisation applied: **False**
- density radius 80.0 um; neighbour radius 80.0 um

`y_count` marginal:

| count | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21+ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cells | 349,427 | 113,597 | 41,729 | 18,936 | 10,503 | 6,289 | 4,184 | 2,929 | 2,157 | 1,758 | 1,389 | 1,162 | 1,007 | 781 | 733 | 651 | 565 | 517 | 432 | 393 | 302 | 2,456 |

**Deviations from spec:**

1. **Density and neighbourhood (spec 1.3 / 1.1).** Kept at this project's radius definitions (80.0 um density, 80.0 um neighbour mean) rather than fixed-k kNN, so the mixture is comparable with the existing NB models. Spec 1.3's own first sentence asks for the existing definition; its fixed-k preference is tested as a Stage D sensitivity. User-approved.
2. **Cycling archetypes (not in spec).** The four explicitly cycling celltype_minor components [] are dropped from the cell's OWN state. Their reference centroids are defined by cell-cycle genes and they correlate 0.47-0.57 with log1p(y_count), so leaving them in the mixing weight would predict P(cycling) from a cycling score built on the outcome's own transcripts. They are kept in the neighbourhood state, where phi_bar excludes the cell itself. User-approved.

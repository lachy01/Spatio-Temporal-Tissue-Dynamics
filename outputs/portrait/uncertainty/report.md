# Portrait uncertainty (portrait spec 8)

## Parametric bootstrap (spec 8.1, block-cluster sandwich): 500 replicates

### barycentric state

Topology preserved (in-support fixed points, spec 8.3 headline): **36%**; all interior fixed points: 1%. Point estimate's in-support key: ['CAFs x Myeloid|q50|saddle'].

| slice | class | in support | n (point) | found | same class | n 95% range |
| --- | --- | --- | --- | --- | --- | --- |
| Cancer Epithelial x CAFs q25 | unstable | False | 57.9 | 0.812 | 0.264 | 24.4 to 163 |
| Cancer Epithelial x Myeloid q25 | saddle | False | 7.63 | 0.636 | 0.492 | 7.48 to 187 |
| CAFs x Myeloid q50 | saddle | True | 66.3 | 0.864 | 0.708 | 54.8 to 91.1 |
| Cancer Epithelial x CAFs q50 | stable | False | 84.1 | 0.926 | 0.73 | 64 to 185 |

| archetype | point | share with ceiling | ceiling n 95% | share with threshold | threshold n 95% |
| --- | --- | --- | --- | --- | --- |
| Endothelial | unstable 129; stable 189 | 0.91 | 21.4 to 228 | 0.852 | 69.1 to 149 |
| CAFs | unstable 71 | 0.332 | 94.1 to 280 | 1 | 66 to 77.4 |
| PVL | shrinks | 0.394 | 16.9 to 134 | 0.458 | 28.1 to 282 |
| B-cells | stable 132 | 1 | 58.2 to 159 | 0.522 | 13.7 to 139 |
| T-cells | stable 154 | 1 | 44.9 to 164 | 0.54 | 20.2 to 131 |
| Myeloid | unstable 83 | 0.44 | 160 to 278 | 1 | 71.4 to 100 |
| Normal Epithelial | stable 185 | 0.998 | 118 to 225 | 0.138 | 10.7 to 134 |
| Plasmablasts | stable 211 | 0.948 | 179 to 257 | 0.042 | 12.4 to 43.7 |
| Cancer Epithelial | stable 79 | 0.95 | 53 to 111 | 0.428 | 7.94 to 154 |

| archetype | point | 95% range |
| --- | --- | --- |
| Endothelial | 0.0368 | 0.035 to 0.0389 |
| CAFs | 0.0218 | 0.0208 to 0.0233 |
| PVL | 0.0559 | 0.0536 to 0.0586 |
| B-cells | 0.0895 | 0.0848 to 0.0938 |
| T-cells | 0.11 | 0.105 to 0.114 |
| Myeloid | 0.0424 | 0.0406 to 0.0442 |
| Normal Epithelial | 0.0917 | 0.0888 to 0.0945 |
| Plasmablasts | 0.114 | 0.11 to 0.118 |
| Cancer Epithelial | 0.203 | 0.199 to 0.207 |

Tumour ceiling on the composition path: point 85.67961614734219-173.86990948235965; lowest ceiling 95% 17.4 to 111, highest 91.7 to 215; replicates with folds 57%.

### counts state

Topology preserved (in-support fixed points, spec 8.3 headline): **50%**; all interior fixed points: 2%. Point estimate's in-support key: none.

| slice | class | in support | n (point) | found | same class | n 95% range |
| --- | --- | --- | --- | --- | --- | --- |
| Cancer Epithelial x CAFs q25 | unstable | False | 61.2 | 0.824 | 0.28 | 37.8 to 103 |
| Cancer Epithelial x Myeloid q25 | saddle | False | 21.2 | 0.654 | 0.494 | 20.9 to 179 |
| Cancer Epithelial x CAFs q50 | stable | False | 84.6 | 0.918 | 0.724 | 64.7 to 173 |
| Cancer Epithelial x T-cells q50 | saddle | False | 47.1 | 0.48 | 0.48 | 45.1 to 52.1 |

| archetype | point | share with ceiling | ceiling n 95% | share with threshold | threshold n 95% |
| --- | --- | --- | --- | --- | --- |
| Endothelial | unstable 129; stable 189 | 0.91 | 21.4 to 228 | 0.852 | 69.1 to 149 |
| CAFs | unstable 71 | 0.332 | 94.1 to 280 | 1 | 66 to 77.4 |
| PVL | shrinks | 0.394 | 16.9 to 134 | 0.458 | 28.1 to 282 |
| B-cells | stable 132 | 1 | 58.2 to 159 | 0.522 | 13.7 to 139 |
| T-cells | stable 154 | 1 | 44.9 to 164 | 0.54 | 20.2 to 131 |
| Myeloid | unstable 83 | 0.44 | 160 to 278 | 1 | 71.4 to 100 |
| Normal Epithelial | stable 185 | 0.998 | 118 to 225 | 0.138 | 10.7 to 134 |
| Plasmablasts | stable 211 | 0.948 | 179 to 257 | 0.042 | 12.4 to 43.7 |
| Cancer Epithelial | stable 79 | 0.95 | 53 to 111 | 0.428 | 7.94 to 154 |

| archetype | point | 95% range |
| --- | --- | --- |
| Endothelial | 0.0368 | 0.035 to 0.0389 |
| CAFs | 0.0218 | 0.0208 to 0.0233 |
| PVL | 0.0559 | 0.0536 to 0.0586 |
| B-cells | 0.0895 | 0.0848 to 0.0938 |
| T-cells | 0.11 | 0.105 to 0.114 |
| Myeloid | 0.0424 | 0.0406 to 0.0442 |
| Normal Epithelial | 0.0917 | 0.0888 to 0.0945 |
| Plasmablasts | 0.114 | 0.11 to 0.118 |
| Cancer Epithelial | 0.203 | 0.199 to 0.207 |

Tumour ceiling on the composition path: point 85.67961614734219-171.09371976562247; lowest ceiling 95% 16.4 to 107, highest 89 to 215; replicates with folds 54%.

## Spatial block bootstrap with refits (spec 8.2): 50 replicates

### barycentric state

Topology preserved (in-support fixed points, spec 8.3 headline): **38%**; all interior fixed points: 4%. Point estimate's in-support key: ['CAFs x Myeloid|q50|saddle'].

| slice | class | in support | n (point) | found | same class | n 95% range |
| --- | --- | --- | --- | --- | --- | --- |
| Cancer Epithelial x CAFs q25 | unstable | False | 57.9 | 0.78 | 0.38 | 33.7 to 102 |
| Cancer Epithelial x Myeloid q25 | saddle | False | 7.63 | 0.76 | 0.66 | 7.57 to 93.6 |
| CAFs x Myeloid q50 | saddle | True | 66.3 | 0.8 | 0.74 | 54.1 to 84.5 |
| Cancer Epithelial x CAFs q50 | stable | False | 84.1 | 0.94 | 0.74 | 61.2 to 171 |

| archetype | point | share with ceiling | ceiling n 95% | share with threshold | threshold n 95% |
| --- | --- | --- | --- | --- | --- |
| Endothelial | unstable 129; stable 189 | 0.96 | 21.8 to 228 | 0.92 | 80.9 to 148 |
| CAFs | unstable 71 | 0.36 | 97 to 212 | 1 | 66.7 to 76.3 |
| PVL | shrinks | 0.34 | 16.6 to 136 | 0.44 | 34.9 to 279 |
| B-cells | stable 132 | 0.96 | 22.8 to 166 | 0.5 | 19.1 to 135 |
| T-cells | stable 154 | 1 | 37.7 to 158 | 0.64 | 25.9 to 131 |
| Myeloid | unstable 83 | 0.46 | 164 to 264 | 1 | 72.1 to 97.1 |
| Normal Epithelial | stable 185 | 1 | 122 to 225 | 0.16 | 13 to 108 |
| Plasmablasts | stable 211 | 0.94 | 180 to 268 | 0.08 | 14.3 to 40.1 |
| Cancer Epithelial | stable 79 | 0.96 | 49.6 to 152 | 0.52 | 21.9 to 148 |

| archetype | point | 95% range |
| --- | --- | --- |
| Endothelial | 0.0368 | 0.0344 to 0.0393 |
| CAFs | 0.0218 | 0.0199 to 0.0244 |
| PVL | 0.0559 | 0.0515 to 0.0598 |
| B-cells | 0.0895 | 0.0834 to 0.0957 |
| T-cells | 0.11 | 0.105 to 0.118 |
| Myeloid | 0.0424 | 0.0404 to 0.0449 |
| Normal Epithelial | 0.0917 | 0.0879 to 0.0988 |
| Plasmablasts | 0.114 | 0.108 to 0.121 |
| Cancer Epithelial | 0.203 | 0.196 to 0.21 |

Tumour ceiling on the composition path: point 85.67961614734219-173.86990948235965; lowest ceiling 95% 44.9 to 184, highest 92.8 to 213; replicates with folds 40%.

### counts state

Topology preserved (in-support fixed points, spec 8.3 headline): **46%**; all interior fixed points: 0%. Point estimate's in-support key: none.

| slice | class | in support | n (point) | found | same class | n 95% range |
| --- | --- | --- | --- | --- | --- | --- |
| Cancer Epithelial x CAFs q25 | unstable | False | 61.2 | 0.7 | 0.34 | 42.9 to 87.6 |
| Cancer Epithelial x Myeloid q25 | saddle | False | 21.2 | 0.72 | 0.62 | 22.5 to 94.6 |
| Cancer Epithelial x CAFs q50 | stable | False | 84.6 | 0.92 | 0.68 | 63 to 163 |
| Cancer Epithelial x T-cells q50 | saddle | False | 47.1 | 0.64 | 0.64 | 46.5 to 54.8 |

| archetype | point | share with ceiling | ceiling n 95% | share with threshold | threshold n 95% |
| --- | --- | --- | --- | --- | --- |
| Endothelial | unstable 129; stable 189 | 0.96 | 21.8 to 228 | 0.92 | 80.9 to 148 |
| CAFs | unstable 71 | 0.36 | 97 to 212 | 1 | 66.7 to 76.3 |
| PVL | shrinks | 0.34 | 16.6 to 136 | 0.44 | 34.9 to 279 |
| B-cells | stable 132 | 0.96 | 22.8 to 166 | 0.5 | 19.1 to 135 |
| T-cells | stable 154 | 1 | 37.7 to 158 | 0.64 | 25.9 to 131 |
| Myeloid | unstable 83 | 0.46 | 164 to 264 | 1 | 72.1 to 97.1 |
| Normal Epithelial | stable 185 | 1 | 122 to 225 | 0.16 | 13 to 108 |
| Plasmablasts | stable 211 | 0.94 | 180 to 268 | 0.08 | 14.3 to 40.1 |
| Cancer Epithelial | stable 79 | 0.96 | 49.6 to 152 | 0.52 | 21.9 to 148 |

| archetype | point | 95% range |
| --- | --- | --- |
| Endothelial | 0.0368 | 0.0344 to 0.0393 |
| CAFs | 0.0218 | 0.0199 to 0.0244 |
| PVL | 0.0559 | 0.0515 to 0.0598 |
| B-cells | 0.0895 | 0.0834 to 0.0957 |
| T-cells | 0.11 | 0.105 to 0.118 |
| Myeloid | 0.0424 | 0.0404 to 0.0449 |
| Normal Epithelial | 0.0917 | 0.0879 to 0.0988 |
| Plasmablasts | 0.114 | 0.108 to 0.121 |
| Cancer Epithelial | 0.203 | 0.196 to 0.21 |

Tumour ceiling on the composition path: point 85.67961614734219-171.09371976562247; lowest ceiling 95% 44.9 to 173, highest 91.2 to 213; replicates with folds 46%.

cases 50, voxel agreement between the two models 0.985
| alpha (Swin weight) | mean Dice |
|---|---|
| 1 | 0.6843 |
| 0.9 | 0.6843 |
| 0.8 | 0.6842 |
| 0.7 | 0.6833 |
| 0.6 | 0.6806 |
| 0.5 | 0.6702 |
| 0.3 | 0.6303 |
| 0 | 0.6059 |

per organ: Swin alone (alpha 1), UNETR alone (alpha 0), best alpha 0.9

| organ | Swin | UNETR | ensemble | gain vs Swin |
|---|---|---|---|---|
| spleen | 0.834 | 0.760 | 0.834 | -0.000 |
| r_kidney | 0.858 | 0.812 | 0.859 | +0.000 |
| l_kidney | 0.820 | 0.776 | 0.820 | -0.000 |
| gallbladder | 0.550 | 0.355 | 0.551 | +0.001 |
| esophagus | 0.563 | 0.509 | 0.563 | -0.000 |
| liver | 0.911 | 0.867 | 0.911 | +0.000 |
| stomach | 0.670 | 0.552 | 0.671 | +0.000 |
| aorta | 0.787 | 0.685 | 0.786 | -0.001 |
| ivc | 0.724 | 0.663 | 0.724 | -0.000 |
| pancreas | 0.590 | 0.526 | 0.590 | +0.000 |
| r_adrenal | 0.490 | 0.421 | 0.491 | +0.000 |
| l_adrenal | 0.414 | 0.346 | 0.413 | -0.001 |

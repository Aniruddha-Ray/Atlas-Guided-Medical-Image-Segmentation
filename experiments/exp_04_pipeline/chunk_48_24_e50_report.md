# Run report: exp_04_pipeline/chunk_48_24_e50

Epochs completed: 50 of 50

| Measure | Value |
|---|---|
| Best validation Dice | **0.3138** (epoch 50) |
| Last validation Dice | 0.3138 |
| Atlas vote alone, same validation cases (96^3) | 0.4214 |
| Model minus atlas vote (best epoch) | -0.1075 |
| Model with the prior removed at evaluation (epoch 50) | 0.0577 |

Per organ at the best epoch (1 to 12):

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model | 0.66 | 0.68 | 0.63 | 0.01 | 0.00 | 0.83 | 0.50 | 0.00 | 0.00 | 0.46 | 0.00 | 0.00 |
| atlas vote | 0.55 | 0.65 | 0.57 | 0.14 | 0.26 | 0.75 | 0.44 | 0.58 | 0.59 | 0.32 | 0.16 | 0.05 |

Training curve (validation Dice): e1 0.012, e13 0.212, e26 0.240, e38 0.304, e50 0.314

Registration and prior validation: `chunk_48_24_e50_validation.md`.

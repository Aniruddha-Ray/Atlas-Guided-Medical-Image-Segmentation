# Run report: exp_04_pipeline/chunk_48_24_e50_gamma1

Epochs completed: 50 of 50

| Measure | Value |
|---|---|
| Best validation Dice | **0.5708** (epoch 50) |
| Last validation Dice | 0.5708 |
| Atlas vote alone, same validation cases (96^3) | 0.4214 |
| Model minus atlas vote (best epoch) | +0.1495 |
| Model with the prior removed at evaluation (epoch 50) | 0.0552 |

Per organ at the best epoch (1 to 12):

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model | 0.69 | 0.76 | 0.70 | 0.30 | 0.52 | 0.83 | 0.53 | 0.65 | 0.69 | 0.51 | 0.39 | 0.28 |

Training curve (validation Dice): e1 0.433, e13 0.518, e26 0.555, e38 0.566, e50 0.571

Registration and prior validation: `chunk_48_24_e50_gamma1_validation.md`.

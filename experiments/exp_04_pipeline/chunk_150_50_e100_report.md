# Run report: exp_04_pipeline/chunk_150_50_e100

Epochs completed: 100 of 100

| Measure | Value |
|---|---|
| Best validation Dice | **0.6059** (epoch 88) |
| Last validation Dice | 0.6054 |
| Atlas vote alone, same validation cases (96^3) | 0.4242 |
| Model minus atlas vote (best epoch) | +0.1817 |
| Model with the prior removed at evaluation (epoch 100) | 0.3019 |

Per organ at the best epoch (1 to 12):

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model | 0.76 | 0.81 | 0.78 | 0.35 | 0.51 | 0.87 | 0.55 | 0.69 | 0.66 | 0.53 | 0.42 | 0.35 |
| atlas vote | 0.57 | 0.67 | 0.63 | 0.15 | 0.27 | 0.74 | 0.41 | 0.57 | 0.57 | 0.33 | 0.13 | 0.05 |

Training curve (validation Dice): e1 0.487, e26 0.572, e51 0.588, e76 0.602, e100 0.605

Registration and prior validation: `chunk_150_50_e100_validation.md`.

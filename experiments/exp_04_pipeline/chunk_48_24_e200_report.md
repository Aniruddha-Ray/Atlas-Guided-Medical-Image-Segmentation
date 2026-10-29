# Run report: exp_04_pipeline/chunk_48_24_e200

Epochs completed: 51 of 200 (still running or interrupted)

| Measure | Value |
|---|---|
| Best validation Dice | **0.4032** (epoch 50) |
| Last validation Dice | 0.4003 |
| Model with the prior removed at evaluation (epoch 50) | 0.0569 |

Per organ at the best epoch (1 to 12):

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model | 0.67 | 0.75 | 0.66 | 0.29 | 0.00 | 0.82 | 0.50 | 0.00 | 0.67 | 0.48 | 0.00 | 0.00 |
| atlas vote | 0.55 | 0.65 | 0.57 | 0.14 | 0.26 | 0.75 | 0.44 | 0.58 | 0.59 | 0.32 | 0.16 | 0.05 |

Training curve (validation Dice): e1 0.012, e13 0.209, e26 0.294, e39 0.342, e51 0.400

Registration and prior validation: `chunk_48_24_e200_validation.md`.

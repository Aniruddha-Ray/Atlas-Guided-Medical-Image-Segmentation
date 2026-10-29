# Run report: exp_04_pipeline/chunk_48_24_e50_imageonly

Epochs completed: 50 of 50

| Measure | Value |
|---|---|
| Best validation Dice | **0.0602** (epoch 50) |
| Last validation Dice | 0.0602 |
| Atlas vote alone, same validation cases (96^3) | 0.4214 |
| Model minus atlas vote (best epoch) | -0.3611 |
| Model with the prior removed at evaluation (epoch 50) | 0.0602 |

Per organ at the best epoch (1 to 12):

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.72 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Training curve (validation Dice): e1 0.002, e13 0.041, e26 0.056, e38 0.059, e50 0.060

Registration and prior validation: `chunk_48_24_e50_imageonly_validation.md`.

# Run report: exp_04_pipeline/chunk_150_50_e100_swin_prior

Epochs completed: 100 of 100

| Measure | Value |
|---|---|
| Best validation Dice | **0.6963** (epoch 96) |
| Last validation Dice | 0.6959 |
| Atlas vote alone, same validation cases (96^3) | 0.4242 |
| Model minus atlas vote (best epoch) | +0.2721 |
| Model with the prior removed at evaluation (epoch 100) | 0.5847 |

Per organ at the best epoch (1 to 12):

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model | 0.85 | 0.86 | 0.82 | 0.58 | 0.58 | 0.92 | 0.69 | 0.80 | 0.73 | 0.60 | 0.49 | 0.43 |

Training curve (validation Dice): e1 0.513, e26 0.633, e51 0.669, e76 0.687, e100 0.696

Registration and prior validation: `chunk_150_50_e100_swin_prior_validation.md`.

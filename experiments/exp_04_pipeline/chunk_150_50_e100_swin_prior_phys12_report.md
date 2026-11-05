# Run report: exp_04_pipeline/chunk_150_50_e100_swin_prior_phys12

Epochs completed: 12 of 12

| Measure | Value |
|---|---|
| Best validation Dice | **0.6924** (epoch 1) |
| Last validation Dice | 0.6844 |
| Atlas vote alone, same validation cases (96^3) | 0.4242 |
| Model minus atlas vote (best epoch) | +0.2682 |
| Model with the prior removed at evaluation (epoch 12) | 0.5700 |

Per organ at the best epoch (1 to 12):

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model | 0.84 | 0.86 | 0.82 | 0.57 | 0.58 | 0.92 | 0.69 | 0.80 | 0.73 | 0.60 | 0.48 | 0.43 |

Training curve (validation Dice): e1 0.692, e4 0.679, e7 0.684, e10 0.685, e12 0.684

Registration and prior validation: `chunk_150_50_e100_swin_prior_phys12_validation.md`.

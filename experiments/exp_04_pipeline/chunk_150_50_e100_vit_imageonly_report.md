# Run report: exp_04_pipeline/chunk_150_50_e100_vit_imageonly

Epochs completed: 100 of 100

| Measure | Value |
|---|---|
| Best validation Dice | **0.3087** (epoch 99) |
| Last validation Dice | 0.3085 |
| Atlas vote alone, same validation cases (96^3) | 0.4242 |
| Model minus atlas vote (best epoch) | -0.1154 |
| Model with the prior removed at evaluation (epoch 100) | 0.3085 |

Per organ at the best epoch (1 to 12):

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model | 0.71 | 0.63 | 0.65 | 0.00 | 0.00 | 0.86 | 0.52 | 0.00 | 0.00 | 0.34 | 0.00 | 0.00 |

Training curve (validation Dice): e1 0.001, e26 0.085, e51 0.254, e76 0.292, e100 0.308

Registration and prior validation: `chunk_150_50_e100_vit_imageonly_validation.md`.

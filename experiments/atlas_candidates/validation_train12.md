# Registration and prior-insertion validation

Atlas-vote Dice is the overlap of argmax(P) with the label on the 96^3 training grid (evaluation only).

| Split | Cases | Vote Dice mean / median | Min / max | After augmentation | Centroid error (median) | Fallback atlases |
|---|---|---|---|---|---|---|
| train | 12 | 0.426 / 0.419 | 0.273 / 0.567 | 0.425 | 12.7 mm | 5 |
| all | 12 | 0.426 / 0.419 | 0.273 / 0.567 | 0.425 | 12.7 mm | 5 |

Per organ (1 to 12), all cases:

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote Dice | 0.59 | 0.68 | 0.58 | 0.20 | 0.25 | 0.76 | 0.43 | 0.56 | 0.56 | 0.27 | 0.17 | 0.06 |
| support recall | 0.92 | 0.97 | 0.96 | 0.67 | 0.81 | 0.97 | 0.88 | 0.82 | 0.97 | 0.84 | 0.73 | 0.70 |

By source: amos 0.417 (n=4), img 0.319 (n=2), s 0.468 (n=6)

**Invariants:** all passed

| Case | Split | Vote Dice | Augmented | 2 mm Dice | Fallbacks | Inliers (median) | Centroid err mm | w mean |
|---|---|---|---|---|---|---|---|---|
| amos_0296 | train | 0.314 | 0.312 | 0.336 | 1 | 125 | 17.5 | 0.96 |
| amos_0200 | train | 0.567 | 0.567 | 0.608 | 0 | 170 | 5.7 | 0.98 |
| amos_0085 | train | 0.384 | 0.384 | 0.421 | 0 | 122 | 11.3 | 0.98 |
| amos_0388 | train | 0.403 | 0.403 | 0.397 | 1 | 107 | 9.0 | 0.96 |
| s0339 | train | 0.378 | 0.376 | 0.394 | 1 | 120 | 15.3 | 0.97 |
| s1046 | train | 0.539 | 0.538 | 0.569 | 0 | 127 | 9.5 | 0.96 |
| s0078 | train | 0.436 | 0.430 | 0.439 | 0 | 118 | 13.1 | 0.97 |
| s0916 | train | 0.454 | 0.453 | 0.518 | 0 | 134 | 9.8 | 0.97 |
| s0275 | train | 0.566 | 0.559 | 0.581 | 0 | 159 | 12.3 | 0.96 |
| s0190 | train | 0.438 | 0.437 | 0.435 | 0 | 99 | 14.6 | 0.97 |
| img0033 | train | 0.273 | 0.270 | 0.305 | 1 | 118 | 34.5 | 0.97 |
| img0021 | train | 0.365 | 0.365 | 0.390 | 1 | 104 | 15.4 | 0.97 |

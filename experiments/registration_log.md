# Atlas → target registration log

Atlas-only quality of the registration stage, measured **before** any network
training. Target GT is used here for evaluation only, never to build priors.
All numbers are for the 10 temporary atlases in `atlas/atlas_selection.json`,
registered onto the first 2 validation cases (`amos_0278`, `s0703`) on a 2 mm
grid. This is a small local chunk, not a full-dataset result.

## Setup

- Atlas library: `scripts/build_atlas_library.py` gives ~220 reliable landmarks
  per atlas (`min_score=0.3`; see below for why not 0.5).
- Registration: `scripts/register_targets.py` runs coarse registration, then
  2-pass NCC patch matching, then RANSAC affine, then smoothing TPS.

## Run 1: initial pipeline (unconstrained coarse MI affine, 1 matching pass)

| Target | TPS / fallback | Majority-vote Dice |
|---|---|---|
| amos_0278 | 9 / 1 | 0.334 |
| s0703 | 9 / 1 | 0.328 |

Failure found: the unconstrained coarse affine collapsed to 0.19× z-scale on
one atlas pair (`img0004`) when the fields of view differed. Coarse centroid
errors were 12–30 mm, often beyond the ±20 mm patch-search window.

## Run 2: fixed coarse stage + second matching pass + TPS smoothing ladder

Coarse stage is now: z-offset search, then similarity, then an affine accepted
only if all axis scales stay within 0.8–1.25. A second NCC pass runs around the
RANSAC affine (±12 mm). The TPS λ ladder goes 5 → 5120.

| Target | TPS / fallback | RANSAC inliers per atlas | Majority-vote Dice |
|---|---|---|---|
| amos_0278 | 9 / 1 | 51–117 | 0.321 |
| s0703 | 10 / 0 | 58–149 | 0.321 |

The mechanics are much more robust, but majority-vote Dice is unchanged.

## Run 3: TPS + B-spline refinement (body-masked MI, LBFGS2, 80 → 40 mm mesh)

| Target | TPS only | TPS + B-spline |
|---|---|---|
| amos_0278 | 0.321 | **0.265** |
| s0703 | 0.321 | 0.316 |

**Negative result.** Refinement did not help and slightly hurt.

## Diagnostic: which stage helps? (amos_0278, single atlas)

| Atlas | Coarse affine | RANSAC affine | TPS |
|---|---|---|---|
| amos_0001 | 0.201 (MI −0.134) | **0.276** (MI −0.071) | 0.265 (MI −0.073) |
| amos_0004 | 0.164 (MI −0.134) | **0.306** (MI −0.107) | 0.284 (MI −0.102) |
| amos_0007 | 0.363 (MI −0.169) | **0.439** (MI −0.083) | 0.421 (MI −0.080) |

- **Landmark RANSAC affine** clearly beats intensity-only coarse registration,
  by +0.08 to +0.15 Dice.
- **TPS** is currently slightly below RANSAC affine. Inlier correspondence RMSE
  is ~7–8 mm, about the size of the deformation TPS is trying to model.
- **Whole-body MI is anti-correlated with organ overlap.** The best-MI mapping
  has the worst Dice. MI is dominated by body outline, bone and fat, which
  explains why MI-driven B-spline refinement cannot help.

## Other checks

- **Label semantics are consistent across sources** (amos / img / s): organ
  volumes and positions relative to the liver match for all 12 labels.
  1 spleen, 2 R kidney, 3 L kidney, 4 gallbladder, 5 esophagus, 6 liver,
  7 stomach, 8 aorta, 9 IVC, 10 pancreas, 11 R adrenal, 12 L adrenal.
- **Reliability threshold.** `min_score=0.5` is the 90th percentile and kept
  only ~38 landmarks per atlas, too few for RANSAC. `0.3` (median) keeps ~220.
- **Runtime** on this laptop (RTX 3050): ~110 s per target with TPS, ~220 s
  with B-spline, both with 10 atlases.

## Methodological note

Registration hyperparameters (TPS λ, RANSAC threshold, patch size, search
radius) must be tuned on **training** targets that are not atlases, never on
validation cases. Otherwise validation Dice is optimistically biased.

# Atlas → target registration log

Atlas-only quality of the registration stage, measured **before** any network
training. Target GT is used here for evaluation only, never to build priors.
All numbers are for the 10 temporary atlases in `atlas/atlas_selection.json`,
registered onto the first 2 validation cases (`amos_0278`, `s0703`) on a 2 mm
grid. This is a small local chunk, not a full-dataset result.

## Correction: the first three runs used a faulty TPS

`atlas/tps.py` had the smoothing term with the wrong sign (`K + λI` instead of
`K − λI`). For the kernel φ(r) = r, K is conditionally *negative* definite, so
the wrong sign is near-singular whenever λ matches an eigenvalue of −K. The
fits were erratic rather than smooth: on synthetic noisy data the control-point
residual went 10 → 1575 → 373 → 94 mm at λ = 5, 20, 50, 320 (a correct spline rises
monotonically from 0 to the plain affine fit, 19 mm there), with weights up to
~470.

How it was found: running the notebook's golden-transformation cell reported a
negative Jacobian determinant. A λ sweep on real atlas pairs then showed
residuals up to 4100 mm and 26 to 55 % of control points folding at λ = 50 and
320, which a smoothing spline should never do. My earlier "no folding" remark had checked only
20 points of one organ.

Fixed in `atlas/tps.py`, with a regression test
(`test_smoothing_moves_monotonically_from_interpolation_to_affine_fit`). The atlas
library and every cached registration were rebuilt (`--force`). No hyperparameter
was changed; only the sign.

| | before fix | after fix |
|---|---|---|
| amos_0278 atlas-only Dice | 0.321 | **0.445** |
| s0703 atlas-only Dice | 0.321 | **0.481** |
| TPS vs RANSAC affine (single atlas) | TPS slightly worse | TPS clearly better |
| golden-transform folding at control points | 1 to 10 % | 0 % at every λ tried |

Runs 1 to 3 and the first diagnostic table below are kept for history only. They
were produced with the faulty TPS, and their conclusion "TPS adds nothing" was an
artefact of the bug.

## Setup (current)

- Atlas library: `scripts/build_atlas_library.py`, golden-transform λ = 5,
  tolerance T = 15 mm. `min_score = 0.3` keeps ~300 landmarks per atlas, about
  76 % of them (see "Open tuning items").
- Registration: `scripts/register_targets.py`: coarse registration, 2-pass NCC
  patch matching, RANSAC affine, smoothing TPS.

## Run 4: after the TPS fix (current)

| Target | TPS / fallback | RANSAC inliers per atlas | Majority-vote Dice |
|---|---|---|---|
| amos_0278 | 9 / 1 | 87–163 (9 for the fallback) | **0.445** |
| s0703 | 10 / 0 | 97–189 | **0.481** |

Per organ (1 to 12):
- amos_0278: 0.73 0.65 0.71 0.00 0.55 0.75 0.40 0.36 0.66 0.48 0.02 0.03
- s0703: 0.78 0.81 0.80 0.00 0.00 0.79 0.40 0.62 0.65 0.30 0.50 0.10

The one fallback (`coarse_affine_only`, atlas 7 on amos_0278) had only 9 RANSAC
inliers out of 39 candidates. Across the 5 cached targets (2 val, 3 train) 49 of
50 registrations ended in a TPS. All 49 settled on the **first** λ rung (5),
with ~1.8 mm residual on their own inliers: the spline follows the RANSAC
inliers closely and never needed the extra smoothing ladder to avoid folding.
Organ 4 scores 0.00 on both cases. Organs 11 and 12 (only ~4 to 5 landmarks each)
are ~0.0 on amos_0278 but 0.50 and 0.10 on s0703, and organ 5 is 0.55 versus 0.00,
so two cases say little about individual organs.

## Diagnostic: which stage helps? (amos_0278, single atlas, after the fix)

| Atlas | Coarse affine | RANSAC affine | TPS |
|---|---|---|---|
| amos_0001 | 0.202 (MI −0.135) | 0.259 (MI −0.075) | 0.263 (MI −0.079) |
| amos_0004 | 0.164 (MI −0.134) | 0.315 (MI −0.106) | **0.389** (MI −0.106) |
| amos_0007 | 0.362 (MI −0.169) | 0.439 (MI −0.101) | **0.543** (MI −0.112) |

- **Landmark RANSAC affine beats intensity-only coarse registration** by +0.06 to
  +0.15 Dice, and **the TPS adds up to +0.10 on top of it** (equal for amos_0001).
- **Whole-body MI is anti-correlated with organ overlap.** The coarse stage has the
  best (lowest) MI and the worst Dice in every row. MI is dominated by body outline,
  bone and fat, so maximising it is the wrong objective for organ overlap. This
  held before and after the fix.

For the record, the same table before the fix: TPS 0.265 / 0.284 / 0.421, i.e.
slightly *below* RANSAC affine (0.276 / 0.306 / 0.439).

## Golden-transformation λ sweep (corrected TPS, 4 atlas pairs)

Mean residual over control points, and share of landmarks with reliability > 0.
The fold fraction was 0.0 % for every pair and every λ.

| Pair (i ← j) | λ = 5 | λ = 320 | λ = 5120 | λ = 100000 |
|---|---|---|---|---|
| 0 ← 3 | 8.7 mm, 88 % | 12.3 mm, 75 % | 13.6 mm, 68 % | 13.9 mm, 66 % |
| 0 ← 1 | 7.8 mm, 91 % | 11.6 mm, 81 % | 13.1 mm, 71 % | 13.4 mm, 68 % |
| 5 ← 3 | 10.4 mm, 77 % | 13.9 mm, 64 % | 16.4 mm, 50 % | 17.2 mm, 45 % |
| 1 ← 5 | 8.8 mm, 82 % | 13.0 mm, 66 % | 15.0 mm, 55 % | 15.7 mm, 53 % |

The residual is not ~0 even at λ = 5 because the nearest-neighbour correspondences
are many-to-one (several atlas-i points match the same atlas-j point), so they
cannot all be satisfied exactly. A larger λ makes the residual reflect
disagreement with a smooth global deformation rather than correspondence
ambiguity, which is closer to the paper's intent, but λ = 5 is kept as the default
until this is tuned (see below).

## Library scores (after the fix)

3925 landmarks over 10 atlases: score median 0.43, p90 0.62 (before the fix:
0.32 and 0.50). Share kept by threshold: ≥ 0.3: 76 %, ≥ 0.4: 57 %, ≥ 0.5: 33 %,
≥ 0.6: 13 %. Organs 11 and 12 keep ~4.7 and ~4.5 landmarks per atlas.

## B-spline refinement (dense, whole-body MI)

Re-run after the fix (same 2 targets, same settings: B-spline on body-masked mutual
information, LBFGS2, 80 → 40 mm mesh, initialised from the landmark TPS).

| Target | TPS only | TPS + B-spline |
|---|---|---|
| amos_0278 | 0.445 | 0.443 |
| s0703 | 0.481 | 0.458 |

The optimised metric improved in every one of the 20 pairs (for example
−0.083 → −0.095) and nothing folded, yet organ overlap did not improve. The
refinement is therefore not used: no gain, at about 1.8× the runtime (~370 s per
target against ~210 s). This agrees with the stage diagnostic above: whole-body
MI is the wrong objective for organ overlap. `--refinement bspline` stays in the
code as an ablation. The earlier version of this ablation (0.265 and 0.316
against 0.321) used the faulty TPS as its starting point and is void.

## History: Runs 1 to 3 (faulty TPS, do not cite)

| Run | Change | amos_0278 | s0703 |
|---|---|---|---|
| 1 | first pipeline (unconstrained coarse affine) | 0.334 | 0.328 |
| 2 | constrained coarse stage, second matching pass, λ ladder 5 → 5120 | 0.321 | 0.321 |
| 3 | + B-spline refinement on whole-body MI | 0.265 | 0.316 |

Run 1 also exposed a separate real problem, since fixed: the unconstrained coarse
affine collapsed to 0.19× z-scale on one atlas pair when the fields of view
differed, and coarse centroid errors (12 to 30 mm) often exceeded the ±20 mm
patch-search window. The z-offset search, similarity stage and scale limits fixed
that, and it is unrelated to the TPS sign.

## Other checks

- **Label semantics are consistent across sources** (amos / img / s): organ
  volumes and positions relative to the liver match for all 12 labels
  (checked on 6 cases per source). Organ names are inferred from volume and
  position, following the AMOS ordering, and are not confirmed against the
  dataset documentation: 1 spleen, 2 R kidney, 3 L kidney, 4 gallbladder,
  5 esophagus, 6 liver, 7 stomach, 8 aorta, 9 IVC, 10 pancreas, 11 R adrenal,
  12 L adrenal.
- **Runtime** on this laptop (RTX 3050): ~190 to 250 s per target with TPS (10 atlases,
  ~300 landmarks each), ~370 s with the B-spline variant. Earlier figures of
  ~110 s came from the smaller pre-fix landmark set (~220 per atlas).

## Open tuning items

All of these must be tuned on **training** targets that are not atlases, never on
validation cases; otherwise validation Dice is optimistically biased. Nothing
below was tuned on validation. The only change since the first runs is the sign
fix, which was motivated by an atlas-only consistency check.

- **Golden-transform λ and tolerance T** (sweep above). Decides how discriminating
  the reliability scores are.
- **Reliability threshold.** 0.3 was chosen when the score median was 0.32; after
  the fix it keeps 76 % of landmarks, so it is now barely selective.
- **Registration TPS λ.** Always the first rung (5), i.e. near-interpolation of
  RANSAC inliers. A larger λ may generalise better or worse; untested.
- **RANSAC threshold (12 mm), patch size, search windows.**
- **Small organs** (4, 11, 12) are still the weakest part.

## Methodological note

Two validation targets is a very small chunk, and these two cases were also the
ones inspected while building the pipeline. Treat the numbers as evidence that
each stage works and of its rough size, not as an estimate of full-set
performance.

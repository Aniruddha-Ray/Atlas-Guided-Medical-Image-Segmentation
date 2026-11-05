# Cascade study: plan and model walkthrough

Started 2026-11-05. Replaces the epoch-efficiency framing (AMOS fold 2: the CT-to-CT prior led early, +0.55 pseudo-Dice at epoch 0, but tied image-only by epoch ~60; image-only reached 0.858 at 100 epochs).
Goal now: a better segmentation result, tested on a different dataset, with Dice and HD95.

## 1. Question and claim

Does a label-to-label atlas prior, built from a first-stage prediction, improve segmentation over (A) image-only and (B) the same cascade without atlases?

- **C must beat B.** If the plain cascade matches C, the atlases add nothing.
- Expected gain is in HD95, missed organs and worst cases rather than mean Dice.

## 2. Data and split

| Set | Cases | Role |
|---|---|---|
| AMOS atlases | 30 | Atlas library (most typical cases that pass quality filters; includes the 10 existing atlases). `atlas/atlas_selection_amos30.json` |
| AMOS stage 1 | 42 | Train the stage-1 model (spent: never used for validation, predictions would be in-sample) |
| AMOS stage 2 | 200 | Train stage 2 (fold 0) |
| BTCV | 30 | Test set for every arm (Dice, HD95) |

Split file: `experiments/exp_cascade/cascade_split.json` (seed 42). Merged dataset Dataset060 is never modified.
Datasets: 080 stage 1; 081 A image-only; 082 B plain cascade; 083 C atlas cascade. All share the nnUNetPlans_4G plans (same CT normalisation, patch, batch).

## 3. Pipeline (scripts/cascade_queue.py, resumable, markers in `<root>/cascade/_done_<step>`)

1. **wait**: for Dataset073 prior fold 2 to finish.
2. **split**: write the split (skipped if present).
3. **btcv_reg**: CT-to-CT register the 30 BTCV cases to the 10 stage-1 atlases (~1 h).
4. **s1_build / s1_prep**: build Dataset080, fingerprint, move plans from 073, preprocess.
5. **s1_train**: `nnUNetTrainerAtlasPrior_50epochs`, 42 cases, validated on BTCV.
6. **s1_pred**: predict 200 + 30 cases (no TTA).
7. **s1_eval**: stage-1 Dice and HD95 on BTCV.
8. **priors**: label-to-label registration of 30 atlases to each prediction (~2-2.5 h with 8 workers).
9. **s2_build / s2_prep**: datasets 081-083.
10. **train_C, train_A, train_B**: 100 epochs each on 200 cases.
11. **eval**: `experiments/exp_cascade/results_btcv.{csv,md}`.

Pause or resume at any time: `python scripts/amos_pause.py pause|resume|status`. Pause stops between steps, or after the current epoch during training (exit code 75, resume with `--c`).

## 4. Model walkthrough

### 4.1 Backbone
nnU-Net v2.8.1, 3d_fullres PlainConvUNet, plans `nnUNetPlans_4G`, 12 organs + background, `nnUNet_compile=f`, 250 iterations per epoch. Extension: `nnunet_ext/nnUNetTrainerAtlasPrior.py` (reinstall with `python nnunet_ext/install.py` after edits).

### 4.2 Input: CT plus a 25-channel prior ("pnw" layout, uint8 x250)
| Channels | Meaning |
|---|---|
| P_1..12 | Probability of each organ at each voxel |
| N_1..12 | Nearness: 1 - min(d,50)/50, d = distance (mm) to {P_k >= 0.5 x peak} |
| w | Confidence: 1 - H/log C (1 when the prior is one-hot) |

Prior dropout 0.3 during training. Intensity augmentation is applied to the CT only.

### 4.3 Output fusion
`logits' = logits + gamma_k * w * log(P_k + 1e-3)`, gamma learned per class. The log prior is computed under `no_grad`, in half precision when autocast is on (4 GB GPU).

### 4.4 Where the prior comes from, per model

| Model | Prior | How built |
|---|---|---|
| Stage 1 | CT-to-CT, 10 atlases | Existing pipeline (`register_targets.py`), top-10 fused by image similarity |
| A (image-only) | none | 1 input channel, `nnUNetTrainerResumable_100epochs` |
| B (plain cascade) | stage-1 prediction, one-hot | P = one-hot, N from distances, w = 1 |
| C (atlas cascade) | 30 atlases warped to the stage-1 prediction, fused | Section 4.5 |

### 4.5 Label-to-label registration (atlas/label_registration.py)
1. Global alignment: similarity transform from organ centroids shared by atlas and prediction (translation only if fewer than 4 common organs).
2. Per-organ affine on signed distance maps (MeanSquares, 50 iterations, 10% sampling), 2 mm working grid for small organs (gallbladder, esophagus, adrenals; labels 4, 5, 10-12) and 3 mm for large ones. Falls back to the global transform when the affine is implausible (scale > 2, shift > 40 mm).
3. An organ missing from the prediction is placed by the global transform.
4. Weight per atlas and organ = Dice between the warped atlas organ and the prediction; top-10 fusion per organ.
5. Build P, N, w channels and write `_0001.._0025` (resumable), with QA JSON per case in `experiments/exp_cascade/label_reg/`.

Limit: the per-organ affine keeps the atlas's organ shape, so even a perfect prediction gives a fused prior Dice of ~0.77-0.80 (CT-to-CT prior: 0.456). With real stage-1 predictions it will be lower. This is deliberate: it does not copy errors from the prediction.

### 4.6 Metrics (scripts/eval_seg.py)
Per organ and case: Dice and HD95 (max of the two directed 95th-percentile surface distances, mm). An organ missing from prediction or truth gets Dice 0 and HD95 = image diagonal, and is counted as "missed" or "false organ". Reports per-organ means, missed organs, worst case, and cases with mean Dice < 0.7.

## 5. Fairness rules
- A, B and C get the same 200 cases, epochs, plans and seed.
- The image-only baseline gets every labelled case the pipeline uses (the 42 stage-1 cases are used by stage 1 only; stage 2 sees 200 for all arms).
- Stage-1 predictions of the 200 stage-2 cases come from a model that never saw them.
- BTCV is never used for training stage 2 or for choosing atlases. (Stage 1 validates on BTCV, so BTCV stage-1 numbers are for reporting only.)

## 6. Time estimate
Before stage-2 training about 6-16 h, then about 27 h or more for the three arms at normal speed (more if the Windows GPU memory spill to shared RAM returns). Total about 2-3 days.

## 7. Checks done
- 14 tests pass (tests/test_nnunet_ext.py, tests/test_label_registration.py).
- Smoke test on amos_0001 and amos_0004 with ground truth as a stand-in prediction: fused prior Dice 0.774 and 0.802, 220-290 s per case.
- Two bugs fixed first: refinement silently never ran (transform type error and an optimizer that stopped after 8 iterations), then registration was too slow (45-108 s per atlas, now ~7 s).

## 8. Open items
- Update RUN_GUIDE.md, research_audit.md, registration_log.md, novelty.md, prototype.ipynb for this study.
- Point 5 (learning curve; folds 1-3 of stage 2 exist at 25/50/100 cases but are not queued) and point 6 (converged baseline, ResEnc, resolution): instructions pending from the user.
- Nothing committed or pushed.

## 9. Files
`atlas/label_registration.py`, `scripts/cascade_split.py`, `scripts/cascade_make_datasets.py`, `scripts/cascade_label_priors.py`, `scripts/eval_seg.py`, `scripts/cascade_queue.py`, `scripts/amos_pause.py`, `tests/test_label_registration.py`.

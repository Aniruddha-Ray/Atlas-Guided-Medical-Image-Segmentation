# Research Audit — Atlas-Guided ViT-UNETR

Snapshot as of this audit. The repo lives at
`github.com/Aniruddha-Ray/Atlas-Guided-Medical-Image-Segmentation` (branch `main`,
last commit `3a86826 "Clean 100 epoch commit"`). Everything described below has
now been extracted out of the single notebook (`AtlasGuidedUNETR_fixed.ipynb`)
into importable modules (`data/`, `models/`, `atlas/`) so it can be tested and
reused from `scripts/`; the notebook itself is otherwise unchanged and still
trains/validates exactly as before.

## A. Current architecture

`AtlasGuidedViTUNETR` (`models/segmentation_model.py`) subclasses MONAI's
`UNETR` and keeps its encoder1-4 / decoder2-5 / segmentation head untouched.
`self.vit` is replaced with `CustomViT`, a MONAI `ViT` whose 12
`TransformerBlock`s are all built with `with_cross_attention=True` (MONAI's
built-in cross-attention — not a hand-rolled one). `AnatomicalPriorGenerator`
mean-pools `probability_maps`/`distance_maps` per 16×16×16 ViT patch
(`avg_pool3d`) and projects the pooled `[P, D]` vector through a 2-layer MLP
into a `hidden_size`-dim anatomical token, fed to every block as
cross-attention K/V (image tokens are Q; anatomical info never gets added
directly to Q). This matches the research report's architecture sections
(1.9–1.19) closely. Config actually used: `hidden_size=768, mlp_dim=3072,
num_heads=12, feature_size=16, img_size=96³, patch_size=16³` (216 patches),
`NUM_CLASSES=13` (12 organs + background).

## B. Current data flow

```
CT (.nii.gz) --LoadImaged/EnsureChannelFirstd/Orientationd(RAS)-->
  CropForegroundd(source=image) --Spacingd(1mm iso)-->
  ScaleIntensityRanged(HU[-1000,1000] -> [0,1]) --Resized(96^3)-->
  [train only: RandSpatialCropd(no-op, roi==size)/RandFlipd x3/RandRotated] -->
  CastToTyped --> model --> logits (13, 96,96,96)
```
(`data/transforms.py::build_transforms`, ported from notebook cell 14.)

## C. Current anatomical prior flow — P(v)/D(v)

**Confirmed: ground-truth leakage.** In the same transform chain, the label
is one-hot encoded (`AsDiscreted(to_onehot=13)`) and used directly as `P(v)`;
`CustomSpatialDistanceTransformd` runs `scipy.ndimage.distance_transform_edt`
directly on that one-hot ground truth to produce `D(v)`. Both are fed to the
model as `probability_maps`/`distance_maps` in *training and validation*. The
model is given the answer as an input feature the whole time — the reported
Dice numbers describe a supervised approximation with an oracle anatomical
prior, not the atlas-derived design in the research report.

A real, non-GT pipeline (`atlas/atlas_pipeline.py`, ported from an
uncommitted notebook draft — see Section F) already implements
atlas-to-target registration (SimpleITK affine + B-spline), label warping,
multi-atlas voting → `P(v)`, entropy confidence → `w(v)` (Section 12/13
formula, exact match), and an argmax-voted distance map → `D(v)`. **It is not
wired into any dataloader or the training loop.** There is also currently no
atlas/target split anywhere in the data — `Dataset060_Merged_Def` is 680
fully-labeled cases used as one homogeneous pool (`atlas/atlas_pipeline.py`
now has `select_atlas_library()` to make that split explicit and
deterministic, but nothing calls it yet).

## D. Current loss

```
L = DiceCELoss(softmax=True, to_onehot_y=False)(logits, one_hot_label)
```
Pure supervised Dice+CE. None of `Latlas`, `Ledge`, `LHU`, `Lsmooth` exist.

## E. Checkpoint compatibility

- `best_model.pth` (state_dict only) **is** the epoch-101 checkpoint: Dice
  peaked at epoch 101 (0.6505) and epochs 102–103 were both lower, so it was
  never overwritten after that.
- `latest_checkpoint.pth` is a full training-state dict
  (`epoch`/`model_state_dict`/`optimizer_state_dict`/`best_dice`) — inspected
  directly: `epoch: 102` (0-indexed) → printed as "Epoch 103/200", i.e.
  training actually continued 2 epochs past the reported stopping point
  before it was halted. `best_dice` stored inside it is still 0.6505.
- Both load cleanly into `AtlasGuidedViTUNETR(in_channels=1, out_channels=13,
  img_size=(96,96,96), feature_size=16, hidden_size=768, mlp_dim=3072,
  num_heads=12)` — 258 state_dict keys, verified directly against the
  instantiated model config; no shape/name mismatch expected since the
  checkpoint was produced by this exact class.
- Load via `model.load_state_dict(torch.load("best_model.pth"))` (or, for
  `latest_checkpoint.pth`, `torch.load(...)["model_state_dict"]`). See
  `scripts/evaluate_checkpoint.py`.

## F. Research gap table

| Research Component | Existing Code | Status | Missing |
|---|---|---|---|
| Golden Transformation (atlas↔atlas) | none | Not started | Landmark/feature correspondence, TPS fit between atlas pairs |
| Reliable feature selection (residual `s_k`, Top-K) | none | Not started | Feature descriptors, residual scoring, thresholding |
| RANSAC outlier rejection | `atlas/atlas_pipeline.py::register_atlas_to_target` uses MI-based affine optimization instead | Substituted | True RANSAC over sparse correspondences (current approach is dense intensity-based, not feature+RANSAC) |
| Affine + non-rigid refinement | `register_atlas_to_target` (SimpleITK affine + B-spline) | Implemented, unwired | Wiring into training/eval; registration-quality validation |
| Label warping | `warp_label` | Implemented, unwired | Wiring into dataset; caching |
| Multi-organ probability map `P(v)` | `build_atlas_priors` (voting) | Implemented, unwired | An actual atlas/target split (`select_atlas_library` exists but is uncalled) |
| Confidence `w(v)` | `build_atlas_priors` (entropy formula, matches report) | Implemented, unwired | Use in `Latlas` (not consumed anywhere) |
| Distance map `D(v)` | `build_atlas_priors` (EDT on voted argmax — correct semantics) | Implemented, unwired | — |
| Anatomical MLP / prior token | `AnatomicalPriorGenerator` | Implemented & trained | — |
| Cross-attention | `CustomViT` (MONAI `TransformerBlock(with_cross_attention=True)`) | Implemented & trained | — |
| Transformer encoder | `CustomViT`, 12 blocks | Implemented & trained | — |
| UNETR decoder | Stock MONAI `UNETR` decoder2-5 | Implemented & trained | — |
| Supervised loss `Lsup` | `DiceCELoss` | Implemented & trained | — |
| Atlas consistency loss `Latlas` | none | Not started | Everything |
| CT edge/boundary loss `Ledge` | none | Not started | Everything |
| HU consistency loss `LHU` | none | Not started | Everything (images are rescaled to [0,1] via `ScaleIntensityRanged`, monotonic — usable if applied consistently) |
| Smoothness loss | none | Not started | Everything |
| Caching (registrations/warped labels/P/D) | none | Not started | Everything |
| Atlas vs. target data split | `select_atlas_library()` (deterministic, seeded) | Function exists, uncalled | A decision on `atlas_count` and which pipeline stage calls it |

## G. Data leakage audit — and a confirmed bug

**Training/validation leakage of GT labels into P(v)/D(v): confirmed**, both
during training and validation (Section C). Per Section 29 of the brief, this
makes every reported Dice number (including 0.6505) a measurement of the
architecture with an oracle prior, not of the atlas-guided design.

**Train/validation split leakage: none found.** `random.seed(42)` + shuffle +
80/20 split on the full case list happens once, and the same seed/logic is
reused verbatim in `scripts/evaluate_checkpoint.py::build_val_files` — no
case appears in both splits, and re-running with the same seed reproduces
the same split deterministically.

**Bug found and fixed:** `CustomSpatialDistanceTransformd` had
`to_closest_foreground=True` call `distance_transform_edt(binary_mask)`.
`scipy.ndimage.distance_transform_edt` measures, for every *nonzero* element,
the distance to the nearest *zero* element — so this returned the distance
from inside the organ to its own boundary (and exactly 0 for every background
voxel), the opposite of the intended `D_c(v) = min_{y∈O_c} ||v−y||` (0 inside
the organ, growing outward). Practically: every epoch-101/103 training and
validation step fed the model a distance channel that carried almost no
positional signal outside organ boundaries — plausibly part of why the
smallest organs (10, 11, 12: Dice 0.55 / 0.45 / 0.40) underperformed, since a
locating signal is most valuable for small/hard-to-find structures.

Fixed in `data/transforms.py` (now the single source of truth, imported by
both the notebook and `scripts/evaluate_checkpoint.py`), with a
`legacy_buggy_distance` flag that reproduces the exact original behavior —
needed only to faithfully reproduce the historical epoch-101 Dice number
against the already-trained `best_model.pth` (see `baseline_reproduction.md`).
**Any new training run should leave `legacy_buggy_distance` at its default
(`False`).** Resuming training from `latest_checkpoint.pth` after this fix is
a real distribution shift for the model (it was fine-tuned against the buggy
signal for 103 epochs) — treat that as a new, deliberately-noted experiment,
not a seamless continuation.

Independently, `atlas/atlas_pipeline.py::build_atlas_priors` already computed
its distance map correctly (`distance_transform_edt(voted_label != c)`), so
that code was never affected by this bug.

## H. Update: atlas pipeline implemented (supersedes parts of Sections C and F)

**Status of the gap table (Section F).**

| Research component | Now implemented in | Status |
|---|---|---|
| Atlas set (10 temporary atlases) | `scripts/select_atlas_cases.py`, `atlas/atlas_selection.json` | Done. All 12 organs present, source-balanced (4 amos / 3 img / 3 s), drawn from the training split only, excluded from training targets |
| Landmarks | `atlas/landmarks.py` | Surface farthest-point sampling for large organs, interior points for organs under 8 mL (11 and 12 are adrenal-sized, 2.6–5.3 mL) |
| Golden transformation | `atlas/golden_transform.py`, `atlas/tps.py` | Kabsch pre-alignment on organ centroids, per-organ nearest-neighbour correspondence, regularised 3D TPS |
| Reliable features (`s_k`, selection) | `atlas/reliable_features.py` | Score averaged over all 9 other atlases, threshold 0.3 (≈ median), per-organ coverage floor |
| Atlas → target registration | `atlas/matching.py`, `atlas/ransac.py`, `atlas/registration.py` | z-search, similarity, constrained affine → 2-pass NCC patch matching → RANSAC affine → smoothing TPS with a folding check. Optional B-spline refinement (no gain, see `experiments/registration_log.md`) |
| Label warping | `atlas/label_warping.py` | Pull-back through the target → atlas mapping, cached as a 4D stack per target |
| P(v), D(v), w(v) | `data/transforms.py::AtlasPriorsd` | Computed in the 96³ training space after augmentation. D is in mm, clipped at 50 mm and normalised |
| Atlas-consistency / CT losses | — | Not started (Experiments 2–4) |

**Leakage status.** With `--prior-source atlas`, no target label is used to build
priors, for training or validation targets. Labels serve only as the supervised
target and as the evaluation reference. The 10 atlases are never training
targets. `register_targets.py` and `data/splits.py` enforce this.

**Additional bugs found**
1. `RandRotated(range_x=(-15, 15))`: MONAI ranges are in radians, so the
   baseline trained with arbitrary rotations of up to ±15 rad, not ±15°. The
   default is kept for comparability; `--rotate-range-deg 15` is the corrected
   ablation.
2. Label resize uses PyTorch `nearest`, which is floor-based. That shifts the
   label about 0.5–0.9 output voxel relative to the CT, which is resized with
   `area`. It affects the baseline too and is documented in
   `tests/test_atlas_priors.py`. Not changed.
3. **Sign error in my own `atlas/tps.py`** (found while validating
   `prototype.ipynb`): the smoothing term was `K + λI` where it must be
   `K − λI`, which made every TPS erratic (residuals up to 4100 mm, 26–55 % of
   control points folding at moderate λ). Fixed, with a regression test. The
   library and all cached registrations were rebuilt. It had hidden the benefit
   of the TPS: the earlier conclusion "TPS adds nothing" was an artefact.
   `experiments/exp_01_atlas_prior/smoke_local` predates the fix;
   `smoke_local_tpsfix` supersedes it.

**Key findings.** Details, tables and open tuning items are in
`experiments/registration_log.md`.
- Atlas-only majority-vote Dice (evaluation only) is **0.445 and 0.481** on the
  first two validation targets after the TPS fix (0.321 and 0.321 before). Landmark
  RANSAC affine beats intensity-only coarse registration by +0.06 to +0.15 per
  atlas, and the TPS adds up to +0.10 on top. Small organs (4, 11, 12) are still
  the weakest.
- Whole-body mutual information is anti-correlated with organ overlap.
  B-spline refinement driven by it gives 0.443 and 0.458 against TPS-only 0.445
  and 0.481: no gain at ~1.8× the runtime, so it is not used.
- Smoke test only (2 validation cases and 3 training steps, not a result):
  the epoch-101 weights score **0.119 with atlas priors before fine-tuning and
  0.198 after 3 steps**. The same weights with ground-truth priors score 0.73 on
  amos_0278 alone (0.68 on the first three cases, 0.65 on the full set). So the
  baseline mostly learned to read the ground-truth prior, and 0.6505 cannot be
  compared with atlas-prior results; a no-prior UNETR is still needed as the
  reference.
- **Unexplained, worth understanding:** that 0.119 is essentially the same as the
  0.120 measured before the TPS fix, although atlas-only Dice rose from 0.32 to
  about 0.46. Untested hypothesis: the model only sees the priors after average
  pooling over 16³ patches (roughly 40 to 60 mm per patch on this grid) and the
  epoch-101 weights are tuned to exact, binary priors, so a ~10 mm improvement
  in registration is invisible to it. If true, registration accuracy is not the
  bottleneck at this architecture's prior resolution, which Experiment 1 and a
  sensitivity test (shifting the priors by known amounts) can check.
- Runtime is ~190 to 250 s per target for registration on this laptop, which
  matters for planning the ~670-target server run.

## Architecture Plan

The diagnosis above leads to a new architecture plan: make the model learn to *use* uncertain atlas priors instead of treating them as corrupted ground truth. The plan adds seven changes (A–G), each switchable via training flags, so the epoch-101 checkpoint remains compatible. Changes include prior dropout, confidence gating, voxel-level prior pathways, output fusion, and atlas-consistency loss.

**See:** `new_model_plan.md` for the full design, implementation details, experiment flow and success metrics.

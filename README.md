# Atlas-Guided Medical Image Segmentation

## IIT ISM Internship Project

**Atlas-Guided ViT-UNETR for Multi-Organ CT Segmentation**

This document is an onboarding report for the `Atlas-Guided-Medical-Image-Segmentation` repository.
It explains the intended research pipeline, the current notebook implementation, the key divergence between them, and the next milestone.

---

## Contents

1. Purpose
2. Theoretical Pipeline
   1. Multi-Atlas Dataset
   2. Golden Transformation Construction
   3. Global Reliable Feature Selection
   4. Atlas-to-Target Registration
   5. Label Transfer (Warping)
   6. Multi-Organ Probability Map Construction
   7. Distance Map Construction
   8. Patch Aggregation
   9. Anatomical Prior Token
   10. Patch Embedding
   11. Cross-Attention Anatomical Fusion
   12. Transformer Encoder
   13. UNETR Decoder
   14. Segmentation Head & Final Prediction
3. Current Implementation Walkthrough
4. Caution: Supervised Approximation
5. Next Milestone: Semi-Supervised Atlas-Guided Segmentation

---

## 0. Purpose of This Document

This report explains:

- how the Atlas-Guided ViT-UNETR project is meant to work,
- what the current notebook (`AtlasGuidedUNETR_fixed.ipynb`) actually does,
- where the theory and implementation diverge,
- the next steps to move from a supervised approximation to a true atlas-guided system.

It is written for a new team member to understand the intended research pipeline and the current codebase state without reverse-engineering either from scratch.

The report has four parts:

- Part 1: full mathematical/conceptual pipeline from the architecture design document
- Part 2: walkthrough of the current notebook mapped to the theory
- Part 3: caution that the notebook is a supervised approximation, not the true atlas-guided system
- Part 4: roadmap to semi-supervised atlas-guided segmentation on unlabeled data

---

## 1. Theoretical Pipeline: Atlas-Guided Anatomical Prior ViT-UNETR

The core idea is to inject anatomical knowledge into a Vision Transformer (ViT) encoder using cross-attention, before a UNETR decoder reconstructs a segmentation mask.

Anatomical knowledge is produced from a library of labeled atlases, and is used to generate soft probability maps and distance maps for a new unlabeled CT scan.

### 1.1 Multi-Atlas Dataset

An atlas library `A = {A₁, A₂, …, Aₙ}` is assembled, where each atlas is a pair:

- `Aᵢ = (Iᵢ, Lᵢ)`
- `Iᵢ = CT volume`
- `Lᵢ = multi-organ ground-truth labels`

These scans are already segmented, typically by expert annotation. They form the anatomical knowledge base used by the rest of the pipeline.

### 1.2 Golden Transformation Construction (atlas ↔ atlas)

Every atlas pair `(Aᵢ, Aⱼ)` is registered to each other using organ masks and image intensities, producing a reliable golden transformation `(Tⱼ,ᵢ)_G`.

This process includes:

- atlas-to-atlas registration
- surface mesh extraction
- landmark correspondence generation
- Thin Plate Spline (TPS) fitting

Example TPS formula:

```
T(x) = A x + b + Σ_{m=1..M} wₘ · U(‖x − pₘ‖)
```

Where:

- `A` is an affine matrix
- `b` is a translation
- `wₘ` are TPS coefficients
- `pₘ` are landmark points

These golden transforms are a reference for later registration quality evaluation.

### 1.3 Global Reliable Feature Selection

Anatomical features are extracted from each atlas and scored for reliability.

- `Fᵢ = {f₁, …, fₙ}`
- each `f = (i, x, d)` contains an index, coordinate, and descriptor

For each correspondence, the residual against the golden transform is measured:

```
r̃_k = ‖x_k − (Tⱼ,ᵢ)_G(x̃_k)‖₂
```

A bounded reliability score is computed with tolerance `T`:

```
s̃_k = max(T − r̃_k, 0) / T
Score(f_k) = ω_r · Σ_{k∈R} s̃_k
```

Only the top-K most reliable features are kept:

- `F_reliable = {f₁, …, f_k}`

This step discards unreliable correspondences before they can affect target registration.

### 1.4 Atlas-to-Target Registration

Given a new unlabeled target CT volume `Iₜ`, reliable atlas features are matched in two stages:

- RANSAC affine registration
  - removes outliers
  - estimates affine transform `Aᵢ` such that `xₜ = Aᵢ(x_a)`
- non-rigid refinement
  - TPS or NiftyReg refines the affine estimate
  - produces a dense deformation field `ϕᵢ`

Final atlas-to-target transformation:

```
Tᵢ(x) = ϕᵢ(Aᵢ(x))
```

### 1.5 Label Transfer (Warping)

Atlas labels are warped into target space using the inverse of the final transform:

```
L′ᵢ(x) = Lᵢ(Tᵢ⁻¹(x))
```

Each atlas then provides a candidate label opinion for every voxel in the target scan.

### 1.6 Multi-Organ Probability Map Construction

For `C` organ classes, warped atlas labels are combined by voting.

Probability that voxel `v` belongs to class `c`:

```
P(c | v) = #(Atlases voting class c) / N
```

The probability vector is:

```
P(v) = [P₁(v), …, P_C(v)]
```

Example:

- `P(v) = [0.82, 0.10, 0.05, 0.03]`
  - 82% liver
  - 10% kidney
  - 5% spleen
  - 3% background

This soft atlas-derived probability map is generated without using the target scan's ground truth.

### 1.7 Distance Map Construction

For each organ class `c`, compute the Euclidean distance from each voxel to the nearest voxel in the organ region `O_c`:

```
D_c(v) = min_{y ∈ O_c} ‖v − y‖₂
D(v) = [D₁(v), …, D_C(v)]
```

Example:

- `D(v) = [3, 80, 65]` mm from liver, kidney, spleen

This distance map encodes relative anatomical position.

### 1.8 Patch Aggregation

The volume is divided into the same 3D patches used by the ViT.

For each patch `Xᵢ` with voxel set `Vᵢ`:

```
Pᵢ = (1/|Vᵢ|) Σ_{v∈Vᵢ} P(v)
Dᵢ = (1/|Vᵢ|) Σ_{v∈Vᵢ} D(v)
```

### 1.9 Anatomical Prior Token

The pooled probability and distance vectors are concatenated and projected into the ViT embedding dimension `d`:

```
Mᵢ = [Pᵢ, Dᵢ]
Aᵢ = MLP(Mᵢ)
MLP: ℝ^{2C} → ℝ^d
```

This anatomical token carries atlas knowledge into the transformer.

### 1.10 Patch Embedding (Image Side)

The CT patch is linearly embedded and summed with positional encoding:

```
Hᵢ = Xᵢ + Posᵢ
```

### 1.11 Cross-Attention Anatomical Fusion

The image tokens are queries, and the anatomical tokens are keys and values:

```
Q = H
K = A
V = A
CrossAttn(Q, K, V) = Softmax(QKᵀ / √d_k) · V
```

### 1.12 Transformer Encoder

The encoder stacks transformer blocks with self-attention, cross-attention, and feed-forward layers:

```
MHSA(H) = Softmax(QKᵀ / √d_k) · V
FFN(x) = W₂ · σ(W₁ x + b₁) + b₂
x = x + MHSA(x)
x = x + FFN(x)
```

The final encoder output is:

- `Z_enc ∈ ℝ^{N_p × d}`

Where `N_p` is the number of patches.

### 1.13 – 1.14 UNETR Decoder

`Z_enc` feeds a UNETR-style decoder with four upsampling stages.
Each stage uses:

- transposed convolution
- skip connection from encoder stage
- convolutional refinement

```
Up(x) = ConvTranspose(x)
Fₗ = Concat(Encoderₗ, Decoderₗ)
F′ₗ = Conv(Fₗ)
```

### 1.15 – 1.16 Segmentation Head & Final Prediction

A `1×1×1` convolution produces logits `Y`, then softmax converts them to probabilities:

```
Ŷ(v, c) = exp(Y(v, c)) / Σ_{j=1..C} exp(Y(v, j))
Label(v) = argmax_c Ŷ(v, c)
```

### 1.17 Overall Pipeline

End-to-end theoretical pipeline:

```
Atlases → Golden Transform → Reliable Features → RANSAC → TPS/NiftyReg → Warp Labels → Probability Maps → Distance Maps → Anatomical Tokens → Cross-Attention ViT → UNETR Decoder → Multi-Organ Segmentation
```

---

## 2. Current Implementation Walkthrough (`AtlasGuidedUNETR_fixed.ipynb`)

This section maps the notebook onto the theory above.

**Key point:** The notebook trains on a dataset with ground-truth organ labels, so it does not implement sections 1.1–1.5 of the theory.

Instead, it generates probability and distance maps directly from the true label.

### 2.1 Environment

The notebook uses:

- PyTorch (CUDA build)
- MONAI 1.6.0 (Colab) / 1.5.2 (local)
- PyTorch Lightning
- einops
- nibabel
- numpy
- pandas

Device selection falls back to CPU if CUDA is unavailable.

### 2.2 Data Preparation

The notebook loads NIfTI image files (`*_0000.nii.gz`) and matching label files.

It then:

- pairs images and labels by filename
- splits data 80/20 into train/validation
- stores examples in MONAI format: `{"image": ..., "label": ...}`

### 2.3 Transforms: Building the Probability and Distance Maps from Ground Truth

The MONAI transform chain performs:

- loading
- channel-first conversion
- RAS orientation
- foreground cropping
- 1 mm isotropic resampling
- HU intensity scaling to `[0, 1]`
- resize/crop to `96×96×96`
- light spatial augmentation (flips, small rotations)

Project-specific transforms:

- `mt.AsDiscreted(keys=["label"], to_onehot=NUM_CLASSES)`
  - converts integer labels to a one-hot tensor with 13 channels
  - this tensor is used as the notebook's probability map
- `CustomSpatialDistanceTransformd`
  - custom wrapper around `scipy.ndimage.distance_transform_edt`
  - computes distance per organ channel from the true mask

This means the notebook uses the ground-truth label as if it were the atlas-voted probability map and distance map.

### 2.4 Model Architecture: `AtlasGuidedViTUNETR`

The model is built on top of MONAI's UNETR and consists of three main components:

- `AnatomicalPriorGenerator`
  - mean-pools the probability and distance maps inside each ViT patch
  - projects the pooled vector through a 2-layer MLP to the ViT hidden size
  - implements patch aggregation and anatomical prior token generation

- `CustomViT`
  - subclasses MONAI's ViT
  - replaces standard transformer blocks with `TransformerBlock(..., with_cross_attention=True)`
  - injects anatomical tokens as cross-attention context

- `AtlasGuidedViTUNETR`
  - subclasses MONAI's UNETR
  - keeps the encoder, decoder, and segmentation head
  - swaps in `CustomViT` and `AnatomicalPriorGenerator`
  - forwards `probability_maps` and `distance_maps` through the ViT and decoder

Engineering note:

Earlier drafts reimplemented cross-attention and added nonstandard encoder/decoder layers, causing shape mismatches and attribute errors. The current version reuses MONAI's built-in cross-attention and UNETR stack for a simpler and more stable architecture.

### 2.5 Loss, Optimizer, Training Loop

Training uses:

- `MONAI DiceCELoss(softmax=True, to_onehot_y=False)`
- `AdamW` optimizer
- `DiceMetric` for validation

The training loop is a standard MONAI epoch loop:

- move image, label, and distance maps to device
- forward through `AtlasGuidedViTUNETR`
- compute loss
- backpropagate
- update weights

**Important note:**

Because the probability and distance maps are derived from ground truth during data loading, the model is effectively using a copy of the answer as input. This validates the architecture, but it is not the true atlas-guided design.

---

## 3. Caution: Read Before Extending This Notebook

⚠️ The notebook is a supervised approximation of the atlas-guided design — not the atlas-guided pipeline.

### Why this matters

- The notebook uses real organ labels for every scan.
- It does not generate probability maps by warping and voting across a library of atlases.
- It does not build distance maps from warped atlas labels.

Instead, it:

- one-hot encodes the real label as the probability map
- computes the Euclidean distance transform directly from the real label

This is useful for validating the architecture, but it means the model is trained with perfect anatomical priors that would not exist for a real unlabeled scan.

### True atlas-guided learning requires:

- a small set of doctor-annotated atlas scans
- a large collection of unlabeled target scans
- atlas-to-target registration
- label warping and voting
- voted probability maps and distance maps

That machinery is the next development task.

---

## 4. Next Milestone: Semi-Supervised, Atlas-Guided Segmentation on Unlabeled CT Data

The next phase is to move from the current supervised approximation to a realistic semi-supervised atlas-guided pipeline.

### 4.1 New components to build

- Atlas-to-atlas golden transform construction
- landmark correspondence and TPS fitting
- residual-based reliable feature scoring and Top-K selection
- atlas-to-target registration using RANSAC affine + non-rigid refinement
- label warping and multi-atlas voting to produce true probability maps
- distance maps computed from warped atlas label regions

### 4.2 Reusable components

The following can be reused with minimal changes:

- `AnatomicalPriorGenerator`
- `CustomViT`
- `AtlasGuidedViTUNETR`
- MONAI preprocessing chain (orientation, spacing, intensity scaling, resizing)

These already consume generic probability and distance map inputs.

### 4.3 CT-physics-guided loss (new work)

Because supervision becomes weak and uncertain, the loss should go beyond standard Dice/Cross-Entropy.

Potential additions:

- atlas voting loss term
- CT intensity consistency penalties
- Hounsfield unit range consistency for tissue classes
- intensity-gradient alignment at predicted boundaries

The exact composite loss should be validated on a labeled hold-out set.

### 4.4 Suggested task breakdown

- implement atlas-to-atlas golden transform + TPS fitting
- implement reliable feature scoring and Top-K selection
- implement atlas-to-target registration (RANSAC affine + TPS/NiftyReg)
- implement label warping + multi-atlas voting and distance map generation
- replace the current label-derived probability/distance maps with atlas-derived maps
- design and validate a CT-physics-guided composite loss
- retrain `AtlasGuidedViTUNETR` using true semi-supervised conditions and compare to the baseline

---

## Summary

This repository currently contains a strong architectural prototype for atlas-guided segmentation, but the existing notebook is a supervised proof-of-concept.

The next milestone is to implement true atlas-based probability and distance map generation for unlabeled target scans and then evaluate the model under realistic semi-supervised conditions.

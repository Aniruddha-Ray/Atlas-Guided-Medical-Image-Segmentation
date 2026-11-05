# Novelty of the paper: status on 2026-10-31

This is a working note for the authors, not paper text. Every claim is rated by how far the current evidence carries it.
No literature search has been done in the sessions that produced this project, so **novelty against prior work is
unchecked**; sections 5 and 6 say what to check and what is still missing.

## 1. One-sentence idea

A label-free anatomical prior (per-voxel class probability, distance to each organ, and a confidence map), built from only
10 quality-selected atlases by sparse landmark matching, RANSAC and thin-plate-spline registration, is injected into a
Swin UNETR at three places (input channels, cross-attention to anatomical tokens, and a log-prior term on the output
logits), and trained with prior dropout and a confidence-weighted consistency loss so that the network uses the atlas
without depending on it.

## 2. Candidate contributions, with evidence status

| # | Contribution | Evidence now | Strength |
|---|---|---|---|
| N1 | **Small, quality-selected atlas set as a prior source.** 10 atlases chosen by slice thickness (at most 3 mm), complete coverage, clean labels and typical anatomy. Choosing them this way raised the atlas-only Dice from 0.380 to 0.450 on 12 training targets (`experiments/registration_log.md`). | Atlas vote alone: 0.424 Dice on the 50 validation cases, about 12 mm mean centroid error. Registration needs no learned model and no target labels. | Moderate. The selection rule is simple; the finding is that the choice matters. |
| N2 | **Three-path prior integration for a hierarchical transformer.** (a) 27-channel input (CT, P, D, w) into patch embedding and the full-resolution branch; (b) 216 anatomical tokens (per-16³-patch pooling of P, D, w, then an MLP) read by gated cross-attention adapters after Swin stages 2, 3 and 4; (c) output fusion `z_c + gamma_c * w * log(P_c + 1e-3)`. | Swin + prior reaches 0.696 against 0.424 for the atlas vote alone. Per-path ablations (tokens only, voxels only, fusion only) are **not run**, so the value of each path is not shown. | Moderate for the design; unproven per path. |
| N3 | **Start at the atlas vote.** Initialising the learnable per-class fusion weight at gamma = 1 makes the untrained network equal the atlas vote. With gamma = 0 the network collapsed to 0 Dice on several organs (0.314 against 0.571 at 48 training cases; 0.606 at 150 for ViT-UNETR). | Measured on the ViT backbone at two data sizes. Not re-run for Swin with gamma = 0. | Small but concrete, and cheap to state. |
| N4 | **Prior dropout plus a confidence-weighted KL consistency loss.** Priors are replaced by uninformative maps with probability 0.3; the network is pulled toward the atlas only where the confidence w is high (weight 0.1, ramped over 10 epochs). | With the atlas removed at test time the Swin model still scores 0.578 (ViT-UNETR 0.302): the image pathway keeps learning. This is **not** an image-only baseline. | Moderate, but only if an image-only Swin run backs it up (see 5). |
| N5 | **Negative results on physics-informed terms and test-time refinement.** HU-consistency and edge-aware losses, HU-likelihood input channels with a learned fusion weight, and post-hoc refinement did not help (section 3). | Several clean negatives on one seed and 50 cases. | Useful as an honest ablation. Not a headline. |

**What is not a contribution.** The physics-informed ("PINN-style") components do not improve Dice, so the paper should
not claim them as a method. Averaging Swin and UNETR also gave nothing (the two agree on 98.5 % of voxels).

## 3. Results behind the claims

Setup: 150 training and 50 validation cases, 96³ input, 100 epochs, seed 0, learning rate 2e-4 (untuned), mean Dice over the
12 organs. The best epoch is picked on the same 50 cases, so the numbers are slightly optimistic.

| Model | Mean Dice |
|---|---|
| Atlas vote alone | 0.424 |
| ViT-UNETR, image only (six organs at 0) | 0.309 |
| ViT-UNETR + prior | 0.606 |
| **Swin UNETR + prior** (best epoch 96; last epoch 0.6959) | **0.696** |
| Swin + prior, then 12 epochs with HU + edge losses only | 0.684 (last epoch) |
| Swin + prior, then 12 epochs with HU channels, fusion term and both losses | 0.685 (last epoch) |
| Swin image only | **not run** |

- The prior is worth about +0.30 Dice for ViT-UNETR and lifts six organs from 0 to 0.35 to 0.69.
- Swin beats ViT-UNETR on all 12 organs (+0.088 mean); the largest gains are gallbladder, stomach and aorta.
- Weakest organs for Swin + prior: left adrenal 0.43, right adrenal 0.49, gallbladder 0.58, esophagus 0.58.
- After training the learned gates and weights barely move (mean trust gate 0.986, mean gamma 0.975), so the adaptive
  parts act close to their starting values.
- Test-time refinement on the best checkpoint: largest-component filtering +0.001, HU Bayes update from -0.0001 to -0.033
  as its weight grows, edge diffusion -0.0005 to -0.0098.

## 4. Where the contribution is not (what a reviewer will say)

- **The absolute score is low.** 0.696 at 96³ with 150 training cases is far from strong full-resolution systems. The
  case for the paper rests on the **relative** gain from the prior and on data efficiency, not on beating them.
- **One seed, 50 validation cases, an untuned learning rate.** A gap under about 0.01 is not a result until the seed
  spread is measured.
- **The comparison is not parameter-matched:** Swin has about 67 M parameters and ViT-UNETR about 122 M, and the voxel
  prior also enters Swin's patch embedding.
- **There is no external baseline** (a standard Swin UNETR or nnU-Net on the same data and split), so the claim
  "the atlas helps" is shown only against our own image-only ViT run.
- **The atlas set was chosen with label information** (clean labels, typical anatomy). The atlases are never targets and the
  rule does not depend on a fold, but it should be stated.
- **The adaptive parts are barely used** (gates and gamma stay near 1), which invites the question whether tokens and
  gates add anything over the voxel path and output fusion alone.

## 5. What is needed before the claims are defensible

1. **5-fold cross-validation on the whole data** (670 cases, 536 / 134 per fold), mean ± sd over folds. Runner:
   `server/run_cv.py`; written but not yet run on the server.
2. **Swin image-only baseline** (queued earlier, never run), to turn the 0.578 no-prior number into a real comparison.
3. **Path ablation:** P only, P + D, P + D + w, and each integration path alone (plan steps 6 and 7).
4. **An external baseline** on the identical folds.
5. **Seed spread** (at least 2 more seeds on one fold) and a learning-rate check.
6. **Small-organ handling,** the clear weak spot: class-weighted Dice, a learned deformation of the atlas prior, or higher
   resolution. None is tried yet; the loss currently has no class weights.
7. **Literature check** of: atlas or probabilistic-atlas inputs to segmentation networks, registration-based priors in
   deep segmentation, cross-attention conditioning of a transformer on anatomical priors, and the UNETR, Swin UNETR and
   nnU-Net lines. Until this is done, "first" and "novel" must not appear in the text.

## 6. Suggested framing for now

A data-efficient atlas-guided Swin UNETR: priors from a handful of quality-selected atlases, injected at three levels and
trained so the network keeps its own image pathway; the gain from the prior is large and measured against an image-only
model, the weak organs are analysed honestly, and physics-based additions are reported as a negative result. Promote or
drop each claim in section 2 according to what items 1 to 5 above show.

## 7. Prior-work check (2026-10-31, web search only)

A first-pass search was run (about eight queries). It found **no paper that combines all of our parts**, but it found
earlier work for most single ingredients. Only titles and search summaries were read, not the papers, so **every entry
below must be read before it is cited or used to argue novelty**; the search is not exhaustive.

| Ingredient | Closest earlier work found | Effect on our claim |
|---|---|---|
| Probabilistic atlas / spatial map as extra network input for CT organs | 3D U-JAPA-Net (3D U-Net + organ probabilistic atlas, 47 CTs); multi-planar 2D U-Nets with spatial occurrence maps (arXiv 2606.07717); STIPPLE (organ-presence map via a prior-driven softmax); atlas-aware ConvNet (Liang et al., ACML 2020) | N1 and the input path of N2 are **not new in principle**. What would be new is the source (few quality-selected atlases, label-free registration), the three outputs P, D, w, and the backbone. |
| Registration-based, target-specific atlas priors for abdominal CT | Wolz et al., hierarchical multi-atlas registration, IEEE TMI 2013 (non-deep) | Same: the prior construction has precedent. |
| Atlas prior combined with a network in a Bayesian way | Dalca et al., MICCAI 2019 (probabilistic atlas + deformation model); Deep Atlas Prior (prior in the loss) | N3 (log-prior fused with the logits, start at the vote) needs a careful comparison to these; the fusion is a Bayes-style product, so do not oversell it. |
| Anatomical prior in a transformer | Focused Decoder (atlas-restricted attention masks for 3D anatomical detection); Swin UNETR's own anatomy proxy tasks (pretraining) | Cross-attention to **learned prior tokens built from registered atlas maps** was not found, which is the best remaining candidate for N2. |
| Prior dropout for robustness to the atlas | Nothing close found (only generic MC-dropout uncertainty work) | N4 may be open, but the search was shallow. |
| HU-based and edge-aware losses for CT segmentation | HU-weighted regulariser for organ-at-risk segmentation (arXiv 2303.10796); boundary-aware network with edge losses (arXiv 1908.08071) | Our losses are not new and did not help; keep them as a reported negative. |

**Net reading for now:** a paper claiming "atlas priors help deep segmentation" would not be novel. The defensible
space is the specific combination (a Swin UNETR with the prior read through cross-attention tokens and a log-prior
fusion, built from a few quality-selected atlases, trained with prior dropout and a confidence-weighted consistency
loss) plus an honest evaluation. Even that must be checked against 2024 to 2026 work, which this search barely touched,
and the closest papers above must be read and, where possible, used as baselines.

### 7.1 Second pass (2026-10-31, deeper, 2024 to 2026 focus)

Three more extended searches and two abstract-level reads. Still no paper with our combination.

- **AtlasSeg** (arXiv 2411.02867, fetal brain MRI, 2024): a dual-branch convolutional network, one branch for an atlas with
  segmentation labels, fused with multi-scale spatial attention in encoder and decoder. The abstract mentions no
  cross-attention or transformer, no number of atlases and no prior dropout. This is the nearest design to our "atlas
  branch fused at several depths"; it must be read in full, and it argues against claiming multi-level fusion as new.
- **FW-Atlas / FW-AtlasMC** (Frontiers in Neurology, 2026, brain MRI): two atlas-derived channels (confidence, entropy)
  concatenated to a 2.5D U-Net; the atlas is built from training labels in a shared voxel grid, **without registration**.
  The atlas channels are kept fixed when an image modality is missing; **no dropout of the prior channels** is described.
  Main claim: the prior did not significantly improve full-input Dice but stabilised degraded inputs.
  This supports our prior-dropout idea as not found elsewhere, and it is a caution that atlas priors can give little
  gain when the network is already strong (our own image-only Swin run decides this for us).
- **CPNet** (3D semantic relation and geometry context prior, compared against nnU-Net, UNETR and Swin UNETR): priors
  propagated into multi-scale features; not atlas label propagation as far as the search shows. Read before claiming
  token-based injection as new.
- No 2024 to 2026 paper was found that feeds registered multi-atlas probability, distance and confidence maps into a
  Swin UNETR through cross-attention tokens. Absence in a web search is weak evidence; a proper search of PubMed, Google
  Scholar, arXiv and the MICCAI proceedings is still needed.
- On AMOS, nnU-Net is generally the stronger baseline than Swin UNETR (differences of about 1 to 3 Dice in the sources
  found), so a reviewer will ask for nnU-Net on the same folds.

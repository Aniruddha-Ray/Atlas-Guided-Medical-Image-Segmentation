# Architecture Plan: Atlas-Aware ViT-UNETR

**Objective:** Make the model learn to *use* uncertain atlas priors instead of treating them as corrupted ground truth.

## Problem Diagnosis

The epoch-101 baseline (trained on ground-truth P(v)) fails with atlas priors:
- **With GT prior:** Dice 0.727 (perfect information)
- **With atlas prior (44.5% overlap):** Dice 0.130 (throws away the prior)
- **Atlas-only (no model):** Dice 0.445 (what the registration provides)

### Root causes

1. **Train-test prior mismatch.** The model was trained with binary ground-truth priors where the easiest solution was to copy the prior. It never learned to question it or lean on the image.
2. **Prior only enters at 16³-patch resolution.** Average pooling into one token per patch (216 tokens total) loses voxel-level spatial detail. A small organ is a tiny fraction of a pooled region.
3. **Confidence map w(v) is computed but unused.** The model can't tell where atlases agree; it sees all priors equally.
4. **No fallback to the image.** No prior dropout, no prior-free baseline to measure against.

## Proposed Architecture

All changes are **additive and switchable** via training flags. The epoch-101 checkpoint remains compatible.

```
CT (1ch) ──────────────┬─────────────► ViT patch embed ─► 12 blocks ─► UNETR decoder ─► logits_net
                       │                    ▲ cross-attn x gate g_i (init 1)          │
P (13) D (12) w (1) ───┼─► PriorEncoder ────┘  (tokens from pooled P, D, w)                  │
      │                │                                                                    ▼
      │                └─► encoder1 conv1/conv3 input: [CT, P, D, w] (new ch zero-init)    logits = logits_net
      │                                                                              + γ_c · w · log(P+ε)
      └──────────────────────────────────────────────────────────────────────────►  (γ_c init 0)
```

## Changes (A through G)

Each change is **independent** and can be toggled on/off during training.

### A. Train with atlas priors only (no GT prior fallback)

**Where:** `scripts/train.py --prior-source atlas` (already implemented)

**What:** Epoch 0 evaluates the epoch-101 weights with atlas priors (diagnostic upper bound). Every epoch after trains on atlas priors, matching what the model will see at inference time.

**Why:** Removes train-test prior mismatch. The model learns that priors are soft and uncertain, not binary.

**Code impact:** None. Already in `scripts/train.py`.

**Epoch-101 compatible:** Yes. Checkpoint loads unchanged.

---

### B. Prior dropout (p ≈ 0.3)

**Where:** New transform in `data/transforms.py`

**What:** With probability 0.3 per sample during training, replace:
- P with uniform (1/13 everywhere)
- D with 1 everywhere (maximum distance, no prior info)
- w with 0 everywhere (no confidence)

**Why:** Keeps the image pathway strong. The model must remain useful when atlases fail. Also acts as regularization to prevent overfit to prior distribution.

**Implementation:**
```python
class PriorDropoutd(RandomizableTransform, MapTransform):
    """Training pipeline only, placed after AtlasPriorsd (so 'prob', 'dist', 'conf' exist)."""
    def __init__(self, prob=0.3):
        RandomizableTransform.__init__(self, prob=prob)
        MapTransform.__init__(self, keys=["prob", "dist", "conf"])

    def __call__(self, data):
        d = dict(data)
        self.randomize(None)                 # sets self._do_transform from self.prob (MONAI API)
        if self._do_transform:
            d["prob"] = torch.full_like(d["prob"], 1.0 / d["prob"].shape[0])  # uninformative: uniform over classes
            d["dist"] = torch.ones_like(d["dist"])                            # D is in [0, 1]; 1 = far from every organ
            d["conf"] = torch.zeros_like(d["conf"])                           # "no atlas evidence"
        return d
```

**Code impact:** 10 lines in `data/transforms.py`, add to training pipeline.

**Epoch-101 compatible:** Yes (a data-side change only). It is never applied to validation, so validation numbers stay comparable across runs.

---

### C. Confidence as an input, plus a trust gate on each cross-attention

**Where:** `models/segmentation_model.py` -> `AnatomicalPriorGenerator`, `CustomViT`

**What:**
1. **Confidence token input.** Pool w(v) per patch like P and D and feed it to the token MLP: the first `Linear` goes from 25 to 26 inputs (13 P + 12 D + 1 w). The epoch-101 weights fill the first 25 columns; the new column is **zero-initialised**, so the tokens are unchanged at start.
2. **Trust gate on the cross-attention residual only.** MONAI's block computes `x = x + attn(norm1(x)); x = x + cross_attn(norm_cross_attn(x), context); x = x + mlp(norm2(x))`. Wrap it in a subclass that scales just the cross-attention term: `x = x + g_i * cross_attn(...)`, with one learnable scalar `g_i` per block, **initialised to 1.0** (identity, so epoch-101 behaviour is exactly preserved). Training can then lower g_i where the prior hurts.

**Correction to an earlier draft:** it multiplied the whole residual stream by `tanh(g)` with g = 0. That zeroes every image token at start and would destroy the pretrained model. The gate must touch only the cross-attention branch, and must start at 1, not 0.

**Why:** the model can learn how much to trust the prior per depth, and it is told where the atlases agree (w). Both start as a no-op.

**Sketch:**

```python
class GatedTransformerBlock(TransformerBlock):        # same state-dict keys as the stock block + one scalar
    def __init__(self, *a, **k):
        super().__init__(*a, with_cross_attention=True, **k)
        self.cross_gate = nn.Parameter(torch.ones(()))
    def forward(self, x, context=None, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.cross_gate * self.cross_attn(self.norm_cross_attn(x), context=context)
        return x + self.mlp(self.norm2(x))
```

**Code impact:** ~25 lines. **Epoch-101 compatible:** yes, with `strict=False` for the new keys (`cross_gate`, the extra MLP column); a unit test must show the output is identical to the old model at init.

---

### D. Voxel-level prior path (full-resolution)

**Where:** `models/segmentation_model.py` -> `AtlasGuidedViTUNETR.__init__/forward`

**What:** `encoder1` is MONAI's `UnetrBasicBlock` (verified): `conv1` (16 x 1 x 3x3x3), `conv2` (16 x 16 x 3x3x3) and a 1x1x1 residual `conv3` (16 x 1 x 1x1x1). Widen the **input of `conv1` and `conv3`** from 1 to 27 channels ([CT, P(13), D(12), w(1)]). Copy the CT slice of each weight and **zero-initialise the 26 new channels**, so the output is identical at start.

**Correction to an earlier draft:** it replaced a 64-channel stride-2 conv (`encoder1[0][0]`), which does not exist. Only the two input convs above change.

**Why:** today the prior reaches the output only through 16^3-patch tokens. This gives the full-resolution skip the prior at voxel level, which is where thin boundaries and the small organs (4, 11, 12) need it.

**Code impact:** ~25 lines (+ 27 x 16 x 27 + 27 x 16 = ~12 k weights). **Epoch-101 compatible:** yes, with a unit test for identical output at init.

---

### E. Output-level prior fusion

**Where:** End of `AtlasGuidedViTUNETR.forward`

**What:** After the decoder outputs logits, add the atlas prior as a bias term:
```
logits_final = logits + γ ⊙ w(v) ⊙ log(P(v) + ε)
```

where:
- `γ` is a learnable vector of shape [13] (one scalar per class), initialized to 0.
- `w(v)` masks the prior to regions where atlases agree (confidence > threshold).
- `log(P(v) + ε)` is the log-prior of each class. Adding a per-class log-prior to logits is the softmax-consistent form (a naive Bayes update); the log-odds form `log P - log(1-P)` is wrong for a softmax over 13 classes. Use ε = 1e-3 so the term is bounded (about -6.9 where P = 0).
- ⊙ denotes element-wise multiplication.

**Why:**
- At init γ = 0, so the fusion does nothing (epoch-101 output unchanged).
- With just this term, the model could match the atlas-only Dice (0.42–0.48).
- It's interpretable: γ_c shows how much organ c relies on the atlas.
- It's efficient: one parameter per class.

**Implementation:**

```python
class AtlasGuidedViTUNETR(UNETR):
    def __init__(self, ..., output_fusion=False, ...):
        super().__init__(...)
        self.output_fusion = output_fusion
        if output_fusion:
            self.output_fusion_weight = nn.Parameter(torch.zeros(13))

    def forward(self, x_in, probability_maps, distance_maps, confidence_maps=None):
        # ... (encoder, ViT, decoder as before)
        logits = self.out(out)
        
        if self.output_fusion and confidence_maps is not None:
            log_prior = torch.log(probability_maps.float() + 1e-3)           # float32 even under AMP
            gamma = self.output_fusion_weight.view(1, -1, 1, 1, 1)
            logits = logits + gamma * confidence_maps.float() * log_prior
        
        return logits
```

**Code impact:** ~15 lines in `AtlasGuidedViTUNETR.__init__` and `forward`.

**Epoch-101 compatible:** Yes. New parameter `output_fusion_weight` initialized to 0.

---

### F. Atlas-consistency loss (Experiment 2)

**Where:** `training/engine.py` → `train_one_epoch`

**What:** Add a weighted KL divergence term:
```
loss_total = DiceCELoss(logits, labels) + λ · mean(w(v) · KL(P(v) ‖ softmax(logits)))
```

where λ starts small (0.01) and ramps up to 0.1 over 10 epochs.

**Why:** This is the paper's Experiment 2: pull the model toward the prior only where the atlases agree (high w). Helps avoid overfitting to noisy atlas estimates.

**Implementation:**

```python
def train_one_epoch(..., consistency_weight=0.0, consistency_ramp_epochs=10, current_epoch=0, ...):
    # ... existing code ...
    
    # Ramp up consistency loss
    if consistency_weight > 0 and current_epoch < consistency_ramp_epochs:
        lambda_t = consistency_weight * (current_epoch / consistency_ramp_epochs)
    else:
        lambda_t = consistency_weight
    
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        prob = batch[prob_key].to(device)
        dist = batch[dist_key].to(device)
        conf = batch.get("conf", torch.ones_like(prob[:, :1, :, :, :])).to(device)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            logits = model(images, prob, dist, conf)
            dice_ce_loss = criterion(logits.float(), labels)
            
            if lambda_t > 0:
                pred_probs = F.softmax(logits, dim=1)
                kl_loss = F.kl_div(
                    torch.log(pred_probs + 1e-7),
                    prob,
                    reduction='none'
                )
                # Weight by confidence
                weighted_kl = (conf * kl_loss).mean()
                loss = dice_ce_loss + lambda_t * weighted_kl
            else:
                loss = dice_ce_loss
        
        # ... backward, optimizer step as before ...
```

**Code impact:** ~25 lines in `train_one_epoch`.

**Epoch-101 compatible:** Yes. With `--consistency-weight 0` (default) it behaves as before.

---

### G. CT HU-consistency loss and edge loss (Experiments 3 and 4)

**Where:** `training/physics_losses.py`, wired into `scripts/train_prototype_v2.py` (`--hu-weight`, `--edge-weight`, both 0 by default, ramped in like the consistency weight).

**What:** (full derivation and formulas in `model_walkthrough.md`, Section 6.3)
- **HU consistency:** voxels predicted as organ c should match this scan's HU for organ c. Per-scan median and robust sigma come from the confident atlas core (P_c >= 0.9, w >= 0.8, at least 50 voxels), never from the target label. Huber penalty, normalised by the predicted organ volume, averaged over the organs that have a core.
- **Edge-aware:** penalises predicted boundaries where the smoothed CT is flat, weight exp(-|grad I| / kappa), gradients in mm from each sample's spacing.

**Why:** Paper Experiments 3 and 4.

**Status:** implemented and tested (11 unit tests; a 3-epoch mixed-precision smoke run had no NaNs or skipped steps). On 12 real training scans the HU loss ranks the true labels above the atlas vote on 10 of 12 scans; the edge loss detects shifted boundaries but cannot separate truth from the atlas vote (6 of 12), so it is expected to act mainly as a smoothness term. Cost: +56 % step time, +0.28 GiB peak memory. Suggested starting weights from loss magnitudes: `--hu-weight 0.03`, `--edge-weight 50`. **Not yet evaluated in a real training run.**

---

## Current architecture vs planned architecture

"Current" was checked against the code (`models/segmentation_model.py`, `training/engine.py`, `data/transforms.py`) and by instantiating the model: 122.3 M parameters, of which the ViT is 116.7 M, the prior generator 1.22 M and the CNN encoder/decoder/head 4.4 M.

| Component | Current (as built) | Planned | Change |
|---|---|---|---|
| Image backbone | 12-layer ViT, hidden 768, MLP 3072, 12 heads, 16^3 patches, 216 tokens | same | none |
| Decoder and skips | stock UNETR decoder; skips from ViT layers 3, 6, 9 and `encoder1` | same | none |
| Prior channels | P (13, incl. background) and D (12) | P, D and the confidence w(v) (1) | + w |
| Prior -> tokens | avg-pool per 16^3 patch, 2-layer MLP (25 -> 1536 -> 768), one token set shared by all 12 blocks | same, MLP input 26 (w added, new column zero-init) | small (C) |
| Prior fusion in the ViT | cross-attention in all 12 blocks, always at full strength | same cross-attention, each scaled by a learnable gate g_i (init 1) | small (C) |
| Prior at full resolution | none: `encoder1` sees the CT only | `encoder1` conv1/conv3 also see [P, D, w] at voxel level (new channels zero-init) | new path (D) |
| Prior at the output | none: logits come only from the segmentation head | logits + gamma_c * w * log(P + eps), gamma init 0 | new term (E) |
| Use of w(v) | computed in the data pipeline, **never used by the model or the loss** | used as input (C, D), output mask (E), loss weight (F) | now used |
| Loss | DiceCE only | DiceCE + lambda * w-weighted KL(P \|\| softmax(logits)), lambda ramped 0 -> 0.1 | new term (F) |
| Training priors | baseline: the case's own label (ground truth); atlas priors only via `--prior-source atlas` | atlas priors only, for both training and validation | policy (A) |
| Prior robustness | none: the prior is always present and always exact | prior dropout p = 0.3 per sample (P uniform, D = 1, w = 0) | new transform (B) |
| Image-only reference | none | stock UNETR trained without priors (step 0) | new run |
| Extra parameters | 0 | ~14 k (12 gates, 13 fusion weights, 1 MLP column, ~12 k voxel-path weights) = 0.01 % | negligible |
| Behaviour at init | epoch-101 | **identical** to epoch-101 (every new weight is 0, or 1 for the gates); must be proven by unit tests | compatible |

### How different is it?

- **Structurally small.** About 95 % of the parameters (ViT, decoder, skips) are untouched. The changes are two new entry points for the prior (full-resolution `encoder1` input and output logits), a gate and a column in existing layers, one transform and one loss term: roughly 150 new lines.
- **Behaviourally different.** The current model can only read the prior through 16^3 averaged tokens, always at full trust, and it was trained on an exact prior. The planned one sees the prior at three resolutions (patch tokens, voxels, logits), is told its reliability, can turn it down, and is trained on the noisy atlas prior and with the prior sometimes removed.
- **Reversible.** Each item (A to F) is a flag, and all flags off gives epoch-101.

### What is a measurement and what is a hypothesis

Measured: the epoch-101 weights give 0.727 with ground-truth priors and 0.13 with atlas priors on amos_0278, while the atlas vote alone gives 0.42 (96^3) to 0.445 (2 mm grid) on that case. Hypotheses, not yet tested: that the patch pooling is the main reason, that prior dropout and the voxel path will fix it, and the size of any gain from E and F. The experiment flow below is designed to test them one at a time.

---

## Experiment Flow

All runs on the same train/val split, augmentation and 96³ grid. **No server: every run is on the laptop (RTX 3050, 4 GB)**; the chunk is 150 train / 50 val.

| Step | What | Flag(s) | Why | Validation gate |
|---|---|---|---|---|
| 0 | **Stock UNETR, no priors** | `--image-only` (added 2026-10-28; uninformative priors in training and validation) | Baseline: image alone | Essential reference R |
| 1 | Fine-tune epoch-101 with atlas priors | `--prior-source atlas` | Current architecture + A | Dice ≥ R? |
| 2 | + prior dropout | `--prior-dropout 0.3` | Add B | Dice ≥ step 1? |
| 3 | + conf input/gate + output fusion | `--conf-gating --output-fusion` (new) | Add C + E | Beat the atlas-only vote (0.392 on the full validation set)? |
| 4 | + voxel-level prior | `--voxel-prior` | Add D | Per-organ Dice on small organs? |
| 5 | + atlas-consistency loss | `--consistency-weight 0.1` | Add F (Exp 2) | Dice gain? |
| 6 | Sensitivity: shift priors | `--shift-prior 5` (mm) | Diagnostic | Is registration tuning worth it? |
| 7 | **Prior-channel ablation**: P, P+D, P+D+w | `--prior-channels P`, `PD`, `PDw` (new) | Do D and w earn their place? | See below |
| 8 | + HU-consistency loss | `--hu-weight 0.03` | Add G (Exp 3) | Dice gain over the best of steps 2 to 7? |
| 9 | + edge-aware loss | `--edge-weight 50` | Add G (Exp 4) | Dice gain over step 8 (or step 7 if 8 is dropped)? |
| 10 | Swin UNETR + prior | `--backbone swin` | Backbone ablation, cell B (see below) | Beats the UNETR + prior run by more than the seed spread? |
| 11 | UNETR, image only | `--image-only` | Ablation cell C | Reference for the prior's effect on UNETR |
| 12 | Swin UNETR, image only | `--backbone swin --image-only` | Ablation cell D | Reference for the prior's effect on Swin |

### Prior-channel ablation (step 7): P vs P+D vs P+D+w

**Why.** P, D and w are all computed from the same 10 atlas votes: D is the distance to the region where P is high (`P_c >= 0.5 * max P_c`), and w is one minus the normalised entropy of P. They add no new information, only different views of it, so each of D and w has to justify the extra inputs and code.

| Arm | Channels the model receives | Used by the model |
|---|---|---|
| P | 13 | token MLP and voxel path take P only; the output-fusion term and the consistency loss use w = 1 everywhere |
| P+D | 13 + 12 | as above, plus D |
| P+D+w | 13 + 12 + 1 | the full redesign: w as an input, an output mask and a loss weight |

**Implementation.** `--prior-channels` builds the model with only those input channels (the token MLP width and the voxel-path width adapt). Channels are removed, not zeroed, so a dead input cannot influence the arm. Everything else is identical across arms.

**Protocol.** Run it on the best configuration from steps 2 to 5, with the same chunk, epochs, seed and validation set for every arm. Repeat the P+D+w arm with a second seed to measure the seed-to-seed spread. **Keep D or w only if it improves mean validation Dice by more than that spread**; otherwise drop it and simplify the model. Report mean Dice, per-organ Dice (small organs 4, 11, 12 in particular), and the atlas-only vote as the reference.

**Cost.** On the laptop, with a ~150-case training chunk, about 3 hours per 50-epoch run, so four runs (three arms plus the extra seed) are about 12 hours. This estimate is extrapolated from the 24-case run, not measured at that size.

## Backbone ablation: UNETR vs Swin UNETR (steps 10 to 12)

**Status (2026-10-28).** The UNETR + prior cell is the 150/50, 100-epoch run in progress (`chunk_150_50_e100`). The other three cells are queued behind it by `scripts/queue_ablation.py` and start automatically.

**Motivation.** The anatomical tokens and the voxel path already give the network global anatomy, so global attention between, say, aorta and liver voxels is not what the backbone has to supply. Organ boundaries and texture are local. Two further reasons to expect Swin to help: the UNETR token grid is only 6x6x6 (16^3 patches), which is coarse for the weakest organs (organs 4, 11, 12), and a hierarchy with local windows is a stronger inductive bias for a from-scratch model on 150 cases. Swin is not cheaper: UNETR attends over 216 tokens, Swin over 110 k at its first level. Its last stage is effectively global (3^3 tokens), so locality is a bias, not a restriction. This is a hypothesis; the runs below test it.

**Design.** 2 x 2 grid, every cell with the same chunk (150 train / 50 val), 100 epochs, seed 0, optimizer type, schedule shape, augmentation, AMP and effective batch 2, and a learning rate tuned per backbone (below), trained from scratch (no pretrained weights for either backbone).

| Cell | Backbone | Prior | Run name | Step |
|---|---|---|---|---|
| A | UNETR (ViT, 122.4 M) | P, D, w | `chunk_150_50_e100` | 7 (done by the main run) |
| B | Swin UNETR | P, D, w | `chunk_150_50_e100_swin_prior` | 10 |
| C | UNETR | none (`--image-only`) | `chunk_150_50_e100_vit_imageonly` | 11 |
| D | Swin UNETR | none (`--image-only`) | `chunk_150_50_e100_swin_imageonly` | 12 |

"Image only" feeds uninformative priors (P = 1/13, D = 1, w = 0) in training and validation, so it is the same network with the atlas switched off. Readings: B versus A is the backbone effect with the prior; D versus C is the backbone effect without it; (A - C) versus (B - D) shows whether the prior helps the two backbones equally. All are read against the atlas-only vote on the same 50 cases (0.424).

**Hyperparameters: tuning is deferred to the server (decision 2026-10-28).** The local queue runs both backbones at the default learning rate 2e-4, which was never tuned and may favour UNETR; read the local ablation with that caveat. Learning-rate tuning per backbone is built into the queue (`--tune`, off by default) and is to be run on the server with the whole data. The protocol it implements:

- Grid 5e-5, 1e-4, 2e-4, 4e-4, with the minimum learning rate at LR/20 (the same ratio as 2e-4 → 1e-5), 25 epochs per run, with the full prior setup (`--gamma-init 1.0`).
- **No selection on the real validation set.** Each tuning run trains on 100 of the 150 training cases and is scored on the other 50 (`--tune-holdout`, fixed shuffle, 27 `s` / 19 `amos` / 4 `img`; checked disjoint from the real 50 validation cases). The score is the mean validation Dice over the last 3 epochs.
- The best learning rate is used for every cell of that backbone, including the image-only cell (so each backbone's with-prior and image-only runs differ only in the prior). If UNETR's best is not 2e-4, cell A is re-run at the new rate (`chunk_150_50_e100_vit_prior_lr<value>`), and the earlier 2e-4 run stays as a reference.
- Fixed for both: AdamW, weight decay 1e-4, gradient clip 1.0, warm-up 1 epoch, cosine decay, effective batch 2, prior dropout 0.3, consistency weight 0.1. Weight decay and batch size are **not** tuned.
- Caveats: a 25-epoch schedule favours higher learning rates than a 100-epoch one, 25 epochs on 100 cases is a noisy proxy (single seed), and a best value at the edge of the grid is logged as such. The image-only cells may have a different optimum than the with-prior ones; they share the with-prior rate on purpose.
- Cost: 8 tuning runs (4 per backbone) of about 1.2 h each for UNETR, Swin unknown, before the main runs. Total roughly 50 h or more on the laptop; the tuning ranking is logged in `logs/queue_ablation.log` as it completes, so it can be cut short.

**Prior integration for Swin** (`models/swin_atlas.py`, same paths as the ViT model, adapted to Swin's stages):

| Path | UNETR | Swin UNETR |
|---|---|---|
| Voxel path | encoder1 sees [CT, P, D, w] | input has 27 channels; patch embedding and encoder1 both see it |
| Anatomical tokens | 216 tokens cross-attended in all 12 blocks | the same tokens, cross-attended after Swin stages 2, 3, 4 (12^3, 6^3, 3^3 grids) |
| Trust gate | scalar per block, init 1 | scalar per adapter, init 1 |
| Output fusion | `z + gamma_c * w * log(P + 1e-3)`, `--gamma-init 1.0` | identical |
| Consistency loss, prior dropout | KL weight 0.1 ramped over 10 epochs, p = 0.3 | identical |

Stage 1 (24^3 tokens) has no cross-attention (13.8 k queries are costly on a 4 GB card, and the voxel path covers that resolution). Differences that remain between the cells: Swin has a different parameter count, and its voxel prior also enters the patch embedding. The comparison is "each backbone with its best-effort version of the same prior integration", not an identical parameter budget.

**Memory and speed.** The laptop GPU has 4 GB. Before the first Swin run the queue benchmarks, in order, feature size 48, feature size 48 with activation checkpointing, and feature size 24 with checkpointing, and takes the first that fits. The chosen configuration and the measured seconds per epoch are in `logs/queue_ablation.log`. If Swin fits in none, the Swin cells are skipped and logged. Swin epoch time is unmeasured; a UNETR epoch is about 250 s, so the three runs together are at least 21 h if Swin is as fast as UNETR and probably more.

**Caveats.** One seed and 50 validation cases: a gap below about 0.01 is not a result (the seed spread has not been measured; step 7 plans to measure it). The image-only UNETR collapsed at 48 cases (0.06), so cell C may be a weak reference rather than a fair "image-only" ceiling; if Swin learns image-only where UNETR does not, that is itself a finding about inductive bias.

**Decision rule.** Adopt Swin as the main backbone only if B beats A by more than the seed spread and the gain shows on the small organs (4, 5, 11, 12) and not just on large ones, at an epoch time the laptop can afford.

## Implementation Plan

**Order of implementation** (each part independent, compile/test between):

1. **Transforms (B):** Prior dropout in `data/transforms.py` + test.
2. **Confidence input (C, E):** Extend `AnatomicalPriorGenerator` to accept confidence + gating in `CustomViT` + output fusion scalar. Test at init against epoch-101 checkpoint.
3. **Voxel prior (D):** Modify `encoder1` input handling + zero-init new channels. Test.
4. **Script flags:** Add `--conf-gating`, `--voxel-prior`, `--output-fusion`, `--consistency-weight`, `--prior-dropout` to `scripts/train.py`.
5. **Consistency loss (F):** Add to `train_one_epoch` with ramp-up logic.
6. **Unit tests:** Each piece tested for init-time equivalence to epoch-101 checkpoint.

**Lines of code:** ~150 new, ~10 modified.

**Parameters added:** 12 gates + 13 output-fusion weights + 1 MLP column (768*2 = 1.5 k) + ~12 k voxel-path weights = ~14 k, against 122.3 M (0.01 %). Memory and speed are essentially unchanged.

**Memory:** No change. Confidence input is a new channel to the model but reuses existing memory for prior computation.

**Speed:** No change. All new operations are linear in batch size.

## Backward compatibility

- **Checkpoint loading:** `strict=False` on first run to load epoch-101 weights (missing new params). After that, `strict=True`.
- **Inference:** Set all flags to False (default) to reproduce epoch-101 behavior exactly.
- **Data:** No change to registration pipeline or data loading. Confidence maps already in the cache.

## Success metrics

- Step 0: Establishes image-alone baseline.
- Step 3: Model output ≥ atlas-only Dice on validation (currently 0.13 << 0.44).
- Step 4: Small organs improve per-organ Dice.
- Step 5: Consistency loss adds ≥ 1–2% Dice.
- Step 6 (diagnostic, read in both directions): if shifting the priors by 5 to 20 mm lowers Dice clearly, the model uses registration detail and further registration tuning is worth its cost; if Dice barely moves, registration precision is not the bottleneck and effort belongs in the architecture (steps 2 to 5) instead.

- Step 7: D and w are kept only if they beat the P-only arm by more than the seed-to-seed spread.

## References

- `research_audit.md` Section A: current architecture.
- `experiments/registration_log.md`: prior quality numbers.
- `prototype.ipynb`: smoke test baseline numbers.
- Paper: Experiments 1–4 correspond to steps 1–5 above.
- Note: this plan is ~400 lines; an earlier chat message that said 970 lines was wrong.

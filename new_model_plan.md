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

### G. CT edge loss and HU loss (Experiment 3 & 4, future)

**Where:** `training/engine.py` → new loss functions

**What:** 
- **CT edge loss:** Encourage high confidence where CT edges exist (organ boundaries).
- **HU loss:** Regularize the model to respect Hounsfield unit ranges per organ.

**Why:** Paper Experiments 3 and 4. Only implement if Experiment 2 helps.

**Status:** Documented, not yet implemented.

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

All runs on the same train/val split, augmentation and 96³ grid. **Heavy runs go to the server.** Local smoke tests only.

| Step | What | Flag(s) | Why | Validation gate |
|---|---|---|---|---|
| 0 | **Stock UNETR, no priors** | new (`--prior-source none` does **not exist yet**; `PRIOR_KEYS` has only `gt` and `atlas`; needs a prior-free model path) | Baseline: image alone | Essential reference R |
| 1 | Fine-tune epoch-101 with atlas priors | `--prior-source atlas` | Current architecture + A | Dice ≥ R? |
| 2 | + prior dropout | `--prior-dropout 0.3` | Add B | Dice ≥ step 1? |
| 3 | + conf input/gate + output fusion | `--conf-gating --output-fusion` (new) | Add C + E | Beat atlas-only (0.44–0.48)? |
| 4 | + voxel-level prior | `--voxel-prior` | Add D | Per-organ Dice on small organs? |
| 5 | + atlas-consistency loss | `--consistency-weight 0.1` | Add F (Exp 2) | Dice gain? |
| 6 | Sensitivity: shift priors | `--shift-prior 5` (mm) | Diagnostic | Is registration tuning worth it? |

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

## References

- `research_audit.md` Section A: current architecture.
- `experiments/registration_log.md`: prior quality numbers.
- `prototype.ipynb`: smoke test baseline numbers.
- Paper: Experiments 1–4 correspond to steps 1–5 above.
- Note: this plan is ~400 lines; an earlier chat message that said 970 lines was wrong.

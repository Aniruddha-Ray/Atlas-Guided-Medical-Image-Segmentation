# Atlas-guided ViT-UNETR: how the whole model works

A walkthrough for someone new to the project. It follows the data from a raw CT scan to a segmentation: why UNETR, how atlases become three prior maps (P, D, w), how those priors enter the transformer through self- and cross-attention, and how the loss is extended with CT physics. Equations are included; each is preceded by a plain-language statement of what it does.

**Status labels used throughout.** *Implemented* = code exists and runs. *Prototype* = implemented in `prototype.ipynb` / `scripts/train_prototype_v2.py`, not yet in `models/`. *Proposed* = formulated here, no code yet. Numbers are a snapshot from late October 2026 and come from small runs; `research_audit.md` has the current record.

---

## 1. The task in one paragraph

Given an abdominal CT volume, label every voxel as background or one of 12 organs. A plain network learns this from labelled examples only. Our idea is to also give it **domain knowledge**: a handful of fully labelled "atlas" scans are aligned (registered) onto the new scan, their labels are warped across and combined, and the result is shown to the network as three extra maps. The network then only has to refine a rough anatomical guess instead of finding organs from scratch.

```
new CT ──────────────────────────────────────────────────────────┐
                                                                  ▼
10 atlas CTs + labels ──► register onto new CT ──► warp labels ──► P, D, w ──► ViT-UNETR ──► 13-class mask
   (landmarks, TPS)         (Section 4)           (vote)                        (Section 5)      ▲
                                                                                               loss (Section 6)
```

Data: 680 paired CT/label cases from three sources (`amos`, `img`, `s`), 12 organs plus background (13 classes), split 80/20 into training and validation with a fixed seed. Every network input is a **96×96×96** volume made by cropping to the body, resampling to 1 mm, scaling intensity and resizing.

Notation used below: $\Omega$ is the 96³ voxel grid, $I(v)$ the CT in Hounsfield units (HU), $y_c(v)\in\{0,1\}$ the one-hot label, $z_c(v)$ the network logits and $\hat y_c(v)=\mathrm{softmax}_c(z(v))$ the predicted probability, with $c\in\{0,\dots,12\}$ and $C=13$.

---

## 2. Why UNETR

UNETR (Hatamizadeh et al., 2022) is a 3D segmentation network with a **Vision Transformer (ViT) encoder** and a **convolutional decoder** joined by skip connections. We use MONAI's implementation and subclass it.

Reasons it fits this project:

1. **Cross-attention needs tokens.** The core idea is to let image features *ask questions of* anatomical information. A transformer works on a sequence of tokens, so anatomical information can be turned into tokens and attended to. A pure CNN U-Net has no such interface.
2. **Global context.** Organ positions are related (the spleen is on the opposite side to the liver, the pancreas sits near the stomach). Self-attention lets every part of the volume see every other part from the first layer, which a CNN only reaches after many layers.
3. **The decoder keeps detail.** The transformer works on coarse patches; the CNN decoder and the full-resolution skip recover boundaries.
4. **Convenient to modify.** MONAI's `TransformerBlock` already supports cross-attention, and its blocks can be subclassed.

Honest limits: the ViT holds 116.7 M of the model's 122.3 M parameters, so it is data-hungry for 680 cases, and its 16³ patches are coarse (a 96³ input becomes only 6×6×6 tokens). We have not benchmarked UNETR against alternatives such as nnU-Net in this project.

### 2.1 The mechanics

**Patch embedding.** The input is cut into $N=(96/16)^3=216$ non-overlapping 16³ patches. Each is flattened and linearly projected to a 768-dimensional token, and a learned position embedding is added:

$$z_0 = [\,x_1E;\ x_2E;\ \dots;\ x_NE\,] + E_{\text{pos}},\qquad E\in\mathbb R^{4096\times 768}.$$

**Transformer block (self-attention).** Each of the 12 blocks lets every token look at every other token, then applies a small per-token network (MLP):

$$z'_\ell = z_{\ell-1} + \mathrm{MSA}\big(\mathrm{LN}(z_{\ell-1})\big),\qquad z_\ell = z'_\ell + \mathrm{MLP}\big(\mathrm{LN}(z'_\ell)\big).$$

Multi-head self-attention uses 12 heads of size $d_h=768/12=64$. For one head, with queries $Q$, keys $K$, values $V$ all computed from the same tokens $z$:

$$\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\Big(\frac{QK^{\top}}{\sqrt{d_h}}\Big)V,\qquad Q=zW_Q,\ K=zW_K,\ V=zW_V .$$

Read it as: each token scores its similarity to every token (a $216\times216$ matrix), turns the scores into weights, and takes the weighted average of the others' values.

**Skip connections and decoder.** Features from three intermediate blocks (indices 3, 6, 9 of 0–11, i.e. the 4th, 7th and 10th) and the final output are reshaped back into 3D grids and upsampled with transposed convolutions. A separate convolutional branch ("encoder1") processes the raw CT at full resolution. The decoder merges them from coarse to fine and a final convolution outputs 13 logits per voxel. Base feature width is 16.

---

## 3. Adding domain knowledge: three prior maps

An **atlas** is a CT scan with an expert organ labelling. We use 10 of them. After registration, each atlas gives its own opinion about what every voxel of the new scan is. Combining the 10 opinions gives three maps on the same grid as the CT:

| Map | Channels | Plain meaning | Range |
|---|---|---|---|
| **P(v)** probability | 13 | "What fraction of the atlases say this voxel is organ c?" | 0 to 1 per class, sums to 1 |
| **D(v)** distance | 12 | "How far is this voxel from where the atlases put organ c?" | 0 inside the atlas organ, 1 far away |
| **w(v)** confidence | 1 | "How much do the atlases agree about this voxel?" | 0 to 1 |

Why three? P says *what*. D adds *how near*: P is piecewise constant (with 10 atlases it only takes 11 values), so it has no information a few millimetres outside an organ, whereas D varies smoothly and tolerates small misalignment. w says *how much to trust* the other two. All three are computed from the same 10 votes, so they carry the same raw information in different forms; whether D and w earn their place is an open experiment (the P / P+D / P+D+w ablation in `new_model_plan.md`).

**Leakage rule.** The new scan's own label is never used to build P, D or w. It is used only as the training target and, for validation scans, as the evaluation reference. The original prototype broke this rule (it used the scan's own label as P); that is why its 0.65 Dice cannot be compared with atlas-based results.

---

## 4. Building P, D and w from the atlases

### 4.1 One-time: the atlas library (`scripts/build_atlas_library.py`)

Registration needs landmarks that can be found reliably in a new scan. We choose them from the atlas labels and keep only the dependable ones.

1. **Landmarks.** For each organ, take 40 points spread evenly over the organ surface (farthest-point sampling). Small organs (under 8 mL) get 5 interior points instead, because surface points are noisiest for tiny organs. Coordinates are in millimetres in world space.
2. **Golden transformation.** For every ordered atlas pair $(i,j)$, both fully labelled, match the landmarks organ by organ (after a rigid alignment on the organ centres) and fit a smooth warp $T_G^{(j\to i)}$, a regularised thin-plate spline (TPS, defined in 4.2). Each matched pair $k$ has a residual

$$r_k^{(i,j)}=\big\|\,x_{i,k}-T_G^{(j\to i)}(x_{j,k})\,\big\|.$$

A landmark with a large residual after a *smooth* global fit is a locally ambiguous point, not one to hunt for in a new scan.

3. **Reliability score**, with tolerance $T=15$ mm, averaged over the other $N-1=9$ atlases:

$$s_k^{(i,j)}=\frac{\max(T-r_k^{(i,j)},\,0)}{T},\qquad \mathrm{Score}_i(f_k)=\frac1{N-1}\sum_{j\neq i}s_k^{(i,j)}.$$

Landmarks with $\mathrm{Score}\ge 0.3$ are kept, plus up to 3 best per organ so no organ loses all its landmarks.

### 4.2 Per scan: registering each atlas onto the new CT (`scripts/register_targets.py`)

The new scan has no labels, so landmarks must be found from image intensities alone. For each atlas:

**Step 1, coarse alignment.** Mutual-information registration at 4 mm on body masks, done in stages so scale is never free before position is right: align body centres in-plane, search over z offsets (scans cover different stretches of the body), fit a similarity transform (rigid plus one scale), then an affine accepted only if every axis scale stays within 0.8–1.25. This only needs to bring each landmark inside the next step's search window.

**Step 2, patch matching.** For each landmark, take a 9³-voxel patch (18 mm at 2 mm resolution) from the atlas and search a window of the new scan for the offset $\Delta$ with the highest normalised cross-correlation:

$$\mathrm{NCC}(\Delta)=\frac{\sum_{u}\big(a(u)-\bar a\big)\big(b(u+\Delta)-\bar b\big)}{\sqrt{\sum_u (a(u)-\bar a)^2\ \sum_u (b(u+\Delta)-\bar b)^2}} .$$

NCC ignores linear brightness changes, so it tolerates differences in contrast phase and scanner. Intensities are first windowed to soft tissue ($[-200,300]$ HU). A landmark is discarded if its patch is nearly flat (it would match anywhere) or if its best NCC is below 0.3. The search is done twice: ±20 mm around the coarse result, then ±12 mm around the RANSAC result of the first pass.

**Step 3, RANSAC.** Patch matching always returns a best offset, even for wrong matches. RANSAC repeatedly fits an affine map $M$ to 4 random correspondences and keeps the largest consistent set:

$$M^\*=\arg\max_M\ \big|\{k:\ \|M a_k - t_k\|<12\ \text{mm}\}\big|,$$

with each hypothesis required to be plausible (no mirroring, $\det M>0$; per-axis scale within 0.6–1.6). At least 15 inliers are required.

**Step 4, thin-plate spline.** A TPS is a smooth non-rigid warp through the inlier correspondences, mapping new-scan points to atlas points:

$$T(x)=Ax+b+\sum_{k}w_k\,\|x-t_k\|.$$

The weights come from the linear system

$$(K-\lambda I)\,w+P\,\begin{pmatrix}b\\A\end{pmatrix}=a,\qquad P^{\top}w=0,\qquad K_{kl}=\|t_k-t_l\|,$$

which minimises the fit error plus $\lambda$ times a bending energy. (Note the minus sign: for this kernel $K$ is conditionally negative definite. An earlier version of our code had $K+\lambda I$ and produced erratic warps; see `research_audit.md`, bug 3.) The warp must not fold: the fraction of points with $\det\nabla T\le 0$ must stay below 0.001, otherwise $\lambda$ is multiplied by 4 and the fit repeated. The mapping is fitted from the *new scan to the atlas*, so no spline inversion is needed.

**Recorded fallbacks.** If the TPS keeps folding the result is the RANSAC affine; if there are too few inliers it is the coarse affine. The status of every atlas is stored, never silent. Targets where 3 or more of the 10 atlases fell back averaged Dice 0.08 against 0.43 for none.

### 4.3 Label warping (`atlas/label_warping.py`)

With the mapping $T_i$ from the new scan to atlas $i$, warping is a pull-back resample with nearest-neighbour interpolation: every voxel of the new scan looks up its label in the atlas,

$$L'_i(v)=L_i\big(T_i(v)\big).$$

The TPS is evaluated on a 6 mm grid and turned into a displacement field for speed. The 10 warped labels per scan are saved together as one 4D file. P, D and w are **not** saved.

### 4.4 From warped labels to P, D and w (`data/transforms.py`, `AtlasPriorsd`)

They are computed at training time, on the final 96³ grid after cropping, resizing and random flips/rotations, so that they match the augmented CT exactly. The warped stack is resampled onto that grid with nearest neighbour, then:

$$P_c(v)=\frac1N\sum_{i=1}^{N}\mathbf 1\!\left[L'_i(v)=c\right]\qquad(N=10).$$

**Confidence** is one minus the normalised entropy of the vote:

$$w(v)=1-\frac{H\big(P(v)\big)}{\log C},\qquad H(P)=-\sum_{c}P_c\log P_c .$$

It is 1 when all atlases agree, and lower the more the vote is split.

**Distance.** First define the atlas region of organ $c$ as $\mathcal O_c=\{u:\ P_c(u)\ge 0.5\,\max_v P_c(v)\}$. The relative threshold keeps a poorly agreeing small organ from ending up with an empty region. Then

$$D_c(v)=\min\!\Big(1,\ \frac{1}{50\ \text{mm}}\,\min_{u\in\mathcal O_c}\|v-u\|_{\text{mm}}\Big).$$

Distances use true voxel spacing in millimetres (the 96³ grid is anisotropic) and are divided by the clip value so $D$ lies in $[0,1]$ like $P$. An organ that no atlas places in view gets $D_c=1$ everywhere. (The original prototype computed a raw voxel-unit distance from the scan's own label, with inverted semantics, which was a bug.)

### 4.5 Training-time prior dropout (prototype)

With probability $p=0.3$ per training sample the priors are replaced by "no information", so the network cannot rely on them completely:

$$(P,D,w)\ \leftarrow\ \big(\tfrac1{13}\mathbf 1,\ \mathbf 1,\ \mathbf 0\big).$$

It is never applied to validation scans.

### 4.6 How good are the priors?

*Atlas set note: the ten atlases were replaced on 2026-10-28 by a quality-selected set (slice thickness at most 3 mm, complete coverage, clean labels, typical anatomy). The figures in this subsection were measured with the **previous** ten; on 12 separate training targets the new set raised the atlas-only Dice from 0.380 to 0.450 (see `experiments/registration_log.md`). Validation numbers for the new set are still to be produced.*

The vote alone ($\arg\max_c P_c$), scored against the true label for evaluation only, gives a mean Dice of **0.392** over the 136 validation scans (range 0.00 to 0.59). Failures come from scans that cover a short stretch of the body and from atlases that fall back to the coarse affine. The current 10 atlases were chosen without a quality check, and a better-selected set is being evaluated (`research_audit.md`, Section I).

---

## 5. How the priors enter the network

### 5.1 Anatomical tokens: P, D **and w** (original model uses P and D only)

Each prior map is averaged inside every 16³ patch, so there is one summary per patch location, then turned into a token by a small MLP. With $\Omega_n$ the voxels of patch $n$:

$$\bar P_n=\frac1{|\Omega_n|}\sum_{v\in\Omega_n}P(v)\in\mathbb R^{13},\qquad \bar D_n\in\mathbb R^{12},\qquad \bar w_n\in\mathbb R^{1}.$$

**Redesigned model (main path):** all three are concatenated and sent through the MLP,

$$A_n=W_2\,\mathrm{GELU}\big(W_1[\bar P_n;\bar D_n;\bar w_n]+b_1\big)+b_2,\qquad W_1\in\mathbb R^{1536\times26},\ W_2\in\mathbb R^{768\times1536}.$$

**Original model (epoch-101):** the same MLP without $\bar w_n$, so $W_1\in\mathbb R^{1536\times25}$. It never saw w.

The result is $A\in\mathbb R^{216\times768}$, one anatomical token per patch, in the same space as the image tokens. When the epoch-101 weights are loaded into the redesigned model, the new column of $W_1$ that reads $\bar w_n$ is zero-initialised, so w has no effect at the start and is learned during training. (In the code this token change and the trust gates of 5.2 are switched on together by one flag, `conf_gating`.)

**Where each prior enters the network:**

| Entry point | P | D | w |
|---|---|---|---|
| Anatomical tokens, read by cross-attention in all 12 blocks (5.1, 5.2) | yes | yes | yes (redesigned model only) |
| Full-resolution CNN branch, `encoder1` input (5.3) | yes | yes | yes |
| Output fusion on the logits (5.3) | yes (as $\log P$) | no | yes (as a mask) |
| Atlas-consistency loss (6.2) | yes (the target) | no | yes (the weight) |
| Prior dropout (4.5) | all three | | |

So w is in the main path, and also in the other three places. D enters only through the tokens and the voxel branch. In the P / P+D / P+D+w ablation (`new_model_plan.md`), the P-only arm removes D and w from the tokens and the voxel branch, and uses $w\equiv1$ in the output fusion and the loss.

### 5.2 Self-attention plus cross-attention in every block (implemented; the gate is prototype)

Each of the 12 blocks does three things in order: image tokens attend to each other, then attend to the anatomical tokens, then pass through the MLP. The cross-attention uses the same formula as before but the keys and values come from a *different* source:

$$z'_\ell=z_{\ell-1}+\mathrm{MSA}\big(\mathrm{LN}(z_{\ell-1})\big)\qquad\text{(self)}$$
$$z''_\ell=z'_\ell+g_\ell\cdot\mathrm{CrossAttn}\big(\mathrm{LN}(z'_\ell),\,A\big)\qquad\text{(cross)}$$
$$z_\ell=z''_\ell+\mathrm{MLP}\big(\mathrm{LN}(z''_\ell)\big)$$

$$\mathrm{CrossAttn}(u,A)=\mathrm{softmax}\!\Big(\frac{(uW_Q)(AW_K)^{\top}}{\sqrt{d_h}}\Big)(AW_V).$$

Read it as: each image token forms a query from what it currently knows and asks all 216 anatomical tokens "which of you is relevant to me?"; the answer is a weighted mix of anatomical information added to the token. Image tokens are always the queries, so anatomy never overwrites the image features, it is only added to them.

The scalar $g_\ell$ is a learnable "trust gate", one per block, initialised to 1 (the original model has $g_\ell\equiv1$). Training can lower it where the prior hurts.

One design point worth testing: the anatomical tokens $A$ carry no positional embedding, so the model infers location from content only.

### 5.3 Two more entry points for the prior (prototype)

**Voxel-level path.** Pooling to 16³ patches (roughly 40–60 mm) discards fine detail, so the full-resolution CNN branch also receives the priors directly. Its input has 27 channels:

$$\tilde X=\big[\tilde I,\ P,\ D,\ w\big]\in\mathbb R^{(1+13+12+1)\times 96^3},\qquad \tilde I=\frac{I+1000}{2000}.$$

**Output fusion.** A learnable weight $\gamma_c$ per class adds the log-prior directly to the logits:

$$\tilde z_c(v)=z_c(v)+\gamma_c\,w(v)\,\log\big(P_c(v)+\varepsilon\big),\qquad \varepsilon=10^{-3}.$$

This has a useful property: if the network's own logits were zero and $\gamma_c=1$, then $\mathrm{softmax}(\tilde z)\propto P^{\,w}$, which reproduces the atlas vote. So $\gamma$ lets the model start from the vote and learn corrections.

All new weights are zero-initialised (gates at 1), so by design the redesigned model starts identical to the original when loaded from the epoch-101 checkpoint. (The notebook has checks for this; I have not re-run them.)

---

## 6. The loss

### 6.1 Supervised loss (implemented)

Dice measures overlap and cross-entropy measures per-voxel accuracy; the loss is their sum:

$$\mathcal L_{\sup}=\underbrace{1-\frac1C\sum_{c}\frac{2\sum_v\hat y_c y_c}{\sum_v\hat y_c+\sum_v y_c}}_{\text{Dice}}\ \underbrace{-\ \frac1{|\Omega|}\sum_v\sum_c y_c\log\hat y_c}_{\text{cross-entropy}}$$

(MONAI's `DiceCELoss`, with small smoothing constants in the Dice denominators.)

### 6.2 Atlas consistency (prototype)

Pull the prediction toward the atlas vote, but only where the atlases agree (large $w$). KL divergence measures how far $\hat y$ is from $P$:

$$\mathcal L_{\text{atlas}}=\frac{1}{|\Omega|}\sum_{v}w(v)\sum_{c}P_c(v)\log\frac{P_c(v)}{\hat y_c(v)}$$

with its weight ramped up: $\lambda_a(t)=\lambda_{\max}\min(1,t/T)$, using $\lambda_{\max}=0.1$ and $T=10$ epochs.

### 6.3 CT physics (implemented in `training/physics_losses.py`; not yet evaluated in a real training run)

The two terms below follow the physics of how a CT image is formed. The original research report's exact definitions are not in this repository, so these formulations are ours and still need to be reconciled with the report.

**What the CT physically measures.** X-rays are attenuated along their path as $I_{\text{det}}=I_0\exp\!\big(-\int\mu\,dl\big)$, and reconstruction recovers the attenuation coefficient $\mu$ at each voxel. The Hounsfield unit expresses it relative to water:

$$\mathrm{HU}(v)=1000\,\frac{\mu(v)-\mu_{\text{water}}}{\mu_{\text{water}}}.$$

Three consequences matter for segmentation:

1. **Tissue has a characteristic HU.** Air is about −1000, fat about −100, water 0, soft-tissue organs roughly 30 to 150 depending on the organ and contrast, with contrast-filled vessels and bone higher still. Within one scan an organ's HU is fairly uniform. (In the 10 candidate atlases we inspected, the mean liver HU ranged from 34 to 121.)
2. **That HU shifts between scans**, with contrast phase, dose and scanner. Fixed population constants would be wrong, so tissue statistics must be estimated **per scan**.
3. **Attenuation adds linearly**, so a voxel partly filled with two tissues has the volume-weighted mean HU (the partial-volume effect): $I(v)\approx\sum_c f_c(v)\,\mathrm{HU}_c$. Our 96³ CT is made by area-averaging a finer grid, so this holds for the network input too, and it is why soft $\hat y_c$ (fractions) are the right output.

**HU consistency loss.** Voxels the network assigns to organ $c$ should look like organ $c$ in this scan. Per-scan statistics $\mu_c,\sigma_c$ are estimated only from the confident core of the atlas prior (never the target label), with $\Omega_c=\{v:P_c(v)\ge0.9,\ w(v)\ge0.8\}$:

$$\mu_c=\operatorname{median}_{v\in\Omega_c}I(v),\qquad \sigma_c=\max\!\big(1.4826\,\mathrm{MAD}_{v\in\Omega_c}I(v),\ \sigma_{\min}\big).$$

$$\mathcal L_{HU}=\frac{1}{|\mathcal V|}\sum_{c\in\mathcal V}\frac{\sum_v \hat y_c(v)\ \rho_\delta\!\big((I(v)-\mu_c)/\sigma_c\big)}{\sum_v\hat y_c(v)+\varepsilon},\qquad \mathcal V=\{c\ge1:\ |\Omega_c|\ge 50\}$$

Here $\rho_\delta$ is the Huber function, which grows only linearly for large deviations so that vessels or lesions inside an organ do not dominate. Only organs enter (background HU is a mix of air, fat, muscle and bone), and each organ's cost is divided by its predicted volume. Without that normalisation, shrinking the predicted organs would trivially lower the loss. Even normalised, the loss can be lowered by predicting only the best-fitting voxels, so it must stay a small regulariser next to $\mathcal L_{\sup}$. An organ contributes only if its confident atlas core has at least 50 voxels. That is also how a failed registration switches the term off: its $w$ is low, so no organ has a core and the term is zero for that scan. The statistics use the priors from before prior dropout (4.5), since they are not a network input.

**Edge loss.** Tissue boundaries appear where $\mu$ changes, as an intensity gradient. A predicted organ boundary in a flat region of the CT is therefore suspect. With $I_s$ a lightly Gaussian-smoothed CT (CT noise is quantum noise), gradients taken in millimetres, and $\kappa$ the median gradient magnitude of the scan:

$$\mathcal L_{\text{edge}}=\frac1{|\Omega|}\sum_{v}\sum_{c=1}^{12}\big\|\nabla\hat y_c(v)\big\|\ \exp\!\Big(-\frac{\|\nabla I_s(v)\|}{\kappa}\Big).$$

It penalises prediction changes where the CT is flat and barely penalises them on strong edges (an edge-aware total-variation term). Some true organ boundaries have almost no contrast (pancreas against duodenum), so this is a gentle regulariser that discourages noisy speckle, not a target. Its minimum is a prediction with no boundaries at all, so it only makes sense together with $\mathcal L_{\sup}$.

**Implementation details.** Huber threshold $\delta=2$ (in units of $\sigma_c$), $\sigma_{\min}=10$ HU, $\varepsilon=1$ voxel. The CT is smoothed with a Gaussian of 1 voxel; $\kappa$ is the median gradient magnitude over body voxels (HU above −500), floored at $10^{-4}$. Gradients use each sample's voxel spacing read from its affine (on the 96³ grid it is strongly anisotropic, for example 4.7 × 3.3 × 6.6 mm). Everything is computed in float32, also under mixed precision. Flags in `scripts/train_prototype_v2.py`: `--hu-weight`, `--edge-weight` (both 0 by default), plus `--hu-delta`, `--hu-core-p`, `--hu-core-w`, `--hu-min-voxels`, `--hu-sigma-min`, `--edge-sigma-vox`. 11 unit tests in `tests/test_physics_losses.py`.

**First checks on real scans** (12 training scans through the training pipeline, before any training; lower is better):

| Prediction scored | $\mathcal L_{HU}$ | $\mathcal L_{\text{edge}}$ |
|---|---|---|
| True labels | 4.20 | 0.0018 |
| True labels shifted by 2 voxels | 7.59 | 0.0022 |
| Atlas vote | 6.29 | 0.0019 |
| Uniform guess | 69.1 | 0.0000 |

- The HU loss points the right way: it prefers the truth to the atlas vote on 10 of 12 scans and to the shifted truth on 12 of 12.
- The edge loss detects misplaced boundaries (12 of 12) but cannot tell the truth from the atlas vote (6 of 12), so expect it to act mainly as a smoothness term.
- The HU term gets statistics for 1 to 7 of the 12 organs per scan (median 6); small organs never have a confident core.
- Cost on the laptop GPU: training step 0.88 → 1.37 s (+56 %), peak memory 2.95 → 3.23 GiB (fits in 4 GiB).
- Scale: for a near-uniform (untrained) prediction the HU loss is about 70 to 110, against about 3 for $\mathcal L_{\sup}$, which is why both weights are ramped in like $\lambda_a$. Starting weights that make each term about 10 % of $\mathcal L_{\sup}$ for a reasonable prediction are $\lambda_h\approx0.03$ and $\lambda_e\approx50$. These come from loss magnitudes, not from tuning.

### 6.4 Total loss

$$\mathcal L=\mathcal L_{\sup}+\lambda_a(t)\,\mathcal L_{\text{atlas}}+\lambda_h(t)\,\mathcal L_{HU}+\lambda_e(t)\,\mathcal L_{\text{edge}},\qquad \lambda_\bullet(t)=\lambda_\bullet\min(1,\,t/T)$$

| Term | Status |
|---|---|
| $\mathcal L_{\sup}$ | Implemented |
| $\mathcal L_{\text{atlas}}$ | Prototype (`scripts/train_prototype_v2.py`) |
| $\mathcal L_{HU}$, $\mathcal L_{\text{edge}}$ | Implemented and tested (`training/physics_losses.py`, wired into `scripts/train_prototype_v2.py`); not yet evaluated in a real training run. To be added one at a time and kept only if validation Dice improves |
| Smoothness term from the research report | Not started |

---

## 7. How results are judged

- **Metric:** mean Dice over the 12 organs, on a fixed validation set of 136 scans that is never used to tune anything.
- **Two reference points that any atlas-guided model must beat:** (1) the atlas vote alone (0.392), and (2) an image-only network with no priors (not yet trained). If the model does not beat both, the atlases have not helped.
- **Leakage guards:** the 10 atlases are never training or validation targets; registration settings are tuned on training scans only.
- **One change at a time:** each addition (prior dropout, gates, voxel path, output fusion, consistency loss, each physics term, and the P / P+D / P+D+w ablation) is a flag, run separately; the order and decision rules are in `new_model_plan.md`.

### Snapshot of results so far

| Measure | Value |
|---|---|
| Atlas vote alone, 136 validation scans | 0.392 mean Dice |
| Original model, ground-truth priors (cannot be compared, uses the answer) | 0.65 |
| Original model, atlas priors, one scan | about 0.13, while the vote alone gives 0.42 there |
| Redesigned model, from scratch, 24 training and 6 validation scans | 0.168, against 0.350 for the vote on the same scans |

The redesigned model is not yet reading the priors effectively; the output-fusion weight barely moved in that short run. This is one small experiment, not a verdict.

---

## 8. Where things live

| What | File |
|---|---|
| Model (original) | `models/segmentation_model.py` |
| Redesigned model, prior dropout, consistency loss | `prototype.ipynb` (cell "v2-cell definitions"), `scripts/train_prototype_v2.py` |
| Atlas library, landmarks, golden transform | `atlas/landmarks.py`, `atlas/golden_transform.py`, `atlas/reliable_features.py` |
| Registration: matching, RANSAC, TPS | `atlas/matching.py`, `atlas/ransac.py`, `atlas/tps.py`, `atlas/registration.py` |
| Label warping, P / D / w | `atlas/label_warping.py`, `data/transforms.py` (`AtlasPriorsd`) |
| Server run commands | `RUN_GUIDE.md` |
| Findings, bugs, results | `research_audit.md`, `experiments/registration_log.md` |
| Redesign plan and experiment order | `new_model_plan.md` |

## 9. Glossary

- **Atlas:** a CT with an expert organ labelling, used as a reference.
- **Registration:** finding the spatial warp that maps one scan onto another.
- **Landmark:** a distinctive point (here on an organ surface or inside a small organ) used to guide registration.
- **TPS (thin-plate spline):** a smooth non-rigid warp that passes close to a set of point correspondences.
- **NCC (normalised cross-correlation):** a similarity score between two image patches that ignores linear brightness differences.
- **RANSAC:** a robust fitting method that repeatedly fits on random subsets and keeps the fit most points agree with, discarding outliers.
- **Dice:** overlap score between prediction and truth, 1 for perfect and 0 for no overlap.
- **HU (Hounsfield unit):** the CT intensity scale; water 0, air −1000.
- **Token:** a vector representing one patch (or one anatomical summary) inside a transformer.
- **Cross-attention:** attention where queries come from one set of tokens and keys and values from another.

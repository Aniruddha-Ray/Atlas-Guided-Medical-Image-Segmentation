# Atlas-guided Swin UNETR: how the whole model works

A walkthrough for someone new to the project. It follows the data from a raw CT scan to a segmentation: why Swin UNETR is the base network, how atlases become three prior maps (P, D, w), how those priors enter the network (at the input, as anatomical tokens read by cross-attention, and at the output), and how the loss is extended with CT physics. The ViT-UNETR version we started with is kept as the comparison backbone (Section 2.2). Equations are included; each is preceded by a plain-language statement of what it does.

**Base network.** Swin UNETR became the base backbone on 2026-10-30, after the 150-train / 50-validation comparison (0.694 against 0.606 for ViT-UNETR with the same prior integration). That is one seed with an untuned learning rate, so it still has to be confirmed by the seed spread, the image-only Swin run and the whole-data runs (`new_model_plan.md`).

**Status labels used throughout.** *Implemented* = code exists and runs (the Swin model in `models/swin_atlas.py`, 5 unit tests; the physics losses in `training/physics_losses.py`). *Prototype* = implemented in `prototype.ipynb` / `scripts/train_prototype_v2.py`, not yet in `models/` (the ViT-UNETR redesign, the prior dropout and the consistency loss, which both backbones share). *Proposed* = formulated here, no code yet. Numbers are a snapshot from 2026-10-30 and come from laptop runs on a 150 / 50 case chunk; `research_audit.md` and `new_model_plan.md` have the current record.

---

## 1. The task in one paragraph

Given an abdominal CT volume, label every voxel as background or one of 12 organs. A plain network learns this from labelled examples only. Our idea is to also give it **domain knowledge**: a handful of fully labelled "atlas" scans are aligned (registered) onto the new scan, their labels are warped across and combined, and the result is shown to the network as three extra maps. The network then only has to refine a rough anatomical guess instead of finding organs from scratch.

```
new CT ──────────────────────────────────────────────────────────┐
                                                                  ▼
10 atlas CTs + labels ──► register onto new CT ──► warp labels ──► P, D, w ──► Swin UNETR ──► 13-class mask
   (landmarks, TPS)         (Section 4)           (vote)                        (Section 5)      ▲
                                                                                               loss (Section 6)
```

Data: 680 paired CT/label cases from three sources (`amos`, `img`, `s`), 12 organs plus background (13 classes), split 80/20 into training and validation with a fixed seed. Every network input is a **96×96×96** volume made by cropping to the body, resampling to 1 mm, scaling intensity and resizing.

Notation used below: $\Omega$ is the 96³ voxel grid, $I(v)$ the CT in Hounsfield units (HU), $y_c(v)\in\{0,1\}$ the one-hot label, $z_c(v)$ the network logits and $\hat y_c(v)=\mathrm{softmax}_c(z(v))$ the predicted probability, with $c\in\{0,\dots,12\}$ and $C=13$.

---

## 2. Why Swin UNETR

Swin UNETR (Hatamizadeh et al., 2022) is a 3D segmentation network with a **Swin Transformer encoder** and a **convolutional decoder** joined by skip connections from every encoder stage. We use MONAI's `SwinUNETR` and subclass it (`models/swin_atlas.py`). It is the base network of this project; the ViT-UNETR we started with is the comparison backbone (Section 2.2).

Reasons it fits this project:

1. **Boundaries and texture are local; global anatomy is supplied separately.** The anatomical tokens and the voxel-level prior already tell the network where the organs are in the whole volume, so it does not have to relate aorta voxels to liver voxels by itself. What the backbone must supply is fine local appearance, which windowed attention and a feature hierarchy are good at.
2. **A hierarchy instead of one coarse grid.** ViT-UNETR works on a single 6×6×6 grid of 16³ patches. Swin keeps feature maps at 48³, 24³, 12³, 6³ and 3³, so small organs (gallbladder, adrenal glands) are represented at a resolution where they exist.
3. **The cross-attention interface survives.** Anatomical information is still turned into 216 tokens and attended to, by adapters after the three coarsest stages (Section 5.2).
4. **It measured better on our data.** On 150 training and 50 validation cases (one seed, learning rate 2e-4, untuned), Swin with the prior reached 0.694 against 0.606 for ViT-UNETR with the prior, better on all 12 organs, and scored 0.578 against 0.302 with the prior removed at test time. That is strong evidence on this chunk, not a verdict: the seed spread, the image-only Swin run and the whole-data runs are still to come.

Honest limits. Swin UNETR (feature size 48, with our additions) has 67.1 M parameters against about 122 M for ViT-UNETR, but it is about four times slower per epoch on the laptop (about 1,200 s against 295 s), because its first level works on 110,592 tokens; it peaks at 5.26 GiB and fits the 4 GiB card only by spilling into shared memory. Windows are local: long-range relations (the spleen on the opposite side to the liver) travel through the anatomical tokens, the voxel path and the last stage (3³ tokens, effectively global), not through attention between distant voxels. No pretrained weights are used for either backbone, and we have not benchmarked against nnU-Net in this project.

### 2.1 The mechanics

**Patch embedding.** The 27-channel input (Section 5.3) is cut into non-overlapping 2×2×2 patches and each is linearly projected to $C=48$ channels: a 48³ grid of 110,592 tokens.

**Window attention.** Attention is computed only inside non-overlapping windows of $7^3=343$ tokens, so the cost grows linearly with the volume instead of quadratically. Consecutive blocks alternate between the regular window grid and a grid shifted by half a window, so information crosses window borders:

$$\hat z_\ell=z_{\ell-1}+\mathrm{W\text{-}MSA}\big(\mathrm{LN}(z_{\ell-1})\big),\qquad z_\ell=\hat z_\ell+\mathrm{MLP}\big(\mathrm{LN}(\hat z_\ell)\big)$$

$$\hat z_{\ell+1}=z_{\ell}+\mathrm{SW\text{-}MSA}\big(\mathrm{LN}(z_{\ell})\big),\qquad z_{\ell+1}=\hat z_{\ell+1}+\mathrm{MLP}\big(\mathrm{LN}(\hat z_{\ell+1})\big)$$

Inside one window, a head computes $\mathrm{softmax}\big(QK^{\top}/\sqrt{d_h}+B\big)V$ over the window's tokens, where $B$ is a learned relative position bias.

**Stages and patch merging.** There are four stages of two blocks each (eight blocks), with 3, 6, 12 and 24 heads. After each stage, patch merging concatenates every 2×2×2 neighbourhood (8C channels) and projects it to 2C channels: the grid halves in every direction and the width doubles. The encoder therefore gives five feature maps, all of which feed the decoder:

| Level | Grid | Channels | Used as |
|---|---|---|---|
| after patch embedding | 48³ | 48 | skip (`encoder2`) |
| after stage 1 | 24³ | 96 | skip (`encoder3`) |
| after stage 2 | 12³ | 192 | skip (`encoder4`) |
| after stage 3 | 6³ | 384 | skip (`decoder5`) |
| after stage 4 | 3³ | 768 | bottleneck (`encoder10`) |

**Decoder.** A convolutional branch (`encoder1`) processes the raw input at full 96³ resolution. The decoder starts from the 3³ bottleneck and upsamples with transposed convolutions, adding a skip at each step (6³, 12³, 24³, 48³ and the 96³ branch); a final 1×1×1 convolution outputs 13 logits per voxel. The base feature width is 48 (the published size, `--swin-feature-size 48`).

### 2.2 The ViT-UNETR backbone (comparison)

UNETR (Hatamizadeh et al., 2022) has a **Vision Transformer (ViT) encoder** and the same kind of convolutional decoder. We use MONAI's implementation, subclassed in `prototype.ipynb`. It was the first backbone of this project and is the other side of the backbone ablation. Its strengths here: self-attention gives every part of the volume a view of every other part from the first layer, and MONAI's `TransformerBlock` already supports cross-attention, which made the anatomical tokens easy to add. Its limits: the ViT holds 116.7 M of the model's 122.3 M parameters, so it is data-hungry for 680 cases, and its 16³ patches are coarse (a 96³ input becomes only 6×6×6 tokens).

**Patch embedding.** The input is cut into $N=(96/16)^3=216$ non-overlapping 16³ patches. Each is flattened and linearly projected to a 768-dimensional token, and a learned position embedding is added:

$$z_0 = [\,x_1E;\ x_2E;\ \dots;\ x_NE\,] + E_{\text{pos}},\qquad E\in\mathbb R^{4096\times 768}.$$

**Transformer block (self-attention).** Each of the 12 blocks lets every token look at every other token, then applies a small per-token network (MLP):

$$z'_\ell = z_{\ell-1} + \mathrm{MSA}\big(\mathrm{LN}(z_{\ell-1})\big),\qquad z_\ell = z'_\ell + \mathrm{MLP}\big(\mathrm{LN}(z'_\ell)\big).$$

Multi-head self-attention uses 12 heads of size $d_h=768/12=64$. For one head, with queries $Q$, keys $K$, values $V$ all computed from the same tokens $z$:

$$\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\Big(\frac{QK^{\top}}{\sqrt{d_h}}\Big)V,\qquad Q=zW_Q,\ K=zW_K,\ V=zW_V .$$

Read it as: each token scores its similarity to every token (a $216\times216$ matrix), turns the scores into weights, and takes the weighted average of the others' values.

**Skip connections and decoder.** Features from three intermediate blocks (indices 3, 6, 9 of 0–11) and the final output are reshaped back into 3D grids and upsampled with transposed convolutions. A separate convolutional branch ("encoder1") processes the raw CT at full resolution. The decoder merges them from coarse to fine and a final convolution outputs 13 logits per voxel. Base feature width is 16.

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

*Atlas set note: the ten atlases were replaced on 2026-10-28 by a quality-selected set (slice thickness at most 3 mm, complete coverage, clean labels, typical anatomy). On 12 separate training targets the new set raised the atlas-only Dice from 0.380 to 0.450 (`experiments/registration_log.md`).*

On the 200 cases of the 150 / 50 chunk (`scripts/validate_priors.py`, `experiments/exp_04_pipeline/chunk_150_50_e100_validation.md`), the vote alone ($\arg\max_c P_c$), scored against the true label for evaluation only, gives a mean Dice of **0.431** on the 150 training cases and **0.424** on the 50 validation cases, with a mean organ-centroid error of about 12 mm; the validation invariants pass. The earlier figure of 0.392 on the 136 official validation scans was measured with the previous ten atlases and is superseded for the laptop runs; the whole-data figure for the new set is still to be produced. Failures come from scans that cover a short stretch of the body and from atlases that fall back to the coarse affine.

---

## 5. How the priors enter the network

The priors enter at three places, and the Swin model (`models/swin_atlas.py`) and the ViT model use the same three: anatomical tokens read by cross-attention, a full-resolution voxel path, and a fusion term on the logits. What differs is where the cross-attention sits and how many parameters the paths touch (table in 5.4).

### 5.1 Anatomical tokens: P, D and w

Each prior map is averaged inside every 16³ patch, so there is one summary per patch location, then turned into a token by a small MLP (`PriorTokens` in the Swin model). With $\Omega_n$ the voxels of patch $n$:

$$\bar P_n=\frac1{|\Omega_n|}\sum_{v\in\Omega_n}P(v)\in\mathbb R^{13},\qquad \bar D_n\in\mathbb R^{12},\qquad \bar w_n\in\mathbb R^{1}.$$

All three are concatenated and sent through the MLP,

$$A_n=W_2\,\mathrm{GELU}\big(W_1[\bar P_n;\bar D_n;\bar w_n]+b_1\big)+b_2,\qquad W_1\in\mathbb R^{1536\times26},\ W_2\in\mathbb R^{768\times1536}.$$

The result is $A\in\mathbb R^{216\times768}$, one anatomical token per 16³ patch, whatever the backbone: the token grid is 6×6×6 for both. (The epoch-101 ViT model used P and D only, $W_1\in\mathbb R^{1536\times25}$; when its weights are loaded into the ViT redesign the new column for $\bar w_n$ is zero-initialised. The Swin model is trained from scratch.)

**Where each prior enters the network:**

| Entry point | P | D | w |
|---|---|---|---|
| Anatomical tokens, read by cross-attention: after Swin stages 2, 3 and 4 (ViT: in all 12 blocks) (5.1, 5.2) | yes | yes | yes |
| Full-resolution input (5.3): Swin: the 27-channel input goes to the patch embedding and to `encoder1` (ViT: `encoder1` only) | yes | yes | yes |
| Output fusion on the logits (5.3) | yes (as $\log P$) | no | yes (as a mask) |
| Atlas-consistency loss (6.2) | yes (the target) | no | yes (the weight) |
| Prior dropout (4.5) | all three | | |

So w is in the main path and also in the other three places; D enters only through the tokens and the voxel path. In the P / P+D / P+D+w ablation (`new_model_plan.md`), the P-only arm removes D and w from the tokens and the voxel path, and uses $w\equiv1$ in the output fusion and the loss.

### 5.2 Cross-attention between image features and anatomical tokens

**Swin UNETR (base).** The Swin blocks themselves only attend inside local windows and do not see the tokens. The tokens are read by **adapters placed after stages 2, 3 and 4**, where the feature maps are 12³ (192 channels), 6³ (384) and 3³ (768). The feature map $h_s$ after stage $s$ is flattened into tokens, which act as the *queries*; the 216 anatomical tokens give the *keys and values*:

$$h'_s=h_s+g_s\cdot\mathrm{CrossAttn}\big(\mathrm{LN}(h_s),\,A\big),\qquad s\in\{2,3,4\},$$

$$\mathrm{CrossAttn}(u,A)=\mathrm{softmax}\!\Big(\frac{(uW_Q)(AW_K)^{\top}}{\sqrt{d_h}}\Big)(AW_V),$$

with 12, 24 and 24 heads at the three stages ($A$ is projected from 768 to the stage width). That is 1,728, 216 and 27 queries per volume. Read it as: each image location forms a query from what it currently knows and asks the 216 anatomical tokens "which of you is relevant to me?"; the answer is a weighted mix of anatomical information added to the feature. Image features are always the queries, so anatomy never overwrites them, it is only added. Stage 1 (24³, 13,824 queries) has no adapter: the memory cost is high on a 4 GiB card and the voxel path already carries the prior at that resolution.

The scalar $g_s$ is a learnable **trust gate**, one per adapter, initialised to 1. Training can lower it where the prior hurts.

**ViT-UNETR (comparison).** Each of the 12 blocks does three things in order: image tokens attend to each other, then to the anatomical tokens, then pass through the MLP, with one trust gate $g_\ell$ per block:

$$z'_\ell=z_{\ell-1}+\mathrm{MSA}\big(\mathrm{LN}(z_{\ell-1})\big),\qquad z''_\ell=z'_\ell+g_\ell\cdot\mathrm{CrossAttn}\big(\mathrm{LN}(z'_\ell),\,A\big),\qquad z_\ell=z''_\ell+\mathrm{MLP}\big(\mathrm{LN}(z''_\ell)\big).$$

One design point worth testing in both models: the anatomical tokens $A$ carry no positional embedding, so the model infers location from content only.

### 5.3 The voxel path and the output fusion

**Voxel-level path.** Pooling to 16³ patches (roughly 40–60 mm) discards fine detail, so the network also receives the priors directly at full resolution. The input has 27 channels:

$$\tilde X=\big[\tilde I,\ P,\ D,\ w\big]\in\mathbb R^{(1+13+12+1)\times 96^3},\qquad \tilde I=\frac{I+1000}{2000}.$$

In Swin UNETR this tensor feeds both the patch embedding (so the prior is visible to every Swin stage) and the full-resolution `encoder1` branch; in ViT-UNETR only the `encoder1` branch sees the priors.

**Output fusion.** A learnable weight $\gamma_c$ per class adds the log-prior directly to the logits:

$$\tilde z_c(v)=z_c(v)+\gamma_c\,w(v)\,\log\big(P_c(v)+\varepsilon\big),\qquad \varepsilon=10^{-3}.$$

This has a useful property: if the network's own logits were zero and $\gamma_c=1$, then $\mathrm{softmax}(\tilde z)\propto P^{\,w}$, which reproduces the atlas vote. So $\gamma$ lets the model start from the vote and learn corrections. **All current runs initialise $\gamma_c=1$** (`--gamma-init 1.0`): starting at the vote was the key fix, because with $\gamma=0$ the network collapsed to 0 Dice on several organs (0.314 against 0.571 at 48 training cases; 0.606 at 150 for ViT-UNETR). After 88 epochs the Swin model's mean $\gamma$ is 0.976 and its mean adapter gate 0.986, so neither moved far from 1.

### 5.4 The two backbones side by side

| | Swin UNETR (base) | ViT-UNETR (comparison) |
|---|---|---|
| Parameters | 67.1 M (feature size 48) | about 122 M |
| Encoder | 8 window-attention blocks, 4 stages, grids 48³ to 3³ | 12 global-attention blocks on a 6³ token grid |
| Prior tokens | 216 tokens, MLP 26 → 1536 → 768 | the same |
| Cross-attention | adapters after stages 2, 3, 4 (12³, 6³, 3³) | in all 12 blocks |
| Trust gates | 3, init 1 | 12, init 1 |
| Voxel path | 27-channel input to patch embedding and `encoder1` | `encoder1` only |
| Output fusion | $\gamma$ init 1.0 | the same |
| Consistency loss, prior dropout | KL weight 0.1 ramped over 10 epochs, $p=0.3$ | the same |
| Laptop speed (150 / 50 chunk) | about 1,200 s per epoch | about 295 s per epoch |

Differences that remain between the two: parameter count, and the voxel prior also entering Swin's patch embedding. The comparison is "each backbone with its best-effort version of the same prior integration", not an identical parameter budget.

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

### 6.3 CT physics (implemented in `training/physics_losses.py`; queued as one combined run on Swin, no result yet)

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
| $\mathcal L_{HU}$, $\mathcal L_{\text{edge}}$ | Implemented and tested (`training/physics_losses.py`, wired into `scripts/train_prototype_v2.py`); queued on 2026-10-30 as one combined run on Swin (`--hu-weight 0.03 --edge-weight 50`); no result yet. Kept only if validation Dice improves |
| Smoothness term from the research report | Not started |

---

## 7. How results are judged

- **Metric:** mean Dice over the 12 organs. The laptop ablation uses the 50 validation cases of the 150 / 50 chunk; the whole-data runs planned on the server use the official 136-scan validation set. Neither is used to tune anything, except that the best epoch is picked on the same cases, so the reported numbers are slightly optimistic.
- **Two reference points that any atlas-guided model must beat:** (1) the atlas vote alone (0.424 on the 50 validation cases), and (2) an image-only network with no priors (0.309 for ViT-UNETR; the Swin image-only run is queued). If the model does not beat both, the atlases have not helped.
- **Leakage guards:** the 10 atlases are never training or validation targets (checked against the 150 / 50 chunk); registration settings are tuned on training scans only.
- **One change at a time:** each addition (prior dropout, gates, voxel path, output fusion, consistency loss, each physics term, the backbone, and the P / P+D / P+D+w ablation) is a flag, run separately where possible; the order and decision rules are in `new_model_plan.md`.

### Snapshot of results so far (2026-10-30)

150 training / 50 validation cases, 100 epochs, seed 0, learning rate 2e-4 (untuned), `--gamma-init 1.0`:

| Measure | Mean Dice |
|---|---|
| Atlas vote alone | 0.424 |
| ViT-UNETR, image only (six organs at 0) | 0.309 |
| ViT-UNETR + prior | 0.606 |
| **Swin UNETR + prior** (epoch 88 of 100, best epoch 86) | **0.694** |
| Swin UNETR + prior + HU + edge losses | queued |
| Swin UNETR, image only | queued |

- The prior is worth about +0.30 Dice for ViT-UNETR, and turns six organs from 0 into 0.35 to 0.69. Swin beats ViT-UNETR on all 12 organs (+0.088 on the mean; the largest gains are on the gallbladder, stomach and aorta).
- With the atlas removed at test time, the Swin model still scores 0.578 (ViT-UNETR 0.302): its image pathway keeps learning under prior dropout. That is not the same as an image-only model, so the Swin image-only run is needed.
- Averaging the Swin and ViT-UNETR predictions gave no gain on the 50 cases (the models agree on 98.5 % of voxels), so an ensemble of the two is not worth a second forward pass.
- Caveats: one seed, 50 cases, an untuned learning rate, different parameter counts, and an unfinished Swin run. A gap below about 0.01 is not a result until the seed spread is measured. Earlier prototype numbers (0.65 with ground-truth priors, 0.168 at 24 training cases) are superseded.

---

## 8. Where things live

| What | File |
|---|---|
| Swin UNETR with the atlas priors (base model) | `models/swin_atlas.py`, `tests/test_swin_atlas.py` |
| ViT-UNETR (original) and its redesign, prior dropout, consistency loss | `models/segmentation_model.py`; `prototype.ipynb` (cell "v2-cell definitions") |
| Trainer for both backbones (`--backbone vit` or `swin`) | `scripts/train_prototype_v2.py` |
| Whole pipeline runners | `run.py` (ViT-UNETR), `run_swin.py` (Swin UNETR) |
| Run queue (laptop) | `scripts/queue_ablation.py` |
| Atlas library, landmarks, golden transform | `atlas/landmarks.py`, `atlas/golden_transform.py`, `atlas/reliable_features.py` |
| Registration: matching, RANSAC, TPS | `atlas/matching.py`, `atlas/ransac.py`, `atlas/tps.py`, `atlas/registration.py` |
| Label warping, P / D / w | `atlas/label_warping.py`, `data/transforms.py` (`AtlasPriorsd`) |
| Physics losses | `training/physics_losses.py`, `tests/test_physics_losses.py` |
| Per-organ Dice with names; two-model ensemble check | `scripts/show_organ_dice.py`, `scripts/ensemble_eval.py` |
| Server environment setup | `server/setup_server.sh` |
| Run commands | `RUN_GUIDE.md` (not yet updated for `run.py` and the Swin model) |
| Findings, bugs, results | `research_audit.md`, `experiments/registration_log.md` |
| Plan, experiment order, status | `new_model_plan.md` |

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
- **Swin Transformer:** a transformer that computes attention inside small local windows and shifts the windows between blocks; it builds a hierarchy of feature maps by merging neighbouring patches.
- **Window attention (W-MSA, SW-MSA):** self-attention restricted to a window of $7^3$ tokens; the shifted version (SW-MSA) moves the window grid by half a window so neighbouring windows exchange information.
- **Patch merging:** joins each 2×2×2 neighbourhood of tokens into one token with twice the channels, halving the grid in every direction.
- **Adapter (cross-attention adapter):** a small cross-attention layer inserted after a Swin stage, with a learnable gate, through which image features read the anatomical tokens.
- **Trust gate:** a learnable scalar (initialised to 1) that scales how strongly the anatomical tokens are added; training can lower it where the prior hurts.
- **Output fusion ($\gamma$):** the learnable per-class weight that adds $w\log P$ to the logits; initialised to 1 so the network starts at the atlas vote.
- **No-prior Dice:** the validation Dice of a prior-trained model re-scored with uninformative priors, logged during training; not the same as an image-only model trained without any prior.

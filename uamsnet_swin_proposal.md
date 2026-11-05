# UAMSNet ideas for a 3D Swin UNETR: reconstruction, adaptation and novelty check

Working note, 2026-11-03. Setting: AMOS22 CT, 12 organs + background, 3D Swin UNETR (MONAI `SwinUNETR`, feature size
48; `models/swin_atlas.py` builds on it).

**How the sources were checked.**
- **UAMSNet:** I read the paper PDF in full: Ju et al., *Pattern Recognition* 172 (2026) 112386,
  doi:10.1016/j.patcog.2025.112386.
- **UAMSNet code:** I downloaded and read the public code (github.com/HeyJGJu/UAMSNet: `main.py`, `MyModel.py`,
  `unet_parts.py`, `modules.py`).
- **A-DCN:** I read the paper's text: Chen et al., *Neurocomputing* 453 (2021) 853-864, doi:10.1016/j.neucom.2020.06.128.
- **Everything else:** abstracts and search summaries only, marked "(abstract only)" where it is used. The literature
  search was small and is **not exhaustive**.

Throughout, **[paper]** marks an equation copied from a source, **[code]** marks what the released code does, and
**[proposed]** marks my adaptation.

---

## 1. Executive summary

1. **The paper's loss cannot be reconstructed as printed.**
   - In Eq. (8) of UAMSNet, the false-positive term is identically zero. For a class that is absent from the ground
     truth, $G_{v,\tilde c}=0$ at every voxel, so both its Dice and its $G\log P$ term are zero.
   - The released code implements a different loss: per-class sigmoid, Dice + BCE, and a weight
     `(1/1+math.exp(50-epoch))`, which Python evaluates as $1+e^{50-e}$, not $1/(1+e^{-(e-50)})$.
   - The code also does not run as published (a syntax error, a NameError and a wrong call signature).
   - The λ ablation in Table 7 changes the results, so the trained loss had a working false-positive term, but neither
     the paper nor the code shows what that term was.
2. **MBA in the code is not what the paper text describes.**
   - The text says "average pooling". The code's `BoundaryAttention` uses `torch.mean(x, dim=1)`, which is a
     channel-wise mean, not a spatial pool.
   - So the code's "peak map" is each voxel's feature vector minus its own channel mean. It is not a spatial high-pass
     filter.
   - The released module is a CBAM-style spatial attention. The "large edge peak" argument in the paper describes the
     spatial-pooling version, which may not be what was trained.
3. **HRFE's deformable branch is A-DCN, without citation.** UAMSNet's Eq. (7) is A-DCN's Eq. (3) symbol for symbol,
   including the $(1-s_k)\Delta m_k$ modulation and the sentence describing $s_k$. UAMSNet cites only DCNv1 and DCNv2.
   HRFE is A-DCN in parallel with a dilated-convolution context block (ASPP/DenseASPP-like), fused by a 1×1 conv plus a
   residual. It is a combination of existing operations, not a new operator.
4. **Moving to a 3D Swin UNETR:**
   - **Bottleneck placement fails.** UAMSNet puts HRFE at the bottleneck. At 96³ input that is a **3×3×3** map, where
     dilations 2 to 4 and learned offsets mostly sample padding.
   - **The deep stages gain nothing.** From 1/8 resolution down, Swin windows already cover most or all of the map.
   - **Only the high-resolution skips leave room.** Inductive bias has room to help only at the 1/2 and 1/4 skip
     features, where windows are local and boundaries span several voxels.
5. **Recommended design (small).**
   - One gated, zero-initialised 3D A-DCN adapter on the 1/4-resolution skip (24³ × 96 channels).
   - Optionally, a 3D MBA on the 1/2 and 1/4 skips.
   - An absent-class penalty normalised by the predicted mass, $\log(1+\sum_v P_{c,v}/\kappa)$, which differs
     mathematically from what CE already does (Section 8). It only matters with **patch** training, because with
     whole-volume inputs almost every organ is present in every sample.
6. **Novelty is low for all three components** when ported on their own. Section 10 gives the ratings and the closest
   prior work. For this project the more defensible use is as controlled ablations next to the atlas prior, or an
   atlas-informed version of the absent-class penalty (Section 14E).

---

## 2. Mathematical reconstruction of UAMSNet

UAMSNet is **2D**: three consecutive axial slices as a 3-channel input, crops of 448×448, a U-Net with 4 encoder stages
and 4 decoder stages, MBA in every encoder and decoder block, HRFE at the bottleneck, Adam at learning rate 1e-4 with
poly decay, 100 epochs, λ = 50 [paper §3, §4.1, §4.3].

### 2.1 Backbone

$$F_{i+1}=E_i(F_i),\qquad G_i=D_i\big(\mathrm{Up}(G_{i+1}),\,F_i\big),\qquad i=1..4$$

- $E_i$ is a 2×2 max-pool followed by a convolution block, which in UAMSNet is MBA [code `EcMBA`].
- $D_i$ concatenates the upsampled decoder feature with the skip $F_i$, then applies a convolution block (MBA)
  [code `UpMBA`].
- The bottleneck is $F_5=\mathrm{HRFE}(\mathrm{MaxPool}(F_4))$ [code `BnDCN_Context`].

**Limits of plain convolution named by the paper:**
- the sampling grid is fixed, so it adapts neither to shape (e.g. the duodenum) nor to scale (adrenal versus liver);
- boundary detail is lost through downsampling;
- Dice/CE with fixed weights cannot adapt to which classes are present in each slice.

### 2.2 MBA

**[paper]** Eqs. (1) to (5), and the text after Eq. (5):

$$F_{in}=BN(Conv(ReLU(BN(Conv(F_{ori})))))\tag{1}$$
$$F_{ca}=CA(F_{in})\otimes F_{in}\tag{2}$$
$$F_{peak}=F_{ca}-avgpool(F_{ca})\tag{3}$$
$$M_{ba}=\sigma\big(Conv(concat(Conv(F_{peak}),\,F_{pool}))\big)\tag{4}$$
$$F_{spa}=M_{ba}\otimes F_{ca}\tag{5}$$
$$F_{MBA}=F_{ori}+F_{spa}\quad\text{(text, no equation number)}$$

- $CA$ is CBAM channel attention [ref 32 of the paper].
- $\otimes$ is called a "dot-product", meaning an elementwise product with broadcasting.
- The paper gives no pooling kernel, stride or padding. It does not say whether $M_{ba}$ has one channel or $C$
  channels.

**[code]** `BasicBlock` + `BoundaryAttention` in `unet_parts.py`, which resolves these gaps:

$$\bar F=\tfrac1C\textstyle\sum_{c}F_{ca}[c]\in\mathbb R^{1\times H\times W}\quad(\texttt{torch.mean(x, dim=1)})$$
$$F_{peak}=F_{ca}-\bar F\quad(\text{broadcast over channels})$$
$$M_{ba}=\sigma\big(W_2 * [\,W_1 * F_{peak}\,;\,\bar F\,]\big)\in(0,1)^{1\times H\times W},\qquad W_1:C\!\to\!1,\ W_2:2\!\to\!1,\ \text{both }3\times3$$
$$F_{out}=ReLU\big(F_{ori}+M_{ba}\otimes(CA(F_{in})\otimes F_{in})\big)$$

The code also differs from the text in two smaller ways:
- a 3×3 Conv-BN-ReLU sits before the block (`ConvWithMBA`);
- a final ReLU is applied after the residual.

**What each operation does.**

1. **Subtracting an average.**
   - **Spatial version (paper text).** With a stride-1 box filter of half-width $a$, a second-order Taylor expansion
     in each axis gives
     $$\mathrm{avg}_a(F)(x)\approx F(x)+\tfrac{a^2}{6}\nabla^2F(x),$$
     so
     $$F-\mathrm{avg}_a(F)\approx-\tfrac{a^2}{6}\nabla^2F.$$
     This is a Laplacian-type high-pass filter applied to *feature maps* (not to HU values). For a step edge, the
     Laplacian is **zero at the edge itself**, with lobes of opposite sign on either side. So $|F_{peak}|$ is largest
     next to a boundary, not on it. The learned $3\times3$ conv in (4) has to turn this into a band. It also passes most
     white noise (for $n$ pooled voxels, the noise variance of $F-\mathrm{avg}$ is $\sigma^2(1-1/n)$) and responds
     to internal texture, such as liver vessels, bowel gas and contrast inhomogeneity. It is a local-contrast map,
     **not an edge detector**.
   - **Channel version (code).** $F-\bar F$ centres each voxel's feature vector. It contains no spatial derivative at
     all. Spatial selectivity comes only from the learned $3\times3$ kernels $W_1, W_2$. This makes it a learned spatial
     attention in the CBAM family: CBAM concatenates the channel mean and channel max, then applies a 7×7 conv and a
     sigmoid. Here the max is replaced by $W_1*(F-\bar F)$.
2. **Why the result could relate to boundaries.** Features of different tissues differ, so feature-space contrast is
   high where tissues meet. Nothing in the loss forces the attention onto boundaries. Whether it lands there is an
   empirical question (see the test in 5.4).
3. **Multiplicative application.** $M\in(0,1)$ gates features. It can suppress a region but cannot create a signal,
   and its gradient is scaled by the feature magnitude: $\partial(MF)/\partial M=F$.
4. **Residual.** Since $F_{spa}=M\otimes F_{ca}$ can be near zero (when $M\approx0$), the skip
   $F_{ori}+\cdot$ keeps the identity path open for gradient flow, as in a ResNet block. Without it, a saturated
   sigmoid would cut the signal.
5. **How it differs from plain channel or spatial attention.** It is channel attention followed by spatial attention,
   like CBAM, but the spatial branch receives a centred/high-passed input instead of channel max/mean pooling. This is a
   modest variant, not a new kind of attention.

### 2.3 HRFE

**[paper]** Eq. (6):
$$F_{HRFE}=Conv\big(concat(Dil(F_{e4}),\,Def(F_{e4}))\big)+F_{e4}\tag{6}$$

- $Dil$: a 1×1 conv that compresses channels, then **four cascaded** dilated branches with rates 1, 2, 3, 4, then a 1×1
  conv that restores the input channel count.
- **[code]** `ContextBlock`: 1×1 conv, then **four parallel** 3×3 convolutions with dilations 1, 2, 3, 4 (padding equal
  to the dilation), LeakyReLU(0.2), concatenation, and a 1×1 conv.
  - The text says "cascaded"; the code is parallel.
  - The prompt that requested this note wrote $F_{dil}=\mathrm{Concat}(D_1..D_4)$, which matches the code, not the
    text.
- $Def$ (Eq. 7, **[paper]**, reproduced exactly):

$$F_{Def}(v)=\sum_{k=1}^{K}w_k\cdot F_{e4}\big(v+s_k\cdot v_k+\Delta v_k\big)\cdot\big((1-s_k)\cdot\Delta m_k\big)\tag{7}$$

  where:
  - $s_k\in[0,1]$ is the "adaptive dilation factor that contains the general distance information";
  - $v_k$ is the fixed grid, e.g. $\{-1,0,1\}^2$ for dilation 1;
  - $\Delta v_k$ is the learned offset;
  - $\Delta m_k\in[0,1]$ is the modulation.

  The paper's Fig. 2(b) shows three convolutions producing $2K$ offsets, $N=\sqrt K$ dilation factors and $K$
  modulation scalars. So $s$ has $N$ channels in the figure and $K$ in the notation $\{s_k\}_{k=1}^K$. The paper does
  not settle this.
- **[code]** `BnDCN_Context`:
  $$x\leftarrow\mathrm{MaxPool}(x);\quad \mathrm{Conv}_{1\times1}\big([\,\mathrm{DCN}^{+}_{\times2}(x)\,;\,\mathrm{Context}(x)\,]\big)+x$$
  - The deformable op `DeformConv2dplus` is imported from `model/DCNv2puls`, which is **not in the repository**, so
    Eq. (7) cannot be checked against code.
  - The modulation is switched on (`modulation=True`), and two deformable convs are stacked.

**How the generic deformable formula differs from the paper's.** Generic modulated DCN (DCNv2):
$$y(p_0)=\sum_k w_k\,x(p_0+p_k+\Delta p_k)\,m_k$$
UAMSNet/A-DCN differs in two places:
- **scaled grid:** $s_kp_k$, where $s_k\in[0,1]$ can only *shrink* the predefined grid; expansion comes from
  $\Delta v_k$, or from a predefined dilation $d$ in A-DCN;
- **coupled modulation:** the modulation is $(1-s_k)\Delta m_k$, so a tap moved outward by a larger $s_k$ is
  down-weighted. A-DCN states this intent: "the channel weight of far sampling point is depressed".

**How the branches complement each other.**
- $Dil$ gives a fixed bank of receptive fields: with 3×3 kernels at rates 1 to 4, extents 3, 5, 7, 9.
- $Def$ gives input-dependent sampling positions, which the paper intends for irregular shapes.
- The 1×1 fusion learns a per-channel mix of the two, and the residual keeps the identity path.

### 2.4 Loss

**[paper]** Eq. (8), reproduced exactly:
$$L_{seg}=1-\sum_{\hat c\in C_{gt}}\Big(\frac{2\sum_v G_{v\hat c}P_{v\hat c}}{\sum_v(G_{v\hat c}^2+P_{v\hat c}^2)}+\sum_v G_{v\hat c}\log P_{v\hat c}\Big)-\sum_{\tilde c\in(C_{pr}-C_{gt})}\beta\Big(\frac{2\sum_v G_{v\tilde c}P_{v\tilde c}}{\sum_v(G_{v\tilde c}^2+P_{v\tilde c}^2)}+\sum_v G_{v\tilde c}\log P_{v\tilde c}\Big)$$
$$\beta=\frac1{1+e^{-(epo-\lambda)}},\quad\lambda=50.$$

So $\beta\to0$ for $e\ll\lambda$, $\beta=0.5$ at $e=\lambda$, and $\beta\to1$ for $e\gg\lambda$. The intended schedule
is: learn the organs that are present first, and add false-positive suppression later, once predictions stop being
noise.

**What the printed equation actually computes:**
- **The false-positive term vanishes.** For $\tilde c\notin C_{gt}$, $G_{v\tilde c}=0$ for all $v$. So the Dice
  numerator is $0$, giving $0/\sum P^2=0$, and $\sum_vG\log P=0$. The false-positive term is **identically zero, with
  zero gradient**, for every β. As printed, $L_{seg}$ equals Dice + one-sided CE over present classes only.
- **The first term has no normalisation.** Dice is summed (not averaged) over classes, and the CE is summed over all
  $V$ pixels, not averaged. CE therefore outweighs Dice by a factor of about $V$, unless normalisation was used and not
  printed.
- **Present-only CE adds no false-positive signal.** Restricting CE to present classes leaves nothing for false
  positives *in the CE either*: under a softmax the CE gradient for a non-label class $c$ at voxel $v$ is $P_{vc}$,
  which already pushes false positives down, but that comes from the softmax coupling, not from the extra term.

**[code]** `MyLoss` in `main.py`:
- **Per-class sigmoid outputs.** `pred = torch.sigmoid(pred)`, so this is multi-label, not a softmax.
- **Per-class term.** `combinLoss = diceCoeffv2(p,g) + BCEWithLogitsLoss(p,g)`. A sigmoid is applied twice, because
  BCE-with-logits receives probabilities.
- **Sign error.** The loss is `1 - mean(class terms)`, so the **BCE is maximised**, the opposite sign to Eq. (8).
- **Which classes count as false positives.** Present classes come from `labelindex`. Prediction-only classes come
  from `getIndex` on the argmax mask, and these get weight `(1/1+math.exp(50-epoch))`, which equals $1+e^{50-e}$ (about
  $5\times10^{21}$ at epoch 0).
- **It does not run.** There is a full-width colon (SyntaxError), `super(newMyDiceLoss, …)` (NameError), and the active
  call `criterion2(pred, label)` is missing two arguments (TypeError).

**Conclusion.** The loss that produced Tables 6 and 7 is **not recoverable** from either the paper or the code. Any
"target-driven loss" we build is our own design, and it must not be described as UAMSNet's loss.

How the *intended* idea differs from plain losses:
- Dice + CE with fixed weights treats every class the same way in every sample.
- The intended UAMSNet loss changes the class set per sample (present versus predicted-only) and ramps the weight of
  the predicted-only part over training.

A side note on protocol: on Synapse, UAMSNet uses an 18/12 split with no separate test set, yet hyper-parameters
"are determined by the validation set" and λ is ablated on that same set (Table 7). The Synapse numbers are therefore
tuned on the cases they report.

---

## 3. Mathematical comparison

### 3.1 U-Net, DCN, A-DCN, UAMSNet

$$\text{conv:}\ y(p_0)=\sum_k w_k\,x(p_0+p_k)$$
$$\text{DCNv1:}\ y(p_0)=\sum_k w_k\,x(p_0+p_k+\Delta p_k)$$
$$\text{DCNv2:}\ \ldots\cdot\Delta m_k$$
$$\text{A-DCN / UAMSNet Def:}\ y(p_0)=\sum_k w_k\,x(p_0+s_kp_k+\Delta p_k)\,(1-s_k)\Delta m_k$$

- **Conv:** the grid is fixed. The receptive field is set by the kernel and the dilation, and it is the same at every
  location and for every input.
- **DCNv1:** each tap gets a learned offset, predicted per location from the input, so the field becomes
  input-dependent. Positions are fractional, so bilinear interpolation is needed.
- **DCNv2:** adds a per-tap amplitude $\Delta m_k$.
- **A-DCN / UAMSNet:** factor each tap's displacement into a shared radial scale $s_kp_k$ (the "large step") plus a
  residual $\Delta p_k$ (the "small step"), and tie the amplitude to the scale.

What UAMSNet changes relative to a U-Net:
- **Encoder and decoder blocks are modified.** Every block becomes MBA: the U-Net double conv plus channel attention,
  a spatial gate and a residual.
- **The bottleneck is replaced** by HRFE.
- **The loss is replaced.**
- Skip connections, pooling and upsampling are unchanged.

### 3.2 DCN, A-DCN and UAMSNet HRFE

| Mathematical aspect | Standard DCN (v1 / v2) | A-DCN [paper, Eq. 3-4] | UAMSNet HRFE [paper Eq. 6-7; code] |
|---|---|---|---|
| Regular sampling grid | $p_k$ (with dilation) | $d_{ij}\,p_k$, with a predefined dilation $d_{ij}$ | $v_k$; the paper's example is dilation 1 |
| Learned offsets | $\Delta p_k$ (2K channels) | $\Delta p_k$ (2K channels) | $\Delta v_k$ (2K channels) |
| Adaptive dilation | none | $s_k\in[0,1]$ via sigmoid, from $\sqrt K$ channels | the same $s_k\in[0,1]$; $N=\sqrt K$ channels in Fig. 2(b) |
| Modulation | v2: $\Delta m_k\in[0,1]$ | $(1-s_k)\Delta m_k$ | $(1-s_k)\Delta m_k$, identical to A-DCN |
| Multi-scale dilated branch | none | none in the operator (A-DCN tests deformable "organizations" inside DenseASPP) | parallel dilations 1 to 4 in the code; "cascaded" in the text |
| Feature fusion | none | none | concat, then 1×1 conv |
| Residual connection | depends on the host block | depends on the host block | $+F_{e4}$ |
| Intended role | geometric invariance (detection, segmentation) | stronger deformation for detection and segmentation (VOC, Cityscapes, COCO) | shape and size adaptation for organs at the bottleneck |

**Verdict.** HRFE's deformable operator *is* A-DCN. Its dilated branch is a standard ASPP/DenseASPP-style context
block. HRFE is an **application-specific combination** of existing mechanisms: two branches, a 1×1 fusion and a
residual. A-DCN itself already studies deformable convolution inside DenseASPP.

### 3.3 Novelty decomposition of UAMSNet

| Component | (1) From U-Net | (2) From DCN / A-DCN | (3) Combination | (4) Possibly task-specific |
|---|---|---|---|---|
| MBA | double conv + residual | – | CBAM channel attention + a spatial gate on centred features | the "peak map" input to the spatial gate (small variant of CBAM) |
| HRFE | bottleneck position | the whole Def branch (A-DCN, uncited) | A-DCN ‖ dilated context block, fused | none beyond the combination |
| Loss | Dice + CE | – | Dice + CE over the classes in each sample + a ramped weight | the idea of a ramped false-positive term over prediction-only classes. **The printed formula does not implement it.** Close prior art: exclusion and marginal losses for partially labelled multi-organ data (Shi et al., MedIA 2021, *to be read*) |

---

## 4. Why the modules cannot be transferred directly to a 3D Swin UNETR

The MONAI `SwinUNETR` at 96³ input with feature size 48 has these features:

| Tensor | Produced by | Size | Channels | Physical size if trained on 1 mm patches |
|---|---|---|---|---|
| enc0 | `encoder1`: residual conv block on the input | 96³ | 48 | 1 mm voxel |
| hs0 → enc1 | patch embed (2³), then `encoder2` | 48³ | 48 | 2 mm |
| hs1 → enc2 | Swin stage 1, then `encoder3` | 24³ | 96 | 4 mm |
| hs2 → enc3 | Swin stage 2, then `encoder4` | 12³ | 192 | 8 mm |
| hs3 | Swin stage 3 | 6³ | 384 | 16 mm |
| hs4 → dec4 | Swin stage 4, then `encoder10` | **3³** | 768 | 32 mm |

The decoders (`decoder5` … `decoder1`) upsample by transposed convolution and fuse each skip through residual conv
blocks.

Inside each Swin stage, the window attention is
$$\mathrm{Attn}=\mathrm{Softmax}\big(QK^\top/\sqrt d+B\big)V,$$
with window 7³ and a shifted window in every second block. MONAI shrinks the window to the feature size when the map
is smaller. With windows of 7 voxels:
- at 24³ the window spans **28 mm**;
- at 12³ it covers 7 of 12 voxels per axis (56 mm), and the shift links neighbouring windows;
- at 6³ and 3³ the attention is **global**.

Consequences:
- **No room at the bottleneck.** UAMSNet's choice is the 3³ map. A 3³ kernel at dilation $r\ge2$ (padding $r$)
  reaches outside the 3-voxel map for every non-centre tap. Learned offsets have nowhere to move. Global attention
  already relates every position to every other.
- **Dilated convolutions repeat what the deep stages already do.** Their receptive-field bank (3 to 9 voxels) adds
  little at 1/8 and below, where windows already span 7 to 12 voxels or the whole map.
- **Room at the high-resolution skips.** At 1/2 and 1/4, windows are local (14 to 28 mm), and boundaries and thin
  organs (adrenal limbs, esophageal wall, the gallbladder neck) span several voxels. Attention is a softmax-weighted
  convex combination, and has been argued to act as a low-pass filter (Park & Kim, ICLR 2022, *abstract only*). The
  skip encoders (`encoder2`, `encoder3`) are already residual conv blocks, so some local high-frequency processing
  exists there.
- **Cost of 3D deformable convolution.** torchvision has no 3D deformable conv (`deform_conv2d` is 2D only). It must be
  written with `grid_sample` (trilinear), which stores $K=27$ sampled copies of the feature map. At 24³×96 channels:
  $27\cdot96\cdot13{,}824\approx3.6\times10^7$ floats, about 143 MB in fp32 per sample, before the backward pass. At
  48³×48 channels it is about 573 MB. On the 4 GB RTX 3050, only the 1/4 level is realistic.
- **Anisotropy.** Our pipeline resamples to 1 mm isotropic, so voxel offsets are isotropic in millimetres. The through-
  plane detail is still interpolated from slices of up to about 5 mm. Isotropic sampling is geometrically right, but
  the z-axis carries less real information. An anisotropic variant is in 6.2.
- **Our current pipeline differs.** It resizes the whole cropped volume to 96³ (about 3 to 5 mm per voxel) instead of
  taking 1 mm patches. At that resolution an adrenal gland is 1 to 3 voxels thick, and no module will recover detail
  that the resampling removed. **The patch-based pipeline is a precondition** for everything below, and it is also the
  setting in which the absent-class loss has any effect (Section 8).

---

## 5. Proposed 3D MBA

### 5.1 Definition [proposed]

For $F\in\mathbb R^{B\times C\times D\times H\times W}$:

$$F_{ca}=\sigma\big(\mathrm{MLP}(\mathrm{GAP}(F))+\mathrm{MLP}(\mathrm{GMP}(F))\big)\otimes F\qquad\text{(CBAM channel attention, as in the code)}$$

Two choices for the "peak" (both will be ablated):

$$\text{MBA-S (spatial, paper text):}\quad F_{peak}=F_{ca}-\mathrm{AvgPool3d}_{k_z\times k\times k,\ \mathrm{stride}\ 1}(F_{ca})$$
$$\text{MBA-C (channel, code):}\quad F_{peak}=F_{ca}-\tfrac1C\textstyle\sum_cF_{ca}[c]$$

$$M_{ba}=\sigma\Big(W_2*\big[\,\mathrm{DW}(F_{peak})\cdot u\,;\ \bar F_{ca}\,\big]\Big)\in(0,1)^{B\times1\times D\times H\times W}$$
$$F^{3D}_{MBA}=F+\alpha\odot(M_{ba}\otimes F_{ca}),\qquad\alpha\in\mathbb R^{C},\ \alpha_{init}=0$$

- **Weights.** $\mathrm{DW}$ is a depthwise 3×3×3 conv. $u\in\mathbb R^{C}$ is a learned 1×1×1 projection to one
  channel, replacing the code's full $C\to1$ 3×3 conv. $W_2$ is a 3×3×3 conv from 2 channels to 1. $\bar F_{ca}$ is the
  channel mean.
- **Parameters:** about $27C+C+54$ for the gate, plus $2C^2/16$ for the channel attention. For $C=96$ that is about
  3.8k. Activation memory is one extra $C$-channel map.
- **Pooling kernel:** $k=3$ and $k_z=\max(1,\mathrm{round}(k\,s_{xy}/s_z))$, which equals $k$ for isotropic data.
- **Zero-initialised α** (a LayerScale/ReZero-style gate) makes the network start as the baseline Swin UNETR, which
  makes the ablation clean. For the gradient at $\alpha=0$ see 7.2.

### 5.2 Placement

- **On the skips enc1 (48³) and enc2 (24³) only**, after `encoder2` and `encoder3`, before they enter the decoder.
- **Why there:**
  - these features carry boundary detail to the full-resolution decoder;
  - windows are local at these resolutions;
  - the cost is linear in $DHW$ and small.
- **Not at enc3 and below:** voxels are 8 mm or more, the attention is near-global, and a boundary is under one voxel
  wide.
- **Not inside the Swin blocks:** that would mean changing pretrained (or carefully initialised) attention blocks, the
  extra mixing duplicates the MLP, and it complicates the ablation.
- **Decoder-side option** (on `decoder3` or `decoder2` outputs): the decoder features are already fused and
  task-shaped, so a boundary gate there acts on semantic contrast, not on the encoder's appearance contrast. This is
  worth one ablation run (E1b), but encoder skips come first: they are where appearance edges exist, and where the
  paper puts most MBA blocks.

### 5.3 Does the peak map capture boundaries?

- MBA-S is a learned gate fed by a feature-Laplacian. It fires on any local contrast: boundaries, vessels, noise.
- MBA-C has no spatial derivative; any boundary preference must be learned by $W_1,W_2$.
- Neither is constrained to boundaries.

### 5.4 Test of the "boundary" claim

- Let $\partial Y$ be the set of voxels within 2 voxels of any organ surface.
- Compute the AUROC of $M_{ba}$ for separating $\partial Y$ from organ interiors, and from the background, on the
  validation cases.
- Compare **MBA-S**, **MBA-C**, **plain CBAM spatial attention** (mean/max, 7³) and a **shuffled control** (the same
  gate with $F_{peak}$ replaced by $F_{ca}$).
- If AUROC is about 0.5, or no better than CBAM, the "boundary" label is unjustified even if Dice improves.
- Also report **NSD (normalised surface Dice) at 1 to 2 mm** and HD95. A boundary module should move those more than
  it moves Dice.

---

## 6. Proposed 3D HRFE

### 6.1 3D A-DCN operator [proposed, a direct 3D extension of A-DCN Eq. 3]

$$y(p_0)=\sum_{k=1}^{27}w_k\;x\big(p_0+s_k\,(d\odot p_k)+\Delta p_k\big)\,(1-s_k)\,\Delta m_k$$

- **Grid:** $p_k\in\{-1,0,1\}^3$, with $p_k=(d_k,h_k,w_k)$ and $\Delta p_k=(\Delta d_k,\Delta h_k,\Delta w_k)$.
- **Predefined per-axis dilation:** $d=(d_z,d_y,d_x)$. Isotropic data uses $d=(2,2,2)$, so $s_k\in[0,1]$ can scale
  each tap between the centre and dilation 2 (8 mm reach at 1/4 resolution). Anisotropic data uses $d_z<d_{xy}$.
- **Predictor:** one 3³ conv from the input predicts $3K$ offsets (unbounded, initialised to 0), $K$ modulation values
  (sigmoid, bias initialised so $\Delta m\approx1$), and $K$ dilation factors (sigmoid, initialised so $s\approx1$).
  That is **5K = 135 channels**.
  - **Per-tap or per-row $s$.** I use per-tap $s_k$ (K channels), which matches the notation $\{s_k\}_{k=1}^K$. A-DCN
    and UAMSNet's figure use $\sqrt K$ channels; that variant can be an ablation.
- **Sampling:** trilinear `grid_sample`, with `align_corners` consistent with the offset units. Write a unit test: zero
  offsets with $s=1$ and $\Delta m=1$ must reproduce `Conv3d(dilation=d)` up to the constant factor $(1-s_k)$.
  - **Caution: the coupled modulation conflicts with "starts as a dilated conv".** A-DCN's $(1-s_k)$ factor is 0 at
    $s=1$.
  - **Option 1:** keep A-DCN's exact form and initialise $s\approx0.5$.
  - **Option 2:** use $(1-\eta s_k)$ with $\eta\in[0,1]$ learned, initialised at 0.
  - I would keep A-DCN's exact form for faithfulness and test option 2 as a variant.

### 6.2 Options

**Option A: bottleneck (3³ × 768).**
- Compatibility: degenerate. Dilation 2 to 4 and offsets mostly sample padding.
- Cost: tiny.
- Receptive field: the map is already covered by global attention.
- Duplicates Swin attention: yes, completely.
- Useful inductive bias: no.
- **Verdict:** reject.

**Option B: after Swin stage 2 or 3 (12³ or 6³).**
- Compatibility: possible.
- Cost: low.
- Receptive field: window 7 versus a dilated extent of 9, so the scales are similar.
- Duplicates Swin attention: mostly.
- Useful inductive bias: weak.
- **Verdict:** at most an ablation.

**Option C: decoder features (24³ to 96³).**
- Compatibility: fine.
- Cost: high at 48³ and above (activation copies of 573 MB or more).
- Receptive field: local.
- Duplicates Swin attention: no; the decoder is convolutional.
- Useful inductive bias: sampling decoder features at deformed positions, which is plausible for thin organs.
- **Verdict:** only at 24³.

**Option D: gated conv adapter on the skip enc2 (24³ × 96).**
- Compatibility: clean, since the input is already a dense 5D tensor.
- Cost: about 0.45M parameters (or about 0.1M with a 1×1 reduction to 48 channels), and about 143 MB of activation
  copies per sample.
- Receptive field: input-dependent, crosses window borders, and has sub-voxel precision.
- Duplicates Swin attention: partly. Shifted windows also cross borders, but at fixed positions and as soft averages.
- Useful inductive bias: yes. Geometric sampling along curved or thin structures; this is the DSCNet argument.
- **Verdict: recommended.**

**Recommended primary design:**
- **Option D at enc2.** Optionally add enc1 later if memory allows (the server, not the laptop).
- **Drop the dilated branch from the primary design.** At 1/4 resolution, the bank of 3 to 9 voxels (12 to 36 mm)
  overlaps the 28 mm Swin window and the receptive fields that the skip encoder and decoder build anyway. Keep it as
  ablation E2c ("dilated only") and E2d ("full HRFE": dilated + deformable).

**Anisotropy.** If the data is resampled to 1 mm (our choice), use isotropic $d$. If a future pipeline keeps the native
z-spacing $s_z>s_{xy}$, set $d_z=\max(1,\mathrm{round}(d_{xy}\,s_{xy}/s_z))$ and scale the offset units per axis, so a
unit offset means the same millimetres on every axis.

---

## 7. Fusing transformer and convolutional features

### 7.1 Comparison of fusion strategies

**Concat + 1×1 conv (UAMSNet).**
- For: expressive.
- Against: at initialisation it changes the baseline function; it adds $2C\cdot C$ parameters; the effect of the
  adapter cannot be separated from the extra fusion capacity.

**Residual addition $F+H(F)$.**
- For: simple; identity path.
- Against: $H$ starts at random scale and perturbs the baseline.

**Gated $F+G\odot H$, with $G=\sigma(\mathrm{Conv}[F,H])$.**
- For: gates per voxel.
- Against: $\sigma(0)=0.5$, so it does not start at the baseline.

**Cross-attention.**
- For: global, content-based.
- Against: 13,824 tokens at 24³; attention costs $O(N^2)$, about $1.9\times10^8$ pairs; it duplicates Swin.

**Adapter with zero-initialised scale (ReZero/LayerScale).**
- For: starts exactly at the baseline, learns how much to use, cheap.
- Against: the gate has a slow start (7.2).

### 7.2 Selected fusion

The selected fusion is a gated residual with a zero-initialised channel scale:
$$G=\sigma\big(W_g*[F,H(F)]\big),\qquad F_{fused}=F+\alpha\odot\big(G\odot H(F)\big),\qquad\alpha\in\mathbb R^{C},\ \alpha_{init}=0$$

- **Gradient at $\alpha=0$.** $\partial L/\partial\alpha_c=\sum_v(\partial L/\partial F_{fused})_{c,v}\,(G\odot H)_{c,v}$,
  which is non-zero. Meanwhile $\partial L/\partial\theta_H=0$ for one step, and both move after that, as in ReZero.
- **It mirrors our atlas design**, which also starts at a known function (γ = 1 makes the untrained net equal the atlas
  vote). Then any gain is attributable to the adapter.
- **Watch the learned α.** In our atlas model the gates stayed near their initial values. A final $|\alpha|\approx0$ is
  a direct sign that the adapter is not used.

---

## 8. 3D absent-class loss

### 8.1 What changes from 2D slices to 3D

- **2D slices.** In UAMSNet, many slices lack many organs, so $C_{pr}\setminus C_{gt}$ is often non-empty.
- **3D whole volumes (our current pipeline).** AMOS scans that cover the abdomen contain nearly all 12 organs
  (exceptions: gallbladder after cholecystectomy, single kidney, partial field of view). The absent set is almost
  always empty, so any absent-class term is inactive and the loss reduces to Dice + CE.
- **3D patches (96³ at 1 mm).** The absent set is often non-empty: an adrenal is absent from most random patches. Note
  that MONAI's foreground-biased sampling raises organ presence.

**First measurement, before building anything:** with the current model and the patch sampler, count per patch the
classes with $\ge k$ predicted voxels and 0 ground-truth voxels (k = 10, 100), and their predicted volume. This answers
Q7 (Section 9.2) with data.

### 8.2 What Dice + CE already do for absent classes

Let $Y$ be one-hot, $P=\mathrm{softmax}(Z)$, $I_c=\mathbb 1[\sum_vY_{c,v}>0]$, $C^+=\{c\ge1:I_c=1\}$,
$C^-=\{c\ge1:I_c=0\}$, and $|\Omega|$ the number of voxels in the patch.

- **CE.**
  $$L_{CE}=-\tfrac1{|\Omega|}\sum_v\log P_{y_v,v}$$
  Its gradient with respect to $Z_{c,v}$ is $(P_{c,v}-Y_{c,v})/|\Omega|$. For $c\in C^-$ that is $P_{c,v}/|\Omega|$
  at **every** voxel. So CE already penalises absent-class mass, with a per-voxel weight that does not depend on how
  large or small the spurious blob is.
- **Soft Dice with smoothing ε** (MONAI's default is ε ≈ 1e-5), for $c\in C^-$:
  $$L_c=1-\frac{\varepsilon}{\sum_vP_{c,v}+\varepsilon}$$
  $$\frac{\partial L_c}{\partial P_{c,v}}=\frac{\varepsilon}{(\sum_vP_{c,v}+\varepsilon)^2}$$
  For a blob of 10 voxels this is about $10^{-7}$: effectively **zero gradient**. Absent classes add a constant of
  about 1 to the mean Dice loss and dilute the present classes.

### 8.3 Candidate penalties for absent class $c\in C^-$

**(a) Mean probability** (the form suggested in the request):
$$L^{(a)}_c=\tfrac1{|\Omega|}\sum_vP_{c,v}$$
$$\frac{\partial L^{(a)}_c}{\partial Z_{j,v}}=\tfrac1{|\Omega|}P_{c,v}(\delta_{cj}-P_{j,v})$$
The gradient vanishes when $P_{c,v}\to1$, so **confident false positives are barely corrected**. Reject.

**(b) Log complement:**
$$L^{(b)}_c=-\tfrac1{|\Omega|}\sum_v\log(1-P_{c,v})$$
$$\frac{\partial L^{(b)}_c}{\partial Z_{c,v}}=P_{c,v}/|\Omega|$$
This is **the same gradient as CE** for that class. It amounts to multiplying CE's weight on absent classes by
$(1+\beta)$, which is a reweighting, not a new mechanism. Use it at most as a control.

**(c) Mass-normalised penalty [proposed]:**
$$L^{(c)}_c=\log\Big(1+\tfrac{1}{\kappa}\sum_vP_{c,v}\Big)$$
$$\frac{\partial L^{(c)}_c}{\partial P_{c,v}}=\frac{1}{\kappa+\sum_uP_{c,u}}$$
- **Small blobs.** For a small false-positive blob ($\sum P\ll\kappa$) the gradient is about $1/\kappa$, much larger
  than CE's $1/|\Omega|$ (with $|\Omega|=884{,}736$ for 96³).
  - Small spurious islands are exactly what inflates HD95, and CE barely sees them.
- **Large blobs.** For large false-positive mass the gradient decays like $1/\sum P$, so one huge error cannot
  dominate the batch (stability).
- **Choice of κ.** Set κ near the smallest organ volume of interest, e.g. κ = 50 to 100 voxels (an adrenal at 1 mm is
  a few thousand voxels; κ only needs to be well above noise).
- **Relation to Dice.** This is the absent-class counterpart of Dice's normalisation by predicted mass.

**Proposed loss:**
$$L=\underbrace{\tfrac1{|C^+\cup\{0\}|}\sum_{c\in C^+\cup\{0\}}\big(1-\mathrm{Dice}_c\big)+L_{CE}}_{\text{Dice over present classes + full softmax CE}}\;+\;\beta(e)\,\tfrac1{|C^-|}\sum_{c\in C^-}L^{(c)}_c$$
$$\beta(e)=\frac{\beta_{max}}{1+\exp(-(e-\lambda)/\tau)}$$

**Choices:**
- **Dice over present classes only** removes the constant, gradient-free terms of absent classes (a rescaling, not a
  new signal).
- **CE stays over all classes**, so the softmax coupling is unchanged.
- **Schedule:** $\lambda$ is set as a fraction of training (e.g. 30 to 50 % of epochs), and $\tau$ is a temperature;
  $\tau=1$ in epochs reproduces UAMSNet's sharp switch.
- **The predicted set $C_{pr}$ is not used.**
  - UAMSNet gates the term by $C_{pr}$, computed from the argmax, which has no gradient. Penalising all of $C^-$ is
    smoother: a class with zero predicted mass costs nothing anyway, since $L^{(c)}=0$.
  - So the $C_{pr}$ gate is unnecessary in this formulation.

### 8.4 Other losses

- **Focal loss** acts on hard voxels and is orthogonal to absent-class presence.
- **Boundary loss or generalised Dice** reweights present classes.
- **Class weights** are not used at present.

None of these is justified by this analysis on its own. Add at most one, and only if E4 shows the absent-class term
helps and a per-organ analysis points to a specific weakness. **Do not stack losses.**

### 8.5 Risks

- **Missing labels.** If an organ is present but not labelled (partial annotation, or merged datasets with different
  label sets such as our `s` source), the penalty actively suppresses real anatomy. AMOS22 annotates all 15 organs
  when present, as far as I know, but this should be verified with a per-case label census.
- **Truncated organs at patch borders.** A few voxels are labelled, so the class is in $C^+$ and gets no penalty.
  This is safe.
- **Overlap with the atlas prior.** Our output fusion $z_c+\gamma_cw\log(P^{atlas}_c+10^{-3})$ already lowers the
  logits by up to about 6.9 wherever the atlas says an organ is confidently absent. That is a spatial false-positive
  suppressor. In the atlas model, the loss's added value may be small. Test it in both the image-only and the prior
  models.

---

## 9. Complete architecture and critical questions

### 9.1 Stage-by-stage changes

Data flow:
$X$ (96³ patch, 1 mm) → `encoder1` → patch embed → Swin stages 1 to 4 → skip encoders (**MBA on enc1 and enc2**,
**A-DCN adapter on enc2**) → decoders → logits, trained with Dice + CE + β·absent-class term.

| Stage | Original Swin UNETR | Proposed change | Operation | Reason |
|---|---|---|---|---|
| enc0 (96³) | `encoder1` residual conv | none | – | already convolutional at full resolution |
| hs0 → enc1 (48³ × 48) | patch embed + `encoder2` | 3D MBA (optional, E1) | $F+\alpha\odot(M_{ba}\otimes F_{ca})$ | boundary detail; local windows |
| hs1 → enc2 (24³ × 96) | Swin stage 1 + `encoder3` | 3D MBA (E1) + **gated 3D A-DCN adapter (E2)** | $F+\alpha\odot(G\odot\mathrm{ADCN}(F))$ | geometric sampling at the finest affordable scale |
| hs2 → enc3 (12³) | Swin stage 2 + `encoder4` | none | – | window covers about 60 % of the map |
| hs3 (6³), hs4 (3³) | Swin stages 3 and 4, `encoder10` | none | – | attention is global; a 3³ map is degenerate |
| decoders | `decoder5`…`decoder1` | none (E1b: MBA on the `decoder3` output) | – | keep the ablation clean |
| loss | Dice + CE | present-class Dice + CE + $\beta(e)\,L^{(c)}$ | Section 8 | small spurious islands |

Unchanged: patch embedding, all Swin blocks, window and shift settings, skip encoders' internals, decoders, output
head, and (in the atlas model) all three prior paths.

### 9.2 Critical questions

1. **Does MBA add to Swin window attention?**
   - Possibly, but only at 1/2 and 1/4 resolution, as a high-pass or contrast gate that attention, a smoothing
     operator, does not provide.
   - The skip encoders already include convs.
   - Expect a small effect, and test it with NSD and the AUROC check in 5.4.
2. **Does the deformable branch add to shifted windows?**
   - It adds input-dependent sub-voxel sampling positions that are not tied to the window grid.
   - It is the most plausible of the three ideas, and only at 24³ or finer.
3. **Is the dilated branch redundant?**
   - Largely yes, at every level. At the fine levels it overlaps the window size and the skip-encoder convs; at the
     deep levels attention is global.
4. **HRFE at the bottleneck only, or at every stage?**
   - Neither. The bottleneck is degenerate at 3³, and every stage multiplies the cost.
   - Use one fine-scale skip.
5. **Encoder or decoder features for MBA?**
   - Encoder skips first, because appearance edges exist there. Run the decoder variant as one ablation.
6. **Does the absent-class loss fix a real Dice + CE weakness?**
   - **Partly.** CE already penalises absent-class mass, and Dice gives effectively none. The real gap is
     *size-normalisation for small spurious islands*, which $L^{(c)}$ targets.
   - The printed UAMSNet loss does not address it at all.
7. **Are absent-class false positives common in AMOS22?**
   - Unknown. They are rare with whole-volume inputs and probably frequent with patches. Measure first (8.1).
8. **Can the loss destabilise training?**
   - (a) and (b) cannot blow up. (c) is bounded in gradient by $1/\kappa$ per voxel.
   - The real risk is missing labels (8.5).
   - Ramping β and capping it at $\beta_{max}\le1$ limits early damage.
9. **Cost of 3D deformable convolution.**
   - Parameters: about 0.45M at 24³ × 96.
   - Memory: about 143 MB of activation copies per sample forward, about 2 to 3 times that with the backward pass.
   - Compute: roughly 5 to 10 GFLOP-equivalent with interpolation.
   - A grid_sample implementation is slower than a fused kernel; expect +15 to 30 % step time (an estimate, to be
     measured).
10. **What is novel?**

| Component | New operation? | New integration? | New application? | Existing method in a new setting? |
|---|---|---|---|---|
| 3D MBA | no (CBAM variant) | a gate on Swin UNETR skips: minor | no | **yes** |
| 3D A-DCN adapter | no (A-DCN in 3D) | a zero-initialised gated adapter on a Swin skip: minor | thin abdominal organs: weak | **yes** |
| Absent-class loss $L^{(c)}$ | a mass-normalised absent-class penalty: **possibly, unchecked** (exclusion losses exist) | – | – | the scheduled β is borrowed |

Using a component with Swin UNETR does not make it novel.

---

## 10. Literature and novelty assessment (partial search)

| Work | Operation | Where inserted | How it differs from the proposal | Does it weaken the novelty claim? |
|---|---|---|---|---|
| UAMSNet (Ju et al., PR 2026) | MBA, HRFE, scheduled false-positive loss | 2D U-Net blocks, bottleneck | 2D; loss not reproducible | it is the source: an adaptation cannot claim its ideas |
| A-DCN (Chen et al., Neurocomputing 2021) | $s_kp_k$ + $(1-s_k)\Delta m_k$ deformable conv; deformable "organisations" inside DenseASPP | backbones; DenseASPP | 2D; detection and segmentation of natural images | **yes**: the deformable operator is A-DCN's |
| DCNv1 / DCNv2 (Dai 2017; Zhu 2019) | offsets; modulation | backbones | – | foundational |
| CBAM (Woo et al., ECCV 2018) | channel + spatial attention | CNN blocks | spatial input is mean/max, not centred | **yes**: MBA as coded is a CBAM variant |
| DSCNet (Qi et al., ICCV 2023) | snake (deformable) convolution for tubular structures | – | 2D/3D tubular | weakens a "deformable for thin organs" claim |
| BA-Net (2022, arXiv 2208.13774, *abstract only*) | a boundary decoder whose maps act as attention; 89.29 % Dice on AMOS CT | encoder + two decoders | CNN, multi-task | **yes** for boundary attention on AMOS |
| 3D boundary-constrained networks (arXiv 2210.04285, *abstract only*) | boundary prediction as an auxiliary task; up to +3.5 % relative Dice on BTCV | 3D encoder-decoders | multi-task, not attention | partly |
| Swin UNETR (Hatamizadeh et al. / Tang et al. 2022) | 3D Swin encoder + conv decoder | – | baseline | – |
| Park & Kim, "How do vision transformers work?" (ICLR 2022, *abstract only*) | attention as a low-pass filter | – | an argument, not a method | supports a high-pass gate |
| Shi et al., marginal and exclusion loss (MedIA 2021, *from memory, to verify*) | penalises predicting organs known to be absent | loss | partial-label setting | **likely yes** for any "absent-class loss" claim |
| nnU-Net Revisited (Isensee et al. 2024, arXiv 2404.09556, *abstract only*) | strict benchmark; CNN U-Nets (ResEnc) match or beat new architectures | – | – | raises the bar: gains must hold against nnU-Net ResEnc |

Not found, in a shallow search: a 3D A-DCN gated adapter on Swin UNETR skips, or a mass-normalised absent-class
penalty in this form. Absence from a few web searches is weak evidence. Before any claim, search PubMed, arXiv and the
MICCAI, MIDL and ISBI proceedings for: "deformable convolution Swin UNETR", "3D deformable convolution abdominal",
"boundary attention transformer AMOS", "exclusion loss", "false positive penalty absent organ", "presence-aware loss".

---

## 11. Experiments and ablations

All runs use the same fixed 5 folds on AMOS22 (atlas cases excluded), the same 96³ 1 mm patch sampler, the same number
of iterations, learning rate and augmentation, sliding-window inference with 0.5 overlap, and **at least 2 seeds** for
E0 and the best model.

| Exp | Architecture | Loss | Purpose |
|---|---|---|---|
| E0 | Swin UNETR | Dice + CE | baseline |
| E0p | Swin UNETR + a **parameter-matched plain conv adapter** at enc2 | Dice + CE | capacity control: separates "deformable" from "more parameters" |
| E1 | + 3D MBA-S on enc1 and enc2 | Dice + CE | isolate MBA |
| E1c | + 3D MBA-C (code version) | Dice + CE | paper text versus code |
| E1b | MBA on the `decoder3` output instead | Dice + CE | encoder versus decoder placement |
| E2 | + gated 3D A-DCN adapter on enc2 | Dice + CE | isolate the deformable branch |
| E2a | as E2 with $s\equiv1$ fixed (DCNv2 3D) | Dice + CE | is adaptive dilation needed? |
| E2c | dilated branch only (rates 1 to 4) at enc2 | Dice + CE | is the dilated bank redundant? |
| E2d | full 3D HRFE (dilated ‖ deformable, concat fusion) at enc2 | Dice + CE | UAMSNet-faithful fusion |
| E2e | full 3D HRFE at the bottleneck (3³) | Dice + CE | test the claim that the bottleneck is degenerate |
| E3 | E1 + E2 | Dice + CE | combination |
| E4 | Swin UNETR | + $\beta L^{(c)}$ | isolate the loss |
| E4b | Swin UNETR | + $\beta L^{(b)}$ (CE reweighting) | is the normalisation what matters? |
| E4c | Swin UNETR | + constant β | is the schedule needed? |
| E4λ | Swin UNETR | λ ∈ {0.2, 0.4, 0.6} × epochs | schedule sensitivity |
| E5 | E3 | + $\beta L^{(c)}$ | full model |
| E6 | E5 with the atlas prior on | all | does it add anything on top of the prior? |

Further variants if needed: gated fusion versus concat (E2 versus E2d); isotropic versus anisotropic $d$ (only if a
native-spacing pipeline is used).

**Metrics:**
- **Accuracy:** mean and per-organ Dice; HD95; ASD; NSD at 1 and 2 mm.
- **Error types:** sensitivity; precision; false-positive volume per organ, plus the number of false-positive
  connected components and absent-class false-positive volume (for the loss).
- **Cost:** parameters; FLOPs (fvcore or ptflops with the custom op counted by hand); peak GPU memory; step time;
  total training time.

**Statistics:**
- Per-case paired Wilcoxon tests against E0, with Holm correction across the ablations.
- Report the fold mean ± sd and the seed spread.
- A difference smaller than the seed spread is not reported as an effect.

---

## 12. Feasibility

**Laptop (RTX 3050, 4 GB):**
- Swin UNETR at 96³ is already near the memory limit; our runs needed batch 1, accumulation, and checkpointing as a
  fallback.
- MBA is cheap.
- The A-DCN adapter at enc2 adds about 0.3 to 0.5 GB, which is tight; use activation checkpointing on the adapter.
- Use the laptop for E0 versus E2 and E4 on **one fold** as a feasibility check.

**Server (2 × V100, 32 GB):** the full ablation table. At 5 folds × about 15 runs × 2 seeds for the key ones, this is
the dominant cost and must be planned (see the 5-fold time estimate earlier: weeks of GPU time at 100 to 200 epochs).

**Implementation:**
- **Deformable op:** a `grid_sample`-based 3D A-DCN with a unit test (zero offsets must equal a dilated `Conv3d`, up to
  the modulation constant).
- **Speed:** check that `torch.compile` or channels-last does not break `grid_sample`.

---

## 13. Risks and failure modes

- **Swin is already good at these resolutions.** E1 and E2 may tie with E0 (the "innovation bias" finding of nnU-Net
  Revisited). Plan the paper so that a null result is still reportable as an ablation.
- **The gates may stay closed.** A learned α that stays near 0 means the adapters are unused. Our atlas gates behaved
  this way.
- **Deformable offsets may diverge.** Use a lower learning rate for the offset head (e.g. 0.1×), initialise offsets at
  zero, and optionally clip $|\Delta p|\le$ 4 voxels.
- **The loss may hurt** if labels are incomplete or if the patch sampler rarely produces absent classes. Measure first.
- **The current whole-volume 96³ resize hides all fine-scale effects.** Do not run these ablations before switching to
  1 mm patches.
- **The UAMSNet numbers are not a reference.** They come from 2D, different label sets, and a tuned-on-test Synapse
  protocol.

---

## 14. Final recommendation

### A. Minimal model

Swin UNETR + **one zero-initialised, gated 3D A-DCN adapter on the 1/4 skip (enc2)**, trained with Dice + CE (E2), next
to the parameter-matched control (E0p).

Separately and at no extra cost, the absent-class term (E4), but only after the measurement in 8.1 shows that absent-
class false positives exist in patch training.

### B. Full model

E5: A-DCN adapter at enc2 + 3D MBA on enc1 and enc2 + present-class Dice + CE + $\beta(e)L^{(c)}$. Add the dilated
branch only if E2c or E2d beats E2.

### C. Novelty ratings

**3D MBA: low.**
- As coded, it is CBAM spatial attention with a centred input.
- As described, it is a feature-Laplacian gate.
- Boundary attention exists on AMOS already (BA-Net).

**3D HRFE / A-DCN adapter: low.**
- The operator is A-DCN.
- The fusion is ASPP-like.
- 3D plus a Swin skip is "an existing method in a new setting".

**Absent-class penalty $L^{(c)}$: low to moderate (unchecked).**
- The gradient analysis shows it differs from CE and from smoothed Dice (Section 8.2/8.3).
- Exclusion-type losses exist and must be compared before any claim.

### D. Implementation roadmap

1. Switch to 96³ patches at 1 mm with sliding-window inference. Re-run E0 on one fold.
2. Measure absent-class false positives (8.1). Run a label census per case for missing organs.
3. Write the 3D A-DCN op with unit tests and a memory/speed benchmark.
4. Train E0, E0p and E2 on one fold.
5. Add MBA (E1, E1c) and run the AUROC boundary test.
6. Run E4 with the loss variants, only if step 2 finds false positives.
7. On the server: 5 folds for E0, the best single component, and E5; seeds for E0 and the best model.
8. E6 with the atlas prior.

### E. Research recommendation

**As a standalone paper ("UAMSNet ideas in a 3D Swin UNETR"), the methodological substance is thin.** Every component
is an existing operator moved to 3D and to a new host network. The weakest point is the deformable branch: it is
A-DCN unchanged, and UAMSNet itself already reused it without citation. Reviewers who know DCN, A-DCN, CBAM and BA-Net
will see this immediately.

**What would strengthen it:**
1. A clear, multi-seed, 5-fold gain over **nnU-Net ResEnc**, not only over Swin UNETR.
2. Mechanism evidence: NSD and HD95 gains concentrated on thin or small organs; MBA's boundary AUROC; the learned α and
   the offset statistics.
3. A component that is actually new.

The most promising new component is in our own pipeline. The atlas prior gives each patch an *expected* set of organs
and their locations. An **atlas-informed absent-class penalty**:
- uses $P^{atlas}$ to weight $L^{(c)}$ voxel by voxel;
- can be applied even where labels are missing.

This links the target-driven idea to the atlas contribution instead of diluting it, and I found no prior work for it
in this quick search.

For the current paper, I would include at most the A-DCN adapter and the loss as **ablations** next to the atlas prior,
and keep the atlas prior as the contribution.

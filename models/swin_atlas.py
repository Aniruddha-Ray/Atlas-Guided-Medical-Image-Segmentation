"""
Swin UNETR with the same atlas-prior integration as AtlasAwareViTUNETR (prototype.ipynb, "#v2-cell definitions").

Where each prior enters, against the ViT model:

| Prior path | ViT-UNETR | Swin UNETR (this file) |
|---|---|---|
| Voxel path | encoder1 sees [CT, P, D, w] | the input has 1 + 13 + 12 + 1 channels, so both the patch embedding and encoder1 see it |
| Anatomical tokens | 216 tokens (16^3 patches, 26-input MLP) cross-attended in all 12 blocks | the same 216 tokens, cross-attended after Swin stages 2, 3 and 4 (12^3, 6^3 and 3^3 token grids) |
| Trust gate | one scalar per block, init 1 | one scalar per adapter, init 1 |
| Output fusion | z + gamma_c * w * log(P + eps), gamma init 0 (--gamma-init sets it) | identical |

Stage 1 (24^3 tokens) gets no cross-attention: 13.8 k queries are costly on a 4 GB GPU and the voxel path already
carries the prior at that resolution. Swin's windows are local; the anatomical tokens are the only global signal.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR

LOG_PRIOR_EPS = 1e-3
CROSS_ATTN_STAGES = (2, 3, 4)  # indices into the swinViT outputs [x0 (48^3), x1 (24^3), x2 (12^3), x3 (6^3), x4 (3^3)]


class PriorTokens(nn.Module):
    """Per-patch average pooling of [P, D, w] followed by an MLP: the same tokens as ConfidencePriorGenerator."""

    def __init__(self, patch_size, n_in, hidden_size):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.mlp = nn.Sequential(nn.Linear(n_in, hidden_size * 2), nn.GELU(), nn.Linear(hidden_size * 2, hidden_size))

    def forward(self, prob, dist, conf):
        m = torch.cat([prob, dist, conf], dim=1)
        m = F.avg_pool3d(m, kernel_size=self.patch_size, stride=self.patch_size)
        return self.mlp(m.flatten(2).transpose(1, 2))


class GatedCrossAttention(nn.Module):
    """x + gate * CrossAttn(LN(x), tokens) on a (B, C, D, H, W) feature map; gate = 1 at init."""

    def __init__(self, dim, context_dim, num_heads):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, kdim=context_dim, vdim=context_dim, batch_first=True)
        self.gate = nn.Parameter(torch.ones(()))

    def forward(self, x, context):
        b, c, d, h, w = x.shape
        q = x.flatten(2).transpose(1, 2)
        out, _ = self.attn(self.norm(q), context, context, need_weights=False)
        return x + self.gate * out.transpose(1, 2).reshape(b, c, d, h, w)


class AtlasAwareSwinUNETR(SwinUNETR):
    def __init__(self, in_channels, out_channels, img_size, feature_size=48, hidden_size=768, patch_size=16,
                 conf_gating=True, voxel_prior=True, output_fusion=True, use_checkpoint=False, **swin_kw):
        n_prob, n_dist = out_channels, out_channels - 1
        self.voxel_prior = voxel_prior
        super().__init__(in_channels=in_channels + (n_prob + n_dist + 1 if voxel_prior else 0), out_channels=out_channels,
                         feature_size=feature_size, use_checkpoint=use_checkpoint, **swin_kw)
        self.conf_gating, self.output_fusion = conf_gating, output_fusion
        if conf_gating:
            self.prior_tokens = PriorTokens((patch_size,) * 3, n_prob + n_dist + 1, hidden_size)
            dims = {i: feature_size * 2 ** i for i in range(5)}
            heads = {i: n for i, n in zip(range(1, 5), (6, 12, 24, 24))}
            self.cross = nn.ModuleDict({str(i): GatedCrossAttention(dims[i], hidden_size, heads[i] if dims[i] % heads[i] == 0 else 1)
                                        for i in CROSS_ATTN_STAGES})
        if output_fusion:
            self.output_fusion_weight = nn.Parameter(torch.zeros(out_channels))

    def _swin_stages(self, x, tokens):
        sw = self.swinViT
        x0 = sw.pos_drop(sw.patch_embed(x))
        outs = [sw.proj_out(x0, self.normalize)]
        h = x0
        for i, layer in enumerate((sw.layers1, sw.layers2, sw.layers3, sw.layers4), start=1):
            h = layer[0](h.contiguous())
            if tokens is not None and str(i) in self.cross:
                h = self.cross[str(i)](h, tokens)
            outs.append(sw.proj_out(h, self.normalize))
        return outs

    def forward(self, x_in, probability_maps, distance_maps, confidence_maps=None):
        if confidence_maps is None:
            confidence_maps = torch.ones_like(probability_maps[:, :1])
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._check_input_size(x_in.shape[2:])
        tokens = self.prior_tokens(probability_maps, distance_maps, confidence_maps) if self.conf_gating else None
        x = torch.cat([x_in, probability_maps, distance_maps, confidence_maps], 1) if self.voxel_prior else x_in
        hs = self._swin_stages(x, tokens)
        enc0 = self.encoder1(x)
        enc1 = self.encoder2(hs[0])
        enc2 = self.encoder3(hs[1])
        enc3 = self.encoder4(hs[2])
        dec4 = self.encoder10(hs[4])
        dec3 = self.decoder5(dec4, hs[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        logits = self.out(self.decoder1(dec0, enc0))
        if self.output_fusion:
            log_prior = torch.log(probability_maps.float() + LOG_PRIOR_EPS)
            logits = logits + self.output_fusion_weight.view(1, -1, 1, 1, 1) * confidence_maps.float() * log_prior
        return logits

    def new_parameters(self):
        out = {}
        if self.conf_gating:
            out["cross_gates"] = torch.stack([m.gate.detach() for m in self.cross.values()])
        if self.output_fusion:
            out["gamma"] = self.output_fusion_weight.detach()
        return out

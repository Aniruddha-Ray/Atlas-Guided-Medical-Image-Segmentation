"""
AtlasGuidedViTUNETR: MONAI UNETR with its ViT swapped for a cross-attention
ViT that consumes anatomical prior tokens (pooled probability + distance
maps). Ported verbatim from the notebook (AtlasGuidedUNETR_fixed.ipynb,
"Model Architecture" section) so it is importable from scripts/ and tests/
without duplicating the class definitions. No behavioral changes here --
see research_audit.md Section A for the architecture writeup and Section E
for checkpoint compatibility notes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import TransformerBlock
from monai.networks.nets import UNETR, ViT


class AnatomicalPriorGenerator(nn.Module):
    """
    Pools the per-organ probability map and per-organ distance map inside each
    ViT patch and projects the pooled vector into a token living in the same
    hidden_size space as the ViT patch tokens. Fully vectorised with
    avg_pool3d (mathematically identical to a per-patch for-loop, just faster).
    """

    def __init__(self, img_size, patch_size, num_prob_channels, num_dist_channels, hidden_size):
        super().__init__()
        self.patch_size = patch_size
        input_dim = num_prob_channels + num_dist_channels
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, probability_maps, distance_maps):
        pD, pH, pW = self.patch_size
        p_pooled = F.avg_pool3d(probability_maps, kernel_size=(pD, pH, pW), stride=(pD, pH, pW))
        d_pooled = F.avg_pool3d(distance_maps, kernel_size=(pD, pH, pW), stride=(pD, pH, pW))
        m = torch.cat([p_pooled, d_pooled], dim=1)  # (B, C_total, nd, nh, nw)
        b, c, nd, nh, nw = m.shape
        m = m.flatten(2).transpose(1, 2)  # (B, num_patches, C_total)
        anatomical_tokens = self.mlp(m)  # (B, num_patches, hidden_size)
        return anatomical_tokens


class CustomViT(ViT):
    """
    MONAI's ViT, except every TransformerBlock is built with
    with_cross_attention=True (MONAI's built-in cross-attention) so every
    block can attend to the anatomical_tokens coming from
    AnatomicalPriorGenerator.
    """

    def __init__(
        self,
        in_channels,
        img_size,
        patch_size,
        hidden_size,
        mlp_dim,
        num_heads,
        proj_type="perceptron",
        dropout_rate=0.0,
        qkv_bias=False,
        save_attn=False,
    ):
        super().__init__(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=12,
            num_heads=num_heads,
            proj_type=proj_type,
            classification=False,
            dropout_rate=dropout_rate,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    mlp_dim,
                    num_heads,
                    dropout_rate,
                    qkv_bias=qkv_bias,
                    save_attn=save_attn,
                    with_cross_attention=True,
                )
                for _ in range(12)
            ]
        )

    def forward(self, x, anatomical_tokens):
        x = self.patch_embedding(x)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x, context=anatomical_tokens)
            hidden_states_out.append(x)
        x = self.norm(x)
        return x, hidden_states_out


class AtlasGuidedViTUNETR(UNETR):
    """
    Subclasses MONAI's UNETR and keeps its encoder1-4 / decoder2-5 / out /
    proj_feat exactly as MONAI built them (i.e. the decoder is unchanged
    stock UNETR). The only difference: self.vit is CustomViT, and forward()
    additionally builds anatomical_tokens from the probability/distance maps
    and feeds them in as cross-attention context.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        proj_type: str = "perceptron",
        norm_name: str = "instance",
        res_block: bool = True,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        num_prob_channels: int = None,
        num_dist_channels: int = None,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            proj_type=proj_type,
            norm_name=norm_name,
            res_block=res_block,
            dropout_rate=dropout_rate,
            qkv_bias=qkv_bias,
        )

        # probability_maps -> one-hot label, has `out_channels` channels (incl. background)
        # distance_maps    -> CustomSpatialDistanceTransformd with include_background=False,
        #                     so it has `out_channels - 1` channels.
        num_prob_channels = num_prob_channels if num_prob_channels is not None else out_channels
        num_dist_channels = num_dist_channels if num_dist_channels is not None else (out_channels - 1)

        self.vit = CustomViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            proj_type=proj_type,
            dropout_rate=dropout_rate,
            qkv_bias=qkv_bias,
        )

        self.anatomical_prior_generator = AnatomicalPriorGenerator(
            img_size=img_size,
            patch_size=self.patch_size,
            num_prob_channels=num_prob_channels,
            num_dist_channels=num_dist_channels,
            hidden_size=hidden_size,
        )

    def forward(self, x_in, probability_maps, distance_maps):
        anatomical_tokens = self.anatomical_prior_generator(probability_maps, distance_maps)
        x, hidden_states_out = self.vit(x_in, anatomical_tokens)

        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(self.proj_feat(hidden_states_out[3]))
        enc3 = self.encoder3(self.proj_feat(hidden_states_out[6]))
        enc4 = self.encoder4(self.proj_feat(hidden_states_out[9]))
        dec4 = self.proj_feat(x)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        return self.out(out)

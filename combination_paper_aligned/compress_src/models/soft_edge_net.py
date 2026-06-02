"""
Patch embedding + temporal Transformer + PixelShuffle decoder (anti-block).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int = 3, patch_size: int = 16, embed_dim: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ps = self.patch_size
        _, _, h, w = x.shape
        pad_h = (ps - h % ps) % ps
        pad_w = (ps - w % ps) % ps
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class FrameDecoder(nn.Module):
    """
    Tokens -> feature map -> PixelShuffle x4 upsample (no per-patch seams).
  Predicts a residual; caller adds the soft-edge frame.
    """

    def __init__(self, embed_dim: int, patch_size: int, out_h: int, out_w: int):
        super().__init__()
        self.patch_size = patch_size
        self.out_h = out_h
        self.out_w = out_w
        pad_h = (patch_size - out_h % patch_size) % patch_size
        pad_w = (patch_size - out_w % patch_size) % patch_size
        self.padded_h = out_h + pad_h
        self.padded_w = out_w + pad_w
        self.gh = self.padded_h // patch_size
        self.gw = self.padded_w // patch_size

        ups = int(round(math.log2(patch_size)))
        if 2**ups != patch_size:
            raise ValueError(f"patch_size must be power of 2, got {patch_size}")

        layers: list[nn.Module] = [
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.GELU(),
        ]
        ch = embed_dim
        for _ in range(ups):
            layers += [
                nn.Conv2d(ch, ch * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.GELU(),
            ]
        layers += [
            nn.Conv2d(ch, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 3, 3, padding=1),
        ]
        self.net = nn.Sequential(*layers)
        last = layers[-1]
        assert isinstance(last, nn.Conv2d)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, _, d = tokens.shape
        x = tokens.transpose(1, 2).contiguous().view(b, d, self.gh, self.gw)
        x = self.net(x)
        return x[:, :, : self.out_h, : self.out_w]


class SoftEdgeReconstructor(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        patch_size: int = 16,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        temporal_window: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.temporal_window = temporal_window
        self.embed = PatchEmbed(3, patch_size, embed_dim)
        self.pos_time = nn.Parameter(torch.randn(1, temporal_window, embed_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.decoder = FrameDecoder(embed_dim, patch_size, height, width)

    def forward(self, soft_seq: torch.Tensor) -> torch.Tensor:
        b, t, _, h, w = soft_seq.shape
        assert t == self.temporal_window
        tokens = [self.embed(soft_seq[:, i]) for i in range(t)]
        x = torch.cat(tokens, dim=1)
        p = tokens[0].shape[1]
        time_bias = self.pos_time.unsqueeze(2).expand(b, -1, p, -1).reshape(b, t * p, -1)
        x = x + time_bias
        x = self.transformer(x)
        last_tokens = x[:, -p:, :]
        base = soft_seq[:, -1]
        residual = self.decoder(last_tokens)
        if residual.shape[-2:] != base.shape[-2:]:
            residual = F.interpolate(
                residual, size=base.shape[-2:], mode="bilinear", align_corners=False
            )
        return (base + residual).clamp(0.0, 1.0)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:9].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_n = (pred - self.mean) / self.std
        tgt_n = (target - self.mean) / self.std
        return F.mse_loss(self.vgg(pred_n), self.vgg(tgt_n))


def composite_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_mse: float = 1.0,
    lambda_perceptual: float = 0.1,
    perc: PerceptualLoss | None = None,
) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    if lambda_perceptual > 0 and perc is not None:
        p = perc(pred, target)
        return lambda_mse * mse + lambda_perceptual * p
    return lambda_mse * mse

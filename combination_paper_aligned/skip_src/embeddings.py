"""Frame embeddings from a torchvision backbone."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms


class FrameEmbedder:
    def __init__(
        self,
        backbone: str = "mobilenet_v3_small",
        device: str | None = None,
        input_size: int = 224,
        use_fp16: bool = True,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.use_fp16 = bool(use_fp16 and self.device.type == "cuda")
        self.input_size = int(input_size)
        self.model, self.dim = self._build(backbone)
        self.model.eval().to(self.device)
        if self.use_fp16:
            self.model.half()
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @staticmethod
    def _build(name: str) -> tuple[nn.Module, int]:
        if name == "mobilenet_v3_small":
            net = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.DEFAULT
            )
            net.classifier = nn.Identity()
            return net, 576
        if name == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            net.fc = nn.Identity()
            return net, 512
        if name == "efficientnet_b0":
            net = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT
            )
            net.classifier = nn.Identity()
            return net, 1280
        raise ValueError(f"Unknown backbone: {name}")

    def _preprocess_tensor(self, frame_bgr: np.ndarray) -> torch.Tensor:
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(rgb)

    @torch.no_grad()
    def encode(self, frame_bgr: np.ndarray) -> np.ndarray:
        x = self._preprocess_tensor(frame_bgr).unsqueeze(0).to(self.device)
        if self.use_fp16:
            x = x.half()
        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            feat = self.model(x).flatten()
        v = feat.float().cpu().numpy().astype(np.float32)
        n = np.linalg.norm(v) + 1e-8
        return v / n

    @torch.no_grad()
    def encode_batch(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        if not frames_bgr:
            return np.zeros((0, self.dim), dtype=np.float32)
        if len(frames_bgr) == 1:
            return self.encode(frames_bgr[0])[np.newaxis, :]
        batch = torch.stack(
            [self._preprocess_tensor(f) for f in frames_bgr], dim=0
        ).to(self.device)
        if self.use_fp16:
            batch = batch.half()
        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            feats = self.model(batch).reshape(len(frames_bgr), -1)
        v = feats.float().cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
        return v / norms


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_sim(a, b)

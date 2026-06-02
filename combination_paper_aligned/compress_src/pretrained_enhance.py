"""Optional pretrained enhancers (download weights on first use)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_WEIGHTS = _ROOT / "outputs" / "pretrained_weights"
_FSRCNN_URL = (
    "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb"
)
_REALESRGAN_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
    "RealESRGAN_x4plus.pth"
)

_realesrgan = None
_lama = None
_opencv_sr = None


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file() or dest.stat().st_size < 1000:
        print(f"Downloading {dest.name} ...")
        urllib.request.urlretrieve(url, dest)
    return dest


def _opencv_superres(rgb: np.ndarray) -> np.ndarray:
    """Sharpen via detail enhance (works with standard opencv-python)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = cv2.detailEnhance(bgr, sigma_s=10, sigma_r=0.12)
    blur = cv2.GaussianBlur(bgr, (0, 0), 1.2)
    bgr = cv2.addWeighted(bgr, 1.25, blur, -0.25, 0)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _realesrgan_enhance(rgb: np.ndarray) -> np.ndarray:
    global _realesrgan
    if _realesrgan is None:
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model_path = _download(_REALESRGAN_URL, _WEIGHTS / "RealESRGAN_x4plus.pth")
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
            _realesrgan = RealESRGANer(
                scale=4,
                model_path=str(model_path),
                model=model,
                tile=256 if dev == "cuda" else 0,
                tile_pad=10,
                pre_pad=0,
                half=dev == "cuda",
                device=dev,
            )
        except ImportError:
            from .rrdbnet import RRDBNet

            model_path = _download(_REALESRGAN_URL, _WEIGHTS / "RealESRGAN_x4plus.pth")
            net = RRDBNet(scale=4)
            state = torch.load(str(model_path), map_location=dev)
            params = state.get("params_ema", state.get("params", state))
            net.load_state_dict(params, strict=True)
            net.eval().to(dev)
            _realesrgan = ("torch", net, dev)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if isinstance(_realesrgan, tuple) and _realesrgan[0] == "torch":
        import torch

        _, net, dev = _realesrgan
        img = torch.from_numpy(bgr).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img = img.to(dev)
        with torch.no_grad():
            out_t = net(img)[0].clamp(0, 1).cpu().numpy()
        out = (out_t.transpose(1, 2, 0) * 255).astype(np.uint8)
        if out.shape[:2] != bgr.shape[:2]:
            out = cv2.resize(out, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    out, _ = _realesrgan.enhance(bgr, outscale=1)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _edge_guided_blend(rgb: np.ndarray, soft_rgb: np.ndarray, strength: float) -> np.ndarray:
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 0:
        return rgb
    gray = cv2.cvtColor(soft_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    edge_w = np.clip(gray * 3.0, 0, 1)[..., None]
    base = soft_rgb.astype(np.float32)
    pred = rgb.astype(np.float32)
    return np.clip(pred * (1.0 - s * edge_w) + base * (s * edge_w), 0, 255).astype(np.uint8)


def enhance_frame(
    rgb: np.ndarray,
    soft_rgb: np.ndarray | None,
    mode: str,
    edge_blend: float = 0.15,
) -> np.ndarray:
    m = (mode or "").lower().strip()
    if not m or m in ("none", "off", "false"):
        return rgb
    out = rgb
    if m in ("opencv_sr", "fsrcnn", "sr"):
        out = _opencv_superres(rgb)
    elif m in ("realesrgan", "real-esrgan", "esrgan"):
        try:
            out = _realesrgan_enhance(rgb)
        except Exception as exc:
            print(f"Real-ESRGAN unavailable ({exc}); using OpenCV FSRCNN.")
            out = _opencv_superres(rgb)
    elif m in ("detail", "opencv_detail"):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr = cv2.detailEnhance(bgr, sigma_s=12, sigma_r=0.15)
        out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unknown pretrained_enhance mode: {mode}")
    if soft_rgb is not None and edge_blend > 0:
        out = _edge_guided_blend(out, soft_rgb, edge_blend)
    return out


def lama_inpaint(rgb: np.ndarray, soft_rgb: np.ndarray) -> np.ndarray:
    """LaMa inpainting: fill non-edge regions using pretrained model."""
    global _lama
    if _lama is None:
        try:
            from simple_lama_inpainting import SimpleLama
        except ImportError as exc:
            raise ImportError(
                "pip install simple-lama-inpainting for fill_mode=lama"
            ) from exc
        _lama = SimpleLama()
    gray = cv2.cvtColor(soft_rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray < 8).astype(np.uint8) * 255
    if mask.sum() == 0:
        return rgb
    guide = baseline_inpaint_guide(soft_rgb)
    from PIL import Image

    pil_img = Image.fromarray(guide)
    pil_mask = Image.fromarray(mask)
    result = _lama(pil_img, pil_mask)
    return np.array(result.convert("RGB"))


def baseline_inpaint_guide(soft_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(soft_rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray > 0).astype(np.uint8) * 255
    if mask.sum() == 0:
        return soft_rgb
    dilated = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    bgr = cv2.cvtColor(soft_rgb, cv2.COLOR_RGB2BGR)
    filled = cv2.inpaint(bgr, dilated, 3, cv2.INPAINT_TELEA)
    return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)

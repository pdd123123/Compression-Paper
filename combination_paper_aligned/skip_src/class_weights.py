"""Inverse-log class frequency weights (YOLO COCO classes)."""

from __future__ import annotations

from collections import deque
from math import log

import numpy as np

# COCO names used by YOLOv8
COCO_NAMES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


class ClassWeightTracker:
    def __init__(
        self,
        window_frames: int = 60,
        rare_classes: list[str] | None = None,
    ):
        self.window = window_frames
        self.rare = set(rare_classes or ["person", "bicycle", "motorcycle", "bus", "truck"])
        self._history: deque[list[int]] = deque(maxlen=window_frames)
        self._counts: dict[int, int] = {}

    def observe(self, class_ids: list[int]) -> None:
        if len(self._history) == self._history.maxlen:
            old = self._history[0]
            for c in old:
                self._counts[c] = self._counts.get(c, 0) - 1
                if self._counts[c] <= 0:
                    self._counts.pop(c, None)
        self._history.append(class_ids)
        for c in class_ids:
            self._counts[c] = self._counts.get(c, 0) + 1

    def weight_for_classes(self, present_ids: list[int]) -> float:
        """Max weight over classes present in the frame."""
        if not present_ids:
            return 1.0
        weights = [self._wc(c) for c in present_ids]
        return float(max(weights))

    def _wc(self, class_id: int) -> float:
        fc = self._counts.get(class_id, 0)
        base = 1.0 / log(1.0 + fc + 1.0)
        name = COCO_NAMES[class_id] if class_id < len(COCO_NAMES) else ""
        if name in self.rare:
            base *= 1.25
        return base


class YoloClassDetector:
    """YOLO detector; loaded on first use."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        imgsz: int = 640,
        half: bool = True,
    ):
        self.model_name = model_name
        self.imgsz = int(imgsz)
        self.half = bool(half)
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_name)

    def detect_class_ids(self, frame_bgr: np.ndarray, conf: float = 0.25) -> list[int]:
        batch = self.detect_class_ids_batch([frame_bgr], conf=conf)
        return batch[0] if batch else []

    def detect_class_ids_batch(
        self, frames_bgr: list[np.ndarray], conf: float = 0.25
    ) -> list[list[int]]:
        if not frames_bgr:
            return []
        try:
            self._load()
        except Exception:
            return [[] for _ in frames_bgr]
        import torch

        use_half = self.half and torch.cuda.is_available()
        results = self._model.predict(
            frames_bgr,
            verbose=False,
            conf=conf,
            imgsz=self.imgsz,
            half=use_half,
        )
        out: list[list[int]] = []
        for res in results:
            if res.boxes is None or len(res.boxes) == 0:
                out.append([])
            else:
                out.append([int(c) for c in res.boxes.cls.cpu().numpy().tolist()])
        return out

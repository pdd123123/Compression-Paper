"""Traffic analytics proxies (paper skip_sampling_combined figure)."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class TrackState:
    track_id: int
    cx: float
    cy: float
    last_frame: int
    speeds_px: list[float] = field(default_factory=list)


class SimpleIoUTracker:
    """Lightweight centroid tracker for unique-ID and speed proxies."""

    def __init__(self, iou_thresh: float = 0.25, max_age: int = 30):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self._tracks: dict[int, TrackState] = {}
        self._next_id = 1
        self._active: dict[int, int] = {}  # tid -> last seen frame

    def update(self, boxes: list[tuple[float, float, float, float]], frame_idx: int, fps: float):
        # boxes: xyxy
        assigned = set()
        for box in boxes:
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            best_tid, best_iou = None, 0.0
            for tid, st in self._tracks.items():
                if frame_idx - st.last_frame > self.max_age:
                    continue
                iou = _box_iou(box, _center_box(st.cx, st.cy, box))
                if iou > best_iou and iou >= self.iou_thresh:
                    best_iou = iou
                    best_tid = tid
            if best_tid is None:
                best_tid = self._next_id
                self._next_id += 1
                self._tracks[best_tid] = TrackState(best_tid, cx, cy, frame_idx)
            st = self._tracks[best_tid]
            if st.last_frame < frame_idx and fps > 0:
                dist = ((cx - st.cx) ** 2 + (cy - st.cy) ** 2) ** 0.5
                dt = (frame_idx - st.last_frame) / fps
                if dt > 1e-6:
                    st.speeds_px.append(dist / dt)
            st.cx, st.cy, st.last_frame = cx, cy, frame_idx
            self._active[best_tid] = frame_idx
            assigned.add(best_tid)

    def unique_ids(self) -> int:
        return len(self._tracks)

    def speed_stats(self) -> tuple[float, float]:
        all_spd = [s for t in self._tracks.values() for s in t.speeds_px]
        if not all_spd:
            return 0.0, 0.0
        return float(np.mean(all_spd)), float(np.max(all_spd))


def _center_box(cx: float, cy: float, ref: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w = ref[2] - ref[0]
    h = ref[3] - ref[1]
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter <= 0:
        return 0.0
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    return inter / (area_a + area_b - inter + 1e-8)


def analyze_video_traffic(
    video_path: str,
    max_frames: int | None = None,
    frame_indices: list[int] | None = None,
    yolo_model: str = "yolov8n.pt",
    vehicle_classes: set[int] | None = None,
) -> dict:
    """
    Run YOLO + tracker. If frame_indices given, only those source frames are processed
  (simulating skip sampling on the original timeline).
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:
        return {"error": str(e)}

    if vehicle_classes is None:
        # person + road vehicles (COCO ids)
        vehicle_classes = {0, 1, 2, 3, 5, 7}

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    model = YOLO(yolo_model)
    tracker = SimpleIoUTracker()
    allow = frame_indices is not None
    index_set = set(frame_indices or [])

    fi = 0
    processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        use = (not allow) or (fi in index_set)
        if use:
            res = model.predict(frame, verbose=False, conf=0.25)[0]
            boxes = []
            if res.boxes is not None and len(res.boxes):
                for b in res.boxes:
                    cid = int(b.cls.item())
                    if cid in vehicle_classes or cid == 0:  # person
                        xyxy = b.xyxy[0].cpu().numpy().tolist()
                        boxes.append(tuple(xyxy))
            tracker.update(boxes, fi, fps)
            processed += 1
        fi += 1
        if max_frames and fi >= max_frames:
            break
    cap.release()

    avg_s, max_s = tracker.speed_stats()
    return {
        "frames_processed": processed,
        "unique_track_ids": tracker.unique_ids(),
        "avg_speed_px_per_s": avg_s,
        "max_speed_px_per_s": max_s,
    }


def compare_to_reference(ref: dict, test: dict) -> dict:
    """Metrics aligned with paper: ID retention %, speed error %."""
    uid_ref = ref.get("unique_track_ids", 0)
    uid_test = test.get("unique_track_ids", 0)
    id_retention = (uid_test / uid_ref * 100) if uid_ref else 100.0

    avg_ref = ref.get("avg_speed_px_per_s", 0)
    max_ref = ref.get("max_speed_px_per_s", 0)
    avg_test = test.get("avg_speed_px_per_s", 0)
    max_test = test.get("max_speed_px_per_s", 0)

    avg_err = abs(avg_test - avg_ref) / (avg_ref + 1e-8) * 100
    max_err = abs(max_test - max_ref) / (max_ref + 1e-8) * 100

    return {
        "unique_id_retention_pct": id_retention,
        "avg_speed_error_pct": avg_err,
        "max_speed_error_pct": max_err,
    }

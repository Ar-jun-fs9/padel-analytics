"""
Model Loader
Downloads / loads YOLOv8 for person + sports-ball detection.
Falls back gracefully if ultralytics is not installed.
"""

import os
from pathlib import Path


MODELS_DIR = Path(__file__).parent.parent / "models"


class ModelLoader:
    def __init__(self, device="cpu", conf=0.3):
        self.device = device
        self.conf = conf
        MODELS_DIR.mkdir(exist_ok=True)

    def load(self):
        """Return a callable YOLO model."""
        try:
            from ultralytics import YOLO
            model_path = MODELS_DIR / "yolov8n.pt"
            print(f"   Loading YOLOv8n  (device={self.device})")
            model = YOLO(str(model_path) if model_path.exists() else "yolov8n.pt")
            model.to(self.device)
            # Wrap so conf threshold is applied
            return _YOLOWrapper(model, self.conf)
        except ImportError:
            print("   [WARN] ultralytics not installed — using mock detector")
            return _MockDetector()
        except Exception as e:
            print(f"   [WARN] YOLO load failed ({e}) — using mock detector")
            return _MockDetector()


# ──────────────────────────────────────────────────────────────────────────────

class _YOLOWrapper:
    """Thin wrapper that sets conf threshold before inference."""

    def __init__(self, model, conf):
        self.model = model
        self.conf = conf

    def __call__(self, frame, **kwargs):
        return self.model(frame, conf=self.conf, **kwargs)


class _MockDetector:
    """
    Deterministic mock used when YOLO is unavailable.
    Produces synthetic ball + player detections so the rest of
    the pipeline can be tested without a GPU / internet.
    """

    class _FakeBox:
        def __init__(self, cls_id, conf, xyxy):
            import torch
            self.cls  = [cls_id]
            self.conf = [conf]
            self.xyxy = [xyxy]

    class _FakeResult:
        def __init__(self, boxes, names):
            self.boxes = boxes
            self.names = names

    def __call__(self, frame, **kwargs):
        import math, random
        h, w = frame.shape[:2]
        t = (id(frame) % 1000) / 30.0          # pseudo-time

        # Simulate ball moving in a sine curve
        bx = int(w * (0.3 + 0.4 * abs(math.sin(t * 0.7))))
        by = int(h * (0.3 + 0.3 * abs(math.cos(t * 1.1))))
        ball_box = self._FakeBox(32, 0.75, [bx - 8, by - 8, bx + 8, by + 8])

        # Two static-ish players
        p1_box = self._FakeBox(0, 0.90, [w * 0.1, h * 0.5, w * 0.25, h * 0.95])
        p2_box = self._FakeBox(0, 0.88, [w * 0.7, h * 0.5, w * 0.85, h * 0.95])

        names = {0: "person", 32: "sports ball"}
        result = self._FakeResult([ball_box, p1_box, p2_box], names)
        return [result]

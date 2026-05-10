"""
Core Padel Analyzer
Handles video processing, object detection, tracking, and shot classification
"""

import cv2
import numpy as np
import json
import csv
import time
import os
from pathlib import Path
from datetime import datetime
from collections import deque

from utils.tracker import MultiObjectTracker
from utils.shot_classifier import ShotClassifier
from utils.ball_tracker import BallTracker
from utils.analytics import AnalyticsEngine
from utils.visualizer import Visualizer
from utils.model_loader import ModelLoader


class PadelAnalyzer:
    def __init__(
        self,
        output_dir="output",
        show_preview=False,
        conf_threshold=0.3,
        device="cpu",
        skip_frames=2,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.show_preview = show_preview
        self.conf_threshold = conf_threshold
        self.device = device
        self.skip_frames = skip_frames

        print("\n[1/5] Loading detection models...")
        self.model_loader = ModelLoader(device=device, conf=conf_threshold)
        self.detector = self.model_loader.load()

        print("[2/5] Initialising trackers...")
        self.ball_tracker = BallTracker()
        self.player_tracker = MultiObjectTracker(
            max_age=30,
            min_hits=1,
            use_court_zones=True,
        )

        print("[3/5] Initialising shot classifier...")
        self.shot_classifier = ShotClassifier()

        print("[4/5] Initialising analytics engine...")
        self.analytics = AnalyticsEngine()

        print("[5/5] Initialising visualizer...")
        self.visualizer = Visualizer()

        self.shots = []
        self.frame_data = []
        self.frame_dims = None  # Will be set on first frame

    # ------------------------------------------------------------------
    def process_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n[INFO] Video: {Path(video_path).name}")
        print(f"   Resolution : {width}×{height}  |  FPS: {fps:.1f}  |  Frames: {total_frames}")

        # Output video writer
        out_path = self.output_dir / "output_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        frame_idx = 0
        processed = 0
        start_time = time.time()

        print("\n[INFO] Processing video …  (press Q in preview to stop)\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Skip frames for speed
            if frame_idx % self.skip_frames != 0:
                writer.write(frame)
                continue

            processed += 1
            timestamp = frame_idx / fps

            # Store frame dimensions (needed for court zone logic)
            h, w = frame.shape[:2]
            if self.frame_dims is None:
                self.frame_dims = (h, w)
                # Update shot classifier with frame height
                self.shot_classifier.frame_height = h

            # ── Detect objects ──────────────────────────────────────────
            detections = self._detect(frame)

            # ── Separate ball vs player detections ─────────────────────
            ball_dets   = [d for d in detections if d["class"] == "sports ball"]
            player_dets = [d for d in detections if d["class"] == "person"]

            # ── Track ──────────────────────────────────────────────────
            ball_pos   = self.ball_tracker.update(ball_dets, frame)
            player_tracks = self.player_tracker.update(player_dets, frame_shape=(h, w))

            # ── Classify shots ─────────────────────────────────────────
            new_shots = self.shot_classifier.classify(
                ball_pos, player_tracks, frame_idx, timestamp
            )
            self.shots.extend(new_shots)

            # ── Analytics ──────────────────────────────────────────────
            self.analytics.update(new_shots, ball_pos, player_tracks, frame_idx)

            # ── Annotate frame ─────────────────────────────────────────
            annotated = self.visualizer.draw(
                frame.copy(),
                ball_pos,
                player_tracks,
                new_shots,
                self.analytics.get_live_stats(),
                frame_idx,
                timestamp,
            )

            writer.write(annotated)

            if self.show_preview:
                cv2.imshow("Padel Analytics", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[Preview] Stopped by user.")
                    break

            # Progress
            if processed % 50 == 0:
                elapsed = time.time() - start_time
                pct = frame_idx / total_frames * 100
                eta = (elapsed / processed) * (total_frames // self.skip_frames - processed)
                print(f"   Frame {frame_idx:>5}/{total_frames}  ({pct:.1f}%)  "
                      f"| Shots: {len(self.shots):>3}  "
                      f"| ETA: {eta:.0f}s")

        cap.release()
        writer.release()
        if self.show_preview:
            cv2.destroyAllWindows()

        print(f"\n[INFO] Processing complete in {elapsed:.1f}s")
        print(f"   Total shots detected: {len(self.shots)}")

        self._save_results(video_path, fps, total_frames)

    # ------------------------------------------------------------------
    def _detect(self, frame):
        """Run YOLO detection and return list of dicts."""
        results = self.detector(frame, verbose=False)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = r.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "class": cls_name,
                    "conf": conf,
                    "bbox": [x1, y1, x2, y2],
                    "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                })
        return detections

    # ------------------------------------------------------------------
    def _save_results(self, video_path, fps, total_frames):
        print("\n[INFO] Saving results …")

        # ── JSON ────────────────────────────────────────────────────────
        summary = self.analytics.get_summary()
        output_json = {
            "metadata": {
                "video": Path(video_path).name,
                "processed_at": datetime.now().isoformat(),
                "total_frames": total_frames,
                "fps": fps,
                "total_shots": len(self.shots),
            },
            "summary": summary,
            "shots": self.shots,
        }
        json_path = self.output_dir / "shots.json"
        with open(json_path, "w") as f:
            json.dump(output_json, f, indent=2, default=str)
        print(f"   JSON  → {json_path}")

        # ── CSV ─────────────────────────────────────────────────────────
        csv_path = self.output_dir / "shots.csv"
        if self.shots:
            fieldnames = list(self.shots[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.shots)
        else:
            with open(csv_path, "w", newline="") as f:
                f.write("frame,timestamp,shot_type,player_id,confidence,ball_x,ball_y,direction\n")
        print(f"   CSV   → {csv_path}")

        # ── Analytics report ────────────────────────────────────────────
        report_path = self.output_dir / "analytics_report.json"
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"   Stats → {report_path}")

        # ── Print summary ───────────────────────────────────────────────
        print("\n" + "=" * 50)
        print("  ANALYTICS SUMMARY")
        print("=" * 50)
        for shot_type, count in summary.get("shot_counts", {}).items():
            bar = "=" * min(count, 40)
            print(f"  {shot_type:<15} {bar} {count}")
        print("=" * 50)
        print(f"\n  Output video -> {self.output_dir / 'output_annotated.mp4'}")
        print(f"  Run dashboard: python main.py --dashboard\n")

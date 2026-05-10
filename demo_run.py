"""
demo_run.py  —  Run a full simulation without a real video file.

Usage:
    python demo_run.py

Generates synthetic detections, classifies shots, and writes:
  output/shots.json
  output/shots.csv
  output/analytics_report.json
  output/demo_annotated.mp4   (synthetic frames)
"""

import cv2
import numpy as np
import json
import csv
import math
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from utils.ball_tracker import BallTracker
from utils.tracker import MultiObjectTracker
from utils.shot_classifier import ShotClassifier
from utils.analytics import AnalyticsEngine
from utils.visualizer import Visualizer

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

W, H    = 1280, 720
FPS     = 30
FRAMES  = 600   # 20-second synthetic clip


def make_frame(frame_idx):
    """Create a synthetic padel court frame."""
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:] = (20, 35, 20)           # dark green background

    # Court lines
    cv2.rectangle(frame, (60, 80),  (W - 60, H - 80), (200, 220, 200), 2)
    cv2.line(frame, (W // 2, 80), (W // 2, H - 80), (200, 220, 200), 2)
    cv2.line(frame, (60, H // 2),  (W - 60, H // 2), (200, 220, 200), 1)

    # Side walls (padel is played with glass walls)
    cv2.rectangle(frame, (0, 0), (55, H), (30, 50, 30), -1)
    cv2.rectangle(frame, (W - 55, 0), (W, H), (30, 50, 30), -1)
    return frame


def synthetic_ball(t):
    """Simulate ball trajectory with shot events."""
    bx = W * 0.5 + math.sin(t * 0.9) * W * 0.28
    by = H * 0.4 + math.sin(t * 1.4) * H * 0.25
    conf = 0.85 + 0.1 * math.sin(t)
    return {"center": [bx, by], "conf": max(0.3, conf), "bbox": [bx-8, by-8, bx+8, by+8], "class": "sports ball"}


def synthetic_players():
    return [
        {"bbox": [80,  380, 220, 680], "conf": 0.92, "class": "person"},
        {"bbox": [1060, 380, 1200, 680], "conf": 0.90, "class": "person"},
    ]


def main():
    print("=" * 60)
    print("  PADEL ANALYTICS — Synthetic Video Demo")
    print("=" * 60)

    ball_tracker   = BallTracker()
    player_tracker = MultiObjectTracker(max_age=30, min_hits=1, use_court_zones=True, frame_width=W)
    shot_clf       = ShotClassifier(frame_height=H)
    analytics      = AnalyticsEngine()
    visualizer     = Visualizer()

    out_path = OUTPUT_DIR / "demo_annotated.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_path), fourcc, FPS, (W, H))

    shots = []
    start = time.time()

    for fi in range(1, FRAMES + 1):
        t   = fi / FPS
        frame = make_frame(fi)

        ball_det  = synthetic_ball(t)
        player_det = synthetic_players()

        ball_pos      = ball_tracker.update([ball_det])
        player_tracks = player_tracker.update(player_det, frame_shape=(H, W))

        new_shots = shot_clf.classify(ball_pos, player_tracks, fi, t)
        shots.extend(new_shots)
        analytics.update(new_shots, ball_pos, player_tracks, fi)

        annotated = visualizer.draw(
            frame, ball_pos, player_tracks, new_shots,
            analytics.get_live_stats(), fi, t,
        )
        writer.write(annotated)

        if fi % 100 == 0:
            print(f"   Frame {fi}/{FRAMES}  | Shots so far: {len(shots)}")

    writer.release()
    elapsed = time.time() - start
    summary = analytics.get_summary()

    # ── Save JSON ──────────────────────────────────────────────────────
    output_json = {
        "metadata": {
            "video": "demo_synthetic",
            "processed_at": datetime.now().isoformat(),
            "total_frames": FRAMES,
            "fps": FPS,
            "total_shots": len(shots),
        },
        "summary": summary,
        "shots": shots,
    }
    json_path = OUTPUT_DIR / "shots.json"
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2, default=str)

    # ── Save CSV ───────────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / "shots.csv"
    if shots:
        with open(csv_path, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=list(shots[0].keys()))
            writer_csv.writeheader()
            writer_csv.writerows(shots)

    # ── Save analytics report ──────────────────────────────────────────
    report_path = OUTPUT_DIR / "analytics_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[OK] Demo complete in {elapsed:.1f}s")
    print(f"   Total shots: {len(shots)}")
    print(f"\n   Shot counts:")
    for st, cnt in summary.get("shot_counts", {}).items():
        bar = "=" * min(cnt, 40)
        print(f"     {st:<15} {bar} ({cnt})")
    print(f"\n   Output video  -> {out_path}")
    print(f"   Shots JSON    -> {json_path}")
    print(f"   Shots CSV     -> {csv_path}")
    print(f"\n   Run dashboard: python main.py --dashboard")


if __name__ == "__main__":
    main()

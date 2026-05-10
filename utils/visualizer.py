"""
Visualizer
Draws detection overlays, trajectories, shot labels, and stats HUD onto frames.
"""

import cv2
import numpy as np
from collections import deque


# Colour palette (BGR)
COLORS = {
    "FOREHAND":     (0,   220, 100),
    "BACKHAND":     (30,  144, 255),
    "SMASH":        (0,   0,   255),
    "VOLLEY":       (255, 165,  0),
    "LOFT/BANDEJA": (180,  0,  180),
    "ball":         (0,   255, 255),
    "player":       (255, 255,   0),
    "hud_bg":       (15,  15,   15),
    "hud_text":     (220, 220, 220),
}

SHOT_DISPLAY_FRAMES = 40   # how long to show shot label


class Visualizer:
    def __init__(self):
        self._recent_shots: deque = deque(maxlen=5)
        self._shot_timer: dict = {}   # shot_type → frames_remaining

    # ------------------------------------------------------------------
    def draw(self, frame, ball_pos, player_tracks, new_shots,
             live_stats, frame_idx, timestamp) -> np.ndarray:

        h, w = frame.shape[:2]

        # Register new shots
        for shot in new_shots:
            st = shot["shot_type"]
            self._recent_shots.appendleft(shot)
            self._shot_timer[st] = SHOT_DISPLAY_FRAMES

        # Decrement timers
        for k in list(self._shot_timer):
            self._shot_timer[k] -= 1
            if self._shot_timer[k] <= 0:
                del self._shot_timer[k]

        # ── Ball trajectory ──────────────────────────────────────────
        if ball_pos and ball_pos.get("trajectory"):
            pts = [(int(p["x"]), int(p["y"])) for p in ball_pos["trajectory"]]
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                color = tuple(int(c * alpha) for c in COLORS["ball"])
                cv2.line(frame, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)

        # ── Ball circle ──────────────────────────────────────────────
        if ball_pos:
            bx, by = int(ball_pos["center"][0]), int(ball_pos["center"][1])
            cv2.circle(frame, (bx, by), 10, COLORS["ball"], 2, cv2.LINE_AA)
            cv2.circle(frame, (bx, by),  4, (255, 255, 255), -1, cv2.LINE_AA)

            # Velocity arrow
            vx, vy = ball_pos.get("velocity", [0, 0])
            scale = 4
            ex, ey = int(bx + vx * scale), int(by + vy * scale)
            cv2.arrowedLine(frame, (bx, by), (ex, ey), COLORS["ball"], 2,
                            cv2.LINE_AA, tipLength=0.4)

        # ── Players ──────────────────────────────────────────────────
        for i, track in enumerate(player_tracks):
            x1, y1, x2, y2 = [int(v) for v in track["bbox"]]
            pid = track["id"]
            # Color based on court side
            side = track.get("player_side", "unknown")
            if side == "left_court":
                color = (0, 200, 255)   # Orange-ish
            elif side == "right_court":
                color = (255, 200, 0)   # Cyan-ish
            else:
                color = COLORS["player"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            label = f"P{pid}"
            if side != "unknown":
                label += f" [{side.split('_')[0]}]"
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        # ── Shot flash labels ────────────────────────────────────────
        y_offset = 80
        for shot_type, remaining in self._shot_timer.items():
            alpha = min(1.0, remaining / 15.0)
            color = COLORS.get(shot_type, (255, 255, 255))
            faded = tuple(int(c * alpha) for c in color)
            cv2.putText(frame, f"► {shot_type}", (w - 280, y_offset),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, faded, 2, cv2.LINE_AA)
            y_offset += 35

        # ── HUD ──────────────────────────────────────────────────────
        self._draw_hud(frame, live_stats, frame_idx, timestamp, w, h)

        return frame

    # ------------------------------------------------------------------
    def _draw_hud(self, frame, stats, frame_idx, timestamp, w, h):
        """Semi-transparent stats overlay in top-left corner."""
        hud_w, hud_h = 210, 160
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (8 + hud_w, 8 + hud_h),
                      COLORS["hud_bg"], -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        lines = [
            ("PADEL ANALYTICS", (255, 80, 80), 0.55),
            (f"Frame: {frame_idx}", COLORS["hud_text"], 0.45),
            (f"Time:  {timestamp:.1f}s", COLORS["hud_text"], 0.45),
            (f"Shots: {stats.get('total_shots', 0)}", (100, 255, 100), 0.48),
        ]
        for st, cnt in stats.get("shot_counts", {}).items():
            color = COLORS.get(st, COLORS["hud_text"])
            lines.append((f"  {st[:12]:<12}: {cnt}", color, 0.42))

        y = 28
        for text, color, scale in lines:
            cv2.putText(frame, text, (14, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            y += int(scale * 38 + 4)

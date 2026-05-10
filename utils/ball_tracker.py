"""
Ball Tracker
Maintains ball position history with minimal lag, computes velocity and direction.
Uses simple low-pass filtering for responsive yet stable tracking.
"""

import numpy as np
from collections import deque


class BallTracker:
    """
    Tracks the padel ball across frames with reduced lag.

    Key improvements:
    - Lower smoothing factor (0.3) for more responsive tracking
    - Velocity computed directly from recent raw positions (not heavily smoothed)
    - Grace period on detection loss (keeps last known pos with decaying confidence)
    """

    HISTORY_LEN = 20
    SMOOTH_ALPHA = 0.3          # EWA smoothing factor (lower = less lag)
    VELOCITY_WINDOW = 3         # Use last N positions for velocity
    MISSING_GRACE = 5           # Frames to keep ball when temporarily lost
    MIN_CONFIDENCE = 0.1

    def __init__(self):
        self.history: deque = deque(maxlen=self.HISTORY_LEN)
        self.smoothed: list | None = None
        self.velocity = [0.0, 0.0]
        self.bounce_count = 0
        self._prev_vy = 0.0
        self._missing_frames = 0

    # ------------------------------------------------------------------
    def update(self, ball_detections: list, frame=None) -> dict | None:
        """
        Parameters
        ----------
        ball_detections : list of detections with 'center' and 'conf'
        frame           : current frame (unused, reserved)

        Returns
        -------
        dict with ball state or None if ball is completely lost
        """
        current_conf = None

        if not ball_detections:
            self._missing_frames += 1
            if self._missing_frames > self.MISSING_GRACE or not self.history:
                return None
            # Use last known position with decaying confidence
            return self._build_output(current_conf, missing=True)

        self._missing_frames = 0
        det = max(ball_detections, key=lambda d: d["conf"])
        cx, cy = det["center"]
        current_conf = det["conf"]

        # Light EWA smoothing
        if self.smoothed is None:
            self.smoothed = [cx, cy]
        else:
            alpha = self.SMOOTH_ALPHA
            self.smoothed[0] = alpha * cx + (1 - alpha) * self.smoothed[0]
            self.smoothed[1] = alpha * cy + (1 - alpha) * self.smoothed[1]

        self.history.append({
            "x": self.smoothed[0],
            "y": self.smoothed[1],
            "raw_x": cx,
            "raw_y": cy,
            "conf": current_conf,
        })

        # Velocity from last few positions (prefer raw positions for responsiveness)
        if len(self.history) >= 2:
            recent = list(self.history)[-self.VELOCITY_WINDOW:]
            if len(recent) >= 2:
                xs = [p.get("raw_x", p["x"]) for p in recent]
                ys = [p.get("raw_y", p["y"]) for p in recent]
                self.velocity = [
                    (xs[-1] - xs[0]) / max(len(xs) - 1, 1),
                    (ys[-1] - ys[0]) / max(len(ys) - 1, 1),
                ]

        # Bounce detection
        bounced = False
        vy = self.velocity[1]
        if (self._prev_vy > 2 and vy < -2) or (self._prev_vy < -2 and vy > 2):
            bounced = True
            self.bounce_count += 1
        self._prev_vy = vy

        return self._build_output(current_conf, bounced)

    # ------------------------------------------------------------------
    def _build_output(self, conf, bounced=False, missing=False) -> dict | None:
        if self.smoothed is None:
            return None

        vx, vy = self.velocity
        speed = (vx**2 + vy**2) ** 0.5

        if abs(vx) > abs(vy) * 0.7:
            direction = "right" if vx > 0 else "left"
        else:
            direction = "down" if vy > 0 else "up"

        conf_val = conf if not missing else max(0.0, (conf or 0.3) - 0.2)
        if missing:
            conf_val = self.MIN_CONFIDENCE

        return {
            "center": list(self.smoothed),
            "velocity": self.velocity,
            "speed": round(speed, 2),
            "direction": direction,
            "confidence": round(conf_val, 3),
            "bounced": bounced,
            "trajectory": [{"x": p["x"], "y": p["y"]} for p in self.history],
        }

    # ------------------------------------------------------------------
    def get_trajectory(self) -> list:
        return [{"x": p["x"], "y": p["y"]} for p in self.history]

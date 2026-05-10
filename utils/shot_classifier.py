"""
Shot Classifier — Court-Aware Version
Classifies padel shots based on:
  - Ball trajectory (velocity, direction, height)
  - Player court position (left_court / right_court)
  - Relative ball-to-player position
  - Speed thresholds

Supported shot types:
  FOREHAND  | BACKHAND  | SMASH  | VOLLEY  | LOFT/BANDEJA
"""

import math
from collections import deque, Counter


# Configuration
SHOT_COOLDOWN_FRAMES = 15
MIN_SHOT_SPEED = 4.0
SMASH_SPEED_THRESHOLD = 18.0
VOLLEY_SPEED_THRESHOLD = 12.0
LOFT_SPEED_THRESHOLD = 10.0
# Vertical position ratio: 0=bottom, 1=top
# Smashes tend to be high in the frame
HIGH_Y_RATIO = 0.4
LOW_Y_RATIO = 0.7  # Near ground


class ShotClassifier:
    def __init__(self, frame_height=None):
        self._last_shot_frame: dict[int, int] = {}
        self._global_last_frame = -SHOT_COOLDOWN_FRAMES
        self._speed_history = deque(maxlen=5)
        self.frame_height = frame_height

        # Track player side assignments over time
        self._player_side_history: dict[int, deque] = {}  # id -> deque of court zones

    # ------------------------------------------------------------------
    def classify(self, ball_pos: dict | None, player_tracks: list,
                 frame_idx: int, timestamp: float) -> list:
        if ball_pos is None:
            return []

        speed = ball_pos.get("speed", 0.0)
        self._speed_history.append(speed)

        # Rising edge detection: speed above threshold and increasing
        if len(self._speed_history) >= 3:
            avg_prev = sum(list(self._speed_history)[-3:-1]) / 2
            speed_rising = speed >= MIN_SHOT_SPEED and speed > avg_prev * 1.1
        else:
            speed_rising = speed >= MIN_SHOT_SPEED

        if not speed_rising:
            return []

        # Cooldown
        if frame_idx - self._global_last_frame < SHOT_COOLDOWN_FRAMES:
            return []

        # Find hitter player
        player = self._find_hitter(ball_pos, player_tracks)
        if player is None:
            return []

        player_id = player["id"]
        player_side = player.get("player_side", self._infer_player_side(player))
        ball_cx, ball_cy = ball_pos["center"]

        # Per-player cooldown
        if player_id in self._last_shot_frame:
            if frame_idx - self._last_shot_frame[player_id] < 10:
                return []

        # Classify shot type
        shot_type = self._determine_shot_type(ball_pos, player_side, player, frame_idx)

        direction = ball_pos.get("direction", "unknown")
        confidence = min(0.95, 0.5 + speed / 40.0)

        shot = {
            "frame": frame_idx,
            "timestamp": round(timestamp, 3),
            "shot_type": shot_type,
            "player_id": player_id,
            "confidence": round(confidence, 3),
            "ball_x": round(ball_pos["center"][0], 1),
            "ball_y": round(ball_pos["center"][1], 1),
            "ball_speed": round(speed, 2),
            "direction": direction,
            "player_side": player_side,
        }

        self._global_last_frame = frame_idx
        self._last_shot_frame[player_id] = frame_idx

        return [shot]

    # ------------------------------------------------------------------
    def _find_hitter(self, ball_pos, player_tracks):
        """
        Identify which player is most likely hitting the ball.
        Uses:
        - Minimum distance to ball
        - Player confirmed (enough hits)
        - Player must be in a valid court zone
        """
        if not player_tracks:
            return None

        bx, by = ball_pos["center"]
        candidates = []

        for p in player_tracks:
            px, py = p["center"]
            dist = math.hypot(px - bx, py - by)

            # Score: distance + bonus for recent confirmation
            score = dist
            if p.get("hits", 0) >= 3:
                score *= 0.7  # Prefer confirmed tracks

            candidates.append((score, p))

        candidates.sort(key=lambda x: x[0])
        best_score = candidates[0][0] if candidates else float("inf")

        # Only accept if reasonably close
        if best_score < 200:  # pixels
            return candidates[0][1]
        return None

    # ------------------------------------------------------------------
    def _infer_player_side(self, player):
        """Infer court side from player's horizontal position."""
        if self.frame_height is None:
            return "unknown"
        cx = player["center"][0]
        return "left_court" if cx < self.frame_height / 2 else "right_court"

    # ------------------------------------------------------------------
    def _determine_shot_type(self, ball_pos, player_side, player, frame_idx):
        """
        Court-aware shot classification using padel heuristics.
        """
        speed = ball_pos.get("speed", 0.0)
        vx, vy = ball_pos.get("velocity", [0, 0])
        bx, by = ball_pos["center"]
        px, py = player["center"]

        # 1. SMASH: High speed + upward trajectory + ball high in frame
        if speed >= SMASH_SPEED_THRESHOLD and vy < -2:
            return "SMASH"

        # 2. VOLLEY: Medium-high speed, mostly horizontal, ball at mid-height
        if speed > VOLLEY_SPEED_THRESHOLD and abs(vx) > abs(vy) * 1.2:
            # Check if ball is not too low (not a bounce)
            if by < 0.75 * self.frame_height if self.frame_height else True:
                return "VOLLEY"

        # 3. LOFT/BANDEJA: Slow speed + upward trajectory + ball higher in frame
        if speed < LOFT_SPEED_THRESHOLD and vy < -1.5:
            if by < 0.5 * self.frame_height if self.frame_height else False:
                return "LOFT/BANDEJA"

        # 4. FOREHAND / BACKHAND based on court side and ball relative position
        # Player's hitting side: for a right-handed player (default assumption):
        #   - On left court: forehand side = left side of body
        #   - On right court: forehand side = right side of body
        # Ball relative position to player:
        rel_x = bx - px  # Positive = ball to right of player

        if player_side == "left_court":
            # Player on left side - assume right-handed, forehand on left side
            # Ball to left (rel_x < 0) → forehand; ball to right → backhand
            return "FOREHAND" if rel_x < 0 else "BACKHAND"
        elif player_side == "right_court":
            # Player on right side - assume forehand on right side
            # Ball to right (rel_x > 0) → forehand; ball to left → backhand
            return "FOREHAND" if rel_x > 0 else "BACKHAND"

        # Fallback: use ball direction relative to player
        return "FOREHAND" if (vx > 0 and player_side == "left_court") or (vx < 0 and player_side == "right_court") else "BACKHAND"

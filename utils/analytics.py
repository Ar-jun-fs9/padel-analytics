"""
Analytics Engine
Accumulates shot events and computes statistics.
"""

from collections import defaultdict


class AnalyticsEngine:
    def __init__(self):
        self.shot_counts: dict[str, int] = defaultdict(int)
        self.player_shot_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.bounce_count = 0
        self.total_frames = 0
        self.rally_lengths: list[int] = []
        self._rally_start_frame = 0
        self._in_rally = False
        self._last_shot_frame = -999

    # ------------------------------------------------------------------
    def update(self, new_shots: list, ball_pos, player_tracks, frame_idx: int):
        self.total_frames = frame_idx

        for shot in new_shots:
            shot_type = shot["shot_type"]
            player_id = shot.get("player_id")
            self.shot_counts[shot_type] += 1
            if player_id is not None:
                self.player_shot_counts[player_id][shot_type] += 1

            # Rally tracking
            if frame_idx - self._last_shot_frame < 90:   # 3s gap → new rally
                if not self._in_rally:
                    self._rally_start_frame = self._last_shot_frame
                    self._in_rally = True
            else:
                if self._in_rally:
                    length = self._last_shot_frame - self._rally_start_frame
                    self.rally_lengths.append(length)
                self._in_rally = False
            self._last_shot_frame = frame_idx

        if ball_pos and ball_pos.get("bounced"):
            self.bounce_count += 1

    # ------------------------------------------------------------------
    def get_live_stats(self) -> dict:
        total = sum(self.shot_counts.values())
        return {
            "total_shots": total,
            "shot_counts": dict(self.shot_counts),
            "bounces": self.bounce_count,
        }

    # ------------------------------------------------------------------
    def get_summary(self) -> dict:
        total = sum(self.shot_counts.values())
        pct = {k: round(v / total * 100, 1) if total else 0
               for k, v in self.shot_counts.items()}

        avg_rally = (
            round(sum(self.rally_lengths) / len(self.rally_lengths), 1)
            if self.rally_lengths else 0
        )

        player_stats = {}
        for pid, counts in self.player_shot_counts.items():
            player_stats[f"player_{pid}"] = {
                "shot_counts": dict(counts),
                "total": sum(counts.values()),
            }

        return {
            "total_shots": total,
            "shot_counts": dict(self.shot_counts),
            "shot_percentages": pct,
            "total_bounces": self.bounce_count,
            "total_frames_processed": self.total_frames,
            "num_rallies": len(self.rally_lengths),
            "avg_rally_length_frames": avg_rally,
            "player_stats": player_stats,
        }

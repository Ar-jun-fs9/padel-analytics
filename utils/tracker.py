"""
Multi-Object Tracker (lightweight SORT-style with stable IDs)
Maintains consistent player IDs across frames using IoU + center-distance hybrid matching.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque, Counter


def iou(box_a, box_b):
    """Intersection-over-Union between two [x1,y1,x2,y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center(box):
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]


class Track:
    _next_id = 1

    def __init__(self, bbox, conf):
        self.id = Track._next_id
        Track._next_id += 1
        self.bbox = bbox
        self.conf = conf
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.center_history = [center(bbox)]
        self.predicted_center = center(bbox)

    def update(self, bbox, conf):
        self.bbox = bbox
        self.conf = conf
        self.hits += 1
        self.time_since_update = 0
        self.center_history.append(center(bbox))
        if len(self.center_history) > 30:
            self.center_history.pop(0)
        self.predicted_center = center(bbox)

    def predict(self):
        """Linear prediction of next position from recent history."""
        self.age += 1
        self.time_since_update += 1
        if len(self.center_history) >= 3:
            # Use last 3 points for velocity
            dx = self.center_history[-1][0] - self.center_history[-3][0]
            dy = self.center_history[-1][1] - self.center_history[-3][1]
            self.predicted_center = (
                self.center_history[-1][0] + dx / 2,
                self.center_history[-1][1] + dy / 2,
            )
        return self.predicted_center

    def is_confirmed(self):
        return self.hits >= self.min_hits()

    @staticmethod
    def min_hits():
        return 1  # Lower threshold for faster initialization

    def to_dict(self):
        cx, cy = center(self.bbox)
        return {
            "id": self.id,
            "bbox": self.bbox,
            "center": [cx, cy],
            "confidence": self.conf,
            "hits": self.hits,
            "confirmed": self.is_confirmed(),
        }


class MultiObjectTracker:
    """
    SORT-style tracker with:
    - Hybrid IoU + center-distance matching
    - Stable ID assignment (low confirmation threshold)
    - Court zone filtering (for CCTV-style fixed cameras)
    - Player side assignment (forehand/backhand court zones)
    """

    def __init__(
        self,
        max_age=30,
        min_hits=1,
        iou_threshold=0.25,
        dist_threshold=150,
        use_court_zones=True,
        frame_width=None,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.dist_threshold = dist_threshold
        self.use_court_zones = use_court_zones
        self.frame_width = frame_width  # Set on first update
        self.tracks: list[Track] = []
        self._next_zone_id = 1  # For assigning sides

    def _get_court_zone(self, cx, cy, frame_width, frame_height):
        """
        Determine court zone based on position.
        Returns: 'left_court' or 'right_court' (the two sides of the padel court)
        Players on the same side as the net-perspective:
        - In padel, court is divided by a center line perpendicular to the net
        - From CCTV view, typically left half = one side, right half = other
        """
        # Use left/right thirds to allow for camera perspective
        left_bound = frame_width * 0.35
        right_bound = frame_width * 0.65

        if cx < left_bound:
            return "left_court"
        elif cx > right_bound:
            return "right_court"
        else:
            # Middle zone - could be net area; assign by nearest boundary
            if cx < frame_width / 2:
                return "left_court"
            else:
                return "right_court"

    def _filter_detections_by_court(self, detections, frame_width, frame_height):
        """
        Keep only detections in valid court playing areas.
        Filters out background spectators, officials, etc.
        Ensures each detection has a 'center' field.
        """
        filtered = []
        for det in detections:
            # Compute center if not present
            if "center" not in det and "bbox" in det:
                x1, y1, x2, y2 = det["bbox"]
                det["center"] = [(x1 + x2) / 2, (y1 + y2) / 2]

            cx, cy = det["center"]
            zone = self._get_court_zone(cx, cy, frame_width, frame_height)

            # Accept both court zones, but must be within vertical bounds
            if 0.10 * frame_height < cy < 0.95 * frame_height:
                det["court_zone"] = zone
                filtered.append(det)
        return filtered

    def _assign_player_side_labels(self, tracks):
        """
        Assign each track a 'side' label (forehand/backhand) based on
        consistent court zone membership over multiple frames.
        """
        # Track which zone each ID has been in recently
        if not hasattr(self, "_zone_history"):
            self._zone_history = {}  # id -> deque of recent zones

        # Update zone assignments
        zone_counts = {}
        for track in tracks:
            tid = track["id"]
            zone = track.get("court_zone")
            if zone is None:
                continue
            if tid not in self._zone_history:
                from collections import deque
                self._zone_history[tid] = deque(maxlen=10)
            self._zone_history[tid].append(zone)

            # Majority vote for this player's side
            recent = list(self._zone_history[tid])
            if len(recent) >= 3:
                from collections import Counter
                side = Counter(recent).most_common(1)[0][0]
                track["player_side"] = side
                zone_counts[side] = zone_counts.get(side, 0) + 1

        return tracks

    def update(self, detections: list, frame_shape=None) -> list:
        """
        detections : list of dicts with 'bbox', 'conf', and optional 'center'
        frame_shape: (height, width) tuple for court zone heuristics
        Returns    : list of active track dicts (confirmed tracks)
        """
        h, w = frame_shape if frame_shape else (None, None)
        if self.use_court_zones and w:
            self.frame_width = w

        # Predict next positions
        for t in self.tracks:
            t.predict()

        if not detections:
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return self._confirmed_tracks()

        # Court zone filtering (only when we know frame dims)
        if self.use_court_zones and w and h:
            detections = self._filter_detections_by_court(detections, w, h)

        det_boxes = [d["bbox"] for d in detections]
        det_confs = [d["conf"] for d in detections]

        if not self.tracks:
            for bbox, conf in zip(det_boxes, det_confs):
                self.tracks.append(Track(bbox, conf))
            confirmed = self._confirmed_tracks()
            return self._assign_player_side_labels(confirmed)

        # Hybrid cost: combine IoU and center-distance
        num_tracks = len(self.tracks)
        num_dets = len(det_boxes)
        cost_matrix = np.zeros((num_tracks, num_dets))

        for ti, track in enumerate(self.tracks):
            pred_cx, pred_cy = track.predicted_center
            for di, dbox in enumerate(det_boxes):
                iou_val = iou(track.bbox, dbox)
                iou_cost = 1.0 - iou_val
                det_cx, det_cy = center(dbox)
                dist = np.hypot(det_cx - pred_cx, det_cy - pred_cy)
                norm_dist = dist / self.dist_threshold
                dist_cost = min(norm_dist, 1.0)
                cost_matrix[ti, di] = 0.3 * iou_cost + 0.7 * dist_cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_tracks = set()
        matched_dets = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 0.6:
                self.tracks[r].update(det_boxes[c], det_confs[c])
                matched_tracks.add(r)
                matched_dets.add(c)

        # Unmatched detections → new tracks
        for di, (bbox, conf) in enumerate(zip(det_boxes, det_confs)):
            if di not in matched_dets:
                self.tracks.append(Track(bbox, conf))

        # Remove stale tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        confirmed = self._confirmed_tracks()
        return self._assign_player_side_labels(confirmed)

    def _confirmed_tracks(self) -> list:
        """Return all tracks that have been confirmed (met min_hits threshold)."""
        return [
            t.to_dict()
            for t in self.tracks
            if t.hits >= self.min_hits
        ]

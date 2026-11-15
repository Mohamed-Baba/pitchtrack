"""Multi-object tracking by detection.

One step per frame: predict every track to the frame time (Kalman), compensate the
camera motion, associate tracks and detections in two stages (high-score
detections first, then low-score ones, ByteTrack), update the life cycle of the tracks
(tentative -> confirmed -> lost -> removed), output the confirmed tracks.

A track id is local to a clip: it is never reused inside a clip, and it says
nothing about who the player is.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np

from .assignment import BIG, linear_assignment
from .boxes import box_iou, xyxy_to_cxcywh
from .camera_motion import warp_box, warp_track_state
from .detector import Detections
from .kalman import CHI2_4_95, KalmanBoxFilter, box_from_mean

TENTATIVE, CONFIRMED, LOST, REMOVED = "tentative", "confirmed", "lost", "removed"


@dataclass
class TrackerConfig:
    motion: str = "kalman"          # "kalman", or "none": compare with the last box only
    two_stage: bool = True          # ByteTrack: second association with low-score detections
    high_thresh: float = 0.5        # first-stage detections (the operating point of the detector)
    low_thresh: float = 0.1         # second-stage detections, between low and high
    new_track_thresh: float = 0.6   # a new track needs a confident detection
    iou_gate_high: float = 0.2      # stage 1: pairs with IoU < 0.2 are never matched
    iou_gate_low: float = 0.5       # stage 2: stricter, low-score detections are less reliable
    maha_gate: float | None = CHI2_4_95   # None disables the Mahalanobis gate
    n_init: int = 3                 # hits needed to confirm a track
    max_lost_s: float = 1.0         # a lost track is removed after this time
    output: str = "filtered"        # "filtered" (Kalman mean) or "detection" (matched box)

    @staticmethod
    def from_dict(d: dict | None) -> "TrackerConfig":
        d = d or {}
        known = {f.name for f in fields(TrackerConfig)}
        return TrackerConfig(**{k: v for k, v in d.items() if k in known})


@dataclass
class Track:
    track_id: int
    mean: np.ndarray            # (8,) Kalman state (cx, cy, w, h, velocities)
    cov: np.ndarray             # (8, 8)
    last_box: np.ndarray        # xyxy of the last matched detection
    last_t: float               # time of the last match, seconds
    score: float
    state: str = TENTATIVE
    hits: int = 1
    updated: bool = True        # matched at the current frame

    def box(self, use_motion: bool = True) -> np.ndarray:
        return box_from_mean(self.mean) if use_motion else self.last_box.copy()


@dataclass
class Tracker:
    config: TrackerConfig = field(default_factory=TrackerConfig)
    kf: KalmanBoxFilter = field(default_factory=KalmanBoxFilter)

    def __post_init__(self):
        if isinstance(self.config, dict):
            self.config = TrackerConfig.from_dict(self.config)
        self.tracks: list[Track] = []
        self.next_id = 1
        self.t_prev: float | None = None

    # ------------------------------------------------------------------- prediction and motion
    @property
    def use_motion(self) -> bool:
        return self.config.motion == "kalman"

    def reset(self) -> None:
        """Forget every track (scene cut). Ids keep increasing: they are never reused."""
        self.tracks = []

    def predict(self, t: float) -> None:
        dt = 0.0 if self.t_prev is None else max(0.0, t - self.t_prev)
        self.t_prev = t
        for tr in self.tracks:
            tr.updated = False
            if self.use_motion and dt > 0:
                tr.mean, tr.cov = self.kf.predict(tr.mean, tr.cov, dt)

    def apply_camera_motion(self, A) -> None:
        """Express every track in the coordinates of the new frame (A: 2 x 3 affine map)."""
        for tr in self.tracks:
            tr.mean, tr.cov = warp_track_state(tr.mean, tr.cov, A)
            tr.last_box = warp_box(tr.last_box, A)

    def cost_matrix(self, track_idx, det_boxes) -> np.ndarray:
        """1 - IoU between predicted track boxes and detections; BIG outside the Mahalanobis gate."""
        if len(track_idx) == 0 or len(det_boxes) == 0:
            return np.zeros((len(track_idx), len(det_boxes)))
        boxes = np.array([self.tracks[i].box(self.use_motion) for i in track_idx])
        cost = 1.0 - box_iou(boxes, det_boxes)
        if self.use_motion and self.config.maha_gate is not None:
            z = xyxy_to_cxcywh(det_boxes)
            for row, i in enumerate(track_idx):
                d2 = self.kf.gating_distance(self.tracks[i].mean, self.tracks[i].cov, z)
                cost[row, d2 > self.config.maha_gate] = BIG
        return cost

    # ------------------------------------------------------------------- association
    def associate(self, dets: Detections):
        """Two-stage association (ByteTrack). Returns (matches, unmatched_tracks, unmatched_high).

        Stage 1: every live track (tentative, confirmed, lost) against the detections with
        score >= high_thresh, cost 1 - IoU, gate max_cost = 1 - iou_gate_high.
        Stage 2 (if two_stage): the tracks still unmatched AND confirmed (not lost, not
        tentative) against the detections with low_thresh <= score < high_thresh, gate
        max_cost = 1 - iou_gate_low. Low-score detections never start a track.
        matches: list of (track index in self.tracks, detection index in dets);
        unmatched_tracks: track indices; unmatched_high: indices of high-score detections.
        """
        cfg = self.config
        high = np.flatnonzero(dets.scores >= cfg.high_thresh)
        low = np.flatnonzero((dets.scores >= cfg.low_thresh) & (dets.scores < cfg.high_thresh))
        live = [i for i, tr in enumerate(self.tracks) if tr.state != REMOVED]

        # Stage 1: associate all live tracks with high-score detections.
        cost_high = self.cost_matrix(live, dets.boxes[high])

        matches_high, unmatched_rows, unmatched_cols = linear_assignment(
            cost_high,
            max_cost=1.0 - cfg.iou_gate_high,
        )

        matches = [
            (live[row], high[col])
            for row, col in matches_high
        ]

        # Convert unmatched row indices back to track indices.
        unmatched_after_high = [
            live[row] for row in unmatched_rows
        ]

        # Convert unmatched high detection indices back to dets indices.
        unmatched_high = [
            high[col] for col in unmatched_cols
        ]

        # Stage 2: only unmatched confirmed tracks can use low-score detections.
        confirmed_unmatched = [
            i for i in unmatched_after_high
            if self.tracks[i].state == CONFIRMED
        ]

        if cfg.two_stage and len(confirmed_unmatched) > 0 and len(low) > 0:
            cost_low = self.cost_matrix(
                confirmed_unmatched,
                dets.boxes[low],
            )

            matches_low, unmatched_rows_low, _ = linear_assignment(
                cost_low,
                max_cost=1.0 - cfg.iou_gate_low,
            )

            matches.extend([
                (confirmed_unmatched[row], low[col])
                for row, col in matches_low
            ])

            matched_low_tracks = {
                confirmed_unmatched[row]
                for row, _ in matches_low
            }

            unmatched_tracks = [
                i for i in unmatched_after_high
                if i not in matched_low_tracks
            ]
        else:
            unmatched_tracks = unmatched_after_high

        return matches, unmatched_tracks, unmatched_high


        
    # ------------------------------------------------------------------- life cycle
    def update_life_cycle(self, dets: Detections, matches, unmatched_tracks, unmatched_high,
                          t: float) -> None:
        """Apply the matches and the life-cycle rules.

        Matched track: Kalman update with the detection (or copy it if motion is "none"),
        hits += 1, last_t = t, score, updated = True; a tentative track becomes confirmed when
        hits >= n_init, a lost track becomes confirmed again.
        Unmatched track: tentative -> removed; confirmed -> lost; lost -> removed once
        t - last_t > max_lost_s.
        Unmatched high-score detection with score >= new_track_thresh: new track with id
        self.next_id (then increment it), tentative (confirmed at once if n_init <= 1).
        Finally drop the removed tracks.
        """
        cfg = self.config
        
        # 1. Update matched tracks.
        for track_idx, det_idx in matches:
            tr = self.tracks[track_idx]
            box = dets.boxes[det_idx]
            z = xyxy_to_cxcywh(box[None, :])[0]

            if self.use_motion:
                tr.mean, tr.cov = self.kf.update(
                    tr.mean, tr.cov, z
                )

            tr.last_box = box.copy()
            tr.hits += 1
            tr.last_t = t
            tr.score = float(dets.scores[det_idx])
            tr.updated = True

            if tr.state == TENTATIVE and tr.hits >= cfg.n_init:
                tr.state = CONFIRMED
            elif tr.state == LOST:
                tr.state = CONFIRMED

        # 2. Handle unmatched tracks.
        for track_idx in unmatched_tracks:
            tr = self.tracks[track_idx]

            if tr.state == TENTATIVE:
                tr.state = REMOVED

            elif tr.state == CONFIRMED:
                tr.state = LOST

            elif tr.state == LOST:
                if t - tr.last_t > cfg.max_lost_s:
                    tr.state = REMOVED

        # 3. Create new tracks from unmatched high-score detections.
        for det_idx in unmatched_high:
            score = float(dets.scores[det_idx])

            if score < cfg.new_track_thresh:
                continue

            box = dets.boxes[det_idx]
            z = xyxy_to_cxcywh(box[None, :])[0]

            mean, cov = self.kf.initiate(z)

            state = TENTATIVE
            if cfg.n_init <= 1:
                state = CONFIRMED

            new_track = Track(
                track_id=self.next_id,
                mean=mean,
                cov=cov,
                last_box=box.copy(),
                last_t=t,
                score=score,
                state=state,
                hits=1,
                updated=True,
            )

            self.tracks.append(new_track)
            self.next_id += 1

        # 4. Remove tracks whose state is REMOVED.
        self.tracks = [
            tr for tr in self.tracks
            if tr.state != REMOVED
        ]


    # ------------------------------------------------------------------- one frame
    def update(self, dets: Detections, t: float, camera_motion=None):
        """One frame. Returns (ids (K,), boxes (K, 4), scores (K,)) of the confirmed tracks
        matched at this frame."""
        self.predict(t)
        if camera_motion is not None:
            self.apply_camera_motion(camera_motion)
        matches, unmatched_tracks, unmatched_high = self.associate(dets)
        self.update_life_cycle(dets, matches, unmatched_tracks, unmatched_high, t)
        out = [tr for tr in self.tracks if tr.state == CONFIRMED and tr.updated]
        if not out:
            return np.zeros(0, dtype=np.int64), np.zeros((0, 4)), np.zeros(0)
        boxes = np.array([tr.box(self.use_motion) if self.config.output == "filtered" else tr.last_box
                          for tr in out])
        return (np.array([tr.track_id for tr in out], dtype=np.int64), boxes,
                np.array([tr.score for tr in out]))


def track_frames(tracker: Tracker, frames, camera_motion=None):
    """Run a tracker over [(frame, t, Detections)], frames in time order.

    camera_motion: optional dict {frame: (A, ok)}; ok=False means a scene cut: reset.
    Yields (frame, t, ids, boxes, scores).
    """
    for frame, t, dets in frames:
        A = None
        if camera_motion is not None and frame in camera_motion:
            A, ok = camera_motion[frame]
            if not ok:
                tracker.reset()
                A = None
        ids, boxes, scores = tracker.update(dets, t, A)
        yield frame, t, ids, boxes, scores


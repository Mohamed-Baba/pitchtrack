"""Analysis of any football video, one frame at a time.

    analyzer = VideoAnalyzer(detector, tracker, frame_size, annotator=LineAnnotator(...), ...)
    for frame in VideoFileSource("match.mp4"):
        result = analyzer.process(frame)       # boxes, track ids, teams, homography, positions
    report = analyzer.finish()                 # teams refitted on the whole video, statistics
    report.save("outputs/match")

Per frame, in this order: detection; camera motion between the previous frame and this one
(a cut resets the tracker); tracking; pitch lines predicted by the line network and calibration
(calibrate_frame), or, if that fails, the homography of the previous frame moved with the image
motion and checked against the predicted lines; team colours; positions on the pitch.

`process` only uses the past, so it can drive a live display: its team labels are a running
estimate (k-means fitted once enough shirt colours are collected). `finish` uses the whole video:
k-means is refitted on every colour, the labels are aligned with the live ones, and the
statistics of each track are computed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import pitch_model as pm
from .boxes import foot_points
from .pitch_calibration import (DEFAULT_QUALITY, ImageMotion, calibrate_frame, image_to_pitch, line_residuals,
                                parse_lines, propagate)
from .pitch_mask import foot_distance_to_pitch, pitch_mask
from .statistics import team_summary, track_table
from .team_classifier import UNKNOWN, assign_teams, fit_team_colours, torso_colour
from .tracker import Tracker
from .video_io import Frame

POSITION_COLUMNS = ["frame", "t", "track_id", "x1", "y1", "x2", "y2", "score", "X", "Y"]
CALIBRATION_COLUMNS = ["frame", "t", "ok", "source", "reason"] + [f"g{i}" for i in range(9)]


@dataclass
class FrameResult:
    """What is known at one frame. Arrays have one row per confirmed track seen at this frame."""

    index: int
    t: float
    ids: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    teams: np.ndarray            # 0, 1 or UNKNOWN: live estimate
    xy: np.ndarray               # (K, 2) metres, NaN where unknown
    G: np.ndarray | None         # image -> pitch homography, None if not calibrated
    calibration: str             # "lines", "propagated" or "none"
    reason: str = ""             # why the frame is not calibrated
    lines: dict | None = None    # pitch lines predicted at this frame (if the network ran)
    n_detections: int = 0


class OnlineTeams:
    """Team colours learned on the fly.

    Every `stride` frames, the torso colour of each tracked box taller than min_height is stored
    with its track id. Once min_samples valid colours exist, k-means (fit_team_colours) is fitted,
    and refitted every refit_every new colours; a track is named by assign_teams (median of its
    own colours, margin and distance rules, UNKNOWN if not sure).
    """

    def __init__(self, cfg: dict, min_samples: int = 150, refit_every: int = 300):
        self.cfg, self.min_samples, self.refit_every = cfg, min_samples, refit_every
        self.feats, self.valid = [], []
        self.rows_of: dict = {}                # track id -> indices of its colours
        self.model = None                      # (centres, team_clusters)
        self._fitted_at, self._team_of, self._dirty = 0, {}, set()

    def observe(self, image, frame_index: int, ids, boxes) -> None:
        c = self.cfg
        if frame_index % c["stride"] != 0 or len(ids) == 0:
            return
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        tall = (boxes[:, 3] - boxes[:, 1]) >= c["min_height"]
        if not tall.any():
            return
        feats, valid = torso_colour(image, boxes[tall], c["min_pixels"], c["hsv_low"], c["hsv_high"])
        for tid, f, v in zip(np.asarray(ids)[tall], feats, valid):
            self.rows_of.setdefault(int(tid), []).append(len(self.feats))
            self.feats.append(f)
            self.valid.append(bool(v))
            self._dirty.add(int(tid))
        n_valid = int(np.sum(self.valid))
        if n_valid >= self.min_samples and (self.model is None or n_valid - self._fitted_at >= self.refit_every):
            self._fit()

    def _fit(self) -> None:
        feats, valid = np.asarray(self.feats), np.asarray(self.valid, dtype=bool)
        centres, teams, _ = fit_team_colours(feats[valid], self.cfg["k"], self.cfg["l_weight"])
        if self.model is not None:                          # keep the meaning of team 0 / team 1
            teams = _aligned(self.model[0][self.model[1]], centres, teams)
        self.model, self._fitted_at = (centres, teams), int(valid.sum())
        self._dirty = set(self.rows_of)                     # every track is named again

    def _name(self, track_ids) -> None:
        c = self.cfg
        centres, teams = self.model
        for tid in track_ids:
            rows = self.rows_of[tid]
            f = np.asarray([self.feats[i] for i in rows])
            v = np.asarray([self.valid[i] for i in rows], dtype=bool)
            self._team_of[tid] = int(assign_teams(f, v, np.full(len(rows), tid), centres, teams, c["margin"],
                                                  c["max_dist"], c["min_frames"], c["l_weight"], level="track")[0])

    def teams_of(self, ids) -> np.ndarray:
        """Current team of each track id (UNKNOWN before the first fit)."""
        if self.model is not None and self._dirty:
            self._name(self._dirty)
            self._dirty = set()
        return np.array([self._team_of.get(int(i), UNKNOWN) for i in np.asarray(ids)], dtype=np.int64)

    def final(self) -> dict:
        """Teams of every track from k-means refitted on all the colours of the video, with the
        labels 0 and 1 aligned with the live ones."""
        if int(np.sum(self.valid)) < max(self.cfg["k"], 10):
            return {}
        self._fit()
        self._name(list(self.rows_of))
        self._dirty = set()
        return dict(self._team_of)


def _aligned(ref_team_centres, centres, teams) -> np.ndarray:
    """teams (2,) cluster ids, swapped if that brings their centres closer to the reference."""
    a = np.linalg.norm(ref_team_centres - centres[teams], axis=1).sum()
    b = np.linalg.norm(ref_team_centres - centres[teams[::-1]], axis=1).sum()
    return teams if a <= b else teams[::-1].copy()


@dataclass
class AnalysisReport:
    positions: pd.DataFrame          # one row per tracked box: POSITION_COLUMNS + team
    calibration: pd.DataFrame        # one row per frame: CALIBRATION_COLUMNS
    players: pd.DataFrame            # one row per track: statistics.track_table
    teams: pd.DataFrame              # one row per team: statistics.team_summary
    info: dict = field(default_factory=dict)

    def team_of(self) -> dict:
        return self.positions.groupby("track_id")["team"].first().astype(int).to_dict() if len(self.positions) else {}

    def save(self, out_dir) -> Path:
        """positions.parquet, calibration.parquet, player_stats.csv, team_stats.csv, report.json,
        summary.md and heatmaps.png in out_dir."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.positions.to_parquet(out / "positions.parquet", index=False)
        self.calibration.to_parquet(out / "calibration.parquet", index=False)
        self.players.to_csv(out / "player_stats.csv", index=False, float_format="%.3f")
        self.teams.to_csv(out / "team_stats.csv", index=False, float_format="%.3f")
        (out / "report.json").write_text(json.dumps(self.info, indent=2, default=str), encoding="utf-8")
        (out / "summary.md").write_text(self.summary_markdown(), encoding="utf-8")
        self.save_heatmaps(out / "heatmaps.png")
        return out

    def summary_markdown(self) -> str:
        i, cal = self.info, self.calibration
        share = cal["ok"].mean() if len(cal) else 0.0
        lines = [f"# {i.get('video', 'video')}", "",
                 f"{i.get('n_frames', 0)} frames, {i.get('duration_s', 0.0):.1f} s; calibrated frames "
                 f"{share:.0%} ({(cal['source'] == 'lines').mean() if len(cal) else 0:.0%} from their own lines); "
                 f"{len(self.players)} tracks, {int(self.players['reliable'].sum()) if len(self.players) else 0} "
                 "with reliable statistics.", "",
                 "| team | tracks | reliable | distance of reliable tracks (m) | median m/min | top speed (m/s) |",
                 "|---|---|---|---|---|---|"]
        for _, r in self.teams.iterrows():
            lines.append(f"| {r['team_name']} | {r['tracks']} | {r['reliable_tracks']} | {r['distance_m']:.0f} | "
                         f"{r['median_m_per_min']:.1f} | {r['top_speed']:.2f} |")
        rel = self.players[self.players["reliable"]] if len(self.players) else self.players
        if len(rel):
            lines += ["", "Fastest reliable tracks:", "", "| track | team | distance (m) | max speed (m/s) |",
                      "|---|---|---|---|"]
            for _, r in rel.sort_values("max_speed", ascending=False).head(5).iterrows():
                lines.append(f"| {int(r['track_id'])} | {int(r['team'])} | {r['distance_m']:.1f} | "
                             f"{r['max_speed']:.2f} |")
        lines += ["", "Distances count reliable segments only (no interpolation across gaps); "
                      "positions are the bottom-middle of the boxes projected with the homography of each frame.",
                  "", "![heatmaps](heatmaps.png)", ""]
        return "\n".join(lines)

    def save_heatmaps(self, path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from .statistics import heatmap
        from .visualization import plot_pitch

        pos = self.positions
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
        for ax, (team, label) in zip(axes, ((0, "team 0"), (1, "team 1"), (UNKNOWN, "unknown"))):
            sel = pos.loc[pos["team"] == team, ["X", "Y"]].to_numpy() if len(pos) else np.zeros((0, 2))
            ax.imshow(heatmap(sel).T, extent=(-pm.HALF_L, pm.HALF_L, pm.HALF_W, -pm.HALF_W), cmap="magma",
                      interpolation="nearest")
            plot_pitch(ax)
            ax.set_title(f"{label}: {int(np.isfinite(sel[:, 0]).sum())} positions")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)


class VideoAnalyzer:
    """Detection, tracking, teams, calibration and positions, frame by frame (see the module doc).

    detector: list of BGR frames -> list of Detections (detector.build_detector);
    tracker: a Tracker; frame_size: (width, height);
    annotator: frames -> pitch-line annotations (line_model.LineAnnotator) or None (no calibration);
    cmc: camera motion for the tracker (camera_motion.CameraMotionEstimator) or None;
    calib_cfg, teams_cfg, stats_cfg: the calibration, teams and statistics sections of
    configs/analysis.yaml; min_score: detections below are dropped (the tracker's low threshold);
    pitch_radius: drop detections whose foot is farther than this from the grass (pixels), None to
    keep all; calib_every: run the line network every n frames (propagation in between).
    """

    def __init__(self, detector, tracker: Tracker, frame_size, annotator=None, cmc=None,
                 calib_cfg: dict | None = None, teams_cfg: dict | None = None, stats_cfg: dict | None = None,
                 min_score: float = 0.1, pitch_radius: float | None = None, pitch_mask_cfg: dict | None = None,
                 calib_every: int = 1, keep_lines: bool = False):
        self.detector, self.tracker, self.annotator, self.cmc = detector, tracker, annotator, cmc
        self.size = (int(frame_size[0]), int(frame_size[1]))
        self.calib_cfg = {**DEFAULT_QUALITY, "propagate": True, "motion_scale": 0.5, **(calib_cfg or {})}
        self.stats_cfg = {"margin_m": 5.0, "max_gap_s": 0.5, "smooth_s": 0.4, "step_s": 0.2, "max_speed": 12.0,
                          "min_duration_s": 2.0, **(stats_cfg or {})}
        teams_default = {"stride": 5, "min_height": 40, "k": 3, "l_weight": 0.5, "margin": 0.15, "max_dist": 25.0,
                         "min_frames": 3, "min_pixels": 20, "hsv_low": [35, 40, 40], "hsv_high": [85, 255, 255]}
        self.teams = OnlineTeams({**teams_default, **(teams_cfg or {})})
        self.min_score, self.pitch_radius = float(min_score), pitch_radius
        self.pitch_mask_cfg = pitch_mask_cfg or {}
        self.calib_every, self.keep_lines = max(1, int(calib_every)), keep_lines
        self.motion = ImageMotion(scale=self.calib_cfg["motion_scale"])
        self._G_prev, self._anchor_t, self._last_net = None, None, None
        self._pos_rows, self._cal_rows, self.n_frames, self.t_last = [], [], 0, 0.0

    # ------------------------------------------------------------------- per frame
    def process(self, frame: Frame) -> FrameResult:
        image, t = frame.image, float(frame.t)
        dets = self.detector([image])[0]
        dets = dets.select(dets.scores >= self.min_score)
        if self.pitch_radius is not None and len(dets):
            pmc = self.pitch_mask_cfg
            mask = pitch_mask(image, pmc.get("hsv_low", (35, 40, 40)), pmc.get("hsv_high", (85, 255, 255)),
                              pmc.get("close_px", 25), pmc.get("scale", 0.5))
            dets = dets.select(foot_distance_to_pitch(dets.boxes, mask) <= self.pitch_radius)

        A = None
        if self.cmc is not None:
            A, ok = self.cmc(image, dets.boxes)
            if not ok:                                        # scene cut: start the tracks again
                self.tracker.reset()
                A = None
        ids, boxes, scores = self.tracker.update(dets, t, A)

        G, source, reason, lines = self._calibrate(image, frame.index, t, boxes)
        self.teams.observe(image, frame.index, ids, boxes)
        teams = self.teams.teams_of(ids)
        xy = np.full((len(ids), 2), np.nan)
        if G is not None and len(ids):
            xy, _ = image_to_pitch(G, foot_points(boxes), self.stats_cfg["margin_m"])

        for i, b, s, p in zip(ids, boxes, scores, xy):
            self._pos_rows.append((frame.index, t, int(i), *b, float(s), *p))
        g = G.reshape(-1) if G is not None else np.full(9, np.nan)
        self._cal_rows.append((frame.index, t, G is not None, source, reason, *g))
        self.n_frames += 1
        self.t_last = t
        return FrameResult(frame.index, t, ids, boxes, scores, teams, xy, G, source, reason,
                           lines if self.keep_lines else None, len(dets))

    def _calibrate(self, image, index: int, t: float, boxes):
        """(G, source, reason, predicted lines) for the current frame."""
        cfg = self.calib_cfg
        M, motion_ok = self.motion(image, boxes)
        lines, cal = None, None
        run_net = self.annotator is not None and (self._G_prev is None or self._last_net is None
                                                  or index - self._last_net >= self.calib_every)
        if run_net:
            lines = self.annotator([image])[0]
            self._last_net = index
            cal = calibrate_frame(lines, self.size, cfg, seed=index)
            if cal.ok:
                self._G_prev, self._anchor_t = cal.G, t
                return cal.G, "lines", "", lines
        reason = cal.reason if cal is not None else ("no line model" if self.annotator is None else "")
        if (self._G_prev is not None and motion_ok and cfg.get("propagate", True)
                and t - self._anchor_t <= cfg["propagate_max_s"]):
            G = propagate(self._G_prev, M)
            lp = parse_lines(lines, *self.size) if lines else None
            if lp is None or len(lp.pts) == 0 or \
                    np.median(line_residuals(G, lp.pts, lp.lines)) <= cfg["propagated_max_median_px"]:
                self._G_prev = G
                return G, "propagated", "", lines
            reason = (reason + "; " if reason else "") + "moved homography does not fit"
        elif self._G_prev is not None:
            reason = (reason + "; " if reason else "") + ("camera cut" if not motion_ok else "too far from the "
                                                          "last calibrated frame")
        self._G_prev = None
        return None, "none", reason or "no calibration", lines

    # ------------------------------------------------------------------- whole video
    def finish(self, info: dict | None = None) -> AnalysisReport:
        pos = pd.DataFrame(self._pos_rows, columns=POSITION_COLUMNS)
        cal = pd.DataFrame(self._cal_rows, columns=CALIBRATION_COLUMNS)
        team_of = self.teams.final()
        pos["team"] = pos["track_id"].map(team_of).fillna(UNKNOWN).astype(int)
        players = track_table(pos, self.stats_cfg)
        teams = team_summary(players)
        meta = {"n_frames": self.n_frames, "duration_s": self.t_last, "frame_size": self.size,
                "calibrated_share": float(cal["ok"].mean()) if len(cal) else 0.0,
                "calibration_sources": cal["source"].value_counts().to_dict() if len(cal) else {},
                "tracks": int(pos["track_id"].nunique()) if len(pos) else 0,
                "calibration": self.calib_cfg, "statistics": self.stats_cfg, "teams": self.teams.cfg,
                **(info or {})}
        return AnalysisReport(pos, cal, players, teams, meta)

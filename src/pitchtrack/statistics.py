"""Per-track statistics in metres.

A track gives positions on the pitch at real timestamps, some of them missing (no calibration,
point above the horizon, frame without detection). The rules are explicit:
the representative point is the bottom-middle of the box; a gap longer than max_gap_s cuts the
track into segments and no distance is counted across it; positions are smoothed over a window
of smooth_s seconds and the distance is summed every step_s seconds (the noise inflates a
distance summed at 25 Hz); a step faster than max_speed is physically implausible and makes its
segment unreliable; only reliable segments of at least min_duration_s count.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import pitch_model as pm


def split_segments(t, max_gap_s: float) -> list[np.ndarray]:
    """Indices of the runs of a sorted time series whose consecutive gaps are <= max_gap_s."""
    t = np.asarray(t, dtype=np.float64)
    if len(t) == 0:
        return []
    cuts = np.flatnonzero(np.diff(t) > max_gap_s + 1e-9) + 1
    return [np.asarray(s) for s in np.split(np.arange(len(t)), cuts)]


def smooth(t, xy, window_s: float) -> np.ndarray:
    """Centred moving average over +- window_s / 2 seconds. Near the ends of the series the
    window shrinks symmetrically (half-width min(window_s / 2, t - t_first, t_last - t)), so that a
    straight run at constant speed is left unchanged. window_s = 0: copy."""
    t = np.asarray(t, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if window_s <= 0 or len(t) < 2:
        return xy.copy()
    half = np.minimum(window_s / 2.0, np.minimum(t - t[0], t[-1] - t))
    lo = np.searchsorted(t, t - half - 1e-9, side="left")
    hi = np.searchsorted(t, t + half + 1e-9, side="right")
    c = np.vstack([np.zeros((1, 2)), np.cumsum(xy, axis=0)])
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


@dataclass
class Kinematics:
    distance_m: float          # over reliable segments only
    duration_s: float          # time covered by reliable segments
    observed_s: float          # time covered by all segments
    max_speed: float           # m/s, over the reliable steps (NaN if none)
    mean_speed: float          # distance_m / duration_s (NaN if no reliable segment)
    n_segments: int
    n_reliable: int
    n_implausible_steps: int

    @property
    def coverage(self) -> float:
        return self.duration_s / self.observed_s if self.observed_s > 0 else 0.0


# --------------------------------------------------------------------------- distance and speed
def track_kinematics(t, xy, max_gap_s: float = 0.5, smooth_s: float = 0.4, step_s: float = 0.2,
                     max_speed: float = 12.0, min_duration_s: float = 2.0) -> Kinematics:
    """Distance and speeds of one track from its pitch positions.

    t (n,) seconds, xy (n, 2) metres, NaN rows for missing positions. Steps: drop the NaN rows,
    sort by time, split_segments(max_gap_s); in each segment: smooth(smooth_s), keep one sample
    every step_s seconds (the first one, then the first sample at least step_s after the previous
    kept one, and always the last one), steps between kept samples, speeds = step length / real
    time difference; the
    segment is reliable if it lasts at least min_duration_s and no speed exceeds max_speed.
    """
    
    # 1. Convert inputs to NumPy arrays
    t = np.asarray(t, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)

    if len(t) != len(xy):
        raise ValueError("t and xy must have the same number of elements")

    # step_s = 0 keeps every sample
    if step_s < 0:
        raise ValueError("step_s must be non-negative")

    # 2. Remove missing or invalid observations
    ok = np.isfinite(xy).all(axis=1) & np.isfinite(t)

    # 3. Sort the remaining observations by timestamp
    order = np.argsort(t[ok], kind="mergesort")
    t, xy = t[ok][order], xy[ok][order]

    # 4. Initialize statistics
    dist = 0.0
    dur = 0.0
    obs = 0.0

    speeds = []
    n_rel = 0
    n_bad = 0

    # 5. Split the trajectory into separate segments
    segments = split_segments(t, max_gap_s)

    # 6. Process each segment independently
    for indices in segments:

        t_seg = t[indices]
        xy_seg = xy[indices]

        # Duration of this segment
        segment_duration = t_seg[-1] - t_seg[0]

        # Count the duration of all observed segments
        obs += segment_duration

        # 7. Smooth the positions
        xy_smoothed = smooth(t_seg, xy_seg, smooth_s)

        # 8. Select samples separated by at least step_s
        keep = [0]

        for i in range(1, len(t_seg)):
            if t_seg[i] - t_seg[keep[-1]] >= step_s:
                keep.append(i)

        # Always include the last sample
        if keep[-1] != len(t_seg) - 1:
            keep.append(len(t_seg) - 1)

        keep = np.asarray(keep, dtype=np.int64)

        t_kept = t_seg[keep]
        xy_kept = xy_smoothed[keep]

        # 9. Compute distances between consecutive kept positions
        steps = np.linalg.norm(
            np.diff(xy_kept, axis=0),
            axis=1
        )

        # 10. Compute actual time differences
        dt = np.diff(t_kept)

        # 11. Compute speeds
        # A non-positive time difference is treated as an
        # implausible step (infinite speed).
        v = np.divide(
            steps,
            dt,
            out=np.full_like(steps, np.inf),
            where=dt > 0
        )

        # 12. Count physically implausible steps
        n_bad += int(np.count_nonzero(v > max_speed))

        # 13. Check whether this segment is reliable
        reliable = (
            segment_duration >= min_duration_s
            and not np.any(v > max_speed)
        )

        if reliable:

            # Add the distance of this reliable segment
            dist += float(np.sum(steps))

            # Add its duration
            dur += float(segment_duration)

            # Count reliable segments
            n_rel += 1

            # Store speeds of reliable steps only
            speeds.extend(v.tolist())

    # 14. Compute final maximum and mean speeds
    max_v = float(np.max(speeds)) if speeds else float("nan")

    mean_v = dist / dur if dur > 0 else float("nan")

    # 15. Return the statistics
    return Kinematics(
        distance_m=float(dist),
        duration_s=float(dur),
        observed_s=float(obs),
        max_speed=max_v,
        mean_speed=mean_v,
        n_segments=len(segments),
        n_reliable=n_rel,
        n_implausible_steps=n_bad
    )



# --------------------------------------------------------------------------- heatmaps and noise
def heatmap(xy, bins=(21, 14), weights=None) -> np.ndarray:
    """Occupation histogram (bins along x, bins along y) over the pitch, normalised to sum 1."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    ok = np.isfinite(xy).all(axis=1)
    h, _, _ = np.histogram2d(xy[ok, 0], xy[ok, 1], bins=bins,
                             range=[[-pm.HALF_L, pm.HALF_L], [-pm.HALF_W, pm.HALF_W]],
                             weights=None if weights is None else np.asarray(weights)[ok])
    s = h.sum()
    return h / s if s > 0 else h


def track_table(positions, cfg: dict, min_share: float = 0.8):
    """Statistics of every track from its positions on the pitch -> DataFrame, one row per track.

    positions: one row per box with track_id, t, X, Y (NaN where unknown) and, if present, seq,
    team and pos_std_m. cfg: max_gap_s, smooth_s, step_s, max_speed, min_duration_s. A track is
    "reliable" if at least one of its segments is reliable, none of its steps is implausible and
    at least min_share of its boxes have a position; distances count reliable segments only.
    """
    import pandas as pd

    keys = ["seq", "track_id"] if "seq" in positions else ["track_id"]
    columns = keys + ["team", "frames", "position_share", "observed_s", "reliable_s", "distance_m", "mean_speed",
                      "max_speed", "segments", "reliable_segments", "implausible_steps", "median_pos_std_m",
                      "reliable"]
    rows = []
    for key, g in positions.groupby(keys):
        k = track_kinematics(g["t"].to_numpy(), g[["X", "Y"]].to_numpy(), cfg["max_gap_s"], cfg["smooth_s"],
                             cfg["step_s"], cfg["max_speed"], cfg["min_duration_s"])
        share = float(np.isfinite(g["X"]).mean())
        key = key if isinstance(key, tuple) else (key,)
        rows.append([*key, int(g["team"].iat[0]) if "team" in g else -1, len(g), share, k.observed_s, k.duration_s,
                     k.distance_m, k.mean_speed, k.max_speed, k.n_segments, k.n_reliable, k.n_implausible_steps,
                     float(g["pos_std_m"].median()) if "pos_std_m" in g else float("nan"),
                     bool(k.n_reliable > 0 and k.n_implausible_steps == 0 and share >= min_share)])
    table = pd.DataFrame(rows, columns=columns)
    table["track_id"] = table["track_id"].astype(int)
    return table


def team_summary(players):
    """One row per team (0, 1, unknown = -1) from track_table: tracks, reliable tracks, total
    distance of the reliable tracks, their median distance per minute and the top speed."""
    import pandas as pd

    rows = []
    for team, name in ((0, "team 0"), (1, "team 1"), (-1, "unknown")):
        g = players[players["team"] == team]
        rel = g[g["reliable"].astype(bool)]
        per_min = 60.0 * rel["distance_m"] / rel["reliable_s"].clip(lower=1e-9)
        rows.append({"team": team, "team_name": name, "tracks": len(g), "reliable_tracks": len(rel),
                     "distance_m": float(rel["distance_m"].sum()),
                     "median_m_per_min": float(per_min.median()) if len(rel) else float("nan"),
                     "top_speed": float(rel["max_speed"].max()) if len(rel) else float("nan")})
    return pd.DataFrame(rows)


def noise_distance_bias(sigma: float, fps: float) -> float:
    """Distance per second added by i.i.d. Gaussian position noise of std sigma (metres, per
    axis) to a STATIC player when steps are summed at fps: fps * sigma * sqrt(pi)."""
    return fps * sigma * np.sqrt(np.pi)

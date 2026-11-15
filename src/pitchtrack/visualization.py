"""Drawing helpers: boxes, labels and the pitch mask outline on BGR frames."""
from __future__ import annotations

import cv2
import numpy as np

from .boxes import as_boxes

GREEN, RED, YELLOW, WHITE, BLUE = (60, 200, 60), (40, 40, 230), (0, 215, 255), (255, 255, 255), (230, 120, 30)


def draw_boxes(image: np.ndarray, boxes, color=RED, labels=None, thickness: int = 2) -> np.ndarray:
    """Draw xyxy boxes (and optional text labels) on a copy of the image."""
    out = image.copy()
    for k, (x1, y1, x2, y2) in enumerate(as_boxes(boxes)):
        p1, p2 = (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))
        cv2.rectangle(out, p1, p2, color, thickness, cv2.LINE_AA)
        if labels is not None:
            text = str(labels[k])
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out, (p1[0], p1[1] - th - 4), (p1[0] + tw + 2, p1[1]), color, -1)
            cv2.putText(out, text, (p1[0] + 1, p1[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1,
                        cv2.LINE_AA)
    return out


def draw_mask_outline(image: np.ndarray, mask: np.ndarray, color=YELLOW, thickness: int = 2) -> np.ndarray:
    out = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness, cv2.LINE_AA)
    return out


def put_header(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (30, 30, 30), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- the pitch
TEAM_COLORS = {0: (40, 40, 230), 1: (230, 120, 30), -1: (160, 160, 160)}   # team 0, team 1, unknown (BGR)


def draw_pitch_lines(image: np.ndarray, H, color=YELLOW, thickness: int = 1) -> np.ndarray:
    """Project the lines of the pitch model with H (pitch -> image) and draw them: a visual check
    of a calibration."""
    from . import pitch_model as pm
    from .pitch_calibration import pitch_to_image
    out = image.copy()
    for name, xy in pm.sample_segments(0.5).items():
        uv = pitch_to_image(H, xy)
        ok = np.isfinite(uv).all(axis=1) & (np.abs(uv) < 1e5).all(axis=1)
        for run in np.split(np.arange(len(uv)), np.flatnonzero(np.diff(ok.astype(int)) != 0) + 1):
            if ok[run[0]] and len(run) > 1:
                cv2.polylines(out, [np.round(uv[run]).astype(np.int32)], False, color, thickness, cv2.LINE_AA)
    return out


def topview(scale: float = 5.0, margin: float = 3.0):
    """Top view of the pitch (BGR image) and the function metres -> pixels of that image."""
    from . import pitch_model as pm
    w, h = int((pm.LENGTH + 2 * margin) * scale), int((pm.WIDTH + 2 * margin) * scale)
    img = np.full((h, w, 3), (60, 130, 60), np.uint8)
    to_px = lambda xy: (np.asarray(xy, dtype=np.float64) + [pm.HALF_L + margin, pm.HALF_W + margin]) * scale
    for name, xy in pm.sample_segments(0.5).items():
        cv2.polylines(img, [np.round(to_px(xy)).astype(np.int32)], name == "Circle central", WHITE, 1, cv2.LINE_AA)
    return img, to_px


def minimap(xy, teams=None, ids=None, scale: float = 4.0) -> np.ndarray:
    """Top view with one dot per player (colour = team)."""
    img, to_px = topview(scale)
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    for k, p in enumerate(xy):
        if not np.isfinite(p).all():
            continue
        u, v = np.round(to_px(p)).astype(int)
        c = TEAM_COLORS.get(int(teams[k]) if teams is not None else -1, TEAM_COLORS[-1])
        cv2.circle(img, (u, v), 4, c, -1, cv2.LINE_AA)
        if ids is not None:
            cv2.putText(img, str(ids[k]), (u + 5, v - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1, cv2.LINE_AA)
    return img


def render_analysis(image: np.ndarray, result, team_of: dict | None = None, title: str = "",
                    minimap_scale: float = 3.0) -> np.ndarray:
    """One frame of an analysed video (analysis.FrameResult): the pitch lines projected with the
    homography of the frame, the tracked boxes coloured by team with their ids, a top view of the
    positions in the bottom-right corner and a header line. team_of (track id -> team) replaces the
    live team estimate of the result, e.g. the final teams of AnalysisReport."""
    out = draw_pitch_lines(image, np.linalg.inv(result.G)) if result.G is not None else image.copy()
    ids = np.asarray(result.ids)
    teams = (np.array([team_of.get(int(i), -1) for i in ids], dtype=int) if team_of is not None
             else np.asarray(result.teams, dtype=int))
    for team in np.unique(teams):
        sel = teams == team
        out = draw_boxes(out, np.asarray(result.boxes)[sel], TEAM_COLORS.get(int(team), TEAM_COLORS[-1]),
                         [int(i) for i in ids[sel]], 2)
    mm = minimap(result.xy, teams, ids, scale=minimap_scale)
    h, w = mm.shape[:2]
    H, W = out.shape[:2]
    if h + 16 < H and w + 16 < W:
        out[H - h - 8:H - 8, W - w - 8:W - 8] = mm
    n = [int((teams == k).sum()) for k in (0, 1, -1)]
    text = (f"{title}  t={result.t:6.2f}s  pitch: {result.calibration}  tracks: {len(ids)} "
            f"(team 0 {n[0]}, team 1 {n[1]}, unknown {n[2]})")
    return put_header(out, text.strip())


def plot_pitch(ax, color="white", lw=1.0):
    """Draw the pitch model on a matplotlib axis (metres, y downwards as in the image)."""
    from . import pitch_model as pm
    for name, xy in pm.sample_segments(0.5).items():
        closed = name == "Circle central"
        pts = np.vstack([xy, xy[:1]]) if closed else xy
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw)
    ax.set_xlim(-pm.HALF_L - 2, pm.HALF_L + 2)
    ax.set_ylim(pm.HALF_W + 2, -pm.HALF_W - 2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

"""The pitch of SoccerNet, in metres.

Frame of SoccerNet (sn-calibration, soccerpitch.py), used by the bbox_pitch positions of GSR:
origin at the centre spot, x from the left goal (x = -52.5) to the right goal (x = +52.5),
y from the far touch line (y = -34, "top" of the image) to the near one (y = +34, "bottom",
on the side of the main camera), z = 0 on the ground and negative above it. Pitch 105 x 68 m.

A straight line of the ground is stored as a homogeneous line (a, b, c) with a^2 + b^2 = 1, so
that a x + b y + c is the signed distance in metres of (x, y) to the line. Every ground line of
the pitch is parallel to an axis: "x-lines" (x = const) and "y-lines" (y = const). Several
annotation classes lie on the same infinite line (the tops of both penalty areas, for example):
`LINE_KEY` gives the infinite line of each class, so that they count once.
"""
from __future__ import annotations

import numpy as np

LENGTH, WIDTH = 105.0, 68.0
HALF_L, HALF_W = LENGTH / 2.0, WIDTH / 2.0
PENALTY_AREA = (16.5, 40.32)          # length (from the goal line), width
GOAL_AREA = (5.5, 18.32)
PENALTY_MARK = 11.0
CIRCLE_RADIUS = 9.15

_PA_X = HALF_L - PENALTY_AREA[0]      # 36.0
_PA_Y = PENALTY_AREA[1] / 2.0         # 20.16
_GA_X = HALF_L - GOAL_AREA[0]         # 47.0
_GA_Y = GOAL_AREA[1] / 2.0            # 9.16

# class name -> segment ((x1, y1), (x2, y2)) on the ground, in metres
SEGMENTS = {
    "Side line top": ((-HALF_L, -HALF_W), (HALF_L, -HALF_W)),
    "Side line bottom": ((-HALF_L, HALF_W), (HALF_L, HALF_W)),
    "Side line left": ((-HALF_L, -HALF_W), (-HALF_L, HALF_W)),
    "Side line right": ((HALF_L, -HALF_W), (HALF_L, HALF_W)),
    "Middle line": ((0.0, -HALF_W), (0.0, HALF_W)),
    "Big rect. left top": ((-HALF_L, -_PA_Y), (-_PA_X, -_PA_Y)),
    "Big rect. left main": ((-_PA_X, -_PA_Y), (-_PA_X, _PA_Y)),
    "Big rect. left bottom": ((-HALF_L, _PA_Y), (-_PA_X, _PA_Y)),
    "Big rect. right top": ((_PA_X, -_PA_Y), (HALF_L, -_PA_Y)),
    "Big rect. right main": ((_PA_X, -_PA_Y), (_PA_X, _PA_Y)),
    "Big rect. right bottom": ((_PA_X, _PA_Y), (HALF_L, _PA_Y)),
    "Small rect. left top": ((-HALF_L, -_GA_Y), (-_GA_X, -_GA_Y)),
    "Small rect. left main": ((-_GA_X, -_GA_Y), (-_GA_X, _GA_Y)),
    "Small rect. left bottom": ((-HALF_L, _GA_Y), (-_GA_X, _GA_Y)),
    "Small rect. right top": ((_GA_X, -_GA_Y), (HALF_L, -_GA_Y)),
    "Small rect. right main": ((_GA_X, -_GA_Y), (_GA_X, _GA_Y)),
    "Small rect. right bottom": ((_GA_X, _GA_Y), (HALF_L, _GA_Y)),
}

# class name -> (centre, radius, x-range kept); the two penalty arcs lie outside the penalty areas
CIRCLES = {
    "Circle central": ((0.0, 0.0), CIRCLE_RADIUS, (-CIRCLE_RADIUS, CIRCLE_RADIUS)),
    "Circle left": ((-HALF_L + PENALTY_MARK, 0.0), CIRCLE_RADIUS, (-_PA_X, -HALF_L + PENALTY_MARK + CIRCLE_RADIUS)),
    "Circle right": ((HALF_L - PENALTY_MARK, 0.0), CIRCLE_RADIUS, (HALF_L - PENALTY_MARK - CIRCLE_RADIUS, _PA_X)),
}

# classes that are not on the ground plane or not identified: never used for the homography
IGNORED_PREFIXES = ("Goal", "Line unknown")


def _line_of_segment(seg) -> np.ndarray:
    (x1, y1), (x2, y2) = seg
    if x1 == x2:                       # x = x1  ->  1 x + 0 y - x1 = 0
        return np.array([1.0, 0.0, -x1])
    return np.array([0.0, 1.0, -y1])   # y = y1


LINES = {name: _line_of_segment(seg) for name, seg in SEGMENTS.items()}
# infinite line of each class: ("x", value) or ("y", value); classes on the same line share it
LINE_KEY = {name: ("x" if l[0] == 1.0 else "y", round(-l[2], 3)) for name, l in LINES.items()}


def is_ground_line(name: str) -> bool:
    return name in LINES


def is_circle(name: str) -> bool:
    return name in CIRCLES


def sample_segments(step: float = 0.5) -> dict:
    """Points every `step` metres along each segment and arc (drawing, synthetic data)."""
    out = {}
    for name, ((x1, y1), (x2, y2)) in SEGMENTS.items():
        n = max(2, int(np.hypot(x2 - x1, y2 - y1) / step) + 1)
        s = np.linspace(0.0, 1.0, n)[:, None]
        out[name] = np.c_[x1 + s * (x2 - x1), y1 + s * (y2 - y1)]
    for name, ((cx, cy), r, (xmin, xmax)) in CIRCLES.items():
        th = np.linspace(0.0, 2 * np.pi, max(16, int(2 * np.pi * r / step)), endpoint=False)
        pts = np.c_[cx + r * np.cos(th), cy + r * np.sin(th)]
        out[name] = pts[(pts[:, 0] >= xmin - 1e-9) & (pts[:, 0] <= xmax + 1e-9)]
    return out


def circle_residual(name: str, xy) -> np.ndarray:
    """Distance in metres between points of the pitch and the circle of class `name`."""
    (cx, cy), r, _ = CIRCLES[name]
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return np.abs(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - r)


def inside_pitch(xy, margin: float = 0.0) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return (np.abs(xy[:, 0]) <= HALF_L + margin) & (np.abs(xy[:, 1]) <= HALF_W + margin)

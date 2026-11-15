"""Frames in, frames out: image folders, video files, timestamps and annotated videos.

Each source yields `Frame(index, t, image)` with a 0-based index, a timestamp t in
seconds and a BGR uint8 image. For a video file, t comes from the container
(presentation timestamps, read through CAP_PROP_POS_MSEC) and is NOT assumed to be
index / fps: re-encoded or variable-frame-rate files break that formula.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, NamedTuple

import cv2
import numpy as np


class Frame(NamedTuple):
    index: int
    t: float
    image: np.ndarray


class ImageFolderSource:
    """Frames stored as numbered images (SoccerNet: img1/000001.jpg ...), constant fps."""

    def __init__(self, folder, fps: float, ext: str = ".jpg", stride: int = 1):
        self.folder, self.fps, self.stride = Path(folder), float(fps), int(stride)
        self.files = sorted(self.folder.glob(f"*{ext}"))
        if not self.files:
            raise FileNotFoundError(f"no {ext} images in {self.folder}")

    def __len__(self) -> int:
        return len(self.files[:: self.stride])

    def __iter__(self) -> Iterator[Frame]:
        for index in range(0, len(self.files), self.stride):
            image = cv2.imread(str(self.files[index]), cv2.IMREAD_COLOR)
            yield Frame(index, index / self.fps, image)


class VideoFileSource:
    """Frames of a video file read with OpenCV; timestamps from the container."""

    def __init__(self, path, stride: int = 1):
        self.path, self.stride = Path(path), int(stride)
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise FileNotFoundError(self.path)
        self.nominal_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        self.n_frames_declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()

    def __iter__(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(str(self.path))
        index = 0
        try:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                t = msec / 1000.0 if msec > 0 or index == 0 else index / self.nominal_fps
                if index % self.stride == 0:
                    yield Frame(index, t, image)
                index += 1
        finally:
            cap.release()

    def timestamps(self) -> np.ndarray:
        """Timestamps of all frames (decodes the whole file once)."""
        stride, self.stride = self.stride, 1
        try:
            return np.array([f.t for f in self], dtype=np.float64)
        finally:
            self.stride = stride


# --------------------------------------------------------------------------- timestamp audit
@dataclass
class TimestampAudit:
    n_frames: int
    duration: float          # t_last - t_first, seconds
    effective_fps: float     # (n_frames - 1) / duration
    n_duplicates: int        # consecutive timestamps closer than half a nominal period
    n_backwards: int         # timestamps going back in time
    n_gaps: int              # intervals longer than 1.5 nominal periods
    max_gap: float           # longest interval, seconds
    est_dropped: int         # frames missing inside the gaps: sum(round(dt * fps) - 1)


def audit_timestamps(t, nominal_fps: float) -> TimestampAudit:
    """Check a timestamp series against the nominal frame rate: intervals going backwards
    (dt < 0), duplicates (0 <= dt < 0.5 / fps) and gaps (dt > 1.5 / fps)."""

    # Convert timestamps to a 1D float64 NumPy array
    t = np.asarray(t, dtype=np.float64).reshape(-1)

    n = len(t)

    # Not enough timestamps to compute time intervals
    if n < 2:
        return TimestampAudit(
            n, 0.0, float("nan"), 0, 0, 0, 0.0, 0
        )

    # Time intervals between consecutive frames: shape (n-1,)
    dt = np.diff(t)

    # Expected time interval between consecutive frames
    period = 1.0 / nominal_fps

    # Detect gaps: intervals longer than 1.5 expected periods
    gaps = dt > 1.5 * period

    # Total video duration
    duration = float(t[-1] - t[0])

    return TimestampAudit(
        n_frames=n,
        duration=duration,

        # Effective frame rate
        effective_fps=(n - 1) / duration if duration > 0 else float("nan"),

        # Number of duplicated timestamps
        n_duplicates=int(np.sum((dt >= 0) & (dt < 0.5 * period))),

        # Number of timestamps going backwards
        n_backwards=int(np.sum(dt < 0)),

        # Number of gaps
        n_gaps=int(np.sum(gaps)),

        # Largest observed interval
        max_gap=float(dt.max()),

        # Estimated number of missing frames
        est_dropped=int(np.sum(np.round(dt[gaps] * nominal_fps) - 1)),
    )
    


class VideoWriter:
    """Annotated video output. Use .mp4 (codec mp4v) or .avi (MJPG, always available)."""

    def __init__(self, path, fps: float, size_wh):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*("MJPG" if self.path.suffix.lower() == ".avi" else "mp4v"))
        self.writer = cv2.VideoWriter(str(self.path), fourcc, float(fps), tuple(int(v) for v in size_wh))
        if not self.writer.isOpened():
            raise RuntimeError(f"cannot open {self.path} for writing")

    def write(self, image: np.ndarray) -> None:
        self.writer.write(image)

    def close(self) -> None:
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

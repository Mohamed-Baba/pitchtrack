"""The video analysis on a rendered broadcast clip: two teams of coloured players running, a camera
that pans, a detector that finds the coloured boxes, and pitch lines given by a stand-in annotator."""
import cv2
import numpy as np
import pytest
from conftest import broadcast_camera, pitch_line_annotation, render_broadcast_frame

from pitchtrack.analysis import VideoAnalyzer
from pitchtrack.detector import Detections
from pitchtrack.tracker import Tracker, TrackerConfig
from pitchtrack.video_io import VideoFileSource, VideoWriter
from pitchtrack.visualization import render_analysis

SIZE, FPS, N = (640, 360), 25.0, 75
RED, BLUE = (40, 40, 210), (210, 90, 30)
SPEED = 4.0                                                     # m/s, along x


def view(f):
    x = -30.0 + 0.04 * f                                        # slow pan to the right
    return broadcast_camera((x, 0.0), (x + 10.0, 74.0, -18.0), *SIZE)


def players(f):
    t = f / FPS
    start = [(-40.0, -15.0), (-36.0, -5.0), (-38.0, 6.0), (-30.0, 12.0), (-26.0, -10.0)]
    return ([(x + SPEED * t, y, RED) for x, y in start] +
            [(x + 5.0 + SPEED * t, y + 2.5, BLUE) for x, y in start])


class ColourDetector:
    """Detections = connected blobs of shirt colour, extended down to the shorts."""

    def __call__(self, images):
        out = []
        for img in images:
            boxes = []
            for colour in (RED, BLUE):
                mask = (np.abs(img.astype(int) - colour).sum(axis=2) < 90).astype(np.uint8)
                n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
                for x, y, w, h, area in stats[1:]:
                    if area >= 6:
                        boxes.append([x, y, x + w, y + h / 0.6])
            out.append(Detections(np.array(boxes).reshape(-1, 4), np.full(len(boxes), 0.9), np.zeros(len(boxes))))
        return out


class TrueLines:
    """Stand-in for the line network: the true annotation, nothing on the frames in `blind`."""

    def __init__(self, blind=()):
        self.f, self.blind = 0, set(blind)

    def __call__(self, images):
        Hm, _, size = view(self.f)
        out = [{} if self.f in self.blind else pitch_line_annotation(Hm, size)]
        self.f += len(images)
        return out


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "clip.avi"
    with VideoWriter(path, FPS, SIZE) as w:
        for f in range(N):
            Hm, P, _ = view(f)
            w.write(render_broadcast_frame(Hm, SIZE, P, players(f), seed=f))
    return path


def test_video_analysis_end_to_end(clip, tmp_path):
    teams_cfg = {"stride": 2, "min_height": 10, "min_pixels": 10}
    analyzer = VideoAnalyzer(ColourDetector(), Tracker(TrackerConfig(n_init=2)), SIZE, TrueLines(blind=range(30, 36)),
                             teams_cfg=teams_cfg, stats_cfg={"min_duration_s": 1.0})
    analyzer.teams.min_samples = 40
    results = [analyzer.process(fr) for fr in VideoFileSource(clip)]
    report = analyzer.finish({"video": "clip"})

    sources = [r.calibration for r in results]
    assert sources.count("lines") >= N - 8
    assert all(sources[f] == "propagated" for f in range(30, 36))       # no lines: moved homography
    # positions: the foot of each box, projected, against the true player positions
    errs = []
    for f in (10, 40, 70):
        truth = np.array([(x, y) for x, y, _ in players(f)])
        xy = results[f].xy[np.isfinite(results[f].xy).all(axis=1)]
        assert len(xy) >= 8
        errs += list(np.min(np.linalg.norm(xy[:, None] - truth[None], axis=2), axis=1))
    assert np.median(errs) < 0.6
    # teams: the two colours end up in two different teams
    team_of = report.team_of()
    pos = report.positions
    by_colour = []
    for f in (60, 70):
        r = results[f]
        frame = next(fr for fr in VideoFileSource(clip) if fr.index == f).image
        for i, (x1, y1, x2, y2) in zip(r.ids, r.boxes):
            c = frame[int(y1 + 0.3 * (y2 - y1)), int((x1 + x2) / 2)].astype(int)
            by_colour.append((int(np.argmin([np.abs(c - RED).sum(), np.abs(c - BLUE).sum()])), team_of.get(int(i), -1)))
    pairs = {p for p in by_colour if p[1] != -1}
    assert len(pairs) == 2 and len({t for _, t in pairs}) == 2
    # statistics: about 4 m/s for the reliable tracks
    rel = report.players[report.players["reliable"]]
    assert len(rel) >= 5
    assert np.median(rel["distance_m"] / rel["reliable_s"]) == pytest.approx(SPEED, rel=0.15)
    assert set(report.teams["team"]) == {0, 1, -1} and len(pos) > 0

    out = report.save(tmp_path / "report")
    for name in ("positions.parquet", "calibration.parquet", "player_stats.csv", "team_stats.csv", "summary.md",
                 "heatmaps.png", "report.json"):
        assert (out / name).is_file()
    img = render_analysis(next(iter(VideoFileSource(clip))).image, results[0], team_of, "clip")
    assert img.shape == (SIZE[1], SIZE[0], 3)


def test_without_line_model_nothing_is_calibrated(clip):
    analyzer = VideoAnalyzer(ColourDetector(), Tracker(), SIZE)
    results = [analyzer.process(fr) for fr in VideoFileSource(clip)]
    assert all(r.G is None and r.calibration == "none" for r in results)
    assert all(np.isnan(r.xy).all() for r in results)
    report = analyzer.finish()
    assert len(report.positions) > 0 and not report.players["reliable"].any()

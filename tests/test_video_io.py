import numpy as np
import pytest

from pitchtrack.video_io import ImageFolderSource, VideoFileSource, VideoWriter, audit_timestamps


# ----------------------------------------------------------------- timestamps
def test_audit_clean_series():
    t = np.arange(250) / 25.0
    a = audit_timestamps(t, 25.0)
    assert a.n_frames == 250 and a.n_gaps == 0 and a.n_duplicates == 0 and a.n_backwards == 0
    assert a.effective_fps == pytest.approx(25.0)
    assert a.duration == pytest.approx(249 / 25)
    assert a.est_dropped == 0 and a.max_gap == pytest.approx(0.04)


def test_audit_gaps_duplicates_backwards():
    t = np.arange(100) / 25.0
    t = np.delete(t, [10, 11, 12, 50])          # 3 frames lost in one gap, 1 in another
    t = np.insert(t, 30, t[29])                 # one duplicated timestamp
    t = np.append(t, t[-1] - 0.01)              # one step back in time
    a = audit_timestamps(t, 25.0)
    assert a.n_gaps == 2 and a.est_dropped == 4
    assert a.n_duplicates == 1 and a.n_backwards == 1
    assert a.max_gap == pytest.approx(4 / 25)


def test_audit_short_series():
    a = audit_timestamps([0.0], 25.0)
    assert a.n_frames == 1 and np.isnan(a.effective_fps)


# ----------------------------------------------------------------- frame sources
def test_image_folder_source(gsr_root):
    src = ImageFolderSource(gsr_root / "valid" / "SNGS-901" / "img1", fps=25, stride=2)
    frames = list(src)
    assert [f.index for f in frames] == [0, 2] and len(src) == 2
    assert frames[1].t == pytest.approx(2 / 25) and frames[0].image.shape == (180, 320, 3)


def test_video_roundtrip(tmp_path):
    path = tmp_path / "clip.avi"
    with VideoWriter(path, 25.0, (64, 48)) as w:
        for k in range(12):
            w.write(np.full((48, 64, 3), 20 * k % 255, np.uint8))
    src = VideoFileSource(path)
    t = src.timestamps()
    assert len(t) == 12 and src.nominal_fps == pytest.approx(25.0)
    assert audit_timestamps(t, src.nominal_fps).n_gaps == 0
    assert [f.index for f in VideoFileSource(path, stride=5)] == [0, 5, 10]

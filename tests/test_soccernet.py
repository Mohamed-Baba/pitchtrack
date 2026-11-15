import numpy as np
import pandas as pd
import pytest

from pitchtrack.audit import audit_sequence, check_game_disjoint, summarize_splits
from pitchtrack.soccernet import (GT_COLUMNS, annotation_to_row, frame_index, list_sequences,
                                  load_sequence)


def test_frame_index_convention():
    assert frame_index("000001.jpg") == 0
    assert frame_index("000750.jpg") == 749


# ----------------------------------------------------------------- ground truth
def test_annotation_to_row_converts_centre_to_corners():
    ann = {"image_id": "x7", "track_id": 5, "supercategory": "object",
           "attributes": {"role": "player", "team": "left", "jersey": "10"},
           "bbox_image": {"x": 90, "y": 40, "x_center": 100.0, "y_center": 60.0, "w": 20.0, "h": 40.0},
           "bbox_pitch": {"x_bottom_middle": -12.5, "y_bottom_middle": 3.0}}
    row = annotation_to_row(ann, {"x7": 41})
    assert set(row) == set(GT_COLUMNS)
    assert row["frame"] == 41 and row["track_id"] == 5
    assert (row["x1"], row["y1"], row["x2"], row["y2"]) == (90.0, 40.0, 110.0, 80.0)
    assert (row["role"], row["team"], row["jersey"]) == ("player", "left", "10")
    assert (row["pitch_x"], row["pitch_y"]) == (-12.5, 3.0)


def test_annotation_to_row_skips_non_objects_and_handles_missing():
    assert annotation_to_row({"supercategory": "pitch", "image_id": "x"}, {"x": 0}) is None
    ann = {"image_id": "x", "track_id": "3", "supercategory": "object",
           "bbox_image": {"x_center": 10, "y_center": 10, "w": 4, "h": 8}}
    row = annotation_to_row(ann, {"x": 0})
    assert row["track_id"] == 3 and row["role"] is None
    assert np.isnan(row["pitch_x"]) and np.isnan(row["pitch_y"])


def test_load_sequence(gsr_root):
    seq = load_sequence(gsr_root / "valid" / "SNGS-901")
    assert seq.info.name == "SNGS-901" and seq.info.split == "valid" and seq.info.game_id == 7
    assert seq.info.fps == 25 and seq.info.n_frames == 4 and (seq.info.width, seq.info.height) == (320, 180)
    assert seq.labeled_frames.tolist() == [0, 1, 2]                    # frame 4 is not labeled
    assert len(seq.gt) == 7 and list(seq.gt.columns) == GT_COLUMNS
    t1 = seq.gt[seq.gt.track_id == 1]
    np.testing.assert_allclose(t1[["x1", "y1", "x2", "y2"]].to_numpy(),
                               [[100, 50, 120, 90], [110, 50, 130, 90], [120, 50, 140, 90]])
    assert t1["frame"].tolist() == [0, 1, 2]
    assert sorted(seq.lines) == [0]
    assert seq.read_frame(0).shape == (180, 320, 3)
    assert seq.frame_path(3).name == "000004.jpg"


def test_list_sequences_sorted(gsr_root):
    names = [p.name for p in list_sequences(gsr_root, "valid")]
    assert names == ["SNGS-901", "SNGS-902"]
    with pytest.raises(FileNotFoundError):
        list_sequences(gsr_root, "test")


# ----------------------------------------------------------------- audit
def test_audit_sequence(gsr_root):
    a = audit_sequence(load_sequence(gsr_root / "valid" / "SNGS-901"), small_px=35)
    assert a["n_boxes"] == 7 and a["n_tracks"] == 4 and a["n_labeled"] == 3
    assert a["boxes_per_frame"] == pytest.approx(7 / 3)
    assert (a["n_player"], a["n_referee"], a["n_goalkeeper"], a["n_other"]) == (3, 2, 1, 1)
    assert a["height_p50"] == pytest.approx(40.0)
    assert a["frac_small"] == pytest.approx(3 / 7)          # heights 32, 32, 30 < 35
    assert a["frac_border"] == pytest.approx(1 / 7)         # the goalkeeper at x = 0
    assert a["frac_no_pitch"] == pytest.approx(1 / 7)
    assert a["frac_degenerate"] == 0.0
    assert a["track_len_median"] == pytest.approx(1.5)      # lengths 3, 2, 1, 1
    assert a["game_id"] == 7 and a["split"] == "valid"


def test_game_disjoint_check():
    audit = pd.DataFrame({"name": ["a", "b", "c", "d"], "split": ["train", "valid", "valid", "test"],
                          "game_id": [1, 2, 2, 1]})
    shared = check_game_disjoint(audit)
    assert shared["game_id"].unique().tolist() == [1]
    assert check_game_disjoint(audit[audit.name != "d"]).empty


def test_summarize_splits(gsr_root):
    rows = [audit_sequence(load_sequence(p)) for split in ("train", "valid")
            for p in list_sequences(gsr_root, split)]
    s = summarize_splits(pd.DataFrame(rows))
    assert s.loc["valid", "n_clips"] == 2 and s.loc["valid", "n_games"] == 1
    assert s.loc["train", "n_boxes"] == 7

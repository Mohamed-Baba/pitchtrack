# pitchtrack

Detection, multi-object tracking and pitch geometry of football players in broadcast video,
written from scratch: from a video file to tracks, teams, positions on the pitch in metres and
running statistics, with every stage measured on SoccerNet-GSR.

```
frames ──► person detector ──► tracker ──► teams (shirt colours)
   │        (YOLO26, COCO)     (Kalman, Hungarian, two-stage association, camera motion)
   │
   └─────► pitch-line network ──► homography ──► positions (m) ──► distances, speeds, heatmaps
           (U-Net, trained on       (DLT on lines, RANSAC, quality checks,
            SoccerNet-GSR)           propagation by image motion)
```

## What is in it

Only the person detector (YOLO26 from Ultralytics, pre-trained on COCO) and the ImageNet encoder of
the line network come pre-trained. Everything else is written here and tested:

- **Tracking**: Kalman filter on boxes with a noise proportional to the box height (Joseph form,
  NEES check), optimal assignment with a gate through an augmented cost matrix, Hungarian method,
  ByteTrack-style two-stage association, track life cycle, camera motion compensation.
- **Metrics**: greedy matching and COCO AP (equal to pycocotools), operating curves, recall by box
  size, HOTA (equal to TrackEval to 1e-12), confidence intervals by resampling clips.
- **Teams**: torso colour in Lab without the grass, k-means++, a decision per track with an
  "unknown" answer for referees, goalkeepers and doubtful tracks.
- **Pitch calibration**: normalised DLT on point-on-line constraints, RANSAC whose consensus counts
  lines, rejection of mirrored models, quality checks on held-out circle points, propagation of
  the homography with the image motion, first-order uncertainty of the positions in metres.
- **Pitch-line network**: a U-Net on a ResNet encoder that predicts the pitch-line annotation of a
  frame, in the format of SoccerNet-GSR; the calibration consumes it unchanged, so it runs on any
  video. Trained on the train split of SoccerNet-GSR.
- **Statistics** with explicit rules: gaps cut the tracks (no interpolation), smoothing and a
  summation step against the noise, physically implausible steps flagged, reliability per track.

## Installation

Install the PyTorch build that matches your GPU first (an RTX 50xx needs a build whose
`torch.cuda.get_arch_list()` contains `sm_120`), for example:

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

then the package:

    pip install -e ".[detector,data,dev,track,video]"
    python scripts/check_env.py --yolo
    pytest

The tests run on the CPU and download nothing.

## Analyse a video

    python scripts/download_sample_video.py
    python scripts/analyze_video.py --video data/videos/2e57b9_0.mp4 --line-model models/pitch_lines.pt --out outputs/videos/2e57b9_0

The output folder holds the annotated video (boxes and ids coloured by team, the pitch lines
projected with the homography of each frame, a top view of the positions), the positions of
every tracked box (`positions.parquet`), the calibration of every frame, the statistics per
player and per team (`player_stats.csv`, `team_stats.csv`), heatmaps and a `summary.md`.
Without `--line-model` the pitch is not calibrated: boxes, tracks and teams only, no metres.
On a CPU, `--weights yolo26n.pt --device cpu` keeps it usable.

The analysis runs one frame at a time and only uses the past (`analysis.VideoAnalyzer.process`),
so it can drive a live display; the teams are then refitted on the whole video for the report.

## Train the pitch-line network

    python scripts/download_gsr.py --root data/SoccerNetGS --splits train valid
    python scripts/train_line_model.py --out models/pitch_lines.pt

Training frames: one in five of the train split, with a horizontal flip that swaps the left and
right classes, random zooms and colour jitter. Model selection on valid with the F1 of the
predicted lines at a 2-pixel tolerance. The 30 epochs of `configs/line_model.yaml` take about
two hours on a laptop RTX 5070 Ti; the best epoch was the 11th (F1 0.966), after which the
validation loss rises, so about 15 epochs are enough. The decisive evaluation is the calibration
it gives, compared with the calibration from the annotated lines:

    python scripts/calibrate.py --split valid --out outputs/calib/valid.parquet
    python scripts/calibrate.py --split valid --line-model models/pitch_lines.pt --out outputs/calib/valid_model.parquet
    python scripts/eval_calibration.py --calib outputs/calib/valid.parquet
    python scripts/eval_calibration.py --calib outputs/calib/valid_model.parquet

## Evaluate on SoccerNet-GSR

Data: `python scripts/download_gsr.py --splits valid` (see `data/README.md`); `test` is used once,
for the final numbers.

Detection:

    python scripts/audit_dataset.py --splits valid
    python scripts/run_detector.py --split valid --stride 5 --out outputs/det/valid_s5.parquet
    python scripts/eval_detection.py --dets outputs/det/valid_s5.parquet
    python scripts/make_overlay.py --clip SNGS-XXX --dets outputs/det/valid_s5.parquet --out outputs/overlay.mp4

Tracking:

    python scripts/run_detector.py --split valid --stride 1 --out outputs/det/valid_s1.parquet
    python scripts/run_tracker.py --dets outputs/det/valid_s1.parquet --cmc --out outputs/trk/v_kalman_cmc.parquet
    python scripts/run_tracker.py --dets outputs/det/valid_s1.parquet --out outputs/trk/v_kalman.parquet
    python scripts/run_tracker.py --dets outputs/det/valid_s1.parquet --motion none --single-stage --out outputs/trk/v_iou.parquet
    python scripts/run_reference_bytetrack.py --dets outputs/det/valid_s1.parquet --out outputs/trk/v_ref.parquet
    python scripts/eval_tracking.py --tracks outputs/trk/v_kalman_cmc.parquet
    python scripts/id_switches.py --tracks outputs/trk/v_kalman_cmc.parquet

`eval_tracking.py` runs the same way on the other three tables.

Teams, pitch and statistics:

    python scripts/run_teams.py --tracks outputs/trk/v_kalman_cmc.parquet --out outputs/teams/valid.parquet
    python scripts/eval_teams.py --teams outputs/teams/valid.parquet
    python scripts/calibrate.py --split valid --dets outputs/det/valid_s1.parquet --out outputs/calib/valid.parquet
    python scripts/eval_calibration.py --calib outputs/calib/valid.parquet
    python scripts/run_statistics.py --tracks outputs/trk/v_kalman_cmc.parquet --calib outputs/calib/valid.parquet --teams outputs/teams/valid.parquet --out outputs/stats/valid
    python scripts/run_statistics.py --oracle --calib outputs/calib/valid.parquet --out outputs/stats/valid_oracle
    python scripts/eval_statistics.py --positions outputs/stats/valid_oracle/positions.parquet
    python scripts/make_final_video.py --clip SNGS-XXX --stats outputs/stats/valid --calib outputs/calib/valid.parquet --out outputs/final.mp4

Each evaluation writes a markdown report to `reports/`.

## Results (validation split of SoccerNet-GSR)

58 clips of 30 s (the line network was trained on the train split); laptop RTX 5070 Ti. The
full reports are in `reports/`.

| detector | imgsz | AP50 | AP@[.5:.95] | precision / recall at τ = 0.5 | DetA at τ | recall, boxes under 40 px | ms / frame |
|---|---|---|---|---|---|---|---|
| YOLO26m (COCO, not fine-tuned) | 1280 | 0.839 | 0.469 | 0.920 / 0.848 | 0.790 | 0.043 | 15.9 |

One frame in five (8,700 frames, 149,771 boxes); τ maximises DetA; boxes under 40 px tall are
7 % of the ground truth.

| tracker (same detections) | HOTA [95 % CI] | DetA | AssA | IDF1 | IDSW |
|---|---|---|---|---|---|
| IoU only | 0.514 [0.494, 0.535] | 0.636 | 0.416 | 0.577 | 4,587 |
| Kalman + IoU, two stages | 0.550 [0.533, 0.569] | 0.634 | 0.478 | 0.639 | 2,529 |
| **Kalman + IoU, two stages, camera motion** | **0.578 [0.562, 0.594]** | **0.645** | **0.518** | **0.675** | **2,007** |
| same, with a 95 % Mahalanobis gate | 0.389 [0.367, 0.411] | 0.615 | 0.249 | 0.395 | 27,099 |
| ByteTrack (Ultralytics, reference) | 0.539 [0.522, 0.557] | 0.638 | 0.457 | 0.618 | 3,012 |

Every frame (43,500), detections with a score of at least 0.1; intervals by resampling clips;
tracking alone takes about 1 ms per frame. The gated row (`reports/tracking_v_byte_cmc.md`) used
`maha_gate: 9.4877` in `configs/tracker.yaml`.

| pitch lines | frames calibrated (from their lines + propagated) | median position error (m), true boxes |
|---|---|---|
| annotated (SoccerNet-GSR) | 68.2 % (49.1 + 19.1) | 0.47 [0.44, 0.51] |
| predicted by the line network | 84.5 % (65.1 + 19.4) | 0.54 |

The error is the distance between the projected foot point of each ground-truth box and its
annotated position, over every player the calibration reaches. The network was selected on this
split (line F1 0.966 at a 2-pixel tolerance), so its row is slightly optimistic; `test` is kept
for a single final run.

| teams and statistics | value |
|---|---|
| team macro-F1 per track (unknown = error), own tracks | 0.917 [0.910, 0.925]: 87.9 % of the tracks named, 98.1 % of them right |
| reliable tracks, median distance per minute (m) | 132 for each team |
| distance bias against the annotated positions, true boxes | +6.5 % (+60 % without smoothing and step) |

What the numbers say:

- **A 95 % Mahalanobis gate wrecked the tracker.** By construction it rejects about 5 % of the
  true matches at each frame (5.5 % measured on the sample clip), so at 25 frames per second most tracks lose their
  match at least once per second (1 − 0.95²⁵ ≈ 72 %), and each rejected detection started a new
  track that took over. Without the gate (IoU gating only, like ByteTrack), HOTA goes from 0.389
  to 0.578 and the identity switches from 27,099 to 2,007. On the sample clip, with the same
  YOLO26n detections: 181 identities with the gate, 40 without, 60 for the Ultralytics ByteTrack. Even a
  99.9 % gate leaves 93: real errors (occlusions, poses) have heavier tails than the Gaussian
  model.
- **Camera motion compensation** adds 0.028 HOTA and removes a fifth of the identity switches.
- **The line network calibrates more frames than the annotations** (84.5 % against 68.2 %) for
  7 cm more median error, and it is what lets the pipeline run on any video.
- **Detection** finds 88 to 97 % of the players taller than 60 px, but 4 % of those under 40 px:
  the next gain is there (fine-tuning on the train split with `scripts/export_yolo.py`, or tiles).
- **Distances**: the position noise (11 cm, against 1.5 cm for the annotated positions) would
  inflate the distances by 60 %; smoothing over 0.4 s and summing every 0.2 s brings the bias
  to +6.5 %.

On the sample clip (a Bundesliga broadcast, a source the line network never saw): every frame
calibrated, 88 % from its own lines; 34 tracks for about 24 people on screen; 7.6 frames per
second with the line network on every frame, 21 without it.

## Repository layout

```
src/pitchtrack/
  boxes.py              box conventions, conversions, IoU / IoS, resizing and letterboxing
  video_io.py           frame sources with real timestamps, timestamp audit, video writer
  soccernet.py          SoccerNet-GSR clips: ground truth, pitch-line annotations
  audit.py              dataset audit
  detector.py           detector contract, NMS, tiled inference, YOLO wrapper
  pitch_mask.py         grass mask, distance of a detection to the pitch
  metrics.py            detection metrics
  kalman.py             Kalman filter on boxes
  assignment.py         assignment with a gate, Hungarian method
  camera_motion.py      camera motion between frames, tracks moved into the new frame
  tracker.py            tracker (association, life cycle)
  mot_metrics.py        HOTA
  team_classifier.py    shirt colours, k-means, team per track
  pitch_model.py        the pitch in metres
  pitch_calibration.py  homography from lines, RANSAC, projection, uncertainty, quality, propagation
  statistics.py         distances, speeds, reliability, heatmaps, tables per track and per team
  line_annotations.py   pitch-line annotations <-> label maps
  line_model.py         pitch-line network, its data and its inference
  analysis.py           frame-by-frame analysis of a video and its report
  visualization.py      drawing
scripts/                one script per experiment or tool (each docstring gives its commands)
configs/                baseline.yaml (data, detector), tracker.yaml, analysis.yaml (teams,
                        calibration, statistics), line_model.yaml
```

## Conventions

Boxes are `(x1, y1, x2, y2)` in pixels of the original frame, continuous coordinates, width
`x2 - x1`; frames are indexed from 0 (`000001.jpg` is frame 0); timestamps are in seconds and come
from the container, never from `index / fps`. Pitch coordinates follow SoccerNet: metres, origin
at the centre spot, x towards the right goal, y towards the near touch line. See
`src/pitchtrack/boxes.py` and `src/pitchtrack/pitch_model.py`.

## Limitations

- The calibration needs two lines in each direction of the pitch, or a recent calibrated frame
  to propagate from: close-ups, replays and some midfield views stay uncalibrated, and their
  positions are missing, never interpolated.
- A homography assumes a flat pitch and no lens distortion.
- People standing near the touch lines (assistant referees, staff by the bench) are tracked like
  players: the grass mask cannot separate them, and the statistics only drop the positions more
  than `margin_m` metres outside the pitch (`configs/analysis.yaml`).
- Track ids are local to a shot: the tracker restarts at a camera cut, and there is no
  re-identification. The ball is not tracked.
- Teams come from k-means on the colours of one video: referees and goalkeepers usually end up
  "unknown".
- The detector is not fine-tuned for football: it misses most players under 40 px tall.
- On a new video there is no ground truth for the positions. On the sample clip the top speeds
  reach 11 m/s, above what footballers reach (about 10 m/s): a sign of residual noise in the
  positions from the predicted lines.

## Licences

Code: MIT (see `LICENSE`). The optional detector backend (Ultralytics) is AGPL-3.0 and is not
included; the tracker, the calibration and the metrics do not depend on it. SoccerNet data and the
sample clips are not redistributed here: use them under their own terms.

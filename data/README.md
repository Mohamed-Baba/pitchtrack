# Data

Nothing in this folder is committed except this file.

## SoccerNet Game State Reconstruction (SoccerNet-GSR, v1.3)

200 clips of 30 s from 12 games of the Swiss Super League (main broadcast camera, 1920 x 1080,
25 fps), the same clips as SoccerNet-Tracking. Each clip: `img1/000001.jpg ... 000750.jpg` and
`Labels-GameState.json` (boxes, track ids, role, team left/right, jersey, foot position on the
pitch in metres, pitch lines). The ball is not annotated in this version.

| split | zip size | used in |
|---|---|---|
| valid | 11.2 GB | every measurement, tuning and model selection |
| train | 9.8 GB | training: pitch-line network, detector fine-tuning (optional) |
| test | 8.9 GB (+ labels) | final numbers, once |

    pip install SoccerNet
    python scripts/download_gsr.py --root data/SoccerNetGS --splits valid

Expected layout:

    data/SoccerNetGS/valid/SNGS-XXX/img1/000001.jpg
    data/SoccerNetGS/valid/SNGS-XXX/Labels-GameState.json

Check `"info" > "version"` in one `Labels-GameState.json`: it must be 1.3 or more.
Terms of use: see https://www.soccer-net.org and https://github.com/SoccerNet/sn-gamestate.
Do not redistribute frames or labels; publish code, metrics and your own figures.

## Videos

`data/videos/` holds the videos to analyse with `scripts/analyze_video.py`. To try the pipeline
on a clip that comes without any annotation:

    python scripts/download_sample_video.py          # -> data/videos/2e57b9_0.mp4

The sample clips (30 s, 1920 x 1080, 25 fps, main broadcast camera) come from the
"DFL - Bundesliga Data Shootout" (Kaggle, 2022), as hosted by Roboflow for its sports examples.
They are not part of this repository and must not be redistributed. A SoccerNet clip can also be
turned into a video, to run the video pipeline on a clip whose ground truth is known:

    python scripts/gsr_clip_to_video.py --clip data/SoccerNetGS/valid/SNGS-021 --out data/videos/SNGS-021.mp4


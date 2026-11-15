"""Analyse a football video: annotated video, tracks, positions on the pitch, statistics.

    python scripts/analyze_video.py --video data/videos/2e57b9_0.mp4 --line-model models/pitch_lines.pt \
        --out outputs/videos/2e57b9_0
    options: --start 5 --duration 10 (seconds), --weights yolo26n.pt, --device cpu, --no-cmc,
             --calib-every 5 (line network every 5 frames), --no-video, --live-teams

Without --line-model the pitch is not calibrated: boxes, tracks and teams only, no metres.
Outputs in --out: annotated.mp4 (re-encoded in H.264 when ffmpeg is on the PATH), positions.parquet,
calibration.parquet, player_stats.csv, team_stats.csv, summary.md, heatmaps.png, report.json.

The analysis runs frame by frame (VideoAnalyzer.process only uses the past); the video is then
drawn in a second pass with the teams refitted on the whole video, unless --live-teams is given,
in which case each frame shows the team estimate available at that time, as a live display would.
"""
import argparse
import shutil
import subprocess
import time
from pathlib import Path

from pitchtrack.analysis import VideoAnalyzer
from pitchtrack.camera_motion import CameraMotionEstimator
from pitchtrack.detector import build_detector
from pitchtrack.kalman import KalmanBoxFilter
from pitchtrack.pipeline import load_config
from pitchtrack.tracker import Tracker, TrackerConfig
from pitchtrack.video_io import VideoFileSource, VideoWriter
from pitchtrack.visualization import render_analysis


def resolve_device(device: str | None) -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if device and str(device).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available: running on the CPU")
        return "cpu"
    return device or ("cuda:0" if torch.cuda.is_available() else "cpu")


def in_window(t: float, start: float, end: float | None) -> bool:
    return t >= start and (end is None or t <= end)


def to_h264(src: Path, dst: Path) -> bool:
    """Re-encode with ffmpeg (H.264, playable in browsers); False if ffmpeg is missing or fails."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    return subprocess.run(cmd).returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--line-model", default=None, help="checkpoint of scripts/train_line_model.py")
    ap.add_argument("--detector-config", default="configs/baseline.yaml")
    ap.add_argument("--tracker-config", default="configs/tracker.yaml")
    ap.add_argument("--config", default="configs/analysis.yaml", help="teams, calibration, statistics")
    ap.add_argument("--weights", default=None, help="override detector.weights (e.g. yolo26n.pt on a CPU)")
    ap.add_argument("--imgsz", type=int, default=None, help="override detector.imgsz")
    ap.add_argument("--device", default=None, help="cuda:0, cpu ... (default: the detector config)")
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--duration", type=float, default=None, help="seconds")
    ap.add_argument("--no-cmc", action="store_true", help="no camera motion compensation in the tracker")
    ap.add_argument("--calib-every", type=int, default=1, help="run the line network every n frames")
    ap.add_argument("--no-video", action="store_true", help="statistics only")
    ap.add_argument("--live-teams", action="store_true", help="draw the live team estimate of each frame")
    args = ap.parse_args()

    det_cfg = load_config(args.detector_config)
    trk_cfg = load_config(args.tracker_config)
    cfg = load_config(args.config)
    device = resolve_device(args.device or det_cfg["detector"].get("device"))
    det_cfg["detector"]["device"] = device
    if args.weights:
        det_cfg["detector"]["weights"] = args.weights
    if args.imgsz:
        det_cfg["detector"]["imgsz"] = args.imgsz

    detector = build_detector(det_cfg)
    tcfg = TrackerConfig.from_dict(trk_cfg.get("tracker"))
    tracker = Tracker(tcfg, KalmanBoxFilter(**(trk_cfg.get("kalman") or {})))
    cmc = None
    if not args.no_cmc:
        cmc_cfg = {k: v for k, v in (trk_cfg.get("cmc") or {}).items() if k != "enabled"}
        cmc = CameraMotionEstimator(**cmc_cfg)
    annotator = None
    if args.line_model:
        from pitchtrack.line_model import LineAnnotator
        annotator = LineAnnotator(args.line_model, device=device)

    source = VideoFileSource(args.video)
    end = None if args.duration is None else args.start + args.duration
    analyzer = VideoAnalyzer(detector, tracker, source.size, annotator, cmc, cfg["calibration"], cfg["teams"],
                             cfg["statistics"], min_score=tcfg.low_thresh, pitch_radius=trk_cfg.get("pitch_radius"),
                             pitch_mask_cfg=det_cfg.get("pitch_mask"), calib_every=args.calib_every)
    print(f"{args.video}: {source.size[0]}x{source.size[1]}, {source.nominal_fps:.2f} fps, "
          f"{source.n_frames_declared} frames; detector {det_cfg['detector']['weights']} on {device}; "
          f"line model: {args.line_model or 'none (no calibration)'}")

    results, tic = {}, time.perf_counter()
    for frame in source:
        if frame.t < args.start:
            continue
        if end is not None and frame.t > end:
            break
        results[frame.index] = analyzer.process(frame)
        if len(results) % 50 == 0:
            r = results[frame.index]
            print(f"  frame {frame.index} t={frame.t:.1f}s: {len(r.ids)} tracks, pitch {r.calibration}, "
                  f"{len(results) / (time.perf_counter() - tic):.1f} frames/s")
    seconds = time.perf_counter() - tic
    report = analyzer.finish(info={"video": Path(args.video).name, "detector": det_cfg["detector"],
                                   "line_model": args.line_model, "cmc": cmc is not None,
                                   "tracker": vars(tcfg), "processing_s": seconds,
                                   "processing_fps": len(results) / max(seconds, 1e-9)})
    out = report.save(args.out)
    print(f"analysis: {len(results)} frames in {seconds:.0f} s ({len(results) / max(seconds, 1e-9):.1f} frames/s)")

    if not args.no_video and results:
        team_of = None if args.live_teams else report.team_of()
        raw = out / "annotated_mp4v.mp4"
        title = Path(args.video).stem
        with VideoWriter(raw, source.nominal_fps, source.size) as writer:
            for frame in VideoFileSource(args.video):
                if frame.index in results:
                    writer.write(render_analysis(frame.image, results[frame.index], team_of, title))
                elif end is not None and frame.t > end:
                    break
        final = out / "annotated.mp4"
        if to_h264(raw, final):
            raw.unlink()
        else:
            raw.replace(final)
        print(f"video -> {final}")
    print("\n".join(report.summary_markdown().splitlines()[2:-3]))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

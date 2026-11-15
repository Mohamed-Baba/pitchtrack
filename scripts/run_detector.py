"""Run the detector on a split and cache all detections (low score threshold).

    python scripts/run_detector.py --split valid --stride 5 --out outputs/det/valid_s5.parquet
    options: --limit 2 (first clips only), --imgsz 1920, --tiled, --weights best.pt, --config ...

Thresholds are NOT applied here (conf is low on purpose): they are swept offline (eval_detection.py).
"""
import argparse

from pitchtrack.detector import build_detector
from pitchtrack.pipeline import detect_split, load_config, save_detections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--stride", type=int, default=None, help="default: data.eval_stride")
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips")
    ap.add_argument("--imgsz", type=int, default=None, help="override detector.imgsz")
    ap.add_argument("--weights", default=None, help="override detector.weights")
    ap.add_argument("--tiled", action="store_true", help="enable tiling.enabled")
    ap.add_argument("--no-mask", action="store_true", help="disable the pitch mask distances")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.imgsz:
        cfg["detector"]["imgsz"] = args.imgsz
    if args.weights:
        cfg["detector"]["weights"] = args.weights
    if args.tiled:
        cfg.setdefault("tiling", {})["enabled"] = True
    if args.no_mask:
        cfg.setdefault("pitch_mask", {})["enabled"] = False
    stride = args.stride or cfg["data"].get("eval_stride", 5)

    detector = build_detector(cfg)
    table, info = detect_split(cfg, detector, args.split, stride, args.limit,
                               progress=lambda name, n: print(f"{name}: {n} frames done"))
    save_detections(table, info, args.out)
    print(f"{info['n_detections']} detections on {info['n_frames']} frames, "
          f"{info['latency_ms_per_frame']:.1f} ms/frame -> {args.out}")


if __name__ == "__main__":
    main()

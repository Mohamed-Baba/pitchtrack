"""Export SoccerNet-GSR frames to the YOLO format to fine-tune the detector.

    python scripts/export_yolo.py --out datasets/gsr_yolo        # train -> train, valid -> val, stride 5
    yolo detect train data=datasets/gsr_yolo/gsr.yaml model=yolo26m.pt imgsz=1280 epochs=30 batch=8

Train on `train` (3 games) and validate on `valid` (3 other games); `test` stays untouched
until the final evaluation. Images are hard-linked when possible (no copy, same disk), copied otherwise.
One class: person (all roles). Label line: "0 cx cy w h", normalised by the image size.
"""
import argparse
import os
import shutil
from pathlib import Path

from pitchtrack.boxes import to_yolo_labels
from pitchtrack.pipeline import frames_to_process
from pitchtrack.soccernet import iter_sequences


def export(root, split, subset, stride, out):
    img_dir, lab_dir = out / "images" / subset, out / "labels" / subset
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for seq in iter_sequences(root, split):
        for f in frames_to_process(seq, stride):
            src = seq.frame_path(int(f))
            dst = img_dir / f"{seq.info.name}_{src.name}"
            if not dst.exists():
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
            gt = seq.gt_frame(int(f))
            lines = to_yolo_labels(gt[["x1", "y1", "x2", "y2"]].to_numpy(), seq.info.width, seq.info.height)
            (lab_dir / dst.with_suffix(".txt").name).write_text("\n".join(lines), encoding="utf-8")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/SoccerNetGS")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="valid")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", default="datasets/gsr_yolo")
    args = ap.parse_args()
    out = Path(args.out)
    n_tr = export(args.root, args.train_split, "train", args.stride, out)
    n_va = export(args.root, args.val_split, "val", args.stride, out)
    (out / "gsr.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n",
        encoding="utf-8")
    print(f"{n_tr} train and {n_va} val images -> {out / 'gsr.yaml'}")


if __name__ == "__main__":
    main()

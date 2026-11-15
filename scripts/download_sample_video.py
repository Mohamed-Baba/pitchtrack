"""Download a sample broadcast clip to try the pipeline on a video without annotations.

    python scripts/download_sample_video.py                  # -> data/videos/2e57b9_0.mp4
    python scripts/download_sample_video.py --clip 08fd33_0
    python scripts/download_sample_video.py --all

Five 30-second clips (1920 x 1080, 25 fps, main camera) from the "DFL - Bundesliga Data Shootout"
(Kaggle, 2022), as hosted by Roboflow for its sports examples (github.com/roboflow/sports,
examples/soccer/setup.sh). They are not part of this repository: use them to test, do not
redistribute them. `pip install gdown` makes the Google Drive download more robust.
"""
import argparse
import urllib.request
from pathlib import Path

import cv2

CLIPS = {
    "2e57b9_0": "19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf",
    "08fd33_0": "1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-",
    "0bfacc_0": "12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF",
    "121364_0": "1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu",
    "573e61_0": "1yYPKuXbHsCxqjA9G-S6aeR2Kcnos8RPU",
}


def is_video(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1_000_000:
        return False
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    return bool(ok)


def download(file_id: str, dst: Path) -> bool:
    try:
        import gdown
        gdown.download(id=file_id, output=str(dst), quiet=False)
    except ImportError:
        url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
        try:
            urllib.request.urlretrieve(url, dst)
        except OSError as e:
            print(f"download failed: {e}")
    except Exception as e:  # gdown raises its own errors when Google Drive refuses the request
        print(f"download failed: {e}")
    return is_video(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="2e57b9_0", choices=sorted(CLIPS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out-dir", default="data/videos")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in (sorted(CLIPS) if args.all else [args.clip]):
        dst = out / f"{name}.mp4"
        if is_video(dst):
            print(f"{dst} already there")
            continue
        if download(CLIPS[name], dst):
            print(f"{name} -> {dst}")
        else:
            print(f"{name}: could not download it. Get it by hand from "
                  f"https://drive.google.com/uc?id={CLIPS[name]} and save it as {dst}")


if __name__ == "__main__":
    main()

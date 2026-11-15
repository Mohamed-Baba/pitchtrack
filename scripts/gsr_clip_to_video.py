"""Turn a SoccerNet-GSR clip (img1/*.jpg) into a video file, to run analyze_video.py on a clip
whose ground truth is known.

    python scripts/gsr_clip_to_video.py --clip data/SoccerNetGS/valid/SNGS-021 --out data/videos/SNGS-021.mp4

Keep the result local: SoccerNet frames must not be redistributed.
"""
import argparse
from pathlib import Path

from pitchtrack.soccernet import load_sequence
from pitchtrack.video_io import ImageFolderSource, VideoWriter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, help="clip folder (img1/ and Labels-GameState.json)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seq = load_sequence(Path(args.clip))
    source = ImageFolderSource(seq.info.img_dir, seq.info.fps)
    with VideoWriter(args.out, seq.info.fps, (seq.info.width, seq.info.height)) as writer:
        for frame in source:
            writer.write(frame.image)
    print(f"{len(source)} frames at {seq.info.fps:g} fps -> {args.out}")


if __name__ == "__main__":
    main()

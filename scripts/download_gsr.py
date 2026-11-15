"""Download and unzip SoccerNet-GSR splits (Hugging Face dataset SoccerNet/SN-GSR-2024).

    python scripts/download_gsr.py --root data/SoccerNetGS --splits valid         # 11.2 GB
    python scripts/download_gsr.py --root data/SoccerNetGS --splits train         #  9.8 GB (training of the detector and of the line network)
    python scripts/download_gsr.py --root data/SoccerNetGS --splits test test_labels   # final numbers, once

Needs `pip install SoccerNet` (it uses huggingface_hub). If the download is refused,
log in with `hf auth login` and accept the dataset conditions on its Hugging Face page.
Check the free disk space first: the zip and the unzipped folder coexist for a while.
"""
import argparse
import shutil
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/SoccerNetGS")
    ap.add_argument("--splits", nargs="+", default=["valid"])
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(root).free / 1e9
    print(f"free disk space under {root}: {free_gb:.1f} GB")

    from SoccerNet.Downloader import SoccerNetDownloader

    SoccerNetDownloader(LocalDirectory=str(root)).downloadDataTask(task="gamestate-2024", split=args.splits)
    for split in args.splits:
        zips = [root / "gamestate-2024" / f"{split}.zip", root / "gamestate-2024" / f"{split}_labels.zip"]
        for z in zips:
            if not z.exists():
                continue
            target = root / split.replace("_labels", "")
            print(f"unzipping {z} -> {target}")
            with zipfile.ZipFile(z) as f:
                f.extractall(target)
            if not args.keep_zip:
                z.unlink()
    for split in ("train", "valid", "test", "challenge"):
        d = root / split
        if d.is_dir():
            n = sum(1 for p in d.iterdir() if p.is_dir())
            print(f"{split}: {n} clip folders")
            nested = d / split
            if nested.is_dir():
                print(f"  note: found {nested}; move its content one level up if the clips are there")


if __name__ == "__main__":
    main()

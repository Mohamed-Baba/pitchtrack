"""Evaluate a tracks table against SoccerNet-GSR -> markdown report + figure.

    python scripts/eval_tracking.py --tracks outputs/trk/valid_base.parquet

HOTA, DetA, AssA, LocA (averaged over 19 thresholds alpha), and if TrackEval is installed
MOTA, IDSW and IDF1. 95 % intervals by resampling CLIPS. The report goes to
reports/tracking_<tracks name>.md unless --out is given.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pitchtrack.mot_metrics import (ALPHAS, clear_identity, combine_clear_identity, combine_hota,  # noqa: E402
                                    hota, sequence_data, summarize)
from pitchtrack.pipeline import frames_to_process  # noqa: E402
from pitchtrack.soccernet import list_sequences, load_sequence  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    tracks_path = Path(args.tracks)
    tracks = pd.read_parquet(tracks_path)
    run = json.loads(tracks_path.with_suffix(".json").read_text(encoding="utf-8"))
    out = Path(args.out or f"reports/tracking_{tracks_path.stem}.md")

    per_clip, h_res, ci_res = [], [], []
    for path in list_sequences(run["data_root"], run["split"]):
        if path.name not in run["clips"]:
            continue
        seq = load_sequence(path, run["split"])
        frames = frames_to_process(seq, run["stride"])
        data = sequence_data(seq.gt, tracks[tracks["seq"] == seq.info.name], frames, seq.info.name)
        res, extra = hota(data), clear_identity(data)
        h_res.append(res)
        ci_res.append(extra)
        per_clip.append({"clip": seq.info.name, **summarize(res),
                         **{k: extra[k] for k in ("IDF1", "MOTA", "IDSW") if k in extra}})

    combined = combine_hota(h_res)
    summary = summarize(combined)
    pooled = combine_clear_identity(ci_res)
    rng = np.random.default_rng(0)
    boot = {"HOTA": [], "DetA": [], "AssA": [], "IDF1": []}
    for _ in range(args.boot):
        idx = rng.integers(0, len(h_res), len(h_res))
        s = summarize(combine_hota([h_res[i] for i in idx]))
        for k in ("HOTA", "DetA", "AssA"):
            boot[k].append(s[k])
        if pooled:
            boot["IDF1"].append(combine_clear_identity([ci_res[i] for i in idx])["IDF1"])
    ci = {k: np.percentile(v, [2.5, 97.5]) for k, v in boot.items() if v}

    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for k in ("HOTA", "DetA", "AssA", "LocA"):
        ax.plot(ALPHAS, combined[k], label=k)
    ax.set(xlabel="localisation threshold alpha", ylim=(0, 1.02), title=tracks_path.stem)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=130)

    def fmt(k):
        v = summary.get(k, pooled.get(k))
        return f"{v:.3f}" + (f" [{ci[k][0]:.3f}, {ci[k][1]:.3f}]" if k in ci else "")

    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# Tracking report -- {tracks_path.name}\n\n")
        f.write(f"Split {run['split']}, {len(h_res)} clips, oracle detections: {run['oracle']}, CMC: {run['cmc']}, "
                f"tracking {run['tracking_ms_per_frame']:.2f} ms/frame. Tracker: `{run['tracker']}`.\n\n")
        row = [fmt("HOTA"), fmt("DetA"), fmt("AssA"), f"{summary['LocA']:.3f}"]
        row += [fmt("IDF1"), f"{pooled['MOTA']:.3f}", str(pooled["IDSW"])] if pooled else ["n/a"] * 3
        f.write("| HOTA | DetA | AssA | LocA | IDF1 | MOTA | IDSW |\n|---|---|---|---|---|---|---|\n")
        f.write("| " + " | ".join(row) + " |\n\n")
        f.write("95 % intervals: clips resampled with replacement. HOTA, DetA and AssA are averaged over "
                "alpha in {0.05, ..., 0.95}.\n\n")
        f.write("## Per clip\n\n" + pd.DataFrame(per_clip).to_markdown(index=False, floatfmt=".3f") + "\n\n")
        f.write(f"![alpha curves]({out.with_suffix('.png').name})\n")
    print(f"HOTA {fmt('HOTA')} | DetA {fmt('DetA')} | AssA {fmt('AssA')}"
          + (f" | IDF1 {fmt('IDF1')} | IDSW {pooled['IDSW']}" if pooled else "") + f" -> {out}")


if __name__ == "__main__":
    main()

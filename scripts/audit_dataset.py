"""Audit the annotations of one or more splits -> reports/data_audit.md (+ CSV).

    python scripts/audit_dataset.py --root data/SoccerNetGS --splits valid
"""
import argparse
from pathlib import Path

import pandas as pd

from pitchtrack.audit import audit_sequence, check_game_disjoint, summarize_splits
from pitchtrack.soccernet import list_sequences, load_sequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/SoccerNetGS")
    ap.add_argument("--splits", nargs="+", default=["valid"])
    ap.add_argument("--small-px", type=float, default=40.0)
    ap.add_argument("--out", default="reports/data_audit.md")
    args = ap.parse_args()

    rows, versions = [], set()
    for split in args.splits:
        for path in list_sequences(args.root, split):
            seq = load_sequence(path, split)
            versions.add(seq.info.version)
            rows.append(audit_sequence(seq, small_px=args.small_px))
            print(f"{split}/{seq.info.name}: {rows[-1]['n_boxes']} boxes, game {seq.info.game_id}")
    audit = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out.with_suffix(".csv"), index=False)

    shared = check_game_disjoint(audit)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Data audit -- SoccerNet-GSR\n\n")
        f.write(f"Dataset versions found: {sorted(versions)} (expected >= 1.3)\n\n")
        f.write("## Per split\n\n" + summarize_splits(audit).to_markdown(floatfmt=".3g") + "\n\n")
        f.write("## Games shared between splits\n\n")
        f.write("None: the splits are game-disjoint.\n\n" if shared.empty else shared.to_markdown() + "\n\n")
        cols = ["name", "split", "game_id", "n_boxes", "n_tracks", "boxes_per_frame", "height_p05",
                "height_p50", "height_p95", "frac_small", "frac_border", "frac_no_pitch", "track_len_median"]
        f.write("## Per clip\n\n" + audit[cols].to_markdown(index=False, floatfmt=".3g") + "\n")
    print(f"wrote {out} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()

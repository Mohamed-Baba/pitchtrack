"""Train the pitch-line network on SoccerNet-GSR -> checkpoint + training log.

    python scripts/train_line_model.py --out models/pitch_lines.pt
    python scripts/train_line_model.py --epochs 1 --limit 2 --max-iters 20 --out models/smoke.pt   # smoke test

Training frames come from the train split (one in train_stride), validation frames from the valid
split. After every epoch: validation loss and the F1 of the predicted lines with a tolerance of a
few pixels, per class; the checkpoint with the best mean F1 is written to --out, the last one to
<out>_last.pt, the history to <out>.json. The decisive evaluation is the calibration it gives:

    python scripts/calibrate.py --split valid --line-model models/pitch_lines.pt --out outputs/calib/valid_model.parquet
    python scripts/eval_calibration.py --calib outputs/calib/valid_model.parquet
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from pitchtrack.line_annotations import CLASSES
from pitchtrack.line_model import (LineDataset, LineNet, f1_from_counts, gsr_line_items, line_loss, save_checkpoint,
                                   tolerant_counts)
from pitchtrack.pipeline import load_config


def evaluate(model, loader, device, amp_dtype, cfg):
    model.eval()
    counts, losses = torch.zeros(4, len(CLASSES) + 1, dtype=torch.float64), []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                logits = model(x)
            losses.append(float(line_loss(logits, y, cfg["train"]["dice_weight"])))
            counts += tolerant_counts(logits.argmax(dim=1), y, cfg["train"]["tolerance_px"]).cpu()
    f1, mean_f1 = f1_from_counts(counts.numpy())
    return float(np.mean(losses)) if losses else float("nan"), f1, mean_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/line_model.yaml")
    ap.add_argument("--root", default=None, help="override data.root")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-pretrained", action="store_true", help="random initial encoder")
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips of each split")
    ap.add_argument("--max-iters", type=int, default=None, help="iterations per epoch (smoke tests)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tr = cfg["train"]
    for key in ("epochs", "batch", "lr", "workers", "device"):
        if getattr(args, key) is not None:
            tr[key] = getattr(args, key)
    if args.root:
        cfg["data"]["root"] = args.root
    if args.no_pretrained:
        cfg["model"]["pretrained"] = False
    device = torch.device(tr["device"] if torch.cuda.is_available() or not str(tr["device"]).startswith("cuda")
                          else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

    d, m = cfg["data"], cfg["model"]
    size = tuple(m["input_size"])
    train_items = gsr_line_items(d["root"], d["train_split"], d["train_stride"], args.limit)
    val_items = gsr_line_items(d["root"], d["val_split"], d["val_stride"], args.limit)
    print(f"{len(train_items)} training frames ({d['train_split']}), {len(val_items)} validation frames "
          f"({d['val_split']}), input {size[0]}x{size[1]}, device {device}")
    loader_args = dict(batch_size=tr["batch"], num_workers=tr["workers"], pin_memory=device.type == "cuda",
                       persistent_workers=tr["workers"] > 0)
    train_loader = torch.utils.data.DataLoader(LineDataset(train_items, size, m["thickness"], cfg["augment"]),
                                               shuffle=True, drop_last=True, **loader_args)
    val_loader = torch.utils.data.DataLoader(LineDataset(val_items, size, m["thickness"]), shuffle=False,
                                             **loader_args)

    model = LineNet(m["encoder"], pretrained=m["pretrained"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    per_epoch = min(len(train_loader), args.max_iters or len(train_loader))
    total, warmup = max(1, tr["epochs"] * per_epoch), max(1, min(500, per_epoch))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, s / total))))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda" and amp_dtype == torch.float16)

    out = Path(args.out)
    ckpt_cfg = {**m, "pretrained": False, "postprocess": cfg["postprocess"]}
    history, best = [], -np.inf
    for epoch in range(1, tr["epochs"] + 1):
        model.train()
        tic, losses = time.perf_counter(), []
        for it, (x, y) in enumerate(train_loader):
            if it >= per_epoch:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                logits = model(x)
            loss = line_loss(logits, y, tr["dice_weight"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss.detach()))
            if (it + 1) % 100 == 0:
                print(f"  epoch {epoch} it {it + 1}/{per_epoch} loss {np.mean(losses[-100:]):.4f}")
        val_loss, f1, mean_f1 = evaluate(model, val_loader, device, amp_dtype, cfg)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "val_f1": mean_f1,
               "val_f1_per_class": {c: (None if np.isnan(v) else round(float(v), 4)) for c, v in zip(CLASSES, f1)},
               "seconds": time.perf_counter() - tic, "lr": opt.param_groups[0]["lr"]}
        history.append(row)
        save_checkpoint(out.with_name(out.stem + "_last.pt"), model, ckpt_cfg, epoch=epoch, metrics=row)
        score = mean_f1 if np.isfinite(mean_f1) else -1.0
        if score > best:
            best = score
            save_checkpoint(out, model, ckpt_cfg, epoch=epoch, metrics=row)
        out.with_suffix(".json").write_text(json.dumps({"config": cfg, "history": history}, indent=2),
                                            encoding="utf-8")
        print(f"epoch {epoch}: train loss {row['train_loss']:.4f}, val loss {val_loss:.4f}, "
              f"val F1 {mean_f1:.3f} (best {best:.3f}), {row['seconds']:.0f} s")
    print(f"best validation F1 {best:.3f} -> {out}")


if __name__ == "__main__":
    main()

"""Pitch-line network: a frame in, the pitch-line annotation of GSR out.

    frame (BGR) -> LineNet -> label map (20 line classes + background) and confidence
                -> line_annotations.label_map_to_lines -> {"Middle line": [{"x": .., "y": ..}, ...], ...}

The output has the format of the SoccerNet-GSR annotations, so `pitch_calibration.calibrate_frame`
consumes it unchanged: the network replaces the annotated lines, nothing downstream changes.

LineNet is a U-Net: a torchvision ResNet encoder (strides 2 to 32) and a decoder that upsamples
back to stride 2 with skip connections; the logits are upsampled to the input size. The deepest
features see the whole frame, which is what separates "Big rect. left main" from "Big rect. right
main": locally the two lines look the same. Training target: the annotated polylines drawn with
their labels (line_annotations.render_label_map); loss: cross-entropy + soft Dice on the line
classes, because line pixels are about 1 % of the image.

Requires torch and torchvision (not installed by `pip install pitchtrack`: install the build
that matches your GPU first).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .line_annotations import (CLASSES, NUM_LABELS, crop_points, flip_points, label_map_to_lines,
                               normalised_points, render_label_map)
from .soccernet import frame_index, list_sequences

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)     # ImageNet statistics, RGB
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ENCODER_CHANNELS = {                                          # stem, layer1 .. layer4
    "resnet18": (64, 64, 128, 256, 512),
    "resnet34": (64, 64, 128, 256, 512),
    "resnet50": (64, 256, 512, 1024, 2048),
}
DEFAULT_POSTPROCESS = {"min_conf": 0.5, "min_pixels": 30, "max_points": 40, "max_run": 8, "line_tol_px": 3.0}


# --------------------------------------------------------------------------- network
class _UpBlock(nn.Module):
    """Upsample to the size of the skip connection, concatenate, two 3x3 convolutions."""

    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in + c_skip, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class LineNet(nn.Module):
    """U-Net with a ResNet encoder. (B, 3, H, W) normalised RGB -> (B, NUM_LABELS, H, W) logits.
    H and W must be multiples of 32."""

    def __init__(self, encoder: str = "resnet34", pretrained: bool = False, num_labels: int = NUM_LABELS):
        super().__init__()
        import torchvision

        if encoder not in ENCODER_CHANNELS:
            raise ValueError(f"encoder must be one of {sorted(ENCODER_CHANNELS)}")
        net = getattr(torchvision.models, encoder)(weights="IMAGENET1K_V1" if pretrained else None)
        c0, c1, c2, c3, c4 = ENCODER_CHANNELS[encoder]
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)          # stride 2
        self.pool = net.maxpool                                           # stride 4
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        self.up4 = _UpBlock(c4, c3, 256)                                  # stride 16
        self.up3 = _UpBlock(256, c2, 128)                                 # stride 8
        self.up2 = _UpBlock(128, c1, 64)                                  # stride 4
        self.up1 = _UpBlock(64, c0, 32)                                   # stride 2
        self.head = nn.Conv2d(32, num_labels, 1)

    def forward(self, x):
        size = x.shape[-2:]
        x0 = self.stem(x)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        d = self.up1(self.up2(self.up3(self.up4(x4, x3), x2), x1), x0)
        return F.interpolate(self.head(d), size=size, mode="bilinear", align_corners=False)


def line_loss(logits, target, dice_weight: float = 1.0):
    """Cross-entropy + soft Dice averaged over the line classes present in the batch.

    logits (B, L, H, W), target (B, H, W) int64. The Dice term of class c is
    1 - (2 sum p_c y_c + 1) / (sum p_c + sum y_c + 1); it ignores the background, so the 99 % of
    background pixels cannot drown the thin lines.
    """
    logits = logits.float()
    ce = F.cross_entropy(logits, target)
    if dice_weight == 0:
        return ce
    prob = logits.softmax(dim=1)
    n_labels = prob.shape[1]
    t = target.reshape(-1)
    p_true = prob.gather(1, target.unsqueeze(1)).reshape(-1)          # probability of the true class
    inter = torch.zeros(n_labels, dtype=prob.dtype, device=prob.device).scatter_add(0, t, p_true)
    n_true = torch.bincount(t, minlength=n_labels).to(prob.dtype)
    mass = prob.sum(dim=(0, 2, 3))
    present = n_true[1:] > 0
    if not present.any():
        return ce + 0.0 * mass.sum()
    dice = 1.0 - (2.0 * inter[1:] + 1.0) / (mass[1:] + n_true[1:] + 1.0)
    return ce + dice_weight * dice[present].mean()


@torch.no_grad()
def tolerant_counts(pred, target, radius: int = 2, num_labels: int = NUM_LABELS) -> torch.Tensor:
    """Per-class counts for a precision / recall with a tolerance of `radius` pixels: a predicted
    pixel is correct if a true pixel of its class is within the radius, and conversely.
    Returns (4, num_labels): correct predicted, predicted, found true, true."""
    out = torch.zeros(4, num_labels, dtype=torch.float64)
    k = 2 * radius + 1
    for c in range(1, num_labels):
        p, t = pred == c, target == c
        n_p, n_t = int(p.sum()), int(t.sum())
        if n_p == 0 and n_t == 0:
            continue
        p_d = F.max_pool2d(p.unsqueeze(1).float(), k, 1, radius).squeeze(1) > 0
        t_d = F.max_pool2d(t.unsqueeze(1).float(), k, 1, radius).squeeze(1) > 0
        out[:, c] += torch.tensor([float((p & t_d).sum()), n_p, float((t & p_d).sum()), n_t], dtype=torch.float64)
    return out


def f1_from_counts(counts) -> tuple[np.ndarray, float]:
    """Per-class tolerant F1 (NaN for classes never seen) and its mean over the seen classes."""
    c = np.asarray(counts, dtype=np.float64)
    prec = c[0] / np.maximum(c[1], 1.0)
    rec = c[2] / np.maximum(c[3], 1.0)
    f1 = np.where(c[3] > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), np.nan)[1:]
    return f1, float(np.nanmean(f1)) if np.isfinite(f1).any() else float("nan")


# --------------------------------------------------------------------------- data
def to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 (H, W, 3) -> normalised RGB float tensor (3, H, W)."""
    rgb = image_bgr[:, :, ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(((rgb - MEAN) / STD).transpose(2, 0, 1)))


def preprocess(images, size_wh) -> torch.Tensor:
    """List of BGR frames of any size -> (B, 3, h, w) batch at the network input size."""
    w, h = size_wh
    return torch.stack([to_tensor(cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)) for im in images])


def gsr_line_items(root, split: str, stride: int = 1, limit: int | None = None) -> list:
    """(image path, raw "lines" dict) for the frames of a GSR split whose pitch is annotated.

    A frame annotated with no visible line (close-up, replay) is kept: it teaches the network to
    answer nothing. Frames flagged has_labeled_pitch = False are skipped.
    """
    items = []
    for seq_dir in list_sequences(root, split)[:limit]:
        with open(seq_dir / "Labels-GameState.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        im_dir = seq_dir / data.get("info", {}).get("im_dir", "img1")
        frame_of, path_of, labeled = {}, {}, set()
        for img in data.get("images", []):
            fr = frame_index(img["file_name"])
            frame_of[img["image_id"]] = fr
            path_of[fr] = str(im_dir / img["file_name"])
            if img.get("has_labeled_pitch", True):
                labeled.add(fr)
        for ann in data.get("annotations", []):
            if ann.get("supercategory") != "pitch" or "lines" not in ann:
                continue
            fr = frame_of[ann["image_id"]]
            if fr % stride == 0 and fr in labeled:
                items.append((path_of[fr], ann["lines"]))
    return items


class LineDataset(torch.utils.data.Dataset):
    """(image path, lines) items -> (normalised image (3, h, w), label map (h, w) int64).

    Augmentations (training): a random zoom (crop of 1 / s of the frame, s uniform in
    [1, zoom_max], resized back), a horizontal flip with the left / right classes swapped, colour
    jitter, blur and noise. The annotation follows every geometric change, then the target is drawn.
    """

    def __init__(self, items, size_wh=(960, 544), thickness: int = 3, augment: dict | None = None):
        self.items, self.size, self.thickness = list(items), tuple(size_wh), int(thickness)
        self.augment = augment or {}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        path, lines = self.items[i]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        points = normalised_points(lines)
        aug, rng = self.augment, np.random.default_rng()
        if aug and rng.random() < aug.get("zoom", 0.0):
            image, points = _random_zoom(image, points, rng, aug.get("zoom_max", 1.5))
        if aug and rng.random() < aug.get("hflip", 0.0):
            image, points = np.ascontiguousarray(image[:, ::-1]), flip_points(points)
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        if aug:
            image = _photometric(image, rng, aug)
        target = render_label_map(points, self.size, self.thickness)
        return to_tensor(image), torch.from_numpy(target.astype(np.int64))


def _random_zoom(image, points, rng, zoom_max: float):
    H, W = image.shape[:2]
    s = rng.uniform(1.0, zoom_max)
    cw, ch = max(2, int(round(W / s))), max(2, int(round(H / s)))
    x0, y0 = int(rng.integers(0, W - cw + 1)), int(rng.integers(0, H - ch + 1))
    box = (x0 / (W - 1), y0 / (H - 1), (x0 + cw - 1) / (W - 1), (y0 + ch - 1) / (H - 1))
    return image[y0:y0 + ch, x0:x0 + cw], crop_points(points, box)


def _photometric(image, rng, aug: dict):
    if rng.random() < aug.get("colour", 0.0):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-6, 6)) % 180
        hsv[..., 1] *= rng.uniform(0.7, 1.3)
        hsv[..., 2] = hsv[..., 2] * rng.uniform(0.7, 1.3) + rng.uniform(-20, 20)
        image = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < aug.get("blur", 0.0):
        k = int(rng.choice([3, 5]))
        image = cv2.GaussianBlur(image, (k, k), 0)
    if rng.random() < aug.get("noise", 0.0):
        image = np.clip(image + rng.normal(0, rng.uniform(2, 6), image.shape), 0, 255).astype(np.uint8)
    return image


# --------------------------------------------------------------------------- checkpoints
def save_checkpoint(path, model: LineNet, config: dict, **extra) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": {**config, "classes": list(CLASSES)}, **extra}, path)


def load_checkpoint(path, map_location="cpu") -> tuple[LineNet, dict]:
    """(model in eval mode, config) from a checkpoint written by save_checkpoint."""
    ckpt = torch.load(path, map_location=map_location)
    cfg = ckpt["config"]
    if list(cfg.get("classes", CLASSES)) != list(CLASSES):
        raise ValueError(f"{path}: trained with other line classes than this version of pitchtrack")
    model = LineNet(cfg.get("encoder", "resnet34"), pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.eval(), cfg


# --------------------------------------------------------------------------- inference
class LineAnnotator:
    """A trained LineNet behind the annotation contract: BGR frames -> GSR-style line dicts.

        annotator = LineAnnotator("models/pitch_lines.pt")
        lines = annotator([frame])[0]             # {"Side line top": [{"x": .., "y": ..}, ...], ...}
        cal = calibrate_frame(lines, (width, height))
    """

    def __init__(self, checkpoint, device=None, half: bool = True, **postprocess):
        self.model, self.config = load_checkpoint(checkpoint)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.half = bool(half) and self.device.type == "cuda"
        self.size = tuple(self.config.get("input_size", (960, 544)))
        self.postprocess = {**DEFAULT_POSTPROCESS, **self.config.get("postprocess", {}), **postprocess}

    @torch.no_grad()
    def label_maps(self, images) -> tuple[np.ndarray, np.ndarray]:
        """(B, h, w) uint8 labels and (B, h, w) float32 confidences at the network input size."""
        x = preprocess(images, self.size).to(self.device)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.half):
            logits = self.model(x)
        conf, labels = logits.float().softmax(dim=1).max(dim=1)
        return labels.to(torch.uint8).cpu().numpy(), conf.cpu().numpy()

    def __call__(self, images) -> list:
        labels, conf = self.label_maps(images)
        return [label_map_to_lines(l, c, **self.postprocess) for l, c in zip(labels, conf)]

"""The line network: shapes, loss, data pipeline, a short overfit, checkpoints and inference."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from conftest import make_line_clip  # noqa: E402

from pitchtrack.line_annotations import FLIP_LUT, NUM_LABELS  # noqa: E402
from pitchtrack.line_model import (LineAnnotator, LineDataset, LineNet, f1_from_counts, gsr_line_items,  # noqa: E402
                                   line_loss, save_checkpoint, tolerant_counts)

VIEWS = [((x, 0.0), (x + 8.0, 74.0, -18.0)) for x in (-38.0, -30.0, -22.0, 22.0, 30.0, 38.0)]


@pytest.fixture
def line_root(tmp_path):
    make_line_clip(tmp_path, "train", "SNGS-801", VIEWS)
    return tmp_path


def test_linenet_shapes():
    net = LineNet("resnet18").eval()
    with torch.no_grad():
        out = net(torch.zeros(2, 3, 64, 96))
    assert out.shape == (2, NUM_LABELS, 64, 96)


def test_line_loss_is_small_for_a_confident_correct_prediction():
    target = torch.zeros(1, 32, 32, dtype=torch.long)
    target[0, 10:13, :] = 3
    good = torch.nn.functional.one_hot(target, NUM_LABELS).permute(0, 3, 1, 2).float() * 20.0
    bad = torch.zeros_like(good)
    assert float(line_loss(good, target)) < 0.05 < float(line_loss(bad, target))


def test_tolerant_f1():
    target = torch.zeros(1, 20, 20, dtype=torch.long)
    target[0, 10, :] = 1
    pred = torch.zeros_like(target)
    pred[0, 11, :] = 1                                     # one pixel off: correct with a tolerance of 2
    f1, mean = f1_from_counts(tolerant_counts(pred, target, radius=2).numpy())
    assert mean == pytest.approx(1.0) and np.isnan(f1[1:]).all()
    f1, mean = f1_from_counts(tolerant_counts(pred, target, radius=0).numpy())
    assert mean == pytest.approx(0.0)


def test_gsr_items_and_flip_augmentation(line_root):
    items = gsr_line_items(line_root, "train")
    assert len(items) == len(VIEWS) and all(isinstance(lines, dict) and lines for _, lines in items)
    plain = LineDataset(items, (192, 96), 2)
    flipped = LineDataset(items, (192, 96), 2, augment={"hflip": 1.0})
    x, y = plain[0]
    xf, yf = flipped[0]
    assert x.shape == (3, 96, 192) and y.shape == (96, 192) and y.dtype == torch.int64
    assert torch.allclose(xf, x.flip(-1), atol=1e-5)
    assert (FLIP_LUT[y.numpy()[:, ::-1]] == yf.numpy()).mean() > 0.99


def test_short_training_lowers_the_loss_and_checkpoint_roundtrip(line_root, tmp_path):
    torch.manual_seed(0)
    data = LineDataset(gsr_line_items(line_root, "train"), (192, 96), 2)
    x = torch.stack([data[i][0] for i in range(len(data))])
    y = torch.stack([data[i][1] for i in range(len(data))])
    net = LineNet("resnet18")
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    losses = []
    for _ in range(30):
        loss = line_loss(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < 0.7 * losses[0]

    path = tmp_path / "lines.pt"
    save_checkpoint(path, net, {"encoder": "resnet18", "input_size": [192, 96]})
    annotator = LineAnnotator(path, device="cpu")
    frame = np.zeros((272, 480, 3), np.uint8)
    labels, conf = annotator.label_maps([frame, frame])
    assert labels.shape == (2, 96, 192) and labels.dtype == np.uint8 and conf.shape == (2, 96, 192)
    out = annotator([frame])
    assert isinstance(out, list) and len(out) == 1 and isinstance(out[0], dict)

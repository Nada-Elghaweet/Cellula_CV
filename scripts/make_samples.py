import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference import predict_mask  # noqa: E402

DATA = ROOT / "data"
SAMPLES = ROOT / "samples"
RESULTS = ROOT / "results"
SAMPLES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

ids = json.load(open(DATA / "ids.json"))
X = np.load(DATA / "features_12.npy")
Y = np.load(DATA / "masks.npy")

with open(DATA / "split.csv", newline="") as f:
    rows = list(csv.DictReader(f))
assert [str(r["id"]) for r in rows] == [str(i) for i in ids], "split.csv and ids.json order differ"
test_idx = [i for i, r in enumerate(rows) if r["split"] == "test"]
print("test images:", len(test_idx))

water = np.array([Y[i].mean() for i in test_idx])
order = np.argsort(water)
picks = [test_idx[order[int(q * (len(order) - 1))]] for q in [0.1, 0.3, 0.5, 0.7, 0.9]]


def metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    return float(iou), float(f1)


lines = ["id,true_water_pct,pred_water_pct,iou,f1"]
for gi in picks:
    sid = ids[gi]
    np.save(SAMPLES / f"sample_{sid}.npy", X[gi].astype(np.float32))
    np.save(SAMPLES / f"sample_{sid}_mask.npy", Y[gi].astype(np.uint8))
    pred = predict_mask(X[gi])
    iou, f1 = metrics(pred, Y[gi])
    lines.append(f"{sid},{Y[gi].mean() * 100:.2f},{pred.mean() * 100:.2f},{iou:.4f},{f1:.4f}")

(RESULTS / "sample_check.csv").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(sorted(p.name for p in SAMPLES.iterdir()))
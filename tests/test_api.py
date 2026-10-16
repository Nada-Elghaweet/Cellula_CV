import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
URL = "http://127.0.0.1:5000"
SAMPLES = ROOT / "samples"
RESULTS = ROOT / "results"


def iou_f1(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    return float(iou), float(f1)


def post(path, **kw):
    with open(path, "rb") as f:
        return requests.post(f"{URL}/predict", files={"file": (Path(path).name, f)}, **kw)


def main():
    assert requests.get(f"{URL}/health").json()["status"] == "ok", "server not running"
    ref = {r["id"]: r for r in csv.DictReader(open(RESULTS / "sample_check.csv"))}
    rows, ok = [], True

    for img in sorted(SAMPLES.glob("sample_*.npy")):
        if img.stem.endswith("_mask"):
            continue
        sid = img.stem.split("_")[1]
        gt = np.load(SAMPLES / f"sample_{sid}_mask.npy")

        r = post(img)
        assert r.status_code == 200 and r.headers["content-type"] == "image/png", r.text
        mask = (np.array(Image.open(io.BytesIO(r.content))) > 0).astype(np.uint8)
        assert mask.shape == gt.shape and set(np.unique(mask)) <= {0, 1}

        j = post(img, params={"format": "json"}).json()
        iou, f1 = iou_f1(mask, gt)
        match = abs(iou - float(ref[sid]["iou"])) < 1e-3 and abs(j["water_pct"] - float(ref[sid]["pred_water_pct"])) < 0.01
        ok = ok and match
        rows.append(dict(id=sid, true_water_pct=round(gt.mean() * 100, 2), api_water_pct=j["water_pct"],
                         iou=round(iou, 4), f1=round(f1, 4), matches_offline=match))
        print(rows[-1])

    bad = post(SAMPLES / "sample_184_mask.npy")
    print("wrong shape ->", bad.status_code, bad.json())
    assert bad.status_code == 400
    bad = post(ROOT / "weights" / "result.json")
    print("wrong type  ->", bad.status_code, bad.json())
    assert bad.status_code == 400

    with open(RESULTS / "api_test_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nall API results match offline inference:", ok)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
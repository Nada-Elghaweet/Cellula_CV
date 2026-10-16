import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import BAND_NAMES, RGB_IDX, load_model

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights" / "best.pt"
NORM = ROOT / "weights" / "norm_stats_z12.json"

N_CHANNELS = 12
THRESHOLD = 0.5
DEVICE = "cpu"

_model = None
_mean = None
_std = None


def _load_norm():
    stats = json.load(open(NORM))
    if stats["bands"] != BAND_NAMES:
        raise RuntimeError("band order in norm_stats_z12.json does not match model.py")
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(1, N_CHANNELS, 1, 1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(1, N_CHANNELS, 1, 1)
    return mean, std


def get_model():
    global _model, _mean, _std
    if _model is None:
        _model = load_model(WEIGHTS, device=DEVICE)
        _mean, _std = _load_norm()
    return _model


def validate_array(arr):
    if not isinstance(arr, np.ndarray):
        raise ValueError("input is not a numpy array")
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"expected shape (12, H, W), got {arr.shape}")
    if arr.shape[0] != N_CHANNELS:
        raise ValueError(f"expected {N_CHANNELS} channels first, got shape {arr.shape}")
    h, w = arr.shape[1:]
    if h % 32 != 0 or w % 32 != 0:
        raise ValueError(f"height and width must be divisible by 32, got {h}x{w}")
    arr = arr.astype(np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("input contains NaN or infinite values")
    return arr


def predict_probs(arr):
    model = get_model()
    arr = validate_array(arr)
    x = ((arr[None] - _mean) / _std).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(DEVICE))
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return probs


def predict_mask(arr, threshold=THRESHOLD):
    probs = predict_probs(arr)
    return (probs >= threshold).astype(np.uint8)


def rgb_preview(arr):
    arr = validate_array(arr)
    rgb = np.transpose(arr[RGB_IDX], (1, 2, 0))
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)
    return (rgb * 255).astype(np.uint8)


def mask_to_png_bytes(mask):
    img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def rgb_to_png_bytes(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def to_b64(png_bytes):
    return base64.b64encode(png_bytes).decode("ascii")


def run(arr, threshold=THRESHOLD):
    mask = predict_mask(arr, threshold)
    return {
        "mask": mask,
        "water_pct": round(float(mask.mean()) * 100, 2),
        "height": int(mask.shape[0]),
        "width": int(mask.shape[1]),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) not in (2, 3):
        print("usage: python inference.py path/to/sample.npy [output_mask.png]")
        raise SystemExit(1)
    a = np.load(sys.argv[1])
    out = run(a)
    print("mask shape:", out["mask"].shape, "| water %:", out["water_pct"])
    if len(sys.argv) == 3:
        Path(sys.argv[2]).write_bytes(mask_to_png_bytes(out["mask"]))
        print("saved", sys.argv[2])
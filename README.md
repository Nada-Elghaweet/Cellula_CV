# Water Segmentation Deployment

Flask REST API and web frontend that segment water bodies in 12-channel satellite patches, using a U-Net with a pretrained ResNet34 encoder.

## Model

- Architecture: `segmentation-models-pytorch` U-Net, ResNet34 encoder, first conv widened to 12 channels (`rgb_zero` init)
- Checkpoint: `weights/best.pt` (seed 42, weight decay 1e-3, stored with Git LFS)
- Training input: z-scored with train-only statistics, saved in `weights/norm_stats_z12.json`
- Threshold: 0.5

## Performance (Week 2 experiments)

- Validation IoU, mean over 5 seeds: 0.797 ± 0.017 (scratch U-Net: 0.748)
- Test IoU, mean over 5 seeds: 0.697 ± 0.027 (scratch U-Net: 0.611)
- The deployed checkpoint is seed 42 (val IoU 0.8118), one run from that distribution
- Weak spot: images with small water bodies (under 7% water), where the model misses water pixels


## Model selection

| Model | Val IoU (5 seeds) | Test IoU (5 seeds) |
|---|---|---|
| ResNet34 U-Net (deployed) | 0.797 | 0.697 |
| EfficientNet-b0 U-Net | 0.787 | 0.697 |
| ResNet18 U-Net | 0.782 | 0.690 |
| Week 1 engineered U-Net (scratch) | 0.757 | not scored |
| Week 1 baseline U-Net (scratch) | 0.749 | not scored |

ResNet34 was chosen for the highest validation IoU. The three pretrained encoders are within seed noise of each other, and all beat every scratch model.


## Input format

- `.npy` file, shape `(12, H, W)` or `(1, 12, H, W)`, float values in 0 to 1
- Band order: Coastal, Blue, Green, Red, NIR, SWIR1, SWIR2, QA, MeritDEM, CopDEM, ESA, WaterOcc
- H and W must be divisible by 32 (training patches are 128 x 128)
- Do not z-score the input, the API does it

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 for the web page.

## API

`POST /predict` with multipart field `file` (a `.npy` file)

- Default: returns a PNG mask (0 = land, 255 = water), with the water percentage in the `X-Water-Pct` header
- `?format=json`: returns `water_pct`, `height`, `width`, `threshold`, `mask_png_b64`, `preview_png_b64`
- `?threshold=0.3`: optional threshold between 0 and 1
- Errors return JSON `{"error": "..."}` with status 400 (wrong shape, NaN values, wrong file type, unreadable file) or 413 (file over 20 MB)

`GET /health` returns `{"status": "ok"}`

```
curl -F "file=@samples/sample_184.npy" "http://127.0.0.1:5000/predict?format=json"
curl -F "file=@samples/sample_184.npy" http://127.0.0.1:5000/predict -o mask.png
```

## Testing

With the server running: `python tests/test_api.py`

Five patches from the test split (never used for model selection) were sent through the API. Results are in `results/api_test_results.csv` and match the offline inference in `results/sample_check.csv`.

| Sample | True water % | Predicted water % | IoU | F1 |
|---|---|---|---|---|
| 188 | 0.00 | 0.00 | 1.0000 | 1.0000 |
| 285 | 3.74 | 3.20 | 0.7849 | 0.8795 |
| 26 | 12.85 | 26.46 | 0.3676 | 0.5376 |
| 184 | 35.28 | 33.91 | 0.9458 | 0.9721 |
| 15 | 58.31 | 49.10 | 0.5627 | 0.7202 |

- Sample 188 has no water, so IoU 1.0 means the model predicted none (empty vs empty), not a segmentation score
- Sample 26 over-predicts water and sample 15 under-predicts it
- Invalid inputs (a 128 x 128 mask file, a `.json` file) return a 400 error with a clear message

Screenshots of the web page are in `results/`.

## Limits

- Only `.npy` input in the Week 1 feature format, raw satellite files are not supported
- Small evaluation sets (47 validation and 46 test images), and every model drops about 0.1 IoU from validation to test
- Runs on CPU, one image per request

## Project structure

```
app.py            Flask app and /predict endpoint
inference.py      loading, normalization, prediction, PNG encoding
model.py          architecture and checkpoint loading
templates/        web frontend
weights/          checkpoint, config, normalization stats
samples/          5 test patches and their ground-truth masks
scripts/          make_samples.py
tests/            test_api.py
results/          test outputs and screenshots
```
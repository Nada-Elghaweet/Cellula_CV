# Water segmentation on multispectral satellite patches

This project marks water pixels in 128x128 satellite patches (30 m resolution, harmonized Sentinel-2 and Landsat). The model is a U-Net written from scratch in PyTorch, with no pretrained weights. We trained it on two different inputs: the plain 12 channels as a baseline, and a smaller set of engineered features.

## Data

There are 306 image and mask pairs. Another 150 labels have no matching image and are ignored.

- Image: (128, 128, 12), int16, channels last
- Mask: (128, 128), 1 = water, 0 = background
- About 25.98% of all pixels are water

The 12 channels, in order: Coastal, Blue, Green, Red, NIR, SWIR1, SWIR2, QA (a bit field), Merit DEM, Copernicus DEM, ESA WorldCover (class codes), Water occurrence.

Some images are near duplicates of each other. 82 images fall into 36 duplicate groups, so there are about 260 truly different scenes. This matters for the split below. The data itself is not stored in this repo.

## What we did

**Split.** We split by duplicate group, not by image, so copies of the same scene never end up in different sets. 70/15/15 for train, validation and test, stratified by how much water a group contains, seed 42. That gives 213, 47 and 46 images.

**Normalization.** Every channel is scaled with statistics from the training images only, computed over the whole training set and kept separately for each channel. Nothing is scaled per image. The numbers are saved in `norm_stats.json`.

- Baseline: the nine real-valued bands are clipped to the 0.1 and 99.9 percentiles and then min-max scaled. QA, ESA and Water occurrence get plain min-max scaling. Missing Merit DEM values (-9999) are filled from the Copernicus DEM.
- Engineered: Green, NIR and SWIR1 are clipped to 0 and the 99.9 percentile, then min-max scaled.

**Engineered input (11 channels).** Green, NIR, SWIR1, two water indices, two QA bits and the ESA class as one-hot flags.

- NDWI = (Green - NIR) / (Green + NIR)
- MNDWI = (Green - SWIR1) / (Green + SWIR1)
- QA bit 4 and bit 5
- ESA one-hot: class 80, 40, 10, and everything else

Coastal, Blue, Red, SWIR2, both DEMs and Water occurrence were dropped because they were redundant, weak, or had missing values. The reasoning is in the exploration notebook.

**Model.** A standard U-Net: four down steps, a bottleneck, four up steps with transposed convolutions, and skip connections. Each step is two 3x3 convolutions with BatchNorm and ReLU. The first level has 32 channels (7.77M parameters) and the output is one logit per pixel. The same code runs with 11 or 12 input channels.

**Training.** AdamW with learning rate 1e-3 and weight decay 1e-4, batch size 16, plain BCE loss. The learning rate halves when validation IoU stalls for 5 epochs, and training stops after 15 epochs without improvement (100 at most). Training images are randomly flipped and rotated by multiples of 90 degrees. Validation and test are left untouched. A run takes one to two minutes on a Colab T4.

**Metrics.** All metrics are for the water class, counted over all pixels of the set (true and false positives and false negatives are summed over images), at threshold 0.5: IoU, precision, recall and F1. We also report IoU without the images that are more than 95% water, since those are easy.

## Results

Validation set, 5 random seeds per input, mean (standard deviation):

| Input | IoU | Precision | Recall | F1 | IoU without full-water images |
|---|---|---|---|---|---|
| Baseline, 12 channels | 0.7485 (0.0137) | 0.895 | 0.822 | 0.8561 (0.0090) | 0.7051 |
| Engineered, 11 channels | 0.7573 (0.0017) | 0.922 | 0.810 | 0.8619 (0.0011) | 0.7141 |

Test results: to be added after the final run. The test set is used once, at the end.

A few things worth knowing when reading these numbers:

- The two inputs end up at about the same level. The mean gap in IoU (0.009) is smaller than its own spread across seeds, so we do not claim the engineered input is more accurate.
- What clearly differs is stability. The engineered input gives almost the same score on every seed. One baseline seed trained badly (IoU 0.724), and without it the two inputs are within 0.003.
- The validation score is taken at the best epoch, and that epoch is chosen on the same validation set, so it is a little optimistic.
- Precision is higher than recall for both inputs, so the models miss more water than they invent.
- We think label noise limits the score, for example temporary or flooded water that looks like land. This is a guess and we have not tested it.

## Still to do

- Ablations: with and without QA and ESA, NDWI, QA bit 4, RGB instead of Green, Copernicus DEM
- Hyperparameter tuning, with the same budget for both inputs
- Tune the decision threshold on validation, then evaluate on test once
- Look at the images where the model fails

## How to run

The notebooks are written for Google Colab with the data on Google Drive.

1. Put the images (`.tif`) and masks (`.png`) in `MyDrive/data/images` and `MyDrive/data/labels`.
2. Run the notebook cells from top to bottom. Everything the pipeline creates (split, statistics, feature arrays, checkpoints, results) goes to `MyDrive/water_seg`.
3. After a runtime restart, run the cell that reloads data from Drive first, then the model, loss and metrics, checkpoint and trainer cells.

Training runs save a checkpoint every epoch, so a run that is cut off resumes where it stopped.

## References

- McFeeters, 1996, NDWI
- Xu, 2006, MNDWI

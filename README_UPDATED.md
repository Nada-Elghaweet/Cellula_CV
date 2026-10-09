# Water segmentation on multispectral satellite patches

This project marks water pixels in 128x128 satellite patches (30 m resolution, harmonized Sentinel-2 and Landsat). We first trained a U-Net from scratch in PyTorch on two inputs: the plain 12 channels as a baseline, and a smaller set of engineered features. In week 2 we replaced the encoder with ImageNet-pretrained ones (ResNet18, ResNet34, EfficientNet-b0) and compared them with the scratch models.

## Data

There are 306 image and mask pairs. Another 150 labels have no matching image and are ignored.

- Image: (128, 128, 12), int16, channels last
- Mask: (128, 128), 1 = water, 0 = background
- About 25.98% of all pixels are water

The 12 channels, in order: Coastal, Blue, Green, Red, NIR, SWIR1, SWIR2, QA (a bit field), Merit DEM, Copernicus DEM, ESA WorldCover (class codes), Water occurrence.

Some images are near duplicates of each other. 82 images fall into 36 duplicate groups, so there are about 260 truly different scenes. This matters for the split below. The data itself is not stored in this repo.

## What we did

**Split.** We split by duplicate group, not by image, so copies of the same scene never end up in different sets. 70/15/15 for train, validation and test, stratified by how much water a group contains, seed 42. That gives 213, 47 and 46 images. The same split is used in both weeks.

**Normalization (week 1).** Every channel is scaled with statistics from the training images only, computed over the whole training set and kept separately for each channel. Nothing is scaled per image. The numbers are saved in `norm_stats.json`.

- Baseline: the nine real-valued bands are clipped to the 0.1 and 99.9 percentiles and then min-max scaled. QA, ESA and Water occurrence get plain min-max scaling. Missing Merit DEM values (-9999) are filled from the Copernicus DEM.
- Engineered: Green, NIR and SWIR1 are clipped to 0 and the 99.9 percentile, then min-max scaled.

**Engineered input (11 channels).** Green, NIR, SWIR1, two water indices, two QA bits and the ESA class as one-hot flags.

- NDWI = (Green - NIR) / (Green + NIR)
- MNDWI = (Green - SWIR1) / (Green + SWIR1)
- QA bit 4 and bit 5
- ESA one-hot: class 80, 40, 10, and everything else

Coastal, Blue, Red, SWIR2, both DEMs and Water occurrence were dropped because they were redundant, weak, or had missing values. The reasoning is in the exploration notebook.

**Scratch model (week 1).** A standard U-Net: four down steps, a bottleneck, four up steps with transposed convolutions, and skip connections. Each step is two 3x3 convolutions with BatchNorm and ReLU. The first level has 32 channels (7.77M parameters) and the output is one logit per pixel. The same code runs with 11 or 12 input channels.

**Scratch training.** AdamW with learning rate 1e-3 and weight decay 1e-4, batch size 16, plain BCE loss. The learning rate halves when validation IoU stalls for 5 epochs, and training stops after 15 epochs without improvement (100 at most). Training images are randomly flipped and rotated by multiples of 90 degrees. Validation and test are left untouched. A run takes one to two minutes on a Colab T4.

**Metrics.** All metrics are for the water class, counted over all pixels of the set (true and false positives and false negatives are summed over images), at threshold 0.5: IoU, precision, recall and F1. We also report IoU without the images that are more than 95% water, since those are easy. The test set has no such images, so there the two IoU values are equal.

## Week 2: pretrained encoders

**Models.** A U-Net from `segmentation-models-pytorch` with an ImageNet-pretrained encoder: ResNet18 (14.4M parameters), ResNet34 (24.5M) and EfficientNet-b0 (6.3M). The input stays at 12 channels.

**Input.** The 12 week-1 baseline channels, z-scored per channel with training-set statistics only (ImageNet filters expect roughly zero mean and unit variance).

**First layer.** The pretrained first convolution has 3 input channels, so we replace it with a 12-channel one. We tried two ways to initialize it:

- avg: the mean of the RGB filters, repeated for all 12 channels and scaled by 3/12
- rgb_zero: the RGB filters are copied to the Red, Green and Blue channels of our array (indices 3, 2, 1), the other 9 channels start at zero

Both use all 12 channels. In rgb_zero the zeros are only the starting values, and those weights are trained like the rest. We checked that at initialization the adapted model gives the same output as the 3-channel pretrained model on the RGB bands (max difference below 1e-4).

**Training.** AdamW, encoder learning rate 1e-4 and decoder learning rate 1e-3, weight decay 1e-3 (no decay on BatchNorm and bias), batch 16, plain BCE, and the same augmentation, scheduler and early stopping as week 1. We also trained the scratch U-Net on the same z-scored input as a control, with weight decay 1e-4 and 1e-3. Every model was trained with 5 seeds (42 to 46).

**Choices.**

- First-layer strategy: avg and rgb_zero tied on ResNet34 (0.7868 val IoU for both, 3 seeds), so we kept rgb_zero.
- Weight decay: tried 1e-4, 1e-3, 1e-2 and 1e-1 on ResNet34 with seed 42 only, and picked 1e-3. The differences between values (up to 0.017) are about the size of the seed noise, so this is a reasonable setting and not a finding.
- Longer training: raising the epoch limit from 100 to 250 did not help (mean val IoU 0.7924 vs 0.7968). No run reached the new limit, so 100 epochs was enough.

## Results

Validation set, 5 random seeds per model, mean (standard deviation):

| Model | Params | IoU | Precision | Recall | F1 | IoU without full-water images |
|---|---|---|---|---|---|---|
| ResNet34, pretrained | 24.5M | 0.7968 (0.0174) | 0.906 | 0.869 | 0.8868 | 0.7615 |
| EfficientNet-b0, pretrained | 6.3M | 0.7870 (0.0133) | 0.909 | 0.855 | 0.8808 | 0.7497 |
| ResNet18, pretrained | 14.4M | 0.7820 (0.0160) | 0.905 | 0.852 | 0.8776 | 0.7440 |
| Engineered, 11 channels (week 1) | 7.8M | 0.7573 (0.0017) | 0.922 | 0.810 | 0.8619 | 0.7141 |
| Scratch, z-scored input, wd 1e-3 | 7.8M | 0.7497 (0.0026) | 0.901 | 0.817 | 0.8569 | 0.7061 |
| Baseline, 12 channels (week 1) | 7.8M | 0.7485 (0.0137) | 0.895 | 0.822 | 0.8561 | 0.7051 |
| Scratch, z-scored input, wd 1e-4 | 7.8M | 0.7484 (0.0042) | 0.906 | 0.813 | 0.8561 | 0.7044 |

Test set, scored once at the end, 5 seeds, mean (standard deviation):

| Model | IoU | Precision | Recall | F1 |
|---|---|---|---|---|
| EfficientNet-b0, pretrained | 0.6972 (0.0318) | 0.894 | 0.761 | 0.8212 |
| ResNet34, pretrained | 0.6966 (0.0265) | 0.889 | 0.764 | 0.8209 |
| ResNet18, pretrained | 0.6896 (0.0271) | 0.888 | 0.757 | 0.8161 |
| Scratch, z-scored input, wd 1e-3 | 0.6208 (0.0177) | 0.877 | 0.681 | 0.7659 |
| Scratch, z-scored input, wd 1e-4 | 0.6113 (0.0098) | 0.881 | 0.667 | 0.7587 |

The week 1 models were never scored on the test set, so the scratch controls are the only test reference.

A few things worth knowing when reading these numbers:

- All three pretrained encoders beat every scratch model, on validation and on test. The smallest gain is ResNet18 over the week 1 engineered input (+0.025 val IoU), which is larger than that input's own seed spread. ResNet34 gains about 0.04 on validation and 0.085 over the scratch control on test.
- The three encoders are close to each other. Their gaps (up to 0.015) are smaller than the spread across seeds, so we do not claim one is better. EfficientNet-b0 matches ResNet34 on test with a quarter of the parameters.
- The scratch controls on z-scored input score the same as the week 1 baseline (0.748 vs 0.749), so the new normalization and the higher weight decay did not help scratch training. The gain comes from the pretrained encoder.
- The gain comes mostly from recall (about 0.86 vs 0.81). Precision stays about the same.
- Pretrained models are less stable between seeds (std about 0.015 vs 0.003 for the scratch controls) and train longer (47 to 77 epochs on average vs 11 to 19).
- Every model drops by about 0.1 IoU from validation to test. We think the test set is harder (it has no full-water images, validation has 2), but we have not tested this.
- Weight decay and the first-layer strategy were chosen on the validation set, and the best epoch is chosen there too, so the validation scores are a little optimistic.
- The 0.8 validation IoU was reached by single seeds (best 0.8118) but not by the mean (0.7968). Running the same seed again can move the score by about 0.01 on GPU.
- Precision is higher than recall for all models, so they miss more water than they invent. The 10 worst validation images (ResNet34, seed 42) all have under 7% water and are mostly missed pixels. We think small water bodies and label noise are the main problem, but this is a guess and we have not tested it.

## Still to do

- Test-time augmentation and averaging the 5 seeds (no retraining needed)
- Tune the decision threshold on validation, then evaluate on test once
- Dice term in the loss, to see if it helps small water bodies
- Ablations: with and without QA and ESA, NDWI, QA bit 4, RGB instead of Green, Copernicus DEM
- Satellite-pretrained encoders (SSL4EO) and a larger EfficientNet

## How to run

The notebooks are written for Google Colab with the data on Google Drive.

1. Put the images (`.tif`) and masks (`.png`) in `MyDrive/data/images` and `MyDrive/data/labels`.
2. Run the week 1 notebook from top to bottom. Everything the pipeline creates (split, statistics, feature arrays, checkpoints, results) goes to `MyDrive/water_seg`.
3. Run `Water_Seg_Week2_Pretrained.ipynb` from top to bottom. It needs a GPU runtime and installs `segmentation-models-pytorch` itself. Runs, checkpoints and results go to `MyDrive/water_seg/runs_w2`.
4. Finished runs are skipped when you run the week 2 notebook again, so a cut-off session only loses the run that was in progress. The week 1 trainer saves a checkpoint every epoch and resumes where it stopped.

The test set is scored in section 15 of the week 2 notebook. The result is saved to `test_results.csv` and later runs only reload it.

## References

- McFeeters, 1996, NDWI
- Xu, 2006, MNDWI

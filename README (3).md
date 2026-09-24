# Eye Disease Classification from Fundus Images

This project builds and compares two deep learning models for classifying retinal fundus images into 8 categories: Normal, Diabetic Retinopathy, Glaucoma, Cataract, AMD, Hypertension, Myopia, and Others. It's the first week of a larger effort, so the focus here is getting a solid baseline pipeline working end to end rather than squeezing out state of the art numbers.

## Dataset

The dataset is a unified collection of 11,839 fundus images pulled together from 4 public sources:

- ODIR5K (7000 images)
- APTOS2019 (3484 images)
- ACRIMA (705 images)
- ORIGA (650 images)

| Class | Total | Train | Val | Test |
|---|---|---|---|---|
| Normal | 4698 | 3808 | 485 | 423 |
| Diabetic Retinopathy | 4113 | 3318 | 408 | 369 |
| Others | 1102 | 893 | 110 | 99 |
| Glaucoma | 930 | 753 | 93 | 84 |
| Cataract | 340 | 275 | 34 | 31 |
| Myopia | 294 | 238 | 30 | 26 |
| AMD | 274 | 221 | 28 | 25 |
| Hypertension | 88 | 71 | 9 | 8 |

The validation split (`val.csv`) came bundled with the original dataset, so it was left untouched. Train and test were carved out ourselves from the remaining images using a stratified 90/10 split, since only val was pre-defined.

Before any of that, the raw data was checked for corrupt files, duplicate images, and inconsistent dimensions:

- 0 corrupt images
- 465 unique image dimensions (ranging from 178px to 5184px, expected given 4 different sources)
- 126 exact duplicate image pairs, 39 of which crossed between splits (e.g. the same image showing up in both train and test)

That cross-split duplication is a real data leakage risk, so those 39 files were removed from val/test and kept only in train before any model touched the data.

## The imbalance problem

The dataset is heavily skewed. Normal and Diabetic Retinopathy alone make up about 74% of all images, while Hypertension has only 88 total. That's a 53:1 ratio between the largest and smallest class, which is a real problem for training since a model can get away with mostly ignoring the small classes and still post a decent accuracy number.

To deal with this, three things were combined:

1. **Augmentation** on every image (flip, rotation, brightness/contrast jitter), with a heavier version applied specifically to the four smallest classes (Cataract, Myopia, AMD, Hypertension) for extra variety on top of the stronger oversampling they get.
2. **Balanced batch sampling** via `WeightedRandomSampler`, so the model sees minority classes far more often per epoch than their raw frequency would suggest.
3. **Focal loss**, used instead of plain cross-entropy, so the model focuses more on harder/misclassified examples during training.

One thing worth flagging: early on, focal loss was also given per-class alpha weights on top of the balanced sampler, and stacking both corrections at once badly overcorrected the model, to the point where it stopped predicting Normal and Diabetic Retinopathy entirely. Dropping the alpha weighting and letting the sampler do the balancing on its own fixed this. That's a useful lesson on its own: imbalance techniques don't just add up cleanly, and combining too many at full strength can break things in ways that are hard to spot from the loss curve alone.

Given the severity of the imbalance, accuracy alone isn't a meaningful metric here, so per-class precision, recall, and F1 (plus macro-F1 as the main model-selection metric) are used throughout instead.

## Models

### Scratch CNN

A custom convolutional network built from scratch, with light residual connections added between conv blocks to help gradient flow. Kept relatively small on purpose (~1.26M parameters) since the training set is under 10,000 images and a deeper network would likely just overfit.

Architecture: a stem conv layer, followed by 4 residual blocks (32 → 64 → 128 → 256 channels), global average pooling, then a small dropout-regularized classifier head.

Trained with focal loss (no per-class alpha, balanced sampler handling exposure), Adam optimizer, a learning rate scheduler on plateau, and early stopping based on validation macro-F1.

**Test set results:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| AMD | 0.000 | 0.000 | 0.000 |
| Cataract | 0.667 | 0.194 | 0.300 |
| Diabetic Retinopathy | 0.918 | 0.381 | 0.538 |
| Glaucoma | 0.312 | 0.571 | 0.403 |
| Hypertension | 0.000 | 0.000 | 0.000 |
| Myopia | 0.800 | 0.615 | 0.696 |
| Normal | 0.701 | 0.583 | 0.636 |
| Others | 0.182 | 0.657 | 0.284 |

Overall accuracy: **49.2%**, macro-F1: **0.357**

AMD and Hypertension both collapsed to zero here. Hypertension only has 8 test images and 71 training images to begin with, so this isn't hugely surprising, but AMD has more data (221 training images) and still failed completely, which points to it being visually harder to separate from other classes rather than a pure data scarcity issue.

### Fine-tuned ResNet50

Started from ImageNet-pretrained weights, replaced the final layer with a small dropout + linear head for 8 classes, and fine-tuned with `layer4` and the new head unfrozen (`layer3` and earlier stayed frozen). Differential learning rates were used, a much smaller one for the pretrained backbone layer and a larger one for the newly initialized head, since the backbone already has useful features and shouldn't be pushed around too aggressively.

An earlier attempt that unfroze more of the backbone (`layer3` and `layer4` both, with a higher learning rate) overfit almost immediately, peaking in the first epoch and then trending downward. Freezing more of the network and lowering the learning rates fixed that and gave a much healthier, steadily improving training curve.

**Test set results:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| AMD | 0.200 | 0.040 | 0.067 |
| Cataract | 0.808 | 0.677 | 0.737 |
| Diabetic Retinopathy | 0.887 | 0.469 | 0.613 |
| Glaucoma | 0.368 | 0.762 | 0.496 |
| Hypertension | 0.250 | 0.125 | 0.167 |
| Myopia | 0.917 | 0.846 | 0.880 |
| Normal | 0.836 | 0.483 | 0.613 |
| Others | 0.203 | 0.788 | 0.323 |

Overall accuracy: **53.1%**, macro-F1: **0.487**

## Comparing the two

ResNet50 comes out ahead on essentially every metric that matters: higher overall accuracy (53.1% vs 49.2%), and a noticeably higher macro-F1 (0.487 vs 0.357). The gap is even more obvious at the per-class level. Where the scratch CNN completely failed on AMD and Hypertension, ResNet at least manages some signal on both, and it's clearly stronger on Cataract, Myopia, and Glaucoma too.

That said, neither model handles the smallest classes well. Hypertension in particular stays weak across both, which lines up with what you'd expect given it only has 71 training images to work with even after oversampling. This is really the core finding of the imbalance analysis: augmentation, balanced sampling, and focal loss all help, but none of them can fully substitute for actual data when a class this small is involved. More data for Hypertension and AMD specifically would probably do more for these numbers than further architecture or loss tuning.

### Real-time deployment consideration

For a real-time or clinical deployment setting, ResNet50 is the better choice here, and not just because of the accuracy gap. It benefits from ImageNet pretraining, which means it converges faster and more reliably than a scratch model, and despite being a much bigger architecture on paper, its inference cost per image is still reasonable and well within real-time range on a GPU. The scratch CNN is smaller and would run faster on very limited hardware, but the accuracy tradeoff is too large to justify it given how poorly it performs on several disease classes, which matters a lot more in a medical context than raw speed does. If deployment ever needs to move to constrained edge hardware, a smaller pretrained backbone like MobileNet or EfficientNet-Lite would be a better middle ground to explore than sticking with the from-scratch model.

## Repo structure

```
├── Retinal_CNN.ipynb        # full notebook: EDA, preprocessing, both models, evaluation
├── README.md
```

The notebook is meant to be run top to bottom in Google Colab with a GPU runtime. It downloads the dataset directly from a shared Google Drive link at the start.

## What's next

This is a first-week baseline, so there's plenty of room to improve on:

- More data or targeted collection for Hypertension and AMD specifically, since both are underperforming across the board regardless of technique
- Trying a second pretrained backbone (VGG16 was part of the original brief but wasn't trained end-to-end here, ResNet was prioritized for its better fine-tuning stability and inference speed)
- Testing whether a milder combination of imbalance techniques (e.g. sampling only, no focal loss, or vice versa) does better on the weakest classes
- A proper inference speed benchmark (ms per image) to put real numbers behind the deployment discussion instead of general architecture reasoning

## References

- Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition" (VGG) — https://arxiv.org/abs/1409.1556
- He et al., "Deep Residual Learning for Image Recognition" (ResNet) — https://arxiv.org/abs/1512.03385

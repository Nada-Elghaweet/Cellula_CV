# Shoplifting Detection from Video

Binary video classification (normal shopping vs shoplifting) with a temporal deep learning model trained from scratch, with no pretrained weights.

## 1. Overview

- Task: classify a video clip as `normal` or `shoplifting`
- Model: residual CNN frame encoder + bidirectional LSTM
- Input: 16 frames per clip, tensor shape `(T, C, H, W) = (16, 3, 144, 176)`
- Framework: PyTorch for training, ONNX for deployment
- Everything is implemented in the notebook `shoplifting_detection.ipynb`

## 2. Dataset

- 855 clips in total
  - 531 normal (`non shop lifters`)
  - 324 shoplifting (`shop lifters`)
- Fixed overhead camera, all clips are 704x576 at 25 fps
- Clip length: 75 to 1850 frames, median 348 frames (about 14 seconds)
- No unreadable clips and no zero-frame clips
- Labels come from the folder names

## 3. Pipeline

### 3.1 Frame sampling

- Skip interval is `total_frames / T`, so the samples span the full video
- The sampler takes the midpoint of each of the T equal segments
- Works for any clip length, including clips shorter than T (indices repeat instead of failing)
- Verified on clips of 5, 16, 75, 348, 570 and 1850 frames: the gap between sampled indices differs by at most 1 frame

### 3.2 Consistent augmentation

- All random parameters are sampled once per clip and shared by every frame
  - Random crop offset (160x196 resized frame, 144x176 crop)
  - Horizontal flip with probability 0.5
  - Brightness, contrast and saturation factors in 0.7 to 1.3
- Contrast uses one mean over the whole clip, so the transform is identical for all frames
- Verified with a code check: 50 random augmentations of a clip made of identical frames always give identical output frames
- A negative control (independent per-frame augmentation) fails the same check, which shows the test is meaningful
- Augmentation is applied to the train split only
- Validation and test use resize and center crop only

### 3.3 Dataset class

- `ShopliftingDataset` returns a float tensor `(16, 3, 144, 176)` in `[0, 1]` and an integer label
- Shapes were checked on every clip in the train, validation and test splits
- Sampled frames are cached once as `.npy` files for fast training, with a fallback to reading the raw mp4

### 3.4 Split

- Stratified at the video level, seed 42

| Split | Normal | Shoplifting | Total |
|-------|--------|-------------|-------|
| Train | 372    | 226         | 598   |
| Val   | 80     | 48          | 128   |
| Test  | 80     | 49          | 129   |

## 4. Model

- Architecture: `ResCNNLSTM`, 1,622,690 parameters, randomly initialized (Kaiming init for convolutions)
- Frame encoder
  - Stem: Conv, BatchNorm, ReLU with stride 2
  - Four residual stages with 32, 64, 128 and 256 channels
  - Global average pooling to a 256-d feature per frame
- Temporal model
  - Bidirectional LSTM with 128 hidden units over the 16 frame features
  - Mean and max pooling over all time steps
  - Dropout and a linear layer with 2 outputs
- The encoder is shared across frames by folding the time axis into the batch axis

## 5. Training

- Loss: cross-entropy with class weights from the train split (0.806 normal, 1.317 shoplifting) and label smoothing 0.1
- Optimizer: AdamW, learning rate 1e-3, weight decay 5e-2 (not applied to biases and BatchNorm)
- Cosine learning rate schedule, gradient clipping at 1.0, mixed precision
- Batch size 8, up to 70 epochs, early stopping patience 20
- The checkpoint with the best validation F1 is kept (epoch 55, validation F1 0.9362)
- The test set is used once, after training

## 6. Results

Test set, 129 clips:

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.9457 |
| Precision | 0.9773 |
| Recall    | 0.8776 |
| F1-score  | 0.9247 |

Per class:

| Class       | Precision | Recall | F1     | Support |
|-------------|-----------|--------|--------|---------|
| normal      | 0.9294    | 0.9875 | 0.9576 | 80      |
| shoplifting | 0.9773    | 0.8776 | 0.9247 | 49      |

Confusion matrix (rows are true labels):

|                     | Pred normal | Pred shoplifting |
|---------------------|-------------|------------------|
| True normal         | 79          | 1                |
| True shoplifting    | 6           | 43               |

Notes:

- The model is conservative: it raises very few false alarms (1) but misses some thefts (6)
- Validation curves are noisy during the first 25 epochs, while the model is still at chance level, and stabilize after about epoch 40

## 7. Deployment

The trained model is exported for inference without PyTorch.

- `shop_scratch/shop_scratch.onnx`: ONNX model with a dynamic batch size
- `shop_scratch/shop_scratch.pt`: PyTorch weights
- `shop_scratch/shop_scratch_meta.json`: input shape and preprocessing details

Inference preprocessing:

- Sample 16 frames at the midpoints of 16 equal segments of the video
- Convert BGR to RGB and resize each frame to 160x196
- Center crop to 144x176
- Scale to `[0, 1]` as float32, with no mean or std normalization
- Input name `clip`, shape `(batch, 16, 3, 144, 176)`, output name `logits` with classes `0 = normal`, `1 = shoplifting`

Example:

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("shop_scratch/shop_scratch.onnx", providers=["CPUExecutionProvider"])
clip = np.random.rand(1, 16, 3, 144, 176).astype(np.float32)
logits = sess.run(None, {"clip": clip})[0]
print("shoplifting" if logits.argmax(1)[0] == 1 else "normal")
```

## 8. How to run

- Install dependencies: `pip install -r requirements.txt`
- Place the dataset so that it has the layout `Shop DataSet/non shop lifters/*.mp4` and `Shop DataSet/shop lifters/*.mp4`
- Update the data paths in the first notebook cell
- Run the notebook from top to bottom (a GPU is recommended, it was trained on a Colab T4)

## 9. Repository structure

```
.
├── README.md
├── requirements.txt
├── shoplifting_detection.ipynb
└── shop_scratch/
    ├── shop_scratch.onnx
    ├── shop_scratch.pt
    └── shop_scratch_meta.json
```

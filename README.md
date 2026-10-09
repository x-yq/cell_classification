# Semi-Supervised Bone Marrow Cell Classification

A PyTorch project for classifying segmented bone marrow cell images with limited labeled data. It implements a **Noisy Student** semi-supervised learning pipeline using a pretrained EfficientNet-B0 model.

## 1. Background

Bone marrow cell classification can support cytology and blood-disorder research, but annotated cell images are expensive to obtain. This project explores how unlabeled cell images can improve classification by generating pseudo-labels and using them to train student models.

## 2. Dataset

The thesis experiments used two datasets:

- **Internal dataset:** 11,417 labeled and 238,392 unlabeled segmented bone marrow cell images.
- **External dataset:** 171,374 labeled cell images across 21 classes, reorganized into labeled and unlabeled subsets for evaluation.

The code expects individual segmented cell images, not whole-slide images. Images are resized to **224 × 224 RGB** during preprocessing. The loader supports `.jpg` and `.png`.

### CSV annotation format

The labeled CSV has a header and one row per image:

```csv
id,parent_folder,Class_A,Class_B,Class_C
cell_001,folder_1,1,0,0
cell_002,folder_1,0,1,0
```

- `id`: image ID; the image is expected at `<image_folder>/<parent_folder>/<id>.jpg` or `.png`.
- `parent_folder`: image's parent directory.
- Remaining columns: one-hot encoded class labels.
- The first CSV header cell must be `id`. Class names are read from the labeled CSV header.

For unlabeled data, use the annotation format expected by the supplied CSV files and pipeline.

## 3. Environment Setup

Core dependencies:

- Python 3
- PyTorch and Torchvision
- NumPy
- Pillow
- scikit-learn

Example environment setup:

```bash
conda create -n cell-classification python=3.10 -y
conda activate cell-classification
pip install torch torchvision numpy pillow scikit-learn
```

Choose a PyTorch build compatible with your OS and CUDA version if using a GPU. The repository does not include a pinned dependency file, so package versions may need adjustment.

## 4. Pipeline

```text
Labeled + Unlabeled Cell Images
              ↓
  CSV Parsing & Preprocessing
              ↓
 Pretrained EfficientNet-B0 Teacher
              ↓
     Train on Labeled Data
              ↓
 Validation & Checkpoint Selection
              ↓
   Generate Pseudo-Labels
              ↓
Student Training on Labeled + Pseudo-Labeled Data
              ↓
     Repeat Student Training
              ↓
       Test Evaluation
```

Details:
- ImageNet-pretrained EfficientNet-B0.
- Image resizing, RGB conversion, tensor conversion, and ImageNet normalization.
- Weighted random sampling to mitigate class imbalance.
- RandAugment for student training.
- Micro-F1 for model selection, plus macro-F1 and per-class classification reports.

## 5. Training

Prepare the labeled/unlabeled CSV files and image directory, then run from the repository root:

```bash
python main.py \
  --trial_name cell_experiment \
  --output_folder /path/to/output \
  --model_name efficientnet_b0 \
  --labeled_anno_file /path/to/labeled.csv \
  --unlabeled_anno_file /path/to/unlabeled.csv \
  --image_folder /path/to/cell_images \
  --threshold 250 \
  --stoch_depth_prob 0.3 \
  --drop_out_rate 0.9
```

The script runs the teacher and two student iterations. The command-line arguments are defined in `hyper.py` and consumed by `main.py`, `train.py`, and `data.py`.

### Training parameters

| Parameter | Default / example | Meaning |
|---|---|---|
| `--trial_name` | Required; `cell_experiment` | Name for this run. It is included in output/log paths so experiments can be distinguished. |
| `--model_name` | Required; `efficientnet_b0` | Model architecture name. This project uses EfficientNet-B0. |
| `--labeled_anno_file` | Required; `/path/to/labeled.csv` | Path to the labeled-data annotation CSV. Class names are read from its header. |
| `--unlabeled_anno_file` | Required; `/path/to/unlabeled.csv` | Path to the unlabeled-data annotation CSV used by the data pipeline. |
| `--image_folder` | Required; `/path/to/cell_images` | Root directory containing the segmented cell images. |
| `--output_folder` | Required; `/path/to/output` | Directory where training outputs, logs, checkpoints, and generated pseudo-label files are saved. |
| `--img_size` | `224` | Image side length used for preprocessing; images are resized to `224 × 224` by default. |
| `--epochs` | `3` | Number of epochs for each training iteration. This is the parser default; the thesis experiments may have used different settings. |
| `--batch_size` | `64` | Number of images in a training batch. Larger batches require more GPU memory. |
| `--learning_rate` | `1e-4` | Optimizer learning rate. |
| `--weight_decay` | `1e-5` | Weight-decay regularization strength to help reduce overfitting. |
| `--workers` | `4` | Number of worker processes used by the data loader. |
| `--pin_memory` | `True` | Enables pinned-memory data loading to help CPU-to-GPU transfers. The argument is declared as an integer in `hyper.py`, so check this if you need to explicitly set it from the command line. |
| `--threshold` | Parser default `200000`; example `250` | Upper limit used by the weighted sampler to cap class weights for class-imbalance handling. The example value `250` reflects an experiment setting, not the parser default. |
| `--num_pseudo_labels` | `35977` | Target number of pseudo-labeled unlabeled examples used by the teacher model in the data pipeline. `main.py` also updates this value between training stages. |
| `--stoch_depth_prob` | `0.3` | Stochastic-depth probability configured for the student model, adding regularization during student training. |
| `--drop_out_rate` | `0.9` | Dropout rate configured for the student model's classifier. |

Two additional arguments are accepted by the parser but are primarily for the data pipeline's internal state:
- `--pseudo_labels_csv`: path to a CSV containing pseudo-labels from a previous iteration. `main.py` sets this automatically between student iterations; normally you do not need to pass it for a standard run.

**Important implementation notes**
- `--threshold 250`, `--stoch_depth_prob 0.3`, and `--drop_out_rate 0.9` are example experiment settings; they differ from the parser default for `--threshold`.
- `--epochs` defaults to `3` in the code, while the thesis describes experiments with different epoch settings. Use the setting appropriate to your experiment.

Training outputs include logs, checkpoints such as `model_best_bench.pth.tar`, and generated pseudo-label CSV files.

## 6. Inference

The current code performs inference internally to generate pseudo-labels during training. It does **not** include a standalone inference command for arbitrary images.

For standalone inference, load `model_best_bench.pth.tar`, instantiate the same EfficientNet-B0 architecture with the same number and ordering of classes, and apply the same preprocessing (RGB, 224 × 224, ImageNet normalization).

```python
import torch
import torchvision
from torch import nn

checkpoint = torch.load("/path/to/model_best_bench.pth.tar", map_location="cpu")

model = torchvision.models.get_model("efficientnet_b0", weights=None)
num_classes = len(class_names)  # Match the training CSV header
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)
model.load_state_dict(checkpoint["model_weights"])
model.eval()

# input_tensor: preprocessed image tensor with shape [1, 3, 224, 224]
with torch.no_grad():
    logits = model(input_tensor)
    predicted_index = logits.argmax(dim=1).item()
    predicted_class = class_names[predicted_index]
```

## 7. Results Summary

According to the thesis experiments:

| Dataset / setup | Accuracy (micro-F1) | Macro-F1 |
|---|---:|---:|
| Internal dataset — labeled-only baseline | 0.6486 | ~0.42 |
| Internal dataset — best Noisy Student model | 0.6927 | ~0.43 |
| External dataset — after semi-supervised training | 0.8509 | ~0.62 |

On the internal dataset, semi-supervised training improved accuracy by about **4.4 percentage points** and macro-F1 by about **1 percentage point** over the labeled-only baseline. Some minority classes also improved, although rare-class performance and overfitting remained challenges.

*These results are from the accompanying thesis and may not be reproduced without the same data splits, preprocessing, and hyperparameters.*

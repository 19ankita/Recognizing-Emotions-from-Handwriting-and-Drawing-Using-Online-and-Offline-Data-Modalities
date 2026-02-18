---

# Transfer Learning Regression

This module implements multi-output regression using transfer learning for predicting DASS-21 emotional scores (Depression, Anxiety, Stress, Total) from handwriting and drawing images combined with pseudo-dynamic features.

The approach integrates:

* Image-based features extracted using a pretrained ResNet-18 backbone
* Pseudo-dynamic handwriting features derived from offline images
* Multi-output regression with bounded predictions

---

## 📌 Overview

The model predicts four continuous emotional scores:

* Depression
* Anxiety
* Stress
* Total DASS

Predictions are normalized to ([0,1]) during training and rescaled to original DASS ranges during evaluation.

The architecture combines:

* Image embeddings (ResNet-18 pretrained on ImageNet)
* A small MLP for pseudo-dynamic features
* Feature fusion via concatenation
* A linear regression head with sigmoid activation

---

## 🏗 Model Architecture

```
Image (224×224 RGB)
        ↓
ResNet-18 Backbone (pretrained)
        ↓
512-D Image Features

Pseudo-Dynamic Features (5-D)
        ↓
MLP → 32-D Embedding

Concatenation (512 + 32)
        ↓
Linear Regression Head (4 outputs)
        ↓
Sigmoid Activation
```

Backbone freezing is optional via CLI flag.

---

## 📂 Directory Structure

```
transfer_learning_regression/
│
├── src/
│   ├── model.py
│   ├── dataset.py
│   ├── utils/
│
├── outputs/
│   ├── best_model_*.pth
│   ├── training_metrics.csv
│   ├── best_epoch_summary.csv
│
├── train.py
└── README.md
```

---

## 📊 Dataset Requirements

The dataset should be structured as:

```
task_root/
│
├── CDT/
│   ├── class_folder/
│   │   ├── sample1.png
│   │   ├── sample2.png
│
├── House/
├── Pentagon/
├── Cursive_writing/
├── Words/
```

Additionally, a CSV file must contain DASS scores:

```
id,depression,anxiety,stress,total
sample1,12,8,14,34
sample2,5,3,6,14
...
```

---

## ⚙️ Installation

Create environment:

```bash
conda create -n handwriting_env python=3.10
conda activate handwriting_env
```

Install dependencies:

```bash
pip install torch torchvision albumentations scikit-learn numpy pandas matplotlib tqdm
```

---

## 🚀 Training

### Basic Run

```bash
python train.py \
    --task CDT \
    --task_dir path/to/dataset \
    --label_csv path/to/DASS_scores.csv \
    --epochs 20
```

### Train on All Tasks

```bash
python train.py \
    --task all \
    --task_dir path/to/dataset \
    --label_csv path/to/DASS_scores.csv
```

---

## 🧪 CLI Arguments

| Argument            | Description                                                            |
| ------------------- | ---------------------------------------------------------------------- |
| `--task`            | Task name (e.g., CDT, House, Pentagon, Cursive_writing, Words, or all) |
| `--task_dir`        | Root directory of dataset                                              |
| `--label_csv`       | Path to DASS label CSV                                                 |
| `--epochs`          | Number of training epochs (default: 20)                                |
| `--lr`              | Learning rate (default: 1e-3)                                          |
| `--freeze_backbone` | Freeze ResNet backbone                                                 |
| `--batch_size`      | Batch size (default: 32)                                               |
| `--img_size`        | Input image size (default: 224)                                        |
| `--val_ratio`       | Validation split ratio (default: 0.2)                                  |
| `--num_workers`     | DataLoader workers (default: 2)                                        |

---

## 📈 Training Strategy

* Loss: Mean Squared Error (MSE)
* Optimizer: AdamW (weight decay = 1e-4)
* Learning Rate Schedule:

  * 2-epoch linear warmup
  * Cosine decay
* Mixed Precision (AMP) enabled on GPU
* Best model selected based on validation (R^2)

---

## 📊 Evaluation Metrics

For each epoch:

* Train MSE
* Validation MSE
* Validation RMSE
* Validation (R^2)
* Per-dimension RMSE & (R^2)

Best epoch summary is saved to:

```
outputs/best_epoch_summary.csv
```

---

## 💾 Outputs

* `best_model_<task>_regression.pth`
* `training_metrics.csv`
* Training curves (RMSE and R² plots)
* Best epoch summary

---

## 🧠 Key Design Decisions

* Sigmoid output to enforce bounded predictions
* Label normalization for stable training
* Feature fusion to combine spatial and pseudo-temporal cues
* Optional backbone freezing to evaluate transfer learning strategies

---

## 🔬 Research Purpose

This module is part of a broader study investigating:

* Emotional state recognition from handwriting and drawing
* Comparison between structured drawing and expressive writing tasks
* Online vs offline feature integration
* Transfer learning effectiveness in behavioral modeling

---


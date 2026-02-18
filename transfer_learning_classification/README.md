---

# Transfer Learning Classification

This module implements multi-class severity classification using transfer learning to predict DASS-21 emotional categories (Normal, Mild, Moderate, Severe, Extremely Severe) from handwriting and drawing images combined with pseudo-dynamic features.

Unlike the regression module, this setup performs **single-state classification**, meaning a separate classifier is trained for:

* Depression
* Anxiety
* Stress

---

## 📌 Overview

The model predicts **five severity levels** for a selected emotional state:

1. Normal
2. Mild
3. Moderate
4. Severe
5. Extremely Severe

Each emotional dimension is trained independently using the `--state` argument.

The architecture combines:

* Image embeddings from a pretrained ResNet-18 backbone
* Pseudo-dynamic handwriting features
* Feature fusion via concatenation
* A linear classification head (5 logits)

---

## 🏗 Model Architecture

```
Image (224×224 RGB)
        ↓
ResNet-18 Backbone (ImageNet pretrained)
        ↓
512-D Image Features

Pseudo-Dynamic Features (5-D)
        ↓
MLP → 32-D Embedding

Concatenation (512 + 32)
        ↓
Linear Classifier (5 logits)
```

* No final activation layer in the model
* Softmax is applied implicitly via CrossEntropyLoss

Backbone freezing is optional.

---

## 📂 Directory Structure

```
transfer_learning_classification/
│
├── src/
│   ├── model.py
│   ├── dataset.py
│   ├── utils/
│
├── outputs/
│   ├── best_model_<task>_<state>_cls.pth
│   ├── training_metrics_<task>_<state>.csv
│   ├── confusion_matrix_*.png
│   ├── classification_report_*.txt
│
├── train.py
└── README.md
```

---

## 📊 Dataset Requirements

Dataset structure:

```
task_root/
│
├── CDT/
├── House/
├── Pentagon/
├── Cursive_writing/
├── Words/
```

For single-task runs, a CSV file with DASS scores is required:

```
id,depression,anxiety,stress,total
sample1,12,8,14,34
sample2,5,3,6,14
...
```

Severity classes are derived from DASS thresholds inside the dataset pipeline.

---

## ⚙️ Installation

Create environment:

```bash
conda create -n handwriting_cls python=3.10
conda activate handwriting_cls
```

Install dependencies:

```bash
pip install torch torchvision albumentations scikit-learn numpy pandas matplotlib tqdm seaborn
```

---

## 🚀 Training

### Train on a Single Task

```bash
python train.py \
    --task CDT \
    --task_dir path/to/dataset \
    --label_csv path/to/DASS_scores.csv \
    --state anxiety
```

### Train on All Tasks Combined

```bash
python train.py \
    --task all \
    --task_dir path/to/dataset \
    --state stress
```

---

## 🧪 CLI Arguments

| Argument            | Description                                                      |
| ------------------- | ---------------------------------------------------------------- |
| `--task`            | Task name (CDT, House, Pentagon, Cursive_writing, Words, or all) |
| `--task_dir`        | Root directory of dataset                                        |
| `--label_csv`       | Path to DASS CSV (required for single-task runs)                 |
| `--state`           | Emotional state: depression, anxiety, or stress                  |
| `--epochs`          | Training epochs (default: 30)                                    |
| `--lr`              | Learning rate (default: 1e-3)                                    |
| `--freeze_backbone` | Freeze ResNet backbone                                           |
| `--batch_size`      | Batch size (default: 32)                                         |
| `--img_size`        | Input image size (default: 224)                                  |
| `--val_ratio`       | Validation split ratio (default: 0.2)                            |
| `--num_workers`     | DataLoader workers (default: 2)                                  |

---

## 📈 Training Strategy

* Loss: CrossEntropyLoss
* Optimizer: AdamW (weight decay = 1e-4)
* Learning rate:

  * 2-epoch linear warmup
  * Cosine decay
* Mixed Precision (AMP) enabled on GPU
* Best model selected based on validation accuracy

---

## 📊 Evaluation Metrics

For each epoch:

* Training loss
* Validation loss
* Training accuracy
* Validation accuracy

After training:

* Confusion matrix (per state)
* Precision, recall, F1-score per class
* Macro and weighted averages

Saved outputs:

```
outputs/
├── confusion_matrix_<task>_<state>.png
├── classification_report_<task>_<state>.txt
├── training_curves.png
```

---

## 🧠 Key Observations

* Strong class imbalance may bias predictions toward the Normal class.
* Macro-F1 is more informative than accuracy.
* Stress classification generally performs better than depression and anxiety.
* Expressive writing tasks tend to provide stronger signal than structured drawing tasks.

---

## 🔬 Research Context

This module supports the study of:

* Emotional severity classification from handwriting
* Impact of task structure (drawing vs writing)
* Transfer learning effectiveness
* Fusion of visual and pseudo-dynamic handwriting features



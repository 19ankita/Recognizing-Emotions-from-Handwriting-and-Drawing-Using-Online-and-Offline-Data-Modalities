# EMOTHAW – ResNet18 Transfer Learning with Data Augmentation

This repository contains a full PyTorch pipeline for training a ResNet-18 model on the EMOTHAW handwriting & drawing dataset using transfer learning and aggressive data augmentation techniques designed for sketch-based emotion recognition.

## Features
- ResNet-18 (ImageNet pretrained)
- Transfer learning (freeze or fine-tune backbone)
- Handwriting-specific augmentations:
  - RandomAffine, Perspective, Cutout, ColorJitter, ResizedCrop
- Config-driven training
- Modular folder structure
- Easy inference on new images

## Folder Structure

src/ # Python modules (dataset, model, train, utils, inference)
configs/ # YAML config files
experiments/ # Saved logs + checkpoints
notebooks/ # Jupyter notebooks
outputs/ # Metrics, logs, etc.

## How to Run Training
- pip install -r requirements.txt
# with config file
- python src/train.py --config configs/default.yaml 

## Using CLI
# Train ResNet18 on ALL tasks: 
  - python run_train.py --task all --model resnet18 --task_dir data/emothaw
# Train ResNet18 on task 3:
  - python run_train.py --task task1 --model resnet18 --task_dir data/emothaw


## Plotting the results
- python plot_training.py --history outputs/history.json
- python plot_all.py --history outputs/history.json --model outputs/best_model.pth
- python visualize_aug.py --config configs/default.yaml

# Checking class distributions
- python src/utils/plot_class_distribution.py data/emothaw_tasks/cdt



## Requirements
See `requirements.txt`

- pip install albumentations==1.3.0
- pip install opencv-python


---^
## ResNet-18 Transfer Learning Diagram
                             ┌───────────────────────────────────────────┐
                             │        Input Image (224 × 224 × 3)        │
                             └───────────────────────────────────────────┘
                                              │
                                              ▼
                     ┌──────────────────────────────────────────────────────────┐
                     │        Pretrained ResNet-18 (ImageNet Weights)          │
                     │   (Convolution + BN + ReLU + Residual Blocks × 4)       │
                     └──────────────────────────────────────────────────────────┘
                                              │
                    (Optional: Freeze Backbone Parameters for Transfer Learning)
                                              │
                                              ▼
         ┌──────────────────────────────────────────────────────────────────────────┐
         │                        Global Average Pooling Layer                      │
         │                       Output tensor: 512-dimensional                     │
         └──────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
             ┌────────────────────────────────────────────────────────────┐
             │      Fully Connected Classifier (replaced by you)          │
             │      Original: 1000 classes → New: N Emotion Classes       │
             │                                                            │
             │      FC: Linear(512 → N_classes)                           │
             └────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                           ┌────────────────────────────────────┐
                           │         Softmax Probabilities       │
                           │     (Predicted Emotion Category)    │
                           └────────────────────────────────────┘


## Data Augmentation Pipeline

                ┌─────────────────┐
                │   Raw Image     │
                └─────────────────┘
                          │
                          ▼
     ┌─────────────────────────────────────────────────────────────┐
     │                  Data Augmentation Pipeline                  │
     │--------------------------------------------------------------│
     │ RandomResizedCrop(224)                                       │
     │ RandomHorizontalFlip                                         │
     │ RandomRotation(10°)                                          │
     │ RandomAffine(shear=10)                                       │
     │ RandomPerspective                                            │
     │ ColorJitter                                                  │
     │ Cutout(mask_length=40)                                      │
     │ ToTensor + Normalize                                         │
     └─────────────────────────────────────────────────────────────┘
                          │
                          ▼
                   →  Model Input

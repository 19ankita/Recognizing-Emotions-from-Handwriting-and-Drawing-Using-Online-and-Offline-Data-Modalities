# Prototypical Networks for EMOTHAW (Few-Shot Learning)

This repository contains a **modern, PyTorch-based implementation of Prototypical Networks** rewritten from scratch for the EMOTHAW dataset.  
It replaces the original torchnet-based implementation with a clean, modular, thesis-friendly framework.

## Features
- Fully episodic few-shot training (N-way, K-shot)
- Prototypical Networks (Conv4 encoder)
- Train per-task or all tasks at once
- Windows + Anaconda compatible
- t-SNE embedding visualization
- Automatic results summarizer
- TensorBoard logging support
- No deprecated libraries (torchnet removed completely)

## Dataset Structure (EMOTHAW)
Expected layout for each task:
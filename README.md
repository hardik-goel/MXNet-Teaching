## Overview

This module implements a complete image classification pipeline in MXNet Gluon. It connects all concepts from the course – NDArrays, Gluon neural networks, training loops, evaluation, saving/loading models, and running real-time inference.

The goal is to build a small but production-style CNN image classifier end to end.

## Objectives

- Load and preprocess image data for training and inference.
- Define a CNN model using `gluon.nn.Sequential`.
- Train and evaluate the classifier on a labeled dataset.
- Save and reload the trained model parameters.
- Run inference on unseen images to simulate real deployment.

## Project Structure

Example structure (adapt to your repo):

## Pipeline Stages

1. **Data Loading & Preprocessing**
   - Load images from `data/train` and `data/val`.
   - Apply standard transforms:
     - Resize / center-crop (or random crop) to a fixed size.
     - Normalize pixel values.
     - Optionally add augmentation (flip, random crop, color jitter).
   - Wrap datasets using `gluon.data.Dataset` and iterate with `DataLoader`.

2. **Model Definition (Gluon)**
   - Define a CNN classifier using `gluon.nn.Sequential`, e.g.:
     - Convolution → BatchNorm → Activation → Pooling blocks.
     - One or more dense layers at the end.
   - Set the correct number of output units = number of classes.
   - Initialize parameters and choose the compute context (CPU/GPU).

3. **Training & Evaluation**
   - Use a standard supervised training loop:
     - Forward pass.
     - Compute loss (e.g., softmax cross-entropy).
     - Backward pass using autograd.
     - Parameter update via an optimizer (e.g., SGD/Adam).
   - Track:
     - Training loss and accuracy.
     - Validation accuracy per epoch.
   - Optionally implement:
     - Learning rate scheduling.
     - Early stopping or “best model” checkpointing.

4. **Saving & Loading the Model**
   - After training, save model parameters (e.g., to `saved_models/best_model.params`).
   - In `inference.py`, recreate the same network architecture and load the saved params.
   - This mirrors a production deployment step.

5. **Real-Time Inference**
   - Provide a simple CLI or function that:
     - Takes one or more image file paths as input.
     - Applies the same preprocessing as training.
     - Runs a forward pass through the trained model.
     - Prints top predicted class (and probability / score).
   - This simulates how the model would be used in a real application.

## How to Run

Adjust commands to match your actual file names:

1. (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate

2. Install dependencies
pip install mxnet gluoncv matplotlib numpy

3. Train the model
python -m src.train
--data-root ./data
--batch-size 32
--epochs 10
--lr 0.001
--save-path ./saved_models/best_model.params

4. Run inference on a new image
python -m src.inference
--model-path ./saved_models/best_model.params
--image ./sample_images/test.jpg



Update arguments/flags as per your implementation.

## Prerequisites

- Python 3.x
- MXNet (CPU or GPU build)
- Recommended:
  - `gluoncv` for pre-trained models and utilities.
  - GPU support for faster training (CUDA/cuDNN properly installed).

## Learning Wrap-Up

By completing this module, you:

- Implemented an end-to-end image classification system in MXNet Gluon.
- Applied:
  - NDArray operations,
  - Gluon model building,
  - Autograd-based training,
  - Evaluation and metrics,
  - Saving/loading models,
  - Inference on unseen data.
- Saw how concepts from earlier modules (CNNs, data pipelines, optimization, deployment thinking) come together in a realistic mini project.

Use this project as a base to:
- Swap in different architectures (e.g., ResNet from GluonCV).
- Experiment with data augmentation and regularization.
- Extend to more complex tasks (multi-class, transfer learning, etc.).




# TextureSeg-ML: Texture Image Segmentation via Feature Engineering

This repository contains an end-to-end Machine Learning pipeline developed for pixel-level **Texture Image Segmentation**. The project addresses 
the task of classifying different structural patterns within an image by extracting localized spatial features and applying tabular classification 
techniques.

The framework extracts distinct texture indicators using a custom 2D sliding window and evaluates multiple classifiers based on semantic overlap and 
computation metrics.

## 📌 Project Overview & Specifications
The segmentation architecture satisfies the following implementation constraints and evaluation standards:
- **Inputs:** A raw texture combination image (`input.jpg`) and its corresponding multi-colored ground truth target (`mask.png`) where each distinct
   RGB value represents a specific surface pattern.
- **Tabular Transformation:** Implements a rigorous **2D Sliding Window** feature extraction function with a fixed `stride = 1` and a dynamic window
  neighborhood size bound within the $[1, 20]$ range.
- **Data Pipeline:** Prepares a tabular dataset where each row maps the local window features of a specific coordinate to its respective pixel pattern
   class label.
- **Core Benchmarks:** Models are explicitly ranked and graded using a balanced formula: 50% based on the **Intersection over Union (IoU)** metric across
   the predicted image, and 50% based on code architecture clarity, visual plots, and structural logic.

## 🛠️ Methodology & Pipeline

1. **Spatial Feature Engineering:**
   - Traverses the raw matrix to extract localized window properties (such as pixel neighborhood statistics) to represent surface rough texture variations.
   - Maps unique RGB mask values directly into categorical target classes representing the structural segmentation zones.

2. **Deterministic Data Split:**
   - Enforces a strict seed protocol (`random_seed = 13`) across data operations.
   - Segregates the generated pixel dataset into an **80% Training** subset and a **20% Evaluation/Test** subset to guarantee unbiased validation.

3. **Segmentation Inference & Metrics:**
   - Trains classical classifiers alongside advanced tree ensembles to predict the correct label for every pixel coordinate across the entire scene.
   - Reconstructs the predicted labels back into a 2D image matrix to visually check segment boundaries.
   - Computes explicit operational trade-offs by measuring **Mean IoU scores**, **Training Time**, and **Test Inference Latency**.

## 📊 Model Evaluation & Key Results

- **Performance Rankings:** The evaluation demonstrates that ensemble architectures (e.g., Random Forest or Gradient Boosting variants) capture micro-texture
   discrepancies effectively, yielding highly distinct segmented boundaries.
- **Visual Analytics:** The standalone report provides reconstructed output segment images alongside categorical confusion matrices, charting out precisely
  where spatial boundaries blur or succeed.
- **Latency vs. Accuracy:** The notebook outlines the performance trade-offs, showcasing models that maintain high IoU overlap scores while keeping deployment
   inference times optimized.

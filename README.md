# Multimodal Transformer Fusion of CNN and Keypoint Representations for Yoga Pose Classification

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</div>

This project addresses the yoga posture classification problem by developing an automated system that can identify a pose from a single image. We implement a multimodal pipeline that fuses a **ResNet-18** image-encoded vector with a **YOLOv11-Pose** keypoint vector, feeding them into a lightweight **Multi-Token Transformer**. Our approach, tested on a dataset of 2,800 images across 47 classes, achieves a **+6%** increase in Top-1 accuracy over the best single-modality baseline, demonstrating the effectiveness of feature fusion.

---

## 📋 Table of Contents

1.  [Core Features](#-core-features)
2.  [Methodology](#-methodology)
3.  [Repository Structure](#-repository-structure)
4.  [Setup and Installation](#-setup-and-installation)
5.  [How to Run the Code](#-how-to-run-the-code)
6.  [Model Hyperparameters](#-model-hyperparameters)
7.  [Final Results](#-final-results)
8.  [Credits and Attribution](#-credits-and-attribution)

---

## ✨ Core Features

* **Hybrid Feature Fusion:** Combines 512-D visual embeddings from a fine-tuned ResNet-18 with 51-D geometric keypoint vectors from YOLOv11-Pose.
* **Multi-Token Transformer Architecture:** A custom Transformer that tokenizes keypoints into 17 individual tokens and processes them alongside a single CNN token and a learnable `[CLS]` token for robust feature integration.
* **Systematic Optimization:** Utilizes the **Optuna** framework with a TPESampler and MedianPruner for efficient and automated hyperparameter tuning.
* **Comparative Analysis:** Benchmarks the fused Transformer against two single-modality baselines: a fine-tuned ResNet-18 (CNN-only) and a basic MLP (keypoints-only).
* **Reproducible Pipeline:** Provides a full suite of documented scripts for data processing, training, and evaluation.

---

## 🧠 Methodology

The classification pipeline is designed to leverage the complementary strengths of both visual and geometric data.

1.  **Keypoint Extraction:** A **YOLOv11-Pose** model, pre-trained on the COCO dataset, is used to generate joint coordinates. For each image, it extracts the `(x, y)` coordinates and a confidence score for 17 keypoints, resulting in a **51-dimensional feature vector** (17 joints × 3 features).

2.  **CNN Feature Extraction:** A **ResNet-18** model, pre-trained on ImageNet, is fine-tuned on the yoga dataset. The final classification layer is then replaced with an identity mapping to extract a **512-dimensional feature vector**.

3.  **Multi-Token Transformer Fusion:**
    * **Tokenization:** The 51-D keypoint vector is split into 17 distinct 3-D tokens (one for each joint). The 512-D CNN vector is treated as a single token.
    * **Projection:** All 18 tokens (17 keypoint + 1 CNN) are projected into a shared latent space of **256 dimensions** using separate linear layers.
    * **CLS Token:** A special, learnable `[CLS]` token is prepended to the sequence, resulting in a total of 19 tokens. This token acts as a global summary of the entire input.
    * **Encoding:** Positional embeddings are added to the sequence, which is then fed into a **2-layer Transformer Encoder** with 8 attention heads. This allows the model to learn the relationships between all feature tokens.
    * **Classification:** The final output embedding of the `[CLS]` token is passed through a LayerNorm and a final linear layer to predict one of the 47 yoga pose classes.

---

## 📂 Repository Structure

The project is organized with a clear separation between different models and data processing utilities.
Deep Learning Project- Yoga pose estimation
Custom Gem

Show thinking
Of course. Based on the detailed information in your final report, I have updated the README.md to accurately reflect your project's methodology, architecture, and final results.

This version is much more detailed and aligns perfectly with the academic rigor of your report.

Copy the Full Updated README.md Script Below:
Markdown

# Multimodal Transformer Fusion of CNN and Keypoint Representations for Yoga Pose Classification

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</div>

This project addresses the yoga posture classification problem by developing an automated system that can identify a pose from a single image. We implement a multimodal pipeline that fuses a **ResNet-18** image-encoded vector with a **YOLOv11-Pose** keypoint vector, feeding them into a lightweight **Multi-Token Transformer**. Our approach, tested on a dataset of 2,800 images across 47 classes, achieves a **+6%** increase in Top-1 accuracy over the best single-modality baseline, demonstrating the effectiveness of feature fusion.

---

## 📋 Table of Contents

1.  [Core Features](#-core-features)
2.  [Methodology](#-methodology)
3.  [Repository Structure](#-repository-structure)
4.  [Setup and Installation](#-setup-and-installation)
5.  [How to Run the Code](#-how-to-run-the-code)
6.  [Model Hyperparameters](#-model-hyperparameters)
7.  [Final Results](#-final-results)
8.  [Credits and Attribution](#-credits-and-attribution)

---

## ✨ Core Features

* **Hybrid Feature Fusion:** Combines 512-D visual embeddings from a fine-tuned ResNet-18 with 51-D geometric keypoint vectors from YOLOv11-Pose.
* **Multi-Token Transformer Architecture:** A custom Transformer that tokenizes keypoints into 17 individual tokens and processes them alongside a single CNN token and a learnable `[CLS]` token for robust feature integration.
* **Systematic Optimization:** Utilizes the **Optuna** framework with a TPESampler and MedianPruner for efficient and automated hyperparameter tuning.
* **Comparative Analysis:** Benchmarks the fused Transformer against two single-modality baselines: a fine-tuned ResNet-18 (CNN-only) and a basic MLP (keypoints-only).
* **Reproducible Pipeline:** Provides a full suite of documented scripts for data processing, training, and evaluation.

---

## 🧠 Methodology

The classification pipeline is designed to leverage the complementary strengths of both visual and geometric data.

1.  **Keypoint Extraction:** A **YOLOv11-Pose** model, pre-trained on the COCO dataset, is used to generate joint coordinates. For each image, it extracts the `(x, y)` coordinates and a confidence score for 17 keypoints, resulting in a **51-dimensional feature vector** (17 joints × 3 features).

2.  **CNN Feature Extraction:** A **ResNet-18** model, pre-trained on ImageNet, is fine-tuned on the yoga dataset. The final classification layer is then replaced with an identity mapping to extract a **512-dimensional feature vector**.

3.  **Multi-Token Transformer Fusion:**
    * **Tokenization:** The 51-D keypoint vector is split into 17 distinct 3-D tokens (one for each joint). The 512-D CNN vector is treated as a single token.
    * **Projection:** All 18 tokens (17 keypoint + 1 CNN) are projected into a shared latent space of **256 dimensions** using separate linear layers.
    * **CLS Token:** A special, learnable `[CLS]` token is prepended to the sequence, resulting in a total of 19 tokens. This token acts as a global summary of the entire input.
    * **Encoding:** Positional embeddings are added to the sequence, which is then fed into a **2-layer Transformer Encoder** with 8 attention heads. This allows the model to learn the relationships between all feature tokens.
    * **Classification:** The final output embedding of the `[CLS]` token is passed through a LayerNorm and a final linear layer to predict one of the 47 yoga pose classes.

---

## 📂 Repository Structure

The project is organized with a clear separation between different models and data processing utilities.
```text
.
├── .gitignore
├── README.md
├── datasets/
    ├──
└── src/
    ├── ResNet18_classifier/
    │   └── resnet18_fine_tuned_embbeding_extractor.py
    │   └── resnet_fine_tuning_process.py
    │   └── best_fine_tune_resnet18.pth
    ├── Data_processing/
    │   ├── combined_csv_creator.py
    │   ├── make_counter_classes.py
    │   ├── train_val_test_divide.py
    ├── Pose_keypoints_labeling/
    │   ├── YogaClassifier.py
    │   └── yolo_pose_confidence_labeler.py
    ├── MLP_classifier/
    │   ├── evaluate_MLP_classifier.py
    │   └── MLP_classifier_training.py  
    └── Transformer/
        ├── MultiTokenTransformer.py
        ├── transformer_train_loop.py
        ├── optimize_multi_token_transformer.py
        └── evaluate_best_multi_token_trans.py

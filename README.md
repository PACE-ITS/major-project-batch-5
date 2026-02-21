# ADPM3D: Alzheimer's Disease Prediction with 3D & 2D MRI Analysis

**Batch ID:** Batch-5  
**Course:** Undergrad Major Project 2026  
**Institution:** PACE Institute of Technology and Sciences

---

## 👥 Team Members
| Name | Roll Number | GitHub Handle |
| :--- | :--- | :--- |
| T. Priyanka Reddy | 22KQ1A6128 | @T-Priyanka-Reddy |
| B. Hepsiba Rani | 22KQ1A6104 | @B-Hepsiba-Rani |
| G. Mukesh Anand | 23KQ5A6101 | @G-Mukesh-Anand |
| S. Chandra Sekhar | 22KQ1A6155 | @S-Chandra-Sekhar |
| N. Raja | 22KQ1A6153 | @N-Raja |

---

## 🚀 Project Overview
**Problem Statement:** Alzheimer's Disease (AD) is a progressive neurodegenerative disorder. Early detection is critical for intervention. This project utilizes research-grade Artificial Intelligence to analyze both 3D volumetric MRI and 2D slice sequences for high-precision classification.

**Technical Objective:** To implement a robust, end-to-end pipeline for Alzheimer's classification using specialized deep learning architectures that surpass standard diagnostic baselines.

---

## 🧠 Hybrid Model Architectures
We utilized a unified **CNN + BiLSTM + Attention** strategy tailored for medical imaging:

### 1. 2D MCC Model (Multi-Class Classification)
* **Backbone**: EfficientNet-B0 (Pre-trained on ImageNet)
* **Sequential Layer**: 2-layer Bidirectional LSTM (512 hidden units)
* **Optimization**: Focal Loss for class imbalance + AdamW Optimizer
* **Classification**: 4 distinct phases of Alzheimer's progression.

### 2. 3D MCC Model (Volumetric Multi-Class)
* **Backbone**: Research-grade 3D CNN with Volumetric Attention
* **Sequential Layer**: Bidirectional LSTM for Z-axis voxel dependency
* **Preprocessing**: 3D Skull Stripping & Z-Score Normalization
* **Classification**: 3-way multi-class (Normal, MCI, AD).

---

## 📂 Repository Directory Guide

### 📂 /3D_model
* **`3D_model_MCC.py`**: The primary training module for 3D ADNI analysis. Includes automated skull stripping, volumetric data augmentation, and the hybrid 3D architecture.
* **`test_3D_MCC.py`**: A specialized evaluation suite for the 3D model, performing multi-class inference and generating statistical plots.

### 📂 /2D_model
* **`2D_model_MCC.py`**: The core of the 2D slice pipeline. Implements ROI-based cropping, CLAHE enhancement, and the EfficientNet-BiLSTM hybrid framework.
* **`test_2D_MCC.py`**: Generates terminal-based classification reports and visual performance summaries for 2D data.

### 📂 /results_2D & /results_3D
Contains performance artifacts (Classification Reports, Confusion Matrices, ROC Curves) demonstrating the model's reliability.

### 📂 /scripts
* **`convert_all_dcm_to_nii.py`**: Batch processing utility to convert raw DICOM folders into 3D NIfTI volumes.
* **`datasetprocessing.py`**: Automated data distribution and directory management script.

---

## 📈 Analysis of Results

### 🔬 2D MRI Analysis (4-Class Classification)
The 2D model demonstrated near-perfect identification across the cognitive spectrum.
* **Classes**: Normal AD (NonDemented), Very Mild AD, Mild AD, and Moderate AD.
* **Performance**: **98.33% Accuracy** with a 0.98 F1-Score.

````carousel
![2D Confusion Matrix](file:///c:/Users/HP/OneDrive/Desktop/ADPM3D/results_2D/confusion_matrix.png)
<!-- slide -->
![2D ROC Curve](file:///c:/Users/HP/OneDrive/Desktop/ADPM3D/results_2D/roc_curve.png)
````

### 🔬 3D Volumetric Analysis (3-Class Classification)
The 3D model efficiently captures volumetric shrinkage and feature dependencies in 64x64x64 brain blocks.
* **Classes**: Control Normal (CN), Mild Cognitive Impairment (MCI), and Alzheimer's Disease (AD).
* **Performance**: **91.42% Accuracy** (Phase 5 Optimization).

````carousel
![3D Confusion Matrix](file:///c:/Users/HP/OneDrive/Desktop/ADPM3D/results_3D/confusion_matrix.png)
<!-- slide -->
![3D ROC Curve](file:///c:/Users/HP/OneDrive/Desktop/ADPM3D/results_3D/roc_curve.png)
````

---

## 🛠️ Installation & Execution
```bash
# Install dependencies
pip install -r requirements.txt

# Run 3D Pipeline
python 3D_model/3D_model_MCC.py

# Run 2D Pipeline
python 2D_model/2D_model_MCC.py
```

---

## 📊 Data Sources
* **ADNI LONI (3D Data)**: Collected research volumes for AD, MCI, and CN subjects.
* **Kaggle Alzheimer (2D Data)**: High-resolution slice dataset for 4-class multi-classification.
* **Processed Link**: [Google Drive Repository](https://drive.google.com/drive/folders/1hhdaOP83lqRjZ0VO_imLOJXO5lJNHlfs?usp=sharing)

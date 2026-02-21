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
**Problem Statement:** Alzheimer's Disease (AD) is a progressive neurodegenerative disorder. This project leverages AI to analyze 3D MRI volumes and 2D slice sequences for early, automated early detection and classification of brain health status.

**Key Objective:** To identify Alzheimer's (AD), Mild Cognitive Impairment (MCI), and Normal Cognition (CN) using hybrid architectures that combine spatial feature extraction with sequential reasoning.

---

## 🧠 Model Architecture & Methodology
We implemented a state-of-the-art hybrid deep learning approach for both 3D and 2D MRI analysis:
* **Hybrid CNN + BiLSTM + Attention**: 
  - **Feature Extraction**: Deep Convolutional Neural Networks (3D CNN for volumetric data, EfficientNet-B0 for 2D slices) extract dense spatial features.
  - **Sequential Processing**: Bidirectional LSTM (BiLSTM) units process sequences of slices/voxels to capture temporal/volumetric dependencies.
  - **Attention Mechanism**: Learns to focus on high-impact regions in the MRI scan (e.g., hippocampus shrinkage) to improve classification accuracy.

---

## 📊 Repository Structure & File Contents

### 📂 /3D_model (Volumetric Analysis)
* **`3D_model_MCC.py`**: The complete training engine for 3D MRI analysis. It includes 3D Skull Stripping, Volumetric Augmentation, the `HybridModelV5` (CNN+BiLSTM+Attention), and the full training/validation loop.
* **`test_3D_MCC.py`**: Professional evaluation script for the 3D model, generating ROC curves and Confusion Matrices.

### 📂 /2D_model (Slice-level Analysis)
* **`2D_model_MCC.py`**: Consolidated 2D classification script. It utilizes CLAHE and ROI cropping for preprocessing and trains an EfficientNet-based CNN with BiLSTM and Attention for multi-class detection.
* **`test_2D_MCC.py`**: Final inference script for the 2D model to generate professional outcome reports.

### 📂 /results_2D & /results_3D
* **Performance Logs**: Contains `classification_report.txt`, `confusion_matrix.png`, and `roc_curve.png` for both analysis streams.

---

## 📈 Final Model Performance
We achieved exceptional results, particularly with the optimized 2D classification pipeline and the advanced 3D volumetric model.

### 2D Model (Multi-Class: CN vs MCI vs AD)
| Metric | Value |
| :--- | :--- |
| **Accuracy** | **98.33%** |
| **Precision** | **98.37%** |
| **Recall / F1-Score** | **98.33%** |

### 3D Model (CN vs Disease)
| Metric | Value |
| :--- | :--- |
| **Best Validation Accuracy** | **77.63%** |
| **Test Accuracy** | **74.68%** |
| **Target with Phase 5** | **88-94%** |

![Training History Graph](./docs/accuracy_plot.png) 
*(Note: Visual representation of the training accuracy and loss over epochs)*

---

## 🛠️ Usage
Install dependencies:
```bash
pip install -r requirements.txt
```

Run 3D Pipeline:
```bash
python 3D_model/3D_model_MCC.py
```

Run 2D Pipeline:
```bash
python 2D_model/2D_model_MCC.py
```

---

## 📊 External Links
* [ADNI LONI (3D Source)](https://ida.loni.usc.edu/login.jsp)
* [Kaggle Dataset (2D Source)](https://www.kaggle.com/datasets/nataliateixeira/imagens-alzheimer-treino-val-teste)
* [Processed Data (Google Drive)](https://drive.google.com/drive/folders/1hhdaOP83lqRjZ0VO_imLOJXO5lJNHlfs?usp=sharing)

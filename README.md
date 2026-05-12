# ADPM3D: Alzheimer's Disease Prediction with 3D & 2D MRI Analysis

**Batch ID:** Batch-5  
**Course:** Undergrad Major Project 2026  
**Institution:** PACE Institute of Technology and Sciences, Ongole, Andhra Pradesh

---

## 👥 Team Members

| Name | Roll Number | GitHub Handle |
| :--- | :--- | :--- |
| T. Priyanka Reddy | 22KQ1A6128 | @T-Priyanka-Reddy |
| B. Hepsiba Rani | 22KQ1A6104 | @B-Hepsiba-Rani |
| G. Mukesh Anand | 23KQ5A6101 | @G-Mukesh-Anand |
| S. Chandra Sekhar | 22KQ1A6155 | @S-Chandra-Sekhar |
| N. Raja | 22KQ1A6153 | @N-Raja |

**Guide:** Dr. Yedukondalu Jammisetty, Dept. of AI & ML

---

## 🚀 Project Overview

**Problem Statement:** Alzheimer's Disease (AD) is a progressive neurodegenerative disorder that is the primary cause of dementia worldwide. Early detection during the Mild Cognitive Impairment (MCI) stage is critical for timely clinical intervention. Manual interpretation of volumetric MRI scans is time-consuming, expertise-dependent, and prone to inter-observer variability — making automated diagnostic systems essential.

**Technical Objective:** To implement a hybrid deep learning framework combining 3D CNN, BiLSTM, and Attention mechanisms for robust subject-level Alzheimer's classification using both 3D volumetric MRI (ADNI) and 2D slice-based MRI (Kaggle) datasets. The study evaluates the trade-off between volumetric anatomical fidelity and computational efficiency.

---

## 🧠 Hybrid Model Architectures

The unified pipeline integrates **3D/2D CNN + BiLSTM + Attention** for medical imaging:

### 1. 3D CNN–BiLSTM Model (Volumetric Multi-Class)
- **Backbone**: Stacked 3D Convolutional Neural Network layers
- **Feature Reduction**: Global Average Pooling (GAP)
- **Sequential Layer**: Bidirectional LSTM — captures contextual dependencies between adjacent MRI slices in both forward and backward directions
- **Attention**: Post-BiLSTM attention mechanism for adaptive feature weighting on disease-relevant brain regions
- **Preprocessing**: DICOM → NIfTI conversion, resampled to 128×128×128, Min-Max normalization
- **Classifier**: Fully connected Softmax layer
- **Classification**: 3-way (Cognitively Normal, MCI, Alzheimer's Disease)
- **Optimizer**: AdamW (lr = 1×10⁻⁴) with cosine annealing, dropout, and early stopping

### 2. 2D CNN–BiLSTM Model (Slice-Based Multi-Class)
- **Backbone**: EfficientNet-B0 (pretrained on ImageNet, compound scaling)
- **Sequential Layer**: Bidirectional LSTM for inter-slice contextual learning
- **Attention**: Attention layer for discriminative feature refinement
- **Preprocessing**: Resized to 128×128, Z-score normalization, augmentation (rotation, flipping, brightness)
- **Classifier**: Fully connected Softmax layer
- **Classification**: 4-way (NonDemented, VeryMildDemented, MildDemented, ModerateDemented)
- **Loss Function**: Focal Loss for class imbalance + AdamW Optimizer

---

## 📊 Data Sources

### 3D Dataset — ADNI (Alzheimer's Disease Neuroimaging Initiative)
T1-weighted MRI scans (MPRAGE protocol) from the [ADNI IDA portal](https://ida.loni.usc.edu/login.jsp). Subject-wise split: 70% train / 15% validation / 15% test.

| Class | Subjects | Samples |
| :--- | :---: | :---: |
| Cognitively Normal (CN) | 55 | 304 |
| Mild Cognitive Impairment (MCI) | 55 | 370 |
| Alzheimer's Disease (AD) | 55 | 243 |
| **Total** | **165** | **917** |

### 2D Dataset — Kaggle Alzheimer MRI
6400 MRI slices across 4 dementia stages.

| Class | Images |
| :--- | :---: |
| NonDemented | 2240 |
| VeryMildDemented | 1568 |
| MildDemented | 627 |
| ModerateDemented | 45 |
| **Total** | **6400** |

**Processed Data:** [Google Drive Repository](https://drive.google.com/drive/folders/1hhdaOP83lqRjZ0VO_imLOJXO5lJNHlfs?usp=sharing)

---

## 📈 Results

### 🔬 3D Volumetric Analysis (3-Class)

| Model | Accuracy (%) | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **3D CNN + BiLSTM (Proposed)** | **91.43** | **0.92** | **0.91** | **0.92** |
| Multi-View CNN + LSTM | 86.96 | 0.8619 | 0.8555 | 0.8661 |
| 3D CNN + LSTM | 82.16 | 0.8974 | 0.8120 | 0.8688 |

### 🔬 2D Slice-Based Analysis (4-Class)

| Model | Accuracy (%) | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **EfficientNet + BiLSTM (Proposed)** | **98.33** | **0.9837** | **0.9833** | **0.9833** |
| CNN + LSTM | 83.00 | 0.87 | 0.80 | 0.85 |

### 🏆 Comparison with Existing Models (3D)

| Model | Train Accuracy (%) | Test Accuracy (%) |
| :--- | :---: | :---: |
| **Proposed 3D CNN–BiLSTM** | **92.15** | **91.43** |
| InceptionResNetV2 | 90.9 | 90.7 |
| DenseNet121 | 87.5 | 86.5 |
| MobileNetV2 | 88.4 | 87.3 |
| Xception | 86.5 | 83.8 |
| VGG16 | 83.5 | 84.5 |

> **Key Insight:** While the 2D model achieves higher numerical accuracy, the 3D subject-level model is more anatomically faithful and clinically reliable, as real-world diagnosis is performed per patient — not per image slice.

---

## 📂 Repository Structure

```
ADPM3D/
├── 3D_model/
│   ├── 3D_model_MCC.py        # 3D training: skull stripping, volumetric augmentation, CNN–BiLSTM
│   └── test_3D_MCC.py         # 3D evaluation: multi-class inference, statistical plots
├── 2D_model/
│   ├── 2D_model_MCC.py        # 2D training: ROI cropping, CLAHE, EfficientNet–BiLSTM
│   └── test_2D_MCC.py         # 2D evaluation: classification reports, visual summaries
├── results_2D/                # Confusion matrix, ROC curves (2D)
├── results_3D/                # Confusion matrix, ROC curves (3D)
├── scripts/
│   ├── convert_all_dcm_to_nii.py  # Batch DICOM → NIfTI conversion
│   └── datasetprocessing.py       # Dataset distribution & directory management
└── requirements.txt
```

---

## 🛠️ Installation & Execution

```bash
# Install dependencies
pip install -r requirements.txt

# Run 3D Volumetric Pipeline (ADNI)
python 3D_model/3D_model_MCC.py

# Run 2D Slice-Based Pipeline (Kaggle)
python 2D_model/2D_model_MCC.py

# Convert DICOM to NIfTI (preprocessing)
python scripts/convert_all_dcm_to_nii.py
```

---

## 🔬 Methodology Summary

The overall pipeline for both models:

1. **MRI Preprocessing** — DICOM loading, slice ordering, 3D stacking, NIfTI export, resampling, intensity normalization
2. **Convolutional Feature Extraction** — 3D CNN (volumetric) or EfficientNet-B0 (slice-level)
3. **Sequential Modeling** — BiLSTM captures inter-slice contextual dependencies (forward + backward)
4. **Attention Refinement** — Adaptive weights focus on disease-relevant regions (hippocampal atrophy, cortical thinning)
5. **Classification** — Softmax classifier with multi-class cross-entropy loss

---

## 🔭 Future Work

- Expand dataset with larger, more diverse ADNI cohorts for improved generalization
- Longitudinal MRI studies to model disease stage progression (CN → MCI → AD)
- Multi-modal fusion: PET scans, CSF biomarkers, genetic data, and clinical assessments
- Vision Transformer and CNN-Transformer hybrid architectures for global feature learning
- Explainable AI (Grad-CAM) for clinical interpretability and adoption

---

## 📄 Citation

> T. Priyanka Reddy, B. Hepsiba Rani, G. Mukesh Anand, S. Chandra Sekhar, N. Raja, and J. Yedukondalu, *"Alzheimer's Disease Prediction through 3D/2D MRI Images using 3D CNN and BiLSTM,"* PACE Institute of Technology and Sciences, 2026.

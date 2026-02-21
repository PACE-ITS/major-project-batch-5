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
**Problem Statement:** Alzheimer's Disease (AD) is a progressive neurodegenerative disorder. This project leverages AI to analyze 3D MRI volumes and 2D slice sequences for early detection and classification.

**Key Objective:** To identify Alzheimer's and Mild Cognitive Impairment using hybrid 3D CNN and 2D sequence-based deep learning architectures.

---

## 🧠 Model Architecture
* **3D Module (`/3D_model`)**: Advanced 3D CNN utilizing a HybridModelV5 architecture with Attention mechanisms and BiLSTM for volumetric feature extraction.
* **2D Module (`/2D_model`)**: Optimized CNN-BiLSTM-Attention architecture utilizing CLAHE-preprocessed 2D slices for multi-class impairment detection.

---

## 📈 Results
| Model | Accuracy | F1-Score |
| :--- | :--- | :--- |
| **3D Model** | 77.63% | 0.76 |
| **2D Model** | 58.00% | 0.54 |

---

## 🛠️ Usage
### 1. Requirements
```bash
pip install -r requirements.txt
```

### 2. Execution
**3D Model:**
```bash
python 3D_model/3D_model_MCC.py
```

**2D Model:**
```bash
python 2D_model/2D_model_MCC.py
```

---

## 📊 Dataset Links
* [LONI IDA (3D Source)](https://ida.loni.usc.edu/login.jsp)
* [Kaggle Alzheimer (2D Source)](https://www.kaggle.com/datasets/nataliateixeira/imagens-alzheimer-treino-val-teste)
* [Processed Data (Google Drive)](https://drive.google.com/drive/folders/1hhdaOP83lqRjZ0VO_imLOJXO5lJNHlfs?usp=sharing)

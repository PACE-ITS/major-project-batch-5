
import os
import glob
import time
import random
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
class Config:
    PROJECT_NAME = "AD_Classification_Phase5"
    DATA_DIR = os.path.join(os.getcwd(), 'data', 'nifti_converted')
    RESULTS_DIR = "results_phase5"
    
    INPUT_SHAPE = (64, 64, 64)
    NUM_CLASSES = 3
    CLASSES = ['CN', 'MCI', 'AD']
    
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    WARMUP_EPOCHS = 5
    EPOCHS = 200  # Increased for convergence with high regularization
    WEIGHT_DECAY = 5e-3 # Extreme L2 regularization
    DROPOUT = 0.7       # Extreme dropout to fight overfitting
    LABEL_SMOOTHING = 0.1
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

set_seed(Config.SEED)
os.makedirs(Config.RESULTS_DIR, exist_ok=True)

# ============================================================================
# 2. PREPROCESSING & AUGMENTATION
# ============================================================================
def skull_strip(data):
    """3D Skull Stripping using morphology."""
    mask = data > (data.mean() * 0.5)
    mask = ndimage.binary_fill_holes(mask)
    label_im, nb_labels = ndimage.label(mask)
    if nb_labels > 0:
        sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
        mask_size = sizes < max(sizes)
        remove_pixel = mask_size[label_im]
        mask[remove_pixel] = 0
    mask = ndimage.binary_closing(mask, iterations=2)
    return data * mask

def augment_3d(img):
    """Aggressive 3D Volumetric Augmentation."""
    # 1. Random Rotation (Axial/Z-axis)
    if random.random() > 0.5:
        angle = random.uniform(-15, 15)
        img[0] = torch.tensor(ndimage.rotate(img[0].numpy(), angle, axes=(1, 2), reshape=False, order=1))
    
    # 2. Random Rotation (Sagittal/X-axis)
    if random.random() > 0.7:
        angle = random.uniform(-10, 10)
        img[0] = torch.tensor(ndimage.rotate(img[0].numpy(), angle, axes=(0, 1), reshape=False, order=1))

    # 3. Random Scaling
    if random.random() > 0.7:
        scale = random.uniform(0.9, 1.1)
        # Use zoom for scaling
        h, w, d = img[0].shape
        zoomed = ndimage.zoom(img[0].numpy(), [scale, scale, scale], order=1)
        # Pad/Crop back to original size
        zh, zw, zd = zoomed.shape
        if zh >= h: # Crop
            start_h = (zh - h) // 2
            start_w = (zw - w) // 2
            start_d = (zd - d) // 2
            img[0] = torch.tensor(zoomed[start_h:start_h+h, start_w:start_w+w, start_d:start_d+d])
        else: # Pad
            pad_h = (h - zh) // 2
            pad_w = (w - zw) // 2
            pad_d = (d - zd) // 2
            padded = np.zeros((h, w, d), dtype=np.float32)
            padded[pad_h:pad_h+zh, pad_w:pad_w+zw, pad_d:pad_d+zd] = zoomed
            img[0] = torch.tensor(padded)
            
    # 4. Intensity Shifting
    if random.random() > 0.5:
        shift = random.uniform(-0.1, 0.1)
        img = img + shift
        
    return img

def load_and_preprocess():
    print(f"--- Inventorying Data ---")
    file_list = []
    class_map = {name: i for i, name in enumerate(Config.CLASSES)}
    for cls_name in Config.CLASSES:
        cls_dir = os.path.join(Config.DATA_DIR, cls_name)
        files = glob.glob(os.path.join(cls_dir, "**", "*.nii*"), recursive=True)
        for f in files: file_list.append((f, class_map[cls_name]))
            
    print(f"--- Loading & Preprocessing ({len(file_list)} volumes) ---")
    all_volumes, all_labels = [], []
    start_time = time.time()
    for idx, (path, label) in enumerate(file_list):
        try:
            nii = nib.load(path)
            data = nii.get_fdata().astype(np.float32)
            data = np.nan_to_num(data)
            data = skull_strip(data)
            zoom_factors = [t / s for t, s in zip(Config.INPUT_SHAPE, data.shape)]
            data = ndimage.zoom(data, zoom_factors, order=1)
            # Z-Score
            brain_mask = data > 0
            if np.any(brain_mask):
                data[brain_mask] = (data[brain_mask] - data[brain_mask].mean()) / (data[brain_mask].std() + 1e-8)
            all_volumes.append(data); all_labels.append(label)
            if (idx + 1) % 50 == 0: print(f"  [{idx+1}/{len(file_list)}] volumes loaded...")
        except Exception as e: print(f"Error loading {path}: {e}")

    X = torch.tensor(np.array(all_volumes), dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(all_labels, dtype=torch.long)
    print(f"Preprocessed in {time.time()-start_time:.1f}s")
    return X, y

# ============================================================================
# 3. ARCHITECTURE (Phase 5 Refined)
# ============================================================================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        weights = torch.tanh(self.attn(x))
        weights = F.softmax(weights, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context, weights

class HybridModelV5(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Bottleneck blocks to reduce parameter overfitting
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)
        )
        self.conv2 = self._make_block(24, 48)
        self.conv3 = self._make_block(48, 96)
        self.conv4 = self._make_block(96, 128)
        
        # GAP to reduce spatial dimensionality before LSTM
        self.gap = nn.AdaptiveAvgPool3d((2, 2, 2))
        
        self.lstm = nn.LSTM(128 * 2 * 2, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.attention = Attention(256)
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def _make_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)
        )

    def forward(self, x):
        x = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        x = self.gap(x)
        b, c, d, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b, d, -1)
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        return self.classifier(context)

# ============================================================================
# 4. VISUALIZATION
# ============================================================================
def plot_training_history(history, save_dir):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['t_loss'], label='Train Loss')
    plt.plot(history['v_loss'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['t_acc'], label='Train Acc')
    plt.plot(history['v_acc'], label='Val Acc')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curves.png'))
    plt.close()

def plot_confusion_matrix_custom(y_true, y_pred, classes, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def plot_roc_curve_custom(y_true, y_probs, classes, save_dir):
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{cls} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()

# ============================================================================
# 5. TRAINING ENGINE
# ============================================================================
def train():
    X, y = load_and_preprocess()
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, stratify=y, random_state=Config.SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=Config.SEED)
    
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train.numpy())
    class_weights = torch.tensor(weights, dtype=torch.float32).to(Config.DEVICE)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=Config.BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=Config.BATCH_SIZE)
    
    model = HybridModelV5().to(Config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    # Warmup + Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS - Config.WARMUP_EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=Config.LABEL_SMOOTHING)
    
    history = {'t_loss': [], 't_acc': [], 'v_loss': [], 'v_acc': []}
    best_val_acc = 0
    patience_counter = 0
    
    print("\n--- Phase 5 Training Started ---")
    for epoch in range(Config.EPOCHS):
        model.train()
        t_l, t_c, t_tot = 0, 0, 0
        s_t = time.time()
        
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(Config.DEVICE), lbls.to(Config.DEVICE)
            
            # --- Apply Aggressive 3D Augmentation ---
            aug_imgs = torch.stack([augment_3d(img) for img in imgs.cpu()]).to(Config.DEVICE)
            
            optimizer.zero_grad()
            out = model(aug_imgs)
            loss = criterion(out, lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            t_l += loss.item() * imgs.size(0)
            t_c += (out.argmax(1) == lbls).sum().item()
            t_tot += lbls.size(0)
            
        model.eval()
        v_l, v_c, v_tot = 0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(Config.DEVICE), lbls.to(Config.DEVICE)
                out = model(imgs)
                loss = criterion(out, lbls)
                v_l += loss.item() * imgs.size(0)
                v_c += (out.argmax(1) == lbls).sum().item()
                v_tot += lbls.size(0)
        
        t_acc, v_acc = t_c/t_tot, v_c/v_tot
        if epoch >= Config.WARMUP_EPOCHS: scheduler.step()
        
        history['t_loss'].append(t_l/t_tot); history['v_loss'].append(v_l/v_tot)
        history['t_acc'].append(t_acc); history['v_acc'].append(v_acc)
        
        print(f"E[{epoch+1}/{Config.EPOCHS}] {time.time()-s_t:.1f}s | T_Loss: {t_l/t_tot:.3f} T_Acc: {t_acc:.3f} | V_Loss: {v_l/v_tot:.3f} V_Acc: {v_acc:.3f}")
        
        if v_acc > best_val_acc:
            best_val_acc = v_acc; patience_counter = 0
            torch.save(model.state_dict(), os.path.join(Config.RESULTS_DIR, 'best_model.pth'))
        else: patience_counter += 1
        
        # No early stopping per user request - full 100 epochs
        # if patience_counter >= 20: break

    # Final Evaluation & Plotting
    plot_training_history(history, Config.RESULTS_DIR)
    
    model.load_state_dict(torch.load(os.path.join(Config.RESULTS_DIR, 'best_model.pth')))
    model.eval()
    y_t, y_p, y_s = [], [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(Config.DEVICE)
            out = model(imgs)
            probs = F.softmax(out, dim=1)
            y_t.extend(lbls.numpy())
            y_p.extend(out.argmax(1).cpu().numpy())
            y_s.extend(probs.cpu().numpy())
    
    y_s = np.array(y_s)
    print("\nPhase 5 Final Report:")
    print(classification_report(y_t, y_p, target_names=Config.CLASSES))
    print(f"Final Accuracy: {accuracy_score(y_t, y_p)*100:.2f}%")
    
    plot_confusion_matrix_custom(y_t, y_p, Config.CLASSES, Config.RESULTS_DIR)
    plot_roc_curve_custom(y_t, y_s, Config.CLASSES, Config.RESULTS_DIR)
    print(f"--- Visualizations saved to {Config.RESULTS_DIR} ---")

if __name__ == "__main__":
    train()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np
import os
import cv2
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import random

# ============================================================================
# 1. CORE UTILITIES
# ============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose: print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose: print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# ============================================================================
# 2. DATA PROCESSING
# ============================================================================
class Alzheimer2DDataset(Dataset):
    def __init__(self, data_list, labels, transform=None):
        self.data_list = data_list
        self.labels = labels
        self.transform = transform

    @staticmethod
    def apply_clahe(img):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)

    @staticmethod
    def crop_roi(img):
        mask = img > 10
        coords = np.argwhere(mask)
        if coords.size > 0:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            return img[y0:y1+1, x0:x1+1]
        return img

    def __len__(self): return len(self.data_list)

    def __getitem__(self, idx):
        file_path = self.data_list[idx]
        label = self.labels[idx]
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None: img = np.zeros((128, 128), dtype=np.uint8)
        img = ((img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8) * 255).astype(np.uint8)
        img = self.crop_roi(img)
        img = self.apply_clahe(img)
        if self.transform: img = self.transform(img)
        else:
            img = cv2.resize(img, (128, 128))
            img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        return img, label

def get_dataloaders(base_path, batch_size=32, seed=42):
    classes = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
    all_paths = []; all_labels = []
    for idx, cls in enumerate(classes):
        cls_path = os.path.join(base_path, cls)
        if os.path.exists(cls_path):
            files = []
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                files.extend(glob.glob(os.path.join(cls_path, ext)))
            all_paths.extend(files)
            all_labels.extend([idx] * len(files))
    
    if not all_paths: raise ValueError(f"No data found in {base_path}")
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(all_paths, all_labels, test_size=0.30, stratify=all_labels, random_state=seed)
    val_paths, test_paths, val_labels, test_labels = train_test_split(temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=seed)
    
    train_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((128, 128)), transforms.RandomRotation(10), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    val_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    
    train_loader = DataLoader(Alzheimer2DDataset(train_paths, train_labels, train_transform), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Alzheimer2DDataset(val_paths, val_labels, val_transform), batch_size=batch_size)
    test_loader = DataLoader(Alzheimer2DDataset(test_paths, test_labels, val_transform), batch_size=batch_size)
    return train_loader, val_loader, test_loader, classes

# ============================================================================
# 3. ARCHITECTURE
# ============================================================================
class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))
    def forward(self, x):
        weights = F.softmax(self.attention(x), dim=1)
        return torch.sum(weights * x, dim=1), weights

class CNN_BiLSTM_Attention(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.feature_extractor = self.backbone.features
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.lstm = nn.LSTM(input_size=1280, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = AttentionBlock(hidden_dim=512)
        self.fc = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, num_classes))

    def forward(self, x):
        if x.size(1) == 1: x = x.repeat(1, 3, 1, 1)
        x = self.feature_extractor(x)
        x = self.pool(x)
        B, C, H, W = x.size()
        x = x.view(B, C, H*W).permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        return self.fc(context)

# ============================================================================
# 4. TRAINING & EVALUATION
# ============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train(); running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images); loss = criterion(outputs, labels)
        loss.backward(); optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1); total += labels.size(0); correct += predicted.eq(labels).sum().item()
    return running_loss / total, 100. * correct / total

def validate_epoch(model, loader, criterion, device):
    model.eval(); running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images); loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1); total += labels.size(0); correct += predicted.eq(labels).sum().item()
    return running_loss / total, 100. * correct / total

def evaluate_model(model, loader, device, classes, results_dir):
    model.eval(); all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images); probs = torch.softmax(outputs, dim=1); _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy()); all_labels.extend(labels.cpu().numpy()); all_probs.extend(probs.cpu().numpy())
    
    all_preds, all_labels, all_probs = np.array(all_preds), np.array(all_labels), np.array(all_probs)
    os.makedirs(results_dir, exist_ok=True)
    
    # Visuals
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8)); sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix'); plt.savefig(os.path.join(results_dir, 'confusion_matrix.png')); plt.close()
    
    y_bin = label_binarize(all_labels, classes=range(len(classes)))
    plt.figure(figsize=(10, 8))
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i]); roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{classes[i]} (AUC = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC'); plt.legend(); plt.savefig(os.path.join(results_dir, 'roc_curve.png')); plt.close()
    
    # Report
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds, target_names=classes)
    with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"W-Precision: {precision:.4f}\nW-Recall: {recall:.4f}\nW-F1: {f1:.4f}\n\nDetailed:\n{report}")
    print(f"Evaluation complete. Results saved to {results_dir}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "data/AD-2dimg/OriginalDataset"
    results_dir = "results_2D"
    set_seed(42)
    
    train_loader, val_loader, test_loader, classes = get_dataloaders(data_path, batch_size=16)
    model = CNN_BiLSTM_Attention(num_classes=len(classes)).to(device)
    criterion = FocalLoss(); optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    print(f"Starting Training on {device}...")
    for epoch in range(1, 51):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = validate_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: T-Loss {t_loss:.4f} T-Acc {t_acc:.2f} | V-Loss {v_loss:.4f} V-Acc {v_acc:.2f}")
        early_stopping(v_loss, model, "weights/best_2d_model.pth")
        if early_stopping.early_stop: break
    
    model.load_state_dict(torch.load("weights/best_2d_model.pth"))
    evaluate_model(model, test_loader, device, classes, results_dir)

if __name__ == "__main__":
    main()

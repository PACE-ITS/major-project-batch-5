"""
Testing script for trained model
Evaluate on test set with clean output
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
import json
from scipy import ndimage
import nibabel as nib

class Config:
    DATA_NIFTI_DIR = "D:/ADPM3D/3MC/data/nifti"
    CHECKPOINT_DIR = "D:/ADPM3D/checkpoints"
    MAPPING_FILE = "D:/ADPM3D/3MC/data/subject_label_mapping.json"
    INPUT_SIZE = (64, 64, 64)
    BATCH_SIZE = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class NIfTIDataset(Dataset):
    def __init__(self, nifti_files, labels, input_size=(64, 64, 64)):
        self.nifti_files = nifti_files
        self.labels = labels
        self.input_size = input_size
    
    def __len__(self):
        return len(self.nifti_files)
    
    def __getitem__(self, idx):
        try:
            img = nib.load(self.nifti_files[idx]).get_fdata().astype(np.float32)
            zoom_factors = tuple(s / d for s, d in zip(self.input_size, img.shape))
            img = ndimage.zoom(img, zoom_factors, order=1)
            vmin, vmax = img.min(), img.max()
            if vmax > vmin:
                img = (img - vmin) / (vmax - vmin)
            img = torch.tensor(img).unsqueeze(0).float()
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return img, label
        except:
            return torch.zeros((1, *self.input_size)), torch.tensor(0)

class Enhanced3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.MaxPool3d(2),
            
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.MaxPool3d(2),
            
            nn.Conv3d(128, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((2, 2, 2)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def test_model():
    config = Config()
    
    print("\n" + "="*70)
    print("MODEL TESTING ON TEST SET")
    print("="*70 + "\n")
    
    # Load data
    nifti_dir = Path(config.DATA_NIFTI_DIR)
    with open(config.MAPPING_FILE) as f:
        subject_to_label = json.load(f)
    
    nifti_files = []
    labels = []
    for nifti_path in sorted(nifti_dir.glob("*.nii.gz")):
        filename = nifti_path.name
        parts = filename.split('_')
        if len(parts) >= 3:
            subject_code = f"{parts[0]}_{parts[1]}_{parts[2]}"
            if subject_code in subject_to_label:
                nifti_files.append(str(nifti_path))
                labels.append(subject_to_label[subject_code])
    
    labels = np.array(labels)
    
    # Split
    total = len(nifti_files)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    
    idx = np.arange(total)
    np.random.seed(42)
    np.random.shuffle(idx)
    
    test_files = [nifti_files[i] for i in idx[train_size+val_size:]]
    test_labels = labels[idx[train_size+val_size:]]
    
    test_ds = NIfTIDataset(test_files, test_labels, config.INPUT_SIZE)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE)
    
    # Load model
    model = Enhanced3DCNN().to(config.DEVICE)
    model_path = f"{config.CHECKPOINT_DIR}/best_model.pth"
    
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Train the model first using train_optimized.py")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    
    print(f"Test set size: {len(test_labels)} samples")
    print(f"  CN: {np.sum(test_labels==0)}, Disease: {np.sum(test_labels==1)}\n")
    print("TESTING:")
    print("-" * 70)
    
    correct = 0
    total = 0
    cn_correct = 0
    cn_total = 0
    disease_correct = 0
    disease_total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, batch_labels) in enumerate(test_loader):
            inputs = inputs.to(config.DEVICE)
            batch_labels = batch_labels.to(config.DEVICE)
            
            outputs = model(inputs).squeeze()
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            preds = (outputs > 0.5).long()
            
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)
            
            # Per-class accuracy
            cn_mask = batch_labels == 0
            cn_correct += ((preds == batch_labels) & cn_mask).sum().item()
            cn_total += cn_mask.sum().item()
            
            disease_mask = batch_labels == 1
            disease_correct += ((preds == batch_labels) & disease_mask).sum().item()
            disease_total += disease_mask.sum().item()
    
    test_acc = 100 * correct / total
    cn_acc = 100 * cn_correct / cn_total if cn_total > 0 else 0
    disease_acc = 100 * disease_correct / disease_total if disease_total > 0 else 0
    
    print("-" * 70)
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"Overall Accuracy:   {test_acc:.2f}%")
    print(f"CN Accuracy:        {cn_acc:.2f}% ({cn_correct}/{cn_total})")
    print(f"Disease Accuracy:   {disease_acc:.2f}% ({disease_correct}/{disease_total})")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_model()

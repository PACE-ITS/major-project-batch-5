import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import pydicom
import kagglehub
from sklearn.model_selection import train_test_split
from torchvision import transforms
import cv2
from tqdm import tqdm
import glob

def download_dataset():
    """Download the dataset from Kaggle using kagglehub."""
    print("Downloading dataset from Kaggle...")
    # The specific dataset mentioned by the user
    path = kagglehub.dataset_download("guillermopetcho/brain-oasis-alzheimers-detection-5-models")
    return path

def convert_dicom_to_nifti(dcm_dir, output_nii_path):
    """Convert a directory of DICOM files to a single NIfTI file."""
    if not os.path.exists(dcm_dir):
        return None
    
    dcm_files = sorted(glob.glob(os.path.join(dcm_dir, "*.dcm")))
    if not dcm_files:
        return None
    
    slices = [pydicom.dcmread(f) for f in dcm_files]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    
    # Extract pixel data
    pixel_data = np.stack([s.pixel_array for s in slices])
    pixel_data = pixel_data.transpose(1, 2, 0) # (H, W, Slices)
    
    # Create NIfTI image
    new_image = nib.Nifti1Image(pixel_data, affine=np.eye(4))
    nib.save(new_image, output_nii_path)
    return output_nii_path

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
        """Crop brain region by finding the bounding box of non-zero pixels."""
        mask = img > 10 # Threshold to find brain
        coords = np.argwhere(mask)
        if coords.size > 0:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            return img[y0:y1+1, x0:x1+1]
        return img

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_path = self.data_list[idx]
        label = self.labels[idx]
        
        # Load image
        if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
            img = nib.load(file_path).get_fdata()
            if len(img.shape) == 3:
                img = img[:, :, img.shape[2] // 2]
        else:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            # Fallback for corrupted images
            img = np.zeros((128, 128), dtype=np.uint8)
        
        # 1. Intensity Scaling [0, 255]
        img = ((img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8) * 255).astype(np.uint8)
        
        # 2. Advanced Preprocessing
        img = self.crop_roi(img)
        img = self.apply_clahe(img)
        
        if self.transform:
            img = self.transform(img)
        else:
            img = cv2.resize(img, (128, 128))
            img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
            
        return img, label

def get_dataloaders(base_path, batch_size=32, seed=42):
    """Prepare 70/15/15 stratified split dataloaders."""
    # This assumes dataset structure: base_path/Class_Name/*.nii or *.jpg
    classes = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
    all_paths = []
    all_labels = []
    
    # Prioritize provided path
    data_dir = base_path
    
    for idx, cls in enumerate(classes):
        cls_path = os.path.join(data_dir, cls)
        
        if os.path.exists(cls_path):
            # Find images (JPG, PNG, NII)
            files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.nii", "*.nii.gz"]:
                files.extend(glob.glob(os.path.join(cls_path, ext)))
            
            all_paths.extend(files)
            all_labels.extend([idx] * len(files))
    
    if not all_paths:
        raise ValueError(f"No data found in {base_path}. Check class folder names.")

    # Split: 70% Train, 30% temp
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels, test_size=0.30, stratify=all_labels, random_state=seed
    )
    
    # Split temp: 50% Val, 50% Test (15% each of total)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=seed
    )
    
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    train_ds = Alzheimer2DDataset(train_paths, train_labels, transform=train_transform)
    val_ds = Alzheimer2DDataset(val_paths, val_labels, transform=val_transform)
    test_ds = Alzheimer2DDataset(test_paths, test_labels, transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, classes

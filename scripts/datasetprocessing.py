"""
Data loader for 3D MRI volumes in NIfTI and DICOM formats.
Handles preprocessing, normalization, and augmentation.
"""

import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import warnings
import pydicom
from scipy import ndimage

warnings.filterwarnings('ignore')


class MRIVolumeDataset(Dataset):
    """
    Dataset for loading 3D MRI volumes from NIfTI files.
    
    Directory structure expected:
    DATA/raw/
    ├── CN/ADNI/
    │   ├── 002_S_0295/
    │   ├── 002_S_0413/
    │   └── ...
    ├── AD/ADNI/
    │   ├── 006_S_4153/
    │   ├── 006_S_4192/
    │   └── ...
    └── MCI/ADNI/
        ├── 002_S_0295/
        └── ...
    """
    
    def __init__(self, root_dir, class1='CN', class2='AD', img_size=(32, 32, 32),
                 target_size='original', normalize=True, augment=False, split='train'):
        """
        Args:
            root_dir: Root directory containing class folders
            class1: First class for binary classification (e.g., 'CN')
            class2: Second class for binary classification (e.g., 'AD')
            img_size: Target image size for cropping/padding (D, H, W)
            target_size: 'original' or (D, H, W) for resizing
            normalize: Whether to normalize volumes
            augment: Whether to apply data augmentation
            split: 'train', 'val', or 'test'
        """
        super(MRIVolumeDataset, self).__init__()
        
        self.root_dir = Path(root_dir)
        self.class1 = class1
        self.class2 = class2
        self.img_size = img_size
        self.target_size = target_size if target_size != 'original' else None
        self.normalize = normalize
        self.augment = augment and split == 'train'
        self.split = split
        
        # Find all subject directories
        self.file_paths = []
        self.labels = []
        
        # Search for class folders
        for class_name, label in [(class1, 0), (class2, 1)]:
            class_paths = list(self.root_dir.glob(f'{class_name}/ADNI/*/'))
            
            for subject_dir in class_paths:
                if not subject_dir.is_dir():
                    continue
                
                # Try to find NIfTI files first (priority)
                nii_files = list(subject_dir.glob('**/*.nii.gz')) + list(subject_dir.glob('**/*.nii'))
                
                if nii_files:
                    # Take the first nifti file found
                    self.file_paths.append(('nifti', str(nii_files[0])))
                    self.labels.append(label)
                else:
                    # Try to find DICOM files
                    dcm_files = list(subject_dir.glob('**/*.dcm'))
                    if dcm_files:
                        # Take the first dcm file (will load entire series)
                        self.file_paths.append(('dicom', str(dcm_files[0])))
                        self.labels.append(label)
        
        if len(self.file_paths) == 0:
            raise RuntimeError(f"No NIfTI or DICOM files found in {root_dir}")
        
        print(f"\n[{split.upper()}] Found {len(self.file_paths)} volumes")
        print(f"  - {self.class1}: {sum(1 for l in self.labels if l == 0)}")
        print(f"  - {self.class2}: {sum(1 for l in self.labels if l == 1)}")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """Load and return a 3D volume."""
        try:
            file_info = self.file_paths[idx]
            
            # Handle both new tuple format and old string format
            if isinstance(file_info, tuple):
                file_type, file_path = file_info
            else:
                file_type = 'nifti'
                file_path = file_info
            
            label = self.labels[idx]
            
            # Load volume based on file type
            if file_type == 'nifti':
                volume = self._load_nifti(file_path)
            elif file_type == 'dicom':
                volume = self._load_dicom_series(file_path)
            else:
                volume = self._load_nifti(file_path)  # Fallback to nifti
            
            # Resize to target size if needed
            if self.target_size is not None:
                volume = self._resize_volume(volume, self.target_size)
            else:
                volume = self._crop_or_pad(volume, self.img_size)
            
            # Normalize
            if self.normalize:
                volume = self._normalize(volume)
            
            # Augmentation
            if self.augment:
                volume = self._augment(volume)
            
            # Convert to tensor with correct dtype
            volume_tensor = torch.from_numpy(volume).unsqueeze(0).float()  # Add channel dimension
            
            return volume_tensor, torch.tensor(label, dtype=torch.float32)
        
        except Exception as e:
            print(f"Error loading file {self.file_paths[idx]}: {e}")
            # Return a zero tensor with correct shape as fallback
            return torch.zeros(1, *self.img_size, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)
    
    def _load_nifti(self, nii_path):
        """Load NIfTI file."""
        nii = nib.load(nii_path)
        volume = nii.get_fdata().astype(np.float32)
        
        # Make a copy to ensure C-contiguous array
        volume = np.ascontiguousarray(volume)
        
        # Handle multi-channel volumes (take first channel if needed)
        if len(volume.shape) == 4:
            volume = volume[:, :, :, 0]
        
        return volume
    
    def _load_dicom_series(self, dcm_path):
        """Load DICOM series from a directory."""
        dcm_dir = Path(dcm_path).parent
        
        # Find all DICOM files in the directory
        dcm_files = sorted(dcm_dir.glob('*.dcm'))
        
        if not dcm_files:
            raise RuntimeError(f"No DICOM files found in {dcm_dir}")
        
        # Load all DICOM files
        slices = []
        for dcm_file in dcm_files:
            try:
                ds = pydicom.dcmread(str(dcm_file))
                # Only add if it has pixel data
                if hasattr(ds, 'pixel_array'):
                    slices.append(ds)
            except Exception as e:
                # Skip problematic files
                continue
        
        if not slices:
            raise RuntimeError(f"Could not read any valid DICOM files from {dcm_dir}")
        
        # Sort slices by position if available
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except:
            # If sorting fails, use original order
            pass
        
        # Extract pixel array and stack
        pixel_arrays = []
        for slice_ds in slices:
            try:
                pixel_array = slice_ds.pixel_array.astype(np.float32)
                # Handle 2D arrays
                if len(pixel_array.shape) == 2:
                    pixel_arrays.append(pixel_array)
                else:
                    pixel_arrays.append(pixel_array)
            except Exception as e:
                continue
        
        if not pixel_arrays:
            raise RuntimeError(f"Could not extract pixel arrays from DICOM files")
        
        # Stack to create 3D volume
        volume = np.stack(pixel_arrays, axis=0).astype(np.float32)
        
        # Ensure 3D shape
        if len(volume.shape) < 3:
            volume = np.expand_dims(volume, axis=0)
        
        # Make contiguous
        volume = np.ascontiguousarray(volume)
        
        return volume
    
    def _crop_or_pad(self, volume, target_size):
        """Crop or pad volume to target size."""
        current_shape = volume.shape
        target_shape = target_size
        
        output = np.zeros(target_shape, dtype=np.float32)
        
        # Calculate slices for cropping/padding
        slices = []
        for i in range(3):
            if current_shape[i] >= target_shape[i]:
                # Crop from center
                start = (current_shape[i] - target_shape[i]) // 2
                end = start + target_shape[i]
                slices.append(slice(start, end))
            else:
                # Pad
                slices.append(slice(None))
        
        # Crop volume
        cropped = volume[tuple(slices)]
        
        # Pad if necessary
        if cropped.shape != target_shape:
            for i in range(3):
                if cropped.shape[i] < target_shape[i]:
                    pad_amount = target_shape[i] - cropped.shape[i]
                    pad_before = pad_amount // 2
                    pad_after = pad_amount - pad_before
                    
                    if i == 0:
                        cropped = np.pad(cropped, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant')
                    elif i == 1:
                        cropped = np.pad(cropped, ((0, 0), (pad_before, pad_after), (0, 0)), mode='constant')
                    else:
                        cropped = np.pad(cropped, ((0, 0), (0, 0), (pad_before, pad_after)), mode='constant')
        
        output[:] = cropped[:target_shape[0], :target_shape[1], :target_shape[2]]
        return output
    
    def _resize_volume(self, volume, target_size):
        """Resize volume to target size using interpolation."""
        from scipy import ndimage
        
        zoom_factors = np.array(target_size) / np.array(volume.shape)
        resized = ndimage.zoom(volume, zoom_factors, order=1)
        
        return resized.astype(np.float32)
    
    def _normalize(self, volume):
        """Normalize volume to [0, 1] range."""
        v_min = np.min(volume)
        v_max = np.max(volume)
        
        if v_max - v_min > 0:
            volume = (volume - v_min) / (v_max - v_min)
        else:
            volume = np.zeros_like(volume)
        
        return volume
    
    def _augment(self, volume):
        """Apply light data augmentation."""
        # Random rotation (small angles)
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-5, 5)
            volume = ndimage.rotate(volume, angle, axes=(0, 1), reshape=False, order=1)
        
        # Random flip with proper contiguity
        if np.random.rand() > 0.5:
            axis = np.random.randint(0, 3)
            volume = np.ascontiguousarray(np.flip(volume, axis=axis))
        
        # Random intensity scaling
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            volume = np.clip(volume * scale, 0, 1)
        
        # Random noise
        if np.random.rand() > 0.7:
            noise = np.random.normal(0, 0.01, volume.shape)
            volume = np.clip(volume + noise, 0, 1)
        
        return np.ascontiguousarray(volume)


def create_dataloaders(data_root, class1='CN', class2='AD', batch_size=4,
                       num_workers=0, img_size=(32, 32, 32), split_ratio=0.8):
    """
    Create train/val dataloaders.
    
    Args:
        data_root: Root directory of data
        class1: First class for binary classification
        class2: Second class for binary classification
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        img_size: Target image size
        split_ratio: Train/val split ratio
    
    Returns:
        train_loader, val_loader, num_train, num_val
    """
    
    # Create full dataset
    full_dataset = MRIVolumeDataset(
        data_root, class1=class1, class2=class2,
        img_size=img_size, normalize=True, augment=True, split='train'
    )
    
    # Split dataset
    num_total = len(full_dataset)
    num_train = int(num_total * split_ratio)
    num_val = num_total - num_train
    
    from torch.utils.data import random_split
    train_dataset, val_dataset = random_split(full_dataset, [num_train, num_val])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    
    return train_loader, val_loader, num_train, num_val


if __name__ == "__main__":
    # Test dataset
    data_root = "DATA/raw"
    
    dataset = MRIVolumeDataset(
        data_root,
        class1='CN',
        class2='AD',
        img_size=(32, 32, 32),
        normalize=True,
        split='train'
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        volume, label = dataset[0]
        print(f"Volume shape: {volume.shape}")
        print(f"Label: {label.item()}")
        print(f"Volume dtype: {volume.dtype}")
        print(f"Volume range: [{volume.min():.4f}, {volume.max():.4f}]")
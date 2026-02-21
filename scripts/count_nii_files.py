#!/usr/bin/env python3
"""Count NII files by disease class (CN, MCI, AD)"""

import os
from pathlib import Path
from collections import defaultdict

def count_nii_files(root_dir):
    """Count .nii.gz files for each class"""
    
    root = Path(root_dir)
    
    if not root.exists():
        print(f"Error: Directory does not exist: {root_dir}")
        return
    
    counts = defaultdict(int)
    
    # Get top-level class directories
    for class_dir in root.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name.upper()
            # Count nii.gz files in this class directory
            nii_count = sum(1 for _ in class_dir.rglob("*.nii.gz"))
            counts[class_name] = nii_count
    
    # Print results
    print(f"\n{'='*50}")
    print(f"NII File Count by Class")
    print(f"{'='*50}")
    
    if not counts:
        print("No .nii.gz files found!")
        return
    
    total = 0
    for class_name in sorted(counts.keys()):
        count = counts[class_name]
        total += count
        print(f"{class_name:10s}: {count:6d} files")
    
    print(f"{'-'*50}")
    print(f"{'TOTAL':10s}: {total:6d} files")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        root = sys.argv[1]
    else:
        root = "data/nifti_converted"
    
    count_nii_files(root)

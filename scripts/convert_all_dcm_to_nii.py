#!/usr/bin/env python3
"""Recursively convert DICOM directories to NIfTI.

This script finds directories containing DICOM files (searching deep
into nested folders). For each leaf directory that contains DICOM files
it will attempt to convert the directory to NIfTI using `dicom2nifti`.
If `dicom2nifti` is not available it will try to call the external
`dcm2niix` binary (if installed).

Usage:
  python scripts/convert_all_dcm_to_nii.py --input <input_root> --output <output_root>

Dependencies:
  pip install dicom2nifti pydicom
  or install `dcm2niix` and keep it on PATH.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from typing import List


def parse_args():
    p = argparse.ArgumentParser(description="Recursively convert DICOM to NIfTI")
    p.add_argument("--input", "-i", required=True, help="Input root directory to search for DICOM files")
    p.add_argument("--output", "-o", required=True, help="Output root directory to write NIfTI files")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output folders")
    p.add_argument("--dry-run", action="store_true", help="Show directories that would be converted and exit")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    return p.parse_args()


def is_dicom_file(path: str) -> bool:
    try:
        import pydicom

        pydicom.dcmread(path, stop_before_pixels=True)
        return True
    except Exception:
        return False


def dir_contains_dcm(path: str) -> bool:
    for f in os.listdir(path):
        fp = os.path.join(path, f)
        if os.path.isfile(fp):
            if f.lower().endswith(".dcm"):
                return True
            # some DICOM files have no extension
            if is_dicom_file(fp):
                return True
    return False


def any_subdir_contains_dcm(path: str) -> bool:
    for root, dirs, files in os.walk(path):
        if root == path:
            # skip root itself
            continue
        if dir_contains_dcm(root):
            return True
    return False


def convert_with_dicom2nifti(src: str, out: str) -> None:
    try:
        import dicom2nifti
    except ImportError:
        raise ImportError("dicom2nifti not installed. Run: pip install dicom2nifti")

    # dicom2nifti.convert_directory will convert all series in the folder
    dicom2nifti.convert_directory(src, out)


def convert_with_dcm2niix(src: str, out: str) -> None:
    # dcm2niix -z n -o <out> <src>
    cmd = ["dcm2niix", "-z", "n", "-o", out, src]
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    input_root = os.path.abspath(args.input)
    output_root = os.path.abspath(args.output)

    if not os.path.isdir(input_root):
        logging.error("Input root does not exist: %s", input_root)
        sys.exit(1)

    to_convert: List[str] = []

    # Find leaf directories that contain DICOM files (no child dir that also contains DICOM).
    for root, dirs, files in os.walk(input_root):
        if dir_contains_dcm(root):
            # if any child directory contains dicom files, skip converting this one now
            has_child_dcm = False
            for d in dirs:
                dpath = os.path.join(root, d)
                if dir_contains_dcm(dpath):
                    has_child_dcm = True
                    break
            if not has_child_dcm:
                to_convert.append(root)

    if not to_convert:
        logging.info("No DICOM-containing directories found under %s", input_root)
        return

    logging.info("Found %d DICOM directories to convert", len(to_convert))

    for src_dir in to_convert:
        rel = os.path.relpath(src_dir, input_root)
        out_dir = os.path.join(output_root, rel)

        if os.path.exists(out_dir) and not args.overwrite:
            logging.info("Skipping (exists): %s -> %s", src_dir, out_dir)
            continue

        logging.info("Converting: %s -> %s", src_dir, out_dir)
        if args.dry_run:
            continue

        os.makedirs(out_dir, exist_ok=True)

        # Try dicom2nifti first (Python implementation)
        try:
            convert_with_dicom2nifti(src_dir, out_dir)
            logging.info("Converted with dicom2nifti: %s", src_dir)
            continue
        except Exception as e:
            logging.debug("dicom2nifti failed for %s: %s", src_dir, str(e))

        # Fallback to dcm2niix if installed
        try:
            convert_with_dcm2niix(src_dir, out_dir)
            logging.info("Converted with dcm2niix: %s", src_dir)
            continue
        except Exception as e:
            logging.error("Conversion failed for %s: %s", src_dir, str(e))


if __name__ == "__main__":
    main()

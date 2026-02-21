#!/usr/bin/env python3
"""
SIMPLIFIED AUTOMATED TRAINING RUNNER
Direct execution of all training and testing with improved parameters
"""

import os
import subprocess
import time
from datetime import datetime

def run_cmd(cmd, description, cwd="."):
    """Run command and return success status"""
    print(f"\n{'='*80}")
    print(f"▶ {description}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=1800
        )
        
        print(result.stdout)
        if result.stderr:
            print("ERRORS:", result.stderr[:1000])
        
        success = result.returncode == 0
        print(f"\n{'✓ SUCCESS' if success else '✗ FAILED'}")
        return success, result.stdout
        
    except subprocess.TimeoutExpired:
        print(f"⏱ TIMEOUT (30 minutes exceeded)")
        return False, ""
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False, ""

def main():
    """Main execution"""
    print(f"\n{'#'*80}")
    print(f"# AUTOMATED TRAINING PIPELINE - AUTO MODE")
    print(f"# Target: 92-96% Accuracy")
    print(f"# Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}\n")
    
    results = {}
    
    # ===== PHASE 1: BINARY CLASSIFICATION TASKS =====
    print(f"\n{'*'*80}")
    print(f"PHASE 1: Binary Classification Tasks (from 3MC)")
    print(f"{'*'*80}")
    
    binary_tasks = [
        ("train_cn_vs_ad.py", "CN vs Alzheimer's Disease"),
        ("train_cn_vs_mci.py", "CN vs Mild Cognitive Impairment"),
        ("train_cn_vs_mci_ad.py", "CN vs (MCI+AD) Multi-class"),
    ]
    
    for train_file, description in binary_tasks:
        desc = f"Training: {description}"
        success, output = run_cmd(f"python {train_file}", desc, cwd="3MC")
        results[train_file] = {"success": success, "output": output}
        time.sleep(2)  # Brief pause between trainings
    
    # ===== PHASE 2: MULTI-TASK LEARNING =====
    print(f"\n{'*'*80}")
    print(f"PHASE 2: Multi-task Learning (3D HCCT)")
    print(f"{'*'*80}")
    
    desc = "Training: 3D HCCT with CNN + Transformer (Multi-task)"
    success, output = run_cmd("python train_3cba.py", desc, cwd=".")
    results["train_3cba.py"] = {"success": success, "output": output}
    time.sleep(2)
    
    # ===== PHASE 3: HYPERPARAMETER TUNING =====
    print(f"\n{'*'*80}")
    print(f"PHASE 3: Advanced Hyperparameter Tuning")
    print(f"{'*'*80}")
    
    desc = "Training: Advanced3DCNN with Hyperparameter Optimization"
    success, output = run_cmd("python hyperparameter_tuner.py", desc, cwd=".")
    results["hyperparameter_tuner.py"] = {"success": success, "output": output}
    time.sleep(2)
    
    # ===== PHASE 4: TESTING =====
    print(f"\n{'*'*80}")
    print(f"PHASE 4: Comprehensive Testing")
    print(f"{'*'*80}")
    
    # Test binary classification models
    test_tasks = [
        ("test_cn_vs_ad.py", "Testing CN vs AD"),
        ("test_cn_vs_mci.py", "Testing CN vs MCI"),
        ("test_cn_vs_mci_ad.py", "Testing CN vs (MCI+AD)"),
        ("test_all_models.py", "Testing All Models (3D HCCT)"),
    ]
    
    for test_file, description in test_tasks:
        test_cwd = "3MC" if "test_" in test_file and test_file != "test_all_models.py" else "3MC"
        desc = f"Testing: {description}"
        success, output = run_cmd(f"python {test_file}", desc, cwd=test_cwd)
        results[f"TEST: {test_file}"] = {"success": success, "output": output}
        time.sleep(2)
    
    # ===== FINAL SUMMARY =====
    print(f"\n\n{'#'*80}")
    print(f"# FINAL SUMMARY")
    print(f"{'#'*80}\n")
    
    print(f"{'Task':<40} {'Status':<15} {'Time'}")
    print(f"{'-'*80}")
    
    successful = 0
    failed = 0
    
    for task, result in results.items():
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        if result['success']:
            successful += 1
        else:
            failed += 1
        print(f"{task:<40} {status:<15} {datetime.now().strftime('%H:%M:%S')}")
    
    print(f"\n{'='*80}")
    print(f"Total Tasks: {len(results)} | Successful: {successful} | Failed: {failed}")
    print(f"Success Rate: {100*successful/len(results):.1f}%")
    print(f"{'='*80}\n")
    
    if successful >= len(results) * 0.8:
        print(f"✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    else:
        print(f"⊙ Some tasks failed. Review logs above.")
    
    print(f"Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

if __name__ == "__main__":
    main()

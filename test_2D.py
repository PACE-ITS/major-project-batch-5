import torch
import os
from model_2D import CNN_BiLSTM_Attention
from data_utils_2D import get_dataloaders
from evaluate_2D import evaluate_model
import argparse

def test_outcome():
    device = torch.device("cpu")
    print("--- Final Model Outcome Generation ---")
    
    # Paths
    weights_path = "weights/best_model_optimized.pth"
    data_path = "data/AD-2dimg/OriginalDataset"
    results_dir = "results_2D/outcome"
    os.makedirs(results_dir, exist_ok=True)
    
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        return

    # 1. Load Data
    print(f"Loading test data from {data_path}...")
    _, _, test_loader, classes = get_dataloaders(data_path, batch_size=16)
    
    # 2. Load Model
    print(f"Initializing EfficientNet-B0 + BiLSTM + Attention Model...")
    model = CNN_BiLSTM_Attention(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    # 3. Generate Outcome
    print("\nStarting Inference on Test Set (960 samples)...")
    evaluate_model(model, test_loader, device, classes, results_dir)
    
    # 4. Read and Display Outcome
    report_path = os.path.join(results_dir, "classification_report.txt")
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            print("\n" + "="*40)
            print("         FINAL TEST OUTCOME")
            print("="*40)
            print(f.read())
            print("="*40)
    
    print(f"\nOutcome generation complete. Detailed plots saved in {results_dir}")

if __name__ == "__main__":
    test_outcome()

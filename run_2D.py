import os
import torch
import torch.nn as nn
import torch.optim as optim
from data_utils_2D import download_dataset, get_dataloaders
from model_2D import CNN_BiLSTM_Attention
from train_2D import set_seed, EarlyStopping, train_epoch, validate_epoch, FocalLoss
from evaluate_2D import evaluate_model, plot_learning_curves
import argparse

def main():
    parser = argparse.ArgumentParser(description="2D Alzheimer's MRI Classification Pipeline")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--results_dir", type=str, default="results_2D", help="Directory to save results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs("weights", exist_ok=True)

    # Setup
    set_seed(42)
    device = torch.device("cpu") # Explicitly CPU as requested

    # Data Selection
    base_data_path = "data/AD-2dimg/OriginalDataset"
    
    if not os.path.exists(base_data_path):
        print(f"User dataset not found at {base_data_path}. Attempting download or synthetic fallback.")
        try:
            dataset_path = download_dataset()
            base_data_path = dataset_path
            if os.path.exists(os.path.join(dataset_path, "Data")):
                base_data_path = os.path.join(dataset_path, "Data")
            elif os.path.exists(os.path.join(dataset_path, "OriginalDataset")):
                base_data_path = os.path.join(dataset_path, "OriginalDataset")
        except Exception as e:
            print(f"Download failed: {e}. Falling back to synthetic.")
            base_data_path = "data/synthetic_data"
    
    print(f"Data loading from: {base_data_path}")
    train_loader, val_loader, test_loader, classes = get_dataloaders(base_data_path, batch_size=args.batch_size)
    
    # Model
    model = CNN_BiLSTM_Attention(num_classes=len(classes)).to(device)
    
    # Training Config
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    model_save_path = "weights/best_model_optimized.pth"
    
    # Stage 1: Freeze Backbone for 5 epochs
    print("Fine-tuning: Freezing backbone for initial stability...")
    for param in model.feature_extractor.parameters():
        param.requires_grad = False
    
    print(f"Starting optimized training for {args.epochs} epochs on CPU...")
    for epoch in range(1, args.epochs + 1):
        # Stage 2: Unfreeze Backbone after 5 epochs
        if epoch == 6:
            print("Fine-tuning: Unfreezing backbone for full optimization...")
            for param in model.feature_extractor.parameters():
                param.requires_grad = True
            # Reduce LR for fine-tuning
            for g in optimizer.param_groups:
                g['lr'] = args.lr * 0.1

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch}/{args.epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_loss)
        early_stopping(val_loss, model, model_save_path)
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
            
    # Evaluation
    print("Training finished. Loading best model for evaluation...")
    model.load_state_dict(torch.load(model_save_path))
    
    plot_learning_curves(train_losses, val_losses, train_accs, val_accs, args.results_dir)
    evaluate_model(model, test_loader, device, classes, args.results_dir)
    
    print(f"Pipeline complete. Results saved in {args.results_dir}")

if __name__ == "__main__":
    main()

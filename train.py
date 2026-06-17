import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns

# Import our custom modules
from dataset import get_dataloaders
from model import get_model

class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving.
    """
    def __init__(self, patience=5, delta=1e-4, checkpoint_path='best_model.pth'):
        self.patience = patience
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        print(f"Validation loss decreased. Saving best model checkpoint to '{self.checkpoint_path}'...")
        torch.save(model.state_dict(), self.checkpoint_path)


def save_training_summary(summary, save_path='training_summary.json'):
    """Writes a compact training record that can be inspected after the run."""
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Training summary saved as '{save_path}'.")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images).squeeze(1) # Squeeze output logit to match label size [batch_size]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record statistics
        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on validation data.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    val_loss = running_loss / total
    val_acc = correct / total
    
    # Calculate Precision, Recall, and F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    return val_loss, val_acc, precision, recall, f1


def plot_metrics(history, save_dir='.'):
    """
    Plots training vs validation curves.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Training Acc')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Validation Acc')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()
    print("Training curves plot saved as 'training_curves.png'.")


def evaluate_final(model, loader, device, save_dir='.'):
    """
    Performs final evaluation on the best checkpointed model.
    Generates confusion matrix and prints classification report.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images).squeeze(1)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    # Print Classification Report
    print("\n" + "="*50)
    print("FINAL EVALUATION REPORT (Best Model)")
    print("="*50)
    print(classification_report(
        all_labels,
        all_preds,
        labels=[0, 1],
        target_names=['REAL (0)', 'FAKE (1)'],
        zero_division=0
    ))
    
    # Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['REAL', 'FAKE'], 
                yticklabels=['REAL', 'FAKE'])
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix - Fake Image Identification')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    print("Confusion matrix saved as 'confusion_matrix.png'.")


def main():
    parser = argparse.ArgumentParser(description="AI Fake Image Identification - Model Training")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs. Default is 10.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size. Default is 32.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate. Default is 1e-4.")
    parser.add_argument('--fine_tune', type=str, default='true', help="Set to 'false' to freeze backbone layers. Default 'true'.")
    parser.add_argument('--csv_path', nargs='+', default=['dataset_local.csv'], help="Path(s) to local dataset CSV file(s). Default is dataset_local.csv.")
    parser.add_argument('--initial_weights', type=str, default=None, help="Optional path to model weights to load before training.")
    parser.add_argument('--checkpoint_path', type=str, default='best_model.pth', help="Path where the best checkpoint will be saved.")
    parser.add_argument('--summary_path', type=str, default='training_summary.json', help="Path where the training summary JSON will be saved.")
    args = parser.parse_args()
    args.csv_path = [path for path in args.csv_path if path.lower() != 'and']

    fine_tune = args.fine_tune.lower() == 'true'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using execution device: {device}")

    # Load loaders
    try:
        train_loader, val_loader = get_dataloaders(csv_path=args.csv_path, batch_size=args.batch_size)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please download dataset samples first by running: python download_dataset.py --limit 200")
        return

    # Load model
    print("Initializing model...")
    model = get_model(fine_tune=fine_tune, device=device)
    if args.initial_weights:
        if not os.path.exists(args.initial_weights):
            raise FileNotFoundError(f"Initial weights not found: {args.initial_weights}")
        print(f"Loading initial weights from '{args.initial_weights}'...")
        model.load_state_dict(torch.load(args.initial_weights, map_location=device, weights_only=True))
    
    # Setup loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    
    # Callbacks
    checkpoint_path = args.checkpoint_path
    early_stopping = EarlyStopping(patience=5, checkpoint_path=checkpoint_path)
    
    # Metrics history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }
    best_epoch = None
    best_metrics = None
    
    print("\nStarting Training Loop...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")
        
        # Validate
        val_loss, val_acc, precision, recall, f1 = validate(model, val_loader, criterion, device)
        print(f"Val Loss:   {val_loss:.4f} | Val Accuracy:   {val_acc:.4f}")
        print(f"Val Precision: {precision:.4f} | Val Recall: {recall:.4f} | Val F1-Score: {f1:.4f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        history['val_f1'].append(f1)
        
        # Checkpoint and Early Stopping Check
        previous_best = early_stopping.best_loss
        early_stopping(val_loss, model)
        if previous_best is None or val_loss < previous_best - early_stopping.delta:
            best_epoch = epoch
            best_metrics = {
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_precision': precision,
                'val_recall': recall,
                'val_f1': f1
            }
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}. Stopping training.")
            break

    # Save training curves
    plot_metrics(history)
    training_summary = {
        'csv_path': args.csv_path,
        'requested_epochs': args.epochs,
        'completed_epochs': len(history['train_loss']),
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'fine_tune': fine_tune,
        'device': str(device),
        'initial_weights': args.initial_weights,
        'checkpoint_path': checkpoint_path,
        'best_epoch': best_epoch,
        'best_metrics': best_metrics,
        'history': history
    }
    save_training_summary(training_summary, save_path=args.summary_path)
    
    # Final evaluation using best model
    if os.path.exists(checkpoint_path):
        print("\nLoading best weights checkpoint for final evaluation...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    else:
        print("\nNo checkpoint found. Evaluating final epoch model weights...")
        
    evaluate_final(model, val_loader, device)
    print("\nTraining execution completed successfully!")

if __name__ == '__main__':
    main()

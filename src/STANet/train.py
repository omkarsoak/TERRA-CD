import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
from metrics import calculate_metrics, calculate_effective_weights

def train_model(model, train_loader, val_loader, num_epochs=50, num_classes=3,
                device='cuda', learning_rate=1e-4, weight_decay=0.01,
                checkpoint_path='best_stanet_model.pt',weighting_method='square_balanced'):
    """
    Training function for STANet model with comprehensive metrics tracking.

    Args:
        model: STANet model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        num_classes: Number of classes for change detection
        device: Device to run training on
        learning_rate: Initial learning rate
        weight_decay: Weight decay for optimizer
        checkpoint_path: Path to save best model checkpoint
    """
    # Initialize starting values
    start_epoch = 0
    best_val_loss = float('inf')

    # Initialize metrics history
    history = {
        'train': {
            'loss': [], 'accuracy': [], 'precision': [],
            'recall': [], 'f1_score': [], 'miou': [], 'kappa': []
        },
        'val': {
            'loss': [], 'accuracy': [], 'precision': [],
            'recall': [], 'f1_score': [], 'miou': [], 'kappa': []
        }
    }

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            history = checkpoint['history']
            print(f"Resuming from epoch {start_epoch} with best val loss: {best_val_loss:.4f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch")

    # Setup optimizer and losses
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    class_weights, _ = calculate_effective_weights(train_loader, device, num_classes=num_classes, method=weighting_method)
    print(class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Move model to device
    model = model.to(device)

    def process_epoch(phase, data_loader):
        """Process one epoch of training or validation"""
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_metrics = {
            'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0,
            'recall': 0.0, 'f1_score': 0.0, 'miou': 0.0, 'kappa': 0.0
        }
        samples_count = 0

        # Use tqdm for progress bar
        pbar = tqdm(data_loader, desc=f'{phase.capitalize()} Epoch')

        with torch.set_grad_enabled(phase == 'train'):
            for inputs1, inputs2, labels in pbar:
                # Move data to device
                inputs1 = inputs1.to(device)
                inputs2 = inputs2.to(device)
                labels = labels.to(device)
                batch_size = inputs1.size(0)

                # Zero gradients for training
                if phase == 'train':
                    optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs1, inputs2)
                loss = criterion(outputs, labels)

                # Backward pass for training
                if phase == 'train':
                    loss.backward()
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                # Calculate metrics
                batch_metrics = calculate_metrics(outputs, labels, num_classes=num_classes)
                batch_metrics['loss'] = loss.item()

                # Update running metrics
                for key in running_metrics:
                    running_metrics[key] += batch_metrics[key] * batch_size
                samples_count += batch_size

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{batch_metrics['loss']:.4f}",
                    'miou': f"{batch_metrics['miou']:.4f}"
                })

        # Calculate epoch metrics
        epoch_metrics = {key: value / samples_count for key, value in running_metrics.items()}

        # Store metrics in history
        for key in history[phase]:
            history[phase][key].append(epoch_metrics[key])

        return epoch_metrics

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}:')

        # Training phase
        train_metrics = process_epoch('train', train_loader)

        # Validation phase
        val_metrics = process_epoch('val', val_loader)

        # Print metrics
        def print_metrics(phase, metrics):
            print(f'\n{phase.capitalize()} Metrics:')
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-score: {metrics['f1_score']:.4f}")
            print(f"  mIoU: {metrics['miou']:.4f}")
            print(f"  Kappa: {metrics['kappa']:.4f}")

        print_metrics('train', train_metrics)
        print_metrics('val', val_metrics)

        # Update learning rate scheduler
        scheduler.step(val_metrics['loss'])

        # Save checkpoint if it's the best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'metrics': val_metrics,
                'history': history
            }
            torch.save(checkpoint, checkpoint_path)
            print(f'\nSaved new best model with validation loss: {val_metrics["loss"]:.4f}')

    return model, history

def save_training_files(history, checkpoint_path, history_filename, bestepoch_filename):
    """Save training history and best epoch info to separate JSON files"""

    def convert_to_serializable(value):
        """Recursively convert numpy/torch types to basic Python types"""
        if isinstance(value, (np.ndarray, torch.Tensor)):
            return value.tolist()
        elif isinstance(value, dict):
            return {k: convert_to_serializable(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [convert_to_serializable(item) for item in value]
        return value

    history_data = {
        phase: {
            metric: convert_to_serializable(values)
            for metric, values in metrics.items()
        }
        for phase, metrics in history.items()
    }

    with open(history_filename, 'w') as f:
        json.dump(history_data, f, indent=4)

    # Load checkpoint without weights_only flag
    checkpoint = torch.load(checkpoint_path)
    # print("\nCheckpoint contents:")
    # for key in checkpoint.keys():
    #     print(f"- {key}")

    # Convert metrics to basic Python types
    epoch_data = {
        'best_epoch': checkpoint['epoch'],
        'best_val_loss': checkpoint['best_val_loss'],
        'val_metrics': convert_to_serializable(checkpoint['metrics'])
    }

    with open(bestepoch_filename, 'w') as f:
        json.dump(epoch_data, f, indent=4)

    print(f"\nSaved training history to: {history_filename}")
    print(f"Saved best epoch info to: {bestepoch_filename}")
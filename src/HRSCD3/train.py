import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from metrics import calculate_metrics, calculate_class_weights

def create_semantic_change_mask(binary_pred, lcm_pred_2019, lcm_pred_2024):
    """Convert binary change + LCM predictions to 13-class semantic change mask.

    Optimized version using vectorized operations and pre-computed lookup tables.

    Args:
        binary_pred: Binary change prediction tensor (B, 1, H, W)
        lcm_pred_2019: Land cover prediction tensor for 2019 (B, C, H, W)
        lcm_pred_2024: Land cover prediction tensor for 2024 (B, C, H, W)

    Returns:
        Semantic change mask tensor (B, H, W) with values 0-12
    """
    device = binary_pred.device
    batch_size = binary_pred.shape[0]
    height = binary_pred.shape[2]
    width = binary_pred.shape[3]

    # Pre-compute land cover predictions - do this once
    lcm_2019 = torch.argmax(lcm_pred_2019, dim=1)  # (B, H, W)
    lcm_2024 = torch.argmax(lcm_pred_2024, dim=1)  # (B, H, W)

    # Create the change mask - use threshold without squeeze/unsqueeze
    change_mask = binary_pred[:, 0] > 0.5  # (B, H, W)

    # Initialize output tensor
    semantic_mask = torch.zeros((batch_size, height, width), device=device, dtype=torch.long)

    # Create transition matrix lookup table - speeds up class mapping
    # Format: from_class * num_cd_classes + to_class = semantic_class
    num_cd_classes = 4  # Water, Building, Sparse, Dense
    transitions = torch.full((num_cd_classes * num_cd_classes,), 0, device=device)

    # Populate transition matrix - all transitions not listed default to 0 (no change)
    transition_map = {
        (0, 1): 1,   # Water → Building
        (0, 2): 2,   # Water → Sparse
        (0, 3): 3,   # Water → Dense
        (1, 0): 4,   # Building → Water
        (1, 2): 5,   # Building → Sparse
        (1, 3): 6,   # Building → Dense
        (2, 0): 7,   # Sparse → Water
        (2, 1): 8,   # Sparse → Building
        (2, 3): 9,   # Sparse → Dense
        (3, 0): 10,  # Dense → Water
        (3, 1): 11,  # Dense → Building
        (3, 2): 12,  # Dense → Sparse
    }

    for (from_idx, to_idx), semantic_idx in transition_map.items():
        transitions[from_idx * num_cd_classes + to_idx] = semantic_idx

    # Vectorized computation of semantic classes
    # Only compute for changed pixels to save memory
    changed_pixels = change_mask.nonzero(as_tuple=True)
    if len(changed_pixels[0]) > 0:
        from_classes = lcm_2019[changed_pixels]  # (N,)
        to_classes = lcm_2024[changed_pixels]    # (N,)

        # Compute transition indices
        transition_indices = from_classes * num_cd_classes + to_classes  # (N,)

        # Look up semantic classes from transition matrix
        semantic_classes = transitions[transition_indices]  # (N,)

        # Assign semantic classes to output mask
        semantic_mask[changed_pixels] = semantic_classes

    return semantic_mask

def train_epoch(model, train_loader, cd_criterion, lcm_criterion,
                cd_optimizer, lcm_optimizer, device, num_cd_classes=13):
    """Train for one epoch"""
    model.train()
    total_cd_loss = 0
    total_lcm_loss = 0
    all_predictions = []
    all_targets = []


    for img_2019, img_2024, sem_2019, sem_2024, cd_mask in tqdm(train_loader):
        # Move data to device
        img_2019 = img_2019.to(device)
        img_2024 = img_2024.to(device)
        sem_2019 = sem_2019.to(device)
        sem_2024 = sem_2024.to(device)
        cd_mask = cd_mask.to(device)

        # Create binary change mask for CD network
        binary_mask = (cd_mask > 0).float().unsqueeze(1)

        # Train CD network
        cd_optimizer.zero_grad()
        cd_logits = model.cd_model(torch.cat([img_2019, img_2024], dim=1))
        cd_pred = torch.sigmoid(cd_logits)  # Apply sigmoid for binary prediction
        cd_loss = cd_criterion(cd_pred, binary_mask)
        cd_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.cd_model.parameters(), max_norm=1.0)
        cd_optimizer.step()

        # Train LCM network
        lcm_optimizer.zero_grad()
        lcm_pred_2019 = model.lcm_model(img_2019)
        lcm_pred_2024 = model.lcm_model(img_2024)
        lcm_loss = (lcm_criterion(lcm_pred_2019, sem_2019) +
                   lcm_criterion(lcm_pred_2024, sem_2024)) / 2
        lcm_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.lcm_model.parameters(), max_norm=1.0)
        lcm_optimizer.step()

        # Get semantic predictions for metrics
        with torch.no_grad():
            semantic_pred = create_semantic_change_mask(cd_pred, lcm_pred_2019, lcm_pred_2024)
            all_predictions.append(semantic_pred.cpu())
            all_targets.append(cd_mask.cpu())

        total_cd_loss += cd_loss.item()
        total_lcm_loss += lcm_loss.item()

    # Calculate metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = calculate_metrics(all_predictions, all_targets, num_cd_classes=num_cd_classes)

    # Average losses
    metrics['cd_loss'] = total_cd_loss / len(train_loader)
    metrics['lcm_loss'] = total_lcm_loss / len(train_loader)

    return metrics

def validate(model, val_loader, cd_criterion, lcm_criterion, device, num_cd_classes):
    """Validate model"""
    model.eval()
    total_cd_loss = 0
    total_lcm_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for img_2019, img_2024, sem_2019, sem_2024, cd_mask in val_loader:
            img_2019 = img_2019.to(device)
            img_2024 = img_2024.to(device)
            sem_2019 = sem_2019.to(device)
            sem_2024 = sem_2024.to(device)
            cd_mask = cd_mask.to(device)

            binary_mask = (cd_mask > 0).float().unsqueeze(1)

            # CD predictions
            cd_pred = model.cd_model(torch.cat([img_2019, img_2024], dim=1))
            cd_loss = cd_criterion(cd_pred, binary_mask)

            # LCM predictions
            lcm_pred_2019 = model.lcm_model(img_2019)
            lcm_pred_2024 = model.lcm_model(img_2024)
            lcm_loss = (lcm_criterion(lcm_pred_2019, sem_2019) +
                       lcm_criterion(lcm_pred_2024, sem_2024)) / 2

            # Get semantic predictions
            semantic_pred = create_semantic_change_mask(cd_pred, lcm_pred_2019, lcm_pred_2024)
            all_predictions.append(semantic_pred.cpu())
            all_targets.append(cd_mask.cpu())

            total_cd_loss += cd_loss.item()
            total_lcm_loss += lcm_loss.item()

    # Calculate metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = calculate_metrics(all_predictions, all_targets, num_cd_classes=num_cd_classes)

    # Average losses
    metrics['cd_loss'] = total_cd_loss / len(val_loader)
    metrics['lcm_loss'] = total_lcm_loss / len(val_loader)

    return metrics


def print_metrics_summary(train_metrics, val_metrics, train_total_loss, val_total_loss):
    """Print a summary of the training/validation metrics"""
    print("\nTraining Metrics:")
    print(f"  Total Loss: {train_total_loss:.4f}")
    print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"  Precision: {train_metrics['precision']:.4f}")
    print(f"  Recall: {train_metrics['recall']:.4f}")
    print(f"  F1-Score: {train_metrics['f1_score']:.4f}")
    print(f"  mIoU: {train_metrics['miou']:.4f}")
    print(f"  Kappa: {train_metrics['kappa']:.4f}")

    print("\nValidation Metrics:")
    print(f"  Total Loss: {val_total_loss:.4f}")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall: {val_metrics['recall']:.4f}")
    print(f"  F1-Score: {val_metrics['f1_score']:.4f}")
    print(f"  mIoU: {val_metrics['miou']:.4f}")
    print(f"  Kappa: {val_metrics['kappa']:.4f}")

def train_strategy3(model, train_loader, val_loader, num_epochs=50, device='cuda',
                   checkpoint_path=None, num_cd_classes=13, num_semantic_classes=4):
    """Train Strategy 3 model with checkpointing based on validation loss and overall metrics"""

    # Loss functions
    cd_criterion = nn.BCEWithLogitsLoss()

    class_weights = calculate_class_weights(train_loader, num_cd_classes=num_semantic_classes, device='cuda')  #square_balanced
    print(class_weights)
    lcm_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Optimizers
    cd_optimizer = optim.AdamW(model.cd_model.parameters(), lr=1e-4, weight_decay=0.01)
    lcm_optimizer = optim.AdamW(model.lcm_model.parameters(), lr=1e-4, weight_decay=0.01)

    # Learning rate schedulers
    cd_scheduler = ReduceLROnPlateau(cd_optimizer, mode='min', factor=0.5, patience=5)
    lcm_scheduler = ReduceLROnPlateau(lcm_optimizer, mode='min', factor=0.5, patience=5)

    # Load checkpoint if exists
    start_epoch = 0
    best_total_loss = float('inf')
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        #model.load_state_dict(checkpoint['model_state_dict'])  # Load model weights
        cd_optimizer.load_state_dict(checkpoint['cd_optimizer'])  # Load optimizer state
        lcm_optimizer.load_state_dict(checkpoint['lcm_optimizer'])
        cd_scheduler.load_state_dict(checkpoint['cd_scheduler'])  # Load scheduler state
        lcm_scheduler.load_state_dict(checkpoint['lcm_scheduler'])
        start_epoch = checkpoint['epoch']
        best_total_loss = checkpoint.get('best_total_loss', float('inf'))
        print(f"Loaded checkpoint from epoch {start_epoch}")

    history = {
        'train': {
            'loss': [],  # Combined CD and LCM loss
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'miou': [],
            'kappa': []
        },
        'val': {
            'loss': [],  # Combined CD and LCM loss
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'miou': [],
            'kappa': []
        }
    }

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Training
        train_metrics = train_epoch(
            model, train_loader, cd_criterion, lcm_criterion,
            cd_optimizer, lcm_optimizer, device, num_cd_classes
        )

        # Validation
        val_metrics = validate(
            model, val_loader, cd_criterion, lcm_criterion,
            device, num_cd_classes
        )

        # Calculate total loss for both phases
        train_total_loss = train_metrics['cd_loss'] + train_metrics['lcm_loss']
        val_total_loss = val_metrics['cd_loss'] + val_metrics['lcm_loss']

        # Update learning rates
        cd_scheduler.step(val_metrics['cd_loss'])
        lcm_scheduler.step(val_metrics['lcm_loss'])

        # Store metrics in history - store overall metrics
        for phase in ['train', 'val']:
            metrics = train_metrics if phase == 'train' else val_metrics
            total_loss = train_total_loss if phase == 'train' else val_total_loss

            history[phase]['loss'].append(float(total_loss))
            history[phase]['accuracy'].append(float(metrics['accuracy']))
            history[phase]['precision'].append(float(metrics['precision']))
            history[phase]['recall'].append(float(metrics['recall']))
            history[phase]['f1_score'].append(float(metrics['f1_score']))
            history[phase]['miou'].append(float(metrics['miou']))
            history[phase]['kappa'].append(float(metrics['kappa']))

        print_metrics_summary(train_metrics, val_metrics, train_total_loss, val_total_loss)

        # Save checkpoint
        if checkpoint_path:
            metrics = {
                'total_loss': val_total_loss,
                'accuracy': val_metrics['accuracy'],
                'precision': val_metrics['precision'],
                'recall': val_metrics['recall'],
                'f1_score': val_metrics['f1_score'],
                'miou': val_metrics['miou'],
                'kappa': val_metrics['kappa']
            }

            # Best model checkpoint based on total loss
            if val_total_loss < best_total_loss:
                best_total_loss = val_total_loss
                torch.save(
                    {
                        'epoch': epoch + 1,
                        'cd_model': model.cd_model.state_dict(),
                        'lcm_model': model.lcm_model.state_dict(),
                        'cd_optimizer': cd_optimizer.state_dict(),
                        'lcm_optimizer': lcm_optimizer.state_dict(),
                        'cd_scheduler': cd_scheduler.state_dict(),
                        'lcm_scheduler': lcm_scheduler.state_dict(),
                        'best_total_loss': best_total_loss,
                        'metrics': metrics,
                    },
                    checkpoint_path
                )
                print(f"Saved new best model with total loss: {best_total_loss:.4f}")

    return model, history

import json
def save_training_history(history, checkpoint_path, save_path, save_path_bestepoch):
    """
    Save training history and best epoch information to JSON files.

    Args:
        history (dict): Dictionary containing training and validation metrics
        checkpoint_path (str): Path to the model checkpoint file
        save_path (str): Path to save the training history
        save_path_bestepoch (str): Path to save the best epoch info
    """
    processed_history = {}
    for phase, metrics in history.items():
        if isinstance(metrics, list):  # Check if metrics is a list
            processed_history[phase] = [
                {metric: (value.tolist() if hasattr(value, 'tolist') else value)
                 for metric, value in entry.items()} if isinstance(entry, dict) else entry
                for entry in metrics
            ]
        else:  # Handle non-list entries
            processed_history[phase] = metrics.tolist() if hasattr(metrics, 'tolist') else metrics

    # Save processed history to a JSON file
    with open(save_path, 'w') as f:
        json.dump(processed_history, f, indent=4)

    print(f"Training history saved to: {save_path}")

    # Load checkpoint to get best epoch info
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    # print("\nCheckpoint contents:")
    # for key in checkpoint.keys():
    #     print(f"- {key}")

    # Extract best epoch information
    epoch_data = {
        'best_epoch': int(checkpoint['epoch']),
        'total_loss': float(checkpoint['metrics']['total_loss']),
        'accuracy': float(checkpoint['metrics']['accuracy']),
        'precision': float(checkpoint['metrics']['precision']),
        'recall': float(checkpoint['metrics']['recall']),
        'f1': float(checkpoint['metrics']['f1_score']),
        'miou': float(checkpoint['metrics']['miou']),
        'kappa': float(checkpoint['metrics']['kappa'])
    }

    # Save the best epoch info
    with open(save_path_bestepoch, 'w') as f:
        json.dump(epoch_data, f, indent=4)
    print(f"Best epoch info saved to: {save_path_bestepoch}")
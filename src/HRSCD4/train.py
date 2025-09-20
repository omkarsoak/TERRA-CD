import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from metrics import calculate_metrics

def stage_one_training(model, train_loader, val_loader, device, checkpoint_path, numepochs=50):
    """
    Stage 1: Train only the LCM branches
    """
    print("\nStarting Stage 1: Training LCM branches...")

    # Optimization setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Initialize tracking variables
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}

    # Try to load checkpoint if it exists
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            print(f"Resuming from epoch {start_epoch} with best val loss: {best_val_loss:.4f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch")

    for epoch in range(start_epoch, numepochs):
        print(f"\nEpoch {epoch+1}/{numepochs}")

        # Training phase
        model.train()
        train_loss = 0.0
        for data in tqdm(train_loader, desc='Training'):
            # Move all tensors to device
            img1, img2, mask1, mask2, _ = [x.to(device) for x in data]

            optimizer.zero_grad()

            # Forward pass - only LCM branches
            lcm1_out, lcm2_out, _ = model(img1, img2)

            # Calculate loss
            loss = (criterion(lcm1_out, mask1) + criterion(lcm2_out, mask2)) / 2

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Clear memory
            del lcm1_out, lcm2_out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_metrics_lcm1 = []
        all_metrics_lcm2 = []

        with torch.no_grad():
            for data in tqdm(val_loader, desc='Validation'):
                # Move all tensors to device
                img1, img2, mask1, mask2, _ = [x.to(device) for x in data]

                # Forward pass
                lcm1_out, lcm2_out, _ = model(img1, img2)

                # Calculate loss
                loss = (criterion(lcm1_out, mask1) + criterion(lcm2_out, mask2)) / 2
                val_loss += loss.item()

                # Calculate metrics
                lcm1_preds = torch.argmax(lcm1_out, dim=1)
                lcm2_preds = torch.argmax(lcm2_out, dim=1)

                metrics_lcm1 = calculate_metrics(lcm1_preds.cpu().numpy(),
                                              mask1.cpu().numpy(),
                                              num_classes=4)  # 4 semantic classes
                metrics_lcm2 = calculate_metrics(lcm2_preds.cpu().numpy(),
                                              mask2.cpu().numpy(),
                                              num_classes=4)

                all_metrics_lcm1.append(metrics_lcm1)
                all_metrics_lcm2.append(metrics_lcm2)

                # Clear memory
                del lcm1_out, lcm2_out, lcm1_preds, lcm2_preds
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Calculate average metrics
        avg_val_loss = val_loss / len(val_loader)

        # Aggregate metrics
        val_metrics = {
            'lcm1': {key: np.mean([m[key] for m in all_metrics_lcm1])
                    for key in all_metrics_lcm1[0].keys()},
            'lcm2': {key: np.mean([m[key] for m in all_metrics_lcm2])
                    for key in all_metrics_lcm2[0].keys()}
        }

        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss,
                'train_loss': avg_train_loss,
                'val_metrics': val_metrics
            }, checkpoint_path)
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")

        # Update learning rate based on val loss
        scheduler.step(avg_val_loss)

        # Store metrics
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_metrics'].append(val_metrics)

        # Print epoch results
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"LCM 2019 - mIoU: {val_metrics['lcm1']['miou']:.4f}, Accuracy: {val_metrics['lcm1']['accuracy']:.4f}")
        print(f"LCM 2024 - mIoU: {val_metrics['lcm2']['miou']:.4f}, Accuracy: {val_metrics['lcm2']['accuracy']:.4f}")

    return model, history

def stage_two_training(model, train_loader, val_loader, device, checkpoint_path, numepochs=50):
    """
    Stage 2: Train the full model end-to-end with fixed LCM weights
    """
    print("\nStarting Stage 2: Training CD branch...")

    # Freeze encoder and LCM decoder weights
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.lcm_decoder.parameters():
        param.requires_grad = False

    # Optimization setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Initialize tracking variables
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}

    # Try to load checkpoint if it exists
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            print(f"Resuming from epoch {start_epoch} with best val loss: {best_val_loss:.4f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch")

    for epoch in range(start_epoch, numepochs):
        print(f"\nEpoch {epoch+1}/{numepochs}")

        # Training phase
        model.train()
        train_loss = 0.0
        for data in tqdm(train_loader, desc='Training'):
            # Move all tensors to device
            img1, img2, _, _, cd_mask = [x.to(device) for x in data]

            optimizer.zero_grad()

            # Forward pass
            _, _, cd_out = model(img1, img2)

            # Calculate loss
            loss = criterion(cd_out, cd_mask)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Clear memory
            del cd_out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_metrics_cd = []

        with torch.no_grad():
            for data in tqdm(val_loader, desc='Validation'):
                # Move all tensors to device
                img1, img2, _, _, cd_mask = [x.to(device) for x in data]

                # Forward pass
                _, _, cd_out = model(img1, img2)

                # Calculate loss
                loss = criterion(cd_out, cd_mask)
                val_loss += loss.item()

                # Move tensors to CPU and convert to numpy
                cd_preds = torch.argmax(cd_out, dim=1).cpu().numpy()
                cd_mask_np = cd_mask.cpu().numpy()

                metrics = calculate_metrics(cd_preds,
                                         cd_mask_np,
                                         num_classes=13)  # 13 change classes
                all_metrics_cd.append(metrics)

                # Clear memory
                del cd_out, cd_preds
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Calculate average metrics
        avg_val_loss = val_loss / len(val_loader)
        val_metrics = {key: np.mean([m[key] for m in all_metrics_cd])
                      for key in all_metrics_cd[0].keys()}

        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss,
                'train_loss': avg_train_loss,
                'val_metrics': val_metrics
            }, checkpoint_path)
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")

        # Update learning rate based on val loss
        scheduler.step(avg_val_loss)

        # Store metrics
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_metrics'].append(val_metrics)

        # Print epoch results
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"CD Metrics - mIoU: {val_metrics['miou']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")

    return model, history


import json
def save_training_history(stage1_history, stage2_history, checkpoint_path, 
                          save_path_stage1, save_path_stage2, save_path_bestepoch):
    """
    Save training history and best epoch information to JSON files.

    Args:
        history (dict): Dictionary containing training and validation metrics
        checkpoint_path (str): Path to the model checkpoint file
        save_path (str): Path to save the training history
        save_path_bestepoch (str): Path to save the best epoch info
    """
    processed_history = {}
    for phase, metrics in stage1_history.items():
        if isinstance(metrics, list):  # Check if metrics is a list
            processed_history[phase] = [
                {metric: (value.tolist() if hasattr(value, 'tolist') else value)
                 for metric, value in entry.items()} if isinstance(entry, dict) else entry
                for entry in metrics
            ]
        else:  # Handle non-list entries
            processed_history[phase] = metrics.tolist() if hasattr(metrics, 'tolist') else metrics

    # Save processed history to a JSON file
    with open(save_path_stage1, 'w') as f:
        json.dump(processed_history, f, indent=4)

    print(f"Training history saved to: {save_path_stage1}")

    processed_history2 = {}
    for phase, metrics in stage2_history.items():
        if isinstance(metrics, list):  # Check if metrics is a list
            processed_history2[phase] = [
                {metric: (value.tolist() if hasattr(value, 'tolist') else value)
                 for metric, value in entry.items()} if isinstance(entry, dict) else entry
                for entry in metrics
            ]
        else:  # Handle non-list entries
            processed_history2[phase] = metrics.tolist() if hasattr(metrics, 'tolist') else metrics

    # Save processed history to a JSON file
    with open(save_path_stage2, 'w') as f:
        json.dump(processed_history2, f, indent=4)

    print(f"Training history saved to: {save_path_stage2}")

    # Load checkpoint to get best epoch info
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    # print("\nCheckpoint contents:")
    # for key in checkpoint.keys():
    #     print(f"- {key}")

    # Extract best epoch information
    epoch_data = {
        'best_epoch': int(checkpoint['epoch']),
        'total_loss': float(checkpoint['val_loss']),
        'accuracy': float(checkpoint['val_metrics']['accuracy']),
        'precision': float(checkpoint['val_metrics']['precision']),
        'recall': float(checkpoint['val_metrics']['recall']),
        'f1': float(checkpoint['val_metrics']['f1_score']),
        'miou': float(checkpoint['val_metrics']['miou']),
        'kappa': float(checkpoint['val_metrics']['kappa'])
    }

    # Save the best epoch info
    with open(save_path_bestepoch, 'w') as f:
        json.dump(epoch_data, f, indent=4)
    print(f"Best epoch info saved to: {save_path_bestepoch}")
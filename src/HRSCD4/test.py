import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import json
from metrics import calculate_metrics

def test_model_complete(model, test_loader, device, checkpoint_path, num_classes=13, num_semantic_classes=4):
    """
    Complete test function for multi-task change detection model.
    Tests both LCM and CD branches and provides detailed metrics.
    """
    print("\nStarting model testing...")

    # Load model checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Loaded model checkpoint")

    # Initialize storage for metrics
    all_cd_metrics = []
    all_lcm_metrics = []  # Single list for averaged LCM metrics
    total_cd_loss = 0
    total_lcm_loss = 0
    samples = 0
    criterion = nn.CrossEntropyLoss()

    # Store some random samples for visualization
    random_samples = []

    print("\nProcessing test data...")
    with torch.no_grad():
        for batch_idx, (img1, img2, mask1, mask2, cd_mask) in enumerate(tqdm(test_loader)):
            # Move data to device
            img1, img2 = img1.to(device), img2.to(device)
            mask1, mask2 = mask1.to(device), mask2.to(device)
            cd_mask = cd_mask.to(device)

            # Forward pass
            lcm1_out, lcm2_out, cd_out = model(img1, img2)

            # Calculate losses
            cd_loss = criterion(cd_out, cd_mask)
            lcm_loss = (criterion(lcm1_out, mask1) + criterion(lcm2_out, mask2)) / 2

            batch_size = img1.size(0)
            total_cd_loss += cd_loss.item() * batch_size
            total_lcm_loss += lcm_loss.item() * batch_size
            samples += batch_size

            # Get predictions
            cd_preds = torch.argmax(cd_out, dim=1)
            lcm1_preds = torch.argmax(lcm1_out, dim=1)
            lcm2_preds = torch.argmax(lcm2_out, dim=1)

            # Calculate metrics
            cd_metrics = calculate_metrics(cd_preds.cpu().numpy(),
                                        cd_mask.cpu().numpy(),
                                        num_classes=num_classes)  # 13 change classes

            # Calculate LCM metrics and average them
            lcm1_metrics = calculate_metrics(lcm1_preds.cpu().numpy(),
                                          mask1.cpu().numpy(),
                                          num_classes=num_semantic_classes)   # 4 semantic classes
            lcm2_metrics = calculate_metrics(lcm2_preds.cpu().numpy(),
                                          mask2.cpu().numpy(),
                                          num_classes=num_semantic_classes)   # 4 semantic classes

            # Average the LCM metrics
            lcm_metrics = {}
            for key in lcm1_metrics.keys():
                lcm_metrics[key] = (lcm1_metrics[key] + lcm2_metrics[key]) / 2

            all_cd_metrics.append(cd_metrics)
            all_lcm_metrics.append(lcm_metrics)

            # Store random samples for visualization
            if len(random_samples) < 5 and batch_idx % 10 == 0:  # Every 10th batch
                for i in range(min(batch_size, 5 - len(random_samples))):
                    random_samples.append({
                        'img1': img1[i].cpu(),
                        'img2': img2[i].cpu(),
                        'lcm1_pred': lcm1_preds[i].cpu(),
                        'lcm1_true': mask1[i].cpu(),
                        'lcm2_pred': lcm2_preds[i].cpu(),
                        'lcm2_true': mask2[i].cpu(),
                        'cd_pred': cd_preds[i].cpu(),
                        'cd_true': cd_mask[i].cpu(),
                        'cd_probs': torch.softmax(cd_out[i], dim=0).cpu()
                    })

            # Clear memory
            del lcm1_out, lcm2_out, cd_out
            del cd_preds, lcm1_preds, lcm2_preds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Calculate average losses
    avg_cd_loss = total_cd_loss / samples
    avg_lcm_loss = total_lcm_loss / samples

    # Calculate average metrics
    def aggregate_metrics(metrics_list):
        result = {}
        for key in metrics_list[0].keys():
            result[key] = np.mean([m[key] for m in metrics_list])
        return result

    cd_metrics = aggregate_metrics(all_cd_metrics)
    lcm_metrics = aggregate_metrics(all_lcm_metrics)

    # Print results
    print("\nTest Results:")
    print("\nChange Detection Metrics:")
    print(f"Loss: {avg_cd_loss:.4f}")
    print(f"Accuracy: {cd_metrics['accuracy']:.4f}")
    print(f"mIoU: {cd_metrics['miou']:.4f}")
    print(f"F1-Score: {cd_metrics['f1_score']:.4f}")
    print(f"Kappa: {cd_metrics['kappa']:.4f}")

    print("\nLand Cover Mapping Metrics (Averaged):")
    print(f"Loss: {avg_lcm_loss:.4f}")
    print(f"Accuracy: {lcm_metrics['accuracy']:.4f}")
    print(f"mIoU: {lcm_metrics['miou']:.4f}")
    print(f"F1-Score: {lcm_metrics['f1_score']:.4f}")
    print(f"Kappa: {lcm_metrics['kappa']:.4f}")

    # Visualize results
    if random_samples:
        visualize_results(random_samples)
    else:
        print("\nNo samples available for visualization")

    return {
        'cd_metrics': cd_metrics,
        'lcm_metrics': lcm_metrics,  # Single set of averaged LCM metrics
        'cd_loss': avg_cd_loss,
        'lcm_loss': avg_lcm_loss
    }

def visualize_results(samples):
    """
    Visualize test results including original images, predictions, 
    and ground truth for both LCM and CD
    """
    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 8, figsize=(32, 4*num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for idx, sample in enumerate(samples):
        # Original images
        axes[idx, 0].imshow(sample['img1'].permute(1, 2, 0))
        axes[idx, 0].set_title('Image 2019')
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(sample['img2'].permute(1, 2, 0))
        axes[idx, 1].set_title('Image 2024')
        axes[idx, 1].axis('off')

        # LCM 2019 results
        axes[idx, 2].imshow(sample['lcm1_pred'], cmap='tab10')
        axes[idx, 2].set_title('LCM 2019 Pred')
        axes[idx, 2].axis('off')

        axes[idx, 3].imshow(sample['lcm1_true'], cmap='tab10')
        axes[idx, 3].set_title('LCM 2019 GT')
        axes[idx, 3].axis('off')

        # LCM 2024 results
        axes[idx, 4].imshow(sample['lcm2_pred'], cmap='tab10')
        axes[idx, 4].set_title('LCM 2024 Pred')
        axes[idx, 4].axis('off')

        axes[idx, 5].imshow(sample['lcm2_true'], cmap='tab10')
        axes[idx, 5].set_title('LCM 2024 GT')
        axes[idx, 5].axis('off')

        # CD results
        axes[idx, 6].imshow(sample['cd_pred'], cmap='tab10')
        axes[idx, 6].set_title('CD Prediction')
        axes[idx, 6].axis('off')

        axes[idx, 7].imshow(sample['cd_true'], cmap='tab10')
        axes[idx, 7].set_title('CD Ground Truth')
        axes[idx, 7].axis('off')

    # Add colorbars and adjust layout
    plt.tight_layout()
    plt.show()

def save_test_metrics(history, save_path):
    # Convert tensors or arrays in the history to lists for JSON serialization
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

    print(f"Testing history saved to: {save_path}")
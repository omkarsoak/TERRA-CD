import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import random
import os
import json
from metrics import calculate_metrics, calculate_effective_weights


def test_model(model, test_loader, loss='CE', device='cuda',
               num_classes=3, weighting_method='square_balanced',checkpoint_path='best_stanet_model.pt'):

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {checkpoint_path}")
    model.eval()

    # Calculate class weights
    class_weights, _ = calculate_effective_weights(test_loader, device,
                                                   num_classes=num_classes,
                                                   method=weighting_method)
    print(f"Class weights: {class_weights}")


    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # For visualization and metrics
    random_samples = []
    total_loss = 0.0
    total_samples = 0

    # Collect predictions and labels for comprehensive metrics
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs1, inputs2, labels in test_loader:
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs1, inputs2)
            loss = criterion(outputs, labels)

            # Accumulate loss
            total_loss += loss.item() * inputs1.size(0)
            total_samples += inputs1.size(0)

            # Get predictions
            preds = torch.argmax(outputs, dim=1)

            # Store predictions and labels
            all_predictions.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # Store random samples for visualization
            if len(random_samples) < 5:
                for i in range(min(inputs1.size(0), 5 - len(random_samples))):
                    if random.random() < 0.2:  # 20% chance to select each sample
                        random_samples.append({
                            'image1': inputs1[i].cpu(),
                            'image2': inputs2[i].cpu(),
                            'label': labels[i].cpu(),
                            'pred': preds[i].cpu(),
                            'probabilities': torch.softmax(outputs[i], dim=0).cpu()
                        })

    # Concatenate predictions and labels
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)

    # Calculate metrics
    test_metrics = calculate_metrics(all_predictions, all_labels, num_classes)

    # Add loss to metrics
    test_metrics['loss'] = total_loss / total_samples

    # Make sure we have exactly 5 samples
    while len(random_samples) < 5:
        random_samples.append(random_samples[-1] if random_samples else {
            'image1': torch.zeros(3, 64, 64),
            'image2': torch.zeros(3, 64, 64),
            'label': torch.zeros(64, 64),
            'pred': torch.zeros(64, 64),
            'probabilities': torch.zeros(3, 64, 64)
        })

    return random_samples, test_metrics

def visualize_results(random_samples, num_classes=3):
    # Extract samples and metrics
    # random_samples = random_samples_and_metrics[0]
    # test_metrics = random_samples_and_metrics[1]

    # Create a figure with subplots
    fig, axes = plt.subplots(5, 4, figsize=(25, 25))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    for idx, sample in enumerate(random_samples):
        # Normalize and convert images for display
        img1 = sample['image1'].numpy().transpose(1, 2, 0)
        img2 = sample['image2'].numpy().transpose(1, 2, 0)
        img1 = (img1 - img1.min()) / (img1.max() - img1.min())
        img2 = (img2 - img2.min()) / (img2.max() - img2.min())

        # Get masks
        pred_mask = sample['pred'].numpy()
        true_mask = sample['label'].numpy()

        # Plot images and masks
        axes[idx, 0].imshow(img1)
        axes[idx, 0].set_title('Image 1')
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(img2)
        axes[idx, 1].set_title('Image 2')
        axes[idx, 1].axis('off')

        # Plot predicted mask
        pred_plot = axes[idx, 2].imshow(pred_mask, cmap='tab10', vmin=0, vmax=num_classes-1)
        axes[idx, 2].set_title('Predicted Change')
        axes[idx, 2].axis('off')

        # Plot ground truth mask
        true_plot = axes[idx, 3].imshow(true_mask, cmap='tab10', vmin=0, vmax=num_classes-1)
        axes[idx, 3].set_title('Ground Truth')
        axes[idx, 3].axis('off')

    plt.tight_layout()
    plt.show()

def save_test_metrics(test_metrics, save_dir, model_name, attention_mode, num_classes, num_epochs):
    """Save test metrics to JSON"""
    metrics_file = os.path.join(save_dir, f"{model_name}_{attention_mode}-{num_classes}_classes_{num_epochs}_test_metrics.json")

    # Use the pre-computed metrics directly
    with open(metrics_file, 'w') as f:
        json.dump(test_metrics, f, indent=4)

    print(f"\nSaved test metrics to: {metrics_file}")
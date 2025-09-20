import random
import matplotlib.pyplot as plt
import torch
import tqdm as tqdm
from train import create_semantic_change_mask
import json
from metrics import calculate_metrics

def visualize_predictions(model, img_2019, img_2024, true_mask, seg_mask_2019, seg_mask_2024):
    """
    Visualize model predictions in a single row
    """
    model.eval()
    with torch.no_grad():
        # Remove batch dimension if present
        if img_2019.dim() == 4:
            img_2019 = img_2019.squeeze(0)
        if img_2024.dim() == 4:
            img_2024 = img_2024.squeeze(0)
        if true_mask.dim() == 4:
            true_mask = true_mask.squeeze(0)
        if seg_mask_2019.dim() == 4:
            seg_mask_2019 = seg_mask_2019.squeeze(0)
        if seg_mask_2024.dim() == 4:
            seg_mask_2024 = seg_mask_2024.squeeze(0)

        # Get predictions
        cd_pred = model.cd_model(torch.cat([img_2019.unsqueeze(0), img_2024.unsqueeze(0)], dim=1))
        lcm_pred_2019 = model.lcm_model(img_2019.unsqueeze(0))
        lcm_pred_2024 = model.lcm_model(img_2024.unsqueeze(0))
        semantic_pred = create_semantic_change_mask(cd_pred, lcm_pred_2019, lcm_pred_2024)
        semantic_pred = semantic_pred.squeeze(0)

    # Create single row visualization
    fig, axes = plt.subplots(1, 9, figsize=(24, 3))

    # Plot images without titles
    # Original images
    axes[0].imshow(img_2019.cpu().permute(1,2,0))
    axes[0].set_title('Image2019')
    axes[1].imshow(img_2024.cpu().permute(1,2,0))
    axes[1].set_title('Image2024')

    # Binary change detection
    axes[2].imshow(torch.sigmoid(cd_pred).cpu().squeeze(), cmap='gray')
    axes[2].set_title('BinaryCD')

    # Ground truth segmentation masks
    axes[3].imshow(seg_mask_2019.cpu(), cmap='tab10')
    axes[3].set_title('GTSemMask2019')
    axes[4].imshow(seg_mask_2024.cpu(), cmap='tab10')
    axes[4].set_title('GTSemMask2024')

    # Predicted segmentation masks
    axes[5].imshow(torch.argmax(lcm_pred_2019, dim=1).squeeze(0).cpu(), cmap='tab10')
    axes[5].set_title('PredSemMask2019')
    axes[6].imshow(torch.argmax(lcm_pred_2024, dim=1).squeeze(0).cpu(), cmap='tab10')
    axes[6].set_title('PredSemMask2024')
    # Change detection
    axes[7].imshow(true_mask.cpu(), cmap='tab10')
    axes[7].set_title('GTChangeMask')
    axes[8].imshow(semantic_pred.cpu(), cmap='tab10')
    axes[8].set_title('PredChangeMask')

    # Remove axes and padding
    for ax in axes:
        ax.axis('off')
    plt.subplots_adjust(wspace=0.05, hspace=0)

    return fig

def test_strategy3(model, test_loader, device, num_samples_to_plot=5, 
                   checkpoint_path='best_model.pt', num_cd_classes=13, num_semantic_classes=4):
    """
    Evaluate model and display metrics for both semantic segmentation and change detection
    """
    # Load checkpoint - modified to load separate models
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loading checkpoint from {checkpoint_path}")

    # Load the separate models
    try:
        model.cd_model.load_state_dict(checkpoint['cd_model'])
        model.lcm_model.load_state_dict(checkpoint['lcm_model'])
        print("Successfully loaded both CD and LCM models")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    model.eval()

    # Initialize metric storage
    change_predictions = []
    change_targets = []
    seg_predictions_2019 = []
    seg_targets_2019 = []
    seg_predictions_2024 = []
    seg_targets_2024 = []

    # Store samples for visualization
    stored_samples = []
    total_samples = len(test_loader.dataset)
    random_indices = set(random.sample(range(total_samples), num_samples_to_plot))
    current_idx = 0

    print("Testing model...")
    with torch.no_grad():
        for img_2019, img_2024, sem_2019, sem_2024, cd_mask in tqdm(test_loader):
            # Move to device
            img_2019 = img_2019.to(device)
            img_2024 = img_2024.to(device)
            sem_2019 = sem_2019.to(device)
            sem_2024 = sem_2024.to(device)
            cd_mask = cd_mask.to(device)

            # Get predictions
            cd_pred = model.cd_model(torch.cat([img_2019, img_2024], dim=1))
            lcm_pred_2019 = model.lcm_model(img_2019)
            lcm_pred_2024 = model.lcm_model(img_2024)
            semantic_pred = create_semantic_change_mask(cd_pred, lcm_pred_2019, lcm_pred_2024)

            # Get semantic segmentation predictions
            seg_pred_2019 = torch.argmax(lcm_pred_2019, dim=1)
            seg_pred_2024 = torch.argmax(lcm_pred_2024, dim=1)

            # Append predictions and targets for later metric calculation
            change_predictions.append(semantic_pred.cpu())
            change_targets.append(cd_mask.cpu())
            seg_predictions_2019.append(seg_pred_2019.cpu())
            seg_targets_2019.append(sem_2019.cpu())
            seg_predictions_2024.append(seg_pred_2024.cpu())
            seg_targets_2024.append(sem_2024.cpu())

            # Store random samples
            batch_size = img_2019.size(0)
            for i in range(batch_size):
                if current_idx + i in random_indices:
                    stored_samples.append({
                        'img_2019': img_2019[i],
                        'img_2024': img_2024[i],
                        'cd_pred': cd_pred[i],
                        'lcm_pred_2019': lcm_pred_2019[i],
                        'lcm_pred_2024': lcm_pred_2024[i],
                        'semantic_pred': semantic_pred[i],
                        'true_mask': cd_mask[i],
                        'seg_mask_2019': sem_2019[i],
                        'seg_mask_2024': sem_2024[i]
                    })
            current_idx += batch_size

    # Calculate metrics
    print("\n" + "="*50)
    print("Semantic Segmentation Metrics:")
    print("="*50)

    # 2019 Segmentation Metrics
    seg_preds_2019 = torch.cat(seg_predictions_2019, dim=0).numpy()
    seg_targets_2019 = torch.cat(seg_targets_2019, dim=0).numpy()
    metrics_seg_2019 = calculate_metrics(seg_preds_2019, seg_targets_2019, num_cd_classes=num_semantic_classes)

    print("\n2019 Segmentation:")
    print(f"Accuracy: {metrics_seg_2019['accuracy']:.4f}")
    print(f"Mean IoU: {metrics_seg_2019['miou']:.4f}")
    print(f"F1 Score: {metrics_seg_2019['f1_score']:.4f}")
    print(f"Precision: {metrics_seg_2019['precision']:.4f}")
    print(f"Recall: {metrics_seg_2019['recall']:.4f}")
    print(f"Kappa: {metrics_seg_2019['kappa']:.4f}")

    # 2024 Segmentation Metrics
    seg_preds_2024 = torch.cat(seg_predictions_2024, dim=0).numpy()
    seg_targets_2024 = torch.cat(seg_targets_2024, dim=0).numpy()
    metrics_seg_2024 = calculate_metrics(seg_preds_2024, seg_targets_2024, num_cd_classes=num_semantic_classes)

    print("\n2024 Segmentation:")
    print(f"Accuracy: {metrics_seg_2024['accuracy']:.4f}")
    print(f"Mean IoU: {metrics_seg_2024['miou']:.4f}")
    print(f"F1 Score: {metrics_seg_2024['f1_score']:.4f}")
    print(f"Precision: {metrics_seg_2024['precision']:.4f}")
    print(f"Recall: {metrics_seg_2024['recall']:.4f}")
    print(f"Kappa: {metrics_seg_2024['kappa']:.4f}")

    # Average Segmentation Metrics
    print("\nAverage Segmentation:")
    avg_seg_metrics = {
        'accuracy': (metrics_seg_2019['accuracy'] + metrics_seg_2024['accuracy']) / 2,
        'miou': (metrics_seg_2019['miou'] + metrics_seg_2024['miou']) / 2,
        'f1_score': (metrics_seg_2019['f1_score'] + metrics_seg_2024['f1_score']) / 2,
        'precision': (metrics_seg_2019['precision'] + metrics_seg_2024['precision']) / 2,
        'recall': (metrics_seg_2019['recall'] + metrics_seg_2024['recall']) / 2,
        'kappa': (metrics_seg_2019['kappa'] + metrics_seg_2024['kappa']) / 2
    }
    print(f"Accuracy: {avg_seg_metrics['accuracy']:.4f}")
    print(f"Mean IoU: {avg_seg_metrics['miou']:.4f}")
    print(f"F1 Score: {avg_seg_metrics['f1_score']:.4f}")
    print(f"Precision: {avg_seg_metrics['precision']:.4f}")
    print(f"Recall: {avg_seg_metrics['recall']:.4f}")
    print(f"Kappa: {avg_seg_metrics['kappa']:.4f}")

    print("\n" + "="*50)
    print("Change Detection Metrics:")
    print("="*50)

    # Change Detection Metrics
    change_preds = torch.cat(change_predictions, dim=0).numpy()
    change_targets = torch.cat(change_targets, dim=0).numpy()
    metrics_change = calculate_metrics(change_preds, change_targets, num_cd_classes=num_cd_classes)

    print(f"Accuracy: {metrics_change['accuracy']:.4f}")
    print(f"Mean IoU: {metrics_change['miou']:.4f}")
    print(f"F1 Score: {metrics_change['f1_score']:.4f}")
    print(f"Precision: {metrics_change['precision']:.4f}")
    print(f"Recall: {metrics_change['recall']:.4f}")
    print(f"Kappa: {metrics_change['kappa']:.4f}")

    # Plot stored samples
    print("\nPlotting random samples...")
    for sample in stored_samples:
        fig = visualize_predictions(
            model,
            sample['img_2019'],
            sample['img_2024'],
            sample['true_mask'],
            sample['seg_mask_2019'],
            sample['seg_mask_2024']
        )
        plt.show()
        plt.close(fig)

    return {
        'segmentation_2019': metrics_seg_2019,
        'segmentation_2024': metrics_seg_2024,
        'segmentation_avg': avg_seg_metrics,
        'change_detection': metrics_change
    }


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
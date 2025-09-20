import torch
import matplotlib.pyplot as plt
import numpy as np
import json
from metrics import calculate_metrics
from tqdm import tqdm

def plot_results_pcc(img1, img2, sem_pred1, sem_pred2, sem_gt1, sem_gt2,
                    change_pred, change_gt):
    """Plot the results from the PCC model in notebook cells"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Plot images
    axes[0, 0].imshow(img1.cpu().permute(1, 2, 0))
    axes[0, 0].set_title('Image 2019')
    axes[0, 1].imshow(img2.cpu().permute(1, 2, 0))
    axes[0, 1].set_title('Image 2024')

    # Plot semantic predictions and ground truth
    axes[0, 2].imshow(sem_pred1.cpu())
    axes[0, 2].set_title('Semantic Pred 2019')
    axes[0, 3].imshow(sem_pred2.cpu())
    axes[0, 3].set_title('Semantic Pred 2024')

    axes[1, 0].imshow(sem_gt1.cpu())
    axes[1, 0].set_title('Semantic GT 2019')
    axes[1, 1].imshow(sem_gt2.cpu())
    axes[1, 1].set_title('Semantic GT 2024')

    # Plot change detection results
    axes[1, 2].imshow(change_pred.cpu())
    axes[1, 2].set_title('Change Prediction')
    axes[1, 3].imshow(change_gt.cpu())
    axes[1, 3].set_title('Change GT')

    plt.tight_layout()
    plt.show()


def test_model_pcc(model, test_loader, checkpoint_path, device, num_samples_to_plot=5,
                   num_cd_classes=3, weighted_metrics=False):
    """Test the PCC model with enhanced metrics for both change detection and semantic segmentation"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {checkpoint_path}")
    model.eval()

    # Initialize metric trackers
    cd_metrics = []  # Change detection metrics
    sem_2019_metrics = []  # Semantic segmentation 2019 metrics
    sem_2024_metrics = []  # Semantic segmentation 2024 metrics
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            img_t2019 = batch['img_t2019'].to(device)
            img_t2024 = batch['img_t2024'].to(device)
            sem_mask_2019 = batch['sem_mask_2019'].to(device)
            sem_mask_2024 = batch['sem_mask_2024'].to(device)
            cd_mask = batch['cd_mask'].to(device)

            sem_out1, sem_out2, change_out = model(img_t2019, img_t2024)

            # Get predictions
            sem_pred1 = torch.argmax(sem_out1, dim=1)
            sem_pred2 = torch.argmax(sem_out2, dim=1)
            change_pred = torch.argmax(change_out, dim=1)

            # Calculate metrics for all tasks
            batch_cd_metrics = calculate_metrics(change_pred.cpu().numpy(),
                                              cd_mask.cpu().numpy(),
                                              num_cd_classes,
                                              weighted_metrics=weighted_metrics)

            batch_sem_2019_metrics = calculate_metrics(sem_pred1.cpu().numpy(),
                                                     sem_mask_2019.cpu().numpy(),
                                                     num_cd_classes,
                                                     weighted_metrics=weighted_metrics)

            batch_sem_2024_metrics = calculate_metrics(sem_pred2.cpu().numpy(),
                                                     sem_mask_2024.cpu().numpy(),
                                                     num_cd_classes,
                                                     weighted_metrics=weighted_metrics)

            cd_metrics.append(batch_cd_metrics)
            sem_2019_metrics.append(batch_sem_2019_metrics)
            sem_2024_metrics.append(batch_sem_2024_metrics)

            # Store predictions for plotting
            all_predictions.append({
                'img_t2019': img_t2019.cpu(),
                'img_t2024': img_t2024.cpu(),
                'sem_pred1': sem_pred1.cpu(),
                'sem_pred2': sem_pred2.cpu(),
                'sem_mask_2019': sem_mask_2019.cpu(),
                'sem_mask_2024': sem_mask_2024.cpu(),
                'change_pred': change_pred.cpu(),
                'cd_mask': cd_mask.cpu()
            })

    # Aggregate metrics
    def aggregate_metrics(metrics_list):
        final_metrics = {}
        for key in metrics_list[0].keys():
            final_metrics[key] = np.mean([m[key] for m in metrics_list])
        return final_metrics

    final_cd_metrics = aggregate_metrics(cd_metrics)
    final_sem_2019_metrics = aggregate_metrics(sem_2019_metrics)
    final_sem_2024_metrics = aggregate_metrics(sem_2024_metrics)

    # Calculate average semantic segmentation metrics
    avg_sem_metrics = {}
    for key in final_sem_2019_metrics.keys():
        avg_sem_metrics[key] = (final_sem_2019_metrics[key] + final_sem_2024_metrics[key]) / 2

    # Print detailed metrics
    print("\n=== Change Detection Metrics ===")
    print(f"Overall Accuracy: {final_cd_metrics['accuracy']:.4f}")
    print(f"Kappa Score: {final_cd_metrics['kappa']:.4f}")
    print(f"mIoU: {final_cd_metrics['miou']:.4f}")
    print(f"F1 score: {final_cd_metrics['f1_score']:.4f}")

    print("\n=== 2019 Semantic Segmentation Metrics ===")
    print(f"Overall Accuracy: {final_sem_2019_metrics['accuracy']:.4f}")
    print(f"Kappa Score: {final_sem_2019_metrics['kappa']:.4f}")
    print(f"mIoU: {final_sem_2019_metrics['miou']:.4f}")
    print(f"F1 score: {final_sem_2019_metrics['f1_score']:.4f}")

    print("\n=== 2024 Semantic Segmentation Metrics ===")
    print(f"Overall Accuracy: {final_sem_2024_metrics['accuracy']:.4f}")
    print(f"Kappa Score: {final_sem_2024_metrics['kappa']:.4f}")
    print(f"mIoU: {final_sem_2024_metrics['miou']:.4f}")
    print(f"F1 score: {final_sem_2024_metrics['f1_score']:.4f}")

    print("\n=== Average Semantic Segmentation Metrics ===")
    print(f"Overall Accuracy: {avg_sem_metrics['accuracy']:.4f}")
    print(f"Kappa Score: {avg_sem_metrics['kappa']:.4f}")
    print(f"mIoU: {avg_sem_metrics['miou']:.4f}")
    print(f"F1 score: {avg_sem_metrics['f1_score']:.4f}")

    # Plot random samples
    total_samples = len(all_predictions)
    random_indices = np.random.choice(total_samples, num_samples_to_plot, replace=False)

    print(f"\nPlotting {num_samples_to_plot} random samples...")
    for idx in random_indices:
        batch = all_predictions[idx]
        plot_results_pcc(
            batch['img_t2019'][0],
            batch['img_t2024'][0],
            batch['sem_pred1'][0],
            batch['sem_pred2'][0],
            batch['sem_mask_2019'][0],
            batch['sem_mask_2024'][0],
            batch['change_pred'][0],
            batch['cd_mask'][0]
        )

    return {
        'change_detection': final_cd_metrics,
        'semantic_2019': final_sem_2019_metrics,
        'semantic_2024': final_sem_2024_metrics,
        'semantic_average': avg_sem_metrics
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

    print(f"Training history saved to: {save_path}")
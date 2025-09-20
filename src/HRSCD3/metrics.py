import numpy as np
import torch
from sklearn.metrics import confusion_matrix

def calculate_class_weights(train_loader, device, num_cd_classes=13):
    """
        method: Weighting method is 'square_balanced'
    """

    # Initialize counters on CPU first
    class_counts = torch.zeros(num_cd_classes)
    total_pixels = 0

    print("Calculating class weights...")
    # Count frequencies on CPU
    for batch in train_loader:
        cd_mask = batch[-1]  # Get the last item which is cd_mask
        unique_labels = torch.unique(cd_mask)
        for label in unique_labels:
            if label < num_cd_classes:  # Safety check
                class_counts[label] += (cd_mask == label).sum().item()
        total_pixels += cd_mask.numel()

    class_frequencies = class_counts / total_pixels
    # Square root of inverse frequencies (less aggressive balancing)
    weights = torch.sqrt(1.0 / class_frequencies)

    # Normalize weights to sum to num_cd_classes
    weights = weights * (num_cd_classes / weights.sum())

    return weights


def calculate_metrics(predictions, targets, num_cd_classes):
    """
    Calculate metrics with per-class IoU and unweighted averaging
    """
    # Flatten predictions and targets
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()

    # Compute confusion matrix
    cm = confusion_matrix(target_flat, pred_flat, labels=range(num_cd_classes))

    # Calculate metrics for each class
    metrics = {}
    class_metrics = []

    # Per-class calculations
    for i in range(num_cd_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - tp - fp - fn

        # Handle divide by zero
        union = tp + fp + fn
        if union == 0:
            iou = 0
        else:
            iou = tp / union

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        class_metrics.append({
            'class': i,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    # Calculate averages - only for classes present in ground truth
    # precision, recall and f1 are weighted
    present_classes = np.unique(target_flat)
    total = np.sum(cm, axis=1)
    metrics['miou'] = np.mean([class_metrics[i]['iou'] for i in present_classes])
    metrics['precision'] = np.average([m['precision'] for m in class_metrics], weights=total)
    metrics['recall'] = np.average([m['recall'] for m in class_metrics] , weights=total)
    metrics['f1_score'] = np.average([m['f1'] for m in class_metrics], weights=total)

    # Overall accuracy
    metrics['accuracy'] = np.sum(np.diag(cm)) / np.sum(cm)

    # Kappa calculation
    n = np.sum(cm)
    sum_po = np.sum(np.diag(cm))
    sum_pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / n
    metrics['kappa'] = (sum_po - sum_pe) / (n - sum_pe + 1e-6)

    return metrics
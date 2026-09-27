"""
Evaluation module for Siamese U-Net.
Provides reusable metrics and evaluation loop.
"""

import torch
import numpy as np

def compute_confusion_matrix(preds, targets, num_classes=4, ignore_index=4):
    """
    Computes a confusion matrix for semantic segmentation.
    preds: [B, H, W] integer predictions
    targets: [B, H, W] integer targets
    ignore_index: pixels with this target value are ignored
    Returns: confusion matrix of shape [num_classes, num_classes]
    """
    valid_mask = targets != ignore_index
    preds_valid = preds[valid_mask]
    targets_valid = targets[valid_mask]
    
    indices = targets_valid * num_classes + preds_valid
    conf_matrix = torch.bincount(indices, minlength=num_classes**2)
    conf_matrix = conf_matrix.reshape(num_classes, num_classes)
    return conf_matrix.cpu().numpy()

def calculate_metrics(conf_matrix):
    """
    Calculates pixel accuracy, IoU, and Dice from a confusion matrix.
    conf_matrix: [num_classes, num_classes] numpy array
    """
    tp = np.diag(conf_matrix)
    fp = conf_matrix.sum(axis=0) - tp
    fn = conf_matrix.sum(axis=1) - tp
    
    total_valid = conf_matrix.sum()
    pixel_acc = tp.sum() / total_valid if total_valid > 0 else 0.0
    
    epsilon = 1e-6
    iou = tp / (tp + fp + fn + epsilon)
    dice = 2 * tp / (2 * tp + fp + fn + epsilon)
    
    # Exclude absent classes from mean calculations to avoid destabilizing metrics
    class_exists = (tp + fp + fn) > 0
    valid_iou = iou[class_exists]
    valid_dice = dice[class_exists]
    
    mean_iou = valid_iou.mean() if len(valid_iou) > 0 else 0.0
    mean_dice = valid_dice.mean() if len(valid_dice) > 0 else 0.0
    
    return {
        "pixel_accuracy": float(pixel_acc),
        "per_class_iou": iou.tolist(),
        "mean_iou": float(mean_iou),
        "per_class_dice": dice.tolist(),
        "mean_dice": float(mean_dice),
        "class_exists": class_exists.tolist()
    }

@torch.no_grad()
def evaluate_epoch(model, dataloader, criterion, device, num_classes=4, ignore_index=4):
    """
    Runs a full evaluation epoch.
    Returns average loss and metrics.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    total_conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    for batch in dataloader:
        pre_img = batch["pre_image"].to(device, non_blocking=True)
        post_img = batch["post_image"].to(device, non_blocking=True)
        mask = batch["damage_mask"].to(device, non_blocking=True)
        
        logits = model(pre_img, post_img)
        loss = criterion(logits, mask)
        
        total_loss += loss.item()
        num_batches += 1
        
        preds = torch.argmax(logits, dim=1)
        conf_matrix = compute_confusion_matrix(preds, mask, num_classes, ignore_index)
        total_conf_matrix += conf_matrix
        
    avg_loss = total_loss / max(1, num_batches)
    metrics = calculate_metrics(total_conf_matrix)
    metrics["loss"] = avg_loss
    return metrics

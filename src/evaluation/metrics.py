"""Evaluation metrics for semantic segmentation."""

import numpy as np
import torch
from tqdm import tqdm


IGNORE_INDEX = 255


def calculate_iou(pred, target, num_classes=19, ignore_index=IGNORE_INDEX):
    """
    Calculate IoU for each class.
    
    Args:
        pred: Predicted labels (numpy array)
        target: Ground truth labels (numpy array)
        num_classes: Number of classes (default: 19)
        ignore_index: Index to ignore (default: 255)
        
    Returns:
        List of IoU values per class
    """
    ious = []
    
    # Ensure numpy arrays
    pred = np.array(pred)
    target = np.array(target)
    
    # Ignore pixels with ignore_index
    mask = (target != ignore_index)
    pred = pred[mask]
    target = target[mask]
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    
    return ious


def evaluate_model(model, val_loader, device, num_classes=19):
    """
    Evaluate model on validation set.
    
    Args:
        model: Model to evaluate
        val_loader: Validation dataloader
        device: Device to run evaluation on
        num_classes: Number of classes (default: 19)
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_correct = 0
    total_pixels = 0
    total_ious = []
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating", position=0, leave=False):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs['out']
            preds = torch.argmax(outputs, dim=1)
            
            # Pixel accuracy (ignore ignore_index)
            valid_mask = (masks != IGNORE_INDEX)
            total_correct += (preds[valid_mask] == masks[valid_mask]).sum().item()
            total_pixels += valid_mask.sum().item()
            
            # IoU per sample
            for i in range(images.size(0)):
                pred_np = preds[i].cpu().numpy()
                mask_np = masks[i].cpu().numpy()
                ious = calculate_iou(pred_np, mask_np, num_classes)
                total_ious.append(ious)
    
    # Calculate average metrics
    pixel_accuracy = total_correct / total_pixels if total_pixels > 0 else 0.0
    
    # Calculate mean IoU per class
    total_ious = np.array(total_ious)
    class_ious = np.nanmean(total_ious, axis=0)
    mean_iou = np.nanmean(class_ious)
    
    return {
        'pixel_accuracy': pixel_accuracy,
        'mean_iou': mean_iou,
        'class_ious': class_ious.tolist()
    }


def print_evaluation_results(metrics):
    """
    Print evaluation results in a formatted way.
    
    Args:
        metrics: Dictionary with evaluation metrics
    """
    class_names = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
    ]
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    
    print("\nClass-wise IoU:")
    for i, (name, iou) in enumerate(zip(class_names, metrics['class_ious'])):
        print(f"  {name:20s}: {iou:.4f}")
    print("="*60)

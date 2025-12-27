"""Visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap, BoundaryNorm


# Cityscapes color palette
CITYSCAPES_COLORS = [
    (128, 64, 128),   # 0 road
    (244, 35, 232),   # 1 sidewalk
    (70, 70, 70),     # 2 building
    (102, 102, 156),  # 3 wall
    (190, 153, 153),  # 4 fence
    (153, 153, 153),  # 5 pole
    (250, 170, 30),   # 6 traffic light
    (220, 220, 0),    # 7 traffic sign
    (107, 142, 35),   # 8 vegetation
    (152, 251, 152),  # 9 terrain
    (70, 130, 180),   # 10 sky
    (220, 20, 60),    # 11 person
    (255, 0, 0),      # 12 rider
    (0, 0, 142),      # 13 car
    (0, 0, 70),       # 14 truck
    (0, 60, 100),     # 15 bus
    (0, 80, 100),     # 16 train
    (0, 0, 230),      # 17 motorcycle
    (119, 11, 32),    # 18 bicycle
]


def get_cityscapes_colormap():
    """Get Cityscapes colormap for visualization."""
    cmap = ListedColormap([(r/255.0, g/255.0, b/255.0) for (r, g, b) in CITYSCAPES_COLORS])
    cmap.set_bad(color=(0.5, 0.5, 0.5))  # grey for ignore=255
    norm = BoundaryNorm(boundaries=list(range(20)), ncolors=cmap.N)
    return cmap, norm


def denormalize_image(image_tensor):
    """
    Denormalize image tensor for visualization.
    
    Args:
        image_tensor: Normalized image tensor (C, H, W)
        
    Returns:
        Denormalized image as numpy array (H, W, C)
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    image = image_tensor * std + mean
    image = torch.clamp(image, 0, 1)
    image = image.permute(1, 2, 0).numpy()
    
    return image


def visualize_predictions(
    model,
    dataloader,
    device,
    num_samples=3,
    save_path=None
):
    """
    Visualize model predictions.
    
    Args:
        model: Model to use for predictions
        dataloader: Dataloader to get samples from
        device: Device to run model on
        num_samples: Number of samples to visualize
        save_path: Path to save figure (None = show only)
    """
    model.eval()
    
    # Get a batch
    batch = next(iter(dataloader))
    if isinstance(batch, dict):
        images = batch['image']
        # If it is (unexpectedly) a list of tensors, stack them
        if isinstance(images, list):
            images = torch.stack(images)
        # If it is (unexpectedly) a list of NumPy arrays, convert it
        if isinstance(images, np.ndarray) and images.dtype.type is np.str_:
            raise ValueError("batch['image'] contiene stringhe, non tensori!")
        images = images.to(device)
        masks = batch['mask'].to(device)
    else:
        images, masks = batch
        images = images.to(device)
        masks = masks.to(device)
    
    # Debug info
    print("images type:", type(images))
    print("images dtype:", images.dtype)
    print("images device:", images.device)
    print("images shape:", images.shape)
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
    
    # Move to CPU
    images = images.cpu()
    masks = masks.cpu()
    preds = preds.cpu()
    
    # Get colormap
    cmap, norm = get_cityscapes_colormap()
    
    # Create figure
    num_samples = min(num_samples, images.size(0))
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Original image
        img = denormalize_image(images[i])
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Sample {i+1} - Image')
        axes[i, 0].axis('off')
        
        # Ground truth mask
        mask = masks[i].numpy().astype(np.int32)
        mask_ma = np.ma.masked_where(mask == 255, mask)
        axes[i, 1].imshow(mask_ma, cmap=cmap, norm=norm, interpolation='nearest')
        axes[i, 1].set_title(f'Sample {i+1} - Ground Truth')
        axes[i, 1].axis('off')
        
        # Predicted mask
        pred = preds[i].numpy().astype(np.int32)
        pred_ma = np.ma.masked_where(pred == 255, pred)
        axes[i, 2].imshow(pred_ma, cmap=cmap, norm=norm, interpolation='nearest')
        axes[i, 2].set_title(f'Sample {i+1} - Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(history, save_path=None):
    """
    Plot training history.
    
    Args:
        history: Dictionary with training history
        save_path: Path to save figure (None = show only)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Validation loss
    axes[0, 1].plot(history['val_loss'], label='Val Loss', color='orange')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Pixel accuracy
    axes[1, 0].plot(history['val_pixel_acc'], label='Pixel Accuracy', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Validation Pixel Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Mean IoU
    axes[1, 1].plot(history['val_miou'], label='Mean IoU', color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('mIoU')
    axes[1, 1].set_title('Validation Mean IoU')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Training history plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

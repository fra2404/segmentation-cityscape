"""DeepLabV3/DeepLabV3+ model creation and configuration."""

import os
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101
import segmentation_models_pytorch as smp


def create_model(num_classes=19, pretrained=True, device='mps', architecture='deeplabv3plus'):
    """
    Create DeepLabV3 or DeepLabV3+ model with ResNet101 backbone.
    
    Args:
        num_classes: Number of output classes (default: 19 for Cityscapes)
        pretrained: Whether to use pretrained weights (default: True)
        device: Device to place model on ('mps', 'cuda', or 'cpu')
        architecture: Model architecture - 'deeplabv3' or 'deeplabv3plus' (default: 'deeplabv3plus')
        
    Returns:
        Model ready for training/inference and device
    """
    print(f"Creating {architecture.upper()} model with ResNet101 backbone...")
    print(f"Number of classes: {num_classes}")
    print(f"Pretrained: {pretrained}")
    
    # Select device
    if device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    if architecture.lower() == 'deeplabv3plus':
        # Use segmentation-models-pytorch for DeepLabV3+
        model = smp.DeepLabV3Plus(
            encoder_name='resnet101',
            encoder_weights=None,  # We'll load weights manually to avoid Hugging Face issues
            in_channels=3,
            classes=num_classes,
        )
        if pretrained:
            # Load ResNet101 weights from local cache to avoid downloads
            import torchvision.models as models
            resnet101 = models.resnet101(weights=None)
            # Load from existing cached file (choose the latest one)
            cache_dir = torch.hub.get_dir()
            checkpoint_path = f"{cache_dir}/checkpoints/resnet101-63fe2227.pth"  # Latest cached version
            if not os.path.exists(checkpoint_path):
                checkpoint_path = f"{cache_dir}/checkpoints/resnet101-5d3b4d8f.pth"  # Fallback
            if os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                resnet101.load_state_dict(state_dict)
                print(f"✅ ResNet101 pretrained weights loaded from: {checkpoint_path}")
            else:
                print("⚠️ No cached ResNet101 weights found, loading from torchvision (may download)")
                resnet101 = models.resnet101(weights='DEFAULT')
            model.encoder.load_state_dict(resnet101.state_dict())
    elif architecture.lower() == 'deeplabv3':
        # Use torchvision for DeepLabV3
        if pretrained:
            model = deeplabv3_resnet101(weights='DEFAULT')
        else:
            model = deeplabv3_resnet101(weights=None)
        
        # Modify classifier for correct number of classes
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Choose 'deeplabv3' or 'deeplabv3plus'")
    
    model = model.to(device)
    
    print(f"Model created and moved to {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model, device


def load_checkpoint(model, checkpoint_path, device):
    """
    Load model from checkpoint.
    
    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on
        
    Returns:
        Model with loaded weights
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print("✅ Checkpoint loaded successfully")
    return model


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save model checkpoint.
    
    Args:
        model: Model instance
        optimizer: Optimizer instance
        epoch: Current epoch
        loss: Current loss value
        path: Path to save checkpoint
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"✅ Checkpoint saved to {path}")

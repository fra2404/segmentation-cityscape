"""DeepLabV3/DeepLabV3+ model creation and configuration."""

import os
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101
import segmentation_models_pytorch as smp
from torchvision import models


def create_model(num_classes=19, pretrained=True, device='mps', architecture='deeplabv3plus', load_weights_path=None, backbone='resnet50'):
    """
    Create DeepLabV3 or DeepLabV3+ model with ResNet backbone.

    Args:
        num_classes: Number of output classes (default: 19 for Cityscapes)
        pretrained: Whether to use pretrained weights (default: True)
        device: Device to place model on ('mps', 'cuda', or 'cpu')
        architecture: Model architecture - 'deeplabv3' or 'deeplabv3plus' (default: 'deeplabv3plus')
        load_weights_path: Path to load model weights from (optional)
        backbone: Backbone architecture - 'resnet50' or 'resnet101' (default: 'resnet50')

    Returns:
        Model ready for training/inference and device
    """
    print(f"Creating {architecture.upper()} model with {backbone.upper()} backbone...")
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
        # Create DeepLabV3+ with ResNet backbone
        model = create_deeplabv3plus_resnet50(num_classes=num_classes, pretrained=pretrained)
    elif architecture.lower() == 'deeplabv3':
        # Use torchvision for DeepLabV3 with ResNet50 and COCO pretrained weights
        from torchvision.models.segmentation import deeplabv3_resnet50
        
        if pretrained:
            # Use COCO pretrained weights (same as COCO_WITH_VOC_LABELS_V1)
            print("✅ Loading DeepLabV3 ResNet50 with COCO pretrained weights")
            model = deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
        else:
            model = deeplabv3_resnet50(weights=None)
        
        # Modify classifier for correct number of classes if needed
        if num_classes != 21:  # COCO+VOC has 21 classes
            model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
            model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
            print(f"✅ Modified classifier for {num_classes} classes")
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Choose 'deeplabv3' or 'deeplabv3plus'")
    
    model = model.to(device)
    
    # Load weights if specified
    if load_weights_path:
        print(f"Loading model weights from {load_weights_path}...")
        checkpoint = torch.load(load_weights_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Model weights loaded successfully")
    
    print(f"Model created and moved to {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model, device


def create_deeplabv3plus_resnet50(num_classes=19, pretrained=True):
    """
    Create DeepLabV3+ model with ResNet50 backbone using torchvision.
    This avoids the Hugging Face download issues.
    """
    # Load ResNet50 backbone
    backbone = models.resnet50(weights='DEFAULT' if pretrained else None)

    # Create ASPP module
    class ASPP(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(ASPP, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu1 = nn.ReLU()

            self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu2 = nn.ReLU()

            self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels)
            self.relu3 = nn.ReLU()

            self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
            self.bn4 = nn.BatchNorm2d(out_channels)
            self.relu4 = nn.ReLU()

            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.conv5 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.bn5 = nn.BatchNorm2d(out_channels)
            self.relu5 = nn.ReLU()

            self.conv6 = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
            self.bn6 = nn.BatchNorm2d(out_channels)
            self.relu6 = nn.ReLU()

        def forward(self, x):
            size = x.shape[2:]
            conv1_out = self.relu1(self.bn1(self.conv1(x)))
            conv2_out = self.relu2(self.bn2(self.conv2(x)))
            conv3_out = self.relu3(self.bn3(self.conv3(x)))
            conv4_out = self.relu4(self.bn4(self.conv4(x)))

            global_pool = self.global_avg_pool(x)
            conv5_out = self.relu5(self.bn5(self.conv5(global_pool)))
            conv5_out = nn.functional.interpolate(conv5_out, size=size, mode='bilinear', align_corners=False)

            out = torch.cat([conv1_out, conv2_out, conv3_out, conv4_out, conv5_out], dim=1)
            out = self.relu6(self.bn6(self.conv6(out)))
            return out

    # Extract layers from ResNet50
    layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
    layer1 = backbone.layer1
    layer2 = backbone.layer2
    layer3 = backbone.layer3
    layer4 = backbone.layer4

    # Modify layer4 for output_stride=8 (remove last stride)
    layer4[0].conv2.stride = (1, 1)
    layer4[0].downsample[0].stride = (1, 1)

    # Create DeepLabV3+ model
    class DeepLabV3Plus(nn.Module):
        def __init__(self, num_classes):
            super(DeepLabV3Plus, self).__init__()
            self.layer0 = layer0
            self.layer1 = layer1
            self.layer2 = layer2
            self.layer3 = layer3
            self.layer4 = layer4

            self.aspp = ASPP(2048, 256)

            # Decoder
            self.conv1 = nn.Conv2d(256, 48, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(48)
            self.relu1 = nn.ReLU()

            self.conv2 = nn.Conv2d(304, 256, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(256)
            self.relu2 = nn.ReLU()

            self.conv3 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(256)
            self.relu3 = nn.ReLU()

            self.conv4 = nn.Conv2d(256, num_classes, 1)

        def forward(self, x):
            # Encoder
            x0 = self.layer0(x)  # 1/4
            x1 = self.layer1(x0)  # 1/4
            x2 = self.layer2(x1)  # 1/8
            x3 = self.layer3(x2)  # 1/16
            x4 = self.layer4(x3)  # 1/16 (modified for stride 8)

            # ASPP
            aspp_out = self.aspp(x4)

            # Decoder
            low_level_features = self.relu1(self.bn1(self.conv1(x1)))  # 1/4 -> 48 channels

            # Upsample ASPP output
            aspp_upsampled = nn.functional.interpolate(aspp_out, size=low_level_features.shape[2:],
                                                     mode='bilinear', align_corners=False)

            # Concatenate
            concat = torch.cat([aspp_upsampled, low_level_features], dim=1)

            decoder_out = self.relu2(self.bn2(self.conv2(concat)))
            decoder_out = self.relu3(self.bn3(self.conv3(decoder_out)))

            # Final classification
            out = self.conv4(decoder_out)

            # Final upsampling to input size
            out = nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
            return out

    return DeepLabV3Plus(num_classes)


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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
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

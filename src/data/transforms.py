"""Data transformations and Cityscapes label mappings."""

import torch
from torchvision import transforms

# Albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def to_train_id(mask_tensor, use_all_classes=False):
    """
    Map Cityscapes labelIds to trainIds or map all labelIds to consecutive 0-33.
    
    Args:
        mask_tensor: Tensor with shape (1, H, W) containing labelIds
        use_all_classes: If True, map all 34 labelIds to consecutive 0-33; if False, map to 19 trainIds
        
    Returns:
        Tensor with shape (H, W) containing consecutive class indices
    """
    mask_tensor = mask_tensor.squeeze(0).long()  # (H, W)
    mapping = torch.full((256,), 255, dtype=torch.long)  # default ignore
    
    if use_all_classes:
        # Map all 34 Cityscapes labelIds to consecutive indices 0-33
        # Cityscapes labelIds: 0-33 but non-consecutive (e.g., missing 14, 15, 16, 29, 30)
        cityscapes_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
                            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 
                            31, 32, 33, -1, -1, -1, -1, -1]  # 34 classes total (some are void/unused)
        for consecutive_idx, label_id in enumerate(cityscapes_labels):
            if label_id >= 0:
                mapping[label_id] = consecutive_idx
        return mapping[mask_tensor]
    
    # Standard 19-class mapping
    mapping = torch.full((256,), 255, dtype=torch.long)  # default ignore
    
    # Official Cityscapes mapping (labelId -> trainId)
    mapping[7] = 0    # road
    mapping[8] = 1    # sidewalk
    mapping[11] = 2   # building
    mapping[12] = 3   # wall
    mapping[13] = 4   # fence
    mapping[17] = 5   # pole
    mapping[19] = 6   # traffic light
    mapping[20] = 7   # traffic sign
    mapping[21] = 8   # vegetation
    mapping[22] = 9   # terrain
    mapping[23] = 10  # sky
    mapping[24] = 11  # person
    mapping[25] = 12  # rider
    mapping[26] = 13  # car
    mapping[27] = 14  # truck
    mapping[28] = 15  # bus
    mapping[31] = 16  # train
    mapping[32] = 17  # motorcycle
    mapping[33] = 18  # bicycle
    
    return mapping[mask_tensor]


def trainid_to_labelid(trainid_tensor):
    """
    Map trainIds (0..18) back to labelIds for submission.
    
    Args:
        trainid_tensor: Tensor with trainIds
        
    Returns:
        Tensor with labelIds
    """
    mapping = torch.full((256,), 0, dtype=torch.long)  # default to void
    
    # Reverse mapping (trainId -> labelId)
    mapping[0] = 7    # road
    mapping[1] = 8    # sidewalk
    mapping[2] = 11   # building
    mapping[3] = 12   # wall
    mapping[4] = 13   # fence
    mapping[5] = 17   # pole
    mapping[6] = 19   # traffic light
    mapping[7] = 20   # traffic sign
    mapping[8] = 21   # vegetation
    mapping[9] = 22   # terrain
    mapping[10] = 23  # sky
    mapping[11] = 24  # person
    mapping[12] = 25  # rider
    mapping[13] = 26  # car
    mapping[14] = 27  # truck
    mapping[15] = 28  # bus
    mapping[16] = 31  # train
    mapping[17] = 32  # motorcycle
    mapping[18] = 33  # bicycle
    mapping[255] = 0  # ignore -> void
    
    return mapping[trainid_tensor]


def get_train_transforms(image_size=(256, 256)):
    """
    Get training transforms with data augmentation.
    
    Args:
        image_size: Tuple of (height, width)
        
    Returns:
        Composed transforms for training images
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(image_size=(256, 256)):
    """
    Get validation transforms (no augmentation).
    
    Args:
        image_size: Tuple of (height, width)
        
    Returns:
        Composed transforms for validation images
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_target_transform(image_size=(256, 256), use_all_classes=False):
    """
    Get target (mask) transforms.
    
    Args:
        image_size: Tuple of (height, width)
        use_all_classes: If True, use all 34 Cityscapes classes; if False, use 19 trainId classes
        
    Returns:
        Composed transforms for segmentation masks
    """
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
        transforms.Lambda(lambda x: to_train_id(x, use_all_classes=use_all_classes))
    ])


# ----------------------------
# Albumentations
# ----------------------------

def get_train_transforms_albu(image_size=(512, 512)):
    """Albumentations training transforms."""
    height, width = image_size
    # Use integer values for interpolation to avoid Pylint error
    INTER_LINEAR = 1
    return A.Compose([
        A.Resize(height=height, width=width, interpolation=INTER_LINEAR),
        A.RandomScale(scale_limit=(0.5, 2.0), p=1.0),
        A.RandomCrop(height=height, width=width),
        A.HorizontalFlip(p=0.5),

        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussianBlur(blur_limit=(3, 3), p=0.1),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
        ToTensorV2()
    ])


def get_val_transforms_albu(image_size=(512, 512)):
    """Albumentations validation transforms (resize + normalize)."""
    height, width = image_size
    INTER_NEAREST = 0
    return A.Compose([
        A.Resize(height=height, width=width, interpolation=INTER_NEAREST),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
        ToTensorV2()
    ])

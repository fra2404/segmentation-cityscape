"""Data transformations and Cityscapes label mappings."""

import torch
from torchvision import transforms


def to_train_id(mask_tensor):
    """
    Map Cityscapes labelIds (0..33) to trainIds (0..18) with 255 as ignore.
    
    Args:
        mask_tensor: Tensor with shape (1, H, W) containing labelIds
        
    Returns:
        Tensor with shape (H, W) containing trainIds
    """
    mask_tensor = mask_tensor.squeeze(0).long()  # (H, W)
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


def get_train_transforms(image_size=(512, 1024)):
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


def get_val_transforms(image_size=(512, 1024)):
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


def get_target_transform(image_size=(512, 1024)):
    """
    Get target (mask) transforms.
    
    Args:
        image_size: Tuple of (height, width)
        
    Returns:
        Composed transforms for segmentation masks
    """
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
        transforms.Lambda(to_train_id)
    ])

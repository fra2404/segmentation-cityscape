"""Cityscapes dataset loading and dataloaders creation."""

import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import datasets
from tqdm import tqdm

from .transforms import to_train_id
from .transforms import (
    get_train_transforms,
    get_val_transforms,
    get_target_transform,
    get_train_transforms_albu,
    get_val_transforms_albu,
)


class CityscapesDataset(Dataset):
    """
    Wrapper around torchvision Cityscapes dataset to support returning filenames.
    """
    def __init__(self, root, split='val', mode='fine', target_type='semantic', 
                 transform=None, target_transform=None, return_filename=False):
        """
        Args:
            root: Path to cityscapes dataset root
            split: 'train', 'val', or 'test'
            mode: 'fine' or 'coarse'
            target_type: 'semantic', 'instance', or 'polygon'
            transform: Transform to apply to images
            target_transform: Transform to apply to targets
            return_filename: Whether to return filename in the sample dict
        """
        self.dataset = datasets.Cityscapes(
            root=root,
            split=split,
            mode=mode,
            target_type=target_type,
            transform=transform,
            target_transform=target_transform
        )
        self.return_filename = return_filename
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        
        if self.return_filename:
            # Extract filename from the image path
            image_path = self.dataset.images[idx]
            filename = os.path.basename(image_path)

            # Ensure image is a tensor
            if isinstance(image, Image.Image):
                import torchvision.transforms as T
                image = T.ToTensor()(image)
            # Ensure mask is a tensor (if present)
            if isinstance(mask, Image.Image):
                mask = torch.from_numpy(np.array(mask)).long()

            return {
                'image': image,
                'mask': mask,
                'filename': filename
            }
        else:
            # Ensure image and mask are tensors
            if isinstance(image, Image.Image):
                import torchvision.transforms as T
                image = T.ToTensor()(image)
            if isinstance(mask, Image.Image):
                mask = torch.from_numpy(np.array(mask)).long()
            return image, mask


class CityscapesRawDataset(Dataset):
    """
    Custom loader: reads raw files from
    leftImg8bit/ and gtFine/ folders and applies Albumentations transforms
    that handle image+mask jointly.
    """

    def __init__(self, root, split='train', transforms=None):
        self.root = root
        self.split = split
        self.transforms = transforms

        self.image_dir = os.path.join(root, 'leftImg8bit', split)
        self.mask_dir = os.path.join(root, 'gtFine', split)

        self.image_list = []
        for city in os.listdir(self.image_dir):
            city_dir = os.path.join(self.image_dir, city)
            for fname in os.listdir(city_dir):
                if fname.endswith('_leftImg8bit.png'):
                    self.image_list.append(os.path.join(city, fname))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        rel_path = self.image_list[idx]
        city, image_name = rel_path.split('/')

        image_path = os.path.join(self.image_dir, city, image_name)

        # First look for labelTrainIds, then for labelIds if labelTrainIds does not exist
        mask_name_trainids = image_name.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
        mask_name_labelids = image_name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
        mask_path_trainids = os.path.join(self.mask_dir, city, mask_name_trainids)
        mask_path_labelids = os.path.join(self.mask_dir, city, mask_name_labelids)

        if os.path.exists(mask_path_trainids):
            mask_path = mask_path_trainids
        elif os.path.exists(mask_path_labelids):
            mask_path = mask_path_labelids
        else:
            raise FileNotFoundError(f"Mask not found for {image_name} in {city}")


        image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(Image.open(mask_path), dtype=np.uint8)

        # If the mask is labelIds, convert to trainId
        if 'labelIds' in mask_path:
            mask = to_train_id(torch.from_numpy(mask).unsqueeze(0)).numpy()

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()

        return image, mask


def compute_dataset_stats(dataset, num_classes=19, max_samples=None):
    """
    Compute class distribution statistics for weighted sampling.
    
    Args:
        dataset: Cityscapes dataset
        num_classes: Number of classes (default: 19)
        max_samples: Maximum number of samples to process (None = all)
        
    Returns:
        Tuple of (class_pixel_counts, images_with_class, total_images)
    """
    class_pixel_counts = torch.zeros(num_classes, dtype=torch.long)
    images_with_class = torch.zeros(num_classes, dtype=torch.long)
    total_images = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    
    for idx in tqdm(range(total_images), desc="Computing dataset stats", position=0, leave=False):
        _, mask = dataset[idx]
        mask_cpu = mask.cpu()
        for cls in range(num_classes):
            present = (mask_cpu == cls)
            count = present.sum().item()
            class_pixel_counts[cls] += count
            if count > 0:
                images_with_class[cls] += 1
                
    return class_pixel_counts, images_with_class, total_images


def create_weighted_sampler(dataset, num_classes=19, max_samples=None):
    """
    Create a weighted sampler to oversample rare classes.
    
    Args:
        dataset: Cityscapes dataset
        num_classes: Number of classes (default: 19)
        max_samples: Maximum number of samples for stats (None = all)
        
    Returns:
        tuple: (WeightedRandomSampler, dataset_stats) - stats to reuse in trainer
    """
    class_pixel_counts, images_with_class, total_images = compute_dataset_stats(
        dataset, num_classes, max_samples
    )
    dataset_stats = (class_pixel_counts, images_with_class, total_images)
    
    print("\n=== DATASET STATS SUMMARY ===")
    for cls in range(num_classes):
        print(f"Class {cls}: images={images_with_class[cls].item()}, "
              f"pixels={class_pixel_counts[cls].item()}")
    
    # Compute rarity scores (inverse frequency)
    rarity = 1.0 / (images_with_class.float() + 1e-6)
    
    # Assign weight to each sample based on classes present
    sample_weights = torch.zeros(len(dataset), dtype=torch.float)
    for idx in tqdm(range(len(dataset)), desc="Building sample weights", position=0, leave=False):
        _, mask = dataset[idx]
        classes_present = torch.unique(mask)
        classes_present = classes_present[(classes_present >= 0) & (classes_present < num_classes)]
        
        if classes_present.numel() > 0:
            sample_weights[idx] = rarity[classes_present].sum().item()
        else:
            sample_weights[idx] = rarity.mean().item()
    
    sampler = WeightedRandomSampler(
        sample_weights, 
        num_samples=len(dataset), 
        replacement=True
    )
    print("✅ Weighted sampler configured to favor rarer classes")
    
    return sampler, dataset_stats


def create_dataloaders(
    root='./data/cityscapes',
    batch_size=2,
    image_size=(512, 1024),
    num_workers=0,
    use_weighted_sampler=True,
    max_samples_for_stats=None,
    filter_city=None,
    use_all_classes=False,
    use_albumentations=True,
):
    """
    Create train and validation dataloaders for Cityscapes.
    
    Args:
        root: Path to cityscapes dataset root
        batch_size: Batch size for training and validation
        image_size: Tuple of (height, width)
        num_workers: Number of data loading workers
        use_weighted_sampler: Whether to use weighted sampling for training
        max_samples_for_stats: Max samples for computing stats (None = all)
        filter_city: Filter validation set to specific city (e.g., 'frankfurt')
        use_all_classes: If True, use all 34 Cityscapes classes; if False, use 19 trainId classes
        
    Returns:
        Tuple of (train_loader, val_loader, train_dataset, val_dataset, dataset_stats)
    """
    print(f"Loading Cityscapes dataset from {root}...")

    if use_albumentations:
        # Albumentations on raw files (trainId masks already encoded)
        train_transform = get_train_transforms_albu(image_size)
        val_transform = get_val_transforms_albu(image_size)

        train_dataset = CityscapesRawDataset(root=root, split='train', transforms=train_transform)
        val_dataset = CityscapesRawDataset(root=root, split='val', transforms=val_transform)
    else:
        # Fallback to torchvision pipeline (kept for compatibility)
        train_transform = get_train_transforms(image_size)
        val_transform = get_val_transforms(image_size)
        target_transform = get_target_transform(image_size, use_all_classes=use_all_classes)

        train_dataset = datasets.Cityscapes(
            root=root,
            split='train',
            mode='fine',
            target_type='semantic',
            transform=train_transform,
            target_transform=target_transform
        )

        val_dataset = datasets.Cityscapes(
            root=root,
            split='val',
            mode='fine',
            target_type='semantic',
            transform=val_transform,
            target_transform=target_transform
        )
    
    # Filter validation set by city if requested
    if filter_city:
        original_len = len(val_dataset)
        frankfurt_indices = [
            i for i, img_path in enumerate(val_dataset.images) 
            if filter_city in img_path
        ]
        val_dataset.images = [val_dataset.images[i] for i in frankfurt_indices]
        val_dataset.targets = [val_dataset.targets[i] for i in frankfurt_indices]
        print(f"Filtered val_dataset to {filter_city}: {len(val_dataset)} samples (was {original_len})")
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create weighted sampler for training if requested 
    sampler = None
    shuffle = True
    dataset_stats = None
    if use_albumentations:
        use_weighted_sampler = False
    if use_weighted_sampler:
        num_classes = 34 if use_all_classes else 19
        sampler, dataset_stats = create_weighted_sampler(
            train_dataset, num_classes=num_classes, max_samples=max_samples_for_stats
        )
        shuffle = False  # Sampler and shuffle are mutually exclusive
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch to avoid batch norm issues
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader, train_dataset, val_dataset, dataset_stats

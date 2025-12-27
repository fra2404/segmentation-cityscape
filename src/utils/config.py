"""Configuration settings."""

from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class Config:
    """Configuration for Cityscapes semantic segmentation."""
    
    # Data settings
    data_root: str = './data/cityscapes'
    image_size: Tuple[int, int] = (512, 1024)
    batch_size: int = 4
    num_workers: int = 0
    filter_city: Optional[str] = None  # Optional city filter for validation
    
    # Training settings
    num_epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_accumulation_steps: int = 1
    max_train_batches: Optional[int] = None  # Limit training batches for quick testing
    
    # Weighted sampling
    use_weighted_sampler: bool = False  
    max_samples_for_stats: Optional[int] = None  # None = use all samples
    
    # Loss and scheduler settings
    use_class_weights: bool = False  
    scheduler: str = 'poly'  # 'poly' or 'cosine'
    min_lr: float = 1e-6  # Minimum LR for cosine scheduler
    
    # Model settings
    num_classes: int = 19
    pretrained: bool = True
    architecture: str = 'deeplabv3plus'  # 'deeplabv3' or 'deeplabv3plus'
    
    # Device settings
    device: str = 'mps'  # 'mps', 'cuda', or 'cpu'
    
    # Checkpoint settings
    checkpoint_dir: str = './checkpoints'
    load_checkpoint: Optional[str] = None  # Path to checkpoint to load
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'data_root': self.data_root,
            'image_size': self.image_size,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'filter_city': self.filter_city,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'max_train_batches': self.max_train_batches,
            'use_weighted_sampler': self.use_weighted_sampler,
            'max_samples_for_stats': self.max_samples_for_stats,
            'use_class_weights': self.use_class_weights,
            'scheduler': self.scheduler,
            'min_lr': self.min_lr,
            'num_classes': self.num_classes,
            'pretrained': self.pretrained,
            'architecture': self.architecture,
            'device': self.device,
            'checkpoint_dir': self.checkpoint_dir,
            'load_checkpoint': self.load_checkpoint,
        }
    
    def print_config(self):
        """Print configuration in a formatted way."""
        print("\n" + "="*60)
        print("CONFIGURATION")
        print("="*60)
        for key, value in self.to_dict().items():
            print(f"{key:30s}: {value}")
        print("="*60 + "\n")

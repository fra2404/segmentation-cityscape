"""Training logic and trainer class."""

import torch
import torch.nn as nn
from tqdm import tqdm
import os

from ..evaluation.metrics import evaluate_model, print_evaluation_results
from ..models.deeplabv3 import save_checkpoint


class Trainer:
    """Trainer class for semantic segmentation model."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        learning_rate=1e-3,
        weight_decay=1e-5,
        gradient_accumulation_steps=2,
        checkpoint_dir='./checkpoints'
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training dataloader
            val_loader: Validation dataloader
            device: Device to train on
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            gradient_accumulation_steps: Steps for gradient accumulation
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=2,
            gamma=0.1
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_pixel_acc': [],
            'val_miou': []
        }
        
        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"  Checkpoint directory: {checkpoint_dir}")
    
    def train_one_epoch(self, epoch, max_batches=None):
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            max_batches: Maximum number of batches to process (None = all)
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        
        # Force batch norm to eval mode to avoid issues with small batches
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        
        total_loss = 0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        # Set progress bar total to max_batches if specified
        total_batches = max_batches if max_batches is not None else len(self.train_loader)
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", total=total_batches)
        for batch_idx, (images, masks) in enumerate(pbar):
            # Stop if max_batches reached
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Validate masks are in correct range
            unique_values = torch.unique(masks)
            if torch.any((unique_values > 18) & (unique_values != 255)):
                print(f"\n⚠️  Warning: Found invalid mask values: {unique_values.tolist()}")
            
            # Forward pass
            outputs = self.model(images)
            # Handle different model outputs (DeepLabV3 returns dict, DeepLabV3+ returns tensor)
            if isinstance(outputs, dict):
                outputs = outputs['out']
            loss = self.criterion(outputs, masks)
            
            # Check for NaN/inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️  Warning: NaN/inf loss detected at batch {batch_idx}")
                print(f"Output stats: min={outputs.min():.4f}, max={outputs.max():.4f}")
                print(f"Mask unique values: {unique_values.tolist()}")
                continue
            
            # Normalize loss by gradient accumulation steps
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights after accumulation steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Track loss (correctly - loss.item() is already normalized)
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar (denormalize for display)
            current_loss = loss.item() * self.gradient_accumulation_steps
            pbar.set_postfix({'loss': f'{current_loss:.4f}'})
        
        # Final optimizer step if there are remaining gradients
        if num_batches % self.gradient_accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def validate(self):
        """
        Validate model on validation set.
        
        Returns:
            Tuple of (val_loss, metrics)
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    outputs = outputs['out']
                loss = self.criterion(outputs, masks)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Compute metrics
        metrics = evaluate_model(self.model, self.val_loader, self.device)
        
        return avg_loss, metrics
    
    def train(self, num_epochs, max_train_batches=None):
        """
        Train model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            max_train_batches: Maximum training batches per epoch (None = all)
        """
        if max_train_batches:
            print(f"\nStarting training for {num_epochs} epochs (max {max_train_batches} batches/epoch)...")
        else:
            print(f"\nStarting training for {num_epochs} epochs...")
        print("="*60)
        
        best_miou = 0.0
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss = self.train_one_epoch(epoch, max_batches=max_train_batches)
            self.history['train_loss'].append(train_loss)
            
            print(f"Training Loss: {train_loss:.4f}")
            
            # Validate
            val_loss, metrics = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['val_pixel_acc'].append(metrics['pixel_accuracy'])
            self.history['val_miou'].append(metrics['mean_iou'])
            
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
            print(f"Mean IoU: {metrics['mean_iou']:.4f}")
            
            # Step learning rate scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f'checkpoint_epoch_{epoch}.pth'
            )
            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                train_loss,
                checkpoint_path
            )
            
            # Save best model
            if metrics['mean_iou'] > best_miou:
                best_miou = metrics['mean_iou']
                best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    train_loss,
                    best_path
                )
                print(f"✅ New best model saved (mIoU: {best_miou:.4f})")
        
        print("\n" + "="*60)
        print("Training completed!")
        print(f"Best mIoU: {best_miou:.4f}")
        print("="*60)
        
        return self.history

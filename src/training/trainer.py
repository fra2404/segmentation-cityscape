"""Robust Trainer for semantic segmentation
Features:
- AMP in training and validation (CUDA/MPS/CPU safe)
- Warmup + Poly LR
- Gradient clipping
- Class weights optional
- Checkpoint save/load (model, optimizer, scaler, epoch, history)
- Resume training from checkpoint
- Early stopping
- Loss spike detection (debug checkpoint + skip)
- torch.cuda.empty_cache() cleanup after validation
"""

import os
import math
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from contextlib import nullcontext
from typing import Optional, Dict, Any, Tuple

from ..evaluation.metrics import evaluate_model
from ..data.dataset import compute_dataset_stats
from .losses import CombinedLoss


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        learning_rate: float = 3e-4,
        lr_backbone: float = 2e-5,
        lr_head: float = 3e-4,
        weight_decay: float = 1e-4,
        gradient_accumulation_steps: int = 1,
        checkpoint_dir: str = './checkpoints',
        use_class_weights: bool = False,
        dataset_stats=None,
        max_samples_for_stats: Optional[int] = None,
        num_classes: int = 19,
        use_amp: bool = True,
        clip_grad_norm: float = 1.0,
        warmup_epochs: int = 0,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 1e-4,
        loss_spike_threshold: Optional[float] = 100.0,  # if loss > threshold -> skip batch and save debug ckpt
        image_size: Tuple[int, int] = (512, 512),
        scheduler_type: str = 'poly',  # 'poly' or 'cosine'
        min_lr: float = 1e-6,  # For cosine scheduler
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.checkpoint_dir = checkpoint_dir
        self.clip_grad_norm = clip_grad_norm
        self.use_amp = use_amp
        self.warmup_epochs = warmup_epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.loss_spike_threshold = loss_spike_threshold
        self.image_size = image_size

        os.makedirs(checkpoint_dir, exist_ok=True)

        # ----- class weights / loss -----
        self.class_weights = None
        if use_class_weights:
            try:
                # Use precomputed stats if available, otherwise compute
                if dataset_stats is not None:
                    print("✅ Using precomputed dataset stats for class weights")
                    _, images_with_class, _ = dataset_stats
                else:
                    print("Computing dataset stats for class weights...")
                    _, images_with_class, _ = compute_dataset_stats(
                        self.train_loader.dataset, num_classes=num_classes, max_samples=max_samples_for_stats
                    )
                epsilon = 1e-6
                inv_freq = 1.0 / (images_with_class.float() + epsilon)
                weights = inv_freq / inv_freq.mean()
                # Move weights to device for MPS compatibility
                self.class_weights = weights.to(device)
                print("✅ Using class weighting for loss (rare classes upweighted)")
                print(f"Class weights: {[round(w.item(), 3) for w in self.class_weights]}")
                self.criterion = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=255)
            except Exception as e:
                print(f"⚠️ Failed to compute class weights, falling back to unweighted loss: {e}")
                self.class_weights = None
                self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        else:
            print("✅ Using standard CrossEntropyLoss (no class weights)")
            self.criterion = nn.CrossEntropyLoss(ignore_index=255)

        # ----- optimizer with param groups (backbone vs head) -----
        def _build_param_groups():
            backbone_prefixes = ('layer0', 'layer1', 'layer2', 'layer3', 'layer4')
            backbone_params, head_params = [], []

            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if name.startswith(backbone_prefixes):
                    backbone_params.append(param)
                else:
                    head_params.append(param)

            groups = []
            if backbone_params:
                groups.append({'params': backbone_params, 'lr': lr_backbone, 'weight_decay': weight_decay})
            if head_params:
                groups.append({'params': head_params, 'lr': lr_head, 'weight_decay': weight_decay})

            # Fallback: single group
            if not groups:
                groups.append({'params': self.model.parameters(), 'lr': learning_rate, 'weight_decay': weight_decay})
            return groups

        param_groups = _build_param_groups()

        # weight_decay handled per-group; set global WD to 0 to avoid double
        self.optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, weight_decay=0.0)

        # ----- device type & AMP scaler -----
        dev_type = 'cuda' if device.type == 'cuda' else ('mps' if device.type == 'mps' else 'cpu')
        self._device_type = dev_type
        self.scaler = None
        # Disable AMP for MPS - causes instability
        if self._device_type == 'mps':
            self.use_amp = False
            print("⚠️  AMP disabled for MPS device (stability)")
        elif self.use_amp and self._device_type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()

        # ----- LR scheduling params (poly + warmup) -----
        self._base_lr = learning_rate
        self._max_epoch = None
        self._poly_power = 0.9
        self._warmup_epochs = max(0, int(warmup_epochs))
        self.scheduler_type = scheduler_type
        self.cosine_scheduler = None
        self.min_lr = min_lr

        # history & bookkeeping
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_pixel_acc': [],
            'val_miou': []
        }

        print("Trainer initialized:")
        print(f"  Device: {device} (interpreted as {self._device_type})")
        print(f"  LR backbone: {lr_backbone} | LR head: {lr_head}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Grad accumulation: {gradient_accumulation_steps}")
        print(f"  Checkpoint dir: {checkpoint_dir}")
        print(f"  AMP enabled: {self.use_amp} (scaler present: {self.scaler is not None})")
        print(f"  Clip grad norm: {self.clip_grad_norm}")
        print(f"  Warmup epochs: {self._warmup_epochs}")
        print(f"  Early stopping patience: {self.early_stopping_patience}")

    # ---------------- helper: autocast context ----------------
    def _amp_context(self):
        # returns a context manager for autocast or nullcontext
        if self.scaler is not None:
            return torch.cuda.amp.autocast()
        if self.use_amp and self._device_type == 'mps':
            # try using torch.autocast for mps if available
            try:
                return torch.autocast(device_type='mps', dtype=torch.float16)
            except Exception:
                return nullcontext()
        if self.use_amp and self._device_type == 'cpu':
            # CPU autocast may use bfloat16 depending on hardware; keep safe
            try:
                return torch.autocast(device_type='cpu', dtype=torch.bfloat16)
            except Exception:
                return nullcontext()
        return nullcontext()

    # ---------------- core: train one epoch ----------------
    def train_one_epoch(self, epoch: int, max_batches: Optional[int] = None) -> float:
        self.model.train()
        # freeze BatchNorm running stats to avoid instability with small batches
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        total_loss = 0.0
        seen_batches = 0
        self.optimizer.zero_grad()

        total_batches = max_batches if max_batches is not None else len(self.train_loader)
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", total=total_batches, position=0, leave=False)

        amp_ctx = self._amp_context()

        for batch_idx, (images, masks) in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            # safety check masks
            uv = torch.unique(masks)
            if torch.any((uv > 18) & (uv != 255)):
                print(f"\n⚠️ Invalid mask values in batch {batch_idx}: {uv.tolist()}")

            with amp_ctx:
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    outputs = outputs.get('out', outputs)
                loss = self.criterion(outputs, masks)

            # Check for NaN/Inf immediately
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️  NaN/Inf loss at batch {batch_idx}, skipping")
                continue

            # Detect spike before scaling/accumulation
            if self.loss_spike_threshold is not None and loss.item() > self.loss_spike_threshold:
                # save debug checkpoint and skip this batch (avoid corruption)
                debug_path = os.path.join(self.checkpoint_dir, f"debug_spike_epoch{epoch}_batch{batch_idx}.pth")
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'loss': loss.item()
                }, debug_path)
                print(f"\n❌ Loss spike detected ({loss.item():.3f}) - saved debug checkpoint to {debug_path} and skipping batch")
                continue

            # normalize loss for gradient accumulation
            loss = loss / float(self.gradient_accumulation_steps)

            # backward (AMP-aware)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # step if enough accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # unscale & clip when using scaler
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                if self.clip_grad_norm and self.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            seen_batches += 1
            pbar.set_postfix({'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}'})

        # final step if accumulation left
        if seen_batches % self.gradient_accumulation_steps != 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            if self.clip_grad_norm and self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

        avg_loss = total_loss / seen_batches if seen_batches > 0 else 0.0
        return avg_loss

    # ---------------- validation ----------------
    def validate(self) -> (float, Dict[str, float]):
        self.model.eval()
        total_loss = 0.0
        seen_batches = 0

        amp_ctx = self._amp_context()

        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Validation", position=0, leave=False):
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                with amp_ctx:
                    outputs = self.model(images)
                    if isinstance(outputs, dict):
                        outputs = outputs.get('out', outputs)
                    loss = self.criterion(outputs, masks)

                total_loss += loss.item()
                seen_batches += 1

        avg_loss = total_loss / seen_batches if seen_batches > 0 else 0.0

        # evaluate metrics (evaluate_model should accept device param)
        metrics = evaluate_model(self.model, self.val_loader, self.device, num_classes=self.num_classes)

        # free some cache if CUDA
        try:
            if self._device_type == 'cuda':
                torch.cuda.empty_cache()
        except Exception:
            pass

        return avg_loss, metrics

    # ---------------- checkpoint helpers ----------------
    def save_full_checkpoint(self, path: str, epoch: int, train_loss: float, extra: Optional[Dict[str, Any]] = None) -> None:
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'history': self.history,
            'image_size': self.image_size,
        }
        if self.scaler is not None:
            state['scaler_state_dict'] = self.scaler.state_dict()
        if extra:
            state.update(extra)
        torch.save(state, path)

    def load_checkpoint(self, path: str, load_optimizer: bool = True, load_scaler: bool = True, current_image_size: Optional[Tuple[int, int]] = None) -> int:
        """Load checkpoint and return starting epoch (next epoch)."""
        # Allow numpy objects in checkpoint for backward compatibility
        try:
            import torch.serialization
            import numpy
            with torch.serialization.safe_globals([numpy._core.multiarray.scalar, torch._C._TensorBase, torch._C._StorageBase]):
                ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except AttributeError:
            # Fallback for older PyTorch versions
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        
        checkpoint_image_size = ckpt.get('image_size')
        resolution_changed = current_image_size is not None and checkpoint_image_size != current_image_size
        
        if resolution_changed:
            print(f"⚠️ Resolution changed from {checkpoint_image_size} to {current_image_size}, resetting optimizer and history")
            load_optimizer = False
            load_scaler = False
            self.history = {
                'train_loss': [],
                'val_loss': [],
                'val_pixel_acc': [],
                'val_miou': []
            }
        
        if load_optimizer and 'optimizer_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if load_scaler and self.scaler is not None and 'scaler_state_dict' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        if 'history' in ckpt and not resolution_changed:
            self.history = ckpt['history']
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"✅ Loaded checkpoint '{path}', resuming from epoch {start_epoch}")
        return start_epoch

    # ---------------- full training loop ----------------
    def train(self, num_epochs: int, max_train_batches: Optional[int] = None, resume_from: Optional[str] = None, current_image_size: Optional[Tuple[int, int]] = None):
        print(f"\nStarting training for {num_epochs} epochs" + (f" (max {max_train_batches} batches/epoch)" if max_train_batches else ""))
        print("=" * 60)

        best_miou = 0.0
        bad_epochs = 0
        self._max_epoch = num_epochs

        start_epoch = 1
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from, current_image_size=current_image_size)

        # Initialize cosine scheduler if needed
        if self.scheduler_type == 'cosine':
            print(f"✅ Using CosineAnnealingLR scheduler (min_lr={self.min_lr})")
            self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
                eta_min=self.min_lr
            )

        for epoch in range(start_epoch, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 60)

            # Update LR based on scheduler type
            if self.scheduler_type == 'cosine' and self.cosine_scheduler is not None:
                # Cosine scheduler - just step
                if epoch > start_epoch:  # Don't step on first epoch if resuming
                    self.cosine_scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
            else:
                # Poly scheduler with warmup
                if self._warmup_epochs > 0 and epoch <= self._warmup_epochs:
                    # linear warmup from 1/10 * base_lr -> base_lr
                    warmup_factor = (epoch / float(max(1, self._warmup_epochs)))
                    current_base = self._base_lr * warmup_factor
                else:
                    current_base = self._base_lr

                # apply poly schedule relative to full horizon
                poly_lr = current_base * (1 - (epoch - 1) / float(max(1, self._max_epoch))) ** self._poly_power
                for pg in self.optimizer.param_groups:
                    pg['lr'] = max(poly_lr, 1e-8)  # safe floor
                current_lr = poly_lr
            print(f"Learning Rate: {current_lr:.8f}")

            # train + validate
            t0 = time.time()
            train_loss = self.train_one_epoch(epoch, max_batches=max_train_batches)
            t_train = time.time() - t0
            self.history['train_loss'].append(train_loss)
            print(f"Training Loss: {train_loss:.4f} (time: {t_train:.1f}s)")

            val_loss, metrics = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['val_pixel_acc'].append(metrics.get('pixel_accuracy', 0.0))
            self.history['val_miou'].append(metrics.get('mean_iou', 0.0))

            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Pixel Accuracy: {metrics.get('pixel_accuracy', 0.0):.4f}")
            print(f"Mean IoU: {metrics.get('mean_iou', 0.0):.4f}")

            ckpt_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            self.save_full_checkpoint(ckpt_path, epoch, train_loss)

            # save best
            miou = metrics.get('mean_iou', 0.0)
            if miou > best_miou + self.early_stopping_min_delta:
                best_miou = miou
                best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                self.save_full_checkpoint(best_path, epoch, train_loss)
                print(f"✅ New best model saved (mIoU: {best_miou:.4f})")
                bad_epochs = 0
            else:
                bad_epochs += 1

            # early stopping
            if self.early_stopping_patience is not None and bad_epochs >= self.early_stopping_patience:
                print(f"\n⏸️ Early stopping triggered (no improvement for {bad_epochs} epochs).")
                break

        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best mIoU: {best_miou:.4f}")
        print("=" * 60)
        return self.history
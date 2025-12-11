#!/usr/bin/env python3
"""Main training script for Cityscapes semantic segmentation."""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config import Config
from src.data.dataset import create_dataloaders
from src.models.deeplabv3 import create_model, load_checkpoint
from src.training.trainer import Trainer
from src.evaluation.metrics import evaluate_model, print_evaluation_results
from src.utils.visualization import visualize_predictions, plot_training_history


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train DeepLabV3 on Cityscapes dataset'
    )
    
    # Data arguments
    parser.add_argument('--data-root', type=str, default='./data/cityscapes',
                        help='Path to Cityscapes dataset')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--image-size', type=int, nargs=2, default=[512, 1024],
                        help='Image size (height width)')
    parser.add_argument('--filter-city', type=str, default=None,
                        help='Filter validation set to specific city (e.g., frankfurt)')
    
    # Training arguments
    parser.add_argument('--num-epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--grad-accum-steps', type=int, default=2,
                        help='Gradient accumulation steps')
    
    # Sampling arguments
    parser.add_argument('--use-weighted-sampler', action='store_true', default=True,
                        help='Use weighted sampling for training')
    parser.add_argument('--no-weighted-sampler', dest='use_weighted_sampler',
                        action='store_false',
                        help='Disable weighted sampling')
    parser.add_argument('--max-samples-stats', type=int, default=None,
                        help='Max samples for computing class stats (None = all)')
    parser.add_argument('--max-train-batches', type=int, default=None,
                        help='Max training batches per epoch (for quick testing)')
    
    # Model arguments
    parser.add_argument('--num-classes', type=int, default=19,
                        help='Number of output classes')
    parser.add_argument('--no-pretrained', action='store_false', dest='pretrained',
                        help='Do not use pretrained weights')
    parser.add_argument('--architecture', type=str, default='deeplabv3plus',
                        choices=['deeplabv3', 'deeplabv3plus'],
                        help='Model architecture (deeplabv3 or deeplabv3plus)')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='mps',
                        choices=['mps', 'cuda', 'cpu'],
                        help='Device to use for training')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--load-checkpoint', type=str, default=None,
                        help='Path to checkpoint to load')
    
    # Action arguments
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'visualize'],
                        help='Mode: train, eval, or visualize')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Create config from arguments
    config = Config(
        data_root=args.data_root,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        filter_city=args.filter_city,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.grad_accum_steps,
        max_train_batches=args.max_train_batches,
        use_weighted_sampler=args.use_weighted_sampler,
        max_samples_for_stats=args.max_samples_stats,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        architecture=args.architecture,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        load_checkpoint=args.load_checkpoint,
    )
    
    # Print configuration
    config.print_config()
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(
        root=config.data_root,
        batch_size=config.batch_size,
        image_size=config.image_size,
        num_workers=config.num_workers,
        use_weighted_sampler=config.use_weighted_sampler,
        max_samples_for_stats=config.max_samples_for_stats,
        filter_city=config.filter_city
    )
    
    # Create model
    print("\nCreating model...")
    model, device = create_model(
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        device=config.device,
        architecture=config.architecture
    )
    
    # Load checkpoint if specified
    if config.load_checkpoint:
        model = load_checkpoint(model, config.load_checkpoint, device)
    
    # Execute requested mode
    if args.mode == 'train':
        # Train model
        print("\n" + "="*60)
        print("TRAINING MODE")
        print("="*60)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            checkpoint_dir=config.checkpoint_dir
        )
        
        history = trainer.train(config.num_epochs, max_train_batches=config.max_train_batches)
        
        # Plot training history
        plot_training_history(history, save_path='training_history.png')
        
        # Final evaluation
        print("\nFinal evaluation on validation set:")
        metrics = evaluate_model(model, val_loader, device)
        print_evaluation_results(metrics)
        
    elif args.mode == 'eval':
        # Evaluate model
        print("\n" + "="*60)
        print("EVALUATION MODE")
        print("="*60)
        
        metrics = evaluate_model(model, val_loader, device)
        print_evaluation_results(metrics)
        
    elif args.mode == 'visualize':
        # Visualize predictions
        print("\n" + "="*60)
        print("VISUALIZATION MODE")
        print("="*60)
        
        visualize_predictions(
            model=model,
            dataloader=val_loader,
            device=device,
            num_samples=3,
            save_path='predictions.png'
        )
        print("✅ Predictions visualized and saved to predictions.png")
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()

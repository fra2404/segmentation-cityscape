#!/usr/bin/env python3
"""Evaluation script for Cityscapes semantic segmentation."""

import argparse
import sys
from pathlib import Path

# Ensure the project root is on PYTHONPATH so `import src` works
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import Config
from src.data.dataset import create_dataloaders
from src.models.deeplabv3 import create_model, load_checkpoint
from src.evaluation.metrics import evaluate_model, print_evaluation_results
from src.utils.visualization import visualize_predictions


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate DeepLabV3 on Cityscapes dataset'
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, default='./data/cityscapes',
                        help='Path to Cityscapes dataset')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size for evaluation')
    parser.add_argument('--image-size', type=int, nargs=2, default=[512, 1024],
                        help='Image size (height width)')
    parser.add_argument('--filter-city', type=str, default=None,
                        help='Filter validation set to specific city')
    parser.add_argument('--device', type=str, default='mps',
                        choices=['mps', 'cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization of predictions')
    parser.add_argument('--num-samples', type=int, default=3,
                        help='Number of samples to visualize')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    print("="*60)
    print("CITYSCAPES EVALUATION")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data root: {args.data_root}")
    print(f"Device: {args.device}")
    print("="*60)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    _, val_loader, _, _, _ = create_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        image_size=tuple(args.image_size),
        num_workers=0,
        use_weighted_sampler=False,  # Not needed for evaluation
        filter_city=args.filter_city
    )
    
    # Create model
    print("\nCreating model...")
    model, device = create_model(
        num_classes=19,
        pretrained=False,  # We'll load from checkpoint
        device=args.device
    )
    
    # Load checkpoint
    model = load_checkpoint(model, args.checkpoint, device)
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluate_model(model, val_loader, device)
    print_evaluation_results(metrics)
    
    # Visualize if requested
    if args.visualize:
        print("\nGenerating visualizations...")
        visualize_predictions(
            model=model,
            dataloader=val_loader,
            device=device,
            num_samples=args.num_samples,
            save_path='evaluation_predictions.png'
        )
        print("✅ Visualizations saved to evaluation_predictions.png")
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()

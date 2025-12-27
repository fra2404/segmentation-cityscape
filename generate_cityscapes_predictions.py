"""
Generate predictions in Cityscapes format for submission.

This script generates predictions on the validation or test set and saves them
in the format required by the Cityscapes evaluation server.
"""

import os
import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure the project root is on PYTHONPATH so `import src` works
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.deeplabv3 import create_model, load_checkpoint
from src.data.dataset import CityscapesDataset
from src.utils.config import Config


# Cityscapes label mapping: trainId -> id
# Source: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
TRAINID_TO_ID = {
    -1: 0,   # unlabeled
    0: 7,    # road
    1: 8,    # sidewalk
    2: 11,   # building
    3: 12,   # wall
    4: 13,   # fence
    5: 17,   # pole
    6: 19,   # traffic light
    7: 20,   # traffic sign
    8: 21,   # vegetation
    9: 22,   # terrain
    10: 23,  # sky
    11: 24,  # person
    12: 25,  # rider
    13: 26,  # car
    14: 27,  # truck
    15: 28,  # bus
    16: 31,  # train
    17: 32,  # motorcycle
    18: 33,  # bicycle
    255: 0   # ignore -> unlabeled
}


def convert_trainid_to_id(prediction):
    """Convert prediction from trainId to id format."""
    prediction_id = np.zeros_like(prediction, dtype=np.uint8)
    for train_id, label_id in TRAINID_TO_ID.items():
        if train_id == -1 or train_id == 255:
            continue
        prediction_id[prediction == train_id] = label_id
    return prediction_id


def save_prediction(prediction, filename, output_dir):
    """
    Save prediction in Cityscapes format.
    
    Args:
        prediction: numpy array with trainId values
        filename: original image filename (e.g., 'frankfurt_000000_000294_leftImg8bit.png')
        output_dir: directory to save predictions
    """
    # Convert trainId to id
    prediction_id = convert_trainid_to_id(prediction)
    
    # Create output filename: replace '_leftImg8bit' with '_gtFine_labelIds'
    output_filename = filename.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
    output_path = os.path.join(output_dir, output_filename)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as PNG
    img = Image.fromarray(prediction_id)
    img.save(output_path)
    
    return output_path


def generate_predictions(model, dataloader, output_dir, device, split='val'):
    """
    Generate predictions for all images in the dataloader.
    
    Args:
        model: trained model
        dataloader: DataLoader for validation or test set
        output_dir: directory to save predictions
        device: device to run inference on
        split: 'val' or 'test'
    """
    model.eval()
    
    print(f"\n{'='*60}")
    print(f"Generating predictions for {split} set")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Generating predictions', position=0, leave=False)):
            images = batch['image'].to(device)
            filenames = batch['filename']
            
            # Get predictions
            outputs = model(images)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                logits = outputs['out']
            else:
                logits = outputs
            
            # Get predicted class for each pixel
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Save each prediction in the batch
            for i, pred in enumerate(predictions):
                filename = filenames[i]
                
                # Resize prediction to original image size if needed
                # Cityscapes images are 2048x1024, but we trained on 1024x512
                pred_resized = np.array(Image.fromarray(pred.astype(np.uint8)).resize(
                    (2048, 1024), Image.Resampling.NEAREST
                ))
                
                output_path = save_prediction(pred_resized, filename, output_dir)
                saved_files.append(output_path)
    
    print(f"\n✅ Generated {len(saved_files)} predictions")
    print(f"Predictions saved to: {output_dir}")
    
    # Print example filenames
    print(f"\nExample files:")
    for f in saved_files[:3]:
        print(f"  {f}")
    
    return saved_files


def main():
    parser = argparse.ArgumentParser(description='Generate Cityscapes predictions for submission')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, default='./data/cityscapes',
                       help='Path to Cityscapes dataset')
    parser.add_argument('--output-dir', type=str, default='./cityscapes_results',
                       help='Directory to save predictions')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                       help='Dataset split to use (val or test)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for inference')
    parser.add_argument('--architecture', type=str, default='deeplabv3plus',
                       choices=['deeplabv3', 'deeplabv3plus'],
                       help='Model architecture')
    parser.add_argument('--device', type=str, default='mps',
                       choices=['mps', 'cuda', 'cpu'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("CITYSCAPES PREDICTION GENERATION")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data root: {args.data_root}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split: {args.split}")
    print(f"Architecture: {args.architecture}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")
    
    # Create model
    print("Creating model...")
    model, device = create_model(
        num_classes=19,
        pretrained=False,
        device=args.device,
        architecture=args.architecture
    )

    # Load checkpoint
    model = load_checkpoint(model, args.checkpoint, device)

    # Create dataset
    print(f"\nLoading {args.split} dataset...")
    import torchvision.transforms as T
    dataset = CityscapesDataset(
        root=args.data_root,
        split=args.split,
        mode='fine',
        target_type='semantic',
        transform=T.ToTensor(),  # Convert PIL to tensor
        return_filename=True  # Important: return filename for saving
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"Total images: {len(dataset)}")
    print(f"Total batches: {len(dataloader)}")
    
    # Generate predictions
    saved_files = generate_predictions(
        model=model,
        dataloader=dataloader,
        output_dir=args.output_dir,
        device=device,
        split=args.split
    )
    
    # Print submission instructions
    print(f"\n{'='*60}")
    print("SUBMISSION INSTRUCTIONS")
    print(f"{'='*60}")
    print("\n1. Create a ZIP file with the predictions:")
    print(f"   cd {args.output_dir}")
    print(f"   zip -r ../cityscapes_predictions.zip .")
    print("\n2. Upload to Cityscapes evaluation server:")
    print("   https://www.cityscapes-dataset.com/submit/")
    print("\n3. Select the appropriate benchmark:")
    print(f"   - Validation set: Use for development/debugging")
    print(f"   - Test set: Use for final evaluation (limited submissions)")
    print(f"\n✅ Done!")


if __name__ == '__main__':
    main()

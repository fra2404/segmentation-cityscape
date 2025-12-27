"""Run inference on a single image or directory of images."""

import argparse
import os
import sys
from pathlib import Path

# Ensure the project root is on PYTHONPATH so `import src` works
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
import matplotlib.pyplot as plt

from src.models.deeplabv3 import create_model, load_checkpoint
from src.data.transforms import get_val_transforms
from src.utils.visualization import get_cityscapes_colormap, denormalize_image


def inference_single_image(model, image_path, device, image_size=(512, 1024), save_path=None):
    """
    Run inference on a single image.
    
    Args:
        model: Trained model
        image_path: Path to input image
        device: Device to run on
        save_path: Path to save visualization (None = show only)
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)
    
    transform = get_val_transforms(image_size)
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, dict):
            output = output['out']
        prediction = torch.argmax(output, dim=1).squeeze(0).cpu()
    
    # Resize prediction back to original size
    prediction_resized = torch.nn.functional.interpolate(
        prediction.unsqueeze(0).unsqueeze(0).float(),
        size=(original_size[1], original_size[0]),
        mode='nearest'
    ).squeeze().long()
    
    # Visualize
    cmap, norm = get_cityscapes_colormap()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Prediction
    axes[1].imshow(prediction_resized, cmap=cmap, norm=norm, interpolation='nearest')
    axes[1].set_title('Segmentation Prediction')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(prediction_resized, cmap=cmap, norm=norm, alpha=0.5, interpolation='nearest')
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return prediction_resized


def main():
    parser = argparse.ArgumentParser(description='Run semantic segmentation inference')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output visualization')
    parser.add_argument('--image-size', type=int, nargs=2, default=[512, 1024],
                        help='Resize (height width) before inference to match training resolution')
    parser.add_argument('--device', type=str, default='mps',
                        choices=['mps', 'cuda', 'cpu'],
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"❌ Error: Image not found at {args.image}")
        return
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found at {args.checkpoint}")
        return
    
    print(f"Loading model from {args.checkpoint}...")
    
    # Create model (with pretrained=True to get the correct architecture, then load trained weights)
    model, device = create_model(num_classes=19, pretrained=True, device=args.device, architecture='deeplabv3plus')
    model = load_checkpoint(model, args.checkpoint, device)
    
    print(f"Running inference on {args.image}...")
    
    # Run inference
    prediction = inference_single_image(
        model,
        args.image,
        device,
        image_size=tuple(args.image_size),
        save_path=args.output
    )
    
    print("✅ Inference completed!")
    print(f"Prediction shape: {prediction.shape}")
    print(f"Unique classes detected: {torch.unique(prediction).tolist()}")


if __name__ == '__main__':
    main()

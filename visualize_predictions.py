"""Visualize and save segmentation predictions (overlays) on Cityscapes."""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

# Ensure the project root is on PYTHONPATH so `import src` works
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import CityscapesDataset
from src.models.deeplabv3 import create_model, load_checkpoint
from src.utils.visualization import CITYSCAPES_COLORS, denormalize_image, visualize_predictions

# Function to overlay the color mask on the original image
def overlay_mask_on_image(image, mask, alpha=0.5):
    # image: numpy array (H, W, 3) in [0,1]
    # mask: numpy array (H, W, 3) in [0,255]
    image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    overlay = (image * (1 - alpha) + mask * alpha).astype(np.uint8)
    return overlay


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize and save segmentation predictions'
    )
    parser.add_argument('--device', type=str, default='mps', choices=['cpu', 'cuda', 'mps'],
                        help='Device to use (cpu, cuda, mps)')
    parser.add_argument('--batch-size', type=int, default=3, help='Batch size')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test', 'train'],
                        help='Dataset split')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Checkpoint path')
    parser.add_argument('--data-root', type=str, default='data/cityscapes',
                        help='Dataset root')
    return parser.parse_args()

def colorize_mask(mask):
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for idx, color in enumerate(CITYSCAPES_COLORS):
        color_mask[mask == idx] = color
    return color_mask

def main():
    args = parse_args()

    checkpoint_path = args.checkpoint
    data_root = args.data_root
    device_name = args.device
    batch_size = args.batch_size
    split = args.split

    # Load model
    model, device = create_model(
        num_classes=19,
        pretrained=False,
        device=device_name,
        architecture='deeplabv3plus'
    )
    model = load_checkpoint(model, checkpoint_path, device)

    # Prepare dataloader
    transform = T.ToTensor()
    dataset = CityscapesDataset(
        root=data_root,
        split=split,
        mode='fine',
        target_type='semantic',
        transform=transform,
        return_filename=True,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- Generate and save overlays for the selected split ---
    output_dir = f"overlays_{split}"
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            filenames = batch['filename']

            output = model(images)
            if isinstance(output, dict):
                output = output['out']
            preds = torch.argmax(output, dim=1).cpu().numpy()

            images_np = images.detach().cpu()
            for i in range(images_np.shape[0]):
                img = denormalize_image(images_np[i]).permute(1, 2, 0).numpy()
                mask_rgb = colorize_mask(preds[i])
                overlay = overlay_mask_on_image(img, mask_rgb)

                out_path = os.path.join(output_dir, filenames[i].replace('.png', '_overlay.png'))
                plt.imsave(out_path, overlay)


if __name__ == '__main__':
    main()

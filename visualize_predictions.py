
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from src.models.deeplabv3 import create_model, load_checkpoint
from src.data.dataset import CityscapesDataset
from src.utils.visualization import visualize_predictions
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Funzione per sovrapporre la maschera colorata all'immagine originale
def overlay_mask_on_image(image, mask, alpha=0.5):
    # image: numpy array (H, W, 3) in [0,1]
    # mask: numpy array (H, W, 3) in [0,255]
    image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    overlay = (image * (1 - alpha) + mask * alpha).astype(np.uint8)
    return overlay


# Argomenti da linea di comando
parser = argparse.ArgumentParser(description='Visualizza e salva predizioni di segmentazione')
parser.add_argument('--device', type=str, default='mps', choices=['cpu', 'cuda', 'mps'], help='Device da usare (cpu, cuda, mps)')
parser.add_argument('--batch-size', type=int, default=3, help='Batch size')
parser.add_argument('--split', type=str, default='val', choices=['val', 'test', 'train'], help='Split del dataset')
parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', help='Path checkpoint')
parser.add_argument('--data-root', type=str, default='data/cityscapes', help='Root dataset')
args = parser.parse_args()

CHECKPOINT_PATH = args.checkpoint
DATA_ROOT = args.data_root
DEVICE = args.device
BATCH_SIZE = args.batch_size
SPLIT = args.split

# Carica modello
model, device = create_model(num_classes=19, pretrained=False, device=DEVICE, architecture='deeplabv3plus')
model = load_checkpoint(model, CHECKPOINT_PATH, device)

# Prepara dataloader
transform = T.ToTensor()
dataset = CityscapesDataset(
    root=DATA_ROOT,
    split=SPLIT,
    mode='fine',
    target_type='semantic',
    transform=transform,
    return_filename=True
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def colorize_mask(mask):
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for idx, color in enumerate(CITYSCAPES_COLORS):
        color_mask[mask == idx] = color
    return color_mask

# --- Genera e salva overlay per tutto il set di validazione ---
import os
from src.utils.visualization import CITYSCAPES_COLORS, denormalize_image

os.makedirs('results_val', exist_ok=True)
model.eval()
with torch.no_grad():
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        filenames = batch['filename'] if 'filename' in batch else [f"img_{batch_idx}_{i}.png" for i in range(images.size(0))]
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        images = images.cpu()
        preds = preds.cpu()
        for i in range(images.size(0)):
            img = denormalize_image(images[i])
            pred_mask = preds[i].numpy().astype(np.int32)
            color_mask = colorize_mask(pred_mask)
            overlay = overlay_mask_on_image(img, color_mask, alpha=0.5)
            fname = os.path.splitext(filenames[i])[0]
            out_path = f'results_val/{fname}_overlay.png'
            plt.imsave(out_path, overlay)
            print(f"✅ Saved: {out_path}")

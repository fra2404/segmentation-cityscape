# Semantic Segmentation Project

This project implements semantic segmentation on Cityscapes with PyTorch. You can now train and run inference from Python scripts (no notebook required), with configurable input resolution (recommended 512x1024 or full 1024x2048).

## Task Description

- **Input**: RGB images from the Cityscapes dataset (high-resolution urban street scenes).
- **Output**: Pixel-wise segmentation masks with 19 classes (e.g., road, car, pedestrian, etc.).
- **Dataset**: Cityscapes (2,975 training images, 500 validation images).
- **Model**: DeepLabV3 with ResNet101 backbone.
- **Metrics**: Pixel Accuracy and Mean Intersection over Union (IoU).

## Project Structure

- `src/cityscapes_seg/`: Python package
  - `datasets.py`: Cityscapes loaders (train/val/test)
  - `transforms.py`: Mappings + transforms (Resize optional; NEAREST for masks)
  - `model.py`: Model factory (`deeplabv3` or `deeplabv3plus`)
  - `metrics.py`: Pixel accuracy, confusion matrix, mIoU
  - `train.py`: CLI training script
  - `predict_test.py`: CLI test inference saving labelId PNGs
- `requirements.txt`: Python dependencies
- `README.md`: This file

## Step-by-Step Setup and Execution

### 1. Environment Setup

- Create and activate a conda environment:
  ```
  conda create -n ai python=3.12 -y
  conda activate ai
  ```
- Install dependencies:
  ```
  pip install -r requirements.txt
  ```
  Alternatively, if using conda:
  ```
  conda install pytorch torchvision numpy matplotlib opencv -c pytorch -c conda-forge -y
  ```

### 2. Download the Dataset

- **Sito ufficiale**: https://www.cityscapes-dataset.com/
- **Registrazione**: Crea un account gratuito (richiede nome, email, affiliazione).
- **Login**: Accedi con le tue credenziali.
- **Scarica i file**:
  - Vai su "Download" → "leftImg8bit" → "leftImg8bit_trainvaltest.zip" (11GB)
  - Vai su "Download" → "gtFine" → "gtFine_trainvaltest.zip" (241MB)
- **Estrazione**:
  ```
  mkdir -p data/cityscapes
  unzip leftImg8bit_trainvaltest.zip -d ./data/cityscapes/
  unzip gtFine_trainvaltest.zip -d ./data/cityscapes/
  ```
- **Struttura finale**:
  ```
  data/cityscapes/
  ├── leftImg8bit/     # Immagini RGB
  │   ├── train/
  │   ├── val/
  │   └── test/
  └── gtFine/          # Annotazioni ground truth
      ├── train/
      ├── val/
      └── test/
  ```

### 3. Run Training (scripts)

Set module path in your shell session:

```zsh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

Quick smoke test at 512x1024 on MPS (macOS):

```zsh
python -m cityscapes_seg.train \
  --root data/cityscapes \
  --epochs 3 \
  --batch-size 2 \
  --workers 4 \
  --drop-last \
  --image-size 512 1024
```

Notes:
- Use `--image-size -1 -1` to keep original 1024x2048. Reduce `--batch-size` to 1 if you hit memory limits.
- For DeepLabV3+ install first: `pip install segmentation-models-pytorch timm`, then add `--model deeplabv3plus --pretrained`.

### 4. Run Test Inference (labelId PNGs)

Uses the best checkpoint saved during training (default `logs/best_model.pth`):

```zsh
python -m cityscapes_seg.predict_test \
  --root data/cityscapes \
  --weights logs/best_model.pth \
  --outdir predictions_test \
  --batch-size 2 \
  --workers 4 \
  --image-size 512 1024
```

Outputs are valid Cityscapes `*_gtFine_labelIds.png` for server submission.

### 5. About Pre-trained Weights

- The model uses `deeplabv3_resnet101(pretrained=True)`, which automatically downloads pre-trained weights from the torchvision model zoo (trained on COCO dataset).
- This provides a strong starting point, improving performance without training from scratch.
- Download size: ~200MB, cached locally after first run.
- If offline, set `pretrained=False` to train from random weights (slower convergence).
- For reproducibility, the weights are deterministic, but training results may vary slightly due to randomness.

### 6. Analysis and Results

- **Data Analysis**: Cityscapes focuses on urban scenes; images are diverse but may have class imbalance (e.g., more road pixels).
- **Algorithm Effectiveness**: DeepLabV3 uses atrous convolutions and ASPP for multi-scale features, achieving high IoU on urban segmentation.
- **Visualization**: Compare predictions with ground truth to assess errors (e.g., misclassifying small objects).
- **Improvements**: Increase epochs, use data augmentation, fine-tune hyperparameters, or try other backbones like MobileNetV3.

## Requirements

- Python 3.8+
- PyTorch 2.2+
- Torchvision 0.17+
- NumPy, Matplotlib, OpenCV

## Notes

- Training on CPU is slow; prefer MPS (macOS) or CUDA.
- For better accuracy, train at 512x1024 or full 1024x2048. Avoid 256x256 as it harms Pixel Accuracy and mIoU.
- Increase epochs (e.g., 50+) for stronger results; tune batch size to fit memory.

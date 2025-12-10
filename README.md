# Semantic Segmentation Project

This project implements semantic segmentation using the Cityscapes dataset with PyTorch and the DeepLabV3 model. The goal is to classify each pixel in an image into one of 19 categories for urban scenes.

## Task Description

- **Input**: RGB images from the Cityscapes dataset (high-resolution urban street scenes).
- **Output**: Pixel-wise segmentation masks with 19 classes (e.g., road, car, pedestrian, etc.).
- **Dataset**: Cityscapes (2,975 training images, 500 validation images).
- **Model**: DeepLabV3 with ResNet101 backbone.
- **Metrics**: Pixel Accuracy and Mean Intersection over Union (IoU).

## Project Structure

- `semantic_segmentation.ipynb`: Jupyter notebook with the complete implementation.
- `requirements.txt`: Python dependencies.
- `README.md`: This file with project details.

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

### 3. Run the Notebook

- Open `semantic_segmentation.ipynb` in Jupyter or VS Code.
- Execute cells step by step:
  - **Imports**: Load PyTorch, torchvision, etc.
  - **Dataset Loading**: Loads Cityscapes with preprocessing (resize to 256x256, normalization).
  - **Model Building**: Loads DeepLabV3 ResNet101 pre-trained on COCO, modifies the classifier for 19 classes.
  - **Training**: Trains for 5 epochs (adjustable) using Adam optimizer and CrossEntropyLoss.
  - **Evaluation**: Computes pixel accuracy and mean IoU on validation set.
  - **Visualization**: Shows original images, true masks, and predicted masks.

### 4. About Pre-trained Weights

- The model uses `deeplabv3_resnet101(pretrained=True)`, which automatically downloads pre-trained weights from the torchvision model zoo (trained on COCO dataset).
- This provides a strong starting point, improving performance without training from scratch.
- Download size: ~200MB, cached locally after first run.
- If offline, set `pretrained=False` to train from random weights (slower convergence).
- For reproducibility, the weights are deterministic, but training results may vary slightly due to randomness.

### 5. Analysis and Results

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

- Training on CPU is slow; use GPU (MPS on macOS) for faster results.
- For full training, increase epochs to 50+ and use a larger batch size.
- This is a baseline implementation; refer to the notebook for code details.

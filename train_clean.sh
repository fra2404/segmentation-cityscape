#!/bin/bash
# Training script optimized for ~40% mIoU on Cityscapes
# Based on successful configuration with DeepLabV3 ResNet50 + COCO pretrained weights

# Activate conda environment
source ~/miniconda3/bin/activate ai

# Train with clean configuration (no class weights, no weighted sampler)
python train.py \
  --architecture deeplabv3 \
  --image-size 256 512 \
  --batch-size 2 \
  --grad-accum-steps 2 \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --num-epochs 60 \
  --scheduler cosine \
  --min-lr 1e-6 \
  --no-class-weights \
  --no-weighted-sampler \
  --load-checkpoint checkpoints/best_model.pth

# Key differences from previous approach:
# 1. DeepLabV3 (not V3+) with ResNet50 + COCO pretrained weights
# 2. Cityscapes-native resolution (256×512)
# 3. No class weights or weighted sampler
# 4. CosineAnnealingLR instead of Poly
# 5. Batch size 2 with grad_accum=2 (effective batch 4)
# 6. 60 epochs for full convergence

# Expected mIoU progression:
# - 10 epochs: ~30%
# - 30 epochs: ~36%
# - 60 epochs: 38-41%

# For Colab T4 GPU (higher batch size, CUDA device):

python train.py \
  --architecture deeplabv3 \
  --image-size 256 512 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --num-epochs 60 \
  --scheduler cosine \
  --min-lr 1e-6 \
  --no-class-weights \
  --no-weighted-sampler \
  --device cuda \
  --load-checkpoint checkpoints/best_model.pth
# Cityscapes Semantic Segmentation

Semantic segmentation on the Cityscapes dataset using PyTorch with DeepLabV3/DeepLabV3+. The pipeline covers training, evaluation, visualization, and generation of submission-ready predictions.

## Task at a Glance

- **Input**: RGB street-scene images (Cityscapes, 1024x2048).
- **Output**: Pixel-wise masks with 19 trainId classes; inference can also export official labelIds for server submission.
- **Metrics**: Pixel Accuracy and Mean IoU (class-wise IoU reported).
- **Model**: DeepLabV3+ (ResNet50 backbone, ImageNet pretrained by default) or DeepLabV3.

## Repository Layout

- train.py — training/visualization CLI
- evaluate.py — validation metrics + optional visualization
- inference.py — single-image inference and overlay
- generate_cityscapes_predictions.py — batch inference to labelIds PNGs (submission format)
- src/
  - data/dataset.py, data/transforms.py — dataloaders, augmentations, label mappings
  - models/deeplabv3.py — model factory and checkpoints
  - training/trainer.py, training/losses.py — training loop and losses
  - evaluation/metrics.py — pixel accuracy, mIoU
  - utils/config.py, utils/visualization.py — config and plotting helpers
- requirements.txt — dependencies

## Environment

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset (Cityscapes)

Download from https://www.cityscapes-dataset.com/ and extract to `data/cityscapes/`:

```
data/cityscapes/
├── leftImg8bit/{train,val,test}
└── gtFine/{train,val,test}
```

## How to Run

1. Training (512x1024 default, weighted sampler on):

```
python train.py \
	--data-root ./data/cityscapes \
	--num-epochs 3 \
	--batch-size 2 \
	--image-size 512 1024 \
	--device mps
```

2. Evaluation of a checkpoint + optional visuals:

```
python evaluate.py \
	--checkpoint ./checkpoints/best_model.pth \
	--visualize --num-samples 3
```

3. Single-image inference with overlay:

```
python inference.py \
	--image path/to/image.png \
	--checkpoint ./checkpoints/best_model.pth \
	--output overlay.png \
	--image-size 512 1024
```

4. Submission-style predictions (labelIds PNGs):

```
python generate_cityscapes_predictions.py \
	--checkpoint ./checkpoints/best_model.pth \
	--data-root ./data/cityscapes \
	--split val \
	--output-dir ./cityscapes_results
```

## Data and Model Notes

- Cityscapes is highly imbalanced (road/building dominate; rider/pole rare). A weighted sampler and class-weighted loss are enabled by default to upweight rare classes.
- Images are resized to 512x1024 during training/eval for a good trade-off between accuracy and memory. Outputs are upsampled back to input size for visualization.
- Ignore index 255 is handled consistently in loss and metrics.
- Pretrained ImageNet weights provide faster convergence; set `--no-pretrained` to disable.

## Experiments and Expected Metrics

- Quick smoke (3 epochs, 512x1024, bs=2, MPS/CPU): sanity check of the pipeline; expect PixelAcc ≈ 0.45–0.55, mIoU ≈ 0.07–0.15.
- Longer run (≈50 epochs, CUDA, bs=4–8): substantially better mIoU; tune LR, scheduler (`poly`/`cosine`), and augmentations for best results.
- Visual outputs (`predictions.png`, `evaluation_predictions.png`) help inspect systematic errors (small objects, thin structures, boundaries).

## Assignment Coverage

- Clear **input/output** contract: RGB image → 19-class trainId mask (+ labelId export for submission).
- **Data handling**: official Cityscapes splits with correct labelId→trainId mapping and ignore handling.
- **Analysis hooks**: PixelAcc/mIoU reporting and class-wise IoU; weighted sampling to mitigate imbalance.
- **Visualization**: overlays and side-by-side plots for qualitative assessment; batch export for server submission.

## Tips

- If memory is tight, lower `--batch-size` or increase `--grad-accum-steps`.
- Use `--filter-city frankfurt` to validate on a single city when iterating quickly.
- Training on CPU is slow; prefer MPS (Apple Silicon) or CUDA.

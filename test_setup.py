"""Test script to verify the installation and basic functionality."""

import sys
from pathlib import Path

# Add project root to PYTHONPATH so `import src` works
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torchvision

print("="*60)
print("TESTING CITYSCAPES SEGMENTATION PROJECT")
print("="*60)

# Test imports
print("\n1. Testing imports...")
try:
    from src.data.transforms import get_train_transforms, get_val_transforms, to_train_id
    from src.data.dataset import create_dataloaders
    from src.models.deeplabv3 import create_model
    from src.training.trainer import Trainer
    from src.evaluation.metrics import calculate_iou, evaluate_model
    from src.utils.config import Config
    from src.utils.visualization import visualize_predictions
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test PyTorch
print("\n2. Testing PyTorch...")
print(f"   PyTorch version: {torch.__version__}")
print(f"   Torchvision version: {torchvision.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
print(f"   MPS available: {torch.backends.mps.is_available()}")

# Test device selection
print("\n3. Testing device selection...")
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print(f"✅ Using MPS device")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✅ Using CUDA device")
else:
    device = torch.device('cpu')
    print(f"✅ Using CPU device")

# Test model creation
print("\n4. Testing model creation...")
try:
    model, device = create_model(num_classes=19, pretrained=False, device='cpu')
    print(f"✅ Model created successfully")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"❌ Model creation error: {e}")
    sys.exit(1)

# Test transforms
print("\n5. Testing transforms...")
try:
    train_transform = get_train_transforms(image_size=(512, 1024))
    val_transform = get_val_transforms(image_size=(512, 1024))
    print(f"✅ Transforms created successfully")
except Exception as e:
    print(f"❌ Transform error: {e}")
    sys.exit(1)

# Test config
print("\n6. Testing configuration...")
try:
    config = Config()
    print(f"✅ Configuration created successfully")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Image size: {config.image_size}")
    print(f"   Num epochs: {config.num_epochs}")
except Exception as e:
    print(f"❌ Config error: {e}")
    sys.exit(1)

# Test label mapping
print("\n7. Testing label mapping...")
try:
    test_tensor = torch.tensor([[[7, 8, 11, 255]]])  # Some labelIds
    trainids = to_train_id(test_tensor)
    expected = torch.tensor([[0, 1, 2, 255]])  # Expected trainIds
    assert torch.equal(trainids, expected), "Mapping mismatch"
    print(f"✅ Label mapping works correctly")
except Exception as e:
    print(f"❌ Label mapping error: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED ✅")
print("="*60)
print("\nYou can now run:")
print("  python train.py --help")
print("  python evaluate.py --help")

import sys
sys.path.insert(0, 'train_code')
from hsi_dataset import TrainDataset
from architecture import model_generator
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Load dataset
train_data = TrainDataset(
    data_root='/ssd7/ICASSP_2026_Hyper-Object_Challenge/track2/dataset-test',
    crop_size=128,
    bgr2rgb=True,
    arg=True,
    stride=8,
    upscale_factor=2
)

# Create dataloader
train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True, num_workers=0)

# Create model and loss
model = model_generator('mst_plus_plus', pretrained_model_path=None, upscale_factor=2)
model.train()
criterion_l1 = nn.L1Loss()

# Test a few iterations
print("Testing L1 loss training:")
for i, (images, labels) in enumerate(train_loader):
    if i >= 3:  # Test 3 batches
        break

    images = images.cuda()
    labels = labels.cuda()

    print(f"\nBatch {i}:")
    print(f"  Images range: [{images.min():.4f}, {images.max():.4f}]")
    print(f"  Labels range: [{labels.min():.6f}, {labels.max():.4f}]")
    print(f"  Labels has zeros: {(labels == 0).any().item()}")

    # Forward pass
    output = model(images)
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")

    # Compute L1 loss
    loss = criterion_l1(output, labels)
    print(f"  L1 Loss: {loss.item():.6f}")

    if loss.item() > 10:
        print(f"  WARNING: Loss still large!")
    else:
        print(f"  ✓ Loss is in normal range")

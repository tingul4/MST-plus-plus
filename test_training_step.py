import sys
sys.path.insert(0, 'train_code')
from hsi_dataset import TrainDataset
from architecture import model_generator
from utils import Loss_MRAE_custom
import torch
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
criterion_mrae = Loss_MRAE_custom()

# Test a few iterations
print("Testing training iterations:")
for i, (images, labels) in enumerate(train_loader):
    if i >= 3:  # Test 3 batches
        break

    images = images.cuda()
    labels = labels.cuda()

    print(f"\nBatch {i}:")
    print(f"  Images shape: {images.shape}, range: [{images.min():.4f}, {images.max():.4f}]")
    print(f"  Labels shape: {labels.shape}, range: [{labels.min():.4f}, {labels.max():.4f}]")

    # Forward pass
    output = model(images)
    print(f"  Output shape: {output.shape}, range: [{output.min():.4f}, {output.max():.4f}]")

    # Compute loss
    loss = criterion_mrae(output, labels)
    print(f"  Loss: {loss.item():.6f}")

    # Check for abnormal values
    if loss.item() > 100:
        print(f"  WARNING: Loss is abnormally large!")
        # Find problematic regions
        with torch.no_grad():
            error = torch.abs(output - labels)
            rel_error = error / (labels + 1e-8)
            print(f"  Max absolute error: {error.max().item():.6f}")
            print(f"  Max relative error: {rel_error.max().item():.6f}")
            print(f"  Labels min value: {labels.min().item():.8f}")

import sys
sys.path.insert(0, 'train_code')
from hsi_dataset import TrainDataset
from architecture import model_generator
import torch
import torch.nn as nn

# Load dataset
train_data = TrainDataset(
    data_root='/ssd7/ICASSP_2026_Hyper-Object_Challenge/track2/dataset-test',
    crop_size=128,
    bgr2rgb=True,
    arg=False,
    stride=8,
    upscale_factor=2
)

# Create model
model = model_generator('mst_plus_plus', pretrained_model_path=None, upscale_factor=2)
model.eval()

# Get a batch
bgr, hyper = train_data[0]
bgr = torch.from_numpy(bgr).unsqueeze(0).cuda()  # Add batch dimension
hyper = torch.from_numpy(hyper).unsqueeze(0).cuda()

print(f"Input shape: {bgr.shape}")
print(f"Target shape: {hyper.shape}")
print(f"Input range: [{bgr.min():.4f}, {bgr.max():.4f}]")
print(f"Target range: [{hyper.min():.4f}, {hyper.max():.4f}]")

# Forward pass
with torch.no_grad():
    output = model(bgr)

print(f"\nOutput shape: {output.shape}")
print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
print(f"Output has NaN: {torch.isnan(output).any().item()}")
print(f"Output has Inf: {torch.isinf(output).any().item()}")

# Check loss
criterion = nn.L1Loss()
loss = criterion(output, hyper)
print(f"\nL1 Loss: {loss.item()}")

# Check if shapes match
if output.shape != hyper.shape:
    print(f"\nWARNING: Shape mismatch!")
    print(f"  Output: {output.shape}")
    print(f"  Target: {hyper.shape}")
else:
    print(f"\nShapes match: {output.shape}")

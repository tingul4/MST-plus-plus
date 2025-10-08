import sys
sys.path.insert(0, 'train_code')
from hsi_dataset import TrainDataset
import torch
import numpy as np

# Load dataset with upscale_factor=2
train_data = TrainDataset(
    data_root='/ssd7/ICASSP_2026_Hyper-Object_Challenge/track2/dataset-test',
    crop_size=128,
    bgr2rgb=True,
    arg=False,  # Disable augmentation for debugging
    stride=8,
    upscale_factor=2
)

print(f"Dataset length: {len(train_data)}")
print(f"Patch per image: {train_data.patch_per_img}")

# Get first sample
bgr, hyper = train_data[0]

print(f"\nFirst sample shapes:")
print(f"BGR shape: {bgr.shape}")
print(f"Hyper shape: {hyper.shape}")
print(f"\nBGR stats:")
print(f"  Min: {bgr.min()}, Max: {bgr.max()}, Mean: {bgr.mean()}")
print(f"\nHyper stats:")
print(f"  Min: {hyper.min()}, Max: {hyper.max()}, Mean: {hyper.mean()}")

# Check if there are any NaN or Inf values
print(f"\nBGR has NaN: {np.isnan(bgr).any()}")
print(f"BGR has Inf: {np.isinf(bgr).any()}")
print(f"Hyper has NaN: {np.isnan(hyper).any()}")
print(f"Hyper has Inf: {np.isinf(hyper).any()}")

# Test a few more samples
print("\n--- Testing 5 more samples ---")
for i in range(5):
    bgr, hyper = train_data[i]
    print(f"Sample {i}: BGR {bgr.shape}, Hyper {hyper.shape}, "
          f"BGR range [{bgr.min():.4f}, {bgr.max():.4f}], "
          f"Hyper range [{hyper.min():.4f}, {hyper.max():.4f}]")

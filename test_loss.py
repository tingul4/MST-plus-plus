import sys
sys.path.insert(0, 'train_code')
from utils import Loss_MRAE, Loss_MRAE_custom
import torch

# Create sample data with small values
output = torch.rand(1, 61, 256, 256) * 0.3
target = torch.rand(1, 61, 256, 256) * 0.3
target[0, 0, 0, 0] = 0.0001  # Add a very small value

print(f"Output range: [{output.min():.6f}, {output.max():.6f}]")
print(f"Target range: [{target.min():.6f}, {target.max():.6f}]")

# Test original loss
try:
    criterion_original = Loss_MRAE()
    loss_original = criterion_original(output, target)
    print(f"\nOriginal Loss_MRAE: {loss_original.item():.6f}")
except Exception as e:
    print(f"\nOriginal Loss_MRAE error: {e}")

# Test custom loss
criterion_custom = Loss_MRAE_custom()
loss_custom = criterion_custom(output, target)
print(f"Custom Loss_MRAE: {loss_custom.item():.6f}")

# Test with zero value
target_with_zero = target.clone()
target_with_zero[0, 0, 0, 0] = 0.0

print(f"\n--- With zero value in target ---")
try:
    loss_original_zero = criterion_original(output, target_with_zero)
    print(f"Original Loss_MRAE: {loss_original_zero.item():.6f}")
except Exception as e:
    print(f"Original Loss_MRAE error: {e}")

loss_custom_zero = criterion_custom(output, target_with_zero)
print(f"Custom Loss_MRAE: {loss_custom_zero.item():.6f}")

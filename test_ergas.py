import torch
from train_code.losses import ERGASLoss


def test_ergas():
    criterion = ERGASLoss()
    pred = torch.rand(1, 61, 128, 128).cuda()
    pred.requires_grad = True
    target = torch.rand(1, 61, 128, 128).cuda()

    loss = criterion(pred, target)
    print(f"Loss: {loss.item()}")

    loss.backward()
    print("Backward OK")


if __name__ == "__main__":
    test_ergas()

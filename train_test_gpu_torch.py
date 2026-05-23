#!/usr/bin/env python3
"""Quick PyTorch GPU training test script.

Usage:
  python train_test_gpu_torch.py --epochs 2 --batch-size 128
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms


def main(args):
    print("PyTorch version:", torch.__version__)

    use_cuda = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count() if use_cuda else 0
    print("CUDA available:", use_cuda)
    print("Number of GPUs:", num_gpus)
    if use_cuda:
        for i in range(num_gpus):
            print(f" GPU {i}:", torch.cuda.get_device_name(i))
    else:
        print("No GPU detected — training will run on CPU.")

    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device:", device)

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_full = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Split train into train/val
    val_size = int(0.1 * len(train_full))
    train_size = len(train_full) - val_size
    train_set, val_set = random_split(train_full, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader, 1):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch}/{args.epochs}  Train loss: {train_loss:.4f}  Train acc: {train_acc:.4f}  Val loss: {val_loss:.4f}  Val acc: {val_acc:.4f}")

    # Test evaluation
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

    test_loss = test_loss / test_total
    test_acc = test_correct / test_total
    print(f"Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    args = parser.parse_args()
    main(args)

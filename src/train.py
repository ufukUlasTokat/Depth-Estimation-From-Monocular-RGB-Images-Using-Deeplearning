import os
import csv
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

from datasets import SunRGBDDataset, NYUDepthDataset
from models import (
    DepthRegressorResNet18,
    DepthRegressorResNet34,
    DepthRegressorDenseNet,
    DepthRegressorMobileNet,
    DepthRegressorEfficientNet
)

def get_model(model_name):
    models = {
        'resnet18': DepthRegressorResNet18,
        'resnet34': DepthRegressorResNet34,
        'densenet': DepthRegressorDenseNet,
        'mobilenet': DepthRegressorMobileNet,
        'efficientnet': DepthRegressorEfficientNet
    }
    return models[model_name]() if model_name in models else None

def get_dataset(dataset_name, transform, resize_size, max_samples, augment=False, virtual_length_multiplier=1):
    if dataset_name == "sunrgbd":
        return SunRGBDDataset(
            transform=transform,
            resize_size=resize_size,
            max_samples=max_samples,
            augment=augment,
            virtual_length_multiplier=virtual_length_multiplier
        )
    elif dataset_name == "nyudepth":
        return NYUDepthDataset(
            transform=transform,
            resize_size=resize_size,
            max_samples=max_samples
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def train_model(model_name: str, dataset_name: str):
    resize_size = (224, 224)
    batch_size = 8
    max_samples = 100000000
    num_epochs = 5
    best_model_path = f"losses/{model_name}_best_{dataset_name}.pth"
    loss_csv_path = f"weigths/{model_name}_losses_{dataset_name}.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = get_dataset(dataset_name, transform, resize_size, max_samples)

    indices = torch.randperm(len(full_dataset)).tolist()
    train_end = int(0.8 * len(indices))
    val_end = int(0.9 * len(indices))
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    if dataset_name == "sunrgbd":
        train_dataset = Subset(
            get_dataset(dataset_name, transform, resize_size, max_samples, augment=True, virtual_length_multiplier=10),
            list(range(train_end * 10))
        )
    else:
        train_dataset = Subset(full_dataset, train_indices)

    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = get_model(model_name).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        print(f"\n Epoch {epoch + 1}/{num_epochs}")
        model.train()
        total_train_loss = 0.0

        for images, depths in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}"):
            images, depths = images.to(device), depths.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, depths)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for images, depths in tqdm(val_loader, desc=f"[Val] Epoch {epoch+1}"):
                images, depths = images.to(device), depths.to(device)
                preds = model(images)
                loss = criterion(preds, depths)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved at {best_model_path}")

    with open(loss_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Val Loss"])
        for i in range(num_epochs):
            writer.writerow([i+1, train_losses[i], val_losses[i]])

    print(f"Losses saved to {loss_csv_path}")
    print(f"Training complete for {model_name} on {dataset_name}")

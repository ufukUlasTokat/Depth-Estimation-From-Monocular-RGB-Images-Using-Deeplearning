import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn

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
        'effnet': DepthRegressorEfficientNet
    }
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not recognized.")
    return models[model_name]()

def get_dataset(dataset_name, transform, resize_size, max_samples):
    if dataset_name == "sunrgbd":
        return SunRGBDDataset(transform=transform, resize_size=resize_size, max_samples=max_samples)
    elif dataset_name == "nyudepth":
        return NYUDepthDataset(transform=transform, max_samples=max_samples, resize_size=resize_size)
    else:
        raise ValueError(f"Dataset '{dataset_name}' not recognized.")

def unnormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)
    return (img_tensor * std + mean).clamp(0, 1)

def show_prediction(model, dataset, index, device):
    model.eval()
    image, true_depth = dataset[index]
    image_batch = image.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_depth = model(image_batch).cpu().squeeze().numpy()

    true_depth = true_depth.squeeze().numpy()
    image_np = unnormalize(image).permute(1, 2, 0).cpu().numpy()

    # === Display plots ===
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title("Input RGB")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(true_depth, cmap='viridis')
    plt.title("True Depth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_depth, cmap='viridis')
    plt.title("Predicted Depth")
    plt.axis("off")

    plt.show()

def test_model(model_name, dataset_name, model_path, visualize_index=None):
    # === Setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    resize_size = (224, 224)
    batch_size = 8
    max_samples = 1000

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # === Load dataset ===
    # === Load dataset ===
    if dataset_name == "sunrgbd":
        image_dir = "/dataset/rgb224"
        depth_dir = "/dataset/depth224"
        dataset = get_dataset(dataset_name, transform, resize_size, max_samples)
    elif dataset_name == "nyudepth":
        dataset = get_dataset(dataset_name, transform, resize_size, max_samples)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"✅ Loaded {len(dataset)} test samples.")

    # === Load model ===
    model = get_model(model_name).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # === Evaluation ===
    criterion = nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for images, depths in tqdm(loader, desc="🧪 Testing"):
            images, depths = images.to(device), depths.to(device)
            outputs = model(images)
            loss = criterion(outputs*255, depths*255)
            total_loss += loss.item()

    avg_test_loss = total_loss / len(loader)
    print(f"\n🔍 Average Test Loss (MSE): {avg_test_loss:.8f}")

    # === Visualization (Optional) ===
    if visualize_index is not None:
        show_prediction(model, dataset, visualize_index, device)

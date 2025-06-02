import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import kagglehub  # make sure to install kagglehub
                  # pip install kagglehub

class NYUDepthDataset(Dataset):
    def __init__(self, transform=None, max_samples=None, resize_size=(224, 224)):
        self.resize_size = resize_size
        self.transform = transform
        self.samples = []

        # === Step 1: Download Dataset if not already ===
        download_path = kagglehub.dataset_download("soumikrakshit/nyu-depth-v2")
        print("✅ NYU Depth V2 dataset path:", download_path)

        root_dir = os.path.join(download_path, "nyu_data/data", "nyu2_train")  # or nyu2_test for test data

        # === Step 2: Parse samples ===
        for folder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                for file in sorted(os.listdir(folder_path)):
                    if file.endswith(".jpg"):
                        base_name = file.split('.')[0]
                        rgb_path = os.path.join(folder_path, base_name + '.jpg')
                        depth_path = os.path.join(folder_path, base_name + '.png')
                        if os.path.exists(depth_path):
                            self.samples.append((rgb_path, depth_path))
                            if max_samples and len(self.samples) >= max_samples:
                                break
                if max_samples and len(self.samples) >= max_samples:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, depth_path = self.samples[idx]
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, self.resize_size)

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = cv2.resize(depth, self.resize_size)
        depth = depth.astype(np.float32) / 1000.0  # from mm to meters

        if self.transform:
            rgb = self.transform(rgb)

        depth = torch.tensor(depth).unsqueeze(0)
        return rgb, depth



class SunRGBDDataset(Dataset):
    def __init__(self, transform=None, max_samples=None, resize_size=(224, 224),
                 augment=False, virtual_length_multiplier=1):
        # Set paths directly
        root_dir = "..\dataset"
        self.image_dir = os.path.join(root_dir, "rgb224")
        self.depth_dir = os.path.join(root_dir, "depth224")
        self.transform = transform
        self.resize_size = resize_size
        self.augment = augment
        self.samples = []

        # Match files based on the shared prefix (excluding _depth and _left)
        depth_files = sorted(f for f in os.listdir(self.depth_dir) if f.endswith("_disp.png"))
        for depth_fname in depth_files:
            base_id = depth_fname.replace("_disp.png", "")
            rgb_fname = base_id + "_left.png"
            rgb_path = os.path.join(self.image_dir, rgb_fname)
            depth_path = os.path.join(self.depth_dir, depth_fname)

            if os.path.exists(rgb_path) and os.path.exists(depth_path):
                self.samples.append((rgb_path, depth_path))
                if max_samples and len(self.samples) >= max_samples:
                    break

        self.virtual_length_multiplier = virtual_length_multiplier
        self.virtual_length = len(self.samples) * self.virtual_length_multiplier

    def __len__(self):
        return self.virtual_length

    def __getitem__(self, idx):
        real_idx = idx % len(self.samples)
        rgb_path, depth_path = self.samples[real_idx]
        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if rgb is None or depth is None:
            print(f"[Uyarı] Bozuk görüntü atlandı: {rgb_path} veya {depth_path}")
            return self.__getitem__((idx + 1) % self.__len__())

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, self.resize_size)
        depth = cv2.resize(depth, self.resize_size)
        depth = depth.astype(np.float32) / 1000.0  # convert mm to meters

        if self.augment:
            if np.random.rand() < 0.5:
                rgb = np.fliplr(rgb).copy()
                depth = np.fliplr(depth).copy()
            if np.random.rand() < 0.3:
                angle = np.random.uniform(-15, 15)
                M = cv2.getRotationMatrix2D((self.resize_size[0] // 2, self.resize_size[1] // 2), angle, 1.0)
                rgb = cv2.warpAffine(rgb, M, self.resize_size)
                depth = cv2.warpAffine(depth, M, self.resize_size)
            if np.random.rand() < 0.4:
                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.random.uniform(0.6, 1.4), 0, 255)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        if self.transform:
            rgb = self.transform(rgb)

        depth = torch.tensor(depth).unsqueeze(0)
        return rgb, depth

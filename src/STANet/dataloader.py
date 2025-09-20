import os
import torch
import rasterio
import numpy as np
from torch.utils.data import Dataset

class ChangeDetectionDatasetTIF(Dataset):
    def __init__(self, t2019_dir, t2024_dir, mask_dir,classes, transform=None):
        self.t2019_dir = t2019_dir
        self.t2024_dir = t2024_dir
        self.mask_dir = mask_dir
        self.classes = classes  # Change detection classes
        self.transform = transform

        # Load all paths
        self.t2019_paths = sorted([f for f in os.listdir(t2019_dir) if f.endswith('.tif')])
        self.t2024_paths = sorted([f for f in os.listdir(t2024_dir) if f.endswith('.tif')])
        self.mask_paths = sorted([f for f in os.listdir(mask_dir) if f.endswith('.tif')])

    def __len__(self):
        return len(self.t2019_paths)

    def __getitem__(self, index):
        # Load images using rasterio
        with rasterio.open(os.path.join(self.t2019_dir, self.t2019_paths[index])) as src:
            img_t2019 = src.read(out_dtype=np.float32) / 255.0
        with rasterio.open(os.path.join(self.t2024_dir, self.t2024_paths[index])) as src:
            img_t2024 = src.read(out_dtype=np.float32) / 255.0
        # Load masks
        with rasterio.open(os.path.join(self.mask_dir, self.mask_paths[index])) as src:
            cd_mask = src.read(1).astype(np.int64)

        # Convert to PyTorch tensors
        img_t2019 = torch.from_numpy(img_t2019)
        img_t2024 = torch.from_numpy(img_t2024)
        cd_mask = torch.from_numpy(cd_mask)

        # Apply transforms if any
        if self.transform is not None:
            img_t2019 = self.transform(img_t2019)
            img_t2024 = self.transform(img_t2024)

        return img_t2019, img_t2024, cd_mask

def describe_loader(loader_type):
    img2019, img2024, cd_mask = next(iter(loader_type))
    print("Batch size:", loader_type.batch_size)
    print("2019 Image Shape:", img2019.shape)
    print("2024 Image Shape:", img2024.shape)
    print("Change Mask Shape:", cd_mask.shape)
    print("Number of images:", len(loader_type.dataset))
    print("Classes:", loader_type.dataset.classes)
    print("Unique CD values:", torch.unique(cd_mask))
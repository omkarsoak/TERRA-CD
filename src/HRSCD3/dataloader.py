import os
import rasterio
import torch
from torch.utils.data import Dataset
import numpy as np

class ChangeDetectionDatasetTIF(Dataset):
    def __init__(self, t2019_dir, t2024_dir, sem_2019_dir, sem_2024_dir, mask_dir,
                 classes, semantic_classes, transform=None):
        self.t2019_dir = t2019_dir
        self.t2024_dir = t2024_dir
        self.mask_dir = mask_dir
        self.sem_2019_dir = sem_2019_dir
        self.sem_2024_dir = sem_2024_dir
        self.classes = classes  # Change detection classes
        self.semantic_classes = semantic_classes  # Land cover classes
        self.transform = transform

        # Load all paths
        self.t2019_paths = sorted([f for f in os.listdir(t2019_dir) if f.endswith('.tif')])
        self.t2024_paths = sorted([f for f in os.listdir(t2024_dir) if f.endswith('.tif')])
        self.mask_paths = sorted([f for f in os.listdir(mask_dir) if f.endswith('.tif')])
        self.sem2019_paths = sorted([f for f in os.listdir(sem_2019_dir) if f.endswith('.tif')])
        self.sem2024_paths = sorted([f for f in os.listdir(sem_2024_dir) if f.endswith('.tif')])

        # Verify all paths match
        assert len(self.t2019_paths) == len(self.t2024_paths) == len(self.mask_paths) == \
               len(self.sem2019_paths) == len(self.sem2024_paths), "Mismatched number of images"

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
        with rasterio.open(os.path.join(self.sem_2019_dir, self.sem2019_paths[index])) as src:
            sem_mask_2019 = src.read(1).astype(np.int64)
        with rasterio.open(os.path.join(self.sem_2024_dir, self.sem2024_paths[index])) as src:
            sem_mask_2024 = src.read(1).astype(np.int64)

        # Convert to PyTorch tensors
        img_t2019 = torch.from_numpy(img_t2019)
        img_t2024 = torch.from_numpy(img_t2024)
        cd_mask = torch.from_numpy(cd_mask)
        sem_mask_2019 = torch.from_numpy(sem_mask_2019)
        sem_mask_2024 = torch.from_numpy(sem_mask_2024)

        # Apply transforms if any
        if self.transform is not None:
            img_t2019 = self.transform(img_t2019)
            img_t2024 = self.transform(img_t2024)

        return img_t2019, img_t2024, sem_mask_2019, sem_mask_2024, cd_mask


def describe_loader(loader_type):
    """Print information about a data loader"""
    img2019, img2024, sem2019, sem2024, cd_mask = next(iter(loader_type))
    print("Batch size:", loader_type.batch_size)
    print("Shapes:")
    print("  2019 Image:", img2019.shape)
    print("  2024 Image:", img2024.shape)
    print("  2019 Semantic Mask:", sem2019.shape)
    print("  2024 Semantic Mask:", sem2024.shape)
    print("  Change Mask:", cd_mask.shape)
    print("Number of images:", len(loader_type.dataset))
    print("Classes:", loader_type.dataset.classes)
    print("Semantic Classes:", loader_type.dataset.semantic_classes)
    print("\nUnique values:")
    print("  Change Mask:", torch.unique(cd_mask))
    print("  Semantic Mask 2019:", torch.unique(sem2019))

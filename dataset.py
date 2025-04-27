import torch
import numpy as np
from util import *


def load_mask(mask_path):
    mask = np.load(mask_path)  # (5, H, W), uint8
    assert mask.shape == (4, 1024, 1024)
    mask = mask.transpose(1, 2, 0)  # (H, W, 4) for augmentation
    return mask.astype(np.float32)  # normalize to [0, 1]


def load_image(image_path):
    with rasterio.open(image_path) as src:
        image = src.read() # Shape: (C, H, W)
    
    assert image.shape == (12, 1024, 1024)
    image = np.nan_to_num(image, nan=0)

    image = image.transpose(1, 2, 0) # (H, W, 4) for augmentation
    return image.astype(np.float32)

def normalize_image(image, mean, std):
    mean = mean.reshape(12, 1, 1)
    std = std.reshape(12, 1, 1)
    return (image - mean) / std


class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, sample_indices, std, mean, augmentations=None, band_indices=None):
        self.image_paths = [f"{data_root}/train_images/train_{i}.tif" for i in sample_indices]
        self.mask_paths = [f"{data_root}/masks/train_{i}.npy" for i in sample_indices]
        self.augmentations = augmentations
        self.band_indices = band_indices
        self.std = std
        self.mean = mean

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = load_image(self.image_paths[idx])  # Load actual image
        mask = load_mask(self.mask_paths[idx])  # Load actual mask

        # Apply augmentations if provided
        if self.augmentations:
            augmented = self.augmentations(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        # Convert (H, W, C) -> (C, H, W)
        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)
        image = normalize_image(image, self.mean, self.std)
        if self.band_indices is not None:
            image = image[self.band_indices, :, :]
        else:
            image = normalize_image(image, self.mean, self.std)

        # Convert to torch tensors
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask  # No more string paths

    

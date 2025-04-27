## Machine Learning Libraries: 
import torch

## Math imports: 
import numpy as np

## Importing our own modules:
from util import *



def load_mask(mask_path):
    '''
    Load the mask from a .npy file and return it as a float32 array.

    Parameters:
    - mask_path (str): Path to the mask file.

    Returns:        
    - mask (np.ndarray): Mask array of shape (H, W, 4) with values normalized to [0, 1].
    '''
    mask = np.load(mask_path)  # (5, H, W), uint8
    assert mask.shape == (4, 1024, 1024)
    mask = mask.transpose(1, 2, 0)  # (H, W, 4) for augmentation
    return mask.astype(np.float32)  # normalize to [0, 1]


def load_image(image_path):
    '''
    Load the image from a .tif file and return it as a float32 array.

    Params:
    - image_path (str): Path to the image file.

    Returns:
    - image (np.ndarray): Image array of shape (H, W, 12) with NaN values replaced by 0.
    '''
    with rasterio.open(image_path) as src:
        image = src.read() # Shape: (C, H, W)
    
    assert image.shape == (12, 1024, 1024)
    image = np.nan_to_num(image, nan=0)

    image = image.transpose(1, 2, 0) # (H, W, 4) for augmentation
    return image.astype(np.float32)


def normalize_image(image, mean, std):
    '''
    Normalize the image using the provided mean and standard deviation.

    Parameters:
    - image (np.ndarray): Image array of shape (H, W, 12).
    - mean (np.ndarray): Mean values for each band.
    - std (np.ndarray): Standard deviation values for each band.

    Returns:
    - normalized_image (np.ndarray): Normalized image array of shape (H, W, 12).
    '''
    mean = mean.reshape(12, 1, 1)
    std = std.reshape(12, 1, 1)
    return (image - mean) / std


class TrainingDataset(torch.utils.data.Dataset):
    '''
    Class representing a torch.utils.data.Dataset for loading training images and saved masks.
    This class handles loading images and masks, applying augmentations, and normalizing the data.

    Attributes:
    - image_paths (list): List of paths to the training images.
    - mask_paths (list): List of paths to the saved masks.
    - augmentations (callable): Augmentation function to apply to the images and masks.
    - band_indices (list): List of band indices to select from the image.
    - std (np.ndarray): Standard deviation values for each band.
    - mean (np.ndarray): Mean values for each band.

    Functions:
    - __init__(): Initializes the dataset.
    - __len__(): Returns the number of samples in the dataset.
    - __getitem__(): Loads and returns the image and mask at the specified index after applying augmentations and normalization.
    '''
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
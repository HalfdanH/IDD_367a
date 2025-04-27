## Machine Learning Libraries: 
import torch.nn as nn

## Math imports: 
import numpy as np
import matplotlib.pyplot as plt

## Plotting / Image processing imports: 
import json
import os
import matplotlib.pyplot as plt
import rasterio


def open_tif_image(file_path):
    '''
    Open a specific tif image and return file and array

    Parameters:
    - file_path (Str): Path to the tif image file

    Returns:
    - src_file (rasterio.io.DatasetReader): Rasterio dataset reader object
    - src_array (numpy.ndarray): Array representation of the image
    '''
    src_file  = rasterio.open(file_path)
    src_array = src_file.read()
    return src_file, src_array


def analyze_tif_image(file_path):
    '''
    Analyze a tif image and print its properties

    Parameters:
    - file_path (Str): Path to the tif image file
    
    Calls:
    - open_tif_image()
    '''
    src_file, src_array = open_tif_image(file_path)

    print(f"Image shape (bands, height, width):\n - {src_array.shape}\n")

    print(f"Image bands (index, dtype, nodataval, NaN count):")
    for i, dtype, band_array in zip(
        src_file.indexes, src_file.dtypes, src_array
    ):
        nan_count = np.isnan(band_array).sum()
        print(f"Band {i}: dtype = {dtype}, NaN count = {nan_count}")


def visualize_bands(path):
    '''
    Visualize individual bands of a tif image

    Parameters:
    - path (Str): Path to the tif image file

    Calls:
    - open_tif_image()
    '''
    _, src_array = open_tif_image(path)
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))  # Adjust figsize for spacing
    fig.suptitle("Visualizing Individual Bands", fontsize=16, y=0.92)  # Main header

    # Loop through bands and plot each one
    for i, ax in enumerate(axes.flat):
        ax.imshow(src_array[i])      
        ax.set_title(f"Band {i + 1}") 
        ax.axis('off')                 

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  
    plt.show()



def calculate_overall_mean_std(image_dir):
    '''
    Calculate the overall mean and standard deviation of bands in a directory of tif images

    Parameters:
    - image_dir (Str): Directory containing tif images

    Returns:
    - overall_means (numpy.ndarray): Overall mean values for each band
    - overall_stds (numpy.ndarray): Overall standard deviation values for each band
    '''
    band_sums = None
    band_squared_sums = None
    num_pixels = None  

    for filename in os.listdir(image_dir):
        if filename.endswith('.tif'):
            file_path = os.path.join(image_dir, filename)
            with rasterio.open(file_path) as src:
                image_array = src.read() 
            
            mask = np.isnan(image_array) # Find nan_values to replace
            image_array[mask] = 0  
            
            if band_sums is None:
                band_sums = np.zeros(image_array.shape[0], dtype=np.float32)
                band_squared_sums = np.zeros(image_array.shape[0], dtype=np.float32)
                num_pixels = np.zeros(image_array.shape[0], dtype=np.float32)

            band_sums += np.sum(image_array, axis=(1, 2))
            band_squared_sums += np.sum(image_array**2, axis=(1, 2))
            num_pixels += image_array.shape[1] * image_array.shape[2] 

    overall_means = band_sums / num_pixels
    overall_stds = np.sqrt(band_squared_sums / num_pixels - overall_means**2)

    return overall_means, overall_stds


def open_json(file_path):
    '''
    Open a JSON file and return its contents
    
    Parameters:
    - file_path (Str): Path to the JSON file

    Returns:
    - annotations (Dict): Parsed JSON data
    '''
    with open(f"{file_path}") as f:
        annotations = json.load(f)
    return annotations


def return_image_annotations(image_num, file_path):
    '''
    Return image annotations for a specific image number from a JSON file

    Parameters:
    - image_num (Int): Image number to retrieve annotations for
    - file_path (Str): Path to the JSON file

    Returns:
    - annotations (Dict): Annotations for the specified image number
    '''

    annotations = open_json(file_path)
    return annotations["images"][image_num]


class MAEEncoder(nn.Module):
    '''
    pytorch nn.module class representing an MAEEncoder for encoding images using a
    transformer-based architecture. 
    
    Parameters:
    - img_size (Int): Size of the input image
    - patch_size (int): Size of each patch 
    - in_chans (int): Number of input channels
    - embed_dim (int): Dimension of the embedding space 
    - depth (int): Number of transformer encoder layers 
    - num_heads (int): Number of attention heads in the transformer 
    
    Functions:
    - __init__(): Initializes the MAEEncoder with the specified parameters.
    - forward(): Defines the forward pass of the encoder, processing the input image.
    '''
    def __init__(self, img_size=1024, patch_size=16, in_chans=9, embed_dim=768, depth=12, num_heads=12):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.patch_embed = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        x = self.patch_embed(x) 
        x = x.flatten(2)          
        x = x.transpose(1, 2)    
        x = self.encoder(x)

        return x
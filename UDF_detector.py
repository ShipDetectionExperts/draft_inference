import os

import logging
logging.getLogger('absl').setLevel('ERROR')

import numpy as np
import tifffile
from skimage import io as tiff
import matplotlib.pyplot as plt
import random
import rasterio
import netCDF4 as nc

import tensorflow as tf
import tensorrt
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Activation, Add
import cv2
from scipy.ndimage import label
from scipy.ndimage import generate_binary_structure, binary_erosion
import geopandas as gpd
from shapely.geometry import box

import functools
from tensorflow.keras.models import load_model
import numpy as np
import gc
from tensorflow.keras.backend import clear_session
from xarray.core.common import ones_like
import xarray
from openeo.udf import XarrayDataCube
from typing import Dict


def calculate_ndwi(nir_band, green_band):
    
    nir_band_float = nir_band.astype(float)
    green_band_float = green_band.astype(float)
    
    nir_band_float = np.nan_to_num(nir_band_float, nan=0.0)
    green_band_float = np.nan_to_num(green_band_float, nan=0.0)
    
    # Replace zero values in the red band with a small positive value (1e-6)
    nir_band_float[nir_band_float == 0] = 1e-6
    green_band_float[green_band_float == 0] = 1e-6

    
    ndwi = (green_band_float - nir_band_float) / (green_band_float + nir_band_float)
    return ndwi


#%%
def spatiotemp_img(blue_band, red_band):
    # Convert input bands to float and handle invalid values
    blue_band_float = blue_band.astype(float)
    red_band_float = red_band.astype(float)

    # Replace NaN and infinity values with zero
    blue_band_float = np.nan_to_num(blue_band_float, nan=0.0)
    red_band_float = np.nan_to_num(red_band_float, nan=0.0)
    
    # Replace zero values in the red band with a small positive value (1e-6)
    red_band_float[red_band_float == 0] = 1e-6

    img = blue_band_float / (red_band_float+blue_band_float)
    
    return img

def inference_tiles(dataset, tile_size):
    # Convert the list of band images to a NumPy array
    band_images = np.array(dataset)

    # Split the dataset into tiles
    tiled_dataset = []
    
    for band_image in band_images:
        if band_image.shape[0] >= 1:
            band_image = np.transpose(band_image, (1, 2, 0))
        
        num_tiles_x = band_image.shape[1] // tile_size
        num_tiles_y = band_image.shape[0] // tile_size
        
        for tile_y in range(num_tiles_y):
            for tile_x in range(num_tiles_x):
                start_x = tile_x * tile_size
                start_y = tile_y * tile_size

                band_tile = band_image[start_y:start_y + tile_size, start_x:start_x + tile_size]
                tiled_dataset.append(band_tile)
        updated_shape = num_tiles_y*tile_size , num_tiles_x*tile_size
    
    print("Raw shape of the dataset was", band_image.shape)
    print("New shape is", updated_shape)
    return tiled_dataset, updated_shape
 

def ship_detector(region_folder, threshold, min_size_threshold=2, kernel_erosion=3, file_format="netcdf"):
    
    band_names = ["B04","B03","B02", "B08"]
    
    # Initialize empty NumPy arrays for each band
    blue = None
    green = None
    red = None
    nir = None

 
    if file_format == "netcdf":
        # Initialize a dictionary to store the band data
        band_data = {}
        nc_file = nc.Dataset(region_folder, "r")
        
        # Iterate through the band names, extract their data, and store in NumPy arrays
        for band_name in band_names:
            try:
                band_data[band_name] = np.array(nc_file.variables[band_name][:])  # Extract and convert to NumPy array
            except KeyError:
                print(f"Variable '{band_name}' not found in the NetCDF file.")
        
        nc_file.close()

        blue = np.transpose(band_data["B02"], (1, 2, 0))
        green = np.transpose(band_data["B03"], (1, 2, 0))
        red = np.transpose(band_data["B04"], (1, 2, 0))
        nir = np.transpose(band_data["B08"], (1, 2, 0))

        blue_max = np.max(blue)
        blue = blue / blue_max
    
        green_max = np.max(green)
        green = green / green_max
    
        red_max = np.max(red)
        red = red / red_max
    
        nir_max = np.max(nir)
        nir = nir / nir_max

        stored_NDWI = calculate_ndwi(nir,green)
        stored_spatio_temp = spatiotemp_img(blue,red)

        bands = [red, green, blue, stored_NDWI, nir, stored_spatio_temp]
    
        # Determine the shape of the first band
        target_shape = bands[0].shape[:2]
    
        # Reshape each band to the target shape
        bands = [band.reshape(target_shape) for band in bands]
     
    elif file_format == "gtiff":
        # Specify the folder containing GeoTIFF files
        folder_path = region_folder
        
        # Iterate over files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.tiff') and any(suffix in filename for suffix in ['B02', 'B03', 'B04', 'B08']):
                file_path = os.path.join(folder_path, filename)
                data = tiff.imread(file_path)
                
                # Determine the band and store it in the corresponding variable
                if 'B02' in filename:
                    blue = data
                elif 'B03' in filename:
                    green = data
                elif 'B04' in filename:
                    red = data
                elif 'B08' in filename:
                    nir = data

        # Check if all bands were loaded
        if blue is not None and green is not None and red is not None and nir is not None:
            print("All bands loaded successfully.")
        else:
            print("Some bands were not found or loaded.")
            
        blue_max = np.max(blue)
        blue = blue / blue_max
    
        green_max = np.max(green)
        green = green / green_max
    
        red_max = np.max(red)
        red = red / red_max
    
        nir_max = np.max(nir)
        nir = nir / nir_max

        stored_NDWI = calculate_ndwi(nir,green)
        stored_spatio_temp = spatiotemp_img(blue,red)

        bands = [red, green, blue, stored_NDWI, nir, stored_spatio_temp]        

    dataset = []
    dataset.append(tuple(bands))
    
    patches, updated_shape = inference_tiles(dataset, tile_size= 64)

        # Provide the path to the model file
    model_path = 'draft_model/Multihead_Attention_UNet_model.h5'
    model = tf.keras.models.load_model(model_path,custom_objects={"K": K}, compile=False)

    binary_masks = []

    for idx, patch in enumerate(patches):
        test_img = patch

        # Expand dimensions of the test image and make predictions
        test_img_input = np.expand_dims(test_img, 0)
        #tf.keras.utils.disable_interactive_logging()
        prediction = (model.predict(test_img_input, verbose = 0)[0, :, :, 0] > threshold).astype(np.uint8)

        # Append the binary mask to the list
        binary_masks.append(prediction)
        
        #Print a consolidated progress line
        print(f'{idx + 1}/{len(patches)} [===========================]', end='\r')

    #Print a newline character to separate the progress line from other output
    print()
    
    tile_size = 64 
     
    large_binary_mask = np.zeros(updated_shape, dtype=np.uint8)

    num_tiles_x = updated_shape[1] // tile_size

    # Loop through the binary masks and populate the large binary mask
    for idx, mask_tile in enumerate(binary_masks):
        # Calculate the tile's position in the reassembled binary mask
        tile_y = idx // num_tiles_x  # Calculate tile index in y direction
        tile_x = idx % num_tiles_x   # Calculate tile index in x direction

        start_x = tile_x * tile_size
        start_y = tile_y * tile_size

        end_x = start_x + tile_size
        end_y = start_y + tile_size

        # Place the mask tile in the corresponding position
        large_binary_mask[start_y:end_y, start_x:end_x] = mask_tile
    
     # Find connected components and label them
    _, labeled_mask = cv2.connectedComponents(large_binary_mask)

    
    unique_labels, label_counts = np.unique(labeled_mask, return_counts=True)
    for label_mask in unique_labels:
        if label_counts[label_mask] < min_size_threshold:
            labeled_mask[labeled_mask == label_mask] = 0
        
        # Find bounding boxes for the labeled vessels
            # Create a structuring element for neighborhood connections
    structure = generate_binary_structure(2, kernel_erosion)
        # Use binary_erosion to label connected components with neighborhood information
    labeled_mask, num_components  = label(large_binary_mask, structure=structure)
        
        # Create a binary mask to merge the bounding boxes
    merged_binary_mask = np.zeros_like(large_binary_mask, dtype=np.uint8)
    
    bounding_boxes = []
    
    # Draw bounding box borders on the merged binary mask
    for label_mask in range(1, num_components + 1):  # Skip label 0 (background)
        vessel_mask = (labeled_mask == label_mask).astype(np.uint8)
        contours, _ = cv2.findContours(vessel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            bounding_boxes.append((x, y, x + w, y + h))
            # Draw bounding box borders on the merged binary mask
            cv2.rectangle(merged_binary_mask, (x, y), (x + w, y + h), 255, 2)  # Draw green bounding box borders
    
    # Count the number of bounding boxes (vessel entities)
    num_vessels = len(bounding_boxes)
    
    # Plot the merged binary mask with labeled vessels and bounding box borders
    plt.figure(figsize=(10, 10))
    plt.imshow(merged_binary_mask, cmap='gray')  # Display the binary mask
    plt.title(f'Binary Mask with {num_vessels} Vessel Entities and Bounding Box Borders')
    plt.axis('off')
    plt.show()
    
    print(f'Number of Vessel Entities: {num_vessels}')
 
    return

            
            
            
            
            
            

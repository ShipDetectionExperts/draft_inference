import os
import logging
logging.getLogger('absl').setLevel('ERROR')

from scipy.ndimage import label
from scipy.ndimage import generate_binary_structure, binary_erosion

import numpy as np
import tifffile
from skimage import io as tiff
import matplotlib.pyplot as plt
import random
import rasterio
import netCDF4 as nc
import geopandas as gpd
from shapely.geometry import box

import tensorflow as tf
import tensorrt
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Activation, Add
import cv2

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

def inf_true_color_images(region_folder):
    
    print("True color process ongoing...")

    red_band_path = None
    green_band_path = None
    blue_band_path = None

    for file in os.listdir(region_folder):
        if 'B04' in file:
            red_band_path = os.path.join(region_folder, file)
            print(f"B04 found, processing file: {file}")
        elif 'B03' in file:
            green_band_path = os.path.join(region_folder, file)
            print(f"B03 found, processing file: {file}")
        elif 'B02' in file:
            blue_band_path = os.path.join(region_folder, file)
            print(f"B02 found, processing file: {file}")

    if red_band_path and green_band_path and blue_band_path:
        with rasterio.open(red_band_path) as red_band_src, \
            rasterio.open(green_band_path) as green_band_src, \
            rasterio.open(blue_band_path) as blue_band_src:

            red_band = red_band_src.read(1)
            green_band = green_band_src.read(1)
            blue_band = blue_band_src.read(1)

            red_band_norm = red_band.astype(float) / np.iinfo(red_band.dtype).max
            green_band_norm = green_band.astype(float) / np.iinfo(green_band.dtype).max
            blue_band_norm = blue_band.astype(float) / np.iinfo(blue_band.dtype).max

            true_color = np.dstack((red_band_norm, green_band_norm, blue_band_norm))

    else:
        print(f"Band files not found for region: {region_folder}")
    return true_color 



def true_color(red_band,green_band,blue_band):
    
    true_color = np.dstack((red_band, green_band, blue_band))
    # Assuming you have a NumPy array called stored_rgb
    normalized_stored_rgb = true_color.astype(float) / np.max(true_color)

    # Scale to the [0, 255] range
    scaled_stored_rgb = (normalized_stored_rgb * 255).astype(np.uint8)

        # Assuming you have a NumPy array called stored_rgb with integer values in [0, 255] range
    scaled_stored_rgb = scaled_stored_rgb.astype(float)  # Convert to float for accurate multiplication
    
    luminance_factor = 4.0 # Adjust this value to control the luminance
    
    # Scale the pixel values to increase luminance
    brightened_image = scaled_stored_rgb * luminance_factor
    
    # Clip the values to ensure they remain within the valid [0, 255] range
    brightened_image = np.clip(brightened_image, 0.0, 255.0)
    
    # Convert the image back to the integer data type (uint8)
    brightened_image = brightened_image.astype(np.uint8)
        
    brightened_image = np.array(brightened_image)

    return brightened_image 




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




def plot_dataset(dataset, band_names):
    num_bands = len(band_names)
    
    plt.figure(figsize=(18, 6))
    
    region_data = dataset[0]  # Assuming there's only one region
    
    for j in range(num_bands):
        plt.subplot(1, num_bands, j + 1)
        plt.imshow(region_data[j], cmap='gray')
        plt.title(f"Band {band_names[j]}")
    
    plt.tight_layout()
    plt.show()



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
 

def plot_tiled_dataset(tiled_dataset):
    num_patches_in_pair = tiled_dataset[0].shape[2]  # Assuming all arrays in the list have the same shape

    # Randomly select an index from the tiled_dataset
    index = random.randint(0, len(tiled_dataset) - 1)

    plt.figure(figsize=(18, 24))

    for band in range(num_patches_in_pair):
        plt.subplot(num_patches_in_pair, num_patches_in_pair, num_patches_in_pair + band + 1)
        plt.imshow(tiled_dataset[index][:, :, band], cmap='gray')  # Using grayscale colormap
        plt.title(f"Patch {index + 1}, Band {band + 1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()            




            
def ship_detector(bbox, temporal_extent, threshold, min_size_threshold=2, kernel_erosion=3):
    from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
)
    from sentinelhub import SHConfig

    config = SHConfig()
    
    config.sh_client_id = '4a1a2f5e-68c8-45bb-9658-1c3d97f83ba9'
    config.sh_client_secret = 'udBzznCtpu8Hbc7Rq5uAVAWsBetZrqhkDtMENAfV'
    config.save()
    if not config.sh_client_id or not config.sh_client_secret:
        print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")
        
    resolution = 10
    queried_bbox = BBox(bbox=bbox, crs=CRS.WGS84)
    queried_size = bbox_to_dimensions(queried_bbox, resolution=resolution)

    evalscript_true_color = """
    //VERSION=3
function setup() {
  return {
    input: [
      {
        bands: ["B02","B03","B04","B08"],      
        units: "DN",            
      }
    ],
    output: [
      {
        id: "default",
        bands: 4,
        sampleType: "UINT16",        
      },    
    ],
  };
}


function evaluatePixel(sample) {
  return [sample.B04, sample.B03, sample.B02, sample.B08];
}
"""

    request_true_color = SentinelHubRequest(
        evalscript=evalscript_true_color,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=temporal_extent,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=queried_bbox,
        size=queried_size,
        config=config,
    )

    bands_queried = request_true_color.get_data()
    red, green, blue, nir = np.split(bands_queried[0], 4, axis=2)

    red = red.squeeze()
    green = green.squeeze()
    blue = blue.squeeze()
    nir = nir.squeeze()
    
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
    stored_rgb =  true_color(red,green,blue)
    bands = [red, green, blue, stored_NDWI, nir, stored_spatio_temp]        

    dataset = []
    dataset.append(tuple(bands))
    
    patches, updated_shape = inference_tiles(dataset, tile_size= 64)

        # Provide the path to the model file
    model_path = 'drafts/best_Multihead_Attention_UNet_model/Multihead_Attention_UNet_model.h5'
    model = tf.keras.models.load_model(model_path,custom_objects={"K": K}, compile=False)

    binary_masks = []

    for idx, patch in enumerate(patches):
        test_img = patch

        # Expand dimensions of the test image and make predictions
        test_img_input = np.expand_dims(test_img, 0)
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
  
    target_height, target_width = large_binary_mask.shape[:2]

    # Crop stored_rgb to match the target size
    cropped_rgb = stored_rgb[:target_height, :target_width, :]
    
    # Create a copy of the resized RGB image
    overlay_rgb = np.copy(cropped_rgb)

     # Find connected components and label them
    _, labeled_mask = cv2.connectedComponents(large_binary_mask)

    # Filter out small components
    unique_labels, label_counts = np.unique(labeled_mask, return_counts=True)
    for label_mask in unique_labels:
        if label_counts[label_mask] < min_size_threshold:
            labeled_mask[labeled_mask == label_mask] = 0
            
    # Create a structuring element for neighborhood connections
    structure = generate_binary_structure(2, kernel_erosion)
    # Use binary_erosion to label connected components with neighborhood information
    labeled_mask, num_components  = label(large_binary_mask, structure=structure)

    # Find bounding boxes for the labeled vessels
    bounding_boxes = []
    for label_mask in range(1, num_components + 1):  # Skip label 0 (background)
        vessel_mask = (labeled_mask == label_mask).astype(np.uint8)
        contours, _ = cv2.findContours(vessel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            bounding_boxes.append((x, y, x + w, y + h))

    num_vessels = len(bounding_boxes)

    # Create a GeoDataFrame with bounding box polygons
    gdf = gpd.GeoDataFrame(
        {'geometry': [box(x1, y1, x2, y2) for x1, y1, x2, y2 in bounding_boxes]},
        crs="EPSG:4326"  # Assuming WGS 84 coordinate reference system
    )

    # Save the GeoDataFrame as a GeoJSON file
    gdf.to_file('vessel_bounding_boxes.geojson', driver='GeoJSON')

    # Overlay the bounding boxes on the RGB image
    overlay_rgb_with_boxes = np.copy(cropped_rgb)
    for x1, y1, x2, y2 in bounding_boxes:
        cv2.rectangle(overlay_rgb_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green bounding boxes

    # Plot the overlayed RGB image with bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay_rgb_with_boxes)
    plt.title(f'RGB Image with Vessel Entities and Bounding Boxes')
    plt.axis('off')
    plt.show()

    print(f'Number of Vessel Entities: {num_vessels}')

    return






           

           
            
                        
            
            
            
            
            
            
            
            
            
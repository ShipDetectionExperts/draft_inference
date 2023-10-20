import urllib

import xarray
from openeo.udf import XarrayDataCube
from openeo.udf.debug import inspect
from typing import Dict
import numpy as np
import functools
import tensorflow as tf
from tensorflow.keras import backend as K
from pathlib import Path

@functools.lru_cache(maxsize=25)
def load_model():
    model_path = 'https://github.com/ShipDetectionExperts/draft_inference/raw/main/draft_model/Multihead_Attention_UNet_model.h5'
    extract_dir = Path.cwd() / 'tmp'
    extract_dir.mkdir(exist_ok=True, parents=True)

    local_model = extract_dir / "model.h5"
    try:
        modelfile, _ = urllib.request.urlretrieve(model_path, filename=local_model)
    except Exception:
        inspect(message = f'Failed to download model from: {model_path}.')
        raise

    model = tf.keras.models.load_model(local_model,custom_objects={"K": K},compile=False)
    return model

def apply_datacube(cube: XarrayDataCube, context: Dict) -> XarrayDataCube:
    inspect(cube.array, message="UDF DEBUG")
    bands = preprocessing(cube.array)

    test_img_input = np.expand_dims(np.rollaxis(np.array(bands),0,3), 0)
    model = load_model()
    inspect(test_img_input, message="UDF input")
    # tf.keras.utils.disable_interactive_logging()
    prediction = model.predict(test_img_input, verbose=0).astype(np.uint8)
    inspect(prediction.shape, message="UDF prediction")

    result_xarray = xarray.DataArray(
        prediction[0],
        dims=["x", "y","bands"],
        coords={"x": cube.array.coords["x"], "y": cube.array.coords["y"],"bands": ["prediction"]},
    ).transpose("bands","y","x")


    return XarrayDataCube(result_xarray)

def preprocessing(array):
    blue = array.sel(bands="B02")
    green = array.sel(bands="B03")
    red = array.sel(bands="B04")
    nir = array.sel(bands="B08")

    blue_max = np.max(blue)
    blue = blue / blue_max

    green_max = np.max(green)
    green = green / green_max

    red_max = np.max(red)
    red = red / red_max

    nir_max = np.max(nir)
    nir = nir / nir_max

    stored_NDWI = calculate_ndwi(nir, green)
    stored_spatio_temp = spatiotemp_img(blue, red)


    return [red, green, blue, stored_NDWI, nir, stored_spatio_temp]


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


def spatiotemp_img(blue_band, red_band):
    # Convert input bands to float and handle invalid values
    blue_band_float = blue_band.astype(float)
    red_band_float = red_band.astype(float)

    # Replace NaN and infinity values with zero
    blue_band_float = np.nan_to_num(blue_band_float, nan=0.0)
    red_band_float = np.nan_to_num(red_band_float, nan=0.0)

    # Replace zero values in the red band with a small positive value (1e-6)
    red_band_float[red_band_float == 0] = 1e-6

    img = blue_band_float / (red_band_float + blue_band_float)

    return img

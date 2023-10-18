import os
import sys
import logging
import math
import argparse

from sentinelhub import (
    BBox,
    CRS,
    DataCollection,
    SentinelHubCatalog,
    DownloadFailedException,
    MimeType,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
)

relative_paths = ["./utils", "./models", "./draft_model"]
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.extend(os.path.join(current_dir, rel_path) for rel_path in relative_paths)

from utils import request_data_sh, ship_detector, output_geojson, add_to_stac_catalog
from utils import *
from models import *

import argparse

args = None
INSTANCE_ID = None
CLIENT_ID = None
CLIENT_SECRET = None
LAYER_NAME = None
BBOX = None
TIME = None
MAX_CC = None
THRESHOLD = None


# TODO: CHECK IF THE BANDS ARE BEING CORRECTLY LOADED AND USED BY THE SHIP DETECTION FUNCTION


def setup_argparse():
    """
    Function to setup the input parameters for the application package
    --instance_id: Sentinel Hub instance ID
    --layer: Sentinel Hub layer name
    --bbox: Bounding box in the format minx,miny,maxx,maxy
    --time: Time range of the image in the format YYYY-MM-DD/YYYY-MM-DD
    --width: Width of the output image, in pixels
    --height: Height of the output image, in pixels
    --maxcc: Maximum cloud cover percentage
    """
    global args, INSTANCE_ID, CLIENT_ID, CLIENT_SECRET, BBOX, TIME, MAX_CC, THRESHOLD

    args = argparse.ArgumentParser(description="Sentinel Hub Inputs")
    args.add_argument(
        "--instance_id",
        type=str,
        default="your_instance_id",
        help="Sentinel Hub instance ID",
    )
    args.add_argument(
        "--client_id", type=str, default="your_client_id", help="Sentinel Hub client ID"
    )
    args.add_argument(
        "--client_secret",
        type=str,
        default="your_client_secret",
        help="Sentinel Hub client secret",
    )
    args.add_argument(
        "--bbox",
        type=str,
        default="13.4,52.5,13.6,52.7",
        help="Bounding box in the format minx,miny,maxx,maxy",
    )
    args.add_argument(
        "--time",
        type=str,
        default="2021-05/2021-08",
        help="Time range of the image in the format YYYY-MM/YYYY-MM",
    )
    args.add_argument(
        "--maxcc", type=int, default=20, help="Maximum cloud cover percentage"
    )
    args.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold for inference"
    )
    args = args.parse_args()

    # Store parsed arguments in global variables
    INSTANCE_ID = args.instance_id
    CLIENT_ID = args.client_id
    CLIENT_SECRET = args.client_secret
    BBOX = args.bbox
    TIME = args.time
    MAX_CC = args.maxcc
    THRESHOLD = args.threshold

    print(
        f"Command line arguments parsed successfully: {BBOX}, {TIME}, {MAX_CC}, {THRESHOLD}"
    )

    return args


def main():
    setup_argparse()

    data = request_data_sh(CLIENT_ID, CLIENT_SECRET, BBOX, TIME, MAX_CC)

    bounding_boxes = ship_detector(data, THRESHOLD, min_size_threshold=10)

    feature_collection = output_geojson(bounding_boxes)

    add_to_stac_catalog(feature_collection)


if __name__ == "__main__":
    main()

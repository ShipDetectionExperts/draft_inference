import argparse
import time
import os
from model_utils import ship_detector
from model_utils import *
from models import *
from package_utils import request_data_sh, output_geojson, to_stac_catalog, get_detection_file


os.environ['KERAS_HOME'] = '/non_existent_directory'
os.environ['MPLCONFIGDIR'] = '/non_existent_directory'
os.environ['SH_CONFIG_PATH'] = '/non_existent_directory'



args = None
INSTANCE_ID = None
CLIENT_ID = None
CLIENT_SECRET = None
LAYER_NAME = None
BBOX = None
TIME = None
MAX_CC = None
THRESHOLD = None


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
        nargs="?",
        const=1,
        help="Bounding box in the format minx,miny,maxx,maxy",
    )
    args.add_argument(
        "--time",
        type=str,
        default="2021-05/2021-08",
        nargs="?",
        const=1,
        help="Time range of the image in the format YYYY-MM/YYYY-MM",
    )
    args.add_argument(
        "--maxcc", 
        type=int,
        default=20,
        nargs="?",
        const=1, 
        help="Maximum cloud cover percentage"
    )
    args.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        nargs="?",
        const=1,
        help="Threshold for inference"
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

    start_time = time.perf_counter()
    setup_argparse()

    data = request_data_sh(CLIENT_ID, CLIENT_SECRET, BBOX, TIME, MAX_CC)

    bounding_boxes = ship_detector(data, THRESHOLD, min_size_threshold=10)

    feature_collection = output_geojson(bounding_boxes)

    time.sleep(3)

    src_path = os.getcwd()

    latest_file = get_detection_file()

    stac_catalog = to_stac_catalog(feature_collection, latest_file)

    end_time = time.perf_counter()

    print(f"Time taken to run the application: {end_time - start_time} seconds")
    print(f'Created STAC catalog, {stac_catalog} at {src_path}')


if __name__ == "__main__":
    main()



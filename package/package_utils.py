import os
import datetime
import json
import glob
from sentinelhub import (
    BBox,
    SentinelHubCatalog,
    SentinelHubRequest,
    SentinelHubDownloadClient,
    SHConfig,
    CRS,
    DataCollection,
    MimeType,
    bbox_to_dimensions,
)
import pystac

args = None
INSTANCE_ID = None
CLIENT_ID = None
CLIENT_SECRET = None
LAYER_NAME = None
BBOX = None
TIME = None
MAX_CC = None
THRESHOLD = None

#TODO: Fix try except and general error handling, perhaps split into two functions
def request_data_sh(CLIENT_ID, CLIENT_SECRET, BBOX, TIME, MAX_CC):
    sh_client_id = CLIENT_ID
    sh_client_secret = CLIENT_SECRET

    if not sh_client_id or not sh_client_secret:
        sh_client_id = os.environ.get("SH_CLIENT_ID")

    if not sh_client_id or not sh_client_secret:
        sh_client_secret = os.environ.get("SH_CLIENT_SECRET")

    if not sh_client_id or not sh_client_secret:
        raise Exception("Please provide the credentials for Sentinel Hub.")

    config = SHConfig()
    config.sh_client_id = sh_client_id
    config.sh_client_secret = sh_client_secret
    config.sh_base_url = "https://services.sentinel-hub.com"

    custom_evalscript = """
    //VERSION=3

function setup() {
  return {
    input: [
      {
        bands: ["B02","B03","B04","B08"],      
        units: "REFLECTANCE",            
      }
    ],
    output: [
      {
        id: "default",
        bands: 4,
        sampleType: "UINT16",        
      },    
    ],
    mosaicking: "SIMPLE",
  };
}


function evaluatePixel(sample) {
  return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02, 2.5 * sample.B08];
}

"""

    catalog = SentinelHubCatalog(config=config)

    time_str = TIME
    start_date_str, end_date_str = time_str.split("/")
    time_interval = (start_date_str, end_date_str)

    print(time_interval)
    bbox = BBox(bbox=BBOX, crs=CRS.WGS84)
    image_size = bbox_to_dimensions(bbox, resolution=10)

    search_iterator = catalog.search(
        DataCollection.SENTINEL2_L2A,
        bbox=bbox,
        time=time_interval,
        filter=f"eo:cloud_cover < {MAX_CC}",
        fields={
            "include": ["id", "properties.datetime", "properties.cloudCover"],
            "exclude": [],
        },
    )

    results = list(search_iterator)
    print(f"Found {len(results)} results")

    process_requests = []

    for result in results:
        print(f"Processing result with ID {result['id']}")
        request = SentinelHubRequest(
            evalscript=custom_evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=time_interval,
                    mosaicking_order="leastCC",
                )
            ],
            responses=[
                SentinelHubRequest.output_response("default", MimeType.TIFF),
            ],
            bbox=bbox,
            size=image_size,
            config=config,
        )

        process_requests.append(request)

        client = SentinelHubDownloadClient(config=config)
        download_requests = [request.download_list[0] for request in process_requests]
        data = client.download(download_requests, max_threads=3)
        print(f"Data downloaded: {len(data)} items")
        for request in process_requests:
            print(f"Processed request with ID {request.get_filename_list()}")

        return data


def output_geojson(bounding_boxes):
    """
    Creates a GeoJSON file from the bounding boxes
    To be stored in a STAC Catalogue during Stage out
    """

    features = []
    for bbox in bounding_boxes:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [bbox[0], bbox[1]],
                        [bbox[2], bbox[1]],
                        [bbox[2], bbox[3]],
                        [bbox[0], bbox[3]],
                        [bbox[0], bbox[1]],
                    ]
                ],
            },
            "properties": {},
        }
        features.append(feature)

    feature_collection = {"type": "FeatureCollection", "features": features}

    filename = f"detections-{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}.geojson"
    with open(filename, "w") as f:
        json.dump(feature_collection, f)

    return feature_collection


def get_detection_file():
    list_file = glob.glob("detections-*.geojson")
    latest_file = max(list_file, key=os.path.getctime)
    return latest_file


def to_stac_catalog(feature_collection, filename):
    """
    Takes the generated bounding boxes in JSON and puts them into a STAC Catalogue
    utilizing pystac
    """
    catalog_id = filename.rsplit('.', 1)[0]  # remove the file extension

    catalog = pystac.Catalog(
        catalog_id, "Ship Detections", "1.0.0", "A catalog of ship detections"
    )

    # Calculate bbox for the entire feature collection
    all_coordinates = [feature["geometry"]["coordinates"][0] for feature in feature_collection["features"]]
    all_coordinates = [coord for sublist in all_coordinates for coord in sublist]  # flatten the list
    minX = min(coord[0] for coord in all_coordinates)
    minY = min(coord[1] for coord in all_coordinates)
    maxX = max(coord[0] for coord in all_coordinates)
    maxY = max(coord[1] for coord in all_coordinates)
    bbox = [minX, minY, maxX, maxY]

    # Create a MultiPolygon geometry that includes all the polygons
    multi_polygon = {
        "type": "MultiPolygon",
        "coordinates": [feature["geometry"]["coordinates"] for feature in feature_collection["features"]],
    }

    # Create a single item with the MultiPolygon as its geometry
    item = pystac.Item(
        id=f"{catalog_id}-{datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')}",
        geometry=multi_polygon,
        bbox=bbox,
        datetime=datetime.datetime.now(),
        properties={},
    )
    catalog.add_item(item)

    catalog.normalize_and_save(root_href=".", catalog_type=pystac.CatalogType.SELF_CONTAINED)

    return catalog


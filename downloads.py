# -*- coding: utf-8 -*-
"""Wrapper to download a sequence of Sentinel-2 images via sentinel-hub."""
import logging
from sentinelhub import MimeType, CRS, BBox, DataCollection, bbox_to_dimensions
from sentinelhub import SHConfig, SentinelHubRequest

logging.basicConfig(level=logging.INFO)

# Load configuration for sentinel-hub API.
config = SHConfig()

# Bands are set in reverse order.
evalscript_true_color = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04"]
            }],
            output: {
                bands: 3
            }
        };
    }

    function evaluatePixel(sample) {
        return [3.5*sample.B04, 3.5*sample.B03, 3.5*sample.B02];

    }
"""

def request_images(coords_wgs84, timeline):
    """
    Arguments:
        coords_wgs84: WGS84 coordinates (e.g. bboxfinder.com)
        timeline: List of time intervals represented as "yyyy-mm-dd"

    Returns:
        List of images represented as NxMx3 Numpy arrays
    """

    # Set resolution and region bb/size.
    resolution = 10
    region_bbox = BBox(bbox = coords_wgs84, crs = CRS.WGS84)
    region_size = bbox_to_dimensions(region_bbox, resolution = resolution)
    print(f'Requesting images with {resolution}m resolution and region size of {region_size} pixels')

    images = []
    for time_int in timeline:

        # Build the request.
        request_true_color = SentinelHubRequest(
            data_folder='sentinel-hub',
            #resolution = resolution,
            evalscript = evalscript_true_color,
            input_data = [
                SentinelHubRequest.input_data(
                    data_collection = DataCollection.SENTINEL2_L1C,
                    time_interval = time_int,
                    mosaicking_order = 'leastCC'
                )
            ],
            responses = [
                SentinelHubRequest.output_response('default', MimeType.PNG)
            ],
            bbox = region_bbox,
            size = region_size,
            config = config
        )

        # By construction, only one image at time is returned.
        true_color_imgs = request_true_color.get_data(save_data=True)
        images.append(true_color_imgs[0])

    return images
